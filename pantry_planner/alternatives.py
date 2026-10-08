"""Ranked alternatives for one planned line, and the pin checks a re-price shares.

A finished plan carries its PlanBasis: the lines as planned, the product bought
for each, and the constraints, location and origin rules it was planned under.
From that alone, with no LLM call and no write, this module answers "what else
could fill this line, and what would choosing it do to my trip?":

  fetch      ONE build_options_sql call finds every product matching the line
             at each level the planner knows (exact, equivalent form, form,
             generic) plus the line's head noun ("related"); a thin pool adds
             same-aisle substitutes (one build_substitute_sql call), as the
             planner's t4 does. The origin exclusion splits off what the plan
             would never buy (held_back), with the planner's own filter_pool.
  facts      origin evidence, review statistics and pack fit, from database
             rows only; what is not known is said to be unknown.
  trip       ONE price-matrix query over the basket and the candidates, then,
             per candidate, the same pricing and trip optimisation flow.reprice
             runs, so a row's trip total is what choosing it would cost.
  order      ORDER below; rank_reason says why each row sits under the one
             above it.

validate_pins is the check every shopper pin passes before anything is priced:
the line is planned, the product is one of its candidates, the origin
exclusion does not hold it back, and it has an offer in range. The chat cart's
swap and the meal plan's pins both go through it, so a client never supplies
trusted state.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import NamedTuple

from sqlalchemy import text
from sqlalchemy.orm import Session

from . import db
from .config import settings
from .models import (
    BasisLine,
    Pin,
    PlanBasis,
    Product,
    ProductOrigin,
)
from .nlsearch.schemas import IngredientSpec
from .nlsearch.units import tokens

# Bounds. A basis arrives from a client (the hub, the browser's meal plan), so
# it is checked like any other public input before a query is built from it.
MAX_PINS = 40
MAX_BASIS_LINES = 60
MAX_LEFT_OUT = 60
MAX_TEXT = 200          # a name, a country, a store
MAX_NOTE = 1000         # a left-out reason or an interpretation chip
SUBSTITUTE_LIMIT = 5
THIN_POOL = 3           # as the planner's t4: fewer same-ingredient hits adds substitutes

# build_options_sql indexes per line: the line at each level, then its head noun.
_LEVELS = ("exact", "form", "form", "generic", "related")
_PER_LINE = len(_LEVELS)
_MATCH_RANK = {"exact": 0, "form": 1, "generic": 2, "related": 3, "substitute": 4,
               "outside": 5}
_TIER = {"exact": "same", "form": "same", "generic": "same", "related": "other",
         "substitute": "other", "outside": "outside"}


class BasisError(ValueError):
    """The basis is malformed or out of bounds (REST 422, MCP ToolError)."""


class PinError(ValueError):
    """A pin names a line that is not planned, an unknown product, or a product
    that is not a choice for the line; the message says which and why."""


class ValidPin(NamedTuple):
    product_id: int
    tier: str
    match: str


# ─── Checks ──────────────────────────────────────────────────

def check_basis(basis: PlanBasis) -> None:
    """Bounds on a client-supplied basis, so no query is built from an outsized one."""
    from .origins import validate_countries

    def short(value: str | None, what: str, limit: int = MAX_TEXT) -> None:
        if value is not None and len(value) > limit:
            raise BasisError(f"{what} is longer than {limit} characters")

    if len(basis.lines) > MAX_BASIS_LINES:
        raise BasisError(f"a basis has at most {MAX_BASIS_LINES} lines, got {len(basis.lines)}")
    nums = [ln.line_no for ln in basis.lines]
    if len(set(nums)) != len(nums):
        raise BasisError("basis lines must have distinct line numbers")
    for ln in basis.lines:
        for value, what in ((ln.name, "a line name"), (ln.form, "a line form"),
                            (ln.prep, "a line prep"), (ln.unit, "a line unit")):
            short(value, what)
    short(basis.recipe_slug, "recipe_slug")
    short(basis.recipe_name, "recipe_name")
    if len(basis.pins) > MAX_PINS:
        raise BasisError(f"at most {MAX_PINS} pins, got {len(basis.pins)}")
    lists = (basis.exclude_origin, basis.preference, basis.interpretation,
             basis.constraints.exclude_tags, basis.constraints.require_tags,
             basis.constraints.categories, basis.constraints.exclude_categories,
             basis.constraints.subcategories, basis.constraints.exclude_subcategories)
    if any(len(lst) > 50 for lst in lists):
        raise BasisError("a basis list has more than 50 entries")
    for lst in lists:
        for value in lst:
            short(value, "a basis list entry", MAX_NOTE)
    for dropped in (basis.not_stocked, basis.out_of_range, basis.skipped):
        if len(dropped) > MAX_LEFT_OUT:
            raise BasisError(f"a left-out list has more than {MAX_LEFT_OUT} entries")
        for d in dropped:
            short(d.ingredient, "a left-out ingredient", MAX_NOTE)
            short(d.reason, "a left-out reason", MAX_NOTE)
            if len(d.suggestions) > 20:
                raise BasisError("a left-out entry has more than 20 suggestions")
            for sug in d.suggestions:
                short(sug, "a suggestion", MAX_NOTE)
    unknown = validate_countries([*basis.exclude_origin, *basis.preference])
    if unknown:
        raise BasisError("Unrecognised country name(s) in the basis: "
                         + ", ".join(repr(k) for k in unknown))


def planned_lines(basis: PlanBasis) -> dict[int, BasisLine]:
    """The lines the plan bought a product for, by line number."""
    return {ln.line_no: ln for ln in basis.lines if ln.product_id is not None}


def _not_planned(line_no: int, planned: dict[int, BasisLine]) -> str:
    listed = ", ".join(str(n) for n in sorted(planned)) or "none"
    return f"line {line_no} is not a planned line; planned: {listed}"


# ─── Fetching candidates ─────────────────────────────────────

def has_location(basis: PlanBasis) -> bool:
    """A library plan made with no location priced lines at catalog prices and has no
    trip; every other plan has a shopping point."""
    return basis.lat is not None and basis.lon is not None


def _point(basis: PlanBasis) -> tuple[float, float]:
    cfg = settings()
    return (basis.lat if basis.lat is not None else cfg.default_lat,
            basis.lon if basis.lon is not None else cfg.default_lon)


def _head_word(name: str) -> str:
    """The word the "related" level matches alone: the last word of the name that says what
    the ingredient is. Colour, form and descriptor words are skipped (the planner's
    WEAK_SUGGESTION_WORDS), so "cumin powder" relates to cumin, not to baking powder. ""
    when the name is that one word already (the exact level covers it) or has no such
    word; an empty spec joins nothing."""
    from .nlsearch.planner import WEAK_SUGGESTION_WORDS

    toks = tokens(name)
    strong = [t for t in toks if t not in WEAK_SUGGESTION_WORDS]
    return strong[-1] if strong and len(toks) > 1 else ""


def _spec(line: BasisLine, name: str | None = None) -> IngredientSpec:
    if name is not None:      # the head noun alone: no purchase form to require
        return IngredientSpec(name=name, quantity=line.quantity, unit=line.unit)
    return IngredientSpec(name=line.name, form=line.form, prep=line.prep,
                          quantity=line.quantity, unit=line.unit)


@dataclass
class _Pool:
    """Every product found for one line: product id -> its strictest match."""
    found: dict[int, str] = field(default_factory=dict)


def _fetch_pools(s: Session, basis: PlanBasis, lines: list[BasisLine],
                 catalog: dict[int, Product], current: dict[int, int]) -> dict[int, _Pool]:
    """One options query for every line in `lines` (five specs each), then a substitute
    query for each line whose same-ingredient pool is thin. Offers are taken from every
    store here (max_km None) so a product sold only out of range is still found, and
    counted as unavailable later instead of vanishing."""
    from .nlsearch.sql_builder import build_options_sql, build_substitute_sql

    specs: list[IngredientSpec] = []
    relaxed: set[int] = set()
    equivalent: set[int] = set()
    generic: set[int] = set()
    for k, line in enumerate(lines):
        base = k * _PER_LINE
        head = _head_word(line.name)
        specs += [_spec(line), _spec(line), _spec(line), _spec(line), _spec(line, head)]
        equivalent.add(base + 1)
        relaxed.add(base + 2)
        generic.add(base + 3)
    lat, lon = _point(basis)
    sql, params = build_options_sql(basis.constraints, specs, relaxed, lat, lon, None,
                                    per_ingredient_limit=10_000, generic=generic,
                                    equivalent=equivalent)
    pools = {line.line_no: _Pool() for line in lines}
    for row in s.execute(text(sql), params).mappings():
        if row["ing_no"] < 0:
            continue
        k, level = divmod(row["ing_no"], _PER_LINE)
        match = _LEVELS[level]
        found = pools[lines[k].line_no].found
        if row["id"] not in found or _MATCH_RANK[match] < _MATCH_RANK[found[row["id"]]]:
            found[row["id"]] = match
    for line in lines:
        pool = pools[line.line_no]
        same = [pid for pid, m in pool.found.items() if _TIER[m] == "same"]
        if len(same) >= THIN_POOL:
            continue
        subcats = Counter(catalog[pid].subcategory for pid in same
                          if pid in catalog and catalog[pid].subcategory)
        pick = catalog.get(current.get(line.line_no, -1))
        sub = (subcats.most_common(1)[0][0] if subcats
               else pick.subcategory if pick is not None else None)
        if not sub:
            continue
        sql, params = build_substitute_sql(basis.constraints, sub, sorted(pool.found), lat, lon,
                                           basis.max_km, limit=SUBSTITUTE_LIMIT)
        for row in s.execute(text(sql), params).mappings():
            pool.found.setdefault(row["id"], "substitute")
    return pools


def _evidence(ids: list[int], catalog: dict[int, Product]
              ) -> tuple[dict[int, ProductOrigin], set[int]]:
    """Resolved origins (exactly as origins.resolve_all resolves them) and the products
    with a demo label photo among their evidence, from one evidence query."""
    from .origins import LABEL_SOURCES, resolve_origin

    evidence = db.load_origin_evidence(ids)
    origins = {pid: resolve_origin(pid, catalog[pid].name, evidence.get(pid, []))
               for pid in ids if pid in catalog}
    demo = {pid for pid, rows in evidence.items()
            if any(r.source in LABEL_SOURCES and (r.source_ref or "").startswith("demo/")
                   for r in rows)}
    return origins, demo


@dataclass
class _Gathered:
    """What one fetch knows: pools per line, the catalog, origins, and what the plan's
    origin exclusion holds back (product id -> (country, field))."""
    pools: dict[int, _Pool]
    catalog: dict[int, Product]
    origins: dict[int, ProductOrigin]
    demo: set[int]
    held: dict[int, tuple[str, str]]


def _gather(s: Session, basis: PlanBasis, line_nos: list[int], catalog: dict[int, Product],
            current: dict[int, int], extra_ids: set[int]) -> _Gathered:
    from .origins import filter_pool

    planned = planned_lines(basis)
    lines = [planned[n] for n in sorted(set(line_nos))]
    pools = _fetch_pools(s, basis, lines, catalog, current)
    ids = sorted(({pid for p in pools.values() for pid in p.found} | extra_ids) & set(catalog))
    origins, demo = _evidence(ids, catalog)
    held: dict[int, tuple[str, str]] = {}
    if basis.exclude_origin:
        # The plan's own filter (apply_origin_constraint calls it), so what is held back
        # here is exactly what the plan would have dropped.
        _kept, dropped = filter_pool([catalog[pid] for pid in ids],
                                     exclude=basis.exclude_origin, origins=origins)
        held = {p.id: (country, fld) for p, country, fld in dropped}
    return _Gathered(pools=pools, catalog=catalog, origins=origins, demo=demo, held=held)


def _catalog() -> dict[int, Product]:
    return {p.id: p for p in db.load_all_products()}


def effective_picks(basis: PlanBasis, pins: dict[int, int] | None = None) -> dict[int, int]:
    """line_no -> the product the cart buys for it: the shopper's pin, else the plan's."""
    pins = pins if pins is not None else {p.line_no: p.product_id for p in basis.pins}
    return {n: pins.get(n, ln.product_id) for n, ln in planned_lines(basis).items()}


# ─── Pins ────────────────────────────────────────────────────

def validate_pins(basis: PlanBasis, pins: list[Pin] | None = None, *,
                  catalog: dict[int, Product] | None = None,
                  session: Session | None = None) -> dict[int, ValidPin]:
    """The basis's own pins merged with `pins` (a later pin for a line wins), each checked:
    the line is planned, the product exists, it is a candidate for the line (any level,
    or a same-aisle substitute), the plan's origin exclusion does not hold it back, and it
    has an offer in range under the plan's price cap. A pin equal to the plan's own pick
    is dropped, which is how a swap is undone. Raises PinError naming the problem."""
    pins = list(pins or [])
    if len(pins) > MAX_PINS:
        raise PinError(f"at most {MAX_PINS} pins, got {len(pins)}")
    planned = planned_lines(basis)
    merged: dict[int, int] = {}
    for p in [*basis.pins, *pins]:
        merged[p.line_no] = p.product_id
    if len(merged) > MAX_PINS:
        raise PinError(f"at most {MAX_PINS} pinned lines, got {len(merged)}")
    catalog = catalog if catalog is not None else (_catalog() if merged else {})
    for line_no, pid in merged.items():
        if line_no not in planned:
            raise PinError(_not_planned(line_no, planned))
        if pid not in catalog:
            raise PinError(f"unknown product id {pid} in pins")
    merged = {n: pid for n, pid in merged.items() if pid != planned[n].product_id}
    if not merged:
        return {}
    own = session is None
    s = session or Session(db.engine())
    try:
        g = _gather(s, basis, list(merged), catalog, effective_picks(basis, {}),
                    set(merged.values()))
        offers, near = _offers(s, basis, set(merged.values()), probe_missing=True)
    finally:
        if own:
            s.close()
    out: dict[int, ValidPin] = {}
    for line_no, pid in sorted(merged.items()):
        line = planned[line_no]
        product = catalog[pid]
        match = g.pools[line_no].found.get(pid)
        if pid in g.held:
            country, fld = g.held[pid]
            raise PinError(f"{product.name} is evidenced as {_where(country, fld)}, which "
                           "this plan excludes")
        if match is None:
            raise PinError(f"{product.name} is not an option for line {line_no} "
                           f"({line.name}): {_why_not(basis, line, product, near.get(pid))}")
        if not _available(basis, offers.get(pid)):
            raise PinError(f"{product.name} cannot be bought for line {line_no} "
                           f"({line.name}): {_unavailable(basis, near.get(pid))}")
        out[line_no] = ValidPin(product_id=pid, tier=_TIER[match], match=match)
    return out


def _where(country: str, fld: str) -> str:
    return {"ingredient_origin": f"{country} (ingredients)",
            "manufactured_in": f"{country} (made or packed there)",
            "conflicting_evidence": f"{country} by some of its evidence"}.get(fld, country)


def _why_not(basis: PlanBasis, line: BasisLine, product: Product, anywhere: dict | None) -> str:
    """Why a product the shopper chose is not among the line's candidates, from its own
    catalog row: a diet, aisle or price limit of the plan, or the words do not match."""
    c = basis.constraints
    tags = {t.strip() for t in product.dietary_tags.split(",") if t.strip()}
    for tag in c.exclude_tags:
        if tag in tags:
            return f"it contains {tag}, which this plan excludes"
    for tag in c.require_tags:
        if tag not in tags:
            return f"it is not marked {tag}, which this plan requires"
    if ((c.categories and product.category not in c.categories)
            or (c.subcategories and product.subcategory not in c.subcategories)
            or product.category in c.exclude_categories
            or product.subcategory in c.exclude_subcategories):
        return "it is outside the aisles this plan is limited to"
    if (c.max_item_price is not None and anywhere is not None
            and anywhere["price"] > c.max_item_price):
        return (f"its cheapest offer, ${anywhere['price']:.2f}, is over the "
                f"${c.max_item_price:.2f} per-item price cap")
    return f"it does not match the words of {line.name!r}"


def _unavailable(basis: PlanBasis, nearest: dict | None) -> str:
    within = (f"within {basis.max_km:g} km" if basis.max_km is not None else "at any store")
    cap = basis.constraints.max_item_price
    if nearest is None:
        return f"no store {within} sells it" + (
            f" at or under the ${cap:.2f} per-item price cap" if cap is not None else "")
    return (f"no store {within} sells it"
            + (f" at or under the ${cap:.2f} per-item price cap" if cap is not None else "")
            + f"; the nearest offer is {nearest['store']}, {nearest['dist_km']:.1f} km away, "
              f"at ${nearest['price']:.2f}")


# ─── Offers ──────────────────────────────────────────────────

def _matrix(s: Session, basis: PlanBasis, ids: set[int], max_km: float | None) -> list:
    from .nlsearch.sql_builder import build_price_matrix_sql

    lat, lon = _point(basis)
    sql, params = build_price_matrix_sql(sorted(ids), lat, lon, max_km)
    return list(s.execute(text(sql), params).mappings())


def _offers(s: Session, basis: PlanBasis, ids: set[int], *, probe_missing: bool = False
            ) -> tuple[dict[int, dict], dict[int, dict]]:
    """(each product's cheapest offer in range, picked as the plan's path picks it; and, with
    `probe_missing`, the nearest offer anywhere for the products with none in range or none
    under the price cap). Without a location every product is offered at its catalog price,
    as such a plan charges."""
    from .flow import _best_offers, _cheapest_offers

    if not ids:
        return {}, {}
    if not has_location(basis):
        return {}, {}
    offers = _cheapest_offers(_matrix(s, basis, ids, basis.max_km), basis.path)
    near: dict[int, dict] = {}
    missing = {pid for pid in ids if not _available(basis, offers.get(pid))}
    if probe_missing and missing:
        near = _best_offers(_matrix(s, basis, missing, None),
                            key=lambda r: (r["dist_km2"], r["price"]))
    return offers, near


def _available(basis: PlanBasis, offer: dict | None) -> bool:
    """An offer in range under the plan's price cap (always, for a plan with no location).
    The cheapest offer in range is under the cap exactly when any offer in range is."""
    if not has_location(basis):
        return True
    cap = basis.constraints.max_item_price
    return offer is not None and (cap is None or offer["price"] <= cap)

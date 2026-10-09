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

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import NamedTuple

from sqlalchemy import text
from sqlalchemy.orm import Session

from . import db
from .config import settings
from .models import (
    AltBuy,
    AltCounts,
    AlternativeRanking,
    AltMove,
    AltOffer,
    AltOrigin,
    AltRating,
    AltTrip,
    BasisLine,
    HeldBack,
    Pin,
    PlanBasis,
    Product,
    ProductOrigin,
    RankedAlternative,
    Reason,
    Selection,
)
from .nlsearch.schemas import IngredientSpec
from .nlsearch.units import head_noun, normalize_quantity, semantic_key, tokens

# Bounds. A basis arrives from a client (the hub, the browser's meal plan), so
# it is checked like any other public input before a query is built from it.
MAX_PINS = 40
MAX_LIMIT = 25
DEFAULT_LIMIT = 12
MAX_BASIS_LINES = 60
MAX_LEFT_OUT = 60
MAX_TEXT = 200          # a name, a country, a store
MAX_NOTE = 1000         # a left-out reason or an interpretation chip
# Trip effects are worked out for at most this many candidates, the best by
# the keys that come before the trip total. A re-optimisation is cheap with
# the seeded stores, but MAX_ENUMERATED_STORES stores would not be.
MAX_TRIP_EVAL = 40
SUBSTITUTE_LIMIT = 5
THIN_POOL = 3           # as the planner's t4: fewer same-ingredient hits adds substitutes

# The ranking keys, in order. Same ingredient first; then how closely the
# words match (closeness: the demo selector's units.semantic_key with its head
# test read from the line's ingredient word, so the ranking does not disagree
# with the plan about which product is closer, and "cumin powder" is about
# cumin, not powder); then whether the cart's packs cover the recipe's amount;
# then the shopper's origin preference, only when one was given (demo mode and
# the week planner put it before price too); then what the swap does to the
# trip total; then the cost of the recipe's amount; then rating, which only
# breaks exact ties because the reviews are synthetic; then the catalog id, so
# the order is total.
ORDER = ["tier", "semantic_key", "pack_fit", "preference", "trip_total", "cost_for_need",
         "rating", "id"]
RANKING_TEXT = (
    "Products that are the same ingredient come first. Then: how many of the recipe's words "
    "the product matches, and whether it is named for and mainly the ingredient itself (for "
    "cumin powder, cumin rather than powder); whether the packs the cart would buy cover the "
    "recipe's amount; your origin preference, only when you gave one; the trip total after "
    "the swap (prices, an extra stop and travel); the cost of the recipe's amount; the "
    "rating, only to break an exact tie (reviews are demo data); and last the catalog number.")
DATA_NOTE = "Store prices, stock at every store and reviews are demo data."

# build_options_sql indexes per line: the line at each level, then its head noun.
_LEVELS = ("exact", "form", "form", "generic", "related")
_PER_LINE = len(_LEVELS)
_MATCH_RANK = {"exact": 0, "form": 1, "generic": 2, "related": 3, "substitute": 4,
               "outside": 5}
_TIER = {"exact": "same", "form": "same", "generic": "same", "related": "other",
         "substitute": "other", "outside": "outside"}
_TIER_RANK = {"same": 0, "other": 1, "outside": 2}
_FIT_RANK = {"covers": 0, "unknown": 1, "short": 2}


class BasisError(ValueError):
    """The basis is malformed or out of bounds (REST 422, MCP ToolError)."""


class PinError(ValueError):
    """A pin names a line that is not planned, an unknown product, or a product
    that is not a choice for the line; the message says which and why."""


class LineNotPlannedError(PinError):
    """The line number names no planned line (REST 422 line_not_planned)."""


class StaleBasisError(BasisError):
    """The catalog changed under the basis: a product it buys is gone or no longer sold in
    range. Planning the recipe again is the remedy (REST 409 stale_basis)."""


class ValidPin(NamedTuple):
    product_id: int
    tier: str
    match: str


# ─── Checks ──────────────────────────────────────────────────

def check_basis(basis: PlanBasis) -> None:
    """Bounds on a client-supplied basis, so no query is built from an outsized one, and no
    infinite, NaN or out-of-range number reaches the pack and trip arithmetic (math.ceil of
    an infinite need raises) or a distance query (a NaN latitude finds no store and would
    blame the catalog)."""
    from .origins import validate_countries

    def short(value: str | None, what: str, limit: int = MAX_TEXT) -> None:
        if value is not None and len(value) > limit:
            raise BasisError(f"{what} is longer than {limit} characters")

    def number(value: float | None, what: str, low: float, high: float = math.inf, *,
               above: bool = False) -> None:
        if value is None:
            return
        if not (math.isfinite(value) and (value > low if above else value >= low)
                and value <= high):
            span = (f"above {low:g}" if above else f"{low:g} or more") if high == math.inf \
                else f"from {low:g} to {high:g}"
            raise BasisError(f"{what} must be a finite number {span}, got {value!r}")

    if len(basis.lines) > MAX_BASIS_LINES:
        raise BasisError(f"a basis has at most {MAX_BASIS_LINES} lines, got {len(basis.lines)}")
    nums = [ln.line_no for ln in basis.lines]
    if len(set(nums)) != len(nums):
        raise BasisError("basis lines must have distinct line numbers")
    for ln in basis.lines:
        for value, what in ((ln.name, "a line name"), (ln.form, "a line form"),
                            (ln.prep, "a line prep"), (ln.unit, "a line unit")):
            short(value, what)
        number(ln.quantity, f"line {ln.line_no}'s quantity", 0)
    short(basis.recipe_slug, "recipe_slug")
    short(basis.recipe_name, "recipe_name")
    number(basis.lat, "lat", -90, 90)
    number(basis.lon, "lon", -180, 180)
    number(basis.max_km, "max_km", 0, above=True)
    c = basis.constraints
    for value, what in ((c.max_item_price, "constraints.max_item_price"),
                        (c.max_total_budget, "constraints.max_total_budget"),
                        (c.max_distance_km, "constraints.max_distance_km")):
        number(value, what, 0)
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


def _strong_head(name: str) -> str:
    """The ingredient word of a line's name: its last word that says what the ingredient is.
    Colour, form and descriptor words are skipped (the planner's WEAK_SUGGESTION_WORDS), so
    "cumin powder" is about cumin and "roma tomatoes" about tomato. "" when every word is
    such a word."""
    from .nlsearch.planner import WEAK_SUGGESTION_WORDS

    strong = [t for t in tokens(name) if t not in WEAK_SUGGESTION_WORDS]
    return strong[-1] if strong else ""


def _head_word(name: str) -> str:
    """The word the "related" level matches alone, the line's ingredient word, so "cumin
    powder" relates to cumin, not to baking powder. "" when the name is that one word
    already (the exact level covers it) or has no such word; an empty spec joins nothing."""
    return _strong_head(name) if len(tokens(name)) > 1 else ""


def closeness(line_name: str, product: Product) -> tuple[int, int, int, int]:
    """How closely `product` answers a line, lower first: units.semantic_key's shared words
    and fresh test; then whether the product's name has the line's ingredient word
    (_strong_head); then whether the product is mainly that word (its head noun).

    semantic_key reads the head from the line's last word, which for "cumin powder" is the
    form word: Curry Powder would rank above Cumin Seeds, and the row would say "not mainly
    powder". For a line that ends in its ingredient word, as most do, this is semantic_key's
    order with some of its ties broken, so the ranking still agrees with the plan about which
    product is closer. The demo selector keeps semantic_key, so no plan changes."""
    overlap, fresh, last_word_miss = semantic_key(line_name, product)
    word = _strong_head(line_name)
    if not word:
        return overlap, fresh, 0, last_word_miss
    return (overlap, fresh, 0 if word in tokens(product.name) else 1,
            0 if head_noun(product.name) == word else 1)


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
    query for each line whose same-ingredient pool is thin. Matches are taken from every
    store (max_km None) so a product sold only out of range is still found, and counted as
    unavailable later instead of vanishing. Substitutes match none of the line's words; they
    are a few same-aisle products, the cheapest the shopper can buy, so like the planner's
    t4 they come only from stores in range, and none is counted as unavailable."""
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
            raise LineNotPlannedError(_not_planned(line_no, planned))
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


# ─── Ranking ─────────────────────────────────────────────────

def _needs(basis: PlanBasis, line_nos: list[int]) -> list[tuple[float, str] | None]:
    by_no = {ln.line_no: ln for ln in basis.lines}
    return [normalize_quantity(by_no[n].quantity, by_no[n].unit) for n in line_nos]


def _fmt_qty(qty: float, uom: str) -> str:
    if uom == "g" and qty >= 1000:
        return f"{qty / 1000:g} kg"
    if uom == "ml" and qty >= 1000:
        return f"{qty / 1000:g} L"
    return f"{qty:g} {uom}" if uom != "each" else f"{qty:g}"


def _total_need(needs: list[tuple[float, str] | None]) -> tuple[float, str] | None:
    if not needs or any(n is None for n in needs) or len({n[1] for n in needs if n}) != 1:
        return None
    return round(sum(n[0] for n in needs if n), 4), needs[0][1]


def _no_need(needs: list[tuple[float, str] | None], lines: list[BasisLine],
             library: bool) -> str:
    """Why the lines have no amount to compare packs with ('' when they have one). A library
    recipe is planned by its ingredient names alone: its lines carry no amount even where
    the database holds demo house amounts for them, so the recipe is not said to give none."""
    if _total_need(needs) is not None:
        return ""
    unmeasured = [ln for ln, n in zip(lines, needs, strict=True) if n is None]
    if not unmeasured:
        return "The recipe's lines give amounts in different units"
    if all(ln.quantity is None for ln in lines):
        return ("Planned without amounts (a library recipe)" if library
                else "Recipe gives no amount")
    if unmeasured[0].quantity is None:
        return f"Recipe gives no amount for line {unmeasured[0].line_no}"
    ln = unmeasured[0]
    amount = f"{ln.quantity:g} {ln.unit or ''}".strip()
    return f"Recipe amount {amount!r} can't be compared with a pack"


def _pack_facts(product: Product, price: float, cart_packs: int,
                needs: list[tuple[float, str] | None], lines: list[BasisLine],
                library: bool = False) -> tuple[str, float | None, Reason]:
    """(pack_fit, cost_for_need, the pack reason) for one candidate."""
    from .packs import pack_count

    need = _total_need(needs)
    if need is None:
        return "unknown", None, Reason(code="pack", text=_no_need(needs, lines, library),
                                       tone="unknown")
    qty, uom = need
    if not product.unit_qty or not product.unit_uom:
        return "unknown", None, Reason(code="pack", text="Pack size not listed", tone="unknown")
    if product.unit_uom != uom:
        return "unknown", None, Reason(
            code="pack", text=f"Pack in {product.unit_uom}, recipe in {uom}", tone="unknown")
    needed = pack_count(product.unit_qty, product.unit_uom, needs, min_lines=1) or 1
    cost = round(price * needed, 2)
    size = product.unit_size or _fmt_qty(product.unit_qty, uom)
    if cart_packs * product.unit_qty + 1e-9 >= qty:
        text_ = (f"Covers the recipe's {_fmt_qty(qty, uom)} in {cart_packs} "
                 f"pack{'s' if cart_packs > 1 else ''}")
        return "covers", cost, Reason(code="pack", text=text_, tone="plus")
    text_ = (f"Recipe needs {_fmt_qty(qty, uom)}; the cart counts {cart_packs} "
             f"pack{'s' if cart_packs > 1 else ''} of {size}")
    return "short", cost, Reason(code="pack", text=text_, tone="minus")


def _unit_price(product: Product, price: float) -> tuple[float | None, str]:
    if not product.unit_qty or product.unit_uom not in {"g", "ml", "each"}:
        return None, ""
    if product.unit_uom == "each":
        return round(price / product.unit_qty, 2), "each"
    return round(price / product.unit_qty * 100, 2), f"100 {product.unit_uom}"


def _alt_origin(o: ProductOrigin | None, demo: bool) -> AltOrigin:
    """The origin as evidence states it, in rank_products' wording; never a percentage, and
    never a country unless the evidence resolved."""
    from .origins import FULL_CLAIMS, PROCESSING_CLAIMS

    if o is None or o.status == "unknown":
        checked = o is not None and o.evidence_count > 0
        return AltOrigin(status="unknown",
                         label="Origin not published" if checked else "Origin not checked")
    if o.status == "conflicting":
        return AltOrigin(status=o.status, label="Sources disagree on origin", demo=demo)
    if o.status != "resolved":
        return AltOrigin(status=o.status, label="Origin not checked")
    claim = ("full" if o.claim_type in FULL_CLAIMS
             else "processing" if o.claim_type in PROCESSING_CLAIMS else "")
    country = o.country
    label = (f"{country}: origin" if claim == "full"
             else f"{country}: processed there, ingredients may be imported"
             if claim == "processing" else f"{country}: as the source states")
    return AltOrigin(status="resolved", country=country, claim=claim, label=label,
                     verbatim=o.verbatim, source=o.source, demo=demo)


@dataclass
class _Row:
    product: Product
    match: str
    tier: str
    current: bool
    offer: AltOffer
    packs: int
    pack_fit: str
    cost_for_need: float | None
    pack_reason: Reason
    unit_price: float | None
    unit_basis: str
    origin: AltOrigin
    pref: int
    rating: AltRating | None
    organic: bool
    semantic: tuple[int, int, int, int]     # closeness()
    preferred: bool = False         # matches an entry of the shopper's preference
    trip: AltTrip | None = None

    def trip_cents(self, located: bool) -> float:
        if self.trip is not None:
            return round(self.trip.total * 100)
        if not located:
            cost = (self.cost_for_need if self.cost_for_need is not None
                    else self.offer.price * self.packs)
            return round(cost * 100)
        return math.inf

    def key(self, located: bool, use_pref: bool) -> tuple:
        rating = (0, -self.rating.avg) if self.rating is not None else (1, 0.0)
        cost = round(self.cost_for_need * 100) if self.cost_for_need is not None else math.inf
        return (_TIER_RANK[self.tier], self.semantic, _FIT_RANK[self.pack_fit],
                self.pref if use_pref else 0, self.trip_cents(located), cost, rating,
                self.product.id)

    def pre_key(self, use_pref: bool) -> tuple:
        return (_TIER_RANK[self.tier], self.semantic, _FIT_RANK[self.pack_fit],
                self.pref if use_pref else 0, self.product.id)


def _trip_of(options) -> tuple[float, list[str], dict[int, str]] | None:
    best = next((o for o in options if o.recommended), None)
    if best is None:
        return None
    return best.total_cost, list(best.stores), {i.product_id: i.store_name for i in best.items}


def rank_alternatives(basis: PlanBasis, line_no: int, limit: int = DEFAULT_LIMIT
                      ) -> AlternativeRanking:
    """The other products that could fill `line_no`, in ORDER, with the cart's pick always
    included and flagged current. Deterministic for the same basis and database; no LLM,
    no write. Raises BasisError or PinError (a bad basis, an unplanned line, a bad pin)."""
    from . import flow
    from .origins import preference_rank

    check_basis(basis)
    if not 1 <= limit <= MAX_LIMIT:
        raise BasisError(f"limit must be 1..{MAX_LIMIT}, got {limit}")
    planned = planned_lines(basis)
    if line_no not in planned:
        raise LineNotPlannedError(_not_planned(line_no, planned))
    catalog = _catalog()
    with Session(db.engine()) as s:
        valid = validate_pins(basis, [], catalog=catalog, session=s)
        picks = effective_picks(basis, {n: v.product_id for n, v in valid.items()})
        current_id = picks[line_no]
        group = sorted(n for n, pid in picks.items() if pid == current_id)
        unknown = sorted({pid for pid in picks.values() if pid not in catalog})
        if unknown:
            raise StaleBasisError(f"product id(s) {unknown} in the basis are no longer in "
                                  "the catalog; plan the recipe again")
        g = _gather(s, basis, group, catalog, picks, {current_id})
        # A shared purchase is swapped for every line it covers, so a candidate must fill
        # each of them; it carries the loosest of its matches.
        found: dict[int, str] = dict(g.pools[group[0]].found)
        for n in group[1:]:
            other = g.pools[n].found
            found = {pid: max(m, other[pid], key=_MATCH_RANK.__getitem__)
                     for pid, m in found.items() if pid in other}
        held = sorted((pid for pid in g.pools[line_no].found if pid in g.held),
                      key=lambda pid: (catalog[pid].name, pid))
        selectable = {pid: m for pid, m in found.items() if pid not in g.held}
        if current_id not in selectable:
            selectable[current_id] = "outside"
        stats = _stats(s, sorted(selectable))
        located = has_location(basis)
        basket = set(picks.values())
        rows_all = _matrix(s, basis, basket | set(selectable), basis.max_km) if located else []

    offers = flow._cheapest_offers(rows_all, basis.path) if located else {}
    if located and current_id not in offers:
        raise StaleBasisError(f"{catalog[current_id].name}, the cart's pick for line "
                              f"{line_no}, has no offer in range any more; plan the recipe "
                              "again")
    unavailable = [pid for pid in selectable
                   if pid != current_id and not _available(basis, offers.get(pid))]
    for pid in unavailable:
        del selectable[pid]

    synthetic = settings().offers_synthetic
    needs = _needs(basis, group)
    lines = [planned[n] for n in group]
    library = basis.path == "library"
    use_pref = bool(basis.preference)
    worst = len(basis.preference) * 2
    baseline = _price(basis, picks, catalog, rows_all)
    base_trip = _trip_of(baseline[1]) if located else None
    base_stores = set(base_trip[1]) if base_trip else set()

    rows: list[_Row] = []
    for pid, match in selectable.items():
        p = catalog[pid]
        if located:
            o = offers[pid]
            offer = AltOffer(store=o["store"], price=o["price"], distance_km=o["dist_km"],
                             on_trip=o["store"] in base_stores)
        else:
            offer = AltOffer(store="", price=p.price)
        cart_packs = flow._packs(p, needs)
        fit, cost, pack_reason = _pack_facts(p, offer.price, cart_packs, needs, lines, library)
        unit_price, unit_basis = _unit_price(p, offer.price)
        st = stats.get(pid)
        rating = (AltRating(avg=st[0], count=st[1], synthetic=synthetic)
                  if st is not None and st[1] > 0 and st[0] is not None else None)
        pref = preference_rank(g.origins.get(pid), basis.preference) if use_pref else worst
        rows.append(_Row(
            product=p, match=match, tier=_TIER[match], current=pid == current_id,
            offer=offer, packs=cart_packs, pack_fit=fit, cost_for_need=cost,
            pack_reason=pack_reason, unit_price=unit_price, unit_basis=unit_basis,
            origin=_alt_origin(g.origins.get(pid), pid in g.demo),
            pref=pref, preferred=pref < worst,
            rating=rating, organic="organic" in tokens(f"{p.name} {p.description}"),
            semantic=closeness(planned[line_no].name, p)))

    if located and base_trip is not None:
        rows.sort(key=lambda r: r.pre_key(use_pref))
        to_eval = rows[:MAX_TRIP_EVAL] + [r for r in rows[MAX_TRIP_EVAL:] if r.current]
        for r in to_eval:
            r.trip, packs_after = _trip_effect(basis, picks, group, r.product.id, catalog,
                                               rows_all, base_trip, baseline[0])
            if packs_after is not None:
                # The product may already fill another line: one purchase, packs for both
                # needs. The cart counts what the re-price buys.
                r.packs = packs_after
            # The cart pays the price where the trip buys the product, which is not always
            # the lowest price in range (a stop there can cost more than it saves), so the
            # cost for the need and the unit price are worked out at that price.
            price = r.trip.buys_at.price if r.trip and r.trip.buys_at else r.offer.price
            r.pack_fit, r.cost_for_need, r.pack_reason = _pack_facts(
                r.product, price, r.packs, needs, lines, library)
            r.unit_price, r.unit_basis = _unit_price(r.product, price)

    rows.sort(key=lambda r: r.key(located, use_pref))
    items = [_item(i + 1, r, rows[i - 1] if i else None, planned[line_no], located, use_pref,
                   synthetic)
             for i, r in enumerate(rows)]
    shown = items[:limit] + [it for it in items[limit:] if it.current]
    need = _total_need(needs)
    return AlternativeRanking(
        line_no=line_no, lines=group, ingredient=" + ".join(ln.name for ln in lines),
        need=_fmt_qty(*need) if need else "", need_note=_no_need(needs, lines, library),
        need_qty=need[0] if need else None, need_uom=need[1] if need else None,
        order=list(ORDER), ranking_text=RANKING_TEXT,
        items=shown,
        held_back=[_held(pid, catalog, g) for pid in held],
        total=len(items), unavailable=len(unavailable),
        counts=AltCounts(
            exact=sum(1 for it in items if it.match == "exact"),
            no_new_stop=sum(1 for it in items if it.trip is not None
                            and it.trip.stops_delta <= 0),
            preferred_origin=sum(1 for r in rows if r.preferred),
            says_organic=sum(1 for it in items if it.says_organic),
            rated=sum(1 for it in items if it.rating is not None)),
        data_note=DATA_NOTE if synthetic else "")


def _stats(s: Session, ids: list[int]) -> dict[int, tuple[float | None, int]]:
    """product id -> (average rating, review count), from build_stats_sql (Postgres returns
    the average as a Decimal; it leaves as a float)."""
    from .nlsearch.sql_builder import build_stats_sql

    sql, params = build_stats_sql(ids)
    out = {}
    for r in s.execute(text(sql), params).mappings():
        avg = r["avg_rating"]
        out[r["product_id"]] = (round(float(avg), 1) if avg is not None else None,
                                int(r["review_count"] or 0))
    return out


def _price(basis: PlanBasis, picks: dict[int, int], catalog: dict[int, Product], rows):
    """Purchases and trip options for these picks, through flow.price_picks: the same code
    reprice runs, which is what makes a row's trip total equal a real re-price."""
    from .flow import price_picks

    sels = [Selection(line_no=n, product_id=pid, confidence=1.0)
            for n, pid in sorted(picks.items())]
    return price_picks(basis, sels, catalog, rows)


def _trip_effect(basis: PlanBasis, picks: dict[int, int], group: list[int], pid: int,
                 catalog: dict[int, Product], rows, base_trip, base_purchases
                 ) -> tuple[AltTrip | None, int | None]:
    """(the recommended trip with `pid` on every line of `group`, the packs the cart would
    then buy of it). The other lines keep their picks; a line that already buys `pid`
    merges with the group into one purchase, as group_purchases merges any two lines."""
    trial = dict(picks)
    for n in group:
        trial[n] = pid
    purchases, options = _price(basis, trial, catalog, rows)
    packs = next((pu.packs for pu in purchases if pu.product.id == pid), None)
    after = _trip_of(options)
    if after is None:
        return None, packs
    total, stores, where = after
    base_total, base_stores, base_where = base_trip
    # the matrix row the optimiser priced the product from at the store it chose
    there = [r for r in rows if r["product_id"] == pid and r["store_name"] == where.get(pid)]
    at = min(there, key=lambda r: (r["price"], r["store_id"])) if there else None
    others = sorted(n for n, q in picks.items() if q == pid and n not in group)
    names = {pu.product.id: pu.product.name for pu in base_purchases}
    moved = [AltMove(product_id=q, product=names.get(q, catalog[q].name),
                     from_store=base_where[q], to_store=where[q])
             for q in base_where
             if q in where and q != picks[group[0]] and base_where[q] != where[q]]
    return AltTrip(total=total, delta=round(total - base_total, 2) + 0.0, stores=stores,
                   stops_delta=len(stores) - len(base_stores),
                   buys_at=AltBuy(store=at["store_name"], price=at["price"],
                                  distance_km=round(math.sqrt(max(at["dist_km2"], 0.0)), 1))
                   if at is not None else None,
                   merges_with_line=others[0] if others else None,
                   moved_items=moved), packs


def _held(pid: int, catalog: dict[int, Product], g: _Gathered) -> HeldBack:
    country, fld = g.held[pid]
    o = g.origins.get(pid)
    return HeldBack(product_id=pid, product=catalog[pid].name, country=country, field=fld,
                    verbatim=o.verbatim if o is not None else "",
                    source=o.source if o is not None else "", demo=pid in g.demo)


def _item(rank: int, r: _Row, above: _Row | None, line: BasisLine, located: bool,
          use_pref: bool, synthetic: bool) -> RankedAlternative:
    p = r.product
    return RankedAlternative(
        rank=rank, current=r.current, product_id=p.id, product=p.name, brand=p.brand,
        size=p.unit_size, tier=r.tier, match=r.match, offer=r.offer, packs=r.packs,
        pack_fit=r.pack_fit, cost_for_need=r.cost_for_need, unit_price=r.unit_price,
        unit_basis=r.unit_basis, trip=r.trip, origin=r.origin, rating=r.rating,
        says_organic=r.organic, reasons=_reasons(r, line, located, synthetic),
        rank_reason=_rank_reason(rank, r, above, line, located, use_pref, synthetic))


def _money(cents: float) -> str:
    return f"${abs(cents) / 100:.2f}"


def _reasons(r: _Row, line: BasisLine, located: bool, synthetic: bool) -> list[Reason]:
    """At most five reasons, each from a fact on the row: the match, the pack, the origin,
    the trip and the rating. Unknowns are stated as unknown."""
    p = r.product
    head = _head_word(line.name)
    match = {
        "exact": Reason(code="match", text=f"Matches every word of {line.name!r}", tone="plus"),
        "form": Reason(code="match", text=f"Matches {line.name!r} in another form",
                       tone="info"),
        "generic": Reason(code="match",
                          text=f"Matches {line.name!r} without its descriptive words",
                          tone="info"),
        "related": Reason(code="match",
                          text=(f"Shares the word {head!r}; not the same ingredient"
                                if head else "Not the same ingredient"), tone="minus"),
        "substitute": Reason(code="match",
                             text=(f"Same aisle ({p.subcategory}); not the same ingredient"
                                   if p.subcategory else "Same aisle; not the same ingredient"),
                             tone="minus"),
        "outside": Reason(code="match",
                          text="The cart's pick; it matches none of the recipe's words",
                          tone="info"),
    }[r.match]
    o = r.origin
    origin_text = o.label + (f': "{o.verbatim}"' if o.verbatim else "") + (
        " (demo label photo)" if o.demo else "")
    origin_tone = "unknown" if o.status != "resolved" else "plus" if r.preferred else "info"
    out = [match, r.pack_reason, Reason(code="origin", text=origin_text, tone=origin_tone)]
    if r.trip is not None:
        t = r.trip
        if t.delta < 0:
            text_, tone = f"${-t.delta:.2f} less on your trip", "plus"
        elif t.delta > 0:
            text_, tone = f"${t.delta:.2f} more on your trip", "minus"
        else:
            text_, tone = "No change to your trip total", "info"
        if t.stops_delta > 0:
            text_ += f" and {t.stops_delta} more stop{'s' if t.stops_delta > 1 else ''}"
            tone = "minus"
        elif t.stops_delta < 0:
            text_ += f" and {-t.stops_delta} stop{'s' if t.stops_delta < -1 else ''} fewer"
        if t.merges_with_line is not None:
            text_ += f"; already bought for line {t.merges_with_line}"
        out.append(Reason(code="trip", text=text_, tone=tone))
    elif located:
        out.append(Reason(code="trip", text="Trip effect not worked out", tone="unknown"))
    else:
        out.append(Reason(code="trip", text="Catalog price: the plan has no shopping location",
                          tone="info"))
    if r.rating is not None:
        out.append(Reason(code="rating",
                          text=f"{r.rating.avg:.1f} of 5 from {r.rating.count} reviews"
                               + (" (demo)" if synthetic else ""), tone="info"))
    else:
        out.append(Reason(code="rating", text="No reviews", tone="unknown"))
    return out[:5]


def _rank_reason(rank: int, r: _Row, above: _Row | None, line: BasisLine, located: bool,
                 use_pref: bool, synthetic: bool) -> str:
    """Why `r` sits below the row above it: the first ranking key on which it is worse,
    in plain words."""
    if above is None:
        return "Ranked first: nothing ranks above it."
    return f"Below #{rank - 1}: {_first_difference(r, above, line, located, use_pref, synthetic)}."


def _first_difference(r: _Row, above: _Row, line: BasisLine, located: bool, use_pref: bool,
                      synthetic: bool) -> str:
    if _TIER_RANK[r.tier] > _TIER_RANK[above.tier]:
        return ("not the same ingredient" if r.tier == "other"
                else "it matches none of the recipe's words")
    if r.semantic != above.semantic:
        if r.semantic[0] != above.semantic[0]:
            return (f"it matches fewer of the recipe's words ({-r.semantic[0]} against "
                    f"{-above.semantic[0]})")
        if r.semantic[1] != above.semantic[1]:
            return "it is not sold as fresh, which the recipe asks for"
        word = _strong_head(line.name)
        if r.semantic[2] != above.semantic[2]:
            return f"its name does not mention {word}"
        toks = tokens(line.name)
        word = word or (toks[-1] if toks else "")
        return f"it is not mainly {word}" if word else "it is less about the ingredient"
    if _FIT_RANK[r.pack_fit] > _FIT_RANK[above.pack_fit]:
        return ("its packs fall short of the recipe's amount" if r.pack_fit == "short"
                else "its pack can't be compared with the recipe's amount")
    if use_pref and r.pref > above.pref:
        return "its origin is further down your preference"
    mine, theirs = r.trip_cents(located), above.trip_cents(located)
    if mine != theirs:
        if mine == math.inf:
            return f"its trip effect was not worked out (only the first {MAX_TRIP_EVAL} are)"
        if located:
            return f"{_money(mine - theirs)} more on your trip"
        return f"it costs {_money(mine - theirs)} more"
    mine_c = round(r.cost_for_need * 100) if r.cost_for_need is not None else math.inf
    theirs_c = round(above.cost_for_need * 100) if above.cost_for_need is not None else math.inf
    if mine_c != theirs_c:
        if mine_c == math.inf:
            return "its cost for the recipe's amount is unknown"
        return f"{_money(mine_c - theirs_c)} more for the recipe's amount"
    if (r.rating is None) != (above.rating is None) or (
            r.rating is not None and above.rating is not None
            and r.rating.avg != above.rating.avg):
        if r.rating is None:
            return "same price to the cent, and it has no reviews"
        return (f"same price to the cent, and it is rated lower ({r.rating.avg:.1f} against "
                f"{above.rating.avg:.1f}{', demo reviews' if synthetic else ''})")
    return "it ties on every count; listed by catalog number"

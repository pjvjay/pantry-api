"""Ranked alternatives for one planned line, and the pin checks a re-price shares.

A finished plan carries its PlanBasis: the lines as planned, the product bought
for each, and the constraints, location and origin rules it was planned under.
From that alone, with no LLM call and no write, this module answers "what else
could fill this line, and what would choosing it do to my trip?":

  fetch      the matcher's candidates for the line (matching/matcher.py): the
             products that are the same item, those that work instead and its
             nearest looser matches, by meaning. A name no seed knows that cannot
             be embedded is matched by words instead, as it always was: ONE
             build_options_sql call finds every product matching the line at
             each level the planner knows (exact, equivalent form, form,
             generic) plus the line's head noun ("related"), and a thin pool
             adds same-aisle substitutes (one build_substitute_sql call), as the
             planner's t4 does; the ranking says which (`matching`). The origin
             exclusion splits off what the plan would never buy (held_back),
             with the planner's own filter_pool, and the hard rules below what
             it must not offer (filtered).
  rules      a candidate is never choosable when it adds a contains-tag the
             plan's own product lacks (peanut butter's nuts, for butter), when
             it contains what the plan's diet, avoid list or constraints leave
             out (a dairy-free plan never sees a dairy substitute), or when no
             store that is on sells it in range (unavailable, out_of_reach).
  facts      origin evidence, review statistics and pack fit, from database
             rows only; what is not known is said to be unknown.
  trip       ONE price-matrix query over the basket and the candidates, then,
             per candidate, the same pricing and trip optimisation flow.reprice
             runs, so a row's trip total is what choosing it would cost.
  order      by meaning: group, then the matcher's score with the trip filled
             in (ORDER_MEANING), then catalog id, so Options agree with the
             plan's first pick; by words: ORDER. rank_reason says why each row
             sits under the one above it.

validate_pins is the check every shopper pin passes before anything is priced:
the line is planned, the product is one of its candidates, the origin
exclusion does not hold it back, and it has an offer in range. The chat cart's
swap and the meal plan's pins both go through it, so a client never supplies
trusted state.
"""
from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import NamedTuple

from sqlalchemy import text
from sqlalchemy.orm import Session

from . import db
from .config import settings
from .matching import matcher
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
    Filtered,
    HeldBack,
    OutOfReach,
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
# By meaning: the group (Same item, Works instead, Looser match), then the matcher's weighted
# sum (matching/matcher.py WEIGHTS: relevance, form and variant, pack fit, value, origin
# preference and the trip total), then the catalog id.
ORDER_MEANING = ["group", "score", "id"]
RANKING_TEXT_MEANING = (
    "Products that are the same item come first, then products that work instead, then looser "
    "matches, as the matcher grouped them by meaning. Within a group, a weighted score: how "
    "close the product is to the ingredient in meaning (35%), whether it is the form and "
    "variant the recipe asks for (15%), whether its packs fit the recipe's amount (15%), its "
    "price per unit against the group's best (15%), your origin preference (10%, only when "
    "you gave one) and the trip total after the swap (10%). Last, the catalog number.")
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
# Groups by meaning, and the words-era match level each one is reported as.
_GROUP_RANK = {"same_item": 0, "works_instead": 1, "looser_match": 2, "outside": 3}
_LEVEL_OF = {"same_item": "exact", "works_instead": "substitute", "looser_match": "related"}
_GROUP_OF = {"exact": "same_item", "form": "same_item", "generic": "same_item",
             "substitute": "works_instead", "related": "looser_match", "outside": "outside"}
GROUP_LABELS = {"same_item": "Same item", "works_instead": "Works instead",
                "looser_match": "Looser match", "outside": "Not a candidate"}
WORDS_NOTE = ("Matched by words: {why}. Products are found by the recipe's words until it can "
              "be matched by meaning again.")
MEANING_NOTE = "Matched by meaning."
# Contains-tags a row's summary names (diets.product_contains, plus honey from the name): the
# priority allergens (diets.ALLERGENS) and the diet tags. "nuts" is left out: peanuts and tree
# nuts name it more exactly.
_TAGS = ("dairy", "egg", "fish", "gluten", "honey", "meat", "mustard", "peanuts", "sesame",
         "shellfish", "soy", "sulphites", "tree_nuts")
# What "no new allergens" was checked against, said on the row so the claim is never wider than it.
_CHECKED = "no new allergens (11 checked)"


class BasisError(ValueError):
    """The basis is malformed or out of bounds (REST 422, MCP ToolError)."""


class PinError(ValueError):
    """A pin names a line that is not planned, an unknown product, or a product
    that is not a choice for the line; the message says which and why, and
    `line_no` names the line when the pin's line is known."""

    def __init__(self, message: str, line_no: int | None = None):
        super().__init__(message)
        self.line_no = line_no


class LineNotPlannedError(PinError):
    """The line number names no planned line (REST 422 line_not_planned)."""


class HeldBackPinError(PinError):
    """The pinned product is one the plan's origin exclusion holds back: `product_id`, the
    excluded `country` its evidence matched and the `field` it matched in, as
    origins.filter_pool reports them."""

    def __init__(self, message: str, line_no: int, product_id: int, country: str, field: str):
        super().__init__(message, line_no)
        self.product_id, self.country, self.field = product_id, country, field


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


def _aisle_of(word: str, pick: Product | None, found: dict[int, str],
              catalog: dict[int, Product]) -> str | None:
    """The aisle the ingredient `word` is in: the pick's subcategory, or with no pick the one
    most of the same-ingredient candidates mainly about `word` share (ties by name); None
    when that is not known."""
    if pick is not None:
        return pick.subcategory or None
    shared = Counter(catalog[pid].subcategory for pid, m in found.items()
                     if _TIER[m] == "same" and pid in catalog and catalog[pid].subcategory
                     and head_noun(catalog[pid].name) == word)
    return min(shared.items(), key=lambda kv: (-kv[1], kv[0]))[0] if shared else None


def _another_food(product: Product, word: str, aisle: str | None) -> bool:
    """A product that matches the line's words but is another food: not mainly the
    ingredient (its head noun is not the line's ingredient word) and from another aisle than
    the ingredient's. Canadian Peanut Butter Cream for butter is; Extra Virgin Olive Oil for
    olive oil, Boneless Skinless Chicken Breast for chicken breast, Tomato Paste for canned
    tomatoes (the canned aisle) are not."""
    return (bool(word) and aisle is not None and head_noun(product.name) != word
            and (product.subcategory or None) != aisle)


def _what_it_is(product: Product, word: str) -> str:
    """What a product that mentions `word` is instead: the word with the one before it in
    the product's name ("peanut butter"), its own head noun when the name starts with the
    word ("mainly sauce"), or its name when only its description mentions it."""
    toks = tokens(product.name)
    if word not in toks:
        return product.name
    if toks.index(word) > 0:
        return f"{toks[toks.index(word) - 1]} {word}"
    head = head_noun(product.name)
    return f"mainly {head}" if head else product.name


def _spec(line: BasisLine, name: str | None = None) -> IngredientSpec:
    if name is not None:      # the head noun alone: no purchase form to require
        return IngredientSpec(name=name, quantity=line.quantity, unit=line.unit)
    return IngredientSpec(name=line.name, form=line.form, prep=line.prep,
                          quantity=line.quantity, unit=line.unit)


@dataclass
class _Pool:
    """Every product found for one line: product id -> its strictest match, and how the line
    was matched (`meaning`); by meaning, each product's matcher candidate too."""
    found: dict[int, str] = field(default_factory=dict)
    meaning: matcher.Meaning | None = None
    cands: dict[int, matcher.Candidate] = field(default_factory=dict)

    @property
    def by_meaning(self) -> bool:
        return self.meaning is not None and self.meaning.matching == "meaning"


def _fetch_pools(s: Session, basis: PlanBasis, lines: list[BasisLine],
                 catalog: dict[int, Product], current: dict[int, int]) -> dict[int, _Pool]:
    """Each line's candidates. By meaning, the matcher's, from the whole catalog, so a
    product sold only out of range is still found, and counted as unavailable later instead
    of vanishing. By words (a name no seed knows that cannot be embedded): one options query
    for every such line (five specs each), then a substitute query for each line whose
    same-ingredient pool is thin. Word matches are taken from every store (max_km None), for
    the same reason. Substitutes match none of the line's words; they are a few same-aisle
    products, the cheapest the shopper can buy, so like the planner's t4 they come only from
    stores in range, and none is counted as unavailable."""
    from .nlsearch.sql_builder import build_options_sql, build_substitute_sql

    meanings = matcher.meanings([(ln.name, ln.form) for ln in lines])
    pools = {line.line_no: _Pool(meaning=m) for line, m in zip(lines, meanings, strict=True)}
    for line in lines:
        pool = pools[line.line_no]
        if pool.by_meaning:
            for c in matcher.candidates(pool.meaning, catalog):
                pool.found[c.product_id] = _LEVEL_OF[c.group]
                pool.cands[c.product_id] = c
    lines = [ln for ln in lines if not pools[ln.line_no].by_meaning]
    if not lines:
        return pools
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

    by_no = {ln.line_no: ln for ln in basis.lines}
    lines = [by_no[n] for n in sorted(set(line_nos))]
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
                  session: Session | None = None,
                  open_lines: bool = False,
                  forbid: frozenset[str] = frozenset()) -> dict[int, ValidPin]:
    """The basis's own pins merged with `pins` (a later pin for a line wins), each checked:
    the line is planned, the product exists, it is a candidate for the line (by meaning:
    the same item, works instead or a looser match; by words: any level, or a same-aisle
    substitute), no hard rule keeps it out (rule: it adds no contains-tag the plan's own
    product lacks, and contains nothing the plan's constraints or `forbid` leave out), the
    plan's origin exclusion does not hold it back, and it has an offer in range under the
    plan's price cap. A pin equal to the plan's own pick is dropped, which is how a swap is
    undone. Raises PinError naming the problem. `open_lines`: a pin may also fill a line of
    the basis the plan bought nothing for (a meal plan's line the resolve left out), checked
    the same way."""
    pins = list(pins or [])
    if len(pins) > MAX_PINS:
        raise PinError(f"at most {MAX_PINS} pins, got {len(pins)}")
    planned = ({ln.line_no: ln for ln in basis.lines} if open_lines
               else planned_lines(basis))
    merged: dict[int, int] = {}
    for p in [*basis.pins, *pins]:
        merged[p.line_no] = p.product_id
    if len(merged) > MAX_PINS:
        raise PinError(f"at most {MAX_PINS} pinned lines, got {len(merged)}")
    catalog = catalog if catalog is not None else (_catalog() if merged else {})
    for line_no, pid in merged.items():
        if line_no not in planned:
            raise LineNotPlannedError(_not_planned(line_no, planned), line_no)
        if pid not in catalog:
            raise PinError(f"unknown product id {pid} in pins", line_no)
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
            raise HeldBackPinError(f"{product.name} is evidenced as {_where(country, fld)}, "
                                   "which this plan excludes", line_no, pid, country, fld)
        if match is None and not _available(basis, offers.get(pid)) and pid not in near:
            # Not found because no store sells it at all (same-aisle substitutes are looked
            # up among offers): that is the reason, not the words.
            raise PinError(f"{product.name} cannot be bought for line {line_no} "
                           f"({line.name}): {_unavailable(basis, None)}", line_no)
        if match is None:
            raise PinError(f"{product.name} is not an option for line {line_no} "
                           f"({line.name}): "
                           f"{_why_not(basis, line, product, near.get(pid), g.pools[line_no])}",
                           line_no)
        broken = rule(product, catalog.get(line.product_id) if line.product_id else None,
                      set(basis.constraints.exclude_tags) | set(forbid), basis.constraints)
        if broken is not None:
            raise PinError(f"{product.name} is not an option for line {line_no} "
                           f"({line.name}): it {broken[2]}", line_no)
        if not _available(basis, offers.get(pid)):
            raise PinError(f"{product.name} cannot be bought for line {line_no} "
                           f"({line.name}): {_unavailable(basis, near.get(pid))}", line_no)
        out[line_no] = ValidPin(product_id=pid, tier=_TIER[match], match=match)
    return out


def _where(country: str, fld: str) -> str:
    return {"ingredient_origin": f"{country} (ingredients)",
            "manufactured_in": f"{country} (made or packed there)",
            "conflicting_evidence": f"{country} by some of its evidence"}.get(fld, country)


def _why_not(basis: PlanBasis, line: BasisLine, product: Product, anywhere: dict | None,
             pool: _Pool | None = None) -> str:
    """Why a product the shopper chose is not among the line's candidates, from its own
    catalog row: a diet, aisle or price limit of the plan, or it does not match (by meaning,
    or by the words)."""
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
    if pool is not None and pool.by_meaning:
        return (f"it is not the same item as {line.name!r}, does not work instead, and is not "
                "near it in meaning")
    return f"it does not match the words of {line.name!r}"


def contains(p: Product) -> set[str]:
    """The contains-tags a hard rule reads: the catalog's own (dietary_tags), with a seafood
    product's meat also read as fish (diets.product_contains), and honey when the name says
    it."""
    from .mealplan.diets import product_contains

    raw = {t.strip() for t in (p.dietary_tags or "").split(",") if t.strip()}
    return raw | product_contains(p) | ({"honey"} if "honey" in tokens(p.name) else set())


def _word(tag: str) -> str:
    """A contains-tag as a shopper reads it: tree_nuts is "tree nuts"."""
    return tag.replace("_", " ")


def _words(tags: list[str]) -> list[str]:
    """Tags as words, without "nuts" when peanuts or tree nuts say it more exactly."""
    exact = {"peanuts", "tree_nuts"} & set(tags)
    return [_word(t) for t in tags if not (t == "nuts" and exact)]


def _and(words: list[str]) -> str:
    return words[0] if len(words) == 1 else ", ".join(words[:-1]) + " and " + words[-1]


def rule(p: Product, original: Product | None, forbid: set[str],
         c=None) -> tuple[str, list[str], str] | None:
    """The hard rule that keeps `p` out of a line whose plan bought `original` (None for a
    line it bought nothing for), as (why, tags, words), or None when none does: it contains
    what `forbid` (the plan's diet, avoid list and excluded tags) leaves out; it lacks a tag
    or aisle the constraints `c` require; or it contains what `original` does not."""
    tags = contains(p)
    bad = sorted(tags & forbid)
    if bad:
        return "diet", bad, f"contains {_and(_words(bad))}, which this plan leaves out"
    if c is not None:
        own = {t.strip() for t in (p.dietary_tags or "").split(",") if t.strip()}
        lacking = [t for t in c.require_tags if t not in own]
        if lacking:
            return ("constraints", lacking,
                    f"is not marked {_and(lacking)}, which this plan requires")
        if ((c.categories and p.category not in c.categories)
                or (c.subcategories and p.subcategory not in c.subcategories)
                or p.category in c.exclude_categories
                or p.subcategory in c.exclude_subcategories):
            return "constraints", [], "is outside the aisles this plan is limited to"
    if original is not None and original.id != p.id:
        added = sorted(tags - contains(original))
        if added:
            return ("adds_tags", added,
                    f"contains {_and(_words(added))}, which {original.name} does not")
    return None


def _unavailable(basis: PlanBasis, nearest: dict | None) -> str:
    within = f" within {basis.max_km:g} km" if basis.max_km is not None else ""
    cap = basis.constraints.max_item_price
    if nearest is None:
        return f"no store{within} sells it" + (
            f" at or under the ${cap:.2f} per-item price cap" if cap is not None else "")
    return (f"no store{within} sells it"
            + (f" at or under the ${cap:.2f} per-item price cap" if cap is not None else "")
            + f"; the nearest offer is {nearest['store']}, {nearest['dist_km']:.1f} km away, "
              f"at ${nearest['price']:.2f}")


# ─── Offers ──────────────────────────────────────────────────

def _matrix(s: Session, basis: PlanBasis, ids: set[int], max_km: float | None,
            stores_off: frozenset[int] = frozenset()) -> list:
    from .nlsearch.sql_builder import build_price_matrix_sql

    lat, lon = _point(basis)
    sql, params = build_price_matrix_sql(sorted(ids), lat, lon, max_km,
                                         exclude_stores=sorted(stores_off))
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
    # why a row the trip effect was worked out for has none (TripEffect): "its price on your
    # plan's trips is unknown (...)"
    trip_note: str = ""
    # Matched by meaning: the row's group, its candidate facts and the matcher's score
    group: str = "outside"
    candidate: matcher.Candidate | None = None
    facts: matcher.Facts | None = None
    by_meaning: bool = False

    @property
    def score(self) -> float | None:
        return self.facts.score() if self.facts is not None else None

    def trip_cents(self, located: bool) -> float:
        if self.trip is not None:
            return round(self.trip.total * 100)
        if not located:
            cost = (self.cost_for_need if self.cost_for_need is not None
                    else self.offer.price * self.packs)
            return round(cost * 100)
        return math.inf

    def key(self, located: bool, use_pref: bool) -> tuple:
        if self.by_meaning:
            return (_GROUP_RANK[self.group], -(self.score or 0.0), self.product.id)
        rating = (0, -self.rating.avg) if self.rating is not None else (1, 0.0)
        cost = round(self.cost_for_need * 100) if self.cost_for_need is not None else math.inf
        return (_TIER_RANK[self.tier], self.semantic, _FIT_RANK[self.pack_fit],
                self.pref if use_pref else 0, self.trip_cents(located), cost, rating,
                self.product.id)

    def pre_key(self, use_pref: bool) -> tuple:
        if self.by_meaning:
            return (_GROUP_RANK[self.group], -(self.score or 0.0), self.product.id)
        return (_TIER_RANK[self.tier], self.semantic, _FIT_RANK[self.pack_fit],
                self.pref if use_pref else 0, self.product.id)


def _score(rows: list[_Row], need: tuple[float, str] | None, preference: list[str],
           located: bool) -> None:
    """The matcher's facts for every row matched by meaning (matching/matcher.py): relevance
    and form/variant from its candidate, pack fit (the packs the cart buys against the need),
    value from unit prices within its group, origin preference, and the trip total against
    its group's cheapest trip (0 when the trip is not worked out)."""
    for group in matcher.GROUPS:
        members = [r for r in rows if r.by_meaning and r.group == group]
        # Unit prices at the cheapest offer in range, as the planner's first pick reads them.
        value = matcher.values({id(r): matcher.unit_price(r.product, r.offer.price)
                                for r in members})
        # The trip total after the swap, or with no location what the cart pays for it.
        trips = [r.trip_cents(located) for r in members if r.trip_cents(located) != math.inf]
        best = min(trips) / 100 if trips else None
        smallest = matcher.smallest_pack([r.product for r in members], need)
        for r in members:
            r.facts = matcher.Facts(
                relevance=r.candidate.relevance, form_variant=r.candidate.form_variant,
                pack_fit=matcher.pack_fit(r.product, need, r.packs, smallest),
                value=value[id(r)],
                origin=matcher.origin_component(r.pref, preference),
                trip=matcher.trip_component(
                    r.trip_cents(located) / 100 if r.trip_cents(located) != math.inf else None,
                    best))


def _trip_of(options) -> tuple[float, list[str], dict[int, str]] | None:
    best = next((o for o in options if o.recommended), None)
    if best is None:
        return None
    return best.total_cost, list(best.stores), {i.product_id: i.store_name for i in best.items}


@dataclass
class TripEffect:
    """The trip effect of another planner, in place of the cart's single trip: the meal
    plan's trips (mealplan/options.py). The ranking keeps everything else (candidates, held
    back, availability, facts, ORDER, reasons); only these come from the other planner:

    needs      line_no -> the amount the purchase is for on that planner's trip, not the
               recipe's one batch;
    on_trip    the stores that trip stops at now;
    packs      the packs that planner buys of a product for `needs`, before its effect is
               worked out;
    effect     product_id -> (the trip with that product on every line of the purchase,
               the packs then bought, what that planner then charges for them, and why
               there is no trip when there is none): its own figures, so a row says what
               choosing it costs there;
    where      how the reasons name the trip total ("your plan's trips");
    stores_off the stores that planner's shopper turned off: no offer of theirs counts, so a
               candidate sold in range only there is unavailable;
    forbid     the contains-tags that planner's shopper leaves out (the plan's diet and avoid
               list, and for a line bought nothing for, the recipe's own diets): no
               candidate that contains one is choosable (rule).

    The current pick may have no offer in range any more (a line no longer stocked, the
    reason to look for another product): it is then left out of the rows instead of making
    the ranking stale."""
    needs: dict[int, tuple[float, str] | None]
    on_trip: set[str]
    packs: Callable[[Product, list], int]
    effect: Callable[[int], tuple[AltTrip | None, int | None, float | None, str]]
    where: str = "your trip"
    max_eval: int = MAX_TRIP_EVAL
    stores_off: frozenset[int] = frozenset()
    forbid: frozenset[str] = frozenset()


def rank_alternatives(basis: PlanBasis, line_no: int, limit: int = DEFAULT_LIMIT, *,
                      effect: TripEffect | None = None,
                      open_lines: bool = False) -> AlternativeRanking:
    """The other products that could fill `line_no`, in order (ORDER_MEANING, or ORDER when
    the line is matched by words), with the cart's pick always included and flagged
    current, unless the plan's origin exclusion now holds the pick back: it is then the first
    of held_back, and no row is current. Deterministic for the same basis and database; no
    LLM, no write. Raises BasisError or PinError (a bad basis, an unplanned line, a bad
    pin). `effect`: another planner's trips stand in for the cart's (TripEffect).

    `open_lines`: `line_no` may be a line the plan bought nothing for. It then has no
    current pick: the rows are every product that could fill it (and each unpicked line of
    the same name, which one choice fills too), none flagged current."""
    from . import flow
    from .matching import events
    from .origins import preference_rank

    check_basis(basis)
    if not 1 <= limit <= MAX_LIMIT:
        raise BasisError(f"limit must be 1..{MAX_LIMIT}, got {limit}")
    planned = ({ln.line_no: ln for ln in basis.lines} if open_lines
               else planned_lines(basis))
    if line_no not in planned:
        raise LineNotPlannedError(_not_planned(line_no, planned))
    catalog = _catalog()
    forbid = set(basis.constraints.exclude_tags) | set(effect.forbid if effect else ())
    with Session(db.engine()) as s:
        valid = validate_pins(basis, [], catalog=catalog, session=s,
                              forbid=frozenset(effect.forbid) if effect else frozenset())
        picks = effective_picks(basis, {n: v.product_id for n, v in valid.items()})
        current_id = picks.get(line_no)
        if current_id is None:
            name = " ".join(planned[line_no].name.casefold().split())
            group = sorted(n for n, ln in planned.items() if n not in picks
                           and " ".join(ln.name.casefold().split()) == name)
        else:
            group = sorted(n for n, pid in picks.items() if pid == current_id)
        unknown = sorted({pid for pid in picks.values() if pid not in catalog})
        if unknown:
            raise StaleBasisError(f"product id(s) {unknown} in the basis are no longer in "
                                  "the catalog; plan the recipe again")
        g = _gather(s, basis, group, catalog, picks,
                    set() if current_id is None else {current_id})
        pool = g.pools[line_no]
        by_meaning = all(g.pools[n].by_meaning for n in group)
        # A shared purchase is swapped for every line it covers, so a candidate must fill
        # each of them; it carries the loosest of its matches.
        found: dict[int, str] = dict(g.pools[group[0]].found)
        for n in group[1:]:
            other = g.pools[n].found
            if by_meaning:
                found = {pid: max(m, other[pid], key=lambda lv: _GROUP_RANK[_GROUP_OF[lv]])
                         for pid, m in found.items() if pid in other}
            else:
                found = {pid: max(m, other[pid], key=_MATCH_RANK.__getitem__)
                         for pid, m in found.items() if pid in other}
        held = sorted((pid for pid in pool.found if pid in g.held),
                      key=lambda pid: (catalog[pid].name, pid))
        # The pick itself when the exclusion now holds it back (evidence added since it was
        # planned or pinned, or the rule turned on after it; a meal plan's Ingredients stage
        # keeps such a pin): no row, so none is current, and the first product held back, as
        # the basket names it first on its line.
        held_pick = current_id is not None and current_id in g.held
        if held_pick:
            held = [current_id, *(pid for pid in held if pid != current_id)]
        selectable = {pid: m for pid, m in found.items() if pid not in g.held}
        # The hard rules (rule): never choosable, and said why.
        original = catalog.get(planned[line_no].product_id) \
            if planned[line_no].product_id is not None else None
        ruled: dict[int, tuple[str, list[str], str]] = {}
        for pid in list(selectable):
            broken = (rule(catalog[pid], original, forbid, basis.constraints)
                      if pid != current_id else None)
            if broken is not None:
                ruled[pid] = broken
                del selectable[pid]
        if current_id is not None and current_id not in selectable and not held_pick:
            selectable[current_id] = "outside"
        stats = _stats(s, sorted(selectable))
        located = has_location(basis)
        basket = set(picks.values())
        rows_all = _matrix(s, basis, basket | set(selectable), basis.max_km,
                           effect.stores_off if effect else frozenset()) if located else []

    offers = flow._cheapest_offers(rows_all, basis.path) if located else {}
    if located and current_id is not None and current_id not in offers:
        if effect is None:
            raise StaleBasisError(f"{catalog[current_id].name}, the cart's pick for line "
                                  f"{line_no}, has no offer in range any more; plan the "
                                  "recipe again")
        # Another planner's line no longer stocked: its options are what the shopper needs.
        selectable.pop(current_id, None)
    unavailable = [pid for pid in selectable
                   if pid != current_id and not _available(basis, offers.get(pid))]
    for pid in unavailable:
        del selectable[pid]

    def best_first(pid: int) -> tuple:
        if by_meaning:
            c = pool.cands.get(pid)
            return (_GROUP_RANK[_GROUP_OF[found[pid]]], -(c.relevance if c else 0.0), pid)
        return (_TIER_RANK[_TIER[found[pid]]], closeness(planned[line_no].name, catalog[pid]),
                pid)

    reach = []
    if effect is not None and located and unavailable:
        reach = _out_of_reach(basis, sorted(unavailable, key=best_first)[:limit], catalog,
                              effect.stores_off)

    synthetic = settings().offers_synthetic
    # By words, a row whose words match but which is another food moves to the other tier.
    word = _strong_head(planned[line_no].name)
    aisle = _aisle_of(word, catalog.get(current_id) if current_id is not None else None,
                      pool.found, catalog)
    needs = [effect.needs.get(n) for n in group] if effect else _needs(basis, group)
    lines = [planned[n] for n in group]
    library = basis.path == "library"
    use_pref = bool(basis.preference)
    worst = len(basis.preference) * 2
    if effect is None:
        baseline = _price(basis, picks, catalog, rows_all)
        base_trip = _trip_of(baseline[1]) if located else None
        base_stores = set(base_trip[1]) if base_trip else set()
    else:
        base_trip, base_stores = None, set(effect.on_trip)

    rows: list[_Row] = []
    for pid, match in selectable.items():
        p = catalog[pid]
        if located:
            o = offers[pid]
            offer = AltOffer(store=o["store"], price=o["price"], distance_km=o["dist_km"],
                             on_trip=o["store"] in base_stores)
        else:
            offer = AltOffer(store="", price=p.price)
        cart_packs = effect.packs(p, needs) if effect else flow._packs(p, needs)
        fit, cost, pack_reason = _pack_facts(p, offer.price, cart_packs, needs, lines, library)
        unit_price, unit_basis = _unit_price(p, offer.price)
        st = stats.get(pid)
        rating = (AltRating(avg=st[0], count=st[1], synthetic=synthetic)
                  if st is not None and st[1] > 0 and st[0] is not None else None)
        pref = preference_rank(g.origins.get(pid), basis.preference) if use_pref else worst
        tier = ("other" if not by_meaning and _TIER[match] == "same"
                and _another_food(p, word, aisle) else _TIER[match])
        rows.append(_Row(
            product=p, match=match, tier=tier, current=pid == current_id,
            offer=offer, packs=cart_packs, pack_fit=fit, cost_for_need=cost,
            pack_reason=pack_reason, unit_price=unit_price, unit_basis=unit_basis,
            origin=_alt_origin(g.origins.get(pid), pid in g.demo),
            pref=pref, preferred=pref < worst,
            rating=rating, organic="organic" in tokens(f"{p.name} {p.description}"),
            semantic=closeness(planned[line_no].name, p),
            group=("looser_match" if tier == "other" and _TIER[match] == "same"
                   else _GROUP_OF[match]),
            candidate=pool.cands.get(pid) if match != "outside" else None,
            by_meaning=by_meaning))

    need = _total_need(needs)
    if by_meaning:
        _score(rows, need, list(basis.preference), located)
    max_eval = effect.max_eval if effect else MAX_TRIP_EVAL
    if located and (base_trip is not None or effect is not None):
        rows.sort(key=lambda r: r.pre_key(use_pref))
        to_eval = rows[:max_eval] + [r for r in rows[max_eval:] if r.current]
        for r in to_eval:
            charged: float | None = None
            if effect is not None:
                r.trip, packs_after, charged, r.trip_note = effect.effect(r.product.id)
            else:
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
            if effect is not None:
                # What the other planner charges for the packs it buys, never a figure of
                # this module's own (unknown when it leaves the packs unknown).
                r.cost_for_need = charged
            r.unit_price, r.unit_basis = _unit_price(r.product, price)
        if by_meaning:
            _score(rows, need, list(basis.preference), located)

    rows.sort(key=lambda r: r.key(located, use_pref))
    where = effect.where if effect else "your trip"
    items = [_item(i + 1, r, rows[i - 1] if i else None, planned[line_no], located, use_pref,
                   synthetic, where=where, max_eval=max_eval, original=original,
                   meaning=pool.meaning)
             for i, r in enumerate(rows)]
    shown = items[:limit] + [it for it in items[limit:] if it.current]
    ruled_rows = sorted(ruled, key=best_first)[:limit]
    ranking = AlternativeRanking(
        line_no=line_no, lines=group, ingredient=" + ".join(ln.name for ln in lines),
        need=_fmt_qty(*need) if need else "", need_note=_no_need(needs, lines, library),
        need_qty=need[0] if need else None, need_uom=need[1] if need else None,
        order=list(ORDER_MEANING if by_meaning else ORDER),
        ranking_text=RANKING_TEXT_MEANING if by_meaning else RANKING_TEXT,
        items=shown,
        held_back=[_held(pid, catalog, g) for pid in held], out_of_reach=reach,
        total=len(items), unavailable=len(unavailable),
        counts=AltCounts(
            exact=sum(1 for it in items if it.match == "exact"),
            no_new_stop=sum(1 for it in items if it.trip is not None
                            and it.trip.stops_delta <= 0),
            preferred_origin=sum(1 for r in rows if r.preferred),
            says_organic=sum(1 for it in items if it.says_organic),
            rated=sum(1 for it in items if it.rating is not None),
            same_item=sum(1 for it in items if it.group == "same_item"),
            works_instead=sum(1 for it in items if it.group == "works_instead"),
            looser_match=sum(1 for it in items if it.group == "looser_match")),
        data_note=DATA_NOTE if synthetic else "",
        matching="meaning" if by_meaning else "words",
        matching_note=(MEANING_NOTE if by_meaning else WORDS_NOTE.format(
            why=next((g.pools[n].meaning.why_words for n in group
                      if g.pools[n].meaning is not None and g.pools[n].meaning.why_words),
                     "this name is not one the matcher knows"))),
        filtered=[Filtered(product_id=pid, product=catalog[pid].name,
                           group=_GROUP_OF[found[pid]] if _GROUP_OF[found[pid]] != "outside"
                           else "looser_match", why=ruled[pid][0], tags=ruled[pid][1],
                           text=f"{catalog[pid].name} {ruled[pid][2]}")
                  for pid in ruled_rows])
    events.shown(ranking)
    return ranking


def _out_of_reach(basis: PlanBasis, pids: list[int], catalog: dict[int, Product],
                  stores_off: frozenset[int]) -> list[OutOfReach]:
    """Why each of these candidates has no offer in range at a store that is on, from every
    store's offers (one price-matrix query): stores_off when every offer within the plan's
    distance is at a store the shopper turned off (those stores, nearest first), out_of_range
    when none is within it (the nearest offer, at a store that is on when one sells it), and
    not_stocked when no store sells it. One over the price cap at a store in range is left
    out: it is neither."""
    with Session(db.engine()) as s:
        rows = _matrix(s, basis, set(pids), None)
    by_pid: dict[int, list] = {}
    for r in sorted(rows, key=lambda r: (r["dist_km2"], r["price"], r["store_id"])):
        by_pid.setdefault(r["product_id"], []).append(r)
    reach2 = basis.max_km ** 2 if basis.max_km is not None else math.inf
    out = []
    for pid in pids:
        sold = by_pid.get(pid, [])
        near = [r for r in sold if r["dist_km2"] <= reach2]
        if near and all(r["store_id"] in stores_off for r in near):
            out.append(OutOfReach(product_id=pid, product=catalog[pid].name, why="stores_off",
                                  stores=list(dict.fromkeys(r["store_name"] for r in near))))
        elif sold and not near:
            r = next((r for r in sold if r["store_id"] not in stores_off), sold[0])
            out.append(OutOfReach(
                product_id=pid, product=catalog[pid].name, why="out_of_range",
                nearest=AltBuy(store=r["store_name"], price=r["price"],
                               distance_km=round(math.sqrt(max(r["dist_km2"], 0.0)), 1))))
        elif not sold:
            out.append(OutOfReach(product_id=pid, product=catalog[pid].name, why="not_stocked"))
    return out


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
          use_pref: bool, synthetic: bool, *, where: str = "your trip",
          max_eval: int = MAX_TRIP_EVAL, original: Product | None = None,
          meaning: matcher.Meaning | None = None) -> RankedAlternative:
    p = r.product
    return RankedAlternative(
        rank=rank, current=r.current, product_id=p.id, product=p.name, brand=p.brand,
        size=p.unit_size, tier=r.tier, match=r.match, offer=r.offer, packs=r.packs,
        pack_fit=r.pack_fit, cost_for_need=r.cost_for_need, unit_price=r.unit_price,
        unit_basis=r.unit_basis, trip=r.trip, origin=r.origin, rating=r.rating,
        says_organic=r.organic, reasons=_reasons(r, line, located, synthetic, where),
        rank_reason=_rank_reason(rank, r, above, line, located, use_pref, synthetic,
                                 where=where, max_eval=max_eval, meaning=meaning),
        group=r.group, group_label=GROUP_LABELS[r.group],
        score=round(r.score, 4) if r.score is not None else None,
        summary=_summary(r, original))


def _type_of(p: Product) -> str:
    from .matching import seeds

    prof = seeds.load().profiles.get(p.id)
    return prof.product_type if prof is not None else p.name.lower()


def _summary(r: _Row, original: Product | None) -> str:
    """The row on one line: what it is, how its pack meets the need, and its allergens
    against the plan's own product: "butter · 454 g covers 80 g · no new allergens (11
    checked)"."""
    p = r.product
    size = p.unit_size or (_fmt_qty(p.unit_qty, p.unit_uom) if p.unit_qty else "")
    pack = r.pack_reason.text
    if r.pack_fit == "covers" and r.pack_reason.text.startswith("Covers the recipe's "):
        need = r.pack_reason.text.removeprefix("Covers the recipe's ").split(" in ")[0]
        pack = f"{size} covers {need}" if size else f"covers {need}"
    elif r.pack_fit == "short" and size:
        pack = f"{size} falls short"
    tags = sorted(contains(p) & set(_TAGS))
    if original is not None:
        added = sorted(set(tags) - contains(original))
        allergens = (_CHECKED if not added
                     else f"adds {_and(_words(added))}")
    else:
        allergens = (f"contains {_and(_words(tags))}" if tags
                     else "none of 11 allergens checked")
    return " · ".join(x for x in (_type_of(p), pack[:1].lower() + pack[1:], allergens) if x)


def _money(cents: float) -> str:
    return f"${abs(cents) / 100:.2f}"


def _reasons(r: _Row, line: BasisLine, located: bool, synthetic: bool,
             where: str = "your trip") -> list[Reason]:
    """At most five reasons, each from a fact on the row: the match, the pack, the origin,
    the trip and the rating. Unknowns are stated as unknown."""
    p = r.product
    head = _head_word(line.name)
    if r.by_meaning:
        return _meaning_reasons(r, line, located, synthetic, where)
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
    if r.tier == "other" and _TIER[r.match] == "same":
        word = _strong_head(line.name)
        match = Reason(code="match", text=f"Mentions {word}, but it is "
                       f"{_what_it_is(p, word)}; not the same ingredient", tone="minus")
    return _fact_reasons(r, match, located, synthetic, where)


def _meaning_reasons(r: _Row, line: BasisLine, located: bool, synthetic: bool,
                     where: str) -> list[Reason]:
    """The reasons of a row matched by meaning: its group first, then the same facts."""
    kind = _type_of(r.product)
    match = {
        "same_item": Reason(code="match", text=f"Same item: {kind}", tone="plus"),
        "works_instead": Reason(code="match",
                                text=f"Works instead of {line.name}: {kind}; not the same item",
                                tone="info"),
        "looser_match": Reason(code="match",
                               text=f"Looser match: {kind}, near {line.name} in meaning; not "
                                    "the same item", tone="minus"),
        "outside": Reason(code="match",
                          text=f"The cart's pick; not a candidate for {line.name}", tone="info"),
    }[r.group]
    return _fact_reasons(r, match, located, synthetic, where)


def _fact_reasons(r: _Row, match: Reason, located: bool, synthetic: bool,
                  where: str) -> list[Reason]:
    o = r.origin
    origin_text = o.label + (f': "{o.verbatim}"' if o.verbatim else "") + (
        " (demo label photo)" if o.demo else "")
    origin_tone = "unknown" if o.status != "resolved" else "plus" if r.preferred else "info"
    out = [match, r.pack_reason, Reason(code="origin", text=origin_text, tone=origin_tone)]
    if r.trip is not None:
        t = r.trip
        if t.delta < 0:
            text_, tone = f"${-t.delta:.2f} less on {where}", "plus"
        elif t.delta > 0:
            text_, tone = f"${t.delta:.2f} more on {where}", "minus"
        else:
            text_, tone = ("No change to your trip total" if where == "your trip"
                           else f"No change to the total of {where}"), "info"
        if t.stops_delta > 0:
            text_ += f" and {t.stops_delta} more stop{'s' if t.stops_delta > 1 else ''}"
            tone = "minus"
        elif t.stops_delta < 0:
            text_ += f" and {-t.stops_delta} stop{'s' if t.stops_delta < -1 else ''} fewer"
        if t.merges_with_line is not None:
            text_ += f"; already bought for line {t.merges_with_line}"
        out.append(Reason(code="trip", text=text_, tone=tone))
    elif located:
        out.append(Reason(code="trip", text=(r.trip_note[:1].upper() + r.trip_note[1:]
                                             if r.trip_note else "Trip effect not worked out"),
                          tone="unknown"))
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
                 use_pref: bool, synthetic: bool, *, where: str = "your trip",
                 max_eval: int = MAX_TRIP_EVAL, meaning: matcher.Meaning | None = None) -> str:
    """Why `r` sits below the row above it: the first ranking key on which it is worse,
    in plain words; by meaning, its group, or the score component that costs it most."""
    if above is None:
        return "Ranked first: nothing ranks above it."
    if r.by_meaning:
        why = _meaning_difference(r, above, line, located, where=where, max_eval=max_eval,
                                  meaning=meaning)
    else:
        why = _first_difference(r, above, line, located, use_pref, synthetic, where=where,
                                max_eval=max_eval)
    return f"Below #{rank - 1}: {why}."


_FIT_WORDS = {"short": "its packs fall short of the recipe's amount",
              "catering": "its pack is many times the recipe's amount",
              "unknown": "its pack can't be compared with the recipe's amount"}


def _meaning_difference(r: _Row, above: _Row, line: BasisLine, located: bool, *,
                        where: str, max_eval: int, meaning: matcher.Meaning | None) -> str:
    if _GROUP_RANK[r.group] > _GROUP_RANK[above.group]:
        return {"works_instead": "it works instead; it is not the same item",
                "looser_match": "it is a looser match: near in meaning, but not the same item "
                                "and not known to work instead",
                "outside": "it is not a candidate for the line"}[r.group]
    if r.facts is None or above.facts is None or r.score == above.score:
        return "it ties on every count; listed by catalog number"
    # The component that costs it most against the row above, weighted.
    gaps = {k: w * (getattr(above.facts, k) - getattr(r.facts, k))
            for k, w in matcher.WEIGHTS.items()}
    worst = max(gaps, key=lambda k: (gaps[k], k))
    scores = f" (score {r.score:.2f} against {above.score:.2f})"
    if worst == "relevance":
        why = f"it is further from {line.name} in meaning"
    elif worst == "form_variant":
        asked = " ".join(x for x in (meaning.form, meaning.variant) if x) if meaning else ""
        why = (f"it is not the {asked} the recipe asks for" if asked
               else "it is not the form or variant the recipe asks for")
    elif worst == "pack_fit":
        fit = {0.3: "short", 0.25: "catering", 0.5: "unknown"}.get(r.facts.pack_fit, "")
        why = _FIT_WORDS.get(fit, "its packs are further from the recipe's amount")
    elif worst == "value":
        why = (f"it costs more per {r.unit_basis}" if r.unit_basis
               else "its price per unit is unknown")
    elif worst == "origin":
        why = "its origin is further down your preference"
    else:
        mine, theirs = r.trip_cents(located), above.trip_cents(located)
        if mine == math.inf:
            why = r.trip_note or f"its trip effect was not worked out (only the first " \
                                 f"{max_eval} are)"
        else:
            why = f"{_money(mine - theirs)} more on {where}"
    return why + scores


def _first_difference(r: _Row, above: _Row, line: BasisLine, located: bool, use_pref: bool,
                      synthetic: bool, *, where: str = "your trip",
                      max_eval: int = MAX_TRIP_EVAL) -> str:
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
        if mine == math.inf and r.trip_note:
            return r.trip_note
        if mine == math.inf:
            return f"its trip effect was not worked out (only the first {max_eval} are)"
        if located:
            return f"{_money(mine - theirs)} more on {where}"
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

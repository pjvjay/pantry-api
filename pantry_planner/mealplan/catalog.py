"""POST /mealplan/catalog: the recipes the meal plan wizard offers in its Meals stage, each with
an estimate of what one meal costs, before anything is resolved. Pure code: no LLM call, in
demo mode or live, and no write.

The recipes are the library's, the demo starters and the shopper's own (RecipeDocs sent with the
request). Each line's product is the one resolve picks in demo mode: the resolve's own
retrieval and origin filter (flow's load_recipe and load_products for a library recipe, and
parse_and_retrieve for a RecipeDoc, called as functions), then the demo selector
(demomode.select_products: shared words, head noun, origin preference, price). Live, the
resolve's selector is a model and may choose another product in the Ingredients stage; the
estimate is this code's, and est_basis says what it rests on.

A line is priced at its product's cheapest offer in range, from the price matrix the trips
read, as its share of the pack: the amount the recipe uses for the household over the pack
size, times the pack price. A line with no amount in the pack's unit is not priced, and
lines_priced says how many were. That share is less than a shopper pays, since packs are
bought whole, so each recipe also lists its lines (product, pack price, pack size and the need
per meal in the pack's unit) for the console to cost any selection by whole packs, the rule
/mealplan/basket buys by (whole_packs): per product, the summed need over the pack size
rounded up, and
at least one pack when a line's need or the pack size is unknown (one in all, however many
lines share the product).

An ingredient no store sells, that no store in range sells, or whose every candidate the
origin exclusion holds back is named in `unavailable` and is never priced: an excluded product
is never picked.

A recipe fits when it satisfies the diet and contains nothing the shopper avoids (diets.py).
The starters and the library carry their diets in the starters file; the shopper's own recipes
are classified from their picks when every line they buy has one, and are unknown otherwise:
diets and contains empty, and fits only when there is no diet and nothing to avoid.
"""
from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field

from burr.core import State

from .. import db, demomode, flow
from ..models import Product, RecipeDoc
from ..nlsearch.planner import PlanAborted, UnparseableRecipe
from ..nlsearch.schemas import Constraints, ParsedInput
from ..nlsearch.units import is_non_purchase, normalize_quantity
from ..recipe_doc import UnconfirmedLines, to_recipe_text, to_spec
from . import diets, schedule
from .models import (
    SLOTS,
    Avoid,
    Catalog,
    CatalogLine,
    CatalogRecipe,
    Diet,
    PlanSettings,
    RecipeRef,
    Slot,
)
from .resolve import recipe_doc, starters_file

# The shopper's own recipes one request may classify and price.
MAX_OWN_RECIPES = 24
NOT_KNOWN = "its diet is not known: not every ingredient has a product to check"


def est_basis(servings: int) -> str:
    people = "1 person" if servings == 1 else f"{servings} people"
    return (f"est_cost_per_meal is one meal for {people}, each ingredient's share of its pack "
            "price at the cheapest store in range (what the recipe uses over the pack size), "
            "while a selection costs whole packs, as the basket's est_total counts them: per "
            "product the total need over the pack size rounded up, and at least one pack when "
            "a line's need or the pack size is unknown (one in all, however many lines share "
            "it), at the line's pack_price.")


def whole_packs(needs: list[float | None], pack_qty: float | None) -> int:
    """The wizard's whole-pack rule for one product bought for several lines: the summed need
    of the lines whose need is known in its pack's unit, over the pack size, rounded up; and at
    least one pack when a line's need, or the pack size, is unknown. That pack is one in all:
    salt "to taste" in three recipes is one bag of salt, not three."""
    known = [n for n in needs if n is not None] if pack_qty else []
    packs = max(0, math.ceil(sum(known) / pack_qty - 1e-9)) if known else 0
    return max(packs, 1) if len(known) < len(needs) else packs


# Why a line has no product: no store sells it, no store in range does, or the plan's origin
# exclusion holds back every candidate in range.
ORIGIN_EXCLUDED, NOT_STOCKED, OUT_OF_RANGE = "origin_excluded", "not_stocked", "out_of_range"
_GATE_KIND = {"excluded_by_origin": ORIGIN_EXCLUDED, "missing_ingredients": NOT_STOCKED,
              "unavailable_within_constraints": OUT_OF_RANGE}


@dataclass
class Picks:
    """One recipe's demo-mode picks, line_no -> product id. A line the resolve's retrieval
    left out has none, and left_out says why: line_no -> (kind, the planner's reason), kind
    one of ORIGIN_EXCLUDED, NOT_STOCKED, OUT_OF_RANGE. checked is False when the recipe could
    not be planned at all (an unconfirmed or empty recipe), so nothing about its lines is
    known."""
    by_line: dict[int, int] = field(default_factory=dict)
    left_out: dict[int, tuple[str, str]] = field(default_factory=dict)
    checked: bool = True


def _retrieve(ref: RecipeRef, doc: RecipeDoc, s: PlanSettings, exclude: list[str]) -> State:
    """The resolve's retrieval and origin filter, as its flow runs them."""
    if ref.slug is not None:
        _r, st = flow.load_recipe(State(), recipe_slug=ref.slug)
        _r, st = flow.load_products(st, exclude=exclude, preference=list(s.preference),
                                    allow_partial=True)
        return st
    parsed = ParsedInput(recipe=to_spec(doc), constraints=Constraints())
    _r, st = flow.parse_and_retrieve(
        State(), recipe_text=to_recipe_text(doc), lat=s.lat, lon=s.lon, exclude=exclude,
        preference=list(s.preference), max_km=s.max_km, allow_partial=True, parsed=parsed)
    return st


def _lines_left_out(doc: RecipeDoc, by_line: dict[int, int],
                    why: dict[str, tuple[str, str]]) -> dict[int, tuple[str, str]]:
    return {ln.line_no: why[ln.name] for ln in doc.lines
            if ln.line_no not in by_line and ln.name in why}


def picks(ref: RecipeRef, doc: RecipeDoc, s: PlanSettings) -> Picks:
    """What resolve buys for each line in demo mode, under the plan's shopping point and origin
    rules, with no LLM in either mode, and why a line it leaves out is left out. A library
    recipe's planner leaves a line out only for the origin exclusion; a RecipeDoc's leaves one
    out of range for either reason, so the same retrieval without the exclusion tells them
    apart: what it keeps, the exclusion emptied."""
    exclude = list(s.exclude_origin)
    try:
        st = _retrieve(ref, doc, s, exclude)
    except (UnconfirmedLines, UnparseableRecipe):
        return Picks(checked=False)
    except PlanAborted as e:
        # A gate left nothing to plan: every line it names is unavailable, for its reason.
        alert = e.execution.aborted
        kind = _GATE_KIND.get(alert.code.value, OUT_OF_RANGE) if alert else OUT_OF_RANGE
        return Picks(left_out=_lines_left_out(doc, {}, {
            d["name"]: (kind, d["reason"]) for d in (alert.details if alert else [])}))
    chosen = demomode.select_products(
        st["recipe"].ingredients, st["products"], model="demo-deterministic",
        origins_by_id=st.get("origins") or {}, preference=st.get("preference") or [],
        substitutes=st.get("substitutes"))
    by_line = {sel.line_no: sel.product_id for sel in chosen.selections}
    why = {d.ingredient: (NOT_STOCKED, d.reason) for d in st.get("not_stocked") or []}
    for d in st.get("out_of_range") or []:
        why.setdefault(d.ingredient,
                       (ORIGIN_EXCLUDED if ref.slug is not None else OUT_OF_RANGE, d.reason))
    if ref.slug is None and exclude and any(k == OUT_OF_RANGE for k, _r in why.values()):
        try:
            base = _retrieve(ref, doc, s, [])
            still = {d.ingredient for d in [*(base.get("not_stocked") or []),
                                            *(base.get("out_of_range") or [])]}
        except PlanAborted:
            still = set(why)
        why = {n: (ORIGIN_EXCLUDED, r) if k == OUT_OF_RANGE and n not in still else (k, r)
               for n, (k, r) in why.items()}
    return Picks(by_line=by_line, left_out=_lines_left_out(doc, by_line, why))


# Picks depend only on the recipe and the plan's shopping point and origin rules, not on the
# diet, the household or the budget the shopper changes in this stage: cached in process.
_CACHE_SIZE = 256
_cache: OrderedDict[str, Picks] = OrderedDict()
_cache_lock = threading.Lock()


def clear_cache() -> None:
    with _cache_lock:
        _cache.clear()


def cached_picks(ref: RecipeRef, doc: RecipeDoc, s: PlanSettings) -> Picks:
    body = {"ref": ref.model_dump(mode="json"), "lat": s.lat, "lon": s.lon,
            "max_km": s.max_km, "exclude": list(s.exclude_origin),
            "preference": list(s.preference)}
    key = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    with _cache_lock:
        hit = _cache.get(key)
        if hit is not None:
            _cache.move_to_end(key)
            return hit
    out = picks(ref, doc, s)
    with _cache_lock:
        _cache[key] = out
        while len(_cache) > _CACHE_SIZE:
            _cache.popitem(last=False)
    return out


def _recipes(own: list[RecipeRef]) -> list[RecipeRef]:
    """The library, the demo starters, then the shopper's own (a key twice is one recipe)."""
    out = [RecipeRef(key=f"lib:{r.slug}", slug=r.slug) for r in db.load_all_recipes()]
    out += [RecipeRef(key=s["doc_key"], starter=s["key"]) for s in starters_file()["starters"]]
    seen = {r.key for r in out}
    for ref in own[:MAX_OWN_RECIPES]:
        if ref.key not in seen:
            seen.add(ref.key)
            out.append(ref)
    return out


def _share(line, product: Product, price: float, scale: float) -> float | None:
    """The line's share of the pack price for one meal, or None when its amount is not in
    the pack's unit."""
    need = normalize_quantity(line.quantity, line.unit)
    if need is None or not product.unit_qty or need[1] != product.unit_uom:
        return None
    return need[0] * scale / product.unit_qty * price


def catalog(*, household_servings: int, diet: Diet, avoid: list[Avoid], settings: PlanSettings,
            slots: list[Slot] | None = None, own: list[RecipeRef] | None = None) -> Catalog:
    """Every recipe in the given slots (all of them by default), with its diets, whether it
    fits, and its estimate. Raises MealPlanError (unknown_recipe_key) for a library slug or
    starter the shopper's list names that does not exist."""
    t0 = time.perf_counter()
    refs = _recipes(own or [])
    metas = {ref.key: diets.seed_meta(ref) for ref in refs}
    slot_of = {ref.key: (metas[ref.key] or {}).get("slot", "dinner") for ref in refs}
    refs = [ref for ref in refs if slots is None or slot_of[ref.key] in slots]
    docs = {ref.key: recipe_doc(ref) for ref in refs}
    found = {ref.key: cached_picks(ref, docs[ref.key], settings) for ref in refs}
    products = {p.id: p for p in db.load_all_products()}
    lat, lon, max_km = schedule.shopping_point(settings)
    ids = sorted({pid for pk in found.values() for pid in pk.by_line.values()} & products.keys())
    offers = flow._cheapest_offers(schedule._offers(ids, lat, lon, max_km), "spec")

    out = []
    for ref in refs:
        doc, pk, meta = docs[ref.key], found[ref.key], metas[ref.key]
        bought = [ln for ln in doc.lines if not is_non_purchase(ln.name)]
        scale = household_servings / doc.servings if doc.servings else None
        shares: list[float] = []
        unavailable: list[str] = []
        held_back: list[str] = []
        lines: list[CatalogLine] = []
        for ln in bought:
            pid = pk.by_line.get(ln.line_no)
            p = products.get(pid)
            offer = offers.get(pid) if p is not None else None
            held = pk.left_out.get(ln.line_no, ("",))[0] == ORIGIN_EXCLUDED
            need = normalize_quantity(ln.quantity, ln.unit)
            lines.append(CatalogLine(
                ingredient=ln.name, product_id=pid if p is not None else None,
                pack_price=offer["price"] if offer else None,
                pack_qty=(p.unit_qty or None) if p is not None else None,
                pack_uom=(p.unit_uom or None) if p is not None else None,
                need_per_meal=(round(need[0] * scale, 4) if p is not None and need is not None
                               and scale is not None and need[1] == p.unit_uom else None),
                held_back=held))
            if offer is None:
                if pk.checked:
                    unavailable.append(ln.name)
                if held:
                    held_back.append(ln.name)
                continue
            share = (None if scale is None
                     else _share(ln, products[pid], offer["price"], scale))
            if share is not None:
                shares.append(share)
        if meta is not None:
            known, dts, contains = True, meta["diets"], meta["contains"]
        elif pk.checked and all(ln.line_no in pk.by_line for ln in bought):
            dts, contains = diets.classify((ln.name, products[pk.by_line[ln.line_no]])
                                           for ln in bought)
            known = True
        else:
            known, dts, contains = False, [], []
        if known:
            fits, why_not = diets.fits(contains, diet, avoid)
        else:
            fits = diet == "any" and not avoid
            why_not = None if fits else NOT_KNOWN
        out.append(CatalogRecipe(
            key=ref.key, ref=ref, title=doc.title, slot=slot_of[ref.key], diets=dts,
            contains=contains, servings=doc.servings, fits=fits, why_not=why_not,
            est_cost_per_meal=round(sum(shares), 2) + 0.0 if shares else None,
            lines_total=len(bought), lines_priced=len(shares), unavailable=unavailable,
            held_back=held_back, lines=lines))
    out.sort(key=lambda r: (SLOTS.index(r.slot), r.title.casefold(), r.key))
    return Catalog(recipes=out, est_basis=est_basis(household_servings),
                   latency_ms=int((time.perf_counter() - t0) * 1000))

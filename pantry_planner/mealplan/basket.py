"""POST /mealplan/basket: the meal plan wizard's Ingredients stage. Every chosen recipe's
ingredients, each with the product the plan buys for it, the store near the shopper, the price,
the country of origin and the product's labels, and whatever is not available nearby. Pure
code: no LLM call and no write.

The products are the draft's own: the resolve's pick for each line, replaced by the shopper's
pin, read and checked as /mealplan/schedule reads them (schedule.prepare). Nothing is resolved
here: a recipe that is not resolved lists no lines, and its status says so. A recipe is chosen
when it has a meal (place.expand: its wanted count and the meals already on the board).

Every product is bought once for every meal of every recipe that uses it, at its cheapest
offer in range (the price matrix the trips read; the catalog's pack_price), by the catalog's
whole-pack rule (catalog.whole_packs): the summed need over the pack size rounded up, and at
least one pack (one in all) when a line's need or the pack size is unknown. So the console's
whole-pack cost of a selection in the Meals stage is this est_total. price is one pack; a
line's cost_for_need is its recipe's share of what the product's packs cost: by need, or
split evenly when no line's need is known; a line of unknown need beside known ones has none
(the packs the others buy cover it). The lines add up to est_total. The trips buy on several
dates and can buy more packs; the Shop stage has their figures.

The plan's origin exclusion applies as everywhere else (origins.filter_pool): a product
evidenced as from an excluded country is not available and is never priced.

Each recipe names its picture as the catalog does (catalog.image).

An unavailable line says why, as unavailable_kind: origin_excluded (its product, or every
candidate in range, is evidenced as from an excluded country), stores_off (sold in range only
at stores the shopper turned off), out_of_range (sold, but by no store in range) or not_stocked
(no store sells it). For a line the resolve bought nothing for,
the kind and the reason are the planner's own (catalog.cached_picks: the same retrieval under
the same rules); a line left without a product for any other reason has no kind.
"""
from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction

from .. import alternatives, flow
from ..models import Product
from ..nlsearch.units import is_non_purchase, normalize_quantity, tokens
from ..origins import _match_preference, filter_pool
from . import catalog, diets, place, schedule
from .models import (
    SLOTS,
    Basket,
    BasketLine,
    BasketOrigin,
    BasketProduct,
    BasketRecipe,
    BasketStore,
    Meal,
    MealPlanDraft,
)
from .needs import USABLE, fmt_qty, meal_servings, recipe_servings

NO_PRODUCT = "no product was chosen for it"
STORES_OFF = "stores_off"
_STATUS = {"resolved": "evidenced", "conflicting": "conflicting"}


def _claim(alt) -> str:
    """The evidence in a shopper's words: "From Thailand" for a full origin claim, "Made in
    Canada, ingredients may be imported" for a processing one, "From X, as the source states"
    for any other; "Origin unclear: sources disagree" or "Origin unknown" without one."""
    if alt.status == "conflicting":
        return "Origin unclear: sources disagree"
    if alt.status != "resolved" or not alt.country:
        return "Origin unknown"
    if alt.claim == "full":
        return f"From {alt.country}"
    if alt.claim == "processing":
        return f"Made in {alt.country}, ingredients may be imported"
    return f"From {alt.country}, as the source states"


def _origin(o, demo: bool, preference: list[str]) -> BasketOrigin:
    alt = alternatives._alt_origin(o, demo)
    hit = _match_preference(o, preference) if o is not None and o.status == "resolved" else None
    return BasketOrigin(country=alt.country if alt.status == "resolved" else None,
                        claim=_claim(alt), status=_STATUS.get(alt.status, "unknown"),
                        preferred_rank=hit[0] + 1 if hit else None)


def _product(p: Product) -> BasketProduct:
    return BasketProduct(id=p.id, name=p.name, size=p.unit_size,
                         labels=sorted(diets.product_contains(p)),
                         says_organic="organic" in tokens(f"{p.name} {p.description}"),
                         demo_product=p.id in schedule.demo_product_ids())


def _left_out(resolved) -> dict[str, str]:
    """What the resolve said of the lines it bought nothing for, by ingredient."""
    if resolved is None:
        return {}
    return {d.ingredient: d.reason
            for d in [*resolved.not_stocked, *resolved.out_of_range, *resolved.skipped]}


@dataclass
class Purchases:
    """What a draft's basket buys, before any offer is read: the draft read and checked
    (schedule.prepare), every meal (place.expand), the chosen recipes in order and the
    resolved ones among them, each line's need for all its recipe's meals, the lines buying
    each product, and the products the origin exclusion holds back (id -> (country, field)),
    with the evidence read for them."""
    pre: schedule.Prepared
    meals: list[Meal]
    count: Counter
    keys: list[str]
    usable: list[str]
    need: dict[tuple[str, int], tuple[Fraction, str] | None]
    buying: dict[int, list[tuple[str, int]]]
    origins: dict
    demo: set[int]
    held: dict[int, tuple[str, str]]


def purchases(draft: MealPlanDraft) -> Purchases:
    """Raises MealPlanError (REST 422) as /mealplan/schedule does."""
    pre = schedule.prepare(draft)
    products, docs, chosen = pre.products, pre.docs, pre.chosen
    meals = place.expand(draft)
    count = Counter(m.recipe_key for m in meals)
    keys = sorted((k for k in draft.recipes if count[k]),
                  key=lambda k: (SLOTS.index(draft.recipes[k].slot), docs[k].title.casefold(), k))
    usable = [k for k in keys
              if draft.resolved.get(k) is not None and draft.resolved[k].status in USABLE]

    # Each line's need for all its recipe's meals: needs_for_meal's arithmetic, summed.
    need: dict[tuple[str, int], tuple[Fraction, str] | None] = {}
    for key in usable:
        doc = docs[key]
        yields = recipe_servings(draft, key, doc)
        people = sum(meal_servings(draft, m) for m in meals if m.recipe_key == key)
        for ln in doc.lines:
            canon = normalize_quantity(ln.quantity, ln.unit)
            need[key, ln.line_no] = (None if yields is None or canon is None
                                     else (Fraction(repr(canon[0])) * Fraction(people, yields),
                                           canon[1]))
    buying: dict[int, list[tuple[str, int]]] = {}
    for key in usable:
        for ln in docs[key].lines:
            pid = chosen[key].get(ln.line_no)
            if pid is not None and not is_non_purchase(ln.name):
                buying.setdefault(pid, []).append((key, ln.line_no))

    origins, demo = alternatives._evidence(sorted(buying), products)
    _kept, dropped = filter_pool([products[pid] for pid in sorted(buying)],
                                 exclude=list(draft.settings.exclude_origin), origins=origins)
    return Purchases(pre=pre, meals=meals, count=count, keys=keys, usable=usable, need=need,
                     buying=buying, origins=origins, demo=demo,
                     held={p.id: (country, fld) for p, country, fld in dropped})


def basket(draft: MealPlanDraft) -> Basket:
    """The basket for a draft. Raises MealPlanError (REST 422) as /mealplan/schedule does."""
    t0 = time.perf_counter()
    pu = purchases(draft)
    products, docs, chosen = pu.pre.products, pu.pre.docs, pu.pre.chosen
    count, keys, usable, need, buying = pu.count, pu.keys, pu.usable, pu.need, pu.buying
    origins, demo, held = pu.origins, pu.demo, pu.held
    s = draft.settings
    lat, lon, max_km = schedule.shopping_point(s)
    evidence = catalog.evidence_version(s)
    offer = flow._cheapest_offers(
        schedule._offers(sorted(buying.keys() - held.keys()), lat, lon, max_km, s.stores_off),
        "spec")
    stocked = set(offer)
    missing = sorted(buying.keys() - held.keys() - stocked)
    # Sold in range only at stores the shopper turned off: those stores, nearest first.
    only_off: dict[int, list[str]] = {}
    if missing and s.stores_off:
        for r in sorted(schedule._offers(missing, lat, lon, max_km),
                        key=lambda r: (r["dist_km2"], r["store_id"])):
            names = only_off.setdefault(r["product_id"], [])
            if r["store_name"] not in names:
                names.append(r["store_name"])
    nearest = flow._best_offers(schedule._offers(missing, lat, lon, None, s.stores_off),
                                key=lambda r: (r["dist_km2"], r["price"])) if missing else {}
    # The whole-pack rule, per product: what its packs cost and each line's share of them.
    cost: dict[int, float] = {}
    share: dict[tuple[str, int], float] = {}
    for pid in sorted(stocked):
        p, price = products[pid], offer[pid]["price"]
        known = {c: float(need[c][0]) for c in buying[pid]
                 if need[c] is not None and need[c][1] == p.unit_uom} if p.unit_qty else {}
        packs = catalog.whole_packs([known.get(c) for c in buying[pid]], p.unit_qty)
        cost[pid] = round(price * packs, 2) + 0.0
        total = sum(known.values())
        for c in buying[pid]:
            if not known:                                 # the one pack, split evenly
                share[c] = round(price * packs / len(buying[pid]), 2) + 0.0
            elif c in known:
                share[c] = round(price * packs * known[c] / total, 2) + 0.0 if total else 0.0

    recipes: list[BasketRecipe] = []
    unavailable = 0
    for key in keys:
        doc = docs[key]
        resolved = draft.resolved.get(key)
        lines: list[BasketLine] = []
        left = _left_out(resolved)
        planner = (catalog.cached_picks(draft.recipes[key].ref, doc, s, evidence)
                   if key in usable and any(chosen[key].get(ln.line_no) is None
                                            and not is_non_purchase(ln.name)
                                            for ln in doc.lines) else None)
        pins = draft.pins.get(key) or {}
        planned = {ln.line_no: ln.product_id for ln in resolved.lines} if resolved else {}
        for ln in (doc.lines if key in usable else []):
            if is_non_purchase(ln.name):
                continue
            pid = chosen[key].get(ln.line_no)
            n = need[key, ln.line_no]
            line = {"line_no": ln.line_no, "ingredient": ln.name,
                    "need": fmt_qty(*(n if n else (None, None))),
                    "pinned": str(ln.line_no) in pins
                    and pins[str(ln.line_no)] != planned.get(ln.line_no),
                    "shared_with": sorted({k for k, _no in buying.get(pid, []) if k != key}),
                    "product": None, "store": None, "price": None, "cost_for_need": None,
                    "origin": _origin(origins.get(pid), pid in demo, list(s.preference)),
                    "available": False, "unavailable_kind": None, "why_unavailable": None}
            if pid is None:
                kind, why = planner.left_out.get(ln.line_no, (None, None)) if planner else (
                    None, None)
                line.update(unavailable_kind=kind,
                            why_unavailable=why or left.get(ln.name, NO_PRODUCT))
            else:
                line["product"] = _product(products[pid])
                if pid in held:
                    country, fld = held[pid]
                    line.update(unavailable_kind=catalog.ORIGIN_EXCLUDED, why_unavailable=(
                        f"{products[pid].name} is evidenced as "
                        f"{alternatives._where(country, fld)}, which this plan excludes"))
                elif pid in only_off:
                    line.update(unavailable_kind=STORES_OFF, why_unavailable=(
                        "Sold only at stores you turned off: " + ", ".join(only_off[pid])))
                elif pid not in stocked:
                    near = nearest.get(pid)
                    line.update(unavailable_kind=(catalog.OUT_OF_RANGE if near
                                                  else catalog.NOT_STOCKED),
                                why_unavailable=_not_in_range(max_km, near))
                else:
                    o = offer[pid]
                    line.update(available=True, price=o["price"],
                                store=BasketStore(name=o["store"], distance_km=o["dist_km"]),
                                cost_for_need=share.get((key, ln.line_no)))
            unavailable += not line["available"]
            lines.append(BasketLine(**line))
        recipes.append(BasketRecipe(key=key, title=doc.title,
                                    image=catalog.image(draft.recipes[key].ref, doc),
                                    status=resolved.status if resolved else "unresolved",
                                    meals=count[key], lines=lines))
    return Basket(rev=draft.rev, recipes=recipes, unavailable_count=unavailable,
                  est_total=round(sum(cost.values()), 2) + 0.0, budget=draft.prefs.budget,
                  latency_ms=int((time.perf_counter() - t0) * 1000))


def _not_in_range(max_km: float | None, nearest: dict | None) -> str:
    """alternatives' words for a product no store in range sells."""
    within = f" within {max_km:g} km" if max_km is not None else ""
    if nearest is None:
        return f"no store{within} sells it"
    return (f"no store{within} sells it; the nearest offer is {nearest['store']}, "
            f"{nearest['dist_km']:.1f} km away, at ${nearest['price']:.2f}")

"""POST /mealplan/basket: the meal plan wizard's Ingredients stage. Every chosen recipe's
ingredients, each with the product the plan buys for it, the store near the shopper, the price,
the country of origin and the product's labels, and whatever is not available nearby. Pure
code: no LLM call and no write.

The products are the draft's own: the resolve's pick for each line, replaced by the shopper's
pin, read and checked as /mealplan/schedule reads them (schedule.prepare). Nothing is resolved
here: a recipe that is not resolved lists no lines, and its status says so. A recipe is chosen
when it has a meal (place.expand: its wanted count and the meals already on the board).

Every product is bought once for every meal of every recipe that uses it: its packs are the
summed need over the pack size (packs.pack_count), and its store is the trip optimizer's for
that one basket (tripopt over the in-range offers the trips read; a product whose packs are
unknown is listed at its cheapest offer among the stops, as a trip lists it). price is one pack
there. A line's cost_for_need is its recipe's share of what the product's packs cost, in
proportion to its need, so the lines add up to est_total. The trips buy on several dates and
can buy more packs; the Shop stage has their figures.

The plan's origin exclusion applies as everywhere else (origins.filter_pool): a product
evidenced as from an excluded country is not available and is never priced.

An unavailable line says why, as unavailable_kind: origin_excluded (its product, or every
candidate in range, is evidenced as from an excluded country), out_of_range (sold, but by no
store in range) or not_stocked (no store sells it). For a line the resolve bought nothing for,
the kind and the reason are the planner's own (catalog.cached_picks: the same retrieval under
the same rules); a line left without a product for any other reason has no kind.
"""
from __future__ import annotations

import math
import time
from collections import Counter
from fractions import Fraction

from .. import alternatives, flow, tripopt
from ..config import settings
from ..models import Product
from ..nlsearch.units import is_non_purchase, normalize_quantity, tokens
from ..origins import _match_preference, filter_pool
from ..packs import pack_count
from . import catalog, diets, place, schedule
from .models import (
    SLOTS,
    Basket,
    BasketLine,
    BasketOrigin,
    BasketProduct,
    BasketRecipe,
    BasketStore,
    MealPlanDraft,
)
from .needs import USABLE, fmt_qty, meal_servings, recipe_servings

NO_PRODUCT = "no product was chosen for it"
_STATUS = {"resolved": "evidenced", "conflicting": "conflicting"}


def _origin(o, demo: bool, preference: list[str]) -> BasketOrigin:
    alt = alternatives._alt_origin(o, demo)
    hit = _match_preference(o, preference) if o is not None and o.status == "resolved" else None
    return BasketOrigin(country=alt.country if alt.status == "resolved" else None,
                        claim=alt.label, status=_STATUS.get(alt.status, "unknown"),
                        preferred_rank=hit[0] + 1 if hit else None)


def _product(p: Product) -> BasketProduct:
    return BasketProduct(id=p.id, name=p.name, size=p.unit_size,
                         labels=sorted(diets.product_contains(p)),
                         says_organic="organic" in tokens(f"{p.name} {p.description}"),
                         demo_product=p.id in schedule.demo_product_ids())


def _stores(rows: list[dict], packs: dict[int, int | None], products: dict[int, Product],
            lat: float, lon: float) -> dict[int, str]:
    """Each stocked product's store: the trip optimizer's for the products whose packs are
    known, and for the rest their cheapest offer among its stops, else in range."""
    combined = {pid: n for pid, n in packs.items()
                if n is not None and any(r["product_id"] == pid for r in rows)}
    options = tripopt.optimize_trips(
        [r for r in rows if r["product_id"] in combined],
        [(pid, products[pid].name) for pid in sorted(combined)], home_lat=lat, home_lon=lon,
        cost_per_km=settings().travel_cost_per_km, packs=combined) if combined else []
    best = next((o for o in options if o.recommended), None)
    out = {it.product_id: it.store_name for it in best.items} if best else {}
    for pid in sorted({r["product_id"] for r in rows} - out.keys()):
        offers = sorted((r["price"], r["store_name"]) for r in rows if r["product_id"] == pid
                        and (best is None or r["store_name"] in best.stores))
        offers = offers or sorted((r["price"], r["store_name"]) for r in rows
                                  if r["product_id"] == pid)
        out[pid] = offers[0][1]
    return out


def _left_out(resolved) -> dict[str, str]:
    """What the resolve said of the lines it bought nothing for, by ingredient."""
    if resolved is None:
        return {}
    return {d.ingredient: d.reason
            for d in [*resolved.not_stocked, *resolved.out_of_range, *resolved.skipped]}


def basket(draft: MealPlanDraft) -> Basket:
    """The basket for a draft. Raises MealPlanError (REST 422) as /mealplan/schedule does."""
    t0 = time.perf_counter()
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

    s = draft.settings
    origins, demo = alternatives._evidence(sorted(buying), products)
    _kept, dropped = filter_pool([products[pid] for pid in sorted(buying)],
                                 exclude=list(s.exclude_origin), origins=origins)
    held = {p.id: (country, fld) for p, country, fld in dropped}
    lat, lon, max_km = schedule.shopping_point(s)
    rows = schedule._offers(sorted(buying.keys() - held.keys()), lat, lon, max_km)
    stocked = {r["product_id"] for r in rows}
    missing = sorted(buying.keys() - held.keys() - stocked)
    nearest = flow._best_offers(schedule._offers(missing, lat, lon, None),
                                key=lambda r: (r["dist_km2"], r["price"])) if missing else {}
    packs = {pid: pack_count(
        products[pid].unit_qty, products[pid].unit_uom,
        [None if need[c] is None else (float(need[c][0]), need[c][1]) for c in buying[pid]],
        min_lines=1) for pid in stocked}
    store_of = _stores(rows, packs, products, lat, lon)
    offer = {(r["store_name"], r["product_id"]): r for r in rows}
    cost = {pid: round(offer[store_of[pid], pid]["price"] * packs[pid], 2) + 0.0
            for pid in stocked if packs[pid] is not None}

    recipes: list[BasketRecipe] = []
    unavailable = 0
    for key in keys:
        doc = docs[key]
        resolved = draft.resolved.get(key)
        lines: list[BasketLine] = []
        left = _left_out(resolved)
        planner = (catalog.cached_picks(draft.recipes[key].ref, doc, s)
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
                elif pid not in stocked:
                    near = nearest.get(pid)
                    line.update(unavailable_kind=(catalog.OUT_OF_RANGE if near
                                                  else catalog.NOT_STOCKED),
                                why_unavailable=_not_in_range(max_km, near))
                else:
                    row = offer[store_of[pid], pid]
                    line.update(
                        available=True, price=row["price"],
                        store=BasketStore(name=row["store_name"], distance_km=round(
                            math.sqrt(max(row["dist_km2"], 0.0)), 1)),
                        cost_for_need=_share(cost.get(pid), n, [need[c] for c in buying[pid]]))
            unavailable += not line["available"]
            lines.append(BasketLine(**line))
        recipes.append(BasketRecipe(key=key, title=doc.title,
                                    status=resolved.status if resolved else "unresolved",
                                    meals=count[key], lines=lines))
    any_line = any(ln.available for r in recipes for ln in r.lines)
    total = None if any_line and not cost else round(sum(cost.values()), 2) + 0.0
    return Basket(rev=draft.rev, recipes=recipes, unavailable_count=unavailable, est_total=total,
                  budget=draft.prefs.budget, latency_ms=int((time.perf_counter() - t0) * 1000))


def _share(cost: float | None, mine, all_needs: list) -> float | None:
    """This line's part of the product's cost, by need. None when the cost is unknown."""
    if cost is None or mine is None:
        return None
    total = sum(n[0] for n in all_needs)
    return round(cost * float(mine[0] / total), 2) + 0.0 if total else None


def _not_in_range(max_km: float | None, nearest: dict | None) -> str:
    """alternatives' words for a product no store in range sells."""
    within = f" within {max_km:g} km" if max_km is not None else ""
    if nearest is None:
        return f"no store{within} sells it"
    return (f"no store{within} sells it; the nearest offer is {nearest['store']}, "
            f"{nearest['dist_km']:.1f} km away, at ${nearest['price']:.2f}")

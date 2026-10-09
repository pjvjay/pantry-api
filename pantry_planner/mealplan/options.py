"""Options for a trip line (POST /mealplan/alternatives): the products that could take the
place of one purchase on a meal-plan trip, in the planner's order, and what choosing each
does to the plan. Pure code: no LLM call and no write.

A trip line is one product bought on one date under one strategy, for one or more meals; it
can cover lines of several recipes (garlic for the pizza and for the fried rice). Choosing
another product pins every recipe line the purchase covers (pins[recipe_key][line_no]), so
the ranking is the chat cart's own, alternatives.rank_alternatives, over a basis made of
those lines (each rebuilt on the server as resolve planned it, pins.meal_basis): a candidate
must fill every one of them, the plan's origin exclusion holds products back, and a product
no store in range sells is not offered. What the ranking cannot know, the meal plan supplies
through alternatives.TripEffect:

- the need is what these meals need on this trip, not one batch of the recipe;
- each row's trip is the plan re-scheduled with that product pinned (the schedule's own
  _strategy, the products, recipes and placement read once): trip.total is the total of the
  strategy's trips, as the Shop band sums them; packs and cost_for_need are what the trip
  line then buys and charges; buys_at the store and pack price there. A re-schedule of the
  draft with those pins gives the same figures to the cent, because it is the same code.

A line whose product no store in range sells any more (no_longer_stocked) still has its
options: the product is left out of the rows and `stocked` says why.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import math

from .. import alternatives
from ..models import AltBuy, AltMove, AltTrip, Product
from ..packs import pack_count
from . import schedule
from .models import (
    CoveredLine,
    MealPlanDraft,
    MealPlanError,
    StrategyResult,
    Trip,
    TripLine,
    TripLineOptions,
)
from .needs import Need
from .pins import meal_basis
from .place import day_label

WHERE = "your plan's trips"
STRATEGY_NAMES = {"fresh": "Shop fresh", "fewest_trips": "Fewest trips"}


class NoTripLineError(LookupError):
    """The trip has no line for that product under that strategy: the plan changed since
    the console drew it (REST 409 no_trip_line)."""


def _run(draft: MealPlanDraft, pre: schedule.Prepared, needs: list[Need], name: str,
         offers: list[dict]) -> StrategyResult:
    ctx = schedule._Ctx(draft, pre.products, needs, pre.board, pre.cook, offers=offers)
    result, _warnings = schedule._strategy(ctx, name)
    return result


def _trip_on(result: StrategyResult, date: dt.date) -> Trip | None:
    return next((t for t in result.trips if t.date == date), None)


def _total_need(needs: list[Need]) -> tuple[float, str] | None:
    if not needs or any(n.qty is None for n in needs) or len({n.uom for n in needs}) != 1:
        return None
    return float(sum(n.qty for n in needs)), needs[0].uom


def _need_note(needs: list[Need], titles: dict[str, str]) -> str:
    """Why these meals' amount is unknown, in the meal plan's words ('' when it is known)."""
    if _total_need(needs) is not None:
        return ""
    waiting = sorted({titles[n.recipe_key] for n in needs if n.unknown == "needs_servings"})
    if waiting:
        return (f"How many {' and '.join(waiting)} serves is not known yet, so the amount "
                "for these meals is unknown")
    if any(n.unknown == "amount_unknown" for n in needs):
        return "The recipe gives no amount that can be compared with a pack"
    return "The recipe's lines give amounts in different units"


def trip_line_options(draft: MealPlanDraft, trip_date: dt.date, product_id: int,
                      strategy: str | None = None,
                      limit: int = alternatives.DEFAULT_LIMIT) -> TripLineOptions:
    """The options for the line buying `product_id` on the trip of `trip_date` under
    `strategy` (the draft's own by default). Raises MealPlanError for a draft no schedule
    can be computed for, NoTripLineError when that trip buys no such product, and
    alternatives.BasisError for a bad limit."""
    name = strategy or draft.prefs.strategy
    pre = schedule.prepare(draft)
    lat, lon, max_km = schedule._location(draft)
    # Every in-range offer, read once: each re-schedule below takes its rows from these.
    offers = schedule._offers(sorted(pre.products), lat, lon, max_km)
    base = _run(draft, pre, pre.needs, name, offers)
    trip = _trip_on(base, trip_date)
    here = [ln for ln in trip.lines if ln.product.id == product_id] if trip else []
    if not here:
        raise NoTripLineError(
            f"no trip on {day_label(trip_date)} buys product {product_id} under "
            f"{STRATEGY_NAMES.get(name, name)}; the plan changed since")
    meals = {m.meal_id for ln in here for m in ln.for_meals}
    bought = [n for n in pre.needs if n.product_id == product_id and n.meal_id in meals]
    covered = sorted({(n.recipe_key, n.line_no) for n in bought})
    cover_set = set(covered)
    titles = {key: doc.title for key, doc in pre.docs.items()}

    # One basis for the purchase: each covered line as its recipe planned it, numbered 1..k.
    bases = {key: meal_basis(draft, key, pre.docs[key], pre.chosen[key])
             for key in sorted({k for k, _ in covered})}
    lines = []
    for i, (key, line_no) in enumerate(covered, start=1):
        src = next(ln for ln in bases[key].lines if ln.line_no == line_no)
        lines.append(src.model_copy(update={"line_no": i}))
    first = bases[covered[0][0]]
    basis = first.model_copy(update={
        "lines": lines, "pins": [], "ingredient_count": len(lines),
        "recipe_slug": "mealplan", "recipe_name": " + ".join(titles[k] for k in bases),
        "not_stocked": [], "out_of_range": [], "skipped": [], "interpretation": []})
    needs = {i: _total_need([n for n in bought if (n.recipe_key, n.line_no) == c])
             for i, c in enumerate(covered, start=1)}

    def packs(product: Product, line_needs: list) -> int:
        return pack_count(product.unit_qty, product.unit_uom, line_needs, min_lines=1) or 1

    unit = {(r["store_name"], r["product_id"]): r for r in offers}
    shared: dict[int, list[str]] = {}

    def effect(pid: int) -> tuple[AltTrip | None, int | None, float | None, str]:
        trial = [dataclasses.replace(n, product_id=pid)
                 if (n.recipe_key, n.line_no) in cover_set else n for n in pre.needs]
        after = _run(draft, pre, trial, name, offers)
        found: list[tuple[Trip, TripLine]] = [
            (t, ln) for t in after.trips for ln in t.lines
            if ln.product.id == pid and meals & {m.meal_id for m in ln.for_meals}]
        same_day = [(t, ln) for t, ln in found if t.date == trip_date]
        found = same_day or found
        if not found:
            return None, None, None, "it would not be bought for these meals (you left it off)"
        t_after = found[0][0]
        got = [ln for _t, ln in found]
        n_packs = None if any(ln.packs is None for ln in got) else sum(ln.packs for ln in got)
        charged = (None if any(ln.price is None for ln in got)
                   else round(sum(ln.price for ln in got), 2) + 0.0)
        if after.total_cost is None or base.total_cost is None:
            return None, n_packs, charged, f"the total of {WHERE} is unknown"
        if charged is None and pid != product_id:
            # A line with no price drops out of the total: the total would fall by what
            # this product costs, which nobody knows.
            why = ("its packs and price are unknown: how many to buy for these meals can't "
                   "be worked out from its pack size" if n_packs is None
                   else "its price is unknown")
            return None, n_packs, None, why
        others = sorted({m.title for ln in got for m in ln.for_meals if m.meal_id not in meals}
                        - {titles[k] for k, _ in covered})
        if others:
            shared[pid] = others
        store = got[0].store
        row = unit.get((store, pid)) if store else None
        before = {ln.product.id: (ln.product.name, ln.store) for ln in trip.lines}
        now = {ln.product.id: ln.store for ln in t_after.lines} if t_after.date == trip_date \
            else {}
        moved = [AltMove(product_id=q, product=nm, from_store=a, to_store=now[q])
                 for q, (nm, a) in sorted(before.items())
                 if q not in (pid, product_id) and q in now and a and now[q] and a != now[q]]
        return AltTrip(
            total=after.total_cost,
            delta=round(after.total_cost - base.total_cost, 2) + 0.0,
            stores=list(t_after.stores),
            stops_delta=sum(len(t.stores) for t in after.trips)
            - sum(len(t.stores) for t in base.trips),
            buys_at=AltBuy(store=row["store_name"], price=row["price"],
                           distance_km=round(math.sqrt(max(row["dist_km2"], 0.0)), 1))
            if row is not None else None,
            moved_items=moved), n_packs, charged, ""

    ranking = alternatives.rank_alternatives(basis, 1, limit, effect=alternatives.TripEffect(
        needs=needs, on_trip=set(trip.stores), packs=packs, effect=effect, where=WHERE))

    # The meal plan's words where the cart's would mislead: the ingredient once per name, the
    # reason an amount is unknown, and a purchase the product would share with other meals.
    names = list(dict.fromkeys(ln.name for ln in lines))
    note = _need_note(bought, titles)
    for item in ranking.items:
        with_whom = " and ".join(shared.get(item.product_id, []))
        for r in item.reasons:
            if with_whom and r.code == "trip":
                r.text += f"; one purchase with what {with_whom} needs"
    ranking = ranking.model_copy(update={"ingredient": " + ".join(names),
                                         "need_note": note if note else ranking.need_note})
    pins = draft.pins
    resolved = {key: {ln.line_no: ln.product_id for ln in draft.resolved[key].lines}
                for key in bases if key in draft.resolved}
    out_lines = []
    for key, line_no in covered:
        doc_line = next(ln for ln in pre.docs[key].lines if ln.line_no == line_no)
        out_lines.append(CoveredLine(
            recipe_key=key, title=titles[key], line_no=line_no, ingredient=doc_line.name,
            planner_product_id=resolved.get(key, {}).get(line_no),
            pinned_product_id=(pins.get(key) or {}).get(str(line_no))))
    return TripLineOptions(
        rev=draft.rev, strategy=name, trip_date=trip_date, product_id=product_id,
        product=here[0].product.name,
        stocked=any(r["product_id"] == product_id for r in offers),
        pinned=any(c.pinned_product_id is not None
                   and c.pinned_product_id != c.planner_product_id for c in out_lines),
        lines=out_lines, plan_total=base.total_cost, ranking=ranking)


def options_error(e: Exception) -> tuple[int, dict]:
    """(status, detail) for the REST route."""
    if isinstance(e, MealPlanError):
        return 422, {"error": e.code, "detail": e.detail, **e.extra}
    if isinstance(e, NoTripLineError):
        return 409, {"error": "no_trip_line", "detail": str(e)}
    if isinstance(e, alternatives.StaleBasisError):
        return 409, {"error": "stale_basis", "detail": str(e)}
    return 422, {"error": "invalid_basis", "detail": str(e)}

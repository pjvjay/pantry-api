"""Where meals sit: counts become meals, unplaced meals are spread, and "Suggest cook days"
proposes a freshness-aware layout.

The board has one cell per date and slot: one breakfast, lunch and dinner a day and two
snacks. A meal with a date, or pinned, never moves. Meals with no date are spread evenly:
the i-th of a recipe's n meals aims at day floor((i + 0.5) * D / n), and on a collision the
next free day is tried in the order d+1, d-1, d+2, d-2 ... Recipes with more meals go first
(ties by key), so the result depends only on the draft.
"""
from __future__ import annotations

import datetime as dt
import math
import re
from dataclasses import dataclass

from .models import Meal, MealPlanDraft, MealPlanError

CAPACITY = {"breakfast": 1, "lunch": 1, "dinner": 1, "snack": 2}


def natural(meal_id: str) -> tuple:
    """'m10' after 'm9': digits compare as numbers."""
    return tuple(int(p) if p.isdigit() else p for p in re.split(r"(\d+)", meal_id))


def expand(draft: MealPlanDraft) -> list[Meal]:
    """The draft's meals plus one undated meal per wanted occasion not yet on the board:
    recipe K wanted 3 with one meal already gives two more, ids 'K#1', 'K#2' (the first
    free numbers). Meals beyond `wanted` are kept: the shopper placed them."""
    meals = list(draft.meals)
    ids = {m.id for m in meals}
    for key in sorted(draft.recipes):
        have = sum(1 for m in meals if m.recipe_key == key)
        n = 1
        for _ in range(draft.recipes[key].wanted - have):
            while f"{key}#{n}" in ids:
                n += 1
            ids.add(f"{key}#{n}")
            meals.append(Meal(id=f"{key}#{n}", recipe_key=key))
    if len(meals) > 56:
        raise MealPlanError("slot_capacity", f"{len(meals)} meals is more than a plan holds (56)")
    return meals


def slot_of(draft: MealPlanDraft, meal: Meal) -> str:
    return meal.slot or draft.recipes[meal.recipe_key].slot


@dataclass
class Board:
    """Cell occupancy: (day index, slot) -> meals in it."""
    days: int
    used: dict[tuple[int, str], int]

    def free(self, day: int, slot: str) -> bool:
        return 0 <= day < self.days and self.used.get((day, slot), 0) < CAPACITY[slot]

    def take(self, day: int, slot: str) -> None:
        self.used[(day, slot)] = self.used.get((day, slot), 0) + 1


def fixed_board(draft: MealPlanDraft, meals: list[Meal], movable: set[str]) -> Board:
    """The board with every meal that is not being placed. Dated meals over a cell's
    capacity are the shopper's own conflict: 422 slot_capacity, naming the cell."""
    board = Board(days=draft.days, used={})
    for m in meals:
        if m.id in movable or m.date is None:
            continue
        day, slot = draft.index(m.date), slot_of(draft, m)
        if not board.free(day, slot):
            raise MealPlanError("slot_capacity",
                                f"{m.date.isoformat()} {slot} holds {CAPACITY[slot]} "
                                f"meal(s); {m.id} is one too many", meal_id=m.id)
        board.take(day, slot)
    return board


def probe(target: int, lo: int, hi: int):
    """target, target+1, target-1, target+2, ... within [lo, hi]."""
    yield target
    for k in range(1, hi - lo + 2):
        for d in (target + k, target - k):
            if lo <= d <= hi:
                yield d


def target_day(i: int, n: int, first: int, days: int) -> int:
    return first + math.floor((i + 0.5) * (days - first) / n)


def spread(draft: MealPlanDraft, meals: list[Meal], movable: set[str], board: Board,
           *, first: int = 0) -> dict[str, int | None]:
    """Days for the meals in `movable`, spread evenly over [first, days-1]; None when no
    cell of their slot is free (unplaced). `board` is updated."""
    by_recipe: dict[str, list[Meal]] = {}
    for m in meals:
        if m.id in movable:
            by_recipe.setdefault(m.recipe_key, []).append(m)
    out: dict[str, int | None] = {}
    for key in sorted(by_recipe, key=lambda k: (-len(by_recipe[k]), k)):
        group = sorted(by_recipe[key], key=lambda m: natural(m.id))
        for i, m in enumerate(group):
            slot = slot_of(draft, m)
            day = None
            if first < draft.days:
                t = target_day(i, len(group), first, draft.days)
                day = next((d for d in probe(t, first, draft.days - 1) if board.free(d, slot)),
                           None)
            if day is not None:
                board.take(day, slot)
            out[m.id] = day
    return out


def place(draft: MealPlanDraft) -> tuple[list[Meal], dict[str, int | None]]:
    """(every meal, meal id -> day index or None). Dated meals stay where they are; undated,
    unpinned meals are spread; an undated pinned meal stays in the tray."""
    meals = expand(draft)
    for m in meals:
        if m.recipe_key not in draft.recipes:
            raise MealPlanError("unknown_recipe_key",
                                f"meal {m.id} names recipe {m.recipe_key!r}, which the plan "
                                "does not have", meal_id=m.id)
        if m.date is not None and not 0 <= draft.index(m.date) < draft.days:
            raise MealPlanError("invalid_dates", f"meal {m.id} is on {m.date.isoformat()}, "
                                "outside the plan's dates", meal_id=m.id)
    movable = {m.id for m in meals if m.date is None and not m.pinned}
    board = fixed_board(draft, meals, movable)
    days = {m.id: draft.index(m.date) for m in meals if m.date is not None}
    days.update(spread(draft, meals, movable, board))
    for m in meals:
        days.setdefault(m.id, None)
    return meals, days


SHORT = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


def day_label(d: dt.date) -> str:
    """'Sun 18 Oct'."""
    return f"{SHORT[d.weekday()]} {d.day} {d:%b}"


# ─── Suggest cook days ───────────────────────────────────────

@dataclass(frozen=True)
class Horizon:
    """How many days after a shop a meal can be cooked, and what limits it: the shortest
    cited fridge time among its products (kind 'cited', with the rule), else the shopper's
    buy-ahead setting for a product with no cited time ('your_setting'), else no limit."""
    days: int | None
    kind: str | None = None
    product: str | None = None
    rule: dict | None = None


def horizon(draft: MealPlanDraft, products: dict, chosen: dict[int, int]) -> Horizon:
    from . import shelf

    best = Horizon(days=None)
    for pid in sorted(set(chosen.values()), key=lambda p: products[p].name):
        ps = shelf.for_product(pid)
        if ps.bought_frozen or (not ps.mapped and ps.storage_class in {"shelf_stable", "frozen"}):
            continue
        if ps.mapped and ps.fridge_days is not None:
            h = Horizon(ps.fridge_days, "cited", products[pid].name,
                        min(ps.fridge_rules(), key=lambda r: (r["days_min"] is None,
                                                              r["days_min"] or 0)))
        else:
            h = Horizon(draft.prefs.buy_ahead_days, "your_setting", products[pid].name)
        rank = {"cited": 0, "your_setting": 1}
        if best.days is None or (h.days, rank[h.kind]) < (best.days, rank[best.kind]):
            best = h
    return best


def _why(title: str, d: dt.date, h: Horizon, trip: dt.date | None, buy_ahead: int) -> str:
    from . import shelf

    shop = f"; bought on the {day_label(trip)} shop" if trip else ""
    head = f"{title} moved to {day_label(d)}"
    if h.kind == "cited":
        return (f"{head}: {h.product} keeps {h.rule['verbatim']} in the fridge "
                f"({shelf.short_source(h.rule['source'])}, {h.rule['id']}){shop}.")
    if h.kind == "your_setting":
        return (f"{head}: {h.product} has no cited storage time, and your setting buys it at "
                f"most {buy_ahead} day(s) ahead{shop}.")
    return (f"{head}: none of its products has a cited fridge time or falls under your "
            f"buy-ahead setting, so it is spread across the period{shop}.")


def freshness_layout(draft: MealPlanDraft, before) -> dict:
    """POST /mealplan/suggest-cook-days: a proposal that places every unpinned meal by
    freshness, never applied by the server.

    Candidate trip days are the approved and fixed dates, then every day the shopper shops
    (shop_weekdays, less dismissed dates). Each unpinned meal gets a horizon (Horizon above).
    Meals with a horizon go first, shortest first (earliest deadline first), each to the free
    cell of its slot nearest its evenly spread target among the days [trip, trip + horizon]
    after some candidate trip (a breakfast from the day after). The rest are spread evenly
    from the first candidate trip on, as before. Pinned meals never move.

    `before` is the draft's current MealSchedule. Returns {rev, ops, meals,
    warnings_before, warnings_after}: each op moves one meal and carries a reason built by
    code from the limiting product's cited row or the shopper's setting."""
    from .. import db
    from . import schedule as sched
    from . import warnings as warn
    from .needs import USABLE, line_products
    from .resolve import recipe_doc

    products = {p.id: p for p in db.load_all_products()}
    meals = expand(draft)
    movable = {m.id for m in meals if not m.pinned}
    board = fixed_board(draft, meals, movable)
    dismissed = {draft.index(d) for d in draft.dismissed_dates}
    fixed = {draft.index(d) for d in draft.fixed_dates} | {draft.index(t.date)
                                                           for t in draft.trips}
    trip_days = sorted(fixed | {i for i in range(draft.days)
                                if draft.day(i).weekday() in draft.prefs.shop_weekdays
                                and i not in dismissed})
    first = trip_days[0] if trip_days else 0

    horizons: dict[str, Horizon] = {}
    for key, dr in sorted(draft.recipes.items()):
        resolved = draft.resolved.get(key)
        chosen = {}
        if resolved is not None and resolved.status in USABLE:
            chosen = line_products(draft, key, recipe_doc(dr.ref, dr.servings), products)
        horizons[key] = horizon(draft, products, chosen)

    groups: dict[str, list[Meal]] = {}
    for m in meals:
        if m.id in movable:
            groups.setdefault(m.recipe_key, []).append(m)
    targets: dict[str, int] = {}
    for _key, group in groups.items():
        for i, m in enumerate(sorted(group, key=lambda m: natural(m.id))):
            targets[m.id] = target_day(i, len(group), first, draft.days)

    placed: dict[str, int | None] = {}
    shop_for: dict[str, int] = {}
    timed = sorted((m for m in meals if m.id in movable
                    and horizons[m.recipe_key].days is not None),
                   key=lambda m: (horizons[m.recipe_key].days, -len(groups[m.recipe_key]),
                                  m.recipe_key, natural(m.id)))
    rest: set[str] = {m.id for m in meals if m.id in movable} - {m.id for m in timed}
    for m in timed:
        slot, h = slot_of(draft, m), horizons[m.recipe_key].days
        start = 1 if slot == "breakfast" else 0
        days = sorted({d for t in trip_days for d in range(t + start, t + h + 1)
                       if d < draft.days and board.free(d, slot)})
        if not days:
            rest.add(m.id)
            continue
        d = min(days, key=lambda d: (abs(d - targets[m.id]), d))
        board.take(d, slot)
        placed[m.id] = d
        shop_for[m.id] = max(t for t in trip_days if t + start <= d)
    placed.update(spread(draft, meals, rest, board, first=first))

    ops, after_meals = [], []
    for m in sorted(meals, key=lambda m: natural(m.id)):
        if m.id not in movable:
            after_meals.append(m)
            continue
        day = placed.get(m.id)
        new_date = None if day is None else draft.day(day)
        slot = slot_of(draft, m)
        after_meals.append(m.model_copy(update={"date": new_date, "slot": slot}))
        if new_date == m.date and (m.slot or slot) == slot:
            continue
        title = sched_title(draft, m.recipe_key)
        if new_date is None:
            reason = f"{title} has no free {slot} slot left after the layout."
        else:
            trip = shop_for.get(m.id)
            if trip is None:
                trip = max((t for t in trip_days if t <= day), default=None)
            reason = _why(title, new_date, horizons[m.recipe_key]
                          if m.id in shop_for else Horizon(days=None),
                          None if trip is None else draft.day(trip),
                          draft.prefs.buy_ahead_days)
        ops.append({"op": "move_meal", "meal_id": m.id,
                    "from": {"date": None if m.date is None else m.date.isoformat(),
                             "slot": m.slot or slot},
                    "to": {"date": None if new_date is None else new_date.isoformat(),
                           "slot": slot},
                    "reason": reason})
    after = sched.compute(draft.model_copy(update={"meals": after_meals}))
    strategy = draft.prefs.strategy
    return {"rev": draft.rev, "ops": ops,
            "meals": [m.model_dump(mode="json") for m in after_meals],
            "warnings_before": warn.counts(before.warnings, strategy),
            "warnings_after": warn.counts(after.warnings, strategy)}


def sched_title(draft: MealPlanDraft, key: str) -> str:
    from .resolve import recipe_doc

    return recipe_doc(draft.recipes[key].ref, draft.recipes[key].servings).title

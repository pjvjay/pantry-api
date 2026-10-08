"""Where meals sit: counts become meals, and unplaced meals are spread.

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


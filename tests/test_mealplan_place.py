"""Placing meals on the board: counts become meals at household servings, spread evenly and
deterministically; meals with a date or a pin never move; overflow is a must-fix.

The expected days are the design's worked example (start Fri 2026-10-09, 14 days):
floor((i + 0.5) * 14 / n), recipes with more meals first, collisions probed d+1, d-1, ...
"""
from __future__ import annotations

import math

import pytest

from tests.mealplan_fixtures import (
    EXAMPLE,
    client,
    day,
    done_db,
    index,
    schedule,
    starter_draft,
    use_db,
)


@pytest.fixture(scope="module", autouse=True)
def seeded_db(tmp_path_factory):
    use_db(tmp_path_factory, "place")
    yield
    done_db()


@pytest.fixture(scope="module")
def example():
    return starter_draft()


def _days(sched: dict, key: str) -> list[int]:
    return sorted(index(m["date"]) for m in sched["meals"] if m["recipe_key"] == key)


def test_the_example_gives_15_meals_one_per_cell_at_household_servings(example):
    sched = schedule(example)
    assert len(sched["meals"]) == sum(EXAMPLE.values()) == 15
    assert sched["unplaced"] == []
    assert {m["servings"] for m in sched["meals"]} == {2}
    # The arithmetic of the spread, recomputed here: 3 meals over 14 days aim at 2, 7, 11.
    assert [math.floor((i + 0.5) * 14 / 3) for i in range(3)] == [2, 7, 11]
    assert _days(sched, "starter:chicken_biryani") == [2, 7, 11]
    assert _days(sched, "starter:pepperoni_pizza") == [3, 8, 12]      # one day on: taken
    assert _days(sched, "starter:chicken_fried_rice") == [4, 10]
    assert _days(sched, "starter:mango_milkshake") == [1, 3, 5, 7, 9, 11, 13]
    cells = [(m["date"], m["slot"]) for m in sched["meals"] if m["slot"] == "dinner"]
    assert len(cells) == len(set(cells))
    assert {m["placed_by"] for m in sched["meals"]} == {"spread"}


def test_the_same_draft_places_the_same_way(example):
    a = client().post("/mealplan/schedule", json=example)
    b = client().post("/mealplan/schedule", json=example)
    assert a.content == b.content


def test_dated_and_pinned_meals_never_move(example):
    meals = [{"id": "starter:chicken_biryani#1", "recipe_key": "starter:chicken_biryani",
              "date": day(2).isoformat(), "slot": "dinner"},
             {"id": "starter:pepperoni_pizza#1", "recipe_key": "starter:pepperoni_pizza",
              "date": day(0).isoformat(), "pinned": True},
             {"id": "starter:chicken_fried_rice#1", "recipe_key": "starter:chicken_fried_rice",
              "pinned": True}]
    sched = schedule({**example, "meals": meals})
    by_id = {m["id"]: m for m in sched["meals"]}
    assert index(by_id["starter:chicken_biryani#1"]["date"]) == 2
    assert by_id["starter:chicken_biryani#1"]["placed_by"] == "you"
    assert index(by_id["starter:pepperoni_pizza#1"]["date"]) == 0
    # Pinned with no date: kept in the tray, not placed, and not a warning.
    assert by_id["starter:chicken_fried_rice#1"]["date"] is None
    assert "unplaced" not in [w["code"] for w in sched["warnings"]]
    # The meals still count: 15 in all, the missing ones added.
    assert len(sched["meals"]) == 15


def test_overflow_is_unplaced_with_a_must_fix(example):
    sched = schedule({**example, "days": 3})
    unplaced = set(sched["unplaced"])
    # A 3-day board holds 3 dinners (8 wanted) and 6 snacks (7 wanted).
    assert len(unplaced) == (8 - 3) + (7 - 2 * 3)
    must = [w for w in sched["warnings"] if w["code"] == "unplaced"]
    assert {w["level"] for w in must} == {"must_fix"}
    assert {m for w in must for m in w["meal_ids"]} == unplaced


def test_two_snacks_a_day_and_one_of_each_meal():
    draft = starter_draft({"mango_milkshake": 7})
    sched = schedule({**draft, "days": 3})
    assert len(sched["meals"]) - len(sched["unplaced"]) == 6


def test_a_draft_the_board_cannot_hold_is_a_422(example):
    two = [{"id": f"x{i}", "recipe_key": "starter:chicken_biryani", "date": day(1).isoformat()}
           for i in range(2)]
    r = client().post("/mealplan/schedule", json={**example, "meals": two})
    assert (r.status_code, r.json()["detail"]["error"]) == (422, "slot_capacity")
    r = client().post("/mealplan/schedule", json={
        **example, "meals": [{"id": "x", "recipe_key": "starter:nope"}]})
    assert (r.status_code, r.json()["detail"]["error"]) == (422, "unknown_recipe_key")
    r = client().post("/mealplan/schedule", json={
        **example, "meals": [{"id": "x", "recipe_key": "starter:chicken_biryani",
                              "date": day(20).isoformat()}]})
    assert (r.status_code, r.json()["detail"]["error"]) == (422, "invalid_dates")
    r = client().post("/mealplan/schedule", json={**example, "v": 2})
    assert (r.status_code, r.json()["detail"]["error"]) == (422, "version")


def test_a_meals_own_servings_override_the_household(example):
    meals = [{"id": "starter:chicken_biryani#1", "recipe_key": "starter:chicken_biryani",
              "servings": 4}]
    sched = schedule({**example, "meals": meals})
    by_id = {m["id"]: m for m in sched["meals"]}
    assert by_id["starter:chicken_biryani#1"]["servings"] == 4
    assert by_id["starter:chicken_biryani#2"]["servings"] == 2

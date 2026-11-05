"""Trip dates: minimum interval stabbing against brute force, the worked example, freezing in
fewest_trips, Saturday-only shopping, the shopper's buy-ahead setting for unknown storage
times, and breakfast bought the day before.

Expected windows come from the cited rows in shelf_life.json (read here) and day arithmetic.
"""
from __future__ import annotations

import itertools
import random

import pytest

from tests.mealplan_fixtures import (
    SHELF,
    START,
    day,
    doc_draft,
    doc_recipe,
    done_db,
    index,
    lines_of,
    schedule,
    starter_draft,
    strategy,
    trip_days,
    use_db,
)

RULES = {r["id"]: r for r in [*SHELF["rules"], *SHELF["thaw_rules"]]}
POULTRY_DAYS = RULES["fs-poultry-pieces-fridge"]["days_min"]


@pytest.fixture(scope="module", autouse=True)
def seeded_db(tmp_path_factory):
    use_db(tmp_path_factory, "trips")
    yield
    done_db()


@pytest.fixture(scope="module")
def example():
    return starter_draft()


# ─── The stabbing search ─────────────────────────────────────

def _brute(windows: list[tuple[int, int]], fixed: set[int], allowed: list[int]) -> int:
    open_ = [w for w in windows if not any(w[0] <= t <= w[1] for t in fixed)]
    if not open_:
        return 0
    pool = [a for a in allowed if a not in fixed]
    for k in range(1, len(pool) + 1):
        for combo in itertools.combinations(pool, k):
            if all(any(lo <= t <= hi for t in combo) for lo, hi in open_):
                return k
    raise AssertionError("no cover exists")


def test_stabbing_takes_the_fewest_trips_on_200_seeded_instances():
    from pantry_planner.mealplan.trips import choose_dates

    rng = random.Random(20261009)
    for _ in range(200):
        days = rng.randint(3, 10)
        allowed = sorted(rng.sample(range(days), rng.randint(1, days)))
        fixed = set(rng.sample(allowed, rng.randint(0, min(2, len(allowed)))))
        windows = []
        for _w in range(rng.randint(1, 8)):
            lo = rng.randint(0, days - 1)
            hi = rng.randint(lo, days - 1)
            inside = [a for a in allowed if lo <= a <= hi]
            if inside:
                windows.append((inside[0], inside[-1]))
        trips, _made = choose_dates(windows, fixed)
        assert fixed <= set(trips) <= set(allowed)
        assert all(any(lo <= t <= hi for t in trips) for lo, hi in windows)
        assert len(set(trips) - fixed) == _brute(windows, fixed, allowed)


def test_the_worked_example_shops_on_days_1_4_7_10(example):
    sched = schedule(example)
    # Chicken keeps 1 to 2 days (planned 1): biryani on 2, 7, 11 and fried rice on 4, 10
    # need trips within a day before each; everything else fits around them.
    assert POULTRY_DAYS == 1
    assert trip_days(sched, "fresh") == [1, 4, 7, 10]
    assert sched["recommended_strategy"] == "fresh"
    fresh = strategy(sched, "fresh")
    assert not [a for a in fresh["actions"] if a["kind"] in {"freeze", "thaw"}]
    for t in fresh["trips"]:
        for ln in t["lines"]:
            if ln["shelf_life"]["status"] == "cited" and ln["storage"] == "fridge":
                for m in ln["for_meals"]:
                    assert 0 <= index(m["date"]) - index(t["date"]) <= \
                        ln["shelf_life"]["days_planned"]


def test_fewest_trips_freezes_on_arrival_with_cited_thaw_reminders(example):
    sched = schedule(example)
    fewest = strategy(sched, "fewest_trips")
    assert len(fewest["trips"]) < len(strategy(sched, "fresh")["trips"])
    freezes = [a for a in fewest["actions"] if a["kind"] == "freeze"]
    thaws = [a for a in fewest["actions"] if a["kind"] == "thaw"]
    assert freezes and thaws
    for a in thaws:
        assert a["rule_ids"] and set(a["rule_ids"]) <= {"fsis-thaw-fridge-small",
                                                        "fsis-thaw-fridge-large"}
        assert a["basis"] == "cited" and RULES[a["rule_ids"][0]]["verbatim"] in a["text"]
    frozen = [ln for t in fewest["trips"] for ln in t["lines"] if ln["freeze_on_arrival"]]
    assert frozen and all(ln["storage"] == "freezer" for ln in frozen)
    assert all("fs-poultry-pieces-freezer" in ln["shelf_life"]["rule_ids"] for ln in frozen)


def test_a_thawed_pack_is_bought_for_its_own_meal():
    # Two meals of 200 g boneless thighs (450 g packs), shopping only on day 0: fresh buys
    # them stale or not at all; fewest_trips freezes one pack per meal, never sharing a
    # thawed pack between them.
    entry = doc_recipe("my:thighs", "Thigh Skewers", 2, [("Chicken Thighs", 200, "g", 46)])
    meals = [{"id": "a", "recipe_key": "my:thighs", "date": day(5).isoformat()},
             {"id": "b", "recipe_key": "my:thighs", "date": day(6).isoformat()}]
    sched = schedule(doc_draft({"my:thighs": entry}, days=7, meals=meals,
                               prefs={"shop_weekdays": [START.weekday()]}))
    ((_trip, line),) = lines_of(sched, "fewest_trips", 46)
    assert (line["storage"], line["packs"], line["need_qty"]) == ("freezer", 2, 400.0)
    thaws = [a for a in strategy(sched, "fewest_trips")["actions"] if a["kind"] == "thaw"]
    assert sorted(a["meal_id"] for a in thaws) == ["a", "b"]


def test_saturday_only_shopping_without_a_layout_gives_must_fix_remedies(example):
    sched = schedule({**example, "prefs": {"shop_weekdays": [5]}})
    assert {day(i).weekday() for i in trip_days(sched, "fresh")} == {5}
    exceeded = [w for w in sched["warnings"] if w["code"] == "fridge_window_exceeded"]
    assert exceeded and {w["strategy"] for w in exceeded} == {"fresh"}
    assert all(w["level"] == "must_fix" for w in exceeded)
    ops = {r["op"] for w in exceeded for r in w["remedies"]}
    assert {"set_storage", "add_trip", "move_meal"} <= ops
    assert all(len(w["remedies"]) <= 3 for w in sched["warnings"])
    # fewest_trips freezes the chicken instead, so it is the one recommended.
    assert "fridge_window_exceeded" not in [w["code"] for w in sched["warnings"]
                                            if w["strategy"] == "fewest_trips"]
    assert sched["recommended_strategy"] == "fewest_trips"


def test_unknown_storage_uses_the_shoppers_buy_ahead_setting(example):
    for ahead in (7, 2):
        sched = schedule({**example, "prefs": {"buy_ahead_days": ahead}})
        lines = lines_of(sched, "fresh", 24)            # Mozzarella: no cited row
        assert lines
        for trip, ln in lines:
            assert ln["shelf_life"]["status"] == "your_setting"
            assert ln["shelf_life"]["days_planned"] == ahead and not ln["shelf_life"]["verbatim"]
            assert "Your setting" in ln["shelf_life"]["note"]
            for m in ln["for_meals"]:
                assert 0 <= index(m["date"]) - index(trip["date"]) <= ahead
    # Shelf-stable with no cited time is not limited: one purchase of salt for the period.
    assert len(lines_of(schedule(example), "fresh", 122)) == 1


def test_breakfast_is_bought_the_day_before():
    entry = doc_recipe("my:eggs", "Chicken Hash", 2, [("Chicken Breast", 300, "g", 10)])
    for slot, expected in (("breakfast", 4), ("dinner", 5)):
        meals = [{"id": "m", "recipe_key": "my:eggs", "date": day(5).isoformat(),
                  "slot": slot}]
        sched = schedule(doc_draft({"my:eggs": entry}, meals=meals))
        assert trip_days(sched, "fresh") == [expected], slot


def test_fixed_and_dismissed_dates(example):
    sched = schedule({**example, "fixed_dates": [day(0).isoformat()],
                      "dismissed_dates": [day(4).isoformat()]})
    days = trip_days(sched, "fresh")
    assert 0 in days and 4 not in days
    trip0 = strategy(sched, "fresh")["trips"][0]
    assert trip0["reason"] == "Your fixed shopping date."


def test_each_trip_names_a_cited_reason_first(example):
    sched = schedule(example)
    for t in strategy(sched, "fresh")["trips"]:
        assert "fs-poultry-pieces-fridge" in t["reason"] or "your setting" in t["reason"]

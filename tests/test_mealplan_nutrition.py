"""Nutrition on POST /mealplan/schedule: each day, the period summary and the per-recipe
receipts. No LLM (starters resolve in demo mode).

Expected per-meal values are read from the schedule's own recipe_nutrition (checked against
arithmetic in test_nutrition); these tests check how days and the period add them up, what
they say is complete, and the demo-amounts label.
"""
from __future__ import annotations

import pytest

from tests.mealplan_fixtures import (
    START,
    client,
    day,
    doc_draft,
    doc_recipe,
    done_db,
    schedule,
    starter_draft,
    use_db,
)

NUTRIENTS = ("energy_kcal", "protein_g", "fat_g", "satfat_g", "carbohydrate_g", "fibre_g",
             "sugars_g", "sodium_mg")
SLOTS = ["breakfast", "lunch", "dinner", "snack"]
OGL = "Contains information licensed under the Open Government Licence – Canada."


@pytest.fixture(scope="module", autouse=True)
def seeded_db(tmp_path_factory):
    use_db(tmp_path_factory, "mp_nutrition")
    yield
    done_db()


def _meals_by_day(sched: dict) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for m in sched["meals"]:
        if m["date"]:
            out.setdefault(m["date"], []).append(m)
    return out


def test_each_day_adds_one_serving_of_each_meal():
    s = schedule(starter_draft())
    assert s["coverage"]["nutrition"] == "computed"
    per = s["recipe_nutrition"]
    assert set(per) == {"starter:pepperoni_pizza", "starter:chicken_fried_rice",
                        "starter:chicken_biryani", "starter:mango_milkshake"}
    assert all(n["basis"] == "per_serving" for n in per.values())
    by_day = _meals_by_day(s)
    for d in s["days"]:
        n = d["nutrition"]
        meals = by_day.get(d["date"], [])
        assert {m["meal_id"] for m in n["meals"]} == {m["id"] for m in meals}
        order = [SLOTS.index(m["slot"]) for m in n["meals"]]
        assert order == sorted(order)
        for key in NUTRIENTS:
            known = [per[m["recipe_key"]]["totals"][key]["amount"] for m in meals]
            known = [a for a in known if a is not None]
            if known:
                assert n["totals"][key]["amount"] == pytest.approx(sum(known), abs=0.051)
            else:
                assert n["totals"][key]["amount"] is None
        # all four slots are on and no starter is breakfast or lunch: never complete
        assert not n["all_meals_planned"] and not n["complete"]
        assert n["meals_counted"] == ["breakfast", "lunch", "dinner", "snack"]


def test_starter_days_carry_the_demo_amounts_label():
    s = schedule(starter_draft())
    with_meals = [d for d in s["days"] if d["meal_ids"]]
    assert with_meals and all(d["nutrition"]["demo_amounts"] for d in with_meals)
    assert all(d["nutrition"]["amounts_basis"] == ["demo_house_amounts"] for d in with_meals)
    assert s["period_nutrition"]["demo_amounts"]
    empty = [d for d in s["days"] if not d["meal_ids"]]
    for d in empty:
        assert not d["nutrition"]["demo_amounts"]
        assert all(t["amount"] is None and t["status"] == "unknown"
                   for t in d["nutrition"]["totals"].values())


def test_no_complete_day_means_no_average():
    p = schedule(starter_draft())["period_nutrition"]
    assert p["days_total"] == 14 and p["days_complete"] == 0
    assert p["per_day_average_over_complete_days"] is None
    assert len(p["incomplete_days"]) == 14
    assert p["lower_bound_total"]["energy_kcal"]["amount"] > 0
    assert not p["lower_bound_total"]["energy_kcal"]["complete"]


def test_days_complete_counts_only_the_slots_that_are_on():
    """The milkshake is complete per serving. With only the snack slot on, every day with a
    milkshake is complete and the average is the milkshake's own values."""
    s = schedule(starter_draft({"mango_milkshake": 5}, prefs={"slots_on": ["snack"]}))
    shake = s["recipe_nutrition"]["starter:mango_milkshake"]
    assert shake["status"] == "complete"
    p = s["period_nutrition"]
    days_with = {m["date"] for m in s["meals"] if m["date"]}
    assert p["days_complete"] == len(days_with) == 5 and p["slots_counted"] == ["snack"]
    for key in NUTRIENTS:
        assert p["per_day_average_over_complete_days"][key] == shake["totals"][key]["amount"]
    complete = [d["date"] for d in s["days"] if d["nutrition"]["complete"]]
    assert sorted(complete) == sorted(days_with)


def test_a_pasted_recipe_meal_has_no_demo_badge():
    key = "my:paste1"
    entry, resolved = doc_recipe(key, "Plain rice", 2, [("Basmati Rice", 300, "g", 8)],
                                 wanted=1)
    s = schedule(doc_draft({key: (entry, resolved)}, prefs={"slots_on": ["dinner"]}))
    (d,) = [d for d in s["days"] if d["meal_ids"]]
    assert not d["nutrition"]["demo_amounts"]
    assert d["nutrition"]["amounts_basis"] == ["parsed_from_your_paste"]
    assert d["nutrition"]["complete"]               # dinner is the only slot on


def test_unknown_servings_count_for_nothing_until_answered():
    key = "my:paste2"
    entry, resolved = doc_recipe(key, "Rice, serves ?", None, [("Basmati Rice", 300, "g", 8)],
                                 wanted=1)
    draft = doc_draft({key: (entry, resolved)}, prefs={"slots_on": ["dinner"]})
    s = schedule(draft)
    assert s["recipe_nutrition"][key]["basis"] == "per_recipe"
    (d,) = [d for d in s["days"] if d["meal_ids"]]
    assert not d["nutrition"]["complete"]
    assert d["nutrition"]["totals"]["energy_kcal"]["amount"] is None
    assert "Rice, serves ?: servings unknown" in d["nutrition"]["totals"]["energy_kcal"]["gaps"]
    draft["recipes"][key]["servings"] = 3
    s = schedule(draft)
    assert s["recipe_nutrition"][key]["basis"] == "per_serving"
    (d,) = [d for d in s["days"] if d["meal_ids"]]
    assert d["nutrition"]["complete"] and d["nutrition"]["totals"]["energy_kcal"]["amount"] > 0


def test_the_schedule_is_still_byte_identical():
    draft = starter_draft()
    a = client().post("/mealplan/schedule", json=draft)
    b = client().post("/mealplan/schedule", json=draft)
    assert a.status_code == 200 and a.content == b.content


def test_targets_from_the_draft_get_verdicts_only_where_proved():
    s = schedule(starter_draft(nutrition_targets={"energy_kcal": {"max": 100},
                                                  "fibre_g": {"min": 500}}))
    d = next(d for d in s["days"] if d["meal_ids"])
    assert d["nutrition"]["targets"]["energy_kcal"]["max"] == "over"
    assert d["nutrition"]["targets"]["fibre_g"]["min"] == "unknown"     # never 'short'
    bad = client().post("/mealplan/schedule",
                        json=starter_draft(nutrition_targets={"zinc": {"min": 1}}))
    assert bad.status_code == 422


def test_the_nutrition_source_is_cited():
    s = schedule(starter_draft())
    cnf = next(x for x in s["sources"] if x["id"] == "cnf-api")
    assert cnf["credit"].startswith(OGL) and "not the CNF 2026 files" in cnf["credit"]
    assert s["sources"][-1]["id"] == "demo-data"


def test_schedules_keep_working_without_the_tables():
    from pantry_planner import db

    for row in (db.IngredientNutrientMapRow, db.NutrientMeasureRow, db.NutrientAmountRow,
                db.NutrientFoodRow, db.NutrientSourceRow):
        row.__table__.drop(db.engine())
    try:
        s = schedule(starter_draft())
        assert s["coverage"]["nutrition"] == "not_deployed"
        assert s["period_nutrition"] is None and s["recipe_nutrition"] == {}
        assert all(d["nutrition"] is None for d in s["days"])
        assert not any(x["id"] == "cnf-api" for x in s["sources"])
        assert s["strategies"][0]["trips"]
    finally:
        db.seed_from_json()
    assert day(0) == START

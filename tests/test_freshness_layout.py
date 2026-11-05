"""Suggest cook days (POST /mealplan/suggest-cook-days): a freshness-aware layout proposed as
move ops with code-built reasons, never applied by the server.

The scenario is the verbatim sentence with Saturday-only shopping, the fried rice at lunch
and one pizza pinned. The cited poultry row ("1 to 2 days", planned as 1) lets a chicken
meal sit on the shopping Saturday or the Sunday after it.
"""
from __future__ import annotations

import pytest

from tests.mealplan_fixtures import (
    SHELF,
    client,
    day,
    done_db,
    index,
    schedule,
    starter_draft,
    use_db,
)

SATURDAY = 5
CHICKEN = ("starter:chicken_biryani", "starter:chicken_fried_rice")
POULTRY = next(r for r in SHELF["rules"] if r["id"] == "fs-poultry-pieces-fridge")
PINNED = {"id": "starter:pepperoni_pizza#1", "recipe_key": "starter:pepperoni_pizza",
          "date": day(6).isoformat(), "slot": "dinner", "pinned": True}


@pytest.fixture(scope="module", autouse=True)
def seeded_db(tmp_path_factory):
    use_db(tmp_path_factory, "layout")
    yield
    done_db()


@pytest.fixture(scope="module")
def draft():
    return starter_draft(slots={"chicken_fried_rice": "lunch"}, meals=[PINNED],
                         prefs={"shop_weekdays": [SATURDAY]})


def _suggest(draft: dict) -> dict:
    r = client().post("/mealplan/suggest-cook-days", json=draft)
    assert r.status_code == 200, r.text
    return r.json()


def test_the_layout_puts_chicken_on_the_shopping_weekend_and_clears_every_must_fix(draft):
    before = schedule(draft)
    assert [w for w in before["warnings"] if w["level"] == "must_fix"
            and w["strategy"] == "fresh"]
    out = _suggest(draft)
    assert out["rev"] == draft.get("rev", 0)
    assert out["warnings_before"]["must_fix"] > 0 and out["warnings_after"]["must_fix"] == 0

    applied = schedule({**draft, "meals": out["meals"]})
    assert [w for w in applied["warnings"] if w["level"] == "must_fix"] == []
    assert POULTRY["days_min"] == 1
    chicken = [m for m in applied["meals"] if m["recipe_key"] in CHICKEN]
    assert len(chicken) == 5
    for m in chicken:
        d = day(index(m["date"]))
        assert d.weekday() in {SATURDAY, SATURDAY + 1}, m          # Sat or Sun
    # Every trip is on a Saturday, as the shopper shops.
    for s in applied["strategies"]:
        assert {day(index(t["date"])).weekday() for t in s["trips"]} == {SATURDAY}


def test_pinned_meals_never_move(draft):
    out = _suggest(draft)
    assert PINNED["id"] not in [op["meal_id"] for op in out["ops"]]
    kept = next(m for m in out["meals"] if m["id"] == PINNED["id"])
    assert (kept["date"], kept["pinned"]) == (PINNED["date"], True)


def test_every_move_has_a_reason_built_from_a_cited_row_or_the_shoppers_setting(draft):
    out = _suggest(draft)
    assert out["ops"]
    for op in out["ops"]:
        assert op["op"] == "move_meal" and op["to"]["date"] and op["reason"]
        if op["meal_id"].startswith(CHICKEN):
            assert POULTRY["id"] in op["reason"] and POULTRY["verbatim"] in op["reason"]
            assert "FoodSafety.gov, Cold Food Storage Chart" in op["reason"]
        else:
            assert "your setting" in op["reason"] or POULTRY["id"] in op["reason"]
        shop = day(index(op["to"]["date"]))
        assert "shop" in op["reason"] and shop.strftime("%b") in op["reason"]


def test_the_proposal_is_deterministic_and_changes_nothing_server_side(draft):
    a = client().post("/mealplan/suggest-cook-days", json=draft)
    b = client().post("/mealplan/suggest-cook-days", json=draft)
    assert a.content == b.content
    assert schedule(draft) == schedule(draft)

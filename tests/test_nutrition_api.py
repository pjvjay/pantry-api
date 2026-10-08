"""GET /nutrition/recipes, GET /recipes/{slug}/nutrition, and nutrition on /plan/week.

The week runs with a deterministic selector stub (no LLM). Expected numbers come from
nutrition.meal_nutrition over the seed, which test_nutrition checks against arithmetic;
here the tests check what the endpoints carry and that plans keep working without the
tables.
"""
from __future__ import annotations

import json
import os

import pytest

from pantry_planner.db import SEEDS_DIR

RECIPES = json.loads((SEEDS_DIR / "recipes.json").read_text(encoding="utf-8"))
OGL = "Contains information licensed under the Open Government Licence – Canada."


@pytest.fixture(scope="module", autouse=True)
def seeded_db(tmp_path_factory):
    os.environ["DB_URL"] = (os.environ.get("PANTRY_TEST_DB_URL")
                            or f"sqlite:///{tmp_path_factory.mktemp('db') / 'nutr_api.db'}")
    from pantry_planner import config, db
    from pantry_planner.nlsearch import vocab

    config.settings.cache_clear()
    db.seed_from_json()
    vocab.clear_cache()
    yield
    config.settings.cache_clear()
    vocab.clear_cache()


def _client():
    from fastapi.testclient import TestClient

    from pantry_planner.api import app

    return TestClient(app)


def _stub_selector(ingredients, products, *, model, constraints=None, avoid=None, **_kw):
    """The cheapest product sharing the most words with the line; `avoid` maps a line name
    to a product name to skip, to show nutrition does not follow the pick."""
    from pantry_planner.models import Selection, SelectorResult
    from pantry_planner.nlsearch.units import tokens

    sels = []
    for ing in ingredients:
        toks = set(tokens(ing.name))
        pool = [p for p in products if not p.substitute] or products
        pool = [p for p in pool if p.name != (avoid or {}).get(ing.name)] or pool
        pick = min(pool, key=lambda p: (
            -len(toks & set(tokens(f"{p.name} {p.description}"))),
            p.store_price if p.store_price is not None else p.price, p.id))
        sels.append(Selection(line_no=ing.line_no, product_id=pick.id, confidence=0.95,
                              reasoning="stub"))
    return SelectorResult(selections=sels, total_cost=0, model_used="stub", cost_usd=0.0)


@pytest.fixture()
def stubbed_week(monkeypatch):
    from pantry_planner import weekplan

    monkeypatch.setattr(weekplan, "call_selector", _stub_selector)
    return weekplan


# ─── Endpoints ───────────────────────────────────────────────

def test_nutrition_recipes_lists_every_library_recipe_per_serving():
    r = _client().get("/nutrition/recipes")
    assert r.status_code == 200, r.text
    body = r.json()
    assert [x["slug"] for x in body["recipes"]] == sorted(r["slug"] for r in RECIPES)
    assert body["sources"][0]["attribution"] == OGL
    assert "not the CNF 2026 files" in body["sources"][0]["edition"]
    for item in body["recipes"]:
        n = item["nutrition"]
        assert n["basis"] == "per_serving" and n["demo_amounts"] is True
        assert n["totals"]["energy_kcal"]["amount"] > 0
        for t in n["totals"].values():
            assert t["amount"] is None or t["amount"] >= 0
            assert t["status"] in {"complete", "at_least", "unknown"}
    pbj = next(x for x in body["recipes"] if x["slug"] == "pbj_sandwich")["nutrition"]
    assert pbj["status"] == "below_floor"
    assert [m["ingredient"] for m in pbj["missing"]] == ["Peanut Butter and Jelly Jam"]


def test_recipe_nutrition_carries_receipts_and_sources():
    c = _client()
    r = c.get("/recipes/spaghetti_bolognese/nutrition")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["servings"] == 4 and body["sources"][0]["source"] == "cnf-api"
    lines = body["nutrition"]["lines"]
    assert [ln["ingredient"] for ln in lines] == [
        i["name"] for i in next(x for x in RECIPES if x["slug"] == "spaghetti_bolognese")[
            "ingredients"]]
    beef = next(ln for ln in lines if ln["ingredient"] == "Ground Beef")
    assert beef["ref_id"] == "cnf-api:2690" and beef["ref_description"] == "Beef, ground, medium, raw"
    assert beef["grams"] == 500 and beef["status"] == "counted"
    oil = next(ln for ln in lines if ln["ingredient"] == "Olive Oil")
    assert "source's measure" in oil["conversion"]       # 30 ml through CNF's own measure
    two = c.get("/recipes/spaghetti_bolognese/nutrition?portions=2").json()
    assert two["nutrition"]["totals"]["energy_kcal"]["amount"] == pytest.approx(
        2 * body["nutrition"]["totals"]["energy_kcal"]["amount"], abs=0.11)


def test_an_unknown_recipe_is_404_like_get_recipe():
    c = _client()
    r = c.get("/recipes/nope/nutrition")
    assert r.status_code == 404
    assert r.json() == c.get("/recipes/nope").json()
    assert c.get("/recipes/spaghetti_bolognese/nutrition?portions=0").status_code == 422
    assert c.get("/recipes/spaghetti_bolognese").json()["slug"] == "spaghetti_bolognese"


# ─── The week planner ────────────────────────────────────────

def test_week_days_carry_nutrition(stubbed_week):
    wp = stubbed_week.plan_week(days=5)
    assert wp.nutrition_sources and wp.nutrition_sources[0].attribution == OGL
    for day in wp.days:
        assert day.nutrition.source_ids == ["cnf-api"]
        assert day.nutrition.demo_amounts and day.nutrition.basis == "per_serving"
        t = day.day_totals
        assert t.meals_counted == ["dinner"] and not t.all_meals_planned and not t.complete
        assert t.note == "dinner only: other meals are not planned"
        assert t.totals["energy_kcal"].amount == day.nutrition.totals["energy_kcal"].amount
    assert not any("nutrition" in n for n in wp.notes)


def test_week_nutrition_does_not_follow_the_products_picked(stubbed_week, monkeypatch):
    """Nutrition reads the recipe's amounts and reference foods, so picking another ground
    beef changes the price, not the numbers."""
    first = stubbed_week.plan_week(days=5)
    day = next(d for d in first.days if any(li.ingredient_name == "Ground Beef"
                                             for li in d.line_items))
    beef = next(li for li in day.line_items if li.ingredient_name == "Ground Beef")
    monkeypatch.setattr(stubbed_week, "call_selector",
                        lambda *a, **kw: _stub_selector(*a, avoid={"Ground Beef":
                                                                   beef.product_name}, **kw))
    second = stubbed_week.plan_week(days=5)
    again = next(d for d in second.days if d.recipe_slug == day.recipe_slug)
    other = next(li for li in again.line_items if li.ingredient_name == "Ground Beef")
    assert other.product_id != beef.product_id
    assert again.nutrition == day.nutrition


def test_week_targets_give_verdicts_only_where_proved(stubbed_week):
    from pantry_planner.models import NutritionTarget

    wp = stubbed_week.plan_week(days=3, targets={
        "energy_kcal": NutritionTarget(max=100), "fat_g": NutritionTarget(min=500, max=900),
        "protein_g": NutritionTarget(min=1)})
    for day in wp.days:
        checks = day.day_totals.targets
        assert checks["energy_kcal"].max == "over"           # a dinner alone is above 100
        assert checks["fat_g"].min == "unknown"               # never 'short': dinner only
        assert checks["fat_g"].max == "unknown"               # never 'within' either
        assert checks["protein_g"].min == "met"


def test_week_endpoint_takes_targets_and_rejects_unknown_keys(monkeypatch):
    from pantry_planner import weekplan

    monkeypatch.setattr(weekplan, "call_selector", _stub_selector)
    c = _client()
    r = c.post("/plan/week", json={"days": 2, "targets": {"sodium_mg": {"max": 50}}})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["nutrition_sources"][0]["source"] == "cnf-api"
    assert body["days"][0]["day_totals"]["targets"]["sodium_mg"]["max"] in {"over", "unknown"}
    assert c.post("/plan/week", json={"days": 2, "targets": {"vitamin_c": {"min": 1}}}
                  ).status_code == 422
    assert c.post("/plan/week", json={
        "days": 2, "targets": {"energy_kcal": {"max": 2000, "source": "health_canada_dv"}}}
    ).status_code == 422


# ─── Without the tables ──────────────────────────────────────

def test_missing_nutrition_tables_leave_plans_working(stubbed_week):
    """An API that deploys before pantry-db 0008: plans succeed, nutrition is null with a
    note, and no number is shown as 0."""
    from pantry_planner import db
    from pantry_planner.nutrition import NOT_DEPLOYED

    for row in (db.IngredientNutrientMapRow, db.NutrientMeasureRow, db.NutrientAmountRow,
                db.NutrientFoodRow, db.NutrientSourceRow):
        row.__table__.drop(db.engine())
    try:
        wp = stubbed_week.plan_week(days=3)
        assert len(wp.days) == 3 and wp.shopping_list
        assert all(d.nutrition is None and d.day_totals is None for d in wp.days)
        assert NOT_DEPLOYED in wp.notes and wp.nutrition_sources == []
        c = _client()
        body = c.get("/nutrition/recipes").json()
        assert all(r["nutrition"] is None and "not deployed" in r["note"]
                   for r in body["recipes"])
        assert body["sources"] == [] and "not deployed" in body["note"]
        one = c.get("/recipes/chicken_curry/nutrition").json()
        assert one["nutrition"] is None and "pantry-db 0008" in one["note"]
        assert c.get("/recipes/chicken_curry").status_code == 200
    finally:
        db.seed_from_json()

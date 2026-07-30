"""DEMO_MODE: both LLM boundaries replaced, everything else identical.

Runs the real API surface with DEMO_MODE=1 and no ANTHROPIC_API_KEY —
if anything under test tried to call Anthropic it would fail on auth,
so green tests prove the demo path never leaves the box.
"""
from __future__ import annotations

import os

import pytest


@pytest.fixture(scope="module", autouse=True)
def demo_env(tmp_path_factory):
    os.environ["DB_URL"] = f"sqlite:///{tmp_path_factory.mktemp('db') / 'demo.db'}"
    os.environ["DEMO_MODE"] = "1"
    os.environ.pop("ANTHROPIC_API_KEY", None)
    from pantry_planner import config, db
    from pantry_planner.nlsearch import vocab

    config.settings.cache_clear()
    db.seed_from_json()
    vocab.clear_cache()
    yield
    os.environ.pop("DEMO_MODE", None)
    config.settings.cache_clear()
    vocab.clear_cache()


BOLOGNESE = """Spaghetti Bolognese (serves 4)
- 400g spaghetti
- 500g ground beef
- 1 can crushed tomatoes
Notes: under $30, no dairy, only stores within 10km
"""


def test_demo_parse_is_deterministic_and_constraint_aware():
    from pantry_planner.demomode import parse_recipe

    p1, p2 = parse_recipe(BOLOGNESE), parse_recipe(BOLOGNESE)
    assert p1 == p2                                   # same input, same parse
    assert p1.recipe.servings == 4
    assert [i.name for i in p1.recipe.ingredients] == \
        ["spaghetti", "beef", "crushed tomatoes"]
    assert p1.recipe.ingredients[1].form == "ground"  # form split out
    assert p1.recipe.ingredients[2].form == "canned"  # "1 can" implies form
    assert p1.constraints.max_total_budget == 30
    assert p1.constraints.max_distance_km == 10
    assert p1.constraints.exclude_tags == ["dairy"]
    assert p1.cost_usd == 0.0


def test_plan_nl_end_to_end_without_api_key():
    from fastapi.testclient import TestClient

    from pantry_planner.api import app

    resp = TestClient(app).post("/plan/nl", json={"recipe_text": BOLOGNESE})
    assert resp.status_code == 200, resp.text
    plan = resp.json()
    assert plan["total_llm_cost_usd"] == 0.0
    assert all(li["model_used"] == "demo-deterministic"
               for li in plan["line_items"])
    # the real machinery still ran: full trace + trip options + gates armed
    steps = [s["step_id"] for s in plan["plan_trace"]]
    assert steps[:3] == ["t1_existence", "t2_options", "t3_statistics"]
    assert steps[-1] == "t5_trip_optimizer"
    assert plan["trip_options"]
    assert "stores ≤ 10 km" in " ".join(plan["interpretation"])


def test_gates_still_fire_in_demo_mode():
    from fastapi.testclient import TestClient

    from pantry_planner.api import app

    resp = TestClient(app).post("/plan/nl", json={
        "recipe_text": "Exotic\n- 10g saffron\n- 200g spaghetti"})
    assert resp.status_code == 409
    assert resp.json()["detail"]["aborted"]["code"] == "missing_ingredients"


def test_plan_week_in_demo_mode():
    from fastapi.testclient import TestClient

    from pantry_planner.api import app

    resp = TestClient(app).post("/plan/week", json={"days": 3})
    assert resp.status_code == 200, resp.text
    wp = resp.json()
    assert len(wp["days"]) == 3
    assert wp["total_llm_cost_usd"] == 0.0
    assert wp["overlap_savings"] >= 0


def test_health_reports_demo_mode_and_spa_mount():
    from fastapi.testclient import TestClient

    from pantry_planner.api import app
    from pantry_planner import demo_server

    assert TestClient(app).get("/health").json()["demo_mode"] is True
    # demo_server routes /pantry/api/* to the same app (SPA mount is
    # conditional on SPA_DIST, absent in tests)
    resp = TestClient(demo_server.root).get("/pantry/api/health")
    assert resp.status_code == 200 and resp.json()["demo_mode"] is True


def test_three_phase_router_works_in_demo_mode():
    from pantry_planner.models import Recipe, RecipeIngredient
    from pantry_planner import db
    from pantry_planner.router.classifier import call_classifier

    recipe = Recipe(slug="t", name="t", ingredients=[
        RecipeIngredient(line_no=1, name="spaghetti")])
    metrics = call_classifier(recipe, db.load_all_products())
    assert metrics.cost_usd == 0.0
    assert 1 <= metrics.match_confidence_1_to_10 <= 10

"""Library recipe amounts: seeds/recipes.json -> recipe_line_amounts -> GET /recipes/{slug}/doc.

The library's amounts are demo house amounts (synthetic, labelled). Expected values come
from seeds/recipes.json read directly, never from the loader under test.
"""
from __future__ import annotations

import json
import os

import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session

from pantry_planner.db import SEEDS_DIR

RECIPES = json.loads((SEEDS_DIR / "recipes.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module", autouse=True)
def seeded_db(tmp_path_factory):
    os.environ["DB_URL"] = (os.environ.get("PANTRY_TEST_DB_URL")
                            or f"sqlite:///{tmp_path_factory.mktemp('db') / 'amounts.db'}")
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


def _client():
    from fastapi.testclient import TestClient

    from pantry_planner.api import app

    return TestClient(app)


def _count(sql: str) -> int:
    from pantry_planner.db import engine

    with Session(engine()) as s:
        return int(s.execute(text(sql)).scalar_one())


def test_every_library_line_is_quantified_or_noted():
    lines = [ing for r in RECIPES for ing in r["ingredients"]]
    assert len(lines) == 37
    for ing in lines:
        assert {"quantity", "unit", "note"} <= ing.keys(), ing
        assert ing["quantity"] is not None or ing["note"].strip(), ing
        if ing["quantity"] is not None:
            assert ing["quantity"] > 0 and ing["unit"] in {"g", "ml"}, ing


def test_the_loader_returns_exactly_the_seed():
    from pantry_planner import db

    got = db.load_line_amounts([r["slug"] for r in RECIPES])
    want = {r["slug"]: {i: (float(ing["quantity"]), ing["unit"], ing["note"])
                        for i, ing in enumerate(r["ingredients"], start=1)}
            for r in RECIPES}
    assert {s: {n: tuple(a) for n, a in lines.items()} for s, lines in got.items()} == want
    assert db.load_line_amounts([]) == {}
    assert db.load_line_amounts(["no_such_recipe"]) == {}


def test_reseeding_clears_the_amounts_first():
    from pantry_planner import db

    db.seed_from_json()
    db.seed_from_json()
    assert _count("SELECT COUNT(*) FROM recipe_line_amounts") == 37


def test_load_recipe_is_unchanged():
    from pantry_planner import db
    from pantry_planner.models import RecipeIngredient

    r = db.load_recipe("chicken_curry")
    assert set(RecipeIngredient.model_fields) == {"line_no", "name", "category"}
    assert [i.name for i in r.ingredients] == [i["name"] for i in RECIPES[2]["ingredients"]]


def test_recipe_doc_carries_the_labelled_amounts():
    resp = _client().get("/recipes/chicken_curry/doc")
    assert resp.status_code == 200, resp.text
    doc = resp.json()
    seed = RECIPES[2]
    assert (doc["key"], doc["title"], doc["servings"], doc["servings_basis"]) == \
        ("lib:chicken_curry", seed["name"], seed["servings"], "source")
    assert doc["source"]["kind"] == "library" and doc["warnings"] == []
    assert [(ln["name"], ln["quantity"], ln["unit"], ln["note"]) for ln in doc["lines"]] == \
        [(i["name"], float(i["quantity"]), i["unit"], i["note"]) for i in seed["ingredients"]]
    assert {ln["amount_basis"] for ln in doc["lines"]} == {"demo_house_amounts"}
    assert doc["lines"][0]["text"] == "700 g Chicken Thighs"
    assert _client().get("/recipes/no_such_recipe/doc").status_code == 404


def test_a_library_doc_plans_through_plan_spec_as_reviewed():
    doc = _client().get("/recipes/tomato_penne/doc").json()
    plan = _client().post("/plan/spec", json={"doc": doc}).json()
    assert [(ln["name"], ln["quantity"], ln["unit"]) for ln in plan["basis"]["lines"]] == \
        [(ln["name"], ln["quantity"], ln["unit"]) for ln in doc["lines"]]
    # amounts known in the pack's unit: the purchase states its need
    penne = next(li for li in plan["line_items"] if li["line_no"] == 1)
    assert (penne["need_qty"], penne["need_uom"]) == (250.0, "g")


def test_an_old_database_has_no_amounts_and_says_so():
    """Before pantry-db 0007 the table is missing: None (unknown), never {} (no amounts)."""
    from pantry_planner import db
    from pantry_planner.recipe_doc import AMOUNTS_MISSING

    db.RecipeLineAmountRow.__table__.drop(db.engine())
    try:
        assert db.load_line_amounts(["chicken_curry"]) is None
        doc = _client().get("/recipes/chicken_curry/doc").json()
        assert doc["warnings"] == [AMOUNTS_MISSING]
        assert all(ln["quantity"] is None and ln["unit"] == "" for ln in doc["lines"])
        assert _client().get("/recipes/chicken_curry").status_code == 200
    finally:
        db.seed_from_json()           # init_schema puts the table back
    assert db.load_line_amounts(["chicken_curry"]) is not None

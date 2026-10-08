"""seeds/nutrients.json and its tables: the reference data nutrition is computed from.

Expected values come from the seed file read directly, never from the loader under test.
The file is byte-identical to pantry-db's (the platform's cmp step checks that).
"""
from __future__ import annotations

import json
import math
import os

import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session

from pantry_planner.db import SEEDS_DIR
from pantry_planner.nlsearch.units import tokens

DATA = json.loads((SEEDS_DIR / "nutrients.json").read_text(encoding="utf-8"))
RECIPES = json.loads((SEEDS_DIR / "recipes.json").read_text(encoding="utf-8"))
STARTERS = json.loads((SEEDS_DIR / "mealplan_starters.json").read_text())["starters"]
NUTRIENTS = ("energy_kcal", "protein_g", "fat_g", "satfat_g", "carbohydrate_g", "fibre_g",
             "sugars_g", "sodium_mg")
FOODS = {f["ref_id"]: f for f in DATA["foods"]}
OGL = "Contains information licensed under the Open Government Licence – Canada."


@pytest.fixture(scope="module", autouse=True)
def seeded_db(tmp_path_factory):
    os.environ["DB_URL"] = (os.environ.get("PANTRY_TEST_DB_URL")
                            or f"sqlite:///{tmp_path_factory.mktemp('db') / 'nutr_seed.db'}")
    from pantry_planner import config, db

    config.settings.cache_clear()
    db.seed_from_json()
    yield
    config.settings.cache_clear()


def _count(sql: str) -> int:
    from pantry_planner.db import engine

    with Session(engine()) as s:
        return int(s.execute(text(sql)).scalar_one())


# ─── The file ────────────────────────────────────────────────

def test_every_recipe_line_key_has_a_map_row():
    """A tokenizer change that re-keys an ingredient fails here, not silently in a plan."""
    keys = {m["ingredient_key"] for m in DATA["map"]}
    names = [ing["name"] for r in RECIPES for ing in r["ingredients"]]
    names += [ln["name"] for s in STARTERS for ln in s["lines"]]
    assert len(names) == 37 + 36
    for name in names:
        assert " ".join(tokens(name)) in keys, name


def test_every_map_row_points_at_a_food_and_only_none_has_none():
    for m in DATA["map"]:
        assert m["match_kind"] in {"generic", "close", "none"}
        assert (m["match_kind"] == "none") == (m["ref_id"] is None), m
        assert m["ref_id"] is None or m["ref_id"] in FOODS, m
        assert m["match_kind"] != "close" or m["note"], f"a close match says how: {m}"
    assert {m["ingredient_key"] for m in DATA["map"] if m["match_kind"] == "none"} == {
        "garam masala", "peanut butter and jelly jam"}


def test_every_food_is_cited_and_described():
    sources = {s["source"]: s for s in DATA["sources"]}
    for f in DATA["foods"]:
        assert f["ref_id"] == f"{f['source']}:{f['food_code']}"
        assert f["source"] in sources
        assert f["description"].strip() and f["state_note"].strip()
        assert {"energy_kcal", "protein_g"} <= f["per_100g"].keys()
        assert set(f["per_100g"]) <= set(NUTRIENTS)
        assert set(f["absent"]) == set(NUTRIENTS) - set(f["per_100g"])


def test_the_source_carries_the_attribution_and_its_edition_caveat():
    (src,) = DATA["sources"]
    assert src["attribution"] == OGL
    assert "not the CNF 2026 files" in src["edition"]
    assert "not verified" in src["licence"]


def test_values_are_published_numbers_within_bounds():
    for f in DATA["foods"]:
        for n, v in f["per_100g"].items():
            assert isinstance(v, (int, float)) and math.isfinite(v) and v >= 0, (f["ref_id"], n)
            if n.endswith("_g"):
                assert v <= 100, (f["ref_id"], n, v)


def test_energy_passes_the_atwater_check_or_says_why_not():
    exempt = set()
    for f in DATA["foods"]:
        p = f["per_100g"]
        e = p["energy_kcal"]
        atwater = 4 * p["protein_g"] + 9 * p["fat_g"] + 4 * p["carbohydrate_g"]
        if abs(e - atwater) > max(25, 0.2 * e):
            assert f.get("atwater_note"), f["ref_id"]
            exempt.add(f["ref_id"])
    assert exempt == {"cnf-api:174", "cnf-api:177", "cnf-api:195", "cnf-api:4008"}


def test_measures_are_the_sources_own_weights():
    for m in DATA["measures"]:
        assert m["ref_id"] in FOODS and m["grams"] > 0 and m["verbatim"]
        assert (m["volume_ml"] is not None) == m["measure"].startswith("vol_"), m


# ─── The tables ──────────────────────────────────────────────

def test_the_loader_returns_exactly_the_seed():
    from pantry_planner import db

    ref = db.load_reference()
    assert set(ref.foods) == set(FOODS)
    for rid, f in FOODS.items():
        assert ref.foods[rid].per_100g == {n: float(v) for n, v in f["per_100g"].items()}
        assert ref.foods[rid].description == f["description"]
    assert "sugars_g" not in ref.foods["cnf-api:4484"].per_100g     # unknown, not 0
    assert ref.foods["cnf-api:113"].per_100g["fibre_g"] == 0.0        # a published 0
    assert {k: (e.ref_id, e.match_kind) for k, e in ref.map.items()} == {
        m["ingredient_key"]: (m["ref_id"], m["match_kind"]) for m in DATA["map"]}
    assert sum(len(v) for v in ref.measures.values()) == len(DATA["measures"])
    assert ref.measures_deployed
    assert ref.sources["cnf-api"]["attribution"] == OGL


def test_the_loader_filters_by_key():
    from pantry_planner import db

    ref = db.load_reference(["ground beef", "no such key"])
    assert set(ref.map) == {"ground beef"} and set(ref.foods) == {"cnf-api:2690"}
    empty = db.load_reference([])
    assert empty.map == {} and empty.foods == {} and "cnf-api" in empty.sources


def test_reseeding_clears_the_tables_first():
    from pantry_planner import db

    db.seed_from_json()
    db.seed_from_json()
    assert _count("SELECT COUNT(*) FROM nutrient_foods") == len(DATA["foods"])
    assert _count("SELECT COUNT(*) FROM nutrient_amounts") == sum(
        len(f["per_100g"]) for f in DATA["foods"])
    assert _count("SELECT COUNT(*) FROM nutrient_measures") == len(DATA["measures"])
    assert _count("SELECT COUNT(*) FROM ingredient_nutrient_map") == len(DATA["map"])
    assert _count("SELECT COUNT(*) FROM nutrient_sources") == 1


def test_an_old_database_without_the_tables_reads_none_and_recipes_still_load():
    """Before pantry-db 0008 the tables are missing: None (unknown), never an empty
    reference; before 0009 only the measures are missing, and that is said."""
    from pantry_planner import db

    try:
        db.NutrientMeasureRow.__table__.drop(db.engine())
        ref = db.load_reference()
        assert ref is not None and ref.measures == {} and not ref.measures_deployed
        for row in (db.IngredientNutrientMapRow, db.NutrientAmountRow, db.NutrientFoodRow,
                    db.NutrientSourceRow):
            row.__table__.drop(db.engine())
        assert db.load_reference() is None
        assert db.load_reference(["ground beef"]) is None
        assert db.load_recipe("spaghetti_bolognese").ingredients
    finally:
        db.seed_from_json()           # init_schema puts the tables back
    assert db.load_reference() is not None

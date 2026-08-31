"""Empty-list SQL guards.

Every one of these crashed before: a joined list interpolated into SQL with
no empty case. `AS (VALUES )` and `IN ()` are syntax errors, and the paths
that reach them are ordinary — an empty recipe library, a recipe whose only
ingredient tokenises to nothing, a basket the selector could not fill.
"""
from __future__ import annotations

import os

import pytest

_TMP_DB = None


@pytest.fixture(scope="module", autouse=True)
def seeded_db(tmp_path_factory):
    global _TMP_DB
    _TMP_DB = tmp_path_factory.mktemp("db") / "test.db"
    os.environ["DB_URL"] = f"sqlite:///{_TMP_DB}"
    os.environ["DEMO_MODE"] = "1"
    from pantry_planner import config, db

    config.settings.cache_clear()
    db.seed_from_json()
    yield
    os.environ.pop("DEMO_MODE", None)
    config.settings.cache_clear()


def _empty_library():
    from sqlalchemy.orm import Session

    from pantry_planner.db import RecipeIngredientRow, RecipeRow, engine

    with Session(engine()) as s:
        s.query(RecipeIngredientRow).delete()
        s.query(RecipeRow).delete()
        s.commit()


def test_plan_week_on_empty_library_gates_instead_of_500ing():
    """Was: OperationalError 'near ")"' -> HTTP 500."""
    from pantry_planner import db, weekplan
    from pantry_planner.nlsearch import PlanAborted

    _empty_library()
    try:
        with pytest.raises(PlanAborted):
            weekplan.plan_week(days=5)
    finally:
        db.seed_from_json()


def test_plan_week_empty_library_returns_409_not_500():
    from fastapi.testclient import TestClient

    from pantry_planner import api, db

    _empty_library()
    try:
        with TestClient(api.app) as c:
            r = c.post("/plan/week", json={"days": 5})
        assert r.status_code == 409, f"got {r.status_code}: {r.text[:200]}"
    finally:
        db.seed_from_json()


def test_options_sql_survives_a_tokenless_ingredient():
    """A name of only short tokens yields no terms — the VALUES list empties."""
    from pantry_planner.nlsearch.schemas import Constraints, IngredientSpec
    from pantry_planner.nlsearch.sql_builder import build_options_sql

    sql, params = build_options_sql(
        Constraints(), [IngredientSpec(name="of the")], relaxed=set(),
        lat=49.28, lon=-123.12)
    assert "VALUES )" not in sql
    assert "AS (VALUES )" not in sql


def test_existence_sql_survives_a_tokenless_ingredient():
    from pantry_planner.nlsearch.schemas import IngredientSpec
    from pantry_planner.nlsearch.sql_builder import build_existence_sql

    sql, params = build_existence_sql([IngredientSpec(name="2")])
    assert "VALUES )" not in sql


def test_in_clauses_never_emit_the_postgres_syntax_error():
    """`IN ()` parses on SQLite and is a syntax error on Postgres, so an
    empty basket here would crash only in production."""
    from pantry_planner.nlsearch.sql_builder import (
        build_price_matrix_sql,
        build_stats_sql,
    )

    sql, _ = build_price_matrix_sql([], lat=49.28, lon=-123.12, max_km=None)
    assert "IN ()" not in sql
    assert "IN (NULL)" in sql

    sql2, _ = build_stats_sql([])
    assert "IN ()" not in sql2


def test_tokenless_library_does_not_crash_the_week_planner():
    from sqlalchemy.orm import Session

    from pantry_planner import db, weekplan
    from pantry_planner.db import RecipeIngredientRow, RecipeRow, engine
    from pantry_planner.nlsearch import PlanAborted

    _empty_library()
    with Session(engine()) as s:
        s.add(RecipeRow(slug="z", name="Z", servings=2))
        s.add(RecipeIngredientRow(recipe_slug="z", line_no=1, name="2",
                                  category=None))
        s.commit()
    try:
        # Must fail as a gate, never as an OperationalError.
        with pytest.raises(PlanAborted):
            weekplan.plan_week(days=1)
    finally:
        db.seed_from_json()

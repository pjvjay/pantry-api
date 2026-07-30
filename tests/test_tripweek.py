"""4A split-trip optimizer + 5A weekly menu optimizer — no LLM, no network.

tripopt is tested as pure math on synthetic matrices; the flow/API
integration runs with the parse and selector stubbed; weekplan runs
against the seeded SQLite DB with a deterministic selector stub.
"""
from __future__ import annotations

import os

import pytest


@pytest.fixture(scope="module", autouse=True)
def seeded_db(tmp_path_factory):
    os.environ["DB_URL"] = f"sqlite:///{tmp_path_factory.mktemp('db') / 'test.db'}"
    from pantry_planner import config, db
    from pantry_planner.nlsearch import vocab

    config.settings.cache_clear()
    db.seed_from_json()
    vocab.clear_cache()
    yield
    config.settings.cache_clear()
    vocab.clear_cache()


def _stub_selector(ingredients, products, *, model, enable_thinking=False,
                   constraints=None):
    from pantry_planner.models import Selection, SelectorResult
    from pantry_planner.nlsearch.units import tokens

    sels = []
    for ing in ingredients:
        toks = set(tokens(ing.name))
        pool = [p for p in products if not p.substitute] or products
        pick = min(pool, key=lambda p: (
            -len(toks & set(tokens(f"{p.name} {p.description}"))),
            p.store_price if p.store_price is not None else p.price))
        sels.append(Selection(line_no=ing.line_no, product_id=pick.id,
                              confidence=0.95, reasoning="stub"))
    return SelectorResult(selections=sels, total_cost=0, model_used="stub",
                          cost_usd=0.004)


# ─── 4A: tripopt pure math ───────────────────────────────────

def _matrix(rows):
    """rows: (store_id, name, lat, lon, dist_km2, product_id, price)"""
    keys = ("store_id", "store_name", "lat", "lon", "dist_km2", "product_id", "price")
    return [dict(zip(keys, r)) for r in rows]


def test_tripopt_exact_frontier_math():
    """Two stores, two products with opposite cheap stores: the split saves
    on basket but pays travel; totals must be exact."""
    from pantry_planner import tripopt

    home = (49.28, -123.12)
    near = (49.28, -123.12 + 1 / (111.0 * 0.652))   # ~1.0 km east (coslat≈.652)
    far = (49.28 - 10 / 111.0, -123.12)             # 10.0 km south
    rows = _matrix([
        (1, "Near", *near, 1.0, 101, 5.00), (1, "Near", *near, 1.0, 102, 9.00),
        (2, "Far", *far, 100.0, 101, 4.00), (2, "Far", *far, 100.0, 102, 2.00),
    ])
    opts = tripopt.optimize_trips(rows, [(101, "A"), (102, "B")],
                                  home_lat=home[0], home_lon=home[1],
                                  cost_per_km=0.50)
    # Near: basket 14.00 + 2km loop * 0.5 = 15.00
    # Far:  basket  6.00 + 20km loop * 0.5 = 16.00  -> Near wins the 1-stop
    # {Near,Far}: both products cheapest at Far -> collapses to Far (16.00),
    # loses to Near — so the frontier is a single 1-stop option.
    assert len(opts) == 1
    rec = opts[0]
    assert rec.recommended and rec.stores == ["Near"]
    assert rec.basket_cost == 14.00
    assert rec.travel_km == pytest.approx(2.0, abs=0.1)
    assert rec.total_cost == pytest.approx(15.0, abs=0.1)
    assert [i.product_id for i in rec.items] == [101, 102]


def test_tripopt_split_beats_one_stop_when_savings_exceed_travel():
    from pantry_planner import tripopt

    home = (49.28, -123.12)
    near1 = (49.28 + 1 / 111.0, -123.12)            # 1 km north
    near2 = (49.28 - 1 / 111.0, -123.12)            # 1 km south
    rows = _matrix([
        (1, "N1", *near1, 1.0, 1, 1.00), (1, "N1", *near1, 1.0, 2, 50.00),
        (2, "N2", *near2, 1.0, 1, 50.00), (2, "N2", *near2, 1.0, 2, 1.00),
    ])
    opts = tripopt.optimize_trips(rows, [(1, "A"), (2, "B")],
                                  home_lat=home[0], home_lon=home[1],
                                  cost_per_km=0.50)
    rec = next(o for o in opts if o.recommended)
    assert len(rec.stores) == 2                     # $49 basket saving >> ~$1 travel
    assert rec.basket_cost == 2.00
    assert rec.savings_vs_one_stop > 45
    assert {i.store_name for i in rec.items} == {"N1", "N2"}


def test_tripopt_infeasible_store_skipped_and_duplicates_priced_per_line():
    from pantry_planner import tripopt

    home = (49.28, -123.12)
    rows = _matrix([
        (1, "OnlyA", 49.29, -123.12, 1.0, 1, 3.00),   # stocks only product 1
        (2, "Full", 49.27, -123.12, 1.0, 1, 4.00), (2, "Full", 49.27, -123.12, 1.0, 2, 4.00),
    ])
    opts = tripopt.optimize_trips(rows, [(1, "A"), (2, "B"), (1, "A")],
                                  home_lat=home[0], home_lon=home[1],
                                  cost_per_km=0.50)
    # OnlyA alone can't cover product 2 → no 1-stop OnlyA option
    assert all("OnlyA" not in o.stores or len(o.stores) > 1 for o in opts)
    rec = next(o for o in opts if o.recommended)
    assert len(rec.items) == 3                      # duplicate line kept
    one_stop = next(o for o in opts if len(o.stores) == 1)
    assert one_stop.stores == ["Full"] and one_stop.basket_cost == 12.00


def test_price_matrix_sql_binding():
    from pantry_planner.nlsearch.sql_builder import build_price_matrix_sql

    sql, params = build_price_matrix_sql([31, 5], 49.28, -123.12, 10)
    assert ":mp0" in sql and params["mp0"] == 31 and params["mp1"] == 5
    assert params["maxdist2"] == 100
    assert "store_products" in sql and "dist_km2" in sql


# ─── 4A through the flow + API (stubbed LLM) ─────────────────

def test_plan_nl_carries_trip_options(monkeypatch):
    from fastapi.testclient import TestClient

    from pantry_planner import flow
    from pantry_planner.api import app
    from pantry_planner.nlsearch import planner
    from pantry_planner.nlsearch.schemas import (Constraints, IngredientSpec,
                                                 ParsedInput, RecipeSpec)

    parsed = ParsedInput(
        recipe=RecipeSpec(title="T", ingredients=[
            IngredientSpec(name="spaghetti"), IngredientSpec(name="cheddar")]),
        constraints=Constraints())
    monkeypatch.setattr(planner, "parse_input", lambda t, model=None: parsed)
    monkeypatch.setattr(flow, "call_selector", _stub_selector)

    resp = TestClient(app).post("/plan/nl", json={"recipe_text": "x"})
    assert resp.status_code == 200, resp.text
    plan = resp.json()
    assert plan["plan_trace"][-1]["step_id"] == "t5_trip_optimizer"
    opts = plan["trip_options"]
    assert opts and sum(o["recommended"] for o in opts) == 1
    rec = next(o for o in opts if o["recommended"])
    assert rec["total_cost"] == min(o["total_cost"] for o in opts)
    assert len(rec["items"]) == len(plan["line_items"])
    # non-recommended options stay lean
    assert all(o["items"] == [] for o in opts if not o["recommended"])


def test_classic_path_has_no_trip_options(monkeypatch):
    from fastapi.testclient import TestClient

    from pantry_planner import flow
    from pantry_planner.api import app

    monkeypatch.setattr(flow, "call_selector", _stub_selector)
    resp = TestClient(app).post("/plan/grilled_cheese")
    assert resp.status_code == 200, resp.text
    assert resp.json()["trip_options"] == []


# ─── 5A: weekly menu optimizer ───────────────────────────────

@pytest.fixture()
def stubbed_week(monkeypatch):
    from pantry_planner import weekplan

    calls = {"n": 0}

    def counting_stub(*a, **kw):
        calls["n"] += 1
        return _stub_selector(*a, **kw)

    monkeypatch.setattr(weekplan, "call_selector", counting_stub)
    return weekplan, calls


def test_week_single_pass_retrieval_and_pbj_drop(stubbed_week):
    weekplan, _ = stubbed_week
    wp = weekplan.plan_week(days=5)
    w1 = [s for s in wp.plan_trace if s.step_id == "w1_options"]
    assert len(w1) == 1                             # whole library, ONE query
    assert any("Peanut Butter" in n for n in wp.notes)   # no jam product → dropped
    assert len(wp.days) == 5
    assert wp.plan_trace[-1].step_id == "t5_trip_optimizer"
    assert wp.trip_options


def test_week_overlap_is_rewarded_exactly(stubbed_week):
    weekplan, _ = stubbed_week
    wp = weekplan.plan_week(days=5)
    garlic = next(w for w in wp.shopping_list if "Garlic" in w.product_name)
    assert len(garlic.used_by) >= 3                 # shared, listed once
    assert wp.overlap_savings > 0
    assert wp.total_cost + wp.overlap_savings == pytest.approx(wp.standalone_cost, abs=0.02)
    # every shopping-list product appears exactly once
    ids = [w.product_id for w in wp.shopping_list]
    assert len(ids) == len(set(ids))


def test_week_budget_gate_fires_before_llm(stubbed_week):
    from pantry_planner.nlsearch.plan import GateCode
    from pantry_planner.nlsearch.planner import PlanAborted

    weekplan, calls = stubbed_week
    with pytest.raises(PlanAborted) as exc:
        weekplan.plan_week(days=5, max_total_budget=5)
    assert exc.value.execution.aborted.code == GateCode.budget_infeasible
    assert calls["n"] == 0                          # no selector spend on abort
    assert exc.value.execution.steps[-1].outcome == "aborted"


def test_week_dietary_knockout_and_clamp(stubbed_week):
    weekplan, _ = stubbed_week
    wp = weekplan.plan_week(days=7, exclude_tags=["dairy", "plutonium"])
    assert any("plutonium" in n for n in wp.notes)   # unknown tag surfaced
    # PBJ is infeasible (no jam product), so 7 requested -> 6 planned; the
    # cheese/butter pools survive via non-dairy alternatives (fallback), but
    # no picked product may contain dairy.
    assert len(wp.days) == 6
    assert any("remain feasible" in n for n in wp.notes)
    dairy_names = ("Whole Milk", "Cheddar Cheese Block", "Butter Salted",
                   "Mozzarella Shredded")
    for day in wp.days:
        for li in day.line_items:
            assert not any(d in li.product_name for d in dairy_names), li.product_name


def test_week_endpoint(monkeypatch):
    from fastapi.testclient import TestClient

    from pantry_planner import weekplan
    from pantry_planner.api import app

    monkeypatch.setattr(weekplan, "call_selector", _stub_selector)
    client = TestClient(app)
    resp = client.post("/plan/week", json={"days": 3})
    assert resp.status_code == 200, resp.text
    wp = resp.json()
    assert len(wp["days"]) == 3 and wp["shopping_list"]
    resp = client.post("/plan/week", json={"days": 5, "max_total_budget": 4})
    assert resp.status_code == 409
    assert resp.json()["detail"]["aborted"]["code"] == "budget_infeasible"

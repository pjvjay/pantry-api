"""plan_recipe with a shopping location: each line at its cheapest store in range, a trip split,
and a chosen product no store in range sells reported in out_of_range. Without a location the
plan is exactly what it was. Runs in DEMO_MODE on the seeded catalog: no LLM, real SQL.
"""
from __future__ import annotations

import os

import pytest
from mcp.server.mcpserver.exceptions import ToolError
from sqlalchemy import text
from sqlalchemy.orm import Session

DOWNTOWN = {"lat": 49.2827, "lon": -123.1207}


@pytest.fixture(scope="module", autouse=True)
def demo_env(tmp_path_factory):
    os.environ["DB_URL"] = os.environ.get("PANTRY_TEST_DB_URL") or \
        f"sqlite:///{tmp_path_factory.mktemp('db') / 'location.db'}"
    os.environ["DEMO_MODE"] = "1"
    from pantry_planner import config, db
    from pantry_planner.nlsearch import vocab

    config.settings.cache_clear()
    db.seed_from_json()
    vocab.clear_cache()
    yield
    os.environ.pop("DEMO_MODE", None)
    config.settings.cache_clear()
    vocab.clear_cache()


def _query(sql: str, **params) -> list[dict]:
    from pantry_planner import db

    with Session(db.engine()) as s:
        return [dict(r) for r in s.execute(text(sql), params).mappings()]


def _execute(sql: str, **params) -> None:
    from pantry_planner import db

    with Session(db.engine()) as s:
        s.execute(text(sql), params)
        s.commit()


def _cheapest_price(product_id: int) -> float:
    return _query("SELECT MIN(price) AS p FROM store_products WHERE product_id = :p",
                  p=product_id)[0]["p"]


def _lines_covered(plan) -> set[str]:
    names = {name for li in plan.line_items for name in li.ingredient_name.split(" + ")}
    return names | {name for d in plan.out_of_range for name in d.ingredient.split(" + ")}


def test_without_a_location_the_plan_is_unchanged():
    from pantry_planner import flow

    plan = flow.run("tomato_penne")
    assert [li.store_name for li in plan.line_items] == [""] * len(plan.line_items)
    assert plan.trip_options == [] and plan.out_of_range == []


def test_a_location_prices_every_line_at_its_cheapest_store_and_splits_the_trip():
    from pantry_planner import flow

    catalog = flow.run("tomato_penne")
    plan = flow.run("tomato_penne", **DOWNTOWN)                  # any distance
    # The same products are chosen; only where to buy them (and so the price) is new.
    assert [li.product_id for li in plan.line_items] == [li.product_id for li in catalog.line_items]
    for li in plan.line_items:
        assert li.store_name, li.product_name
        assert li.store_price == _cheapest_price(li.product_id)
        assert li.price == round(li.store_price * li.packs, 2)
    assert plan.total_cost == round(sum(li.price for li in plan.line_items), 2)
    assert plan.out_of_range == []
    [best] = [o for o in plan.trip_options if o.recommended]
    assert {i.product_id for i in best.items} == {li.product_id for li in plan.line_items}
    assert best.total_cost == pytest.approx(best.basket_cost + best.travel_cost, abs=0.01)
    assert plan.plan_trace[-1].step_id == "t5_trip_optimizer"


def test_a_product_no_store_in_range_sells_is_reported_with_its_nearest_offer():
    """Every seeded store sells every recipe product, so take garlic off the shelf of the one
    store within 1 km of downtown: garlic is then reported, and the basket, total and trip
    cover the rest."""
    from pantry_planner import flow

    before = flow.run("tomato_penne", **DOWNTOWN, max_km=1)
    [store] = {li.store_name for li in before.line_items}
    garlic = next(li for li in before.line_items if li.ingredient_name == "Garlic")
    shelf = _query("SELECT sp.* FROM store_products sp JOIN stores s ON s.id = sp.store_id "
                   "WHERE s.name = :store AND sp.product_id = :pid", store=store,
                   pid=garlic.product_id)
    assert len(shelf) == 1
    _execute("DELETE FROM store_products WHERE store_id = :store_id AND product_id = :product_id",
             **shelf[0])
    try:
        plan = flow.run("tomato_penne", **DOWNTOWN, max_km=1)
    finally:
        cols = ", ".join(shelf[0])
        _execute(f"INSERT INTO store_products ({cols}) VALUES "
                 f"({', '.join(':' + c for c in shelf[0])})", **shelf[0])
    [dropped] = plan.out_of_range
    assert dropped.ingredient == "Garlic"
    assert dropped.reason.startswith(f"{garlic.product_name} has no offer within 1 km of the "
                                     "shopping location; the nearest is ")
    assert "Garlic" not in {li.ingredient_name for li in plan.line_items}
    recipe_lines = {i.name for i in flow.db.load_recipe("tomato_penne").ingredients}
    assert _lines_covered(plan) == recipe_lines                 # nothing silently dropped
    assert plan.total_cost == round(sum(li.price for li in plan.line_items), 2)
    assert plan.total_cost == round(before.total_cost - garlic.price, 2)
    for option in plan.trip_options:
        assert garlic.product_id not in {i.product_id for i in option.items}


def test_no_store_in_range_at_all_is_a_gate_naming_the_nearest_offers():
    from pantry_planner import flow
    from pantry_planner.nlsearch.planner import PlanAborted

    whistler = {"lat": 50.1163, "lon": -122.9574}             # about 100 km from every store
    with pytest.raises(PlanAborted) as info:
        flow.run("tomato_penne", **whistler, max_km=5)
    alert = info.value.execution.aborted
    assert alert.code.value == "unavailable_within_constraints" and alert.partial_would_plan == 0
    assert "No store within 5 km of the shopping location sells any" in alert.message
    assert len(alert.details) == 5 and all("the nearest is" in d["reason"] for d in alert.details)

    from fastapi.testclient import TestClient

    from pantry_planner.api import app

    resp = TestClient(app).post("/plan/tomato_penne", params={**whistler, "max_km": 5})
    assert resp.status_code == 409
    assert resp.json()["detail"]["aborted"]["code"] == "unavailable_within_constraints"


def test_the_rest_endpoint_takes_a_location_and_validates_it():
    from fastapi.testclient import TestClient

    from pantry_planner.api import app

    client = TestClient(app)
    resp = client.post("/plan/tomato_penne", params={**DOWNTOWN, "max_km": 5})
    assert resp.status_code == 200, resp.text
    plan = resp.json()
    assert all(li["store_name"] for li in plan["line_items"]) and plan["trip_options"]
    for bad in ({"max_km": 0.1}, {"max_km": 500}, {"lat": 95}, {"lon": -200}):
        assert client.post("/plan/tomato_penne", params=bad).status_code == 422, bad


@pytest.mark.asyncio
async def test_the_mcp_tool_takes_a_location_and_reports_stores_and_the_trip():
    from pantry_planner.mcp_server import server

    res = await server.call_tool("plan_recipe", {"slug": "tomato_penne", **DOWNTOWN,
                                                 "max_km": 5,
                                                 "exclude_origin": ["United States"]})
    summary = res.structured_content["summary"]
    assert all(line["store"] for line in summary["lines"])
    assert summary["trip"]["stores"] and summary["trip"]["total_cost"] > 0
    assert summary["origin_status"] in ("verified", "unverified")
    tools = {t.name: t for t in await server.list_tools()}
    props = tools["plan_recipe"].input_schema["properties"]
    assert {"lat", "lon", "max_km"} <= set(props)
    with pytest.raises(ToolError):
        await server.call_tool("plan_recipe", {"slug": "tomato_penne", "max_km": 0.1})


@pytest.mark.asyncio
async def test_each_line_says_where_the_recommended_trip_buys_it():
    from pantry_planner.mcp_server import server

    res = await server.call_tool("plan_recipe", {"slug": "tomato_penne", **DOWNTOWN,
                                                 "max_km": 5})
    summary = res.structured_content["summary"]
    on_trip = {i["product_id"]: i for i in summary["trip"]["items"]}
    for line in summary["lines"]:
        stop = on_trip[line["product_id"]]
        assert (line["trip_store"], line["trip_price"]) == (stop["store"], stop["price"])
    # without a location there is no trip, and the lines say so
    res = await server.call_tool("plan_recipe", {"slug": "tomato_penne"})
    lines = res.structured_content["summary"]["lines"]
    assert all(line["trip_store"] == "" and line["trip_price"] is None for line in lines)


def test_a_plan_times_its_pipeline_steps_and_the_store_list_is_served():
    from fastapi.testclient import TestClient

    from pantry_planner import flow
    from pantry_planner.api import app

    plan = flow.run("tomato_penne", **DOWNTOWN, max_km=5)
    steps = [s["step"] for s in plan.pipeline]
    assert steps[0] == "load_recipe" and steps[-1] == "build_plan" and "optimize_trips" in steps
    assert all(s["ms"] is not None and s["ms"] >= 0 and s["error"] is None for s in plan.pipeline)
    stores = TestClient(app).get("/stores").json()
    assert {"Pantry Mart Downtown", "GreenLeaf Grocers Kitsilano"} <= {s["name"] for s in stores}
    assert all(-90 <= s["lat"] <= 90 for s in stores)


def test_every_plan_call_is_its_own_burr_run():
    from pantry_planner import flow, tracing

    first, second = flow.run("tomato_penne"), flow.run("tomato_penne")
    assert first.burr_run.startswith("run-tomato_penne-") and first.burr_run != second.burr_run
    runs = tracing.TRACKING_DB_DIR / "pantry-planner"
    assert (runs / first.burr_run).is_dir() and (runs / second.burr_run).is_dir()

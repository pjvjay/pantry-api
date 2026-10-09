"""An exclusion that empties an ingredient: what the shopper is offered instead.

Excluding the United States when the only yellow onion is American used to end the plan with
"Affected: Yellow Onion." and nothing else, although the planner had found red and green
onions still on the shelf. The trade now travels with the error (the removed products and the
alternatives left), and allow_partial plans the rest of the recipe with the emptied ingredient
on out_of_range, carrying the same options.
"""
from __future__ import annotations

import os

import pytest

US = ["United States"]
TEXT = "Garlic Pasta (serves 2)\n- 500g penne\n- 2 cloves garlic\n- 1 can crushed tomatoes\n"


@pytest.fixture(scope="module", autouse=True)
def seeded_db(tmp_path_factory):
    db_file = tmp_path_factory.mktemp("db") / "test.db"
    os.environ["DB_URL"] = os.environ.get("PANTRY_TEST_DB_URL") or f"sqlite:///{db_file}"
    os.environ["DEMO_MODE"] = "1"
    os.environ.pop("ANTHROPIC_API_KEY", None)
    from pantry_planner import config, db

    config.settings.cache_clear()
    db.seed_from_json()
    yield
    os.environ.pop("DEMO_MODE", None)
    config.settings.cache_clear()


@pytest.fixture(autouse=True)
def clean_evidence():
    from sqlalchemy.orm import Session

    from pantry_planner.db import ProductOriginEvidenceRow, engine

    with Session(engine()) as s:
        s.query(ProductOriginEvidenceRow).delete()
        s.commit()
    yield


def _mark(ids, country="United States"):
    from pantry_planner import db

    db.save_origin_evidence([
        dict(product_id=pid, source="label-photo", source_ref=f"{pid}.jpg", claim_type="made-in",
             verbatim=f"Made in {country}", ingredient_origin="", manufactured_in=country,
             confidence="high", importer_only=False, note="", observed_at="")
        for pid in ids])


def _ids_named(fragment):
    from pantry_planner import db

    return [p.id for p in db.load_all_products() if fragment in p.name.lower()]


def _planned(plan) -> int:
    return sum(1 + len(li.also_lines) for li in plan.line_items)


# ─── The gate names the trade ────────────────────────────────

def test_gate_lists_the_removed_products_and_the_country():
    from pantry_planner import flow
    from pantry_planner.nlsearch import PlanAborted

    _mark(_ids_named("garlic"))
    with pytest.raises(PlanAborted) as exc:
        flow.run("tomato_penne", exclude=US)
    alert = exc.value.execution.aborted
    assert alert.details[0]["name"] == "Garlic"
    assert "United States" in alert.details[0]["reason"]
    assert any("Fresh Garlic" in s for s in alert.details[0]["suggestions"])
    # four of tomato penne's five ingredients are still plannable
    assert alert.partial_would_plan == 4


@pytest.mark.asyncio
async def test_mcp_error_carries_the_options_and_the_retry():
    from mcp.server.mcpserver.exceptions import ToolError

    from pantry_planner.mcp_server import server

    _mark(_ids_named("garlic"))
    with pytest.raises(ToolError) as exc:
        await server.call_tool("plan_recipe", {"slug": "tomato_penne", "exclude_origin": US})
    message = str(exc.value)
    assert "Garlic (options: Fresh Garlic" in message
    assert "Retry with allow_partial=true to plan the rest." in message


@pytest.mark.asyncio
async def test_mcp_plan_from_text_error_carries_the_alternatives():
    from mcp.server.mcpserver.exceptions import ToolError

    from pantry_planner.mcp_server import server

    _mark(_ids_named("garlic"))
    with pytest.raises(ToolError) as exc:
        await server.call_tool("plan_from_text", {"recipe_text": TEXT, "exclude_origin": US})
    message = str(exc.value)
    assert "still available, not a direct match" in message
    assert "Retry with allow_partial=true" in message


# ─── allow_partial plans the rest ────────────────────────────

def test_classic_partial_plan_reports_the_emptied_ingredient():
    from pantry_planner import flow

    _mark(_ids_named("garlic"))
    plan = flow.run("tomato_penne", exclude=US, allow_partial=True)
    assert "Garlic" not in [li.ingredient_name for li in plan.line_items]
    assert [d.ingredient for d in plan.out_of_range] == ["Garlic"]
    left = plan.out_of_range[0]
    assert "United States" in left.reason
    assert any("Fresh Garlic" in s for s in left.suggestions)
    assert plan.ingredient_count == 5
    assert _planned(plan) + len(plan.out_of_range) == plan.ingredient_count


def test_classic_partial_plan_ships_nothing_excluded():
    from pantry_planner import flow

    garlic = set(_ids_named("garlic"))
    _mark(garlic)
    plan = flow.run("tomato_penne", exclude=US, allow_partial=True)
    assert not garlic & {li.product_id for li in plan.line_items}


def test_partial_plan_still_gates_when_nothing_would_remain():
    from pantry_planner import db, flow
    from pantry_planner.nlsearch import PlanAborted

    _mark([p.id for p in db.load_all_products()])
    with pytest.raises(PlanAborted) as exc:
        flow.run("grilled_cheese", exclude=US, allow_partial=True)
    assert exc.value.execution.aborted.code.value == "excluded_by_origin"
    assert exc.value.execution.aborted.partial_would_plan == 0


def test_nl_partial_plan_reports_the_emptied_ingredient():
    from pantry_planner import flow

    _mark(_ids_named("garlic"))
    plan = flow.run_nl(TEXT, exclude=US, allow_partial=True)
    names = [li.ingredient_name for li in plan.line_items]
    assert "garlic" not in names and names, names
    left = [d for d in plan.out_of_range if d.ingredient == "garlic"]
    assert left and any("not a direct match" in s for s in left[0].suggestions)
    assert _planned(plan) + len(plan.out_of_range) + len(plan.not_stocked) \
        + len(plan.skipped) == plan.ingredient_count


@pytest.mark.asyncio
async def test_mcp_plan_recipe_partial_summary_lists_the_options():
    from pantry_planner.mcp_server import server

    _mark(_ids_named("garlic"))
    result = await server.call_tool("plan_recipe", {
        "slug": "tomato_penne", "exclude_origin": US, "allow_partial": True})
    summary = result.structured_content["summary"]
    assert [d["ingredient"] for d in summary["out_of_range"]] == ["Garlic"]
    assert summary["out_of_range"][0]["suggestions"]
    assert any("planned 4 of 5 ingredients" in n for n in summary["notes"])


def test_rest_partial_plan_is_a_200():
    from fastapi.testclient import TestClient

    from pantry_planner import api

    _mark(_ids_named("garlic"))
    r = TestClient(api.app).post("/plan/tomato_penne",
                                 params={"exclude_origin": US, "allow_partial": True})
    assert r.status_code == 200, r.text[:300]
    assert r.json()["out_of_range"][0]["ingredient"] == "Garlic"

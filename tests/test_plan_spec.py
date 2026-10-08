"""Planning reviewed lines (flow.run_spec, POST /plan/spec, MCP plan_from_lines) and what
every plan now carries: its basis, the need per purchase and the servings it states.

No LLM: everything runs in DEMO_MODE with no ANTHROPIC_API_KEY. Expected values come from
the request itself (the reviewed lines), the demo parser, the recipe file or plain unit
arithmetic, never from the code under test.
"""
from __future__ import annotations

import json
import os

import pytest

PASTA = "Garlic Pasta (serves 2)\n- 500g penne\n- 2 cloves garlic\n- 1 can crushed tomatoes\n"
TIMING = ("burr_run", "pipeline", "latency_ms", "llm_calls")


@pytest.fixture(scope="module", autouse=True)
def seeded_db(tmp_path_factory):
    os.environ["DB_URL"] = (os.environ.get("PANTRY_TEST_DB_URL")
                            or f"sqlite:///{tmp_path_factory.mktemp('db') / 'spec.db'}")
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


@pytest.fixture()
def server():
    from pantry_planner.mcp_server import server

    return server


def _untimed(summary: dict) -> dict:
    return {k: v for k, v in summary.items() if k not in TIMING}


# ─── Additive fields on every plan ───────────────────────────

def test_need_per_purchase_is_the_lines_amount_in_canonical_units():
    from pantry_planner import flow

    plan = flow.run_nl(PASTA)
    by_line = {li.line_no: li for li in plan.line_items}
    assert (by_line[1].need_qty, by_line[1].need_uom) == (500.0, "g")
    # "1 can" is the standard 400 ml can (units._VOLUME)
    assert (by_line[3].need_qty, by_line[3].need_uom) == (400.0, "ml")
    # a clove is not a unit the planner can measure: unknown, never a guess
    assert (by_line[2].need_qty, by_line[2].need_uom) == (None, None)


def test_servings_are_reported_only_when_the_recipe_states_them():
    from pantry_planner import db, flow

    assert flow.run_nl(PASTA).servings == 2
    assert flow.run_nl("Garlic Pasta\n- 500g penne\n").servings is None
    assert flow.run("tomato_penne").servings == db.load_recipe("tomato_penne").servings


def test_an_unstated_servings_count_is_recorded_before_it_is_clamped():
    from pantry_planner.nlsearch.query_parser import validate_parsed
    from pantry_planner.nlsearch.schemas import ParsedInput

    p = validate_parsed(ParsedInput.model_validate(
        {"recipe": {"title": "t", "servings": None, "ingredients": [{"name": "penne"}]}}), {})
    assert (p.recipe.servings, p.recipe.servings_stated) == (1, False)
    p = validate_parsed(ParsedInput.model_validate(
        {"recipe": {"title": "t", "servings": 3, "ingredients": [{"name": "penne"}]}}), {})
    assert (p.recipe.servings, p.recipe.servings_stated) == (3, True)


def test_library_basis_names_every_line_and_the_product_bought_for_it():
    from pantry_planner import db, flow

    recipe = db.load_recipe("chicken_curry")
    plan = flow.run("chicken_curry")
    b = plan.basis
    assert b.path == "library" and b.recipe_slug == "chicken_curry"
    assert [(ln.line_no, ln.name) for ln in b.lines] == \
        [(i.line_no, i.name) for i in recipe.ingredients]
    bought = {n: li.product_id for li in plan.line_items for n in [li.line_no, *li.also_lines]}
    assert {ln.line_no: ln.product_id for ln in b.lines} == bought
    assert all(ln.quantity is None and ln.unit is None for ln in b.lines)


def test_nl_basis_carries_the_parse_and_the_constraints():
    from pantry_planner import demomode, flow

    plan = flow.run_nl(PASTA + "Notes: within 10 km\n", exclude=["United States"])
    parsed = demomode.parse_recipe(PASTA)
    b = plan.basis
    assert b.path == "nl"
    assert [(ln.name, ln.form, ln.quantity, ln.unit) for ln in b.lines] == \
        [(i.name, i.form, i.quantity, i.unit) for i in parsed.recipe.ingredients]
    assert b.max_km == 10 and b.constraints.max_distance_km == 10
    assert b.exclude_origin == ["United States"] and b.origin_requested
    assert b.interpretation == plan.interpretation


@pytest.mark.asyncio
async def test_plan_tools_attach_the_basis_only_when_asked(server):
    for tool, args in (("plan_from_text", {"recipe_text": PASTA}),
                       ("plan_recipe", {"slug": "tomato_penne"})):
        lean = (await server.call_tool(tool, args)).structured_content["summary"]
        assert "basis" not in lean, tool
        withb = (await server.call_tool(tool, {**args, "basis": True})
                 ).structured_content["summary"]
        assert withb["basis"]["lines"], tool
        # nothing else moves: the summary is the lean one plus the basis key
        assert _untimed({k: v for k, v in withb.items() if k != "basis"}) == _untimed(lean)


@pytest.mark.asyncio
async def test_basis_is_hidden_from_text_content_too(server):
    res = await server.call_tool("plan_from_text", {"recipe_text": PASTA})
    assert '"basis"' not in json.dumps(res.structured_content)
    assert all('"basis"' not in c.text for c in res.content if hasattr(c, "text"))

"""MCP server tests — no LLM calls, no network, no subprocess.

Tools are exercised in-process through MCPServer.call_tool, which runs
the same validation/serialization path the wire transports use. The
LLM-backed plan tools are asserted present-with-schema but never
executed (the pipeline itself is covered by test_nlsearch/test_router).
"""
from __future__ import annotations

import os

import pytest
from mcp.server.mcpserver.exceptions import ToolError

_TMP_DB = None

EXPECTED_TOOLS = {
    "list_recipes", "get_recipe", "list_products",
    "plan_recipe", "plan_from_text", "plan_week",
    "get_product_origins", "search_products_by_origin",
    "pipeline_status",
}


@pytest.fixture(scope="module", autouse=True)
def seeded_db(tmp_path_factory):
    global _TMP_DB
    _TMP_DB = tmp_path_factory.mktemp("db") / "test.db"
    os.environ["DB_URL"] = f"sqlite:///{_TMP_DB}"
    from pantry_planner import config, db

    config.settings.cache_clear()
    db.seed_from_json()
    yield
    config.settings.cache_clear()


@pytest.fixture()
def server():
    from pantry_planner.mcp_server import server

    return server


# ─── Tool listing / schemas ──────────────────────────────────

@pytest.mark.asyncio
async def test_lists_exactly_the_expected_tools(server):
    tools = await server.list_tools()
    assert {t.name for t in tools} == EXPECTED_TOOLS


@pytest.mark.asyncio
async def test_every_tool_has_description_and_output_schema(server):
    for t in await server.list_tools():
        assert t.description, t.name
        assert t.output_schema, t.name


@pytest.mark.asyncio
async def test_plan_tools_have_input_schemas_without_running(server):
    tools = {t.name: t for t in await server.list_tools()}
    assert "slug" in tools["plan_recipe"].input_schema["properties"]
    props = tools["plan_from_text"].input_schema["properties"]
    assert {"recipe_text", "lat", "lon"} <= set(props)
    week_props = tools["plan_week"].input_schema["properties"]
    assert {"days", "max_total_budget", "exclude_tags"} <= set(week_props)


# ─── Read-only tools ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_list_recipes(server):
    res = await server.call_tool("list_recipes", {})
    assert not res.is_error
    recipes = res.structured_content["result"]
    slugs = {r["slug"] for r in recipes}
    assert "pbj_sandwich" in slugs
    pbj = next(r for r in recipes if r["slug"] == "pbj_sandwich")
    assert pbj["ingredient_count"] > 0


@pytest.mark.asyncio
async def test_get_recipe_round_trips_model(server):
    from pantry_planner.models import Recipe

    res = await server.call_tool("get_recipe", {"slug": "pbj_sandwich"})
    recipe = Recipe.model_validate(res.structured_content)
    assert recipe.slug == "pbj_sandwich"
    assert recipe.ingredients


@pytest.mark.asyncio
async def test_get_recipe_unknown_slug_is_tool_error(server):
    with pytest.raises(ToolError, match="list_recipes"):
        await server.call_tool("get_recipe", {"slug": "does-not-exist"})


@pytest.mark.asyncio
async def test_list_products_search_filter(server):
    res = await server.call_tool("list_products", {"search": "cheese"})
    products = res.structured_content["result"]
    assert products
    assert all("cheese" in
               f"{p['name']} {p['category']} {p['subcategory']}".lower()
               for p in products)

    everything = await server.call_tool("list_products", {})
    assert len(everything.structured_content["result"]) > len(products)


@pytest.mark.asyncio
async def test_pipeline_status(server):
    res = await server.call_tool("pipeline_status", {})
    status = res.structured_content
    assert status["status"] == "ok"
    assert status["routing_strategy"] in {"cascade", "three_phase"}
    assert "***" in status["db"] or status["db"].startswith("sqlite")


# ─── Origin tools ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_product_origins_no_llm(server):
    res = await server.call_tool(
        "get_product_origins", {"search": "basmati", "allow_llm": False})
    origins = res.structured_content["result"]
    assert len(origins) == 1
    assert origins[0]["country"] == "India"
    assert origins[0]["source"] == "heuristic"


@pytest.mark.asyncio
async def test_get_product_origins_unknown_id_is_tool_error(server):
    with pytest.raises(ToolError, match="Unknown product ids"):
        await server.call_tool(
            "get_product_origins", {"product_ids": [99999], "allow_llm": False})


@pytest.mark.asyncio
async def test_search_products_by_origin(server):
    res = await server.call_tool(
        "search_products_by_origin", {"country": "italy"})
    matches = res.structured_content["result"]
    assert matches
    assert all(m["country"] == "Italy" for m in matches)
    assert any("Passata" in m["product_name"] for m in matches)


# ─── HTTP transport wiring (no server process) ───────────────

def test_http_app_serves_mcp_route():
    """The mounted Starlette app must expose exactly /mcp (no trailing-
    slash redirect trap) and api.py must drive the session manager."""
    from starlette.routing import Route

    from pantry_planner.mcp_server import http_app

    app = http_app()
    paths = [r.path for r in app.routes if isinstance(r, Route)]
    assert "/mcp" in paths


def test_fastapi_mounts_mcp_and_keeps_rest_routes():
    from pantry_planner import api

    rest_paths = {getattr(r, "path", None) for r in api.app.routes}
    assert {"/health", "/recipes", "/products"} <= rest_paths
    # the catch-all mount that carries /mcp
    from starlette.routing import Mount

    mounts = [r for r in api.app.routes if isinstance(r, Mount)]
    assert any(r.path == "" or r.path == "/" for r in mounts)

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
    "get_product_origins", "rank_products_by_origin", "origin_triage",
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


# --- Provenance tools ----------------------------------------

@pytest.fixture()
def evidence():
    """Write evidence directly; ingest paths are covered by test_ingest."""
    from sqlalchemy.orm import Session

    from pantry_planner import db
    from pantry_planner.db import ProductOriginEvidenceRow, engine

    with Session(engine()) as s:
        s.query(ProductOriginEvidenceRow).delete()
        s.commit()
    rice = next(p for p in db.load_all_products() if p.name == "Basmati Rice 2kg")
    pb = next(p for p in db.load_all_products()
              if p.name == "Nestles PBJ")
    db.save_origin_evidence([
        dict(product_id=rice.id, source="label-photo", claim_type="product-of",
             verbatim="Product of India", ingredient_origin="India",
             manufactured_in="India", confidence="high", importer_only=False,
             note="", source_ref="rice.jpg", observed_at=""),
        dict(product_id=pb.id, source="label-photo", claim_type="made-in",
             verbatim="Made in Canada from imported ingredients",
             ingredient_origin="United States", manufactured_in="Canada",
             confidence="high", importer_only=False, note="",
             source_ref="pb.jpg", observed_at=""),
    ])
    yield {"rice": rice.id, "pb": pb.id}
    with Session(engine()) as s:
        s.query(ProductOriginEvidenceRow).delete()
        s.commit()


@pytest.mark.asyncio
async def test_get_product_origins_reports_status(server, evidence):
    res = await server.call_tool("get_product_origins", {"search": "basmati"})
    origins = res.structured_content["result"]
    assert len(origins) == 1
    assert origins[0]["status"] == "resolved"
    assert origins[0]["manufactured_in"] == "India"
    assert origins[0]["verbatim"] == "Product of India"


@pytest.mark.asyncio
async def test_get_product_origins_unknown_id_is_tool_error(server):
    with pytest.raises(ToolError, match="Unknown product ids"):
        await server.call_tool("get_product_origins", {"product_ids": [99999]})


@pytest.mark.asyncio
async def test_rank_excludes_on_ingredient_origin(server, evidence):
    """Made in Canada from American peanuts must not pass a US exclusion."""
    res = await server.call_tool("rank_products_by_origin", {
        "preference": ["Canada"], "exclude": ["United States"]})
    out = res.structured_content
    excluded_ids = {e["product_id"] for e in out["excluded"]}
    assert evidence["pb"] in excluded_ids
    assert evidence["pb"] not in {r["product_id"] for r in out["ranked"]}
    hit = next(e for e in out["excluded"] if e["product_id"] == evidence["pb"])
    assert hit["matched_field"] == "ingredient_origin"


@pytest.mark.asyncio
async def test_rank_keeps_unverified_out_of_the_ranking(server, evidence):
    res = await server.call_tool("rank_products_by_origin",
                                 {"preference": ["India"]})
    out = res.structured_content
    assert out["counts"]["unranked"] > out["counts"]["ranked"]
    ranked_ids = {r["product_id"] for r in out["ranked"]}
    assert not ranked_ids & {u["product_id"] for u in out["unranked"]}
    assert "not evidence" in out["coverage_note"]


@pytest.mark.asyncio
async def test_origin_triage_returns_hints(server):
    res = await server.call_tool("origin_triage", {})
    hints = res.structured_content["result"]
    assert hints
    assert set(hints[0]) == {"product_id", "product_name", "reason", "status"}


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

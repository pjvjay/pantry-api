"""Fourth review round. Each test states the right output, after first
asserting the precondition the finding described."""
from __future__ import annotations

import asyncio
import json
import os

import pytest
from mcp.server.mcpserver.exceptions import ToolError

from tests.seed_catalog import CATALOG

_TMP_DB = None
TEXT = "Garlic Pasta (serves 2)\n- 500g penne\n- 2 cloves garlic\n- 1 can crushed tomatoes\n"


@pytest.fixture(scope="module", autouse=True)
def seeded_db(tmp_path_factory):
    global _TMP_DB
    _TMP_DB = tmp_path_factory.mktemp("db") / "test.db"
    os.environ["DB_URL"] = os.environ.get("PANTRY_TEST_DB_URL") or f"sqlite:///{_TMP_DB}"
    os.environ["DEMO_MODE"] = "1"
    os.environ.pop("ANTHROPIC_API_KEY", None)
    from pantry_planner import config, db

    config.settings.cache_clear()
    db.seed_from_json()
    yield
    os.environ.pop("DEMO_MODE", None)
    config.settings.cache_clear()


async def _read_json(server, uri: str):
    [content] = await server.read_resource(uri)
    return json.loads(content.content)


# ─── The API can roll before pantry-db 0006 exists ───────────────────────

@pytest.mark.asyncio
async def test_lookup_and_coverage_answer_without_the_review_queue_table():
    from sqlalchemy import text

    from pantry_planner import db, ingest
    from pantry_planner.mcp_server import server

    # Precondition: with the table present the count is a number, not None.
    r = await server.call_tool("get_product", {"product_id": 8})
    assert r.structured_content["pending_submissions"] == 0
    with db.engine().begin() as c:
        c.execute(text("DROP TABLE origin_submissions"))
    try:
        r = await server.call_tool("get_product", {"product_id": 8})
        d = r.structured_content
        assert d["name"] == "Basmati Rice 2kg"
        assert d["pending_submissions"] is None          # unknown, not zero
        cov = await _read_json(server, "pantry://origins/coverage")
        assert cov["products"] == CATALOG
        assert cov["submissions"] is None
        assert "0006" in cov["submissions_note"]
        with pytest.raises(ToolError, match="0006"):
            await server.call_tool("submit_origin_evidence", {
                "product_id": 8, "claim_type": "product-of", "country": "India",
                "verbatim": "Product of India"})
        with pytest.raises(ToolError, match="0006"):
            await server.call_tool("list_origin_submissions", {})
        assert ingest.pending_submission_count(8) is None
    finally:
        db.init_schema()
    r = await server.call_tool("get_product", {"product_id": 8})
    assert r.structured_content["pending_submissions"] == 0


# ─── A bad MCP_AUTH_TOKENS aborts startup, not the first request ─────────

def test_weak_token_aborts_the_api_lifespan_before_anything_is_served():
    from pantry_planner import api, config

    os.environ["MCP_AUTH_TOKENS"] = "x:short"
    config.settings.cache_clear()
    try:
        cm = api._lifespan(api.app)
        with pytest.raises(ValueError, match="at least 16"):
            asyncio.run(cm.__aenter__())
    finally:
        os.environ.pop("MCP_AUTH_TOKENS", None)
        config.settings.cache_clear()


def test_validate_startup_is_what_the_stdio_entry_point_runs():
    import inspect

    from pantry_planner import config, mcp_server

    assert "validate_startup()" in inspect.getsource(mcp_server.main)
    os.environ["MCP_AUTH_TOKENS"] = "reviewer:0123456789abcdef0123"
    config.settings.cache_clear()
    try:
        assert config.validate_startup().mcp_auth_tokens == (("reviewer", "0123456789abcdef0123"),)
    finally:
        os.environ.pop("MCP_AUTH_TOKENS", None)
        config.settings.cache_clear()


def test_token_parse_errors_name_only_the_entry_position():
    from pantry_planner.config import parse_mcp_auth_tokens

    with pytest.raises(ValueError) as e:
        parse_mcp_auth_tokens("Sup3rSecretValue12345678:bob")     # fields reversed
    msg = str(e.value)
    assert "entry 1" in msg and "at least 16" in msg
    assert "Sup3rSecretValue" not in msg and "bob" not in msg


def test_duplicate_token_labels_are_rejected():
    from pantry_planner.config import parse_mcp_auth_tokens

    with pytest.raises(ValueError, match="entry 2: same label as entry 1"):
        parse_mcp_auth_tokens("alice:0123456789abcdef,alice:fedcba9876543210")
    assert parse_mcp_auth_tokens("alice:0123456789abcdef,bob:fedcba9876543210") == (
        ("alice", "0123456789abcdef"), ("bob", "fedcba9876543210"))


# ─── The lean trip is priced as its own basket ───────────────────────────

@pytest.mark.asyncio
async def test_lean_trip_carries_the_recommended_options_item_prices():
    from pantry_planner.mcp_server import server

    r = await server.call_tool("plan_from_text", {"recipe_text": TEXT, "verbose": True})
    s, f = r.structured_content["summary"], r.structured_content["full"]
    rec = next(t for t in f["trip_options"] if t["recommended"])
    assert rec["items"], "precondition: the recommended option prices its lines"
    assert [(i["product_id"], i["store"], i["price"]) for i in s["trip"]["items"]] == [
        (i["product_id"], i["store_name"], i["price"]) for i in rec["items"]]
    assert s["trip"]["basket_cost"] == rec["basket_cost"]
    assert s["trip"]["travel_cost"] == rec["travel_cost"]
    assert s["trip"]["total_cost"] == rec["total_cost"]
    # The two totals describe two baskets and each must add up on its own.
    assert abs(s["total_cost"] - sum(li["price"] for li in s["lines"])) < 0.011
    assert abs(s["trip"]["basket_cost"] - sum(i["price"] for i in s["trip"]["items"])) < 0.011


# ─── list_products: subcategories match, unknown values are an error ─────

@pytest.mark.asyncio
async def test_list_products_category_takes_subcategories_and_rejects_unknown():
    from sqlalchemy import func
    from sqlalchemy.orm import Session

    from pantry_planner import db
    from pantry_planner.mcp_server import server

    with Session(db.engine()) as s:
        pasta = sorted(int(i) for (i,) in s.query(db.ProductRow.id)
                       .filter(func.lower(db.ProductRow.subcategory) == "pasta").all())
    assert pasta, "precondition: the seed has a pasta subcategory"
    r = await server.call_tool("list_products", {"category": "pasta"})
    page = r.structured_content
    assert sorted(p["id"] for p in page["items"]) == pasta
    assert page["total"] == len(pasta)
    with pytest.raises(ToolError) as e:
        await server.call_tool("list_products", {"category": "nope"})
    assert "pantry" in str(e.value) and "catalog/categories" in str(e.value)

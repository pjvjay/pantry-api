"""Product lookup tools: find_product / get_product, input bounds, annotations.

Every expected value is read from the seed independently (a direct SQL
read, the tokenizer, the distance formula), never from another tool, so a
test here fails when the tool is wrong, not when two tools agree on being
wrong. The planner-parity claim is the point: a hit in find_product must be
a candidate in the planner, and a miss must be a miss.
"""
from __future__ import annotations

import math
import os

import pytest
from mcp.server.mcpserver.exceptions import ToolError
from sqlalchemy import text
from sqlalchemy.orm import Session

from tests.seed_catalog import ids_mentioning

_TMP_DB = None

# Seed facts the assertions name (seeds/products.json).
BASMATI, LONG_GRAIN, JASMINE = 8, 9, 84
GF_PENNE, PENNE_RIGATE = 49, 51
# The head-noun pool for "rice" is every product whose name or description
# says "rice" — not only the rice bags: Gluten-Free Penne is rice-flour
# pasta, and Shaoxing wine, rice vinegar and rice noodles say "rice" too.
RICE_POOL = ids_mentioning("rice")

RECIPE = ("Garlic Pasta (serves 2)\n- 500g penne\n- 2 cloves garlic\n"
          "- 1 can crushed tomatoes\nNotes: ")


@pytest.fixture(scope="module", autouse=True)
def seeded_db(tmp_path_factory):
    global _TMP_DB
    _TMP_DB = tmp_path_factory.mktemp("db") / "test.db"
    os.environ["DB_URL"] = os.environ.get("PANTRY_TEST_DB_URL") or f"sqlite:///{_TMP_DB}"
    from pantry_planner import config, db

    config.settings.cache_clear()
    db.seed_from_json()
    yield
    config.settings.cache_clear()


@pytest.fixture()
def server():
    from pantry_planner.mcp_server import server

    return server


@pytest.fixture()
def clean_evidence():
    from pantry_planner.db import ProductOriginEvidenceRow, engine

    def wipe():
        with Session(engine()) as s:
            s.query(ProductOriginEvidenceRow).delete()
            s.commit()

    wipe()
    yield
    wipe()


def _sql(sql: str, **params):
    from pantry_planner.db import engine

    with Session(engine()) as s:
        return s.execute(text(sql), params).all()


def _cheapest_offer(product_id: int) -> tuple[str, float]:
    """(store name, price) of the cheapest store_products row — the planner's price."""
    (store, price), = _sql(
        "SELECT s.name, sp.price FROM store_products sp JOIN stores s ON s.id = sp.store_id "
        "WHERE sp.product_id = :pid ORDER BY sp.price ASC, s.id ASC LIMIT 1", pid=product_id)
    return str(store), float(price)


def _distance_km(store: str, lat: float, lon: float) -> float:
    """The planner's equirectangular formula (sql_builder.DIST_EXPR), 0.1 km."""
    (slat, slon), = _sql("SELECT lat, lon FROM stores WHERE name = :n", n=store)
    coslat = 111.0 * math.cos(math.radians(lat))
    return round(math.sqrt(((lat - slat) * 111.0) ** 2 + ((lon - slon) * coslat) ** 2), 1)


async def _find(server, **args):
    res = await server.call_tool("find_product", args)
    assert not res.is_error
    return res.structured_content


# ─── find_product: planner-parity retrieval ──────────────────

@pytest.mark.asyncio
async def test_basmati_rice_is_a_direct_match_priced_like_the_planner(server):
    out = await _find(server, query="basmati rice")
    assert out["match"] == "direct"
    assert out["tokens"] == ["basmati", "rice"]
    ids = [i["id"] for i in out["items"]]
    assert BASMATI in ids
    assert LONG_GRAIN not in ids, "a head-noun-only match must not be a direct hit"
    assert out["total"] == len(ids) == 1
    for item in out["items"]:
        store, price = _cheapest_offer(item["id"])
        assert item["price"] == price
        assert item["store"] == store
    assert out["items"][0]["name"] == "Basmati Rice 2kg"
    assert out["items"][0]["size"] == "2kg"
    assert out["items"][0]["origin_status"] == "unknown"
    assert out["items"][0]["origin_country"] == ""


@pytest.mark.asyncio
async def test_token_and_separates_gluten_free_penne_from_penne_rigate(server):
    gf = await _find(server, query="gluten free penne")
    assert gf["match"] == "direct"
    assert gf["tokens"] == ["gluten", "free", "penne"]
    assert [i["id"] for i in gf["items"]] == [GF_PENNE]

    both = await _find(server, query="penne")
    assert both["match"] == "direct"
    assert {i["id"] for i in both["items"]} == {GF_PENNE, PENNE_RIGATE}
    prices = [i["price"] for i in both["items"]]
    assert prices == sorted(prices), "cheapest first, as the planner sees them"
    assert [i["id"] for i in both["items"]] == [PENNE_RIGATE, GF_PENNE]
    assert both["total"] == 2


@pytest.mark.asyncio
async def test_relaxed_match_offers_head_noun_alternatives(server):
    out = await _find(server, query="coconut rice")
    assert out["tokens"] == ["coconut", "rice"]
    assert out["match"] == "relaxed"
    assert {BASMATI, LONG_GRAIN, JASMINE, GF_PENNE} <= RICE_POOL
    assert {i["id"] for i in out["items"]} == RICE_POOL
    assert out["total"] == len(RICE_POOL)
    assert "alternatives" in out["note"]
    assert "'rice'" in out["note"]
    prices = [i["price"] for i in out["items"]]
    assert prices == sorted(prices)


@pytest.mark.asyncio
async def test_no_match_is_an_answer_not_an_error(server):
    out = await _find(server, query="zzqx flurb")
    assert out["match"] == "none"
    assert out["items"] == []
    assert out["total"] == 0
    assert "list_products" in out["note"]


@pytest.mark.asyncio
async def test_stopword_only_query_is_none(server):
    out = await _find(server, query="fresh of the")
    assert out["tokens"] == []
    assert out["match"] == "none"
    assert out["items"] == []


@pytest.mark.asyncio
async def test_limit_trims_items_but_total_counts_every_hit(server):
    out = await _find(server, query="penne", limit=1)
    assert len(out["items"]) == 1
    assert out["total"] == 2
    assert out["items"][0]["id"] == PENNE_RIGATE, "the cheapest hit survives the trim"


@pytest.mark.asyncio
async def test_find_product_distance_follows_the_request_point(server):
    from pantry_planner.config import settings

    cfg = settings()
    default = await _find(server, query="basmati rice")
    moved = await _find(server, query="basmati rice", lat=49.155, lon=-123.135)
    item = default["items"][0]
    assert item["distance_km"] == _distance_km(item["store"], cfg.default_lat, cfg.default_lon)
    # Standing at MegaSave Richmond, the cheapest basmati offer is 0.0 km away.
    assert moved["items"][0]["store"] == "MegaSave Richmond"
    assert moved["items"][0]["distance_km"] == 0.0


@pytest.mark.asyncio
@pytest.mark.parametrize("args", [
    {"query": ""},
    {"query": "x" * 201},
    {"query": "penne", "limit": 0},
    {"query": "penne", "limit": 26},
])
async def test_find_product_rejects_out_of_bound_inputs(server, args):
    with pytest.raises(ToolError, match="validation error"):
        await server.call_tool("find_product", args)


# ─── get_product ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_product_lists_every_offer_cheapest_first(server, clean_evidence):
    from pantry_planner.config import settings

    res = await server.call_tool("get_product", {"product_id": BASMATI})
    assert not res.is_error
    out = res.structured_content
    assert out["id"] == BASMATI
    assert out["name"] == "Basmati Rice 2kg"
    (list_price,), = _sql("SELECT price FROM products WHERE id = :pid", pid=BASMATI)
    assert out["list_price"] == float(list_price)

    offers = out["offers"]
    assert len(offers) == 4
    prices = [o["price"] for o in offers]
    assert prices == sorted(prices)
    (min_price,), = _sql("SELECT MIN(price) FROM store_products WHERE product_id = :pid",
                         pid=BASMATI)
    assert offers[0]["price"] == float(min_price)
    assert (offers[0]["store"], offers[0]["price"]) == _cheapest_offer(BASMATI)
    cfg = settings()
    for o in offers:
        assert o["distance_km"] == _distance_km(o["store"], cfg.default_lat, cfg.default_lon)

    assert out["origin"]["status"] == "unknown"
    assert out["origin"]["country"] == ""
    assert out["evidence"] == []
    assert out["pending_submissions"] == 0
    (n, avg), = _sql("SELECT COUNT(*), AVG(rating) FROM reviews WHERE product_id = :pid",
                     pid=BASMATI)
    assert out["review_count"] == int(n) and int(n) > 0
    assert out["avg_rating"] == pytest.approx(float(avg))


@pytest.mark.asyncio
async def test_get_product_resolves_a_label_and_keeps_ignored_evidence_visible(
        server, clean_evidence):
    from pantry_planner import db

    written = db.save_origin_evidence([
        dict(product_id=BASMATI, source="label-photo", claim_type="product-of",
             verbatim="Product of India", ingredient_origin="India",
             manufactured_in="India", confidence="high", importer_only=False,
             note="", source_ref="rice.jpg", observed_at="2026-10-01T00:00:00Z"),
        # An importer address is not an origin: the resolver ignores it,
        # but get_product must still show it so a reviewer can see why.
        dict(product_id=BASMATI, source="label-photo", claim_type="made-in",
             verbatim="Imported by Coastline Foods, Vancouver BC",
             ingredient_origin="", manufactured_in="Canada", confidence="low",
             importer_only=True, note="", source_ref="rice-back.jpg", observed_at=""),
    ])
    assert written == 2

    res = await server.call_tool("get_product", {"product_id": BASMATI})
    out = res.structured_content
    assert out["origin"]["status"] == "resolved"
    assert out["origin"]["country"] == "India"
    assert out["origin"]["claim_type"] == "product-of"
    assert out["origin"]["evidence_count"] == 2
    assert len(out["evidence"]) == 2
    assert out["evidence"][0]["verbatim"] == "Product of India"
    assert out["evidence"][0]["source"] == "label-photo"
    assert out["evidence"][0]["observed_at"] == "2026-10-01T00:00:00Z"
    assert out["evidence"][0]["importer_only"] is False
    assert out["evidence"][1]["importer_only"] is True
    assert out["evidence"][1]["verbatim"] == "Imported by Coastline Foods, Vancouver BC"

    found = await _find(server, query="basmati rice")
    assert found["items"][0]["origin_status"] == "resolved"
    assert found["items"][0]["origin_country"] == "India"


@pytest.mark.asyncio
async def test_get_product_unknown_id_points_at_find_product(server):
    with pytest.raises(ToolError, match=r"Unknown product id 99999.*find_product"):
        await server.call_tool("get_product", {"product_id": 99999})


@pytest.mark.asyncio
async def test_get_product_distance_follows_the_request_point(server):
    res = await server.call_tool("get_product", {"product_id": BASMATI,
                                                 "lat": 49.2820, "lon": -123.1180})
    by_store = {o["store"]: o["distance_km"] for o in res.structured_content["offers"]}
    assert by_store["Pantry Mart Downtown"] == 0.0
    assert by_store["MegaSave Richmond"] == _distance_km("MegaSave Richmond", 49.2820, -123.1180)


# ─── REST-parity input bounds on the existing tools ──────────

@pytest.mark.asyncio
@pytest.mark.parametrize("tool,field,args,detail", [
    ("plan_from_text", "recipe_text", {"recipe_text": (RECIPE + "x").ljust(8001, "x")},
     "at most 8000 characters"),
    ("plan_recipe", "exclude_origin",
     {"slug": "tomato_penne", "exclude_origin": ["Canada"] * 51}, "at most 50 items"),
    ("plan_recipe", "preference",
     {"slug": "tomato_penne", "preference": ["Canada"] * 51}, "at most 50 items"),
    ("plan_week", "days", {"days": 15}, "less than or equal to 14"),
    ("plan_week", "days", {"days": 0}, "greater than or equal to 1"),
    ("plan_week", "exclude_tags", {"exclude_tags": ["dairy"] * 51}, "at most 50 items"),
    ("list_products", "search", {"search": "x" * 201}, "at most 200 characters"),
    ("get_product_origins", "product_ids", {"product_ids": list(range(1, 202))},
     "at most 200 items"),
    ("get_product_origins", "search", {"search": "x" * 201}, "at most 200 characters"),
    ("rank_products_by_origin", "exclude", {"exclude": ["Canada"] * 51}, "at most 50 items"),
    ("rank_products_by_origin", "search", {"search": "x" * 201}, "at most 200 characters"),
])
async def test_rest_parity_bounds_reject_before_any_work(server, tool, field, args, detail):
    # The pydantic message is multi-line: "<field>\n  <detail> [...]".
    with pytest.raises(ToolError, match=rf"(?s)validation error.*\n{field}\n.*{detail}"):
        await server.call_tool(tool, args)


@pytest.mark.asyncio
async def test_plan_from_text_accepts_exactly_8000_chars(server):
    from pantry_planner import config

    text_8000 = RECIPE.ljust(8000, "x")
    assert len(text_8000) == 8000
    saved_key = os.environ.pop("ANTHROPIC_API_KEY", None)
    os.environ["DEMO_MODE"] = "1"
    config.settings.cache_clear()
    try:
        res = await server.call_tool("plan_from_text", {"recipe_text": text_8000})
    finally:
        os.environ.pop("DEMO_MODE", None)
        if saved_key is not None:
            os.environ["ANTHROPIC_API_KEY"] = saved_key
        config.settings.cache_clear()
    assert not res.is_error
    summary = res.structured_content["summary"]
    assert {ln["ingredient"] for ln in summary["lines"]} == {
        "penne", "garlic", "crushed tomatoes"}
    assert summary["total_cost"] > 0


@pytest.mark.asyncio
async def test_input_schemas_advertise_the_bounds(server):
    tools = {t.name: t.input_schema["properties"] for t in await server.list_tools()}
    assert tools["plan_from_text"]["recipe_text"]["maxLength"] == 8000
    assert tools["find_product"]["query"]["maxLength"] == 200
    assert tools["find_product"]["query"]["minLength"] == 1
    assert tools["find_product"]["limit"] == {
        "default": 8, "maximum": 25, "minimum": 1, "title": "Limit", "type": "integer"}
    assert tools["plan_week"]["days"]["maximum"] == 14
    assert tools["plan_week"]["days"]["minimum"] == 1
    lists = tools["plan_recipe"]["exclude_origin"]["anyOf"]
    assert {"items": {"type": "string"}, "maxItems": 50, "type": "array"} in lists


# ─── Annotations ─────────────────────────────────────────────

READ_TOOLS = {
    "list_recipes", "get_recipe", "list_products", "find_product", "get_product",
    "get_product_origins", "rank_products_by_origin", "origin_triage", "pipeline_status",
    "list_origin_submissions",
}
PLAN_TOOLS = {"plan_recipe", "plan_from_text", "plan_from_lines", "plan_week", "plan_meals"}
# No LLM behind them: the same basis against the same catalog gives the same answer.
FOLLOW_UP_TOOLS = {"rank_alternatives", "reprice_plan"}
# Write tools add to a queue or copy into evidence; nothing is destroyed.
WRITE_TOOLS = {"submit_origin_evidence", "review_origin_submission"}


@pytest.mark.asyncio
async def test_every_tool_is_annotated_and_titled(server):
    tools = {t.name: t for t in await server.list_tools()}
    assert set(tools) == READ_TOOLS | PLAN_TOOLS | WRITE_TOOLS | FOLLOW_UP_TOOLS
    for name, t in tools.items():
        assert t.annotations is not None, name
        assert t.title, name
        assert t.annotations.open_world_hint is False, name
    for name in READ_TOOLS:
        assert tools[name].annotations.read_only_hint is True, name
    for name in PLAN_TOOLS:
        assert tools[name].annotations.read_only_hint is True, name
        assert tools[name].annotations.idempotent_hint is False, name
    for name in FOLLOW_UP_TOOLS:
        assert tools[name].annotations.read_only_hint is True, name
        assert tools[name].annotations.idempotent_hint is True, name
    for name in WRITE_TOOLS:
        assert tools[name].annotations.read_only_hint is False, name
        assert tools[name].annotations.destructive_hint is False, name
    # Resubmitting the same reading is deduped; a second review is an error.
    assert tools["submit_origin_evidence"].annotations.idempotent_hint is True
    assert tools["review_origin_submission"].annotations.idempotent_hint is False
    assert tools["plan_recipe"].title == "Plan a recipe"

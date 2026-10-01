"""Token-lean MCP results: PlanResult / WeekResult summaries, paged catalog tools.

Content is asserted before size. Every expected value comes from somewhere
other than the tool under test — the recipe file (db.load_recipe), the demo
parser, a direct SQL read, or the full plan the same tool attaches with
verbose=True — so a test here fails when the summary is wrong, not when the
summary agrees with itself. Sizes are compact-JSON chars of the structured
content (what the wire carries, ÷4 ≈ tokens).
"""
from __future__ import annotations

import json
import os

import pytest
from mcp.server.mcpserver.exceptions import ToolError
from sqlalchemy import text
from sqlalchemy.orm import Session

_TMP_DB = None
TEXT = "Garlic Pasta (serves 2)\n- 500g penne\n- 2 cloves garlic\n- 1 can crushed tomatoes\n"
GARLIC, BASMATI = 13, 8          # seeds/products.json: Fresh Garlic, Basmati Rice 2kg
CATALOG = 62


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


@pytest.fixture(autouse=True)
def clean_evidence():
    from pantry_planner.db import ProductOriginEvidenceRow, engine

    def wipe():
        with Session(engine()) as s:
            s.query(ProductOriginEvidenceRow).delete()
            s.commit()

    wipe()
    yield
    wipe()


@pytest.fixture()
def server():
    from pantry_planner.mcp_server import server

    return server


def _chars(res) -> int:
    return len(json.dumps(res.structured_content, separators=(",", ":")))


def _sql(sql: str, **params):
    from pantry_planner.db import engine

    with Session(engine()) as s:
        return s.execute(text(sql), params).all()


def _label(product_id: int, country: str) -> None:
    from pantry_planner import db

    db.save_origin_evidence([dict(
        product_id=product_id, source="label-photo", source_ref=f"{product_id}.jpg",
        claim_type="product-of", verbatim=f"Product of {country}",
        ingredient_origin=country, manufactured_in=country, confidence="high",
        importer_only=False, note="", observed_at="")])


# ─── plan_recipe ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_plan_recipe_lines_follow_the_recipe_and_the_catalog(server):
    from pantry_planner import db
    from pantry_planner.mcp_server import PlanResult

    out = PlanResult.model_validate(
        (await server.call_tool("plan_recipe", {"slug": "tomato_penne"})).structured_content)
    assert out.full is None, "full is attached only on verbose=True"
    s = out.summary
    assert s.recipe_slug == "tomato_penne" and s.recipe_name == "Tomato Penne"

    recipe = db.load_recipe("tomato_penne")
    assert [ln.ingredient for ln in s.lines] == [i.name for i in recipe.ingredients]
    assert [ln.line_no for ln in s.lines] == [1, 2, 3, 4, 5]
    for ln in s.lines:
        (name, brand, size), = _sql(
            "SELECT name, brand, unit_size FROM products WHERE id = :pid", pid=ln.product_id)
        assert (ln.product, ln.brand, ln.size) == (name, brand, size), ln
        # no evidence on a clean DB: the receipt exists and says "unknown"
        assert (ln.origin_status, ln.origin_country) == ("unknown", "")
    assert s.total_cost == round(sum(ln.price for ln in s.lines), 2)
    # nobody asked an origin question, so no coverage is stamped on it
    assert s.origin_status == "not_requested" and s.coverage is None
    assert s.notes == []


@pytest.mark.asyncio
async def test_plan_recipe_verbose_attaches_the_full_plan(server):
    from pantry_planner.mcp_server import PlanResult

    out = PlanResult.model_validate((await server.call_tool(
        "plan_recipe", {"slug": "tomato_penne", "verbose": True})).structured_content)
    assert out.full is not None
    assert out.full.recipe_slug == "tomato_penne"
    assert out.summary.total_cost == out.full.total_cost
    assert ([ln.product_id for ln in out.summary.lines]
            == [li.product_id for li in out.full.line_items])
    assert out.summary.llm_cost_usd == out.full.total_llm_cost_usd
    # the classic path builds no trip frontier, so the summary has no trip
    assert out.full.trip_options == [] and out.summary.trip is None


@pytest.mark.asyncio
async def test_plan_recipe_line_carries_the_resolved_origin(server):
    from pantry_planner.mcp_server import PlanResult

    _label(GARLIC, "Italy")
    out = PlanResult.model_validate(
        (await server.call_tool("plan_recipe", {"slug": "tomato_penne"})).structured_content)
    by_ingredient = {ln.ingredient: ln for ln in out.summary.lines}
    garlic = by_ingredient["Garlic"]
    assert garlic.product_id == GARLIC
    assert (garlic.origin_status, garlic.origin_country) == ("resolved", "Italy")
    assert all(ln.origin_status == "unknown"
               for ln in out.summary.lines if ln.ingredient != "Garlic")


@pytest.mark.asyncio
async def test_plan_recipe_exclude_on_a_clean_db_is_unverified_below_the_floor(server):
    from pantry_planner.mcp_server import PlanResult

    out = PlanResult.model_validate((await server.call_tool(
        "plan_recipe", {"slug": "tomato_penne", "exclude_origin": ["Canada"]})).structured_content)
    s = out.summary
    # nothing is evidenced, so nothing is excluded — every line still ships
    assert len(s.lines) == 5
    assert next(ln for ln in s.lines if ln.ingredient == "Garlic").product_id == GARLIC
    assert s.origin_status == "unverified"
    assert s.coverage is not None
    assert s.coverage.meets_floor is False
    assert (s.coverage.lines_total, s.coverage.lines_known,
            s.coverage.lines_excluded_origin) == (5, 0, 0)
    assert s.coverage.spend_fraction == 0.0 and s.coverage.count_fraction == 0.0
    floor_notes = [n for n in s.notes if n.startswith("coverage below floor")]
    assert len(floor_notes) == 1
    assert "0 of 5 lines" in floor_notes[0] and "floor 60%" in floor_notes[0]


# ─── plan_from_text ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_plan_from_text_summary_then_verbose_trace(server):
    from pantry_planner import demomode
    from pantry_planner.mcp_server import PlanResult

    lean = await server.call_tool("plan_from_text", {"recipe_text": TEXT})
    out = PlanResult.model_validate(lean.structured_content)
    assert out.full is None
    expected = [i.name for i in demomode.parse_recipe(TEXT).recipe.ingredients]
    assert expected == ["penne", "garlic", "crushed tomatoes"]
    assert [ln.ingredient for ln in out.summary.lines] == expected
    garlic = next(ln for ln in out.summary.lines if ln.ingredient == "garlic")
    assert garlic.product == "Fresh Garlic"
    assert out.summary.notes[0] == "Garlic Pasta · serves 2 · 3 ingredients"

    full = await server.call_tool("plan_from_text", {"recipe_text": TEXT, "verbose": True})
    vout = PlanResult.model_validate(full.structured_content)
    assert vout.full is not None
    assert vout.full.plan_trace, "the NL path carries its retrieval trace in full"
    recommended = next(t for t in vout.full.trip_options if t.recommended)
    assert vout.summary.trip is not None
    assert (vout.summary.trip.stores, vout.summary.trip.total_cost) == (
        recommended.stores, recommended.total_cost)
    assert vout.summary.total_cost == vout.full.total_cost

    # size last: the flag has to be real in both directions
    assert _chars(lean) <= 2000, _chars(lean)
    assert _chars(full) >= 8000, _chars(full)


# ─── plan_week ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_plan_week_summary_merges_the_list_once(server):
    from pantry_planner.mcp_server import WeekResult

    lean = await server.call_tool("plan_week", {"days": 3})
    out = WeekResult.model_validate(lean.structured_content)
    assert out.full is None
    s = out.summary
    assert len(s.days) == 3
    ids = [w.product_id for w in s.shopping_list]
    assert len(set(ids)) == len(ids), "a shared product is listed once"
    assert s.overlap_savings == round(s.standalone_cost - s.total_cost, 2)
    assert s.standalone_cost == round(sum(d.day_cost for d in s.days), 2)
    # used_by is exactly the set of dinners whose lines chose that product
    day_products = {w.product_id: {d.recipe_name for d in s.days
                                   if any(ln.product_id == w.product_id for ln in d.lines)}
                    for w in s.shopping_list}
    for w in s.shopping_list:
        assert set(w.used_by) == day_products[w.product_id], w
    assert set(ids) == {ln.product_id for d in s.days for ln in d.lines}
    assert s.origin_status == "not_requested" and s.coverage is None

    full = await server.call_tool("plan_week", {"days": 3, "verbose": True})
    vout = WeekResult.model_validate(full.structured_content)
    assert vout.full is not None
    assert vout.summary.total_cost == vout.full.total_cost
    # the planner's own notes (skipped recipes etc.) survive into the summary
    assert vout.full.notes and all(n in vout.summary.notes for n in vout.full.notes)
    recommended = next(t for t in vout.full.trip_options if t.recommended)
    assert vout.summary.trip is not None and vout.summary.trip.stores == recommended.stores

    # Measured 6178 lean / 24878 verbose on the demo seed (compact JSON).
    assert _chars(lean) <= 6500, _chars(lean)
    assert _chars(full) >= 3 * _chars(lean), (_chars(full), _chars(lean))


# ─── list_products paging ────────────────────────────────────

@pytest.mark.asyncio
async def test_list_products_pages_by_id(server):
    from pantry_planner.mcp_server import ProductPage

    ordered = [int(r[0]) for r in _sql("SELECT id FROM products ORDER BY id")]
    assert len(ordered) == CATALOG

    page = ProductPage.model_validate(
        (await server.call_tool("list_products", {"limit": 10})).structured_content)
    assert [p.id for p in page.items] == ordered[:10]
    assert (page.total, page.next_offset) == (CATALOG, 10)

    tail = ProductPage.model_validate(
        (await server.call_tool("list_products", {"offset": 60})).structured_content)
    assert [p.id for p in tail.items] == ordered[60:]
    assert len(tail.items) == 2 and tail.next_offset is None

    (n_dairy,), = _sql("SELECT COUNT(*) FROM products WHERE category = 'dairy'")
    dairy = ProductPage.model_validate(
        (await server.call_tool("list_products", {"category": "Dairy"})).structured_content)
    assert int(n_dairy) == 12
    assert dairy.total == len(dairy.items) == 12 and dairy.next_offset is None
    assert {p.category for p in dairy.items} == {"dairy"}

    default = await server.call_tool("list_products", {})
    assert len(default.structured_content["items"]) == 50
    # Measured 7892 on the demo seed: 50 ProductSummary rows. (A 2600-char
    # page is not reachable at limit=50 — even id/name/price alone is 2680.)
    assert _chars(default) <= 8000, _chars(default)


# ─── get_product_origins paging and counts ───────────────────

@pytest.mark.asyncio
async def test_get_product_origins_counts_before_the_status_filter(server):
    from pantry_planner.mcp_server import OriginPage

    ordered = [int(r[0]) for r in _sql("SELECT id FROM products ORDER BY id")]
    clean = OriginPage.model_validate(
        (await server.call_tool("get_product_origins", {})).structured_content)
    assert clean.by_status == {"unknown": CATALOG}
    assert (clean.total, len(clean.items), clean.next_offset) == (CATALOG, 50, 50)
    assert [o.product_id for o in clean.items] == ordered[:50]

    _label(BASMATI, "India")
    (n_evidenced,), = _sql("SELECT COUNT(DISTINCT product_id) FROM product_origin_evidence")
    after = OriginPage.model_validate(
        (await server.call_tool("get_product_origins", {})).structured_content)
    assert after.by_status == {"unknown": CATALOG - int(n_evidenced), "resolved": 1}
    assert after.total == CATALOG

    resolved = OriginPage.model_validate((await server.call_tool(
        "get_product_origins", {"status": "resolved"})).structured_content)
    assert [o.product_id for o in resolved.items] == [BASMATI]
    assert resolved.items[0].country == "India"
    assert (resolved.total, resolved.next_offset) == (1, None)
    # by_status still describes the whole catalog, not the filtered page
    assert resolved.by_status == after.by_status

    narrowed = OriginPage.model_validate((await server.call_tool(
        "get_product_origins", {"search": "basmati"})).structured_content)
    assert narrowed.by_status == {"resolved": 1} and narrowed.total == 1


@pytest.mark.asyncio
async def test_get_product_origins_empty_and_invalid(server):
    none = await server.call_tool("get_product_origins", {"search": "zzz-no-such"})
    assert none.structured_content == {"items": [], "total": 0, "by_status": {},
                                       "next_offset": None}
    with pytest.raises(ToolError, match="Unknown origin status.*resolved"):
        await server.call_tool("get_product_origins", {"status": "verified"})

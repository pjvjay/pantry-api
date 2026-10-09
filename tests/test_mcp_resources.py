"""MCP resources and prompts, plus the annotations audit.

A resource is a thin wrapper over code the tools already use, so every
expected value here is read from the seed independently (direct SQL, the
origins tables, settings) rather than from another tool. The countries
resource exists so an agent can spell a country the server accepts, so its
right-answer test is the round trip: every name it publishes must pass
validate_countries.
"""
from __future__ import annotations

import json
import os

import pytest
from mcp.server.mcpserver.exceptions import ResourceError
from sqlalchemy import text
from sqlalchemy.orm import Session

from tests.seed_catalog import CATALOG, categories, count

_TMP_DB = None

RESOURCE_URIS = {
    "pantry://recipes", "pantry://catalog/categories",
    "pantry://countries", "pantry://origins/coverage",
}
RECIPE_TEMPLATE = "pantry://recipes/{slug}"
PROMPTS = {"plan_dinner", "read_label", "review_submissions"}
GARLIC, SPAGHETTI = 13, 21          # seeds/products.json


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
def clean_origin_tables():
    """No evidence and an empty review queue before AND after the test."""
    from pantry_planner.db import OriginSubmissionRow, ProductOriginEvidenceRow, engine

    def wipe():
        with Session(engine()) as s:
            s.query(OriginSubmissionRow).delete()
            s.query(ProductOriginEvidenceRow).delete()
            s.commit()

    wipe()
    yield
    wipe()


def _sql(query: str, **params):
    from pantry_planner.db import engine

    with Session(engine()) as s:
        return s.execute(text(query), params).all()


async def _read_json(server, uri: str):
    [content] = await server.read_resource(uri)
    assert content.mime_type == "application/json", uri
    return json.loads(content.content)


# ─── Listing ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_lists_the_resources_and_the_recipe_template(server):
    resources = await server.list_resources()
    assert {str(r.uri) for r in resources} == RESOURCE_URIS
    assert all(r.title and r.description for r in resources)
    templates = await server.list_resource_templates()
    assert [t.uri_template for t in templates] == [RECIPE_TEMPLATE]


# ─── Recipes ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_recipes_resource_is_the_seeded_library(server):
    data = await _read_json(server, "pantry://recipes")
    seeded = {slug for (slug,) in _sql("SELECT slug FROM recipes")}
    assert len(seeded) == 7
    assert {r["slug"] for r in data} == seeded
    (n,) = _sql("SELECT COUNT(*) FROM recipe_ingredients WHERE recipe_slug = 'tomato_penne'")[0]
    by_slug = {r["slug"]: r for r in data}
    assert by_slug["tomato_penne"]["ingredient_count"] == n == 5
    assert by_slug["tomato_penne"]["name"] == "Tomato Penne"


@pytest.mark.asyncio
async def test_recipe_template_returns_the_recipe_with_ordered_ingredients(server):
    data = await _read_json(server, "pantry://recipes/tomato_penne")
    names = [name for (name,) in _sql(
        "SELECT name FROM recipe_ingredients WHERE recipe_slug = 'tomato_penne' "
        "ORDER BY line_no")]
    assert data["slug"] == "tomato_penne"
    assert [i["name"] for i in data["ingredients"]] == names
    assert names[0] == "Penne"


@pytest.mark.asyncio
async def test_unknown_recipe_slug_keeps_the_list_recipes_hint(server):
    # A plain ValueError would be masked by the SDK as "Error creating
    # resource from template"; ResourceNotFoundError carries the hint.
    with pytest.raises(ResourceError) as ei:
        await server.read_resource("pantry://recipes/no_such_recipe")
    assert "no_such_recipe" in str(ei.value)
    assert "list_recipes" in str(ei.value)


# ─── Categories ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_categories_resource_matches_a_direct_group_by(server):
    data = await _read_json(server, "pantry://catalog/categories")
    expected: dict[str, dict[str, int]] = {}
    for cat, sub, n in _sql(
            "SELECT category, subcategory, COUNT(*) FROM products "
            "GROUP BY category, subcategory"):
        expected.setdefault(cat or "", {})[sub or ""] = n
    assert data == expected
    # ...and the database holds what the seed file says, aisle by aisle
    assert data == categories()
    assert data["pantry"]["pasta"] == count("pantry", "pasta") > 0
    assert data["produce"]["vegetables"] == count("produce", "vegetables") > 0
    assert sum(n for subs in data.values() for n in subs.values()) == CATALOG


# ─── Countries ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_countries_resource_spells_what_the_server_accepts(server):
    from pantry_planner.origins import validate_countries

    data = await _read_json(server, "pantry://countries")
    assert set(data) == {"canonical", "aliases", "ambiguous"}
    canonical = data["canonical"]
    assert "United States" in canonical
    assert "usa" in data["aliases"]["United States"]
    assert "u.s.a" in data["aliases"]["United States"]
    # Synonyms fold to one canonical entry; the list is sorted and unique.
    assert "Czechia" in canonical and "Czech Republic" not in canonical
    assert "czech republic" in data["aliases"]["Czechia"]     # forms are lowercase
    assert canonical == sorted(canonical) and len(canonical) == len(set(canonical))
    # Ambiguous inputs are listed with the real choices, not accepted.
    assert data["ambiguous"]["congo"] == [
        "Republic of the Congo", "Democratic Republic of the Congo"]
    assert validate_countries(["congo"]) == {"congo": data["ambiguous"]["congo"]}
    # The point of the resource: every published spelling passes validation.
    every_form = canonical + [f for forms in data["aliases"].values() for f in forms]
    assert len(every_form) > 300
    assert validate_countries(every_form) == {}


def test_ambiguous_is_public_and_the_old_name_is_an_alias():
    from pantry_planner import origins

    assert origins.AMBIGUOUS is origins._AMBIGUOUS
    assert "korea" in origins.AMBIGUOUS


# ─── Origin coverage ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_origin_coverage_reflects_evidence_and_the_review_queue(
        server, clean_origin_tables):
    from pantry_planner import config, db, ingest

    data = await _read_json(server, "pantry://origins/coverage")
    assert data == {
        "products": CATALOG, "by_status": {"unknown": CATALOG}, "evidence_rows": 0,
        "submissions": {"pending": 0, "approved": 0, "rejected": 0},
        "submissions_note": "",
        "floor": config.settings().origin_min_coverage,
    }
    assert 0 < data["floor"] <= 1

    written = db.save_origin_evidence([dict(
        product_id=GARLIC, source="label-photo", claim_type="grown-in",
        verbatim="Grown in U.S.A.", ingredient_origin="United States",
        manufactured_in="United States", confidence="high", importer_only=False,
        note="", source_ref="garlic.jpg", observed_at="2026-10-01T00:00:00Z")])
    assert written == 1
    sub = ingest.submit_origin({
        "product_id": SPAGHETTI, "claim_type": "made-in", "country": "Italy",
        "verbatim": "Made in Italy", "confidence": "high"}, submitted_by="test")
    assert sub["status"] == "pending" and sub["duplicate"] is False

    data = await _read_json(server, "pantry://origins/coverage")
    assert data["products"] == CATALOG
    assert data["by_status"] == {"resolved": 1, "unknown": CATALOG - 1}
    assert data["evidence_rows"] == 1
    assert data["submissions"] == {"pending": 1, "approved": 0, "rejected": 0}
    (pending,) = _sql("SELECT COUNT(*) FROM origin_submissions WHERE status = 'pending'")[0]
    assert pending == 1


# ─── Prompts ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_lists_the_three_prompts_with_arguments(server):
    prompts = {p.name: p for p in await server.list_prompts()}
    assert set(prompts) == PROMPTS
    assert all(p.title and p.description for p in prompts.values())
    args = {a.name: a.required for a in prompts["plan_dinner"].arguments}
    assert args == {"recipe": True, "exclude_origin": False, "budget": False}
    assert [a.name for a in prompts["read_label"].arguments] == ["product"]
    assert not prompts["review_submissions"].arguments


async def _prompt_text(server, name: str, args: dict) -> str:
    result = await server.get_prompt(name, args)
    assert len(result.messages) == 1 and result.messages[0].role == "user"
    return result.messages[0].content.text


@pytest.mark.asyncio
async def test_read_label_prompt_is_the_vision_protocol(server):
    body = await _prompt_text(server, "read_label", {"product": "garlic"})
    assert "garlic" in body
    for must in ("find_product", "verbatim", "EXACTLY", "importer_only",
                 "Imported by", "submit_origin_evidence", "PENDING",
                 "product-of", "made-in", "high =", "medium =", "low ="):
        assert must in body, must


@pytest.mark.asyncio
async def test_plan_dinner_prompt_reports_coverage_verbatim(server):
    body = await _prompt_text(server, "plan_dinner", {"recipe": "tomato_penne"})
    for must in ("tomato_penne", "list_recipes", "find_product", "plan_recipe",
                 "origin_status", "coverage.spend_fraction", "VERBATIM",
                 "meets_floor", "summary.notes"):
        assert must in body, must
    assert "Exclude products" not in body and "budget" not in body.lower()

    body = await _prompt_text(server, "plan_dinner", {
        "recipe": "tomato_penne", "exclude_origin": "USA", "budget": "$20"})
    assert "coming from: USA" in body and "pantry://countries" in body
    assert "budget is $20" in body


@pytest.mark.asyncio
async def test_review_submissions_prompt_names_the_three_tools(server):
    body = await _prompt_text(server, "review_submissions", {})
    for must in ("list_origin_submissions", "get_product", "review_origin_submission",
                 "approve", "reject", "note", "importer_only", "agent-label"):
        assert must in body, must


# ─── Annotations audit ───────────────────────────────────────

@pytest.mark.asyncio
async def test_every_tool_has_a_title_and_annotations(server):
    tools = await server.list_tools()
    assert len(tools) == 19
    for t in tools:
        assert t.title, t.name
        assert t.annotations is not None, t.name
        assert t.annotations.open_world_hint is False, t.name
    by_name = {t.name: t.annotations for t in tools}
    writes = {"submit_origin_evidence", "review_origin_submission"}
    for name, a in by_name.items():
        if name in writes:
            assert a.read_only_hint is False and a.destructive_hint is False, name
        else:
            assert a.read_only_hint is True, name
    assert by_name["submit_origin_evidence"].idempotent_hint is True
    assert by_name["review_origin_submission"].idempotent_hint is False
    for name in ("plan_recipe", "plan_from_text", "plan_from_lines", "plan_week", "plan_meals"):
        assert by_name[name].idempotent_hint is False, name
    for name in ("rank_alternatives", "reprice_plan"):
        assert by_name[name].idempotent_hint is True, name

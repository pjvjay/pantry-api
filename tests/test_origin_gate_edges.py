"""Edges an independent review found open after the first fix round.

Each test states the RIGHT output. Where a tie or a special case is needed
it is constructed explicitly, never assumed.
"""
from __future__ import annotations

import os

import pytest

_TMP_DB = None
US = ["United States"]
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


@pytest.fixture(autouse=True)
def clean_evidence():
    from sqlalchemy.orm import Session

    from pantry_planner.db import ProductOriginEvidenceRow, engine

    with Session(engine()) as s:
        s.query(ProductOriginEvidenceRow).delete()
        s.commit()
    yield


def _mark(ids, country="United States", claim="made-in", source="label-photo"):
    from pantry_planner import db

    db.save_origin_evidence([
        dict(product_id=pid, source=source, source_ref=f"{pid}.jpg", claim_type=claim,
             verbatim=f"Made in {country}", ingredient_origin="", manufactured_in=country,
             confidence="high", importer_only=False, note="", observed_at="")
        for pid in ids])


def _ids_named(fragment):
    from pantry_planner import db

    return [p.id for p in db.load_all_products() if fragment in p.name.lower()]


# ─── NL path: t4 substitutes are not candidates for the ingredient ────

def test_nl_path_gates_when_only_substitutes_remain():
    """Garlic's NL pool is [Fresh Garlic, onion(sub), ginger(sub), carrots(sub)].
    Excluding the one garlic must gate — not ship an onion, and certainly not
    the Penne Rigate the old code mapped it to."""
    from pantry_planner import flow
    from pantry_planner.nlsearch import PlanAborted

    baseline = {li.ingredient_name: li.product_name for li in flow.run_nl(TEXT).line_items}
    assert baseline["garlic"] == "Fresh Garlic", "precondition: unfiltered NL maps garlic correctly"

    _mark(_ids_named("garlic"))
    with pytest.raises(PlanAborted) as exc:
        flow.run_nl(TEXT, exclude=US)
    alert = exc.value.execution.aborted
    assert alert.code.value == "excluded_by_origin"
    assert [d["name"] for d in alert.details] == ["garlic"]
    assert any("garlic" in s.lower() for s in alert.details[0]["suggestions"])


def test_nl_path_gate_is_a_409_over_rest_and_a_tool_error_over_mcp():
    from fastapi.testclient import TestClient

    from pantry_planner import api

    _mark(_ids_named("garlic"))
    r = TestClient(api.app).post("/plan/nl", json={"recipe_text": TEXT, "exclude_origin": US})
    assert r.status_code == 409, r.text[:200]
    assert r.json()["detail"]["aborted"]["code"] == "excluded_by_origin"


@pytest.mark.asyncio
async def test_mcp_plan_from_text_reports_the_gate_not_a_200():
    from mcp.server.mcpserver.exceptions import ToolError

    from pantry_planner.mcp_server import server

    _mark(_ids_named("garlic"))
    with pytest.raises(ToolError, match="excluded_by_origin"):
        await server.call_tool("plan_from_text", {"recipe_text": TEXT, "exclude_origin": US})


# ─── Classic REST route: the gate is a 409, not a 500 ─────────────────

def test_classic_rest_gate_is_409_with_the_alert_payload():
    from fastapi.testclient import TestClient

    from pantry_planner import api

    _mark(_ids_named("garlic"))
    r = TestClient(api.app, raise_server_exceptions=False).post(
        "/plan/tomato_penne", params={"exclude_origin": US})
    assert r.status_code == 409, f"got {r.status_code}: {r.text[:200]}"
    assert r.json()["detail"]["aborted"]["details"][0]["name"] == "Garlic"


@pytest.mark.asyncio
async def test_mcp_plan_recipe_reports_the_gate_deliberately():
    from mcp.server.mcpserver.exceptions import ToolError

    from pantry_planner.mcp_server import server

    _mark(_ids_named("garlic"))
    with pytest.raises(ToolError, match="excluded_by_origin.*Garlic"):
        await server.call_tool("plan_recipe", {"slug": "tomato_penne", "exclude_origin": US})


# ─── Classic gate must not be stricter than the selector ─────────────

def test_gate_counts_every_candidate_not_just_the_cheapest_eight():
    """Nine term-indexed garlic twins; the eight cheapest excluded; the ninth
    must plan. The old gate judged a cheapest-8 shortlist and 409'd."""
    from sqlalchemy.orm import Session

    from pantry_planner import flow
    from pantry_planner.db import ProductRow, ProductTermRow, StoreProductRow, StoreRow, engine

    with Session(engine()) as s:
        store_id = s.query(StoreRow.id).first()[0]
        ids = []
        for i in range(9):
            pid = 9000 + i
            s.add(ProductRow(id=pid, name=f"Garlic Bulb Pack {i}", description="fresh garlic",
                             price=1.00 + i * 0.10, category="produce", subcategory="vegetables"))
            s.add(StoreProductRow(store_id=store_id, product_id=pid, price=1.00 + i * 0.10))
            for term in ("garlic", "bulb", "pack", "fresh"):
                s.add(ProductTermRow(term=term, product_id=pid))
            ids.append(pid)
        s.commit()
    try:
        _mark(_ids_named("garlic")[:0] + [i for i in ids[:8]] + _ids_named("fresh garlic"))
        plan = flow.run("tomato_penne", exclude=US)
        garlic_line = next(li for li in plan.line_items if li.ingredient_name == "Garlic")
        assert garlic_line.product_id == ids[8], "the ninth, non-excluded garlic must be chosen"
    finally:
        with Session(engine()) as s:
            s.query(ProductTermRow).filter(ProductTermRow.product_id.in_(ids)).delete()
            s.query(StoreProductRow).filter(StoreProductRow.product_id.in_(ids)).delete()
            s.query(ProductRow).filter(ProductRow.id.in_(ids)).delete()
            s.commit()


# ─── Week plan: preference reaches the selector and never costs money ──

def test_week_preference_reaches_the_per_day_selector(monkeypatch):
    from pantry_planner import weekplan

    seen = {}
    real = weekplan.call_selector

    def spy(*a, **kw):
        seen.update(kw)
        return real(*a, **kw)

    monkeypatch.setattr(weekplan, "call_selector", spy)
    weekplan.plan_week(days=1, preference=["Canada"])
    assert seen.get("preference") == ["Canada"]
    assert seen.get("constraints") == {"origin_preference": ["Canada"]}
    assert "origins_by_id" in seen


def _floor_label(plan):
    return next((s.label for s in plan.plan_trace
                 if "cheapest-basket floor" in (s.label or "")), None)


def test_week_preference_does_not_inflate_the_budget_floor():
    """The regression: a dearer preferred product at pool[0] was read as the
    'cheapest' and fired budget_infeasible on a feasible budget. The invariant
    is about the FLOOR and the GATE — not the chosen total, which a
    preference may legitimately raise by picking the preferred product."""
    from pantry_planner import db, weekplan
    from pantry_planner.nlsearch import PlanAborted

    beef = _ids_named("ground beef")
    prices = {p.id: p.price for p in db.load_all_products()}
    _mark([max(beef, key=prices.get)], country="Canada", claim="product-of")

    plain = weekplan.plan_week(days=7)
    floor = _floor_label(plain)
    assert floor, "precondition: the week plan reports its cheapest-basket floor"
    budget = plain.total_cost + 0.50                    # feasible by construction

    try:
        pref = weekplan.plan_week(days=7, max_total_budget=budget, preference=["Canada"])
    except PlanAborted as e:
        pytest.fail(f"soft preference fired a gate: {e.execution.aborted.code.value}")
    assert _floor_label(pref) == floor, "preference must not change the cheapest-basket floor"


# ─── Country names: validation and matching must agree ───────────────

@pytest.mark.parametrize("evidence,user", [
    ("Made in Czechia", "Czech Republic"), ("Produced in Türkiye", "Turkey"),
    ("Product of Burma", "Myanmar"), ("Ivory Coast", "Côte d’Ivoire"),
    ("Viet Nam", "Vietnam"), ("Russian Federation", "Russia"),
    ("UAE", "United Arab Emirates"), ("Bosnia", "Bosnia and Herzegovina"),
    ("Cape Verde", "Cabo Verde"), ("Swaziland", "Eswatini"),
])
def test_synonyms_accepted_by_validation_also_match_evidence(evidence, user):
    from pantry_planner.origins import country_matches, validate_countries

    assert validate_countries([user]) == {}
    assert country_matches(evidence, user), f"{user!r} validated but cannot match {evidence!r}"


def test_legitimate_spellings_are_not_rejected():
    from pantry_planner.origins import validate_countries

    ok = ["The Netherlands", "The UK", "The United States", "Viet Nam", "DR Congo",
          "Russian Federation", "UAE", "Bosnia", "Côte d’Ivoire", "Slovak Republic",
          "Holy See", "Czech Republic", "Türkiye", "Hong Kong", "Taiwan", "Republic of Korea",
          "Korea, South", "U.S.A.", "United States of America", "Holland", "New Zealand",
          "Papua New Guinea", "Trinidad and Tobago", "North Macedonia", "Timor-Leste",
          "Palestine", "Kosovo", "Greenland", "Puerto Rico", "Scotland", "Wales",
          "Northern Ireland", "México"]
    assert validate_countries(ok) == {}


def test_ambiguous_names_get_real_choices_not_difflib_noise():
    from pantry_planner.origins import validate_countries

    assert validate_countries(["Korea"])["Korea"] == ["South Korea", "North Korea"]
    assert validate_countries(["Ontario"])["Ontario"] == ["Canada"]

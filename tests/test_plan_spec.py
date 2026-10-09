"""Planning reviewed lines (flow.run_spec, POST /plan/spec, MCP plan_from_lines) and what
every plan now carries: its basis, the need per purchase and the servings it states.

No LLM: everything runs in DEMO_MODE with no ANTHROPIC_API_KEY. Expected values come from
the request itself (the reviewed lines), the demo parser, the recipe file or plain unit
arithmetic, never from the code under test.
"""
from __future__ import annotations

import json
import math
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


def test_basis_carries_the_servings_and_which_picks_the_selector_called_substitutions(
        monkeypatch):
    """A re-price has no selector reasoning to read: the basis keeps the servings the plan
    reported and whether each pick was called a substitution, so neither is lost later."""
    from pantry_planner import demomode, flow

    plan = flow.run_nl(PASTA)
    assert plan.basis.servings == plan.servings == 2
    assert not any(ln.substitution for ln in plan.basis.lines)
    assert flow.run("tomato_penne").basis.servings == flow.run("tomato_penne").servings

    real = demomode.select_products

    def flagging(ingredients, products, **kw):
        res = real(ingredients, products, **kw)
        for sel in res.selections:
            if sel.line_no == 2:
                sel.reasoning = "Substitution: no fresh garlic in range"
        return res

    monkeypatch.setattr(demomode, "select_products", flagging)
    b = flow.run_nl(PASTA).basis
    assert [ln.line_no for ln in b.lines if ln.substitution] == [2]


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


# ─── Reviewed lines: run_spec, POST /plan/spec, plan_from_lines ──

def _doc(lines, **kw) -> dict:
    return {"key": "imp:1", "title": "Reviewed", "source": {"kind": "pasted", "method": "paste"},
            "lines": [{"line_no": i, "text": ln.get("text", ln["name"]),
                       "amount_basis": "parsed_from_your_paste", **ln}
                      for i, ln in enumerate(lines, start=1)], **kw}


# Every line stocked, so every line is planned; amounts in units the parser would never
# write that way ("0.5 kg", "3 tbsp", none at all, an empty unit) to show none is re-read.
FIXTURES = [
    _doc([{"name": "penne", "quantity": 500, "unit": "g"},
          {"name": "garlic", "quantity": 3, "unit": "cloves", "note": "minced"},
          {"name": "crushed tomatoes", "quantity": 1, "unit": "can"},
          {"name": "olive oil", "quantity": 3, "unit": "tbsp"}], servings=2),
    _doc([{"name": "chicken thighs", "quantity": 0.5, "unit": "kg"},
          {"name": "basmati rice", "quantity": 1.5, "unit": "cups"},
          {"name": "yellow onion", "quantity": 1, "unit": ""},
          {"name": "garam masala", "quantity": None, "unit": "", "note": "to taste"},
          {"name": "ginger", "text": "a thumb of ginger"}]),
    _doc([{"name": "ground beef", "quantity": 1, "unit": "lb"},
          {"name": "broccoli", "quantity": 2, "unit": "heads"},
          {"name": "spaghetti", "quantity": 0.75, "unit": "box"}], servings=4,
         servings_basis="your_setting"),
]


def _client():
    from fastapi.testclient import TestClient

    from pantry_planner.api import app

    return TestClient(app)


def _three(lines) -> str:
    return json.dumps([{k: ln[k] for k in ("name", "quantity", "unit")} for ln in lines],
                      separators=(",", ":"))


def _reviewed(doc: dict) -> list[dict]:
    """The doc's lines as the shopper's reviewed RecipeDoc serialises them."""
    from pantry_planner.models import RecipeDoc

    return RecipeDoc.model_validate(doc).model_dump(mode="json")["lines"]


@pytest.mark.parametrize("doc", FIXTURES, ids=["pasta", "curry", "beef"])
def test_planned_basis_lines_are_the_reviewed_lines_byte_for_byte(doc):
    reviewed = _reviewed(doc)
    resp = _client().post("/plan/spec", json={"doc": doc})
    assert resp.status_code == 200, resp.text
    plan = resp.json()
    assert plan["basis"]["path"] == "spec"
    assert _three(plan["basis"]["lines"]) == _three(reviewed)
    assert not plan["not_stocked"] and not plan["out_of_range"] and not plan["skipped"]
    assert plan["servings"] == doc.get("servings")


def test_parse_lines_output_plans_as_it_stands():
    """A client builds the doc straight from POST /recipes/parse-lines, warnings and all, and
    posts it: stray bullets, a catering yield, a sentence on one line and a run of lines
    with no amount must not make that a 422 the shopper cannot fix. The planned basis is
    the parsed lines, byte for byte."""
    c = _client()
    parsed = c.post("/recipes/parse-lines", json={
        "yield_text": "Makes 150 servings",
        "lines": ["-", "500g penne", "•", "- 2 cloves garlic, minced",
                  "1 can crushed tomatoes", "stir " * 60] + ["salt"] * 25}).json()
    doc = {"key": "imp:1", "title": "Pasted", "servings": parsed["servings"],
           "servings_stated": parsed["servings_stated"],
           "source": {"kind": "pasted", "method": "paste"}, "lines": parsed["lines"],
           "warnings": parsed["warnings"]}
    resp = c.post("/plan/spec", json={"doc": doc, "allow_partial": True})
    assert resp.status_code == 200, resp.text
    plan = resp.json()
    assert _three(plan["basis"]["lines"]) == _three(parsed["lines"])
    assert plan["servings"] is None


def test_a_reviewed_count_has_a_need_when_its_unit_is_each():
    """RecipeLine's contract: a count is unit "each", which is what parse-lines writes. A
    quantity with unit "" is planned as written, so its need is unknown rather than guessed."""
    c = _client()
    (parsed,) = c.post("/recipes/parse-lines", json={"lines": ["2 yellow onions"]}
                       ).json()["lines"]
    assert (parsed["quantity"], parsed["unit"]) == (2.0, "each")
    for unit, need in (("each", (2.0, "each")), ("", (None, None))):
        doc = _doc([{"name": "yellow onion", "quantity": 2, "unit": unit}])
        resp = c.post("/plan/spec", json={"doc": doc})
        assert resp.status_code == 200, resp.text
        (li,) = resp.json()["line_items"]
        assert (li["need_qty"], li["need_uom"]) == need, unit


def test_planning_reviewed_lines_never_calls_a_parser(monkeypatch):
    """Both parsers are patched to raise; planning must still succeed, in demo mode and on
    the live path (fake providers: the only LLM call is the product selection)."""
    from pantry_planner import config, demomode
    from pantry_planner.nlsearch import planner, query_parser
    from tests.llm_fakes import FakeProviders

    def boom(*_a, **_k):
        raise AssertionError("a reviewed recipe was parsed again")

    monkeypatch.setattr(demomode, "parse_recipe", boom)
    monkeypatch.setattr(planner, "parse_input", boom)
    monkeypatch.setattr(query_parser, "parse_input", boom)
    doc = FIXTURES[0]
    resp = _client().post("/plan/spec", json={"doc": doc})
    assert resp.status_code == 200, resp.text
    assert resp.json()["line_items"][0]["model_used"] == "demo-deterministic"

    fakes = FakeProviders().install(monkeypatch)
    config.set_runtime_overrides(demo_mode=False)
    try:
        resp = _client().post("/plan/spec", json={"doc": doc})
    finally:
        config.clear_runtime_overrides()
    assert resp.status_code == 200, resp.text
    tools = [c["tool_choice"]["name"] for c in fakes.anthropic]
    assert tools and set(tools) == {"submit_plan"}, tools
    assert _three(resp.json()["basis"]["lines"]) == _three(_reviewed(doc))


def test_the_trace_says_the_parse_was_skipped():
    from pantry_planner import flow
    from pantry_planner.models import RecipeDoc
    from pantry_planner.recipe_doc import to_recipe_text, to_spec

    doc = RecipeDoc.model_validate(FIXTURES[0])
    plan = flow.run_spec(to_spec(doc), display_text=to_recipe_text(doc))
    first = plan.plan_trace[0]
    assert (first.step_id, first.outcome) == ("parse_input", "skipped")
    assert first.label.startswith("skipped: reviewed lines")
    assert [s.step_id for s in plan.plan_trace[1:4]] == \
        ["t1_existence", "t2_options", "t3_statistics"]
    assert plan.interpretation[0] == "Reviewed · serves 2 · 4 ingredients"


def test_an_unconfirmed_line_is_a_422_naming_it():
    lines = [dict(ln) for ln in FIXTURES[0]["lines"]]
    lines[2]["confirmed"] = False
    resp = _client().post("/plan/spec", json={"doc": {**FIXTURES[0], "lines": lines}})
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["error"] == "unconfirmed_lines" and detail["line_nos"] == [3]


def test_spec_errors_match_plan_nl():
    c = _client()
    resp = c.post("/plan/spec", json={"doc": FIXTURES[0], "exclude_origin": ["Amerca"]})
    assert resp.status_code == 422 and "unknown" in resp.json()["detail"]
    resp = c.post("/plan/spec", json={"doc": _doc([])})
    assert resp.status_code == 422 and resp.json()["detail"]["error"] == "no_lines"
    resp = c.post("/plan/spec", json={"doc": _doc([{"name": "saffron", "quantity": 1,
                                                     "unit": "g"}])})
    assert resp.status_code == 409
    assert resp.json()["detail"]["aborted"]["code"] == "missing_ingredients"
    resp = c.post("/plan/spec", json={"doc": _doc([{"name": "penne"}] * 61)})
    assert resp.status_code == 422


def test_a_422_renders_whatever_the_rejected_input():
    """Python's JSON parser reads 1e309 as inf and accepts NaN, neither of which JSON can
    carry. FastAPI's 422 echoes the rejected input back, so the error itself used to fail
    to render and the client got a 500 for its own mistake."""
    body = json.dumps({"doc": FIXTURES[0], "lat": "LAT"})
    for raw, shown in (("NaN", "nan"), ("1e309", "inf"), ("-1e309", "-inf")):
        resp = _client().post("/plan/spec", content=body.replace('"LAT"', raw),
                              headers={"content-type": "application/json"})
        assert resp.status_code == 422, raw
        (err,) = resp.json()["detail"]
        assert (err["loc"], err["input"]) == (["body", "lat"], shown), raw


def test_a_quantity_that_is_not_a_plannable_number_is_a_422():
    """JSON's 1e309 reads as inf in Python, and NaN parses too. Two such lines bought as one
    product used to overflow pack_count into a 500; one alone planned with a null need and a
    null basis quantity, which is not the reviewed line. A reviewed quantity is a finite
    number from 0 to MAX_LINE_QUANTITY, and the bound's own sum still plans."""
    from pantry_planner.models import MAX_LINE_QUANTITY

    body = json.dumps({"doc": _doc([{"name": "penne", "quantity": "Q", "unit": "g"},
                                    {"name": "penne rigate", "quantity": "Q", "unit": "g"}])})
    bad = {("body", "doc", "lines", n, "quantity") for n in (0, 1)}
    for raw in ("1e309", "-1e309", "NaN", "1e308", str(MAX_LINE_QUANTITY + 1)):
        resp = _client().post("/plan/spec", content=body.replace('"Q"', raw),
                              headers={"content-type": "application/json"})
        assert resp.status_code == 422, raw
        assert {tuple(e["loc"]) for e in resp.json()["detail"]} == bad, raw
    resp = _client().post("/plan/spec", content=body.replace('"Q"', str(MAX_LINE_QUANTITY)),
                          headers={"content-type": "application/json"})
    assert resp.status_code == 200, resp.text
    assert [ln["quantity"] for ln in resp.json()["basis"]["lines"]] == [MAX_LINE_QUANTITY] * 2


def test_lines_validate_parsed_drops_are_named_not_lost():
    """45 reviewed lines: the planner's 40-line cap drops 5, which appear by name on
    `skipped` and in `ignored`; water is never bought and is named too."""
    from pantry_planner.nlsearch.planner import NOT_BOUGHT, OVER_CAP

    lines = ([{"name": "water", "quantity": 1, "unit": "cup"}]
             + [{"name": "penne"}] * 39
             + [{"name": f"extra {n}"} for n in range(5)])
    resp = _client().post("/plan/spec", json={"doc": _doc(lines)})
    assert resp.status_code == 200, resp.text
    plan = resp.json()
    skipped = {(d["ingredient"], d["reason"]) for d in plan["skipped"]}
    assert ("water", NOT_BOUGHT) in skipped
    assert {(f"extra {n}", OVER_CAP) for n in range(5)} <= skipped
    assert "ignored: 5 ingredients over the 40-ingredient cap" in plan["interpretation"]
    assert plan["ingredient_count"] == 45
    covered = sum(1 + len(li["also_lines"]) for li in plan["line_items"])
    assert covered + len(plan["skipped"]) == 45


@pytest.mark.asyncio
async def test_plan_from_lines(server):
    lines = [{k: ln[k] for k in ln if k not in ("line_no", "amount_basis")}
             for ln in FIXTURES[0]["lines"]]
    res = await server.call_tool("plan_from_lines", {
        "doc_key": "imp:1", "lines": lines, "title": "Garlic Pasta", "servings": 2,
        "basis": True})
    assert not res.is_error, res.content
    s = res.structured_content["summary"]
    assert s["recipe_name"] == "Garlic Pasta" and s["basis"]["path"] == "spec"
    assert _three(s["basis"]["lines"]) == _three(_reviewed(FIXTURES[0]))
    lean = (await server.call_tool("plan_from_lines", {"doc_key": "imp:1", "lines": lines})
            ).structured_content["summary"]
    assert "basis" not in lean


@pytest.mark.asyncio
async def test_plan_from_lines_refusals(server):
    from mcp.server.mcpserver.exceptions import ToolError

    with pytest.raises(ToolError, match="No lines for 'imp:9'"):
        await server.call_tool("plan_from_lines", {"doc_key": "imp:9"})
    with pytest.raises(ToolError, match="not confirmed yet: 1"):
        await server.call_tool("plan_from_lines", {
            "doc_key": "imp:1", "lines": [{"name": "penne", "confirmed": False}]})
    with pytest.raises(ToolError):
        await server.call_tool("plan_from_lines", {
            "doc_key": "imp:1", "lines": [{"name": "penne"}] * 61})
    # an amount past RecipeLine's bound is named as the argument it is, not a masked
    # "Error executing tool" from inside
    for q in (math.inf, math.nan, 1e308):
        with pytest.raises(ToolError, match=r"lines\.0\.quantity"):
            await server.call_tool("plan_from_lines", {"doc_key": "imp:1", "lines": [
                {"name": "penne", "quantity": q, "unit": "g"},
                {"name": "penne rigate", "quantity": q, "unit": "g"}]})

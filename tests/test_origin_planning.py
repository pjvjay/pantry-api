"""Origin as a planning constraint — asserting the RIGHT answer.

The first version of this file asserted the absence of the wrong thing
("no American lines") and passed while the planner mapped Garlic to Ginger.
Zero excluded lines is trivially satisfiable by substituting garbage. Every
test here states what the correct output IS: a line comes from its own
ingredient's candidates, or the plan gates and names that ingredient.

DEMO_MODE throughout: no API key, no cost, deterministic selection.
"""
from __future__ import annotations

import os

import pytest

_TMP_DB = None


@pytest.fixture(scope="module", autouse=True)
def seeded_db(tmp_path_factory):
    global _TMP_DB
    _TMP_DB = tmp_path_factory.mktemp("db") / "test.db"
    os.environ["DB_URL"] = os.environ.get("PANTRY_TEST_DB_URL") or f"sqlite:///{_TMP_DB}"
    os.environ["DEMO_MODE"] = "1"
    os.environ.pop("ANTHROPIC_API_KEY", None)   # any live call must fail loudly
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


def _mark(product_ids, country="United States", claim="made-in", source="label-photo"):
    from pantry_planner import db

    db.save_origin_evidence([
        dict(product_id=pid, source=source, source_ref=f"{pid}.jpg",
             claim_type=claim, verbatim=f"Made in {country}",
             ingredient_origin="", manufactured_in=country,
             confidence="high", importer_only=False, note="", observed_at="")
        for pid in product_ids
    ])


def _ids_named(fragment):
    from pantry_planner import db

    return [p.id for p in db.load_all_products() if fragment in p.name.lower()]


def _pools(recipe):
    """Per-ingredient candidate ids — the ground truth a line must respect."""
    from pantry_planner import flow
    from pantry_planner.config import settings

    cfg = settings()
    raw = flow._ingredient_pools(recipe, cfg.default_lat, cfg.default_lon)
    # Direct candidates only: relaxed (head-noun) matches are alternatives the
    # gate may offer, never candidates — see test_origin_round3.py.
    return {recipe.ingredients[k].name: {p.id for p in v["direct"]} for k, v in raw.items()}


# ─── The blocking defect: a line must come from ITS OWN candidates ────

def _emptied_by(pools, excluded_ids):
    """Ingredients whose every candidate is in excluded_ids."""
    return {name for name, ids in pools.items() if ids and ids <= excluded_ids}


@pytest.mark.parametrize("stride", [2, 3, 5])
def test_every_plan_line_comes_from_its_own_ingredient_pool(stride):
    """Mark part of the catalog American. The right answer is one of two
    things, and the test checks WHICH applies rather than accepting either:
    if any ingredient lost all its candidates the plan must gate naming
    exactly those ingredients; otherwise every line must come from its own
    ingredient's candidates, never a neighbour's."""
    from pantry_planner import db, flow
    from pantry_planner.nlsearch import PlanAborted

    marked = [p.id for p in db.load_all_products()][::stride]
    _mark(marked)
    recipe = db.load_recipe("tomato_penne")
    pools = _pools(recipe)
    expected_gate = _emptied_by(pools, set(marked))

    try:
        plan = flow.run("tomato_penne", exclude=["United States"])
    except PlanAborted as e:
        named = {d["name"] for d in e.execution.aborted.details}
        assert named == expected_gate, (
            f"gate named {named}, but the exclusion emptied {expected_gate}")
        return
    assert not expected_gate, f"should have gated on {expected_gate}"
    for li in plan.line_items:
        pool = pools.get(li.ingredient_name)
        if pool:
            assert li.product_id in pool, (
                f"{li.ingredient_name!r} -> {li.product_name!r} is not one of "
                f"its own candidates")


def test_emptied_ingredient_gates_and_names_the_trade():
    """When every garlic is excluded, the answer is a gate that says
    'Garlic' and lists the garlic it removed — not onion in garlic's slot."""
    from pantry_planner import flow
    from pantry_planner.nlsearch import PlanAborted

    garlic = _ids_named("garlic")
    assert garlic, "precondition: catalog has garlic"
    _mark(garlic)

    with pytest.raises(PlanAborted) as exc:
        flow.run("tomato_penne", exclude=["United States"])
    alert = exc.value.execution.aborted
    assert alert.code.value == "excluded_by_origin"
    names = [d["name"] for d in alert.details]
    assert names == ["Garlic"], f"must name exactly the emptied ingredient, got {names}"
    suggestions = alert.details[0]["suggestions"]
    assert suggestions and any("garlic" in s.lower() for s in suggestions), (
        "the trade must list what was removed")


def test_gate_fires_per_recipe_only_where_an_ingredient_is_emptied():
    """grilled_cheese has no garlic; excluding all garlic must not gate it."""
    from pantry_planner import db, flow

    _mark(_ids_named("garlic"))
    recipe = db.load_recipe("grilled_cheese")
    assert not any("garlic" in i.name.lower() for i in recipe.ingredients)
    plan = flow.run("grilled_cheese", exclude=["United States"])
    pools = _pools(recipe)
    for li in plan.line_items:
        if pools.get(li.ingredient_name):
            assert li.product_id in pools[li.ingredient_name]


def test_excluded_origin_never_ships_in_a_week_plan():
    """Guarded against vacuity: the unfiltered run must SHOW American lines
    shipping first, or the filtered assertion proves nothing."""
    from pantry_planner import db, weekplan
    from pantry_planner.nlsearch import PlanAborted

    _mark([p.id for p in db.load_all_products()][:20])
    baseline = weekplan.plan_week(days=3)
    shipped = [w for w in baseline.shopping_list
               if w.origin and w.origin.manufactured_in == "United States"]
    assert shipped, "precondition: unfiltered week plan must ship American lines"

    try:
        wp = weekplan.plan_week(days=3, exclude_origin=["United States"])
    except PlanAborted as e:
        assert e.execution.aborted.code.value == "excluded_by_origin"
        return
    assert not [w for w in wp.shopping_list
                if w.origin and w.origin.manufactured_in == "United States"]


def test_week_gate_names_real_ingredients_with_suggestions():
    from pantry_planner import db, weekplan
    from pantry_planner.nlsearch import PlanAborted

    library_ingredients = {i.name for r in db.load_all_recipes() for i in r.ingredients}
    _mark([p.id for p in db.load_all_products()])      # everything is American

    with pytest.raises(PlanAborted) as exc:
        weekplan.plan_week(days=3, exclude_origin=["United States"])
    alert = exc.value.execution.aborted
    assert alert.code.value == "excluded_by_origin"
    assert alert.details, "must name what was emptied"
    for d in alert.details:
        assert d["name"] in library_ingredients, f"{d['name']!r} is not an ingredient"
        assert d["suggestions"], "each emptied ingredient must list the trade"


# ─── Conflicting evidence is a reason to drop, not a loophole ────────

def test_conflicting_evidence_naming_an_excluded_country_is_dropped():
    from pantry_planner import db
    from pantry_planner.origins import filter_pool, resolve_all

    pid = _ids_named("penne")[0]
    _mark([pid], country="United States", source="label-photo")
    _mark([pid], country="Canada", source="open-food-facts")
    origins = resolve_all([pid])
    assert origins[pid].status == "conflicting"

    product = next(p for p in db.load_all_products() if p.id == pid)
    kept, dropped = filter_pool([product], exclude=["United States"], origins=origins)
    assert not kept
    assert dropped[0][2] == "conflicting_evidence"


def test_conflicting_product_does_not_ship_end_to_end():
    from pantry_planner import flow
    from pantry_planner.nlsearch import PlanAborted

    penne = _ids_named("penne")
    _mark(penne, country="United States")
    _mark(penne, country="Canada", source="open-food-facts")
    try:
        plan = flow.run("tomato_penne", exclude=["United States"])
    except PlanAborted:
        return                                  # gated: nothing conflicting shipped
    assert not {li.product_id for li in plan.line_items} & set(penne)


# ─── Preference is consumed, and only as a tie-break ─────────────────

def _twins():
    from pantry_planner.models import Product, RecipeIngredient

    cheap = Product(id=901, name="Garlic Bulb", description="fresh garlic", price=0.79)
    dear = Product(id=902, name="Garlic Bulb", description="fresh garlic", price=1.49)
    ing = [RecipeIngredient(line_no=1, name="Garlic")]
    return cheap, dear, ing


def test_preference_breaks_a_tie_toward_the_preferred_origin():
    from pantry_planner import demomode
    from pantry_planner.models import ProductOrigin

    cheap, dear, ing = _twins()
    canadian = ProductOrigin(product_id=902, status="resolved", country="Canada",
                             manufactured_in="Canada", claim_type="product-of",
                             manufactured_claim="product-of")
    without = demomode.select_products(ing, [cheap, dear], model="demo")
    assert without.selections[0].product_id == 901, "no preference: cheapest wins"

    with_pref = demomode.select_products(
        ing, [cheap, dear], model="demo", origins_by_id={902: canadian},
        preference=["Canada"])
    assert with_pref.selections[0].product_id == 902, "preference breaks the tie"


def test_preference_never_overrides_semantic_match():
    from pantry_planner import demomode
    from pantry_planner.models import Product, ProductOrigin, RecipeIngredient

    garlic = Product(id=911, name="Garlic Bulb", description="fresh garlic", price=0.79)
    onion = Product(id=912, name="Yellow Onion", description="onion", price=0.50)
    canadian_onion = ProductOrigin(product_id=912, status="resolved", country="Canada",
                                   manufactured_in="Canada", claim_type="product-of",
                                   manufactured_claim="product-of")
    r = demomode.select_products(
        [RecipeIngredient(line_no=1, name="Garlic")], [garlic, onion], model="demo",
        origins_by_id={912: canadian_onion}, preference=["Canada"])
    assert r.selections[0].product_id == 911, "a Canadian onion is still not garlic"


def _equal_match_pair():
    """Find an ingredient with two candidates that tie on semantic match.

    Preference is only a tie-break (rule 7), so the test needs a real tie.
    Searching the seeded recipes for one is honest; inventing equality is not.
    """
    from pantry_planner import db
    from pantry_planner.nlsearch.units import tokens

    products = {p.id: p for p in db.load_all_products()}
    for recipe in db.load_all_recipes():
        for ing_name, ids in _pools(recipe).items():
            toks = set(tokens(ing_name))
            by_overlap: dict[int, list] = {}
            for pid in ids:
                p = products[pid]
                ov = len(toks & set(tokens(f"{p.name} {p.description}")))
                by_overlap.setdefault(ov, []).append(p)
            top = max(by_overlap) if by_overlap else None
            if top is not None and len(by_overlap[top]) >= 2:
                tied = sorted(by_overlap[top], key=lambda p: (p.price, p.id))
                return recipe.slug, ing_name, tied[0], tied[-1]
    return None


def test_preference_reaches_the_classic_planner():
    """A genuine semantic tie, broken by preference toward the Canadian
    product even though it costs more. Skips explicitly if the seeded
    catalog offers no tie — a skip is visible; a vacuous pass is not."""
    from pantry_planner import flow

    pair = _equal_match_pair()
    if pair is None:
        pytest.skip("seeded catalog has no equal-match pair to break a tie on")
    slug, ing_name, cheap, dear = pair
    assert cheap.id != dear.id
    _mark([dear.id], country="Canada", claim="product-of")

    plain = {li.ingredient_name: li.product_id for li in flow.run(slug).line_items}
    pref = {li.ingredient_name: li.product_id
            for li in flow.run(slug, preference=["Canada"]).line_items}
    assert plain[ing_name] == cheap.id, "no preference: cheapest of the tie"
    assert pref[ing_name] == dear.id, "preference: the Canadian one of the tie"


def test_preference_alone_marks_the_plan_as_an_origin_question():
    from pantry_planner import flow

    plan = flow.run("tomato_penne", preference=["Canada"])
    assert plan.origin_status in ("verified", "unverified")
    assert plan.origin_coverage is not None


# ─── Coverage: only when asked, and unmissable when it is ────────────

def test_no_origin_question_means_no_coverage_stamp():
    from pantry_planner import flow

    plan = flow.run("tomato_penne")
    assert plan.origin_coverage is None
    assert plan.origin_status == "not_requested"


def test_origin_question_sets_a_status_the_caller_cannot_miss():
    from pantry_planner import flow

    plan = flow.run("tomato_penne", exclude=["United States"])
    assert plan.origin_coverage is not None
    assert plan.origin_status == "unverified"           # nothing is evidenced
    assert plan.origin_coverage.meets_floor is False
    assert "UNVERIFIED" in plan.origin_coverage.note


def test_coverage_is_spend_weighted_not_just_counted():
    from pantry_planner import db
    from pantry_planner.origins import basket_coverage

    products = db.load_all_products()
    cheap, dear = products[0].id, products[1].id
    _mark([cheap], country="Canada")
    cov = basket_coverage([(cheap, 1.00), (dear, 99.00)])
    assert cov.count_fraction == 0.5
    assert cov.spend_fraction < 0.02
    assert cov.meets_floor is False


def test_products_without_evidence_are_never_excluded():
    from pantry_planner import flow

    plan = flow.run("tomato_penne", exclude=["United States"])
    assert plan.line_items


# ─── Unknown country names fail loudly ───────────────────────────────

def test_unknown_country_is_422_with_suggestions_over_rest():
    from fastapi.testclient import TestClient

    from pantry_planner import api

    r = TestClient(api.app).post("/plan/week", json={"days": 2, "exclude_origin": ["Itly"]})
    assert r.status_code == 422, r.text[:200]
    assert r.json()["detail"]["unknown"] == {"Itly": ["Italy"]}


def test_misspelt_america_suggests_united_states():
    from pantry_planner.origins import validate_countries

    assert validate_countries(["Amerca"])["Amerca"][0] == "United States"
    assert validate_countries(["USA", "Canada", "México"]) == {}


@pytest.mark.asyncio
async def test_unknown_country_is_a_tool_error_over_mcp():
    from mcp.server.mcpserver.exceptions import ToolError

    from pantry_planner.mcp_server import server

    with pytest.raises(ToolError, match="Amerca"):
        await server.call_tool("rank_products_by_origin",
                               {"preference": [], "exclude": ["Amerca"]})

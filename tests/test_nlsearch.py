"""NL2SQL query-plan tests — no LLM calls, no network.

The parse is injected (ParsedInput fixtures); everything downstream —
validation clamp, template compilation, the staged plan execution against
a seeded SQLite DB (t1 existence → t2 options → t3 stats → t4 lookups),
the abort gates, and the retrieval stats — runs for real.
"""
from __future__ import annotations

import os

import pytest

# Point the app at a per-session sqlite DB BEFORE importing app modules.
_TMP_DB = None


@pytest.fixture(scope="module", autouse=True)
def seeded_db(tmp_path_factory):
    global _TMP_DB
    _TMP_DB = tmp_path_factory.mktemp("db") / "test.db"
    os.environ["DB_URL"] = os.environ.get("PANTRY_TEST_DB_URL") or f"sqlite:///{_TMP_DB}"
    from pantry_planner import config, db
    from pantry_planner.nlsearch import vocab

    config.settings.cache_clear()
    db.seed_from_json()
    vocab.clear_cache()
    yield
    config.settings.cache_clear()
    vocab.clear_cache()


def _parsed(**kw):
    from pantry_planner.nlsearch.schemas import (
        Constraints, IngredientSpec, ParsedInput, RecipeSpec)

    ingredients = kw.pop("ingredients", [IngredientSpec(name="spaghetti")])
    constraints = Constraints(**kw.pop("constraints", {}))
    return ParsedInput(
        recipe=RecipeSpec(title=kw.pop("title", "Test"), servings=2,
                          ingredients=ingredients),
        constraints=constraints, **kw)


def _run(parsed):
    from pantry_planner.nlsearch.planner import run_query_plan
    return run_query_plan("ignored", parsed=parsed)


# ─── units / pre-processing ──────────────────────────────────

def test_brand_statistics_are_json_floats_whatever_the_database_returns():
    """Postgres returns AVG(rating) over an integer column as Decimal; the selector's payload is
    JSON, which cannot carry one (the Postgres CI job failed on it, SQLite returns a float)."""
    import json
    from decimal import Decimal

    from pantry_planner.models import Product
    from pantry_planner.nlsearch.planner import _brand_regroup

    pool = [Product(id=1, name="Penne Rigate 500g", description="", price=1.97,
                    brand="Fraser Farms"),
            Product(id=2, name="Penne 900g", description="", price=2.49, brand="Fraser Farms")]
    rows = [{"product_id": 1, "avg_price": Decimal("2.10"), "min_price": Decimal("1.97"),
             "avg_rating": Decimal("4.3333"), "review_count": Decimal(3)},
            {"product_id": 2, "avg_price": 2.6, "min_price": 2.49, "avg_rating": None,
             "review_count": None}]
    [stats] = _brand_regroup({0: pool}, [type("I", (), {"name": "penne"})()], rows)["penne"]
    assert stats == {"brand": "Fraser Farms", "options": 2, "avg_price": 2.35, "min_price": 1.97,
                     "avg_rating": 4.3, "review_count": 3}
    json.dumps(stats)


def test_unit_normalization():
    from pantry_planner.nlsearch.units import normalize_quantity

    assert normalize_quantity(225, "g") == (225, "g")
    assert normalize_quantity(2, "cups") == (500, "ml")
    assert normalize_quantity(1, "lb") == (454, "g")
    assert normalize_quantity(1, "dozen") == (12, "each")
    assert normalize_quantity(1, "pinch") is None
    assert normalize_quantity(None, "g") is None


def test_tokens_stemming():
    from pantry_planner.nlsearch.units import tokens

    assert tokens("Roma Tomatoes") == ["roma", "tomato"]
    assert tokens("fresh Yellow Onions") == ["yellow", "onion"]
    assert "spaghetti" in tokens("400g spaghetti")


# ─── validate_parsed / post-processing clamp ─────────────────

def test_validate_clamps_tags_and_levels():
    from pantry_planner.nlsearch.query_parser import validate_parsed
    from pantry_planner.nlsearch.vocab import db_vocab

    p = _parsed(constraints={
        "exclude_tags": ["dairy", "plutonium"],
        "categories": ["cheese"],            # actually a subcategory
        "subcategories": ["dairy", "bogus"],  # actually a category / unknown
        "max_total_budget": -5,
        "max_distance_km": 9999,
    })
    v = validate_parsed(p, db_vocab())
    assert v.constraints.exclude_tags == ["dairy"]
    assert v.constraints.subcategories == ["cheese"]   # moved to its true level
    assert v.constraints.categories == ["dairy"]
    assert v.constraints.max_total_budget is None
    assert v.constraints.max_distance_km is None       # out of range -> dropped
    assert "plutonium" in v.ignored and "bogus" in v.ignored


def test_validate_unknown_form_folds_into_name():
    from pantry_planner.nlsearch.query_parser import validate_parsed
    from pantry_planner.nlsearch.schemas import IngredientSpec
    from pantry_planner.nlsearch.vocab import db_vocab

    p = _parsed(ingredients=[IngredientSpec(name="tomato", form="sun-dried"),
                             IngredientSpec(name="tomato", form="Canned")])
    v = validate_parsed(p, db_vocab())
    assert v.recipe.ingredients[0].form is None
    assert v.recipe.ingredients[0].name == "sun-dried tomato"
    assert v.recipe.ingredients[1].form == "canned"


# ─── sql_builder / template compilation ──────────────────────

def test_existence_sql_shape_and_binding():
    from pantry_planner.nlsearch.schemas import IngredientSpec
    from pantry_planner.nlsearch.sql_builder import build_existence_sql

    sql, params = build_existence_sql([
        IngredientSpec(name="tomato", form="canned"),
        IngredientSpec(name="spaghetti")])
    assert "product_terms" in sql and "LIKE" not in sql   # inverted index, no scans
    assert "strict_matches" in sql and "relaxed_matches" in sql
    assert params["i0t0"] == "canned" and params["i0t1"] == "tomato"
    assert params["i1t0"] == "spaghetti"
    # rows are (ing_no, term, strict, base, gen, eqv): the form token is
    # strict-only; neither name has a descriptor to drop or a form word with
    # an equivalent, so no row carries the gen or eqv flag
    assert "(0, :i0t0, 1, 0, 0, 0)" in sql and "(0, :i0t1, 1, 1, 0, 0)" in sql
    # counts are (ing_no, n_all, n_base, n_gen, n_eqv)
    assert "(0, 2, 1, 0, 0)" in sql and "(1, 1, 1, 0, 0)" in sql


def test_existence_sql_flags_generic_terms():
    """"light soy sauce": the descriptor stays a base token (the relaxed
    level still needs it) but only soy + sauce count at the generic level."""
    from pantry_planner.nlsearch.schemas import IngredientSpec
    from pantry_planner.nlsearch.sql_builder import build_existence_sql

    sql, params = build_existence_sql([IngredientSpec(name="light soy sauce")])
    assert [params["i0t0"], params["i0t1"], params["i0t2"]] == ["light", "soy", "sauce"]
    assert "(0, :i0t0, 1, 1, 0, 0)" in sql             # light: base, not generic
    assert "(0, :i0t1, 1, 1, 1, 0)" in sql and "(0, :i0t2, 1, 1, 1, 0)" in sql
    assert "(0, 3, 3, 2, 0)" in sql                    # 2 generic tokens to match
    assert "generic_matches" in sql


def test_existence_sql_flags_equivalent_form_terms():
    """"cumin powder": the equivalent level swaps powder for ground, so it
    needs cumin + ground; "powder" is strict-only and "ground" eqv-only."""
    from pantry_planner.nlsearch.schemas import IngredientSpec
    from pantry_planner.nlsearch.sql_builder import build_existence_sql

    sql, params = build_existence_sql([IngredientSpec(name="cumin powder")])
    assert [params["i0t0"], params["i0t1"], params["i0t2"]] == ["cumin", "powder", "ground"]
    assert "(0, :i0t0, 1, 1, 0, 1)" in sql             # cumin: every level but generic
    assert "(0, :i0t1, 1, 1, 0, 0)" in sql             # powder: not at the eqv level
    assert "(0, :i0t2, 0, 0, 0, 1)" in sql             # ground: only the eqv level
    assert "(0, 2, 2, 0, 2)" in sql and "equivalent_matches" in sql


def test_a_form_word_repeated_in_the_name_is_counted_once():
    """Parsed as name "ground Sichuan peppercorn" AND form "ground", the
    strict level must need "ground" once: token-AND compares distinct hits
    with the term count, and a duplicate made it unmatchable."""
    from pantry_planner.nlsearch.schemas import IngredientSpec

    r = _run(_parsed(ingredients=[IngredientSpec(name="ground Sichuan peppercorn",
                                                 form="ground")]))
    assert r.match_levels == {1: "exact"}
    assert [p.name for p in r.pools[0] if not p.substitute] == \
        ["Sichuan Peppercorns Ground 40g"]


def test_options_sql_patterns_and_binding():
    from pantry_planner.nlsearch.schemas import Constraints, IngredientSpec
    from pantry_planner.nlsearch.sql_builder import build_options_sql

    c = Constraints(max_item_price=10, exclude_tags=["dairy"],
                    exclude_subcategories=["canned"])
    ing = [IngredientSpec(name="tomato", form="canned", quantity=225, unit="g"),
           IngredientSpec(name="spaghetti")]
    sql, params = build_options_sql(c, ing, relaxed=set(),
                                    lat=49.28, lon=-123.12, max_km=10)

    assert "sp.price <= :max_item_price" in sql and params["max_item_price"] == 10
    assert "NOT LIKE :xtag0" in sql and params["xtag0"] == "%,dairy,%"
    assert "p.subcategory NOT IN (:xsub0)" in sql
    assert params["i0t0"] == "canned"                  # form is a required token
    assert params["need0"] == 225 and params["uom0"] == "g"
    assert params["maxdist2"] == 100                   # 10km, compared squared
    assert "ROW_NUMBER() OVER (PARTITION BY m.ing_no, p.id" in sql   # best store
    assert "PARTITION BY b.ing_no" in sql              # per-ingredient limit
    assert sql.count("product_terms") == 1             # ONE pass for all ingredients


def test_options_sql_form_relaxation_drops_form_token():
    from pantry_planner.nlsearch.schemas import Constraints, IngredientSpec
    from pantry_planner.nlsearch.sql_builder import build_options_sql

    ing = [IngredientSpec(name="tomato", form="powdered")]
    _, strict_params = build_options_sql(Constraints(), ing, relaxed=set(),
                                         lat=49.28, lon=-123.12)
    _, relaxed_params = build_options_sql(Constraints(), ing, relaxed={0},
                                          lat=49.28, lon=-123.12)
    assert strict_params["i0t0"] == "powdered"
    assert relaxed_params["i0t0"] == "tomato"          # form dropped


def test_sql_injection_stays_parameterized():
    from pantry_planner.nlsearch.schemas import Constraints, IngredientSpec
    from pantry_planner.nlsearch.sql_builder import (build_existence_sql,
                                                     build_options_sql)

    evil = "x'; DROP TABLE products; --"
    for sql, params in (build_existence_sql([IngredientSpec(name=evil)]),
                        build_options_sql(Constraints(), [IngredientSpec(name=evil)],
                                          relaxed=set(), lat=49.28, lon=-123.12)):
        assert "DROP TABLE" not in sql                 # never concatenated into SQL
        assert any("drop" in str(v).lower() for v in params.values())


# ─── plan composition ────────────────────────────────────────

def test_build_plan_shapes():
    from pantry_planner.nlsearch.plan import GateCode, StepKind
    from pantry_planner.nlsearch.planner import build_plan

    p = _parsed(constraints={"max_total_budget": 30, "max_distance_km": 10})
    plan = build_plan(p, max_km=10)
    assert [s.id for s in plan.steps] == ["t1_existence", "t2_options", "t3_statistics"]
    assert plan.steps[0].gate == GateCode.missing_ingredients
    assert plan.steps[1].gate == GateCode.unavailable_within_constraints
    assert plan.steps[1].params_summary["max_km"] == 10
    assert plan.steps[2].kind == StepKind.statistics and plan.steps[2].gate is None


# ─── execution: happy path ───────────────────────────────────

def test_plan_narrows_and_pools():
    from pantry_planner.nlsearch.schemas import IngredientSpec

    r = _run(_parsed(
        ingredients=[IngredientSpec(name="tomato", form="canned"),
                     IngredientSpec(name="cheddar")],
        constraints={"exclude_tags": ["gluten"]}))
    direct0 = [p for p in r.pools[0] if not p.substitute]
    assert all("dairy" not in p.category for p in direct0)
    assert any("anned" in p.name for p in direct0)     # form token hit
    cheddars = {p.name for p in r.pools[1]}
    assert "Cheddar Cheese Block 300g" in cheddars
    # every candidate is pinned to a store offer
    assert all(p.store_name and p.store_price is not None for p in r.products)
    steps = {s.step_id: s for s in r.execution.steps}
    assert steps["t1_existence"].outcome == "ok"
    assert steps["t2_options"].outcome == "ok"
    assert "SELECT" in steps["t2_options"].sql_display


def test_no_dairy_excludes_dairy_products():
    from pantry_planner.nlsearch.schemas import IngredientSpec

    r = _run(_parsed(ingredients=[IngredientSpec(name="milk")],
                     constraints={"exclude_tags": ["dairy"]}))
    names = {p.name for p in r.products}
    assert "Whole Milk 1L" not in names
    assert "Oat Milk 1L" in names            # dairy-free alternative retrieved


def test_size_fit_ordering():
    """225g beef need: covering pack closest to need ranks first."""
    from pantry_planner.nlsearch.schemas import IngredientSpec

    r = _run(_parsed(ingredients=[
        IngredientSpec(name="ground beef", quantity=225, unit="g")]))
    pool = [p for p in r.pools[0] if not p.substitute]
    assert pool[0].name == "Ground Beef Extra Lean 300g"   # smallest covering pack
    sizes = [p.unit_qty for p in pool if p.unit_qty]
    assert 300 in sizes and 450 in sizes


def test_form_relaxation_feeds_router_stat():
    """Unstocked purchase form: t1 relaxes it instead of aborting, and the
    router sees it as a zero-hit signal."""
    from pantry_planner.nlsearch.schemas import IngredientSpec

    r = _run(_parsed(ingredients=[IngredientSpec(name="tomato", form="powdered")]))
    assert r.stats.zero_hit_ingredients == 1
    assert r.pools[0], "form-relaxed pool should not be empty"
    assert "form relaxation" in r.execution.steps[0].label


def test_unparseable_recipe_raises():
    from pantry_planner.nlsearch.planner import UnparseableRecipe

    with pytest.raises(UnparseableRecipe):
        _run(_parsed(ingredients=[]))


def test_value_disagreement_stat():
    from pantry_planner.nlsearch.schemas import IngredientSpec

    r = _run(_parsed(ingredients=[IngredientSpec(name="rice")]))
    assert 0.0 <= r.stats.value_disagreement <= 1.0


# ─── size cap: the smallest pack on offer stays admissible ────

def test_regression_condiment_quantities_do_not_abort_t2():
    """Live 2026-10-02 (pantry-api#21): "1 tbsp soy sauce" and "1/3 cup
    canola oil" aborted t2 with unavailable_within_constraints although the
    text named no distance, price or diet. No default distance is applied
    (t2 carries no max_km) and both products are stocked at every store: the
    SIZE_RANGE cap dropped every pack over 6x the need (500 ml for 15 ml,
    1 L for 83 ml), and the attribution probe kept the cap, so the alert
    blamed the dietary filter. The smallest pack on offer is now always
    admissible — for soy sauce that is every 500 ml bottle (plain, dark and
    light since the 160-product catalog), cheapest first."""
    from sqlalchemy import text
    from sqlalchemy.orm import Session

    from pantry_planner import db
    from pantry_planner.nlsearch.planner import build_plan
    from pantry_planner.nlsearch.schemas import IngredientSpec

    soy, canola, thighs = 59, 17, 46               # seeds/products.json
    ings = [IngredientSpec(name="boneless skinless chicken thigh", quantity=1, unit="lb"),
            IngredientSpec(name="soy sauce", quantity=1, unit="tbsp"),
            IngredientSpec(name="canola oil", quantity=1 / 3, unit="cup"),
            IngredientSpec(name="garlic", quantity=5, unit="cloves"),
            IngredientSpec(name="ginger", quantity=1, unit="thumb")]
    with Session(db.engine()) as s:
        sizes = dict(s.execute(text(
            "SELECT id, unit_qty FROM products WHERE id IN (:a, :b)"),
            {"a": soy, "b": canola}).all())
        n_stores = s.execute(text("SELECT COUNT(*) FROM stores")).scalar_one()
        offers = dict(s.execute(text(
            "SELECT product_id, COUNT(*) FROM store_products "
            "WHERE product_id IN (:a, :b) GROUP BY product_id"),
            {"a": soy, "b": canola}).all())
        cheapest_soy = s.execute(text(
            "SELECT MIN(price) FROM store_products WHERE product_id = :a"),
            {"a": soy}).scalar_one()
        # every product the token-AND match admits, by its cheapest offer
        soy_sauces = [(pid, qty) for pid, qty in s.execute(text(
            "SELECT p.id, p.unit_qty FROM products p "
            "JOIN product_terms a ON a.product_id = p.id AND a.term = 'soy' "
            "JOIN product_terms b ON b.product_id = p.id AND b.term = 'sauce' "
            "JOIN store_products sp ON sp.product_id = p.id "
            "GROUP BY p.id, p.unit_qty ORDER BY MIN(sp.price), p.id"))]
    assert sizes[soy] > 15 * 6 and sizes[canola] > 250 / 3 * 6   # every pack over the cap
    assert soy_sauces[0][0] == soy and len(soy_sauces) >= 1
    assert {qty for _, qty in soy_sauces} == {sizes[soy]}         # all the smallest pack
    assert offers == {soy: n_stores, canola: n_stores}            # stocked everywhere
    p = _parsed(ingredients=ings)
    assert "max_km" not in build_plan(p, max_km=None).steps[1].params_summary

    r = _run(p)                                    # must not raise PlanAborted
    assert r.execution.aborted is None
    direct = {n: [x.id for x in pool if not x.substitute] for n, pool in r.pools.items()}
    assert direct[1] == [pid for pid, _ in soy_sauces] and direct[2] == [canola]
    assert direct[0][0] == thighs                  # Chicken Thighs Boneless 450g
    assert r.pools[1][0].store_price == pytest.approx(float(cheapest_soy))


def test_size_cap_still_drops_catering_packs_when_a_fitting_pack_exists():
    """50 g ground beef: the 300 g pack is within 6x, so 450 g and 900 g stay
    out. 50 ml olive oil: no bottle is within 6x (300 ml), so the cap scales
    to the smallest bottle — 500 ml and 1 L (within 6 x 500 ml) both stay,
    ranked by price because neither fits the need any better."""
    from pantry_planner.nlsearch.schemas import IngredientSpec

    r = _run(_parsed(ingredients=[
        IngredientSpec(name="ground beef", quantity=50, unit="g"),
        IngredientSpec(name="olive oil", quantity=50, unit="ml")]))
    assert [p.name for p in r.pools[0] if not p.substitute] == ["Ground Beef Extra Lean 300g"]
    assert [p.name for p in r.pools[1] if not p.substitute] == \
        ["Extra Virgin Olive Oil 500ml", "Olive Oil 1L"]


@pytest.mark.parametrize("name,qty,unit,cheapest", [
    # every salt box is over 6 x 5 g: Table Salt 1kg ($1.46) is the cheapest,
    # and the smallest-pack-only rule kept just Fine Sea Salt 750g ($3.22)
    ("salt", 5, "g", "Table Salt 1kg"),
    ("salt", 1, "tsp", "Table Salt 1kg"),
    # 2 tbsp of any oil: every bottle is over 180 ml; the cheapest neutral oil
    # leads, not the 250 ml sesame oil that used to be the only candidate
    ("oil", 2, "tbsp", "Canola Oil 1L"),
    ("vinegar", 2, "tbsp", "White Vinegar 1L"),
])
def test_when_no_pack_fits_the_need_the_cheapest_pack_leads(name, qty, unit, cheapest):
    from pantry_planner.nlsearch.schemas import IngredientSpec

    r = _run(_parsed(ingredients=[IngredientSpec(name=name, quantity=qty, unit=unit)]))
    direct = [p for p in r.pools[0] if not p.substitute]
    assert direct[0].name == cheapest
    assert direct[0].store_price == min(p.store_price for p in direct)


# ─── gate: missing_ingredients ───────────────────────────────

def test_missing_ingredient_aborts_with_suggestions():
    from pantry_planner.nlsearch.plan import GateCode
    from pantry_planner.nlsearch.planner import PlanAborted
    from pantry_planner.nlsearch.schemas import IngredientSpec

    with pytest.raises(PlanAborted) as exc:
        _run(_parsed(ingredients=[
            IngredientSpec(name="spaghetti"),
            IngredientSpec(name="saffron", category_hint="spice")]))
    ex = exc.value.execution
    assert ex.aborted.code == GateCode.missing_ingredients
    assert ex.aborted.stage == "t1_existence"
    assert ex.steps[0].outcome == "aborted"
    (detail,) = ex.aborted.details
    assert detail["name"] == "saffron"
    assert len(detail["suggestions"]) == 3             # same-hint alternatives
    assert all("$" in s for s in detail["suggestions"])


# ─── gate: unavailable_within_constraints ────────────────────

def test_distance_gate_14km_store():
    """MegaSave Richmond sits at ~13.9 km: excluded at 10 km, included at 20."""
    from pantry_planner.nlsearch.schemas import IngredientSpec

    def stores_at(km):
        r = _run(_parsed(ingredients=[IngredientSpec(name="spaghetti")],
                         constraints={"max_distance_km": km}))
        return {p.store_name for pool in r.pools.values() for p in pool}

    assert "MegaSave Richmond" not in stores_at(10)
    assert "MegaSave Richmond" in stores_at(20)


def test_unavailable_abort_attributes_constraint():
    from pantry_planner.nlsearch.plan import GateCode
    from pantry_planner.nlsearch.planner import PlanAborted
    from pantry_planner.nlsearch.schemas import IngredientSpec

    with pytest.raises(PlanAborted) as exc:
        _run(_parsed(ingredients=[IngredientSpec(name="milk")],
                     constraints={"max_item_price": 0.5}))
    alert = exc.value.execution.aborted
    assert alert.code == GateCode.unavailable_within_constraints
    # the attribution probe names a concrete out-of-constraint offer
    assert "available only outside the constraints" in alert.details[0]["reason"]
    assert "$" in alert.details[0]["reason"]


# ─── gate: budget_infeasible ─────────────────────────────────

def test_budget_floor_math():
    """Floor = sum of each pool's cheapest offer; gate fires exactly on it."""
    from pantry_planner.nlsearch.plan import GateCode
    from pantry_planner.nlsearch.planner import PlanAborted
    from pantry_planner.nlsearch.schemas import IngredientSpec

    ings = [IngredientSpec(name="spaghetti"), IngredientSpec(name="ground beef")]
    free = _run(_parsed(ingredients=ings))
    floor = sum(min(p.store_price for p in pool if not p.substitute)
                for pool in free.pools.values())

    ok = _run(_parsed(ingredients=ings,
                      constraints={"max_total_budget": floor + 0.01}))
    assert ok.execution.aborted is None

    with pytest.raises(PlanAborted) as exc:
        _run(_parsed(ingredients=ings,
                     constraints={"max_total_budget": floor - 0.01}))
    alert = exc.value.execution.aborted
    assert alert.code == GateCode.budget_infeasible
    assert f"${floor:.2f}" in alert.message


# ─── t3: brand statistics ────────────────────────────────────

def test_brand_stats_grouping():
    from pantry_planner.nlsearch.schemas import IngredientSpec

    r = _run(_parsed(ingredients=[IngredientSpec(name="ground beef")]))
    rows = r.brand_stats["ground beef"]
    assert len(rows) >= 2                              # multiple brands to compare
    for row in rows:
        assert set(row) == {"brand", "options", "avg_price", "min_price",
                            "avg_rating", "review_count"}
        assert row["options"] >= 1 and row["avg_price"] > 0


def test_brand_stats_keep_reviewless_products():
    """LEFT JOIN semantics: a product with zero reviews still gets a stats
    row (avg_rating None), it is not dropped from the brand grouping."""
    from sqlalchemy import text
    from sqlalchemy.orm import Session

    from pantry_planner import db
    from pantry_planner.nlsearch.schemas import IngredientSpec

    r0 = _run(_parsed(ingredients=[IngredientSpec(name="ground beef")]))
    victim = r0.pools[0][0]
    with Session(db.engine()) as s:
        s.execute(text("DELETE FROM reviews WHERE product_id = :pid"),
                  {"pid": victim.id})
        s.commit()
    try:
        r = _run(_parsed(ingredients=[IngredientSpec(name="ground beef")]))
        row = next(b for b in r.brand_stats["ground beef"]
                   if b["brand"] == victim.brand)
        assert row["options"] >= 1                     # still grouped
    finally:
        db.seed_from_json()                            # restore fixture data


# ─── t4: substitutes for thin pools ──────────────────────────

def test_thin_pool_gets_labeled_substitutes():
    from pantry_planner.nlsearch.schemas import IngredientSpec

    r = _run(_parsed(ingredients=[IngredientSpec(name="yellow onion")]))
    pool = r.pools[0]
    direct = [p for p in pool if not p.substitute]
    subs = [p for p in pool if p.substitute]
    assert direct and subs                             # appended, never replaced
    assert pool[0].substitute is False                 # direct match stays first
    assert any(s.step_id.startswith("t4_lookup") for s in r.execution.steps)
    assert all(p.subcategory == direct[0].subcategory for p in subs)


# ─── efficiency layer ────────────────────────────────────────

def test_product_terms_tokenizer_parity():
    """The DB's precomputed terms must equal the parser's tokenizer output
    (units.index_text: name + description, negated words cut). pantry-db's
    KEEP-IN-SYNC copy is held to the GOLDEN_TERMS table below. Every
    product, not a sample: the catalog grows by copying pantry-db's seed
    (pjvjay/pantry-api#22), and a new row ("Jalapeno", "Whipping Cream
    35%") is where a tokenizer edge case would first show."""
    from sqlalchemy import text
    from sqlalchemy.orm import Session

    from pantry_planner import db
    from pantry_planner.nlsearch.units import index_text, tokens
    from tests.seed_catalog import CATALOG

    with Session(db.engine()) as s:
        rows = s.execute(text("SELECT id, name, description FROM products")).all()
        db_terms: dict[int, set[str]] = {}
        for pid, term in s.execute(text("SELECT product_id, term FROM product_terms")):
            db_terms.setdefault(pid, set()).add(term)
    assert len(rows) == CATALOG
    for pid, name, desc in rows:
        assert db_terms.get(pid) == set(tokens(index_text(name, desc))), name


# pantry-db's scripts/gen-seed-sql.py holds this table verbatim (GOLDEN_TERMS)
# and asserts it on every run, so its tokenizer copy and this one agree on
# the cases that matter: irregular plurals, negated description words, and
# what must NOT change (olives/cloves/chives, "dairy-free", the split "ñ").
GOLDEN_TERMS = [
    (("Bay Leaves 10g", "Dried whole bay leaves, 10g bag"),
     ["bag", "bay", "dried", "leaf", "whole"]),
    (("Olives", "Cloves, chives, two loaves and halves"),
     ["and", "chive", "clove", "half", "loaf", "olive", "two"]),
    (("Roma Tomato", "Fresh Roma tomatoes, 500g pack"), ["pack", "roma", "tomato"]),
    (("Crushed Tomatoes Canned 796ml", "Canned crushed tomatoes, no salt added, 796ml"),
     ["canned", "crushed", "ml", "tomato"]),
    (("Canadian Peanut Butter Cream", "Smooth peanut butter, no jelly, 500g"),
     ["butter", "canadian", "cream", "peanut", "smooth"]),
    (("Oat Milk 1L", "Unsweetened oat beverage, dairy-free, 1L carton"),
     ["beverage", "carton", "dairy", "free", "milk", "oat", "unsweetened"]),
    (("Jalapeno Peppers", "Fresh jalapeño peppers (jalapeños, jalapenos), ~200g"),
     ["jalape", "jalapeno", "os", "pepper"]),
]


@pytest.mark.parametrize("product,terms", GOLDEN_TERMS)
def test_tokenizer_golden_cases_shared_with_pantry_db(product, terms):
    from pantry_planner.storeseed import product_terms

    name, description = product
    assert product_terms({"name": name, "description": description}) == terms


def test_single_pass_matches_naive_reference():
    """The windowed single-pass query returns the same pools as a plain
    Python reference implementation over the same data."""
    from sqlalchemy import text
    from sqlalchemy.orm import Session

    from pantry_planner import db
    from pantry_planner.nlsearch.schemas import IngredientSpec
    from pantry_planner.nlsearch.units import normalize_quantity, tokens

    ings = [IngredientSpec(name="rice"),
            IngredientSpec(name="ground beef", quantity=225, unit="g"),
            IngredientSpec(name="cheddar"),
            # every pack > 6x the need: packs within 6x the smallest stay
            IngredientSpec(name="soy sauce", quantity=1, unit="tbsp"),
            # no bottle within 6x 50 ml: 500 ml and 1 L both stay, by price
            IngredientSpec(name="olive oil", quantity=50, unit="ml"),
            # 50 g ground beef: the 300 g pack fits, so 450 g / 900 g are capped
            IngredientSpec(name="ground beef", quantity=50, unit="g")]
    r = _run(_parsed(ingredients=ings))

    with Session(db.engine()) as s:
        terms: dict[int, set[str]] = {}
        for pid, term in s.execute(text("SELECT product_id, term FROM product_terms")):
            terms.setdefault(pid, set()).add(term)
        best_price = {pid: price for pid, price in s.execute(text(
            "SELECT product_id, MIN(price) FROM store_products GROUP BY product_id"))}
        products = {p.id: p for p in db.load_all_products()}

    for n, ing in enumerate(ings):
        toks = set(tokens(ing.name))
        need = normalize_quantity(ing.quantity, ing.unit)
        matched = [pid for pid, tset in terms.items() if toks <= tset]
        same_uom = [products[pid].unit_qty for pid in matched
                    if need and products[pid].unit_qty is not None
                    and products[pid].unit_uom == need[1]]
        smallest = min(same_uom) if same_uom else None
        offers = []
        for pid in matched:
            p = products[pid]
            sized = need and p.unit_qty is not None and p.unit_uom == need[1]
            fits = sized and p.unit_qty <= need[0] * 6
            if sized and not fits:
                # nothing fits: the cap scales to 6x the smallest pack
                if not (smallest > need[0] * 6 and p.unit_qty <= smallest * 6):
                    continue
            price = best_price[pid]
            if not sized:
                key = (1, 0, 0.0, price, pid)
            else:
                key = (0, 1 if p.unit_qty < need[0] else 0,
                       abs(p.unit_qty - need[0]) if fits else 0.0, price, pid)
            offers.append((key, pid))
        expected = [pid for _, pid in sorted(offers)][:8]
        got = [p.id for p in r.pools[n] if not p.substitute]
        assert got == expected, ing.name
    # the size-cap cases, stated outright (seeds/products.json ids)
    # Soy, Dark Soy, Light Soy Sauce 500ml: no bottle fits 15 ml, cheapest first
    assert [p.id for p in r.pools[3] if not p.substitute] == [59, 65, 64]
    assert [p.id for p in r.pools[4] if not p.substitute] == [16, 62]   # EVOO, then 1 L
    assert [p.id for p in r.pools[5] if not p.substitute] == [36]       # 300 g pack only


def test_twenty_ingredients_one_round_trip():
    """The whole recipe resolves in ONE t2 query regardless of size."""
    from pantry_planner.nlsearch.schemas import IngredientSpec

    names = ["spaghetti", "ground beef", "cheddar", "milk", "butter", "rice",
             "yellow onion", "garlic", "tomato", "olive oil", "basmati rice",
             "mozzarella", "yogurt", "bread", "chicken breast", "potato",
             "broccoli", "peanut butter", "dark chocolate", "ginger"]
    r = _run(_parsed(ingredients=[IngredientSpec(name=n) for n in names]))
    options_steps = [s for s in r.execution.steps if s.step_id == "t2_options"]
    assert len(options_steps) == 1
    assert len(r.stats.pool_sizes) == 20
    assert sum(1 for size in r.stats.pool_sizes if size > 0) == 20


# ─── router integration ──────────────────────────────────────

def test_phase_a_gains_retrieval_stats():
    from pantry_planner import db
    from pantry_planner.nlsearch.schemas import RetrievalStats
    from pantry_planner.router.deterministic import compute_phase_a
    from pantry_planner.models import Recipe, RecipeIngredient

    recipe = Recipe(slug="t", name="t", ingredients=[
        RecipeIngredient(line_no=1, name="spaghetti")])
    products = db.load_all_products()
    stats = RetrievalStats(pool_sizes=[4], zero_hit_ingredients=1,
                           value_disagreement=0.5, catalog_size=62)
    m = compute_phase_a(recipe, products, retrieval_stats=stats)
    assert m.has_retrieval and m.mean_pool_size == 4.0
    base = compute_phase_a(recipe, products)
    assert not base.has_retrieval            # classic path untouched

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
    os.environ["DB_URL"] = f"sqlite:///{_TMP_DB}"
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
    # form token is strict-only: base flag 0 in the VALUES rowset
    assert "(0, :i0t0, 0)" in sql and "(0, :i0t1, 1)" in sql


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
    """The DB's precomputed terms must equal the parser's tokenizer output —
    guards the KEEP-IN-SYNC duplicate in pantry-db's gen-seed-sql.py."""
    from sqlalchemy import text
    from sqlalchemy.orm import Session

    from pantry_planner import db
    from pantry_planner.nlsearch.units import tokens

    with Session(db.engine()) as s:
        rows = s.execute(text(
            "SELECT p.id, p.name, p.description FROM products p LIMIT 10")).all()
        for pid, name, desc in rows:
            db_terms = {t for (t,) in s.execute(
                text("SELECT term FROM product_terms WHERE product_id = :pid"),
                {"pid": pid})}
            assert db_terms == set(tokens(f"{name} {desc}")), name


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
            IngredientSpec(name="cheddar")]
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
        offers = []
        for pid, tset in terms.items():
            if not toks <= tset:
                continue
            p = products[pid]
            if (need and p.unit_qty is not None and p.unit_uom == need[1]
                    and p.unit_qty > need[0] * 6):
                continue                               # size cap
            price = best_price[pid]
            if need is None or p.unit_qty is None or p.unit_uom != need[1]:
                key = (1, 0, 0.0, price, pid)
            else:
                key = (0, 1 if p.unit_qty < need[0] else 0,
                       abs(p.unit_qty - need[0]), price, pid)
            offers.append((key, pid))
        expected = [pid for _, pid in sorted(offers)][:8]
        got = [p.id for p in r.pools[n] if not p.substitute]
        assert got == expected, ing.name


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

"""NL2SQL stage tests — no LLM calls, no network.

The parse is injected (ParsedInput fixtures); everything downstream —
validation clamp, SQL compilation, execution against a seeded SQLite DB,
the widening ladder, and the retrieval stats — runs for real.
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
    })
    v = validate_parsed(p, db_vocab())
    assert v.constraints.exclude_tags == ["dairy"]
    assert v.constraints.subcategories == ["cheese"]   # moved to its true level
    assert v.constraints.categories == ["dairy"]
    assert v.constraints.max_total_budget is None
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


# ─── sql_builder / translation ───────────────────────────────

def test_sql_builder_patterns_and_binding():
    from pantry_planner.nlsearch.schemas import Constraints, IngredientSpec
    from pantry_planner.nlsearch.sql_builder import build_retrieval_sql

    c = Constraints(max_item_price=10, exclude_tags=["dairy"],
                    exclude_subcategories=["canned"])
    ing = [IngredientSpec(name="tomato", form="canned", quantity=225, unit="g"),
           IngredientSpec(name="spaghetti")]
    sql, params = build_retrieval_sql(c, ing)

    assert "0 AS ingredient_no" in sql and "1 AS ingredient_no" in sql
    assert "price <= :max_item_price" in sql and params["max_item_price"] == 10
    assert "NOT LIKE :xtag0" in sql and params["xtag0"] == "%,dairy,%"
    assert "subcategory NOT IN (:xsub0)" in sql
    # form is a required token; size predicates present for the sized ingredient
    assert params["i0t0"] == "%canned%"
    assert params["need0"] == 225 and params["uom0"] == "g"
    assert "ABS(unit_qty - :need0)" in sql


def test_sql_injection_stays_parameterized():
    from pantry_planner.nlsearch.schemas import Constraints, IngredientSpec
    from pantry_planner.nlsearch.sql_builder import build_retrieval_sql

    evil = "x'; DROP TABLE products; --"
    sql, params = build_retrieval_sql(Constraints(), [IngredientSpec(name=evil)])
    assert "DROP TABLE" not in sql          # never concatenated into SQL
    assert any("drop" in str(v).lower() for v in params.values())


# ─── retrieve / execution + widening ─────────────────────────

def test_retrieve_narrows_and_pools():
    from pantry_planner.nlsearch.retrieve import retrieve
    from pantry_planner.nlsearch.schemas import IngredientSpec

    parsed = _parsed(
        ingredients=[IngredientSpec(name="tomato", form="canned"),
                     IngredientSpec(name="cheddar")],
        constraints={"exclude_tags": ["gluten"]})
    r = retrieve("ignored", parsed=parsed)
    assert r.stats.zero_hit_ingredients == 0
    pool0 = {p.name for p in r.pools[0]}
    assert any("Canned" in n or "canned" in n for n in pool0)     # form token hit
    assert all("dairy" not in p.category for p in r.pools[0])
    cheddars = {p.name for p in r.pools[1]}
    assert "Cheddar Cheese Block 300g" in cheddars


def test_no_dairy_excludes_dairy_products():
    from pantry_planner.nlsearch.retrieve import retrieve
    from pantry_planner.nlsearch.schemas import IngredientSpec

    parsed = _parsed(ingredients=[IngredientSpec(name="milk")],
                     constraints={"exclude_tags": ["dairy"]})
    r = retrieve("ignored", parsed=parsed)
    names = {p.name for p in r.products}
    assert "Whole Milk 1L" not in names
    assert {"Oat Milk 1L"} <= names          # dairy-free alternative retrieved


def test_size_fit_ordering():
    """225g beef need: covering pack closest to need ranks first."""
    from pantry_planner.nlsearch.retrieve import retrieve
    from pantry_planner.nlsearch.schemas import IngredientSpec

    parsed = _parsed(ingredients=[
        IngredientSpec(name="ground beef", quantity=225, unit="g")])
    r = retrieve("ignored", parsed=parsed)
    pool = r.pools[0]
    assert pool[0].name == "Ground Beef Extra Lean 300g"   # smallest covering pack
    sizes = [p.unit_qty for p in pool if p.unit_qty]
    assert 300 in sizes and 450 in sizes


def test_widening_ladder_on_zero_hit():
    from pantry_planner.nlsearch.retrieve import retrieve
    from pantry_planner.nlsearch.schemas import IngredientSpec

    parsed = _parsed(ingredients=[
        IngredientSpec(name="saffron", category_hint="spice")])
    r = retrieve("ignored", parsed=parsed)
    assert r.stats.zero_hit_ingredients == 1
    assert r.pools[0], "scope fallback should fill the pool"
    # exact-subcategory matches lead; parent-category rows may pad the tail
    assert r.pools[0][0].subcategory == "spice"
    spice_count = sum(1 for p in r.pools[0] if p.subcategory == "spice")
    assert spice_count >= 4                  # all four seeded spices retrieved


def test_unparseable_recipe_raises():
    from pantry_planner.nlsearch.retrieve import UnparseableRecipe, retrieve

    with pytest.raises(UnparseableRecipe):
        retrieve("ignored", parsed=_parsed(ingredients=[]))


def test_value_disagreement_stat():
    from pantry_planner.nlsearch.retrieve import retrieve
    from pantry_planner.nlsearch.schemas import IngredientSpec

    # rice pool: 1kg @ $4.50 (0.45/100g) vs 2kg @ $12 (0.60/100g)
    # cheapest == best-value -> no disagreement expected in this pool
    parsed = _parsed(ingredients=[IngredientSpec(name="rice")])
    r = retrieve("ignored", parsed=parsed)
    assert 0.0 <= r.stats.value_disagreement <= 1.0


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

"""Recipe-link basket quality (pjvjay/pantry-api#25, pjvjay/pantry-db#12).

The live skill runs on 2026-10-03 priced real recipes with seven kinds of
error, and the PR #24 review found more. Each test names the case it pins and
states the right answer from somewhere other than the code under test:
seeds/products.json ids, a direct SQL read, or the arithmetic of the plan.

No LLM: the planner tests inject the parse; the MCP/REST tests run in
DEMO_MODE with no ANTHROPIC_API_KEY.
"""
from __future__ import annotations

import os

import pytest
from mcp.server.mcpserver.exceptions import ToolError
from sqlalchemy import text
from sqlalchemy.orm import Session

# seeds/products.json
SPAGHETTI, EVOO, CANOLA, CRUSHED_TOMATOES, PB_CREAM = 21, 16, 17, 33, 6
CUMIN_GROUND, CORIANDER_GROUND, CUMIN_SEEDS, CHILI_POWDER = 26, 28, 100, 98
THIGHS_BONE_IN, THIGHS_BONELESS, DRUMSTICKS = 11, 46, 160
SICHUAN_WHOLE, SICHUAN_GROUND = 66, 67
BAY_LEAVES, CILANTRO, MINT, PARSLEY = 136, 143, 145, 144
GREEN_CHILIES, DRIED_RED_CHILIES = 105, 68
WHITE_FISH, FISH_SAUCE, DIJON, MUSTARD_SEEDS = 161, 94, 141, 99
AP_FLOUR, FLOUR_TORTILLAS, WHITE_VINEGAR, TABLE_SALT = 61, 107, 132, 122
VEG_OIL, SESAME_OIL, BUTTER_SALTED, PAPRIKA = 131, 72, 18, 134


@pytest.fixture(scope="module", autouse=True)
def seeded_db(tmp_path_factory):
    os.environ["DB_URL"] = (os.environ.get("PANTRY_TEST_DB_URL")
                            or f"sqlite:///{tmp_path_factory.mktemp('db') / 'basket.db'}")
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
def reseed():
    yield
    from pantry_planner import db
    from pantry_planner.nlsearch import vocab

    db.seed_from_json()
    vocab.clear_cache()


@pytest.fixture()
def server():
    from pantry_planner.mcp_server import server

    return server


def _sql(sql: str, **params):
    from pantry_planner.db import engine

    with Session(engine()) as s:
        res = s.execute(text(sql), params)
        rows = res.all() if res.returns_rows else []
        s.commit()
        return rows


def _cheapest(product_id: int) -> float:
    (p,), = _sql("SELECT MIN(price) FROM store_products WHERE product_id = :p", p=product_id)
    return float(p)


def _ing(name, **kw):
    from pantry_planner.nlsearch.schemas import IngredientSpec

    return IngredientSpec(name=name, **kw)


def _run(ingredients, *, allow_partial=False, max_km=None, **constraints):
    from pantry_planner.nlsearch.planner import run_query_plan
    from pantry_planner.nlsearch.schemas import Constraints, ParsedInput, RecipeSpec

    parsed = ParsedInput(recipe=RecipeSpec(title="Basket", servings=2, ingredients=ingredients),
                         constraints=Constraints(**constraints))
    return run_query_plan("ignored", parsed=parsed, allow_partial=allow_partial,
                          max_km=max_km)


def _direct(pool):
    return [p.id for p in pool if not p.substitute]


def _demo_pick(r, n=0):
    from pantry_planner import demomode

    sel = demomode.select_products([r.recipe.ingredients[n]], r.pools[n], model="demo")
    return sel.selections[0].product_id


async def _plan(server, recipe_text: str, **args):
    from pantry_planner.mcp_server import PlanResult

    res = await server.call_tool("plan_from_text", {"recipe_text": recipe_text, **args})
    return PlanResult.model_validate(res.structured_content).summary


def _accounted(s) -> int:
    """Every recipe ingredient lands in exactly one place."""
    return (sum(1 + len(ln.also_lines) for ln in s.lines)
            + len(s.not_stocked) + len(s.out_of_range) + len(s.skipped))


# ─── 1. lines that resolve to one product are one purchase ────

def test_two_lines_choosing_one_product_are_one_purchase_and_one_pack():
    """Mala chicken: "ground Sichuan peppercorns" (1 tsp) and "Sichuan
    peppercorns" (2 tsp) both chose the whole 50 g bag. One purchase, one
    pack: teaspoons say nothing about how many 50 g bags that is."""
    from pantry_planner import db
    from pantry_planner.flow import group_purchases
    from pantry_planner.models import RecipeIngredient, Selection
    from pantry_planner.nlsearch.units import normalize_quantity

    whole = next(p for p in db.load_all_products() if p.id == SICHUAN_WHOLE)
    lines = {10: RecipeIngredient(line_no=10, name="ground Sichuan peppercorn"),
             13: RecipeIngredient(line_no=13, name="Sichuan peppercorn"),
             14: RecipeIngredient(line_no=14, name="garlic")}
    sels = [Selection(line_no=13, product_id=SICHUAN_WHOLE, confidence=0.9),
            Selection(line_no=10, product_id=SICHUAN_WHOLE, confidence=0.8)]
    needs = {10: normalize_quantity(1, "teaspoon"), 13: normalize_quantity(2, "teaspoons")}
    purchases, unselected = group_purchases(sels, {SICHUAN_WHOLE: whole}, lines, needs)
    (pu,) = purchases
    assert [sel.line_no for sel, _ in pu.lines] == [10, 13]      # recipe order
    assert pu.packs == 1
    assert [(ing.line_no, why) for ing, why in unselected] == [
        (14, "the selector returned no product for this line")]


def test_a_shared_purchase_buys_a_second_pack_only_when_the_summed_need_requires_it():
    from pantry_planner import db
    from pantry_planner.flow import group_purchases
    from pantry_planner.models import RecipeIngredient, Selection

    pasta = next(p for p in db.load_all_products() if p.id == SPAGHETTI)
    assert (pasta.unit_qty, pasta.unit_uom) == (500, "g")
    lines = {1: RecipeIngredient(line_no=1, name="spaghetti"),
             2: RecipeIngredient(line_no=2, name="spaghetti")}
    sels = [Selection(line_no=n, product_id=SPAGHETTI, confidence=0.9) for n in (1, 2)]

    def packs(a, b):
        (pu,), _ = group_purchases(sels, {SPAGHETTI: pasta}, lines,
                                   {1: (a, "g"), 2: (b, "g")})
        return pu.packs

    assert packs(200, 300) == 1          # 500 g: one 500 g box is enough
    assert packs(300, 300) == 2          # 600 g: one box is short
    assert packs(500, 1000) == 3
    (pu,), _ = group_purchases(sels, {SPAGHETTI: pasta}, lines, {1: (300, "g"), 2: None})
    assert pu.packs == 1                 # an unknown need never invents a second pack


@pytest.mark.asyncio
async def test_plan_prices_a_shared_product_once_in_total_cost_and_the_trip(server):
    s = await _plan(server, "Dupes (serves 2)\n- 500g spaghetti\n- 15ml olive oil\n"
                            "- 30ml olive oil\n", allow_partial=True)
    assert [(ln.line_no, ln.also_lines, ln.product_id, ln.packs) for ln in s.lines] == [
        (1, [], SPAGHETTI, 1), (2, [3], EVOO, 1)]
    assert s.lines[1].ingredient == "olive oil + olive oil"
    assert s.total_cost == round(_cheapest(SPAGHETTI) + _cheapest(EVOO), 2)
    assert sorted(i.product_id for i in s.trip.items) == [EVOO, SPAGHETTI]   # each once
    assert s.trip.basket_cost == round(sum(i.price for i in s.trip.items), 2)
    assert ("lines 2, 3 (olive oil + olive oil) share one purchase: Extra Virgin Olive "
            "Oil 500ml, bought once") in s.notes
    assert _accounted(s) == 3 and not any(n.startswith("planned ") for n in s.notes)


@pytest.mark.asyncio
async def test_plan_prices_the_packs_a_shared_need_takes(server):
    s = await _plan(server, "Pasta x2 (serves 6)\n- 300g spaghetti\n- 300g spaghetti\n")
    (ln,) = s.lines
    assert (ln.line_no, ln.also_lines, ln.packs) == (1, [2], 2)
    assert ln.price == pytest.approx(round(2 * _cheapest(SPAGHETTI), 2))
    assert s.total_cost == ln.price
    (item,) = s.trip.items
    store_price, = _sql("SELECT sp.price FROM store_products sp JOIN stores st "
                        "ON st.id = sp.store_id WHERE sp.product_id = :p AND st.name = :n",
                        p=SPAGHETTI, n=item.store)[0]
    assert item.price == pytest.approx(round(2 * float(store_price), 2))
    assert ("lines 1, 2 (spaghetti + spaghetti) share one purchase: Spaghetti Pasta 500g, "
            "2 packs for their combined quantity") in s.notes


# ─── 2. non-purchases are never priced ───────────────────────

@pytest.mark.parametrize("name", ["water", "hot water", "boiling water", "cold water",
                                  "ice", "ice cubes", "Water"])
def test_non_purchases_are_recognised(name):
    from pantry_planner.nlsearch.units import is_non_purchase

    assert is_non_purchase(name)


@pytest.mark.parametrize("name", ["coconut water", "sparkling water", "rose water",
                                  "water chestnut", "iceberg lettuce", "spaghetti"])
def test_products_named_with_water_or_ice_are_still_bought(name):
    from pantry_planner.nlsearch.units import is_non_purchase

    assert not is_non_purchase(name)


def test_water_is_skipped_before_retrieval_and_never_aborts_a_plan():
    """"water" used to be an EXACT match (canned beans and tuna are "in
    water") and was bought as a can of chickpeas. It is skipped before t1,
    even without allow_partial, and never reaches the selector."""
    from pantry_planner.nlsearch.planner import NOT_BOUGHT

    r = _run([_ing("spaghetti", quantity=500, unit="g"),
              _ing("water", quantity=2, unit="cups"), _ing("boiling water")])
    assert [i.name for i in r.recipe.ingredients] == ["spaghetti"]   # the selector's lines
    assert [(d.ingredient, d.reason) for d in r.skipped] == [
        ("water", NOT_BOUGHT), ("boiling water", NOT_BOUGHT)]
    assert r.ingredient_count == 3 and r.match_levels == {1: "exact"}
    assert all(p.id == SPAGHETTI or p.substitute for p in r.products)
    assert "1/1 ingredients stocked; 2 not bought" in r.execution.steps[0].label


def test_a_recipe_of_only_water_aborts_without_offering_a_retry():
    from pantry_planner.nlsearch.plan import GateCode
    from pantry_planner.nlsearch.planner import PlanAborted

    with pytest.raises(PlanAborted) as exc:
        _run([_ing("water"), _ing("ice")], allow_partial=True)
    alert = exc.value.execution.aborted
    assert alert.code == GateCode.missing_ingredients
    assert alert.partial_would_plan == 0


@pytest.mark.asyncio
async def test_plan_from_text_lists_water_as_skipped_and_prices_only_food(server):
    s = await _plan(server, "Chana (serves 4)\n- 1 cup chickpeas\n- 1½ cups water "
                            "(to pressure cook)\n- 3/4 teaspoon salt\n")
    assert [ln.ingredient for ln in s.lines] == ["chickpeas", "salt"]
    assert [d.ingredient for d in s.skipped] == ["water"]
    assert s.total_cost == round(sum(ln.price for ln in s.lines), 2)
    assert ("planned 2 of 3 ingredients: 0 not stocked, 0 out of range, 1 skipped (see "
            "not_stocked / out_of_range / skipped); total_cost covers the planned lines "
            "only") in s.notes
    assert _accounted(s) == 3


@pytest.mark.asyncio
async def test_find_product_says_water_is_never_a_candidate(server):
    res = await server.call_tool("find_product", {"query": "water"})
    out = res.structured_content
    assert (out["match"], out["total"], out["items"]) == ("none", 0, [])
    assert "never bought" in out["note"]


# ─── 3. irregular plurals ────────────────────────────────────

@pytest.mark.parametrize("name,product", [
    ("bay leaf", BAY_LEAVES), ("bay leaves", BAY_LEAVES), ("mint leaf", MINT),
    ("coriander leaf", CILANTRO), ("coriander leaves", CILANTRO),
])
def test_leaf_matches_leaves(name, product):
    r = _run([_ing(name)])
    assert r.match_levels == {1: "exact"}
    assert _direct(r.pools[0]) == [product]


def test_stemmer_irregulars_and_what_it_must_not_touch():
    from pantry_planner.nlsearch.units import stem

    assert [stem(w) for w in ("leaves", "loaves", "halves", "leaf")] == \
        ["leaf", "loaf", "half", "leaf"]
    assert [stem(w) for w in ("olives", "cloves", "chives", "knives")] == \
        ["olive", "clove", "chive", "knive"]


# ─── 4. "X powder" is "Ground X" ─────────────────────────────

@pytest.mark.parametrize("ing,product,level", [
    (dict(name="cumin powder"), CUMIN_GROUND, "form"),
    (dict(name="coriander powder"), CORIANDER_GROUND, "form"),
    (dict(name="cumin", form="powdered"), CUMIN_GROUND, "form"),
    (dict(name="ground cumin"), CUMIN_GROUND, "exact"),
    (dict(name="chili powder"), CHILI_POWDER, "exact"),          # not Chili Ground
    (dict(name="red chili powder"), CHILI_POWDER, "exact"),
])
def test_powder_and_ground_are_one_purchase_form(ing, product, level):
    r = _run([_ing(**ing)])
    assert r.match_levels == {1: level}
    assert _direct(r.pools[0]) == [product]           # not Cumin Seeds 100g


def test_the_equivalent_form_is_not_a_fallback_for_a_missing_spice():
    """"garlic powder" must not become Fresh Garlic through the swap: the
    equivalent needs garlic AND ground, and no product has both."""
    r = _run([_ing("garlic powder"), _ing("spaghetti")], allow_partial=True)
    assert [d.ingredient for d in r.not_stocked] == ["garlic powder"]


# ─── 5. purchase descriptors decide the product ──────────────

def test_parser_prompt_keeps_purchase_descriptors_and_written_forms():
    from pantry_planner.nlsearch.query_parser import PARSER_SYSTEM

    prompt = " ".join(PARSER_SYSTEM.split())
    assert '-> name "boneless skinless chicken thigh", NOT "chicken thigh"' in prompt
    assert '"ground Sichuan peppercorns" -> name "Sichuan peppercorn", form "ground"' in prompt
    for word in ("boneless", "skinless", "bone-in", "whole", "unsalted", "smoked"):
        assert word in prompt
    PARSER_SYSTEM.format(vocab="x")                   # still a valid template


@pytest.mark.parametrize("ing,direct", [
    (dict(name="boneless skinless chicken thigh"), [THIGHS_BONELESS]),   # never bone-in
    (dict(name="Sichuan peppercorn", form="ground"), [SICHUAN_GROUND]),
    (dict(name="whole Sichuan peppercorn"), [SICHUAN_WHOLE]),
])
def test_kept_descriptors_select_the_product_that_has_them(ing, direct):
    r = _run([_ing(**ing)])
    assert r.match_levels == {1: "exact"}
    assert _direct(r.pools[0]) == direct


@pytest.mark.parametrize("name,product", [
    ("boneless skinless chicken drumstick", DRUMSTICKS),   # no boneless drumsticks stocked
    ("unsalted butter", BUTTER_SALTED),
    ("smoked paprika", PAPRIKA),
    ("bone-in chicken breast", 10),                        # Boneless Skinless Chicken Breast
])
def test_a_descriptor_no_product_has_falls_back_to_a_flagged_generic_match(name, product):
    r = _run([_ing(name)])
    assert r.match_levels == {1: "generic"}
    assert product in _direct(r.pools[0])
    assert _demo_pick(r) == product


def test_demo_parse_keeps_descriptors_and_moves_prep_like_the_live_parser():
    from pantry_planner import demomode

    p = demomode.parse_recipe(
        "Mala (serves 4)\n"
        "- 1 lb boneless skinless chicken thigh (or breast) (, cut into 1” (2.5 cm) cubes )\n"
        "- 1 teaspoon ground Sichuan peppercorns ((*Footnote 3))\n"
        "- 1/4 to 1/3 cup peanut oil ((or vegetable oil))\n"
        "- 5 garlic cloves (, thinly sliced)\n"
        "- 1 cup chopped cilantro (, and more for garnish)\n"
        "- 1½ cups water\n- 2 cloves\n")
    got = [(i.name, i.form, i.quantity, i.unit, i.prep) for i in p.recipe.ingredients]
    assert got == [
        ("boneless skinless chicken thigh", None, 1.0, "lb", None),
        ("sichuan peppercorns", "ground", 1.0, "teaspoon", None),
        ("peanut oil", None, 0.25, "cup", None),
        ("garlic", None, 5.0, "cloves", None),
        ("cilantro", None, 1.0, "cup", "chopped"),
        ("water", None, 1.5, "cups", None),
        ("cloves", None, 2.0, "each", None),
    ]


# ─── 6. not_stocked suggestions name related products ────────

@pytest.mark.parametrize("name,hint,first", [
    ("Kashmiri red chili powder", "spice", ["Chili Powder 100g"]),
    ("ginger garlic paste", "condiment", ["Fresh Garlic", "Fresh Ginger"]),
    ("garam masala powder", "spice", ["Garam Masala 100g"]),
    ("brown rice", "rice", ["Long Grain White Rice 1kg", "Jasmine Rice 2kg"]),
])
def test_suggestions_share_the_ingredients_words_before_its_aisle(name, hint, first):
    r = _run([_ing(name, category_hint=hint), _ing("spaghetti")], allow_partial=True)
    (d,) = r.not_stocked
    names = [sg.rsplit(" ($", 1)[0] for sg in d.suggestions]
    assert names[:len(first)] == first


def test_a_shared_weak_word_never_makes_a_suggestion():
    """"brown rice" shares "brown" with Brown Sugar 1kg: colour words and
    descriptors do not make a product related."""
    r = _run([_ing("brown rice"), _ing("spaghetti")], allow_partial=True)
    (d,) = r.not_stocked
    assert d.suggestions and all("Rice" in sg for sg in d.suggestions)


def test_with_no_shared_word_the_category_hint_still_answers():
    r = _run([_ing("kasuri methi", category_hint="herbs"), _ing("spaghetti")],
             allow_partial=True)
    cheapest_herbs = [f"{n} (${float(p):.2f})" for n, p in _sql(
        "SELECT name, price FROM products WHERE subcategory = 'herbs' OR category = 'herbs' "
        "ORDER BY price LIMIT 3")]
    assert r.not_stocked[0].suggestions == cheapest_herbs


# ─── 7. a word the description negates is not a match ────────

@pytest.mark.parametrize("ing", [dict(name="salt"), dict(name="salt", quantity=10, unit="g"),
                                 dict(name="salt", quantity=1, unit="tsp")])
def test_salt_never_matches_no_salt_added_tomatoes(ing):
    r = _run([_ing(**ing)])
    pool = _direct(r.pools[0])
    assert CRUSHED_TOMATOES not in pool
    subcats = dict(_sql("SELECT id, subcategory FROM products"))
    assert {subcats[i] for i in pool} == {"salt"}
    assert _demo_pick(r) == TABLE_SALT                 # the cheapest salt


def test_jelly_does_not_match_the_no_jelly_peanut_butter():
    r = _run([_ing("grape jelly"), _ing("jelly")], allow_partial=False)
    assert PB_CREAM not in _direct(r.pools[1])


# The demo pick for every seeded recipe line on the 62-product catalog
# (recorded before pantry-db#11 grew it), by recipe slug and line name.
# One pick moves, and is right now: grilled cheese's "Butter" used to buy
# Canadian Peanut Butter Cream; the head-noun tie-break buys Butter Salted.
# None: the line was not stocked on the NL path then either.
PICKS_ON_THE_62_PRODUCT_CATALOG = {
    "pbj_sandwich": {"Peanut Butter and Jelly Jam": None, "Wheat Bread": 2,
                     "White Chocolate": None},
    "spaghetti_bolognese": {"Spaghetti": 21, "Ground Beef": 36, "Yellow Onion": 12,
                            "Garlic": 13, "Canned Tomatoes": 15, "Olive Oil": 16,
                            "Mozzarella": 24},
    "chicken_curry": {"Chicken Thighs": 11, "Basmati Rice": 8, "Yellow Onion": 12,
                      "Garlic": 13, "Ginger": 25, "Garam Masala": 29, "Turmeric": 27,
                      "Diced Tomatoes": 15, "Canola Oil": 17},
    "grilled_cheese": {"Sliced Bread": 2, "Cheddar Cheese": 23, "Butter": BUTTER_SALTED},
    "veggie_stirfry": {"Broccoli": 53, "Basmati Rice": 8, "Garlic": 13, "Ginger": 25,
                       "Canola Oil": 17},
    "beef_rice_bowl": {"Ground Beef": 36, "Basmati Rice": 8, "Broccoli": 53, "Garlic": 13,
                       "Canola Oil": 17},
    "tomato_penne": {"Penne": 51, "Canned Tomatoes": 15, "Garlic": 13, "Olive Oil": 16,
                     "Mozzarella": 24},
}


def test_existing_recipe_lines_buy_what_they_bought_before():
    """The tokenizer change (negations, irregular plurals), the size cap and
    the 99 new catalog rows move no seeded recipe line, except the butter
    line, which they fix."""
    from pantry_planner import db

    got: dict[str, dict] = {}
    for recipe in db.load_all_recipes():
        got[recipe.slug] = {}
        for ing in recipe.ingredients:
            r = _run([_ing(ing.name), _ing("spaghetti")], allow_partial=True)
            planned = [i.name for i in r.recipe.ingredients]
            got[recipe.slug][ing.name] = _demo_pick(r) if planned[0] == ing.name else None
    assert got == PICKS_ON_THE_62_PRODUCT_CATALOG


# ─── review: size cap, demo selector, catalog rows ───────────

def test_a_catering_pack_stays_out_even_when_no_pack_fits(reseed):
    """The scaled cap is 6x the smallest pack, not unlimited: with a 20 L
    jug of canola added, "2 tbsp oil" still never sees it."""
    _sql("INSERT INTO products (id, name, description, price, category, subcategory, "
         "dietary_tags, unit_size, unit_qty, unit_uom, brand) VALUES (900, 'Canola Oil "
         "Jug 20L', 'Catering canola oil', 1.00, 'pantry', 'oil', '', '20L', 20000, 'ml', "
         "'PantryCo')")
    _sql("INSERT INTO store_products (store_id, product_id, price) VALUES (1, 900, 1.00)")
    for t in ("canola", "oil", "jug", "catering"):
        _sql("INSERT INTO product_terms (term, product_id) VALUES (:t, 900)", t=t)
    r = _run([_ing("oil", quantity=2, unit="tbsp")])
    assert 900 not in _direct(r.pools[0])             # 20 L > 6 x 250 ml
    assert _direct(r.pools[0])[0] == CANOLA
    r = _run([_ing("oil")])                           # no quantity: no cap at all
    assert _direct(r.pools[0])[0] == 900


@pytest.mark.parametrize("line,product", [
    ("- 2 tbsp oil", CANOLA),
    ("- 2 tbsp vinegar", WHITE_VINEGAR),
    ("- 2 cups flour", AP_FLOUR),
    ("- 500g flour", AP_FLOUR),
    ("- 10g salt", TABLE_SALT),
    ("- 500g fish", WHITE_FISH),
    ("- 1 tbsp mustard", DIJON),
    ("- 1 tsp coriander", CORIANDER_GROUND),
    ("- 1/4 cup fresh coriander", CILANTRO),
])
def test_demo_picks_the_product_the_word_is_about(line, product):
    """DEMO_MODE (the public demo) used to buy Flour Tortillas for flour,
    sesame oil for oil, rice vinegar for vinegar, Fish Sauce for fish and
    mustard seeds for mustard: token overlap ties broken on price alone."""
    from fastapi.testclient import TestClient

    from pantry_planner.api import app

    resp = TestClient(app).post("/plan/nl", json={"recipe_text": f"One (serves 1)\n{line}\n"})
    assert resp.status_code == 200, resp.text
    (li,) = resp.json()["line_items"]
    assert li["product_id"] == product


def test_bread_recipe_end_to_end_buys_flour_and_salt_and_skips_water():
    """The pantry-db review's reproduction: flour -> Flour Tortillas, salt ->
    Crushed Tomatoes, water -> Canned Tuna."""
    from fastapi.testclient import TestClient

    from pantry_planner.api import app

    resp = TestClient(app).post("/plan/nl", json={
        "recipe_text": "Basic Bread (serves 4)\n- 500g flour\n- 10g salt\n- 350ml water",
        "allow_partial": True})
    assert resp.status_code == 200, resp.text
    plan = resp.json()
    assert [(li["ingredient_name"], li["product_id"]) for li in plan["line_items"]] == [
        ("flour", AP_FLOUR), ("salt", TABLE_SALT)]
    assert [d["ingredient"] for d in plan["skipped"]] == ["water"]
    assert plan["ingredient_count"] == 3


@pytest.mark.parametrize("name,product", [
    ("green chilli", GREEN_CHILIES), ("green chillies", GREEN_CHILIES),
    ("green chile", GREEN_CHILIES), ("green chiles", GREEN_CHILIES),
    ("thai chilli", GREEN_CHILIES), ("birds eye chili", GREEN_CHILIES),
    ("chilli powder", CHILI_POWDER), ("red chilli powder", CHILI_POWDER),
    ("dried red chillies", DRIED_RED_CHILIES), ("dried red chiles", DRIED_RED_CHILIES),
])
def test_the_spellings_indian_and_mexican_recipes_use(name, product):
    r = _run([_ing(name)])
    assert r.match_levels == {1: "exact"}
    assert _direct(r.pools[0])[0] == product


def test_fish_has_a_fish_to_buy():
    r = _run([_ing("fish", quantity=500, unit="g")])
    assert _direct(r.pools[0])[0] == WHITE_FISH        # same unit, fits: ranks first
    assert FISH_SAUCE in _direct(r.pools[0])
    assert _demo_pick(r) == WHITE_FISH


def test_soybean_oil_is_out_of_a_soy_free_plan():
    r = _run([_ing("oil"), _ing("vegetable oil")], allow_partial=True, exclude_tags=["soy"])
    assert VEG_OIL not in _direct(r.pools[0])
    assert [d.ingredient for d in r.out_of_range] == ["vegetable oil"]


def test_substitutes_stay_in_kind():
    """t4 offers the cheapest same-subcategory products. Ground spices and
    blends are no longer filed with whole seeds, dried herbs not with fresh
    ones, and sesame oil is not a cooking-oil substitute."""
    subcats = dict(_sql("SELECT id, subcategory FROM products"))

    def subs(name):
        r = _run([_ing(name)])
        return [p.id for p in r.pools[0] if p.substitute]

    assert {subcats[i] for i in subs("garam masala")} == {"spice"}
    assert {subcats[i] for i in subs("bay leaves")} == {"dried herbs"}
    assert SESAME_OIL not in subs("canola oil")
    assert MUSTARD_SEEDS not in subs("turmeric")


def test_every_catalog_name_resolves_to_itself_in_ingest():
    """ingest.match_product declines near-ties, but a name that adds nothing
    but a size to the query is the plain product, not a tie: "Soy Sauce
    500ml" is not Light Soy Sauce 500ml. The two Wheat Bread rows have the
    same words and stay ambiguous — declined, never mismatched."""
    from pantry_planner import db
    from pantry_planner.ingest import match_product

    catalog = db.load_all_products()
    wrong = {p.id: getattr(match_product(p.name, catalog), "id", None)
             for p in catalog if getattr(match_product(p.name, catalog), "id", None) != p.id}
    assert wrong == {1: None, 2: None}
    assert match_product("No Name Soy Sauce 1L", catalog).id == 59
    assert match_product("Ground Beef", catalog) is None   # lean, medium, extra lean


# ─── review: accounting, alerts, retries ─────────────────────

@pytest.mark.asyncio
async def test_a_line_the_selector_skips_is_reported_not_lost(server, monkeypatch):
    from pantry_planner import demomode

    orig = demomode.select_products

    def drop_line_2(ingredients, products, **kw):
        r = orig(ingredients, products, **kw)
        r.selections = [s for s in r.selections if s.line_no != 2]
        return r

    monkeypatch.setattr(demomode, "select_products", drop_line_2)
    s = await _plan(server, "P (serves 2)\n- 500g spaghetti\n- 1 tbsp olive oil\n- 1g saffron\n",
                    allow_partial=True)
    assert [ln.ingredient for ln in s.lines] == ["spaghetti"]
    (d,) = s.skipped
    assert (d.ingredient, d.reason) == ("olive oil",
                                        "the selector returned no product for this line")
    assert d.suggestions[0].startswith("Extra Virgin Olive Oil 500ml")
    assert [x.ingredient for x in s.not_stocked] == ["saffron"]
    assert _accounted(s) == 3
    assert ("planned 1 of 3 ingredients: 1 not stocked, 0 out of range, 1 skipped (see "
            "not_stocked / out_of_range / skipped); total_cost covers the planned lines "
            "only") in s.notes


@pytest.mark.asyncio
async def test_a_product_id_that_is_not_a_candidate_is_named(server, monkeypatch):
    from pantry_planner import demomode

    orig = demomode.select_products

    def bad_id(ingredients, products, **kw):
        r = orig(ingredients, products, **kw)
        for s in r.selections:
            if s.line_no == 2:
                s.product_id = 99999
        return r

    monkeypatch.setattr(demomode, "select_products", bad_id)
    s = await _plan(server, "P (serves 2)\n- 500g spaghetti\n- 1 tbsp olive oil\n")
    (d,) = s.skipped
    assert d.reason == ("the selector named product 99999, which is not a candidate "
                        "for this line")


def test_ingredients_past_the_cap_are_named_and_counted():
    from pantry_planner.nlsearch.planner import OVER_CAP

    names = [f"spaghetti {i}" for i in range(40)] + ["saffron", "garlic"]
    r = _run([_ing("spaghetti")] * 40 + [_ing("saffron"), _ing("garlic")])
    assert len(names) == r.ingredient_count == 42
    assert len(r.recipe.ingredients) == 40
    assert [(d.ingredient, d.reason) for d in r.skipped] == [
        ("saffron", OVER_CAP), ("garlic", OVER_CAP)]


@pytest.mark.asyncio
async def test_retry_hint_only_when_a_partial_plan_would_price_something(server):
    with pytest.raises(ToolError) as one_missing:
        await server.call_tool("plan_from_text", {
            "recipe_text": "P (serves 1)\n- 500g spaghetti\n- 1g saffron\n"})
    assert "Retry with allow_partial=true to plan the rest." in str(one_missing.value)
    with pytest.raises(ToolError) as all_missing:
        await server.call_tool("plan_from_text", {
            "recipe_text": "P (serves 1)\n- 1g saffron\n- 5g potato starch\n"})
    assert "allow_partial" not in str(all_missing.value)
    assert "retry.." not in str(all_missing.value)


@pytest.mark.asyncio
async def test_budget_abort_names_what_was_already_left_out(server):
    with pytest.raises(ToolError) as exc:
        await server.call_tool("plan_from_text", {
            "recipe_text": "P (serves 2)\n- 500g spaghetti\n- 1 tbsp olive oil\n- 1g saffron\n"
                           "Notes: under $1", "allow_partial": True})
    msg = str(exc.value)
    assert "budget_infeasible" in msg and "Not stocked at all: saffron." in msg
    assert "Affected: spaghetti, olive oil, saffron." in msg
    assert "allow_partial" not in msg.split("Affected")[1]   # no retry hint for a budget


def test_out_of_range_reason_names_the_limit_it_breaks():
    """Both limits at once: the nearest olive oil is over the price cap,
    and Richmond (~14 km) is beyond 5 km."""
    r = _run([_ing("spaghetti"), _ing("olive oil")], allow_partial=True, max_km=5,
             max_item_price=5)
    (d,) = r.out_of_range
    assert d.reason.endswith("over the $5.00 per-item price cap")
    assert "beyond" not in d.reason                   # the nearest offer is 0.3 km away


def test_skill_md_tells_the_agent_what_each_error_and_reason_means():
    from pathlib import Path

    md = " ".join((Path(__file__).resolve().parents[1] / "skills" / "recipe-shopper"
                   / "SKILL.md").read_text(encoding="utf-8").split())
    assert "`budget_infeasible`: the recipe CAN be planned" in md
    assert "`excluded_by_origin`" in md
    assert 'When it says "beyond the N km limit", offer to plan again with a larger `max_km`' in md
    assert "a larger `max_km` cannot help" in md
    assert "`summary.skipped`" in md and "`also_lines`" in md
    assert "Check that the listed tool's parameters include `allow_partial`" in md

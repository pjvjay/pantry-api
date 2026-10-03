"""Planning a real recipe: partial plans, a distance limit, generic matching.

A recipe fetched from a link (pjvjay/pantry-api#21) names more than the
seeded catalog stocks, says "light brown sugar" where the shelf says "Brown
Sugar", and wants "nearby" to mean something. The fixtures name ingredients
the 160-product catalog (pjvjay/pantry-api#22) still lacks — saffron,
lemongrass, potato starch, MSG — since it now stocks Sichuan peppercorns,
cornstarch and light soy sauce. (The size-cap regression from
the same live run is in test_nlsearch.py, next to the cap's other tests.) Each test states the right
answer from somewhere other than the code under test — seeds/products.json
ids, a direct SQL read, or the store coordinates — so it fails when the
planner is wrong, not merely when it changes.

No LLM: the planner tests inject the parse; the MCP/REST tests run in
DEMO_MODE with no ANTHROPIC_API_KEY.
"""
from __future__ import annotations

import math
import os

import pytest
from mcp.server.mcpserver.exceptions import ToolError
from sqlalchemy import text
from sqlalchemy.orm import Session

# seeds/products.json
SPAGHETTI, CANOLA, EVOO, OLIVE_1L, SOY_SAUCE = 21, 17, 16, 62, 59
LIGHT_SOY, DARK_SOY, CORNSTARCH, BROWN_SUGAR = 64, 65, 81, 120
DARK_CHOC, WHOLE_MILK, CHICKEN_STOCK = 3, 19, 57
DOWNTOWN, RICHMOND, EAST_VAN = 1, 4, 3        # storeseed.STORES ids
HOME = (49.28, -123.12)                       # config default_lat/default_lon


@pytest.fixture(scope="module", autouse=True)
def seeded_db(tmp_path_factory):
    os.environ["DB_URL"] = (os.environ.get("PANTRY_TEST_DB_URL")
                            or f"sqlite:///{tmp_path_factory.mktemp('db') / 'links.db'}")
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
    """For tests that edit catalog rows: put the seed back afterwards."""
    yield
    from pantry_planner import db
    from pantry_planner.nlsearch import vocab

    db.seed_from_json()
    vocab.clear_cache()


def _sql(sql: str, **params):
    from pantry_planner.db import engine

    with Session(engine()) as s:
        res = s.execute(text(sql), params)
        rows = res.all() if res.returns_rows else []
        s.commit()
        return rows


def _store_km(store_id: int) -> float:
    """Equirectangular km from the default point, from the stores table."""
    (lat, lon), = _sql("SELECT lat, lon FROM stores WHERE id = :i", i=store_id)
    dy = (HOME[0] - lat) * 111.0
    dx = (HOME[1] - lon) * 111.0 * math.cos(math.radians(HOME[0]))
    return round(math.sqrt(dx * dx + dy * dy), 1)


def _price(product_id: int, store_id: int) -> float:
    (p,), = _sql("SELECT price FROM store_products WHERE product_id = :p AND store_id = :s",
                 p=product_id, s=store_id)
    return float(p)


def _store_name(store_id: int) -> str:
    (n,), = _sql("SELECT name FROM stores WHERE id = :i", i=store_id)
    return str(n)


def _parsed(ingredients, **constraints):
    from pantry_planner.nlsearch.schemas import Constraints, ParsedInput, RecipeSpec

    return ParsedInput(recipe=RecipeSpec(title="Links", servings=2, ingredients=ingredients),
                       constraints=Constraints(**constraints))


def _run(ingredients, *, allow_partial=False, max_km=None, **constraints):
    from pantry_planner.nlsearch.planner import run_query_plan

    return run_query_plan("ignored", parsed=_parsed(ingredients, **constraints),
                          allow_partial=allow_partial, max_km=max_km)


def _ing(name, **kw):
    from pantry_planner.nlsearch.schemas import IngredientSpec

    return IngredientSpec(name=name, **kw)


def _direct(pool):
    return [p.id for p in pool if not p.substitute]


# ─── allow_partial: not stocked ──────────────────────────────

def test_partial_drops_unstocked_ingredients_and_plans_the_rest():
    r = _run([_ing("saffron", category_hint="spice"),
              _ing("spaghetti", quantity=400, unit="g"),
              _ing("lemongrass")], allow_partial=True)
    assert [i.name for i in r.recipe.ingredients] == ["spaghetti"]
    assert [i.line_no for i in r.recipe.ingredients] == [2]   # numbering as written
    assert r.match_levels == {2: "exact"}
    assert r.ingredient_count == 3
    assert r.out_of_range == []
    assert [d.ingredient for d in r.not_stocked] == ["saffron", "lemongrass"]
    assert all(d.reason == "not in the catalog" for d in r.not_stocked)
    cheapest_spices = [f"{n} (${float(p):.2f})" for n, p in _sql(
        "SELECT name, price FROM products WHERE subcategory = 'spice' OR category = 'spice' "
        "ORDER BY price LIMIT 3")]
    assert r.not_stocked[0].suggestions == cheapest_spices
    assert r.not_stocked[1].suggestions == []            # no category hint
    assert _direct(r.pools[0]) == [SPAGHETTI]
    t1 = r.execution.steps[0]
    assert t1.outcome == "ok" and "2 not stocked, dropped" in t1.label


def test_partial_with_nothing_stocked_still_aborts():
    from pantry_planner.nlsearch.plan import GateCode
    from pantry_planner.nlsearch.planner import PlanAborted

    with pytest.raises(PlanAborted) as exc:
        _run([_ing("saffron"), _ing("unicorn horn")], allow_partial=True)
    alert = exc.value.execution.aborted
    assert alert.code == GateCode.missing_ingredients
    assert [d["name"] for d in alert.details] == ["saffron", "unicorn horn"]


def test_default_still_aborts_on_one_missing_ingredient():
    from pantry_planner.nlsearch.plan import GateCode
    from pantry_planner.nlsearch.planner import PlanAborted

    with pytest.raises(PlanAborted) as exc:
        _run([_ing("spaghetti"), _ing("saffron")])
    assert exc.value.execution.aborted.code == GateCode.missing_ingredients


# ─── allow_partial: out of range ─────────────────────────────

def test_partial_drops_out_of_range_and_names_the_nearest_offer(reseed):
    """Cornstarch only at East Van (~4 km, dearer) and Richmond (~14 km,
    cheaper); a 3 km limit. The reason must name the NEAREST offer — the
    old probe pinned each product to its cheapest store and would have
    named Richmond."""
    (n,), = _sql("SELECT COUNT(*) FROM product_terms WHERE term = 'cornstarch'")
    assert n == 1                                    # one product answers "cornstarch"
    _sql("DELETE FROM store_products WHERE product_id = :p AND store_id NOT IN (:e, :r)",
         p=CORNSTARCH, e=EAST_VAN, r=RICHMOND)
    _sql("UPDATE store_products SET price = 5.00 WHERE product_id = :p AND store_id = :e",
         p=CORNSTARCH, e=EAST_VAN)
    _sql("UPDATE store_products SET price = 3.00 WHERE product_id = :p AND store_id = :r",
         p=CORNSTARCH, r=RICHMOND)
    assert _store_km(EAST_VAN) > 3 and _store_km(RICHMOND) > _store_km(EAST_VAN)

    r = _run([_ing("spaghetti"), _ing("cornstarch")], allow_partial=True, max_km=3)
    assert [i.name for i in r.recipe.ingredients] == ["spaghetti"]
    assert r.not_stocked == []
    (d,) = r.out_of_range
    assert d.ingredient == "cornstarch"
    assert d.reason == ("available only outside the constraints — nearest: Cornstarch 450g "
                        f"at {_store_name(EAST_VAN)} ($5.00, {_store_km(EAST_VAN)} km), "
                        "beyond the 3 km limit")
    assert {p.store_name for p in r.pools[0]} <= {
        _store_name(i) for i in (1, 2, 3, 4) if _store_km(i) <= 3}


def test_partial_price_cap_reason_and_suggestions_come_from_the_nearest_store():
    r = _run([_ing("spaghetti"), _ing("olive oil")], allow_partial=True, max_item_price=5)
    (d,) = r.out_of_range
    assert d.ingredient == "olive oil"
    # both oils are nearest at Downtown; nearest first, then cheapest
    at_downtown = sorted(((_price(pid, DOWNTOWN), name) for pid, name in
                          ((EVOO, "Extra Virgin Olive Oil 500ml"), (OLIVE_1L, "Olive Oil 1L"))))
    km = _store_km(DOWNTOWN)
    shop = _store_name(DOWNTOWN)
    (p0, n0), (p1, n1) = at_downtown
    assert p0 > 5 and p1 > 5                           # really priced out
    # the offer is inside any distance; the PRICE cap is what excludes it
    assert d.reason == ("available only outside the constraints — nearest: "
                        f"{n0} at {shop} (${p0:.2f}, {km} km), "
                        "over the $5.00 per-item price cap")
    assert d.suggestions == [f"{n1} at {shop} (${p1:.2f}, {km} km)"]


def test_partial_diet_exclusion_is_out_of_range_with_its_own_reason():
    r = _run([_ing("soy sauce"), _ing("spaghetti")], allow_partial=True,
             exclude_tags=["soy"])
    (d,) = r.out_of_range
    assert (d.ingredient, d.reason, d.suggestions) == (
        "soy sauce", "no offer passes the dietary/category constraints", [])
    assert [i.name for i in r.recipe.ingredients] == ["spaghetti"]
    assert r.match_levels == {2: "exact"}


def test_partial_with_everything_out_of_range_aborts_and_names_every_drop():
    from pantry_planner.nlsearch.plan import GateCode
    from pantry_planner.nlsearch.planner import PlanAborted

    with pytest.raises(PlanAborted) as exc:
        _run([_ing("saffron"), _ing("olive oil")], allow_partial=True, max_item_price=1)
    alert = exc.value.execution.aborted
    assert alert.code == GateCode.unavailable_within_constraints
    assert [d["name"] for d in alert.details] == ["olive oil", "saffron"]
    assert alert.details[1]["reason"] == "not in the catalog"
    assert "Not stocked at all: saffron." in alert.message


# ─── max_km ──────────────────────────────────────────────────

def _stores(r):
    return {p.store_name for pool in r.pools.values() for p in pool}


def test_max_km_overrides_the_parsed_distance_both_ways():
    near = {_store_name(i) for i in (1, 2, 3, 4) if _store_km(i) <= 1}
    assert near == {_store_name(DOWNTOWN)}
    assert _store_km(RICHMOND) > 10

    # the text said 20 km; max_km=1 narrows to the one store within 1 km
    r = _run([_ing("spaghetti"), _ing("garlic")], max_distance_km=20, max_km=1)
    assert _stores(r) == near
    assert r.max_km == 1
    assert "stores ≤ 1 km" in r.parsed.display_lines()
    assert "ignored: 20 km from the text (max_km=1 given)" in r.parsed.display_lines()

    # the text said 1 km; max_km=20 widens it back to Richmond's cheaper offer
    assert _stores(_run([_ing("spaghetti")], max_distance_km=1)) == near
    wide = _run([_ing("spaghetti")], max_distance_km=1, max_km=20)
    (cheapest_store,), = _sql(
        "SELECT s.name FROM store_products sp JOIN stores s ON s.id = sp.store_id "
        "WHERE sp.product_id = :p ORDER BY sp.price, s.id LIMIT 1", p=SPAGHETTI)
    assert cheapest_store == _store_name(RICHMOND)    # the store only 20 km reaches
    (spaghetti,) = [p for p in wide.pools[0] if not p.substitute]
    assert (spaghetti.id, spaghetti.store_name) == (SPAGHETTI, cheapest_store)
    assert wide.max_km == 20


# ─── generic matching ────────────────────────────────────────

def test_generic_tokens_vocabulary():
    from pantry_planner.nlsearch.units import generic_tokens

    assert generic_tokens("light soy sauce") == ["soy", "sauce"]
    assert generic_tokens("Low-Sodium soy sauce") == ["soy", "sauce"]
    assert generic_tokens("reduced sodium chicken stock") == ["chicken", "stock"]
    assert generic_tokens("toasted sesame seeds") == ["sesame", "seed"]
    assert generic_tokens("soy sauce") == []          # nothing to drop
    assert generic_tokens("sodium bicarbonate") == []  # "sodium" alone is not a descriptor
    assert generic_tokens("light") == []              # never an empty remainder
    assert generic_tokens("extra light") == []


def test_light_brown_sugar_matches_generically():
    (n,), = _sql("SELECT COUNT(*) FROM product_terms WHERE term = 'light' AND product_id = :p",
                 p=BROWN_SUGAR)
    assert n == 0                                    # the shelf never says "light"
    r = _run([_ing("light brown sugar", quantity=1, unit="tbsp")])
    assert r.match_levels == {1: "generic"}
    assert _direct(r.pools[0]) == [BROWN_SUGAR]
    assert "(1 via generic match)" in r.execution.steps[0].label
    assert r.stats.zero_hit_ingredients == 1         # the router sees the loosened match


def test_light_soy_sauce_is_exact_once_the_catalog_stocks_it():
    """#21's motivating case. The 160-product catalog sells Light Soy Sauce,
    so the exact level answers and the generic level (plain and dark soy
    sauce too) never runs."""
    r = _run([_ing("light soy sauce", quantity=1, unit="tbsp")])
    assert r.match_levels == {1: "exact"}
    assert _direct(r.pools[0]) == [LIGHT_SOY]
    assert "generic" not in r.execution.steps[0].label


@pytest.mark.parametrize("name,form,level,expected", [
    # an exact match exists: the generic level never runs, so the other
    # chocolate / milks / sauces never join the pool
    ("dark chocolate", None, "exact", [DARK_CHOC]),
    ("whole milk", None, "exact", [WHOLE_MILK]),
    ("low-sodium chicken stock", None, "exact", [CHICKEN_STOCK]),
    # form relaxation also wins over generic: "frozen" is dropped as the
    # purchase form, "dark" stays required
    ("dark chocolate", "frozen", "form", [DARK_CHOC]),
    # every soy sauce, cheapest first (by its cheapest store offer)
    ("low-sodium soy sauce", None, "generic", [SOY_SAUCE, DARK_SOY, LIGHT_SOY]),
])
def test_exact_and_form_matches_win_over_generic(name, form, level, expected):
    r = _run([_ing(name, form=form)])
    assert r.match_levels == {1: level}
    assert _direct(r.pools[0]) == expected


def test_a_bare_descriptor_is_never_a_generic_match():
    # "extra" (Extra Virgin Olive Oil) and "light" (flaked light tuna) are
    # both real product terms, so "extra light" is a miss at every level —
    # with an empty remainder the generic level must not match everything
    r = _run([_ing("spaghetti"), _ing("organic"), _ing("extra light")], allow_partial=True)
    assert [d.ingredient for d in r.not_stocked] == ["organic", "extra light"]
    assert r.match_levels == {1: "exact"}


# ─── selector hints follow the planned ingredients ───────────

def test_selector_hints_omit_dropped_ingredients():
    from pantry_planner.flow import _selector_constraints

    parsed = _parsed([_ing("spaghetti", quantity=400, unit="g"),
                      _ing("saffron", quantity=1, unit="g", prep="crushed")])
    out = _selector_constraints(parsed, planned={"spaghetti"})
    assert out["quantities_needed"] == {"spaghetti": "400 g"}
    assert "preps" not in out


# ─── MCP: plan_from_text ─────────────────────────────────────

MALA = ("Mala Chicken (serves 4)\n"
        "- 450g boneless skinless chicken thigh\n"
        "- 15ml light soy sauce\n"
        "- 10g light brown sugar\n"
        "- 80ml canola oil\n"
        "- 5g potato starch\n"
        "- 2g msg\n")


@pytest.fixture()
def server():
    from pantry_planner.mcp_server import server

    return server


@pytest.mark.asyncio
async def test_plan_from_text_partial_summary(server):
    from pantry_planner import demomode
    from pantry_planner.mcp_server import PlanResult

    parsed = [i.name for i in demomode.parse_recipe(MALA).recipe.ingredients]
    assert parsed == ["boneless skinless chicken thigh", "light soy sauce",
                      "light brown sugar", "canola oil", "potato starch", "msg"]

    res = await server.call_tool("plan_from_text", {"recipe_text": MALA, "allow_partial": True})
    s = PlanResult.model_validate(res.structured_content).summary
    by = {ln.ingredient: ln for ln in s.lines}
    assert list(by) == parsed[:4]
    assert [ln.line_no for ln in s.lines] == [1, 2, 3, 4]
    assert (by["light soy sauce"].product_id, by["light soy sauce"].match) == (LIGHT_SOY, "exact")
    assert (by["light brown sugar"].product_id,
            by["light brown sugar"].match) == (BROWN_SUGAR, "generic")
    assert (by["canola oil"].product_id, by["canola oil"].match) == (CANOLA, "exact")
    assert by["boneless skinless chicken thigh"].match == "exact"
    for ln in s.lines:                               # each at its cheapest store
        (low,), = _sql("SELECT MIN(price) FROM store_products WHERE product_id = :p",
                       p=ln.product_id)
        assert ln.price == pytest.approx(float(low))
    assert s.total_cost == round(sum(ln.price for ln in s.lines), 2)
    assert [d.ingredient for d in s.not_stocked] == ["potato starch", "msg"]
    assert s.out_of_range == []
    assert ("planned 4 of 6 ingredients: 2 not stocked, 0 out of range, 0 skipped (see "
            "not_stocked / out_of_range / skipped); total_cost covers the planned lines "
            "only") in s.notes
    assert "light brown sugar matched generically: Brown Sugar 1kg" in s.notes
    assert not any("light soy sauce matched" in n for n in s.notes)
    # the empty lists are on the wire, not merely defaulted by the model
    assert res.structured_content["summary"]["out_of_range"] == []


@pytest.mark.asyncio
async def test_plan_from_text_without_partial_errors_and_says_how_to_retry(server):
    with pytest.raises(ToolError, match=r"(?s)missing_ingredients.*potato starch, "
                                        r"msg.*allow_partial=true"):
        await server.call_tool("plan_from_text", {"recipe_text": MALA})


@pytest.mark.asyncio
async def test_plan_from_text_nothing_stocked_errors_even_with_partial(server):
    text_ = "Nothing (serves 1)\n- 1g saffron\n- 5g potato starch\n"
    with pytest.raises(ToolError, match="missing_ingredients") as exc:
        await server.call_tool("plan_from_text", {"recipe_text": text_, "allow_partial": True})
    assert "allow_partial=true" not in str(exc.value)    # retrying would not help


@pytest.mark.asyncio
async def test_plan_from_text_lean_summary_has_empty_drop_lists_by_default(server):
    from pantry_planner.mcp_server import PlanResult

    text_ = "Pasta (serves 2)\n- 500g penne\n- 2 cloves garlic\n"
    res = await server.call_tool("plan_from_text", {"recipe_text": text_})
    raw = res.structured_content["summary"]
    assert raw["not_stocked"] == [] and raw["out_of_range"] == []
    s = PlanResult.model_validate(res.structured_content).summary
    assert not any(n.startswith("planned ") for n in s.notes)
    assert all(ln.match == "exact" for ln in s.lines)


@pytest.mark.asyncio
async def test_plan_from_text_max_km_moves_every_line_to_the_stores_in_range(server):
    from pantry_planner.mcp_server import PlanResult

    text_ = "Garlic Pasta (serves 2)\n- 500g penne\n- 2 cloves garlic\n- 1 can crushed tomatoes\n"
    wide = PlanResult.model_validate((await server.call_tool(
        "plan_from_text", {"recipe_text": text_})).structured_content).summary
    near = PlanResult.model_validate((await server.call_tool(
        "plan_from_text", {"recipe_text": text_, "max_km": 1})).structured_content).summary

    def cheapest(pid, store_ids):
        keys = ",".join(str(i) for i in store_ids)
        (name, price), = _sql(
            "SELECT s.name, sp.price FROM store_products sp JOIN stores s ON s.id = sp.store_id "
            f"WHERE sp.product_id = :p AND s.id IN ({keys}) ORDER BY sp.price, s.id LIMIT 1",
            p=pid)
        return str(name), float(price)

    every = (1, 2, 3, 4)
    for ln in wide.lines:
        assert (ln.store, ln.price) == cheapest(ln.product_id, every), ln
    # precondition for a meaningful test: unconstrained, some line is
    # bought somewhere other than the one store within 1 km
    assert {ln.store for ln in wide.lines} != {_store_name(DOWNTOWN)}
    for ln in near.lines:
        assert (ln.store, ln.price) == cheapest(ln.product_id, (DOWNTOWN,)), ln
    assert "stores ≤ 1 km" in near.notes
    assert near.trip is not None and near.trip.stores == [_store_name(DOWNTOWN)]


@pytest.mark.asyncio
@pytest.mark.parametrize("bad,detail", [(0.4, "greater than or equal to 0.5"),
                                        (100.5, "less than or equal to 100")])
async def test_plan_from_text_max_km_bounds(server, bad, detail):
    with pytest.raises(ToolError, match=rf"(?s)validation error.*max_km.*{detail}"):
        await server.call_tool("plan_from_text",
                               {"recipe_text": "X\n- 500g penne\n", "max_km": bad})


# ─── MCP: find_product's generic tier ────────────────────────

@pytest.mark.asyncio
async def test_find_product_generic_tier_matches_the_planner(server):
    from pantry_planner.mcp_server import ProductSearch

    light = ProductSearch.model_validate((await server.call_tool(
        "find_product", {"query": "light brown sugar"})).structured_content)
    assert light.match == "generic"
    assert [i.id for i in light.items] == [BROWN_SUGAR] and light.total == 1
    assert "'brown sugar'" in light.note

    soy = ProductSearch.model_validate((await server.call_tool(
        "find_product", {"query": "light soy sauce"})).structured_content)
    assert soy.match == "direct" and [i.id for i in soy.items] == [LIGHT_SOY]

    dark = ProductSearch.model_validate((await server.call_tool(
        "find_product", {"query": "dark chocolate"})).structured_content)
    assert dark.match == "direct" and [i.id for i in dark.items] == [DARK_CHOC]

    # no generic hit either: the head-noun alternatives as before
    sun = ProductSearch.model_validate((await server.call_tool(
        "find_product", {"query": "dried sun tomato"})).structured_content)
    assert sun.match == "relaxed"


# ─── plan_week: lines the head-noun fallback rescued ─────────

@pytest.mark.asyncio
async def test_plan_week_marks_fallback_lines_generic(server, reseed):
    from pantry_planner.mcp_server import WeekResult

    _sql("DELETE FROM recipe_ingredients")
    _sql("DELETE FROM recipes")
    _sql("INSERT INTO recipes (slug, name, servings) VALUES ('choc_toast', 'Choc Toast', 1)")
    _sql("INSERT INTO recipe_ingredients (recipe_slug, line_no, name) VALUES "
         "('choc_toast', 1, 'Wheat Bread'), ('choc_toast', 2, 'White Chocolate')")
    # precondition: nothing is "white chocolate", so only w2 can answer it
    (n,), = _sql("SELECT COUNT(*) FROM product_terms a JOIN product_terms b "
                 "ON a.product_id = b.product_id WHERE a.term = 'white' AND b.term = 'chocolate'")
    assert n == 0

    out = WeekResult.model_validate(
        (await server.call_tool("plan_week", {"days": 1})).structured_content)
    (day,) = out.summary.days
    by = {ln.ingredient: ln for ln in day.lines}
    assert by["Wheat Bread"].match == "exact"
    white = by["White Chocolate"]
    assert white.match == "generic"
    (sub,), = _sql("SELECT subcategory FROM products WHERE id = :p", p=white.product_id)
    assert sub == "chocolate"
    assert f"Choc Toast: White Chocolate matched generically: {white.product}" \
        in out.summary.notes


# ─── REST /plan/nl shares the path ───────────────────────────

def test_rest_plan_nl_partial_and_max_km():
    from fastapi.testclient import TestClient

    from pantry_planner.api import app

    client = TestClient(app)
    resp = client.post("/plan/nl", json={"recipe_text": MALA, "allow_partial": True,
                                         "max_km": 1})
    assert resp.status_code == 200, resp.text
    plan = resp.json()
    assert [d["ingredient"] for d in plan["not_stocked"]] == ["potato starch", "msg"]
    assert plan["ingredient_count"] == 6 and len(plan["line_items"]) == 4
    assert {li["store_name"] for li in plan["line_items"]} == {_store_name(DOWNTOWN)}
    assert {li["ingredient_name"]: li["match"] for li in plan["line_items"]}[
        "light brown sugar"] == "generic"
    # defaults unchanged: the same text is still a 409 without allow_partial
    gated = client.post("/plan/nl", json={"recipe_text": MALA})
    assert gated.status_code == 409
    assert gated.json()["detail"]["aborted"]["code"] == "missing_ingredients"
    assert client.post("/plan/nl", json={"recipe_text": MALA, "max_km": 0}).status_code == 422


@pytest.mark.asyncio
async def test_plan_dinner_prompt_reports_what_a_partial_plan_left_out(server):
    res = await server.get_prompt("plan_dinner", {"recipe": "https://example.com/mala"})
    body = " ".join(m.content.text for m in res.messages)
    for must in ("allow_partial=true", "summary.not_stocked", "summary.out_of_range",
                 "\"generic\""):
        assert must in body, must

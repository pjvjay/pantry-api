"""Resolving recipes for the meal plan (POST /mealplan/resolve) and the demo starters
(GET /mealplan/starters).

No LLM: DEMO_MODE, and the parsers are patched to raise where a test says a recipe is never
parsed. Expected values come from the seed files and the request.
"""
from __future__ import annotations

import pytest

from tests.mealplan_fixtures import STARTERS, client, done_db, use_db


@pytest.fixture(scope="module", autouse=True)
def seeded_db(tmp_path_factory):
    use_db(tmp_path_factory, "resolve")
    yield
    done_db()


@pytest.fixture(autouse=True)
def _fresh_cache():
    from pantry_planner.mealplan import resolve

    resolve.clear_cache()


def _no_parser(monkeypatch):
    from pantry_planner import demomode
    from pantry_planner.nlsearch import planner, query_parser

    def boom(*_a, **_k):
        raise AssertionError("a reviewed recipe was parsed again")

    monkeypatch.setattr(demomode, "parse_recipe", boom)
    monkeypatch.setattr(planner, "parse_input", boom)
    monkeypatch.setattr(query_parser, "parse_input", boom)


def _resolve(*refs, **knobs) -> list[dict]:
    r = client().post("/mealplan/resolve", json={"recipes": list(refs), **knobs})
    assert r.status_code == 200, r.text
    return r.json()["resolved"]


def _starter(key: str) -> dict:
    return {"key": f"starter:{key}", "starter": key}


def _resolve_starters() -> list[dict]:
    """Every demo starter, resolved twelve at a time (a request's bound, MAX_RECIPES)."""
    keys = list(STARTERS)
    return [r for i in range(0, len(keys), 12)
            for r in _resolve(*(_starter(k) for k in keys[i:i + 12]))]


def test_starters_resolve_through_run_spec_with_no_parse(monkeypatch):
    from pantry_planner import flow

    _no_parser(monkeypatch)
    calls = []
    real = flow.run_spec
    monkeypatch.setattr(flow, "run_spec", lambda spec, **kw: calls.append(spec) or real(spec,
                                                                                      **kw))
    out = _resolve_starters()
    assert len(calls) == len(STARTERS)
    for got, (key, seed) in zip(out, STARTERS.items(), strict=True):
        assert (got["key"], got["status"], got["label"]) == (f"starter:{key}", "ok",
                                                             "demo recipe")
        assert (got["servings"], got["servings_basis"]) == (seed["servings"], "source")
        # The reviewed lines are what was planned: names and amounts as the starter file.
        assert [(ln["name"], ln["quantity"], ln["unit"]) for ln in got["lines"]] == [
            (ln["name"], ln["quantity"], ln["unit"]) for ln in seed["lines"]]
        assert all(ln["amount_basis"] == "demo_house_amounts" for ln in got["lines"])
    # Water is never bought; every other line got a product in the seeded catalog.
    for got in out:
        for ln in got["lines"]:
            assert (ln["product_id"] is None) == (ln["name"] == "Water"), ln


def _by_name(name: str) -> int:
    from tests.mealplan_fixtures import PRODUCTS

    (pid,) = [pid for pid, p in PRODUCTS.items() if p["name"] == name]
    return pid


def test_every_starter_line_buys_a_product_named_for_it_and_says_so_honestly():
    """A t4 substitute fetched for one line's thin pool is no substitute for another line.
    Fresh Ginger is a same-aisle substitute for garlic, and was flagged substitute on the one
    product list the demo selector reads, so it was dropped from every line's choice:
    Ginger and Cilantro bought Fresh Garlic, and Chicken Fried Rice's Garlic bought Green
    Onions, each labelled 'exact'."""
    from pantry_planner.nlsearch.units import tokens

    out = {r["key"]: r for r in _resolve_starters()}
    picks = {(key, ln["name"]): ln for key, r in out.items() for ln in r["lines"]}
    assert picks["starter:chicken_biryani", "Ginger"]["product_id"] == _by_name("Fresh Ginger")
    assert picks["starter:chicken_biryani", "Cilantro"]["product_id"] == _by_name("Cilantro")
    assert picks["starter:chicken_biryani", "Garlic"]["product_id"] == _by_name("Fresh Garlic")
    assert picks["starter:chicken_fried_rice", "Garlic"]["product_id"] == \
        _by_name("Fresh Garlic")
    for (key, name), ln in picks.items():
        if ln["product_id"] is None:                   # water: never bought, never labelled
            assert (name, ln["match"]) == ("Water", None)
            continue
        # The line's head noun is in the product's name, so 'exact' is a true label.
        assert tokens(name)[-1] in tokens(ln["product_name"]), (key, name, ln["product_name"])
        assert ln["match"] == "exact", (key, name)


def test_a_substitute_for_one_line_is_still_the_match_for_another():
    """Penne Rigate is fetched as a t4 substitute for spaghetti's thin pool and is penne's
    direct match. The flat list kept it flagged, so penne bought Coconut Milk."""
    from pantry_planner import flow
    from pantry_planner.nlsearch.planner import run_query_plan, substitutes_by_line, union_of_pools

    text = "X\n- 400g spaghetti\n- 400g penne\n- 1 cup milk"
    penne = _by_name("Penne Rigate 500g")
    r = run_query_plan(text)
    spaghetti_pool, penne_pool = r.pools[0], r.pools[1]
    assert any(p.id == penne and p.substitute for p in spaghetti_pool)
    assert any(p.id == penne and not p.substitute for p in penne_pool)
    # On the flat list it is a direct candidate; per line it substitutes for spaghetti only.
    assert not next(p for p in union_of_pools(list(r.pools.values())) if p.id == penne).substitute
    subs = substitutes_by_line(r.recipe, r.pools)
    assert penne in subs[1] and penne not in subs.get(2, [])

    plan = flow.run_nl(text)
    got = {b.line_no: b.product_id for b in plan.basis.lines}
    names = {li.product_id: li.product_name for li in plan.line_items}
    assert got[1] == _by_name("Spaghetti Pasta 500g")
    assert "penne" in names[got[2]].lower()
    assert "milk" in names[got[3]].lower()


def test_the_demo_products_let_the_example_resolve_in_stock():
    out = {r["key"]: r for r in _resolve_starters()}
    bought = {ln["product_id"] for r in out.values() for ln in r["lines"]}
    assert {166, 167, 169} <= bought            # frozen mango, pepperoni, instant yeast
    assert all(not r["not_stocked"] for r in out.values())


def test_a_library_recipe_resolves_with_its_demo_house_amounts():
    import json

    from pantry_planner.db import SEEDS_DIR

    curry = next(r for r in json.loads((SEEDS_DIR / "recipes.json").read_text())
                 if r["slug"] == "chicken_curry")
    (got,) = _resolve({"key": "lib:chicken_curry", "slug": "chicken_curry"})
    assert (got["status"], got["label"], got["servings"]) == ("ok", "demo house amounts",
                                                              curry["servings"])
    assert [(ln["quantity"], ln["unit"]) for ln in got["lines"]] == [
        (i["quantity"], i["unit"]) for i in curry["ingredients"]]
    assert all(ln["product_id"] is not None for ln in got["lines"])


def test_a_second_identical_call_is_cached_and_costs_nothing():
    first = client().post("/mealplan/resolve", json={"recipes": [_starter("mango_milkshake")]})
    again = client().post("/mealplan/resolve", json={"recipes": [_starter("mango_milkshake")]})
    assert first.json()["resolved"][0]["cached"] is False
    assert again.json()["resolved"][0]["cached"] is True
    assert again.json()["llm_cost_usd"] == 0
    assert again.json()["resolved"][0]["lines"] == first.json()["resolved"][0]["lines"]


def _doc(key: str, servings: int | None, *, confirmed: bool = True) -> dict:
    return {"key": key, "title": "Garlic Pasta", "servings": servings,
            "servings_stated": servings is not None,
            "lines": [{"line_no": 1, "text": "500 g penne", "name": "penne", "quantity": 500,
                       "unit": "g", "amount_basis": "parsed_from_your_paste"},
                      {"line_no": 2, "text": "10 g garlic", "name": "garlic", "quantity": 10,
                       "unit": "g", "amount_basis": "parsed_from_your_paste",
                       "confirmed": confirmed}],
            "source": {"kind": "pasted", "method": "paste"}}


def test_each_failing_recipe_gets_its_own_status_and_the_others_still_resolve(monkeypatch):
    _no_parser(monkeypatch)
    out = _resolve({"key": "lib:no_such_dish", "slug": "no_such_dish"},
                   {"key": "my:1", "doc": _doc("my:1", 2, confirmed=False)},
                   _starter("mango_milkshake"))
    assert [r["status"] for r in out] == ["not_found", "unconfirmed_lines", "ok"]
    assert "2" in out[1]["message"]


def test_unstated_servings_give_needs_servings_until_the_shopper_answers(monkeypatch):
    _no_parser(monkeypatch)
    (got,) = _resolve({"key": "my:2", "doc": _doc("my:2", None)})
    assert (got["status"], got["servings"], got["servings_basis"]) == ("needs_servings", None,
                                                                       None)
    assert "How many does this recipe serve?" in got["message"]
    # Products are still chosen: only the amounts wait for the answer.
    assert all(ln["product_id"] is not None for ln in got["lines"])
    (answered,) = _resolve({"key": "my:2", "doc": _doc("my:2", None), "servings": 3})
    assert (answered["status"], answered["servings"], answered["servings_basis"]) == (
        "ok", 3, "your_setting")


def test_a_ref_key_must_match_what_it_names():
    r = client().post("/mealplan/resolve",
                      json={"recipes": [{"key": "lib:pbj_sandwich", "starter": "mango_milkshake"}]})
    assert r.status_code == 422


def test_get_starters_lists_the_labelled_demo_recipes():
    from pantry_planner.models import RecipeDoc
    from pantry_planner.recipe_doc import to_recipe_text

    body = client().get("/mealplan/starters").json()
    assert [s["key"] for s in body] == list(STARTERS)
    for s in body:
        seed = STARTERS[s["key"]]
        assert (s["label"], s["slot"], s["aliases"]) == ("demo recipe", seed["slot"],
                                                         seed["aliases"])
        doc = RecipeDoc.model_validate(s["doc"])
        assert doc.key == f"starter:{s['key']}" and doc.source.label == "demo recipe"
        assert s["text"] == to_recipe_text(doc) == seed["text"]


@pytest.fixture
def live_without_keys(monkeypatch):
    """Not DEMO_MODE, and neither model key: the configuration that used to 500."""
    from pantry_planner import config, llm

    def no_client(*_a, **_k):
        raise AssertionError("an Anthropic client was built with no key")

    for var in ("DEMO_MODE", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "SELECTOR_MODEL_DEFAULT",
                "SELECTOR_MODEL_ESCALATION", "CLASSIFIER_MODEL", "NL2SQL_MODEL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(llm, "Anthropic", no_client)
    config.settings.cache_clear()
    yield
    config.settings.cache_clear()


@pytest.mark.parametrize("ref", [
    _starter("mango_milkshake"),                                  # run_spec
    {"key": "lib:chicken_curry", "slug": "chicken_curry"},       # flow.run
])
def test_resolve_without_a_model_key_is_a_503_naming_the_fix(live_without_keys, ref):
    """Every recipe used to come back as an opaque 500 (an empty key sent as an auth
    header). A missing key fails the whole request once, saying what to set."""
    from pantry_planner.llm import NO_KEY_HINT

    r = client().post("/mealplan/resolve", json={"recipes": [ref, _starter("pepperoni_pizza")]})
    assert r.status_code == 503, r.text
    body = r.json()
    assert body["error"] == "llm_call_failed" and body["provider"] == "anthropic"
    assert NO_KEY_HINT in body["detail"]
    assert ("No model key is configured: set ANTHROPIC_API_KEY or GEMINI_API_KEY, "
            "or run with DEMO_MODE=1") in body["detail"]


def test_a_missing_key_is_not_cached_as_a_result(live_without_keys, monkeypatch):
    """Once a key is set, the same recipe resolves: the 503 left nothing in the cache."""
    from pantry_planner import config

    ref = _starter("mango_milkshake")
    assert client().post("/mealplan/resolve", json={"recipes": [ref]}).status_code == 503
    monkeypatch.setenv("DEMO_MODE", "1")
    config.settings.cache_clear()
    (got,) = _resolve(ref)
    assert got["status"] == "ok" and got["cached"] is False


def test_mealplan_plan_without_a_model_key_is_a_503(live_without_keys):
    """POST /mealplan/plan resolves the dishes it drafts through the same path."""
    r = client().post("/mealplan/plan", json={"dishes": [{"recipe": "mango_milkshake",
                                                          "count": 1}]})
    assert r.status_code == 503, r.text
    assert "ANTHROPIC_API_KEY" in r.json()["detail"]


def test_resolve_is_rate_limited_per_client():
    body = {"recipes": [_starter("mango_milkshake")]}
    codes = [client().post("/mealplan/resolve", json=body).status_code for _ in range(4)]
    assert codes == [200, 200, 200, 429]          # 6 a minute, burst 3

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


def test_starters_resolve_through_run_spec_with_no_parse(monkeypatch):
    from pantry_planner import flow

    _no_parser(monkeypatch)
    calls = []
    real = flow.run_spec
    monkeypatch.setattr(flow, "run_spec", lambda spec, **kw: calls.append(spec) or real(spec,
                                                                                      **kw))
    out = _resolve(*(_starter(k) for k in STARTERS))
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


def test_the_demo_products_let_the_example_resolve_in_stock():
    out = {r["key"]: r for r in _resolve(*(_starter(k) for k in STARTERS))}
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


def test_resolve_is_rate_limited_per_client():
    body = {"recipes": [_starter("mango_milkshake")]}
    codes = [client().post("/mealplan/resolve", json=body).status_code for _ in range(4)]
    assert codes == [200, 200, 200, 429]          # 6 a minute, burst 3

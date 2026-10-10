"""The meal plan wizard's Meals stage: POST /mealplan/catalog, on both engines (SQLite here,
Postgres with PANTRY_TEST_DB_URL).

No LLM, in demo mode or live: the catalog runs live too with every model boundary patched to
raise. Expected values come from the seed files, the offers read from the database directly and
the resolve in demo mode, never from the code under test.
"""
from __future__ import annotations

import json

import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session

from pantry_planner.db import SEEDS_DIR
from tests.mealplan_fixtures import (
    PRODUCTS,
    STARTERS,
    client,
    done_db,
    use_db,
)

SEED = json.loads((SEEDS_DIR / "mealplan_starters.json").read_text(encoding="utf-8"))
LIBRARY = {r["slug"]: r for r in json.loads((SEEDS_DIR / "recipes.json").read_text())}
SLOTS = ("breakfast", "lunch", "dinner", "snack")
TUNA, CHEDDAR = 48, 23
WHEAT_BREAD, OTHER_WHEAT_BREAD = 2, 1     # Wheat Bread 2, the resolve's pick, and Wheat Bread 1
FAR_STORE = 4        # MegaSave Richmond, about 14 km from the server's reference point
TUNA_MELT = "starter:tuna_melt"


@pytest.fixture(scope="module", autouse=True)
def seeded_db(tmp_path_factory):
    use_db(tmp_path_factory, "wizard")
    yield
    done_db()


@pytest.fixture()
def reseed():
    """For tests that change offers or evidence: the seed is restored, and the catalog's and
    the resolve's caches forgotten, afterwards."""
    yield
    from pantry_planner import db
    from pantry_planner.db import ProductOriginEvidenceRow
    from pantry_planner.mealplan import catalog, resolve

    with Session(db.engine()) as s:
        s.query(ProductOriginEvidenceRow).delete()
        s.commit()
    db.seed_from_json()
    catalog.clear_cache()
    resolve.clear_cache()


def _sql(stmt: str, **params) -> list[dict]:
    from pantry_planner import db

    with Session(db.engine()) as s:
        rows = s.execute(text(stmt), params)
        out = [dict(r) for r in rows.mappings()] if rows.returns_rows else []
        s.commit()
    return out


def _us_evidence(*pids: int) -> None:
    from pantry_planner import db

    db.save_origin_evidence([dict(
        product_id=pid, source="label-photo", source_ref=f"{pid}.jpg", claim_type="made-in",
        verbatim="Made in United States", ingredient_origin="", manufactured_in="United States",
        confidence="high", importer_only=False, note="", observed_at="") for pid in pids])


def _changed() -> None:
    """Evidence or offers changed: what the catalog and the resolve cached is stale."""
    from pantry_planner.mealplan import catalog, resolve

    catalog.clear_cache()
    resolve.clear_cache()


def cat(**body) -> dict:
    r = client().post("/mealplan/catalog", json=body)
    assert r.status_code == 200, r.text
    return r.json()


def by_key(body: dict) -> dict[str, dict]:
    return {r["key"]: r for r in body["recipes"]}


def _meta() -> dict[str, dict]:
    out = {s["doc_key"]: {**{k: s[k] for k in ("slot", "diets", "contains", "title",
                                                 "servings")},
                          "ref": {"key": s["doc_key"], "starter": s["key"]}}
           for s in SEED["starters"]}
    for slug, m in SEED["library_meta"].items():
        out[f"lib:{slug}"] = {**m, "title": LIBRARY[slug]["name"],
                              "servings": LIBRARY[slug]["servings"],
                              "ref": {"key": f"lib:{slug}", "slug": slug}}
    return out


def _resolved(key: str, **knobs) -> dict[int, int]:
    """line_no -> product id, as resolve_one buys it in demo mode."""
    from pantry_planner.mealplan.models import RecipeRef
    from pantry_planner.mealplan.resolve import resolve_one

    kind, _, name = key.partition(":")
    ref = RecipeRef(key=key, **({"starter": name} if kind == "starter" else {"slug": name}))
    return {ln.line_no: ln.product_id for ln in resolve_one(ref, **knobs).lines
            if ln.product_id is not None}


def _lines(key: str) -> list[dict]:
    kind, _, name = key.partition(":")
    if kind == "starter":
        return STARTERS[name]["lines"]
    return [{"line_no": i, **ing} for i, ing in enumerate(LIBRARY[name]["ingredients"], 1)]


def _cheapest(pid: int) -> float:
    """The product's cheapest offer: every store is in range of the reference point."""
    return min(r["price"] for r in _sql(
        "SELECT price FROM store_products WHERE product_id = :p", p=pid))


# ─── The catalog ─────────────────────────────────────────────

def test_the_catalog_offers_the_library_and_every_starter_in_its_slot():
    body = cat()
    got, meta = by_key(body), _meta()
    assert set(got) == set(meta)
    for key, m in meta.items():
        r = got[key]
        assert (r["slot"], r["diets"], r["contains"], r["title"], r["servings"]) == (
            m["slot"], m["diets"], m["contains"], m["title"], m["servings"]), key
        assert {k: v for k, v in r["ref"].items() if v is not None} == m["ref"]
    assert [SLOTS.index(r["slot"]) for r in body["recipes"]] == sorted(
        SLOTS.index(r["slot"]) for r in body["recipes"])
    count = {s: sum(1 for r in body["recipes"] if r["slot"] == s) for s in SLOTS}
    assert count == {"breakfast": 5, "lunch": 5, "dinner": 9, "snack": 1}
    only = cat(slots=["breakfast", "lunch"])["recipes"]
    assert {r["slot"] for r in only} == {"breakfast", "lunch"} and len(only) == 10
    assert body["v"] == 1 and "whole packs" in body["est_basis"]


def test_the_catalog_calls_no_model_in_demo_mode_or_live(monkeypatch):
    from pantry_planner import config, flow, llm, selector
    from pantry_planner.mealplan import catalog
    from pantry_planner.nlsearch import planner, query_parser
    from pantry_planner.router import classifier

    def boom(*_a, **_k):
        raise AssertionError("the catalog called a model")

    for mod, name in ((llm, "forced_tool_call"), (selector, "forced_tool_call"),
                      (classifier, "forced_tool_call"), (query_parser, "forced_tool_call"),
                      (selector, "call_selector"), (flow, "call_selector"),
                      (query_parser, "parse_input"), (planner, "parse_input")):
        monkeypatch.setattr(mod, name, boom)
    catalog.clear_cache()
    demo = cat(household_servings=3)
    catalog.clear_cache()
    config.set_runtime_overrides(demo_mode=False)
    try:
        assert not config.settings().demo_mode
        live = cat(household_servings=3)
    finally:
        config.clear_runtime_overrides()
    assert demo["recipes"] == live["recipes"]


def test_each_line_buys_what_the_resolve_buys_in_demo_mode():
    from pantry_planner.mealplan import catalog
    from pantry_planner.mealplan.models import PlanSettings, RecipeRef
    from pantry_planner.mealplan.resolve import recipe_doc

    for key, m in _meta().items():
        ref = RecipeRef(**m["ref"])
        assert catalog.picks(ref, recipe_doc(ref), PlanSettings()).by_line == _resolved(key)


@pytest.mark.parametrize("household", [1, 3])
def test_an_estimate_is_each_lines_share_of_its_cheapest_pack_price(household):
    from pantry_planner.nlsearch.units import is_non_purchase, normalize_quantity

    got = by_key(cat(household_servings=household))
    for key, m in _meta().items():
        bought = _resolved(key)
        total, priced, lines = 0.0, 0, 0
        for ln in _lines(key):
            if is_non_purchase(ln["name"]):
                continue
            lines += 1
            p = PRODUCTS[bought[ln["line_no"]]]
            qty, uom = normalize_quantity(ln["quantity"], ln["unit"])
            assert uom == p["unit_uom"], (key, ln["name"])
            total += qty * household / m["servings"] / p["unit_qty"] * _cheapest(p["id"])
            priced += 1
        r = got[key]
        assert (r["lines_total"], r["lines_priced"], r["unavailable"]) == (lines, priced, [])
        assert r["est_cost_per_meal"] == pytest.approx(total, abs=0.006), key


def test_diet_and_avoid_say_what_does_not_fit_and_why():
    for diet, avoid in (("vegetarian", ["gluten"]), ("vegan", []), ("pescatarian", ["egg"]),
                        ("any", ["dairy", "nuts"]), ("any", [])):
        got = by_key(cat(diet=diet, avoid=avoid))
        for key, m in _meta().items():
            fits = ((diet == "any" or diet in m["diets"])
                    and not set(avoid) & set(m["contains"]))
            assert got[key]["fits"] is fits, (diet, avoid, key)
            assert (got[key]["why_not"] is None) == fits
    veg = by_key(cat(diet="vegetarian", avoid=["gluten"]))
    assert veg[TUNA_MELT]["why_not"] == "contains fish and gluten"
    assert veg["lib:chicken_curry"]["why_not"] == "contains meat"
    assert veg["starter:tomato_pilaf"]["fits"] is True
    vegan = by_key(cat(diet="vegan"))
    assert vegan["starter:scrambled_eggs_on_toast"]["why_not"] == "contains dairy and egg"
    assert vegan["starter:yogurt_fruit_bowl"]["why_not"] == "contains dairy"
    # the diet's own tags first, then what the shopper avoids, in the order given
    assert by_key(cat(diet="pescatarian", avoid=["soy", "gluten"]))[
        "starter:chicken_fried_rice"]["why_not"] == "contains meat, soy and gluten"


def test_an_excluded_product_is_never_picked_or_priced(reseed):
    """Canned Tuna is the only tuna, so excluding it leaves the line unavailable, as the
    resolve leaves it out; the wheat bread the resolve picks has another wheat bread to take
    its place, priced instead."""
    _us_evidence(TUNA, WHEAT_BREAD)
    _changed()
    plain = by_key(cat())
    assert plain[TUNA_MELT]["unavailable"] == [] and plain[TUNA_MELT]["lines_priced"] == 4
    rules = {"exclude_origin": ["United States"]}
    excl = by_key(cat(settings=rules))
    melt = excl[TUNA_MELT]
    assert melt["unavailable"] == ["Canned Tuna"]
    assert (melt["lines_total"], melt["lines_priced"]) == (4, 3)
    picks = _resolved(TUNA_MELT, exclude=["United States"])
    assert TUNA not in picks.values() and WHEAT_BREAD not in picks.values()
    assert picks[3] == OTHER_WHEAT_BREAD
    by_no = {ln["line_no"]: ln for ln in STARTERS["tuna_melt"]["lines"]}
    expected = sum(by_no[n]["quantity"] / PRODUCTS[pid]["unit_qty"] * _cheapest(pid)
                   for n, pid in picks.items())
    assert melt["est_cost_per_meal"] == pytest.approx(expected, abs=0.006)
    # Grilled cheese (the library path) buys another bread too.
    assert excl["lib:grilled_cheese"]["unavailable"] == []
    assert WHEAT_BREAD not in _resolved("lib:grilled_cheese", exclude=["United States"]).values()
    # The recipe's diets are the file's, whatever the exclusion picks.
    assert (melt["diets"], melt["contains"]) == (["pescatarian"], ["dairy", "fish", "gluten"])


def test_an_ingredient_no_store_in_range_sells_is_unavailable(reseed):
    _sql("DELETE FROM store_products WHERE product_id IN (:a, :b) AND store_id <> :far",
         a=TUNA, b=CHEDDAR, far=FAR_STORE)
    _changed()
    near = by_key(cat(settings={"max_km": 5}))
    assert near[TUNA_MELT]["unavailable"] == ["Canned Tuna", "Cheddar Cheese"]
    assert near[TUNA_MELT]["lines_priced"] == 2
    # the library path picks from the whole catalog, then finds no offer in range
    assert near["lib:grilled_cheese"]["unavailable"] == ["Cheddar Cheese"]
    assert near["lib:grilled_cheese"]["lines_priced"] == 2
    anywhere = by_key(cat())
    assert anywhere[TUNA_MELT]["unavailable"] == [] == anywhere["lib:grilled_cheese"][
        "unavailable"]


def _doc(key: str, title: str, lines: list[tuple[str, float, str]], servings: int | None = 2,
         **line_extra) -> dict:
    return {"key": key, "doc": {
        "key": key, "title": title, "servings": servings,
        "servings_stated": servings is not None,
        "lines": [{"line_no": i, "text": f"{q:g} {u} {n}", "name": n, "quantity": q, "unit": u,
                   "amount_basis": "parsed_from_your_paste", **line_extra}
                  for i, (n, q, u) in enumerate(lines, start=1)],
        "source": {"kind": "pasted", "method": "paste"}}}


def test_the_shoppers_own_recipes_are_classified_from_their_products():
    bowl = _doc("my:bowl", "Tofu Rice Bowl", [("Extra Firm Tofu", 200, "g"),
                                              ("Jasmine Rice", 150, "g"),
                                              ("Soy Sauce", 15, "ml")])
    stew = _doc("my:stew", "Mystery Stew", [("Unobtainium", 5, "g"), ("Yellow Onion", 100, "g")])
    video = _doc("my:video", "Video Soup", [("Carrots", 100, "g")], confirmed=False)
    got = by_key(cat(diet="vegan", recipes=[bowl, stew, video, bowl]))
    assert [k for k in got if k.startswith("my:")] == ["my:stew", "my:bowl", "my:video"]
    from pantry_planner.mealplan.models import RecipeRef
    from pantry_planner.mealplan.resolve import resolve_one

    resolved = resolve_one(RecipeRef.model_validate(bowl))
    tags = {t for ln in resolved.lines
            for t in PRODUCTS[ln.product_id]["dietary_tags"].split(",") if t}
    assert tags == {"soy", "gluten"}
    b = got["my:bowl"]
    assert (b["slot"], b["diets"], b["contains"], b["fits"], b["why_not"]) == (
        "dinner", ["vegan", "vegetarian", "pescatarian"], ["gluten", "soy"], True, None)
    assert b["lines_priced"] == 3 and b["est_cost_per_meal"] is not None
    for key in ("my:stew", "my:video"):
        r = got[key]
        assert (r["diets"], r["contains"], r["fits"]) == ([], [], False), key
        assert r["why_not"] == ("its diet is not known: not every ingredient has a product "
                                "to check")
    assert got["my:stew"]["unavailable"] == ["Unobtainium"]
    assert got["my:video"]["unavailable"] == [] and got["my:video"]["lines_priced"] == 0
    # with no diet and nothing to avoid an unknown recipe is offered
    plain = by_key(cat(recipes=[stew]))
    assert (plain["my:stew"]["fits"], plain["my:stew"]["why_not"]) == (True, None)


def test_the_catalog_is_bounded():
    for bad in ({"diet": "keto"}, {"avoid": ["meat"]}, {"household_servings": 0},
                {"household_servings": 21}, {"slots": []}, {"slots": ["brunch"]},
                {"settings": {"exclude_origin": ["Atlantis"]}},
                {"recipes": [_doc(f"my:{i}", "X", [("Salt", 1, "g")]) for i in range(25)]}):
        assert client().post("/mealplan/catalog", json=bad).status_code == 422, bad
    r = client().post("/mealplan/catalog", json={"recipes": [
        {"key": "starter:nope", "starter": "nope"}]})
    assert r.status_code == 422 and r.json()["detail"]["error"] == "unknown_recipe_key"
    big = {"recipes": [], "padding": "x" * 300_000}
    assert client().post("/mealplan/catalog", json=big).status_code == 413


def test_the_catalog_has_its_own_sixty_a_minute_bucket(monkeypatch):
    from pantry_planner import limits

    assert limits.LIMITS["/mealplan/catalog"] == limits.Limit(per_minute=60, burst=60)
    monkeypatch.setitem(limits.LIMITS, "/mealplan/catalog", limits.Limit(per_minute=60,
                                                                         burst=1))
    assert client().post("/mealplan/catalog", json={}).status_code == 200
    r = client().post("/mealplan/catalog", json={})
    assert r.status_code == 429 and r.headers["Retry-After"]

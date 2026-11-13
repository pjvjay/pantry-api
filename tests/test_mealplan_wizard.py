"""The meal plan wizard's Meals and Ingredients stages: POST /mealplan/catalog and POST
/mealplan/basket, on both engines (SQLite here, Postgres with PANTRY_TEST_DB_URL).

No LLM in either, in demo mode or live: the catalog runs live too with every model boundary
patched to raise. Expected values come from the seed files, the offers read from the database
directly and the resolve in demo mode, never from the code under test.
"""
from __future__ import annotations

import copy
import json
import math
from fractions import Fraction

import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session

from pantry_planner.db import SEEDS_DIR
from tests.mealplan_fixtures import (
    PRODUCTS,
    STARTERS,
    client,
    doc_draft,
    doc_recipe,
    done_db,
    starter_draft,
    use_db,
)

SEED = json.loads((SEEDS_DIR / "mealplan_starters.json").read_text(encoding="utf-8"))
LIBRARY = {r["slug"]: r for r in json.loads((SEEDS_DIR / "recipes.json").read_text())}
SLOTS = ("breakfast", "lunch", "dinner", "snack")
TUNA, CHEDDAR, THIGHS, BONELESS, SALT, GARLIC = 48, 23, 11, 46, 122, 13
CHEDDAR_SHREDDED = 44
WHEAT_BREAD, OTHER_WHEAT_BREAD = 2, 1     # Wheat Bread 2, the resolve's pick, and Wheat Bread 1
FAR_STORE = 4        # MegaSave Richmond, about 14 km from the server's reference point
TUNA_MELT, BIRYANI, PIZZA = "starter:tuna_melt", "starter:chicken_biryani", \
    "starter:pepperoni_pizza"


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
    out = {s["doc_key"]: {**{k: s[k] for k in ("slot", "diets", "contains", "image_wiki",
                                                 "title", "servings")},
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
        assert r["image"] == {"kind": "wiki", "title": m["image_wiki"]}, key
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


def test_a_product_of_unknown_need_is_one_pack_in_all():
    """Salt "to taste" in three recipes is one bag, not three; a known need still rounds up,
    and a line of unknown need beside it adds no pack of its own."""
    from pantry_planner.mealplan.catalog import whole_packs

    assert whole_packs([None, None, None], 1000) == 1
    assert whole_packs([5.0, None], None) == 1
    assert whole_packs([600.0, 500.0], 500) == 3
    assert whole_packs([600.0, None], 500) == 2
    assert whole_packs([100.0, None], 500) == 1
    assert whole_packs([0.0], 500) == 0


def test_each_recipe_lists_its_lines_for_costing_whole_packs():
    from pantry_planner.nlsearch.units import is_non_purchase, normalize_quantity

    got = by_key(cat(household_servings=3))
    for key, m in _meta().items():
        bought = _resolved(key)
        want = []
        for ln in _lines(key):
            if is_non_purchase(ln["name"]):
                continue
            p = PRODUCTS[bought[ln["line_no"]]]
            qty, _uom = normalize_quantity(ln["quantity"], ln["unit"])
            want.append({"ingredient": ln["name"], "product_id": p["id"],
                         "pack_price": _cheapest(p["id"]), "pack_qty": p["unit_qty"],
                         "pack_uom": p["unit_uom"],
                         "need_per_meal": round(qty * 3 / m["servings"], 4), "held_back": False})
        assert got[key]["lines"] == want, key
        assert len(want) == got[key]["lines_total"]


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
    assert melt["unavailable"] == melt["held_back"] == ["Canned Tuna"]
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
    # Both cheddars: matched by meaning, the shredded one is the same item as the block.
    _sql("DELETE FROM store_products WHERE product_id IN (:a, :b, :c) AND store_id <> :far",
         a=TUNA, b=CHEDDAR, c=CHEDDAR_SHREDDED, far=FAR_STORE)
    _changed()
    near = by_key(cat(settings={"max_km": 5}))
    assert near[TUNA_MELT]["unavailable"] == ["Canned Tuna", "Cheddar Cheese"]
    assert near[TUNA_MELT]["held_back"] == []           # no origin rule: they are out of range
    assert near[TUNA_MELT]["lines_priced"] == 2
    # the library path picks from the whole catalog, then finds no offer in range
    assert near["lib:grilled_cheese"]["unavailable"] == ["Cheddar Cheese"]
    assert near["lib:grilled_cheese"]["lines_priced"] == 2
    anywhere = by_key(cat())
    assert anywhere[TUNA_MELT]["unavailable"] == [] == anywhere["lib:grilled_cheese"][
        "unavailable"]


def test_held_back_names_what_the_origin_rule_left_out_not_what_is_out_of_range(reseed):
    """Garlic's only candidate evidenced as from the United States: with the exclusion, every
    recipe with garlic has it unavailable and held back, the library's (whose planner leaves an
    ingredient out only for the origin rule) and the starters' (whose planner leaves one out
    of range for either reason, told apart by the same retrieval without the rule). Tuna sold
    only out of range is unavailable and not held back, rule or not."""
    _us_evidence(GARLIC)
    _sql("DELETE FROM store_products WHERE product_id = :p AND store_id <> :far", p=TUNA,
         far=FAR_STORE)
    _changed()
    got = by_key(cat(settings={"exclude_origin": ["United States"], "max_km": 5}))
    for key in ("lib:tomato_penne", "starter:tomato_pilaf", "starter:ginger_soy_salmon"):
        assert "Garlic" in got[key]["unavailable"], key
        assert got[key]["held_back"] == ["Garlic"], key
        (line,) = [ln for ln in got[key]["lines"] if ln["ingredient"] == "Garlic"]
        assert (line["held_back"], line["product_id"], line["pack_price"]) == (True, None, None)
    melt = got[TUNA_MELT]
    assert (melt["unavailable"], melt["held_back"]) == (["Canned Tuna"], [])
    for r in got.values():
        assert set(r["held_back"]) <= set(r["unavailable"]), r["key"]
    plain = by_key(cat(settings={"max_km": 5}))
    assert all(r["held_back"] == [] for r in plain.values())
    assert plain["starter:tomato_pilaf"]["unavailable"] == []


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


PAGE_IMAGE = "https://blog.example/img/tofu-bowl.jpg"


def _from_the_web(recipe: dict, image_url: str | None = PAGE_IMAGE) -> dict:
    """The recipe as if imported from a page whose preview image is image_url."""
    return {**recipe, "doc": {**recipe["doc"], "source": {
        "kind": "web", "method": "jsonld", "url": "https://blog.example/tofu-bowl",
        "site": "blog.example", "image_url": image_url}}}


def test_a_shoppers_recipe_shows_its_pages_preview_image_and_a_pasted_one_none():
    """A recipe names its picture, never holds one: a seed's Wikipedia article, the link to the
    preview image of the page a shopper's recipe came from, or nothing."""
    lines = [("Jasmine Rice", 150, "g")]
    web = _from_the_web(_doc("my:web", "Tofu Rice Bowl", lines))
    untitled = _from_the_web(_doc("my:untitled", "Rice", lines), None)
    pasted = _doc("my:pasted", "Plain Rice", lines)
    got = by_key(cat(recipes=[web, untitled, pasted]))
    assert got["my:web"]["image"] == {"kind": "url", "url": PAGE_IMAGE}
    assert got["my:untitled"]["image"] is None and got["my:pasted"]["image"] is None
    assert got[TUNA_MELT]["image"] == {"kind": "wiki", "title": "Melt sandwich"}
    assert got["lib:pbj_sandwich"]["image"] == {"kind": "wiki",
                                                "title": "Peanut butter and jelly sandwich"}
    for bad in ("/img/tofu-bowl.jpg", "javascript:alert(1)", "data:image/png;base64,AAAA",
                "https://blog.example/" + "x" * 2000):
        r = client().post("/mealplan/catalog", json={"recipes": [_from_the_web(web, bad)]})
        assert r.status_code == 422, bad


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


# ─── The basket ──────────────────────────────────────────────

def test_an_address_is_kept_with_the_plan_and_changes_nothing():
    """The shopper's words for where they shop travel with the plan; only lat/lon count."""
    from pantry_planner.mealplan.models import PlanSettings

    s = PlanSettings(lat=49.28, lon=-123.12, address="800 Robson St, Vancouver")
    assert s.address == "800 Robson St, Vancouver"
    with pytest.raises(ValueError):
        PlanSettings(address="x" * 201)
    plain = cat(settings={"lat": 49.28, "lon": -123.12})
    named = cat(settings={"lat": 49.28, "lon": -123.12, "address": "800 Robson St, Vancouver"})
    assert [r["lines"] for r in plain["recipes"]] == [r["lines"] for r in named["recipes"]]


def test_a_pinned_line_is_costed_at_the_shoppers_product():
    """The Meals stage costs what the Ingredients stage buys: a plan's pin puts the shopper's
    product on that line of that recipe only, at its own pack price."""
    plain = by_key(cat())["lib:tomato_penne"]
    oil = next(ln for ln in plain["lines"] if ln["ingredient"] == "Olive Oil")
    other = 16 if oil["product_id"] != 16 else 62
    pinned = by_key(cat(pins={"lib:tomato_penne": {"4": other}}))
    line = next(ln for ln in pinned["lib:tomato_penne"]["lines"] if ln["ingredient"] == "Olive Oil")
    assert line["product_id"] == other and line["pack_qty"] == PRODUCTS[other]["unit_qty"]
    assert line["pack_price"] is not None and line["pack_price"] != oil["pack_price"]
    others = [ln for ln in pinned["lib:tomato_penne"]["lines"] if ln["ingredient"] != "Olive Oil"]
    assert others == [ln for ln in plain["lines"] if ln["ingredient"] != "Olive Oil"]
    assert pinned["lib:spaghetti_bolognese"] == by_key(cat())["lib:spaghetti_bolognese"]
    # a pin to a product nobody sells is the pick again, not an error
    assert by_key(cat(pins={"lib:tomato_penne": {"4": 9999}}))["lib:tomato_penne"]["lines"] \
        == plain["lines"]


COUNTS = {"pepperoni_pizza": 3, "chicken_biryani": 2, "tuna_melt": 2}


@pytest.fixture(scope="module")
def chosen():
    return starter_draft(COUNTS)


def bask(draft: dict) -> dict:
    r = client().post("/mealplan/basket", json={"draft": draft})
    assert r.status_code == 200, r.text
    return r.json()


def _needs(draft: dict) -> dict[tuple[str, int], tuple[Fraction, str]]:
    """Each line's need for all its recipe's meals, from the starter file: quantity x meals x
    household / servings."""
    from pantry_planner.nlsearch.units import normalize_quantity

    out = {}
    for key, n in COUNTS.items():
        s = STARTERS[key]
        for ln in s["lines"]:
            q = normalize_quantity(ln["quantity"], ln["unit"])
            if q is not None:
                out[s["doc_key"], ln["line_no"]] = (Fraction(repr(q[0])) * n * 2
                                                    / s["servings"], q[1])
    return out


def test_the_basket_lists_each_chosen_recipes_lines_with_the_drafts_products(chosen):
    from pantry_planner.mealplan.needs import fmt_qty

    body = bask(chosen)
    assert (body["v"], body["rev"], body["unavailable_count"], body["budget"]) == (1, 0, 0, None)
    got = {r["key"]: r for r in body["recipes"]}
    assert set(got) == {f"starter:{k}" for k in COUNTS}
    needs = _needs(chosen)
    for key, n in COUNTS.items():
        r = got[f"starter:{key}"]
        assert (r["status"], r["meals"], r["title"]) == ("ok", n, STARTERS[key]["title"])
        bought = {ln["line_no"]: ln["product_id"]
                  for ln in chosen["resolved"][f"starter:{key}"]["lines"]}
        lines = [ln for ln in STARTERS[key]["lines"] if ln["name"] != "Water"]
        assert [ln["line_no"] for ln in r["lines"]] == [ln["line_no"] for ln in lines]
        for ln in r["lines"]:
            p = PRODUCTS[bought[ln["line_no"]]]
            assert ln["product"]["id"] == p["id"] and ln["product"]["name"] == p["name"]
            assert ln["product"]["size"] == p["unit_size"]
            assert ln["product"]["demo_product"] == (p["id"] in (166, 167, 168, 169))
            assert set(ln["product"]["labels"]) - {"fish"} <= set(
                t for t in p["dietary_tags"].split(",") if t)
            assert ln["need"] == fmt_qty(*needs[f"starter:{key}", ln["line_no"]])
            assert ln["available"] and ln["why_unavailable"] is None and not ln["pinned"]
    # Salt is one purchase for the pizza and the biryani; each names the other.
    salt = {k: ln for k, r in got.items() for ln in r["lines"] if ln["product"]["id"] == SALT}
    assert salt[PIZZA]["shared_with"] == [BIRYANI] and salt[BIRYANI]["shared_with"] == [PIZZA]


def test_the_baskets_recipes_name_their_pictures_as_the_catalog_does(chosen):
    got = {r["key"]: r["image"] for r in bask(chosen)["recipes"]}
    assert got == {f"starter:{k}": {"kind": "wiki", "title": STARTERS[k]["image_wiki"]}
                   for k in COUNTS}
    web, pasted = (doc_recipe(k, "Rice Bowl", 2, [("Basmati Rice", 200, "g", 8)], wanted=1)
                   for k in ("my:web", "my:pasted"))
    web[0]["ref"] = _from_the_web(web[0]["ref"])
    got = {r["key"]: r["image"]
           for r in bask(doc_draft({"my:web": web, "my:pasted": pasted}))["recipes"]}
    assert got == {"my:web": {"kind": "url", "url": PAGE_IMAGE}, "my:pasted": None}


def _cheapest_offer(pid: int, max_km: float | None = None) -> dict:
    """The product's cheapest offer in range, ties to the lower store id: the price matrix the
    trips read, run directly."""
    from pantry_planner.config import settings
    from pantry_planner.nlsearch.sql_builder import build_price_matrix_sql

    cfg = settings()
    sql, params = build_price_matrix_sql([pid], cfg.default_lat, cfg.default_lon, max_km)
    return min(_sql(sql, **params), key=lambda r: (r["price"], r["store_id"]))


def test_the_basket_buys_whole_packs_at_the_cheapest_offer_in_range(chosen):
    """Every product once, at its cheapest offer in range, its packs the summed need over the
    pack size rounded up; each line's cost its share of the packs by need."""
    body = bask(chosen)
    needs = _needs(chosen)
    buying: dict[int, list[tuple[str, int]]] = {}
    for r in body["recipes"]:
        for ln in r["lines"]:
            buying.setdefault(ln["product"]["id"], []).append((r["key"], ln["line_no"]))
    cost = {}
    for pid, cs in buying.items():
        total = float(sum(needs[c][0] for c in cs))
        packs = math.ceil(total / PRODUCTS[pid]["unit_qty"] - 1e-9)
        at = _cheapest_offer(pid)
        cost[pid] = round(at["price"] * packs, 2)
        for r in body["recipes"]:
            for ln in r["lines"]:
                if ln["product"]["id"] != pid:
                    continue
                assert ln["store"] == {"name": at["store_name"], "distance_km": round(
                    math.sqrt(at["dist_km2"]), 1)}
                assert ln["price"] == at["price"]
                share = at["price"] * packs * float(needs[r["key"], ln["line_no"]][0]) / total
                assert ln["cost_for_need"] == pytest.approx(share, abs=0.006)
    assert body["est_total"] == pytest.approx(sum(cost.values()), abs=0.001)
    assert sum(ln["cost_for_need"] for r in body["recipes"] for ln in r["lines"]) == \
        pytest.approx(body["est_total"], abs=0.02)


def _whole_pack_cost(lines: list[tuple[int, dict]]) -> float:
    """The console's rule over catalog lines, (meals, line) each: per product the total need
    over the pack size rounded up, and at least one pack (one in all) when a line's need or
    the pack size is unknown, at the line's pack price; a line with no pack price is not
    bought."""
    by_product: dict[int, list[tuple[int, dict]]] = {}
    for meals, ln in lines:
        if ln["pack_price"] is not None:
            by_product.setdefault(ln["product_id"], []).append((meals, ln))
    total = 0.0
    for got in by_product.values():
        qty, price = got[0][1]["pack_qty"], got[0][1]["pack_price"]
        known = [m * ln["need_per_meal"] for m, ln in got
                 if ln["need_per_meal"] is not None and qty]
        packs = math.ceil(sum(known) / qty - 1e-9) if known else 0
        packs = max(packs, 1) if len(known) < len(got) else packs
        total += round(price * packs, 2)
    return round(total, 2)


def test_the_catalogs_lines_cost_a_selection_as_the_basket_does(chosen):
    """Stage 2's budget bar and stage 3's total agree: the catalog's lines, costed by whole
    packs for the chosen meals, are the basket's est_total to the cent."""
    recipes = by_key(cat(household_servings=2))
    lines = [(n, ln) for key, n in COUNTS.items() for ln in recipes[f"starter:{key}"]["lines"]]
    assert _whole_pack_cost(lines) == bask(chosen)["est_total"]
    # far more than the per-meal shares the estimate adds up
    shares = sum(n * recipes[f"starter:{k}"]["est_cost_per_meal"] for k, n in COUNTS.items())
    assert bask(chosen)["est_total"] > shares
    one = by_key(cat(household_servings=1))
    lines = [(n, ln) for key, n in COUNTS.items() for ln in one[f"starter:{key}"]["lines"]]
    half = starter_draft(COUNTS, prefs={"household_servings": 1})
    assert _whole_pack_cost(lines) == bask(half)["est_total"]


def test_pins_replace_the_resolved_product_and_are_checked_as_the_schedule_checks_them(chosen):
    body = bask({**chosen, "pins": {BIRYANI: {"1": BONELESS}}})
    (line,) = [ln for r in body["recipes"] if r["key"] == BIRYANI for ln in r["lines"]
               if ln["line_no"] == 1]
    assert line["product"]["id"] == BONELESS and line["pinned"] and line["available"]
    # a pin equal to the resolve's pick is no pin
    same = bask({**chosen, "pins": {BIRYANI: {"1": THIGHS}}})
    assert not any(ln["pinned"] for r in same["recipes"] for ln in r["lines"])
    r = client().post("/mealplan/basket", json={"draft": {**chosen, "pins": {
        BIRYANI: {"99": BONELESS}}}})
    assert r.status_code == 422 and r.json()["detail"]["error"] == "pin_invalid"


def test_an_excluded_product_is_not_available_and_never_priced(chosen, reseed):
    _us_evidence(THIGHS)
    plain = bask(chosen)
    excl = bask({**chosen, "settings": {"exclude_origin": ["United States"]}})
    (line,) = [ln for r in excl["recipes"] if r["key"] == BIRYANI for ln in r["lines"]
               if ln["product"]["id"] == THIGHS]
    assert (line["available"], line["price"], line["store"], line["cost_for_need"]) == (
        False, None, None, None)
    assert line["unavailable_kind"] == "origin_excluded"
    assert line["why_unavailable"] == ("Chicken Thighs Bone-In is evidenced as United States "
                                       "(made or packed there), which this plan excludes")
    assert line["origin"] == {"country": "United States", "claim": "Made in United States, "
                              "ingredients may be imported", "status": "evidenced",
                              "preferred_rank": None}
    # named on its own line, with the offer it would have been bought at
    offer = _cheapest_offer(THIGHS)
    assert line["held_back"] == [{
        "product_id": THIGHS, "product": "Chicken Thighs Bone-In", "country": "United States",
        "claim": "Made in United States", "field": "manufactured_in",
        "store": offer["store_name"], "price": offer["price"]}]
    assert (excl["origin_held_back"], plain["origin_held_back"]) == (1, 0)
    assert excl["unavailable_count"] == 1
    (before,) = [ln for r in plain["recipes"] for ln in r["lines"]
                 if ln["product"]["id"] == THIGHS]
    assert excl["est_total"] == pytest.approx(plain["est_total"] - before["cost_for_need"],
                                              abs=0.001)


def test_a_product_no_store_in_range_sells_names_the_nearest_offer(chosen, reseed):
    _sql("DELETE FROM store_products WHERE product_id = :p AND store_id <> :far",
         p=TUNA, far=FAR_STORE)
    (far,) = _sql("SELECT price FROM store_products WHERE product_id = :p", p=TUNA)
    body = bask({**chosen, "settings": {"max_km": 5}})
    (line,) = [ln for r in body["recipes"] for ln in r["lines"] if ln["product"]["id"] == TUNA]
    assert not line["available"] and line["price"] is None and line["store"] is None
    assert line["unavailable_kind"] == "out_of_range"
    assert line["why_unavailable"].startswith(
        "no store within 5 km sells it; the nearest offer is MegaSave Richmond, ")
    assert line["why_unavailable"].endswith(f"km away, at ${far['price']:.2f}")
    assert body["unavailable_count"] == 1


def test_a_line_the_resolve_left_out_says_which_rule_left_it_out(reseed):
    """Resolved under the exclusion, the pilaf buys no garlic: the line is unavailable
    because of the origin rule, in the planner's words, not "not sold nearby"."""
    from pantry_planner.mealplan.models import RecipeRef
    from pantry_planner.mealplan.resolve import resolve_one

    _us_evidence(GARLIC)
    _changed()
    rules = {"exclude_origin": ["United States"]}
    ref = RecipeRef(key="starter:tomato_pilaf", starter="tomato_pilaf")
    resolved = resolve_one(ref, exclude=["United States"])
    draft = {"start_date": "2026-10-09", "days": 7, "settings": rules,
             "recipes": {ref.key: {"ref": ref.model_dump(mode="json"), "wanted": 2,
                                   "slot": "lunch"}},
             "resolved": {ref.key: resolved.model_dump(mode="json")}}
    body = bask(draft)
    (garlic,) = [ln for r in body["recipes"] for ln in r["lines"] if ln["ingredient"] == "Garlic"]
    assert (garlic["product"], garlic["available"], garlic["unavailable_kind"]) == (
        None, False, "origin_excluded")
    assert garlic["why_unavailable"] == ("its only candidate is evidenced as coming from "
                                         "United States, which this plan excludes")
    assert [(h["product_id"], h["claim"]) for h in garlic["held_back"]] == [
        (GARLIC, "Made in United States")]
    assert body["unavailable_count"] == 1
    assert all(ln["unavailable_kind"] is None for r in body["recipes"] for ln in r["lines"]
               if ln["available"])


def test_a_product_no_store_sells_at_all_is_not_stocked(chosen, reseed):
    _sql("DELETE FROM store_products WHERE product_id = :p", p=TUNA)
    (line,) = [ln for r in bask(chosen)["recipes"] for ln in r["lines"]
               if ln["product"]["id"] == TUNA]
    assert (line["unavailable_kind"], line["why_unavailable"]) == ("not_stocked",
                                                                   "no store sells it")


def test_origin_evidence_and_the_preference_order(chosen, reseed):
    from pantry_planner import db

    db.save_origin_evidence([
        dict(product_id=18, source="label-photo", source_ref="18.jpg", claim_type="product-of",
             verbatim="Product of Canada", ingredient_origin="Canada", manufactured_in="Canada",
             confidence="high", importer_only=False, note="", observed_at=""),
        *(dict(product_id=60, source=src, source_ref=f"60-{c}", claim_type="product-of",
               verbatim=f"Product of {c}", ingredient_origin=c, manufactured_in=c,
               confidence="high", importer_only=False, note="", observed_at="")
          for src, c in (("label-photo", "Canada"), ("off", "France")))])
    body = bask({**copy.deepcopy(chosen), "settings": {"preference": ["France", "Canada"]}})
    origin = {ln["product"]["id"]: ln["origin"] for r in body["recipes"] for ln in r["lines"]}
    assert origin[18] == {"country": "Canada", "claim": "From Canada", "status": "evidenced",
                          "preferred_rank": 2}
    assert origin[SALT] == {"country": None, "claim": "Origin unknown", "status": "unknown",
                            "preferred_rank": None}
    eggs = bask(starter_draft({"scrambled_eggs_on_toast": 1}))
    (egg,) = [ln for r in eggs["recipes"] for ln in r["lines"] if ln["product"]["id"] == 60]
    assert egg["origin"] == {"country": None, "claim": "Origin unclear: sources disagree",
                             "status": "conflicting", "preferred_rank": None}


def test_an_unresolved_recipe_lists_no_lines_and_an_unknown_amount_is_one_pack():
    known = doc_recipe("my:k", "Rice Bowl", 2, [("Basmati Rice", 200, "g", 8)], wanted=2)
    unknown = doc_recipe("my:u", "Salmon", None, [("Atlantic Salmon", 400, "g", 47)], wanted=1)
    draft = doc_draft({"my:k": known, "my:u": unknown}, prefs={"budget": 50})
    draft["recipes"]["starter:tuna_melt"] = {
        "ref": {"key": "starter:tuna_melt", "starter": "tuna_melt"}, "wanted": 1,
        "slot": "lunch"}
    draft["recipes"]["my:idle"] = {**known[0], "ref": {**known[0]["ref"], "key": "my:idle",
                                                       "doc": {**known[0]["ref"]["doc"],
                                                               "key": "my:idle"}},
                                   "wanted": 0}
    body = bask(draft)
    got = {r["key"]: r for r in body["recipes"]}
    assert set(got) == {"my:k", "my:u", "starter:tuna_melt"}      # my:idle has no meal
    assert (got["starter:tuna_melt"]["status"], got["starter:tuna_melt"]["lines"]) == (
        "unresolved", [])
    (salmon,) = got["my:u"]["lines"]
    assert got["my:u"]["status"] == "needs_servings"
    # its need is unknown, so it counts as one pack
    assert (salmon["need"], salmon["available"]) == ("amount unknown", True)
    assert salmon["cost_for_need"] == salmon["price"] == _cheapest_offer(47)["price"]
    (rice,) = got["my:k"]["lines"]
    assert rice["need"] == "400 g"
    assert rice["cost_for_need"] == _cheapest_offer(8)["price"]      # 400 g of a 2 kg bag
    assert body["est_total"] == pytest.approx(rice["cost_for_need"] + salmon["cost_for_need"],
                                              abs=0.001)
    assert body["budget"] == 50


def test_the_basket_refuses_what_the_schedule_refuses(chosen, monkeypatch):
    from pantry_planner import limits

    bad = {**chosen, "recipes": {**chosen["recipes"], "lib:nope": {
        "ref": {"key": "lib:nope", "slug": "nope"}, "wanted": 1}}}
    r = client().post("/mealplan/basket", json={"draft": bad})
    assert r.status_code == 422 and r.json()["detail"]["error"] == "unknown_recipe_key"
    big = {"draft": {**chosen, "padding": "x" * 300_000}}
    assert client().post("/mealplan/basket", json=big).status_code == 413
    assert limits.LIMITS["/mealplan/basket"] == limits.Limit(per_minute=120, burst=120)
    monkeypatch.setitem(limits.LIMITS, "/mealplan/basket", limits.Limit(per_minute=60,
                                                                        burst=1))
    assert client().post("/mealplan/basket", json={"draft": chosen}).status_code == 200
    assert client().post("/mealplan/basket", json={"draft": chosen}).status_code == 429


# ─── Stores the shopper turned off ───────────────────────────

STORE_NAMES = {1: "Pantry Mart Downtown", 2: "GreenLeaf Grocers Kitsilano",
               3: "ValueFoods East Van", 4: "MegaSave Richmond"}
OFF = [1, 3]


def _cheapest_on(pid: int, off: list[int]) -> float:
    return min(r["price"] for r in _sql(
        "SELECT store_id, price FROM store_products WHERE product_id = :p", p=pid)
        if r["store_id"] not in off)


def test_a_store_turned_off_is_never_priced_or_put_on_a_trip(chosen):
    from tests.mealplan_fixtures import schedule, strategy

    off_names = {STORE_NAMES[i] for i in OFF}
    draft = {**chosen, "settings": {"stores_off": OFF}}
    sched = schedule(draft)
    for name in ("fresh", "fewest_trips"):
        st = strategy(sched, name)
        assert st["trips"]
        for t in st["trips"]:
            assert not off_names & set(t["stores"]), t["stores"]
            assert all(ln["store"] not in off_names for ln in t["lines"])
            for o in t["frontier"]:
                assert not off_names & set(o["stores"])
    on_stores = {ln["store"] for t in strategy(schedule(chosen), "fresh")["trips"]
                 for ln in t["lines"]}
    assert on_stores & off_names            # the turned-off stores were on the trips before
    # the basket prices every product at its cheapest offer among the stores left on
    body = bask(draft)
    for r in body["recipes"]:
        for ln in r["lines"]:
            assert ln["store"]["name"] not in off_names
            assert ln["price"] == _cheapest_on(ln["product"]["id"], OFF)
    # and so does the catalog's pack_price
    for key, r in by_key(cat(settings={"stores_off": OFF})).items():
        for ln in r["lines"]:
            assert ln["pack_price"] == _cheapest_on(ln["product_id"], OFF), (key, ln)
    # the Options lists, with and without a trip date, offer nothing from them
    trip = strategy(sched, "fresh")["trips"][0]
    pid = trip["lines"][0]["product"]["id"]
    for extra in ({"trip_date": trip["date"]}, {}):
        r = client().post("/mealplan/alternatives",
                          json={"draft": draft, "product_id": pid, **extra})
        assert r.status_code == 200, r.text
        for it in r.json()["ranking"]["items"]:
            assert it["offer"]["store"] not in off_names
            assert it["trip"] is None or (it["trip"]["buys_at"]["store"] not in off_names
                                          and not off_names & set(it["trip"]["stores"]))


def test_the_calendar_names_only_the_stores_left_on(chosen):
    from tests.mealplan_fixtures import schedule, strategy

    off_names = {STORE_NAMES[i] for i in OFF}
    draft = {**chosen, "settings": {"stores_off": OFF}}
    first = schedule(draft)
    draft["trips"] = [
        {"date": t["date"], "fingerprint": t["fingerprint"], "strategy": "fresh",
         "snapshot": [{"product_id": ln["product"]["id"], "packs": ln["packs"],
                       "storage": ln["storage"], "price_at_approval": ln["price"]}
                      for ln in t["lines"]]} for t in strategy(first, "fresh")["trips"]]
    approved = schedule(draft)["approved_schedule"]
    assert approved["trips"]
    r = client().post("/calendar/preview", json={"schedule": approved})
    assert r.status_code == 200
    for e in r.json()["events"]:
        if e["kind"] == "trip":
            assert not any(n in e["location"] for n in off_names), e["location"]


def test_turning_a_store_off_resolves_nothing_again(chosen, monkeypatch):
    """The resolve ignores stores_off: no recipe is resolved again, and the catalog's picks
    (cached per recipe, shopping point and origin rules) are the same picks."""
    from pantry_planner import flow
    from pantry_planner.mealplan import catalog, resolve

    before = cat()
    calls = []
    real = catalog.picks
    monkeypatch.setattr(catalog, "picks", lambda *a, **k: calls.append(a) or real(*a, **k))

    def boom(*_a, **_k):
        raise AssertionError("resolved again")

    for mod, name in ((resolve, "resolve_one"), (flow, "run"), (flow, "run_spec")):
        monkeypatch.setattr(mod, name, boom)
    rules = {"stores_off": OFF}
    after = cat(settings=rules)
    assert calls == []
    assert [[ln["product_id"] for ln in r["lines"]] for r in after["recipes"]] == [
        [ln["product_id"] for ln in r["lines"]] for r in before["recipes"]]
    draft = {**chosen, "settings": rules}
    assert client().post("/mealplan/schedule", json=draft).status_code == 200
    assert bask(draft)["recipes"]


def test_stores_off_is_bounded(chosen):
    for bad in ([0], [-1], list(range(1, 52)), ["one"]):
        r = client().post("/mealplan/basket",
                          json={"draft": {**chosen, "settings": {"stores_off": bad}}})
        assert r.status_code == 422, bad
    ok = client().post("/mealplan/basket",
                       json={"draft": {**chosen, "settings": {"stores_off": [999]}}})
    assert ok.status_code == 200          # a store id nobody has turns nothing off


def test_a_product_sold_in_range_only_at_stores_turned_off_says_so(chosen, reseed):
    """Canned Tuna sold only at the two stores turned off (and the far one, out of range):
    the line is stores_off, naming them nearest first, and never priced."""
    _sql("DELETE FROM store_products WHERE product_id = :p AND store_id NOT IN (1, 3, 4)",
         p=TUNA)
    rules = {"stores_off": OFF, "max_km": 5}
    body = bask({**chosen, "settings": rules})
    (line,) = [ln for r in body["recipes"] for ln in r["lines"] if ln["product"]["id"] == TUNA]
    assert (line["available"], line["price"], line["store"], line["cost_for_need"]) == (
        False, None, None, None)
    assert line["unavailable_kind"] == "stores_off"
    assert line["why_unavailable"] == ("Sold only at stores you turned off: "
                                       "Pantry Mart Downtown, ValueFoods East Van")
    assert body["unavailable_count"] == 1
    # with one of them back on, it is bought there
    back = bask({**chosen, "settings": {"stores_off": [1], "max_km": 5}})
    (line,) = [ln for r in back["recipes"] for ln in r["lines"] if ln["product"]["id"] == TUNA]
    assert line["available"] and line["store"]["name"] == "ValueFoods East Van"
    # with none in range at all it is out of range, the nearest offer named among stores on
    _sql("DELETE FROM store_products WHERE product_id = :p AND store_id <> 4", p=TUNA)
    (line,) = [ln for r in bask({**chosen, "settings": rules})["recipes"] for ln in r["lines"]
               if ln["product"]["id"] == TUNA]
    assert line["unavailable_kind"] == "out_of_range"
    assert "MegaSave Richmond" in line["why_unavailable"]


def _stores(draft: dict) -> dict:
    r = client().post("/mealplan/stores", json={"draft": draft})
    assert r.status_code == 200, r.text
    return r.json()


def _km(lat: float, lon: float) -> float:
    """The price matrix's equirectangular distance from the server's reference point."""
    from pantry_planner.config import settings

    cfg = settings()
    dy = (cfg.default_lat - lat) * 111.0
    dx = (cfg.default_lon - lon) * 111.0 * math.cos(math.radians(cfg.default_lat))
    return round(math.sqrt(dx * dx + dy * dy), 1)


def test_the_stores_list_every_store_nearest_first_with_what_it_sells(chosen, reseed):
    from pantry_planner.config import settings

    stores = {r["id"]: r for r in _sql("SELECT id, name, lat, lon, address FROM stores")}
    bought = {ln["product"]["id"] for r in bask(chosen)["recipes"] for ln in r["lines"]}
    _sql("DELETE FROM store_products WHERE product_id = :p AND store_id = 2", p=TUNA)
    body = _stores({**chosen, "settings": {"stores_off": [3], "max_km": 5}})
    assert body["v"] == 1 and body["products_total"] == len(bought)
    cfg = settings()
    assert (body["origin"], body["max_km"], body["synthetic"]) == (
        {"lat": cfg.default_lat, "lon": cfg.default_lon}, 5, True)
    got = body["stores"]
    assert [s["id"] for s in got] == sorted(
        stores, key=lambda i: (_km(stores[i]["lat"], stores[i]["lon"]), i))
    for s in got:
        row = stores[s["id"]]
        km = _km(row["lat"], row["lon"])
        assert (s["name"], s["distance_km"], s["in_range"], s["off"]) == (
            row["name"], km, km <= 5, s["id"] == 3)
        assert (s["address"], s["lat"], s["lon"]) == (row["address"], row["lat"], row["lon"])
        stocked = {r["product_id"] for r in _sql(
            "SELECT product_id FROM store_products WHERE store_id = :s", s=s["id"])}
        assert s["sells"] == len(bought & stocked)
    assert next(s for s in got if s["id"] == 2)["sells"] == len(bought) - 1
    # no meal yet: nothing to count
    empty = _stores({"start_date": "2026-10-09", "days": 7})
    assert empty["products_total"] == 0 and all(s["sells"] is None for s in empty["stores"])


def test_the_stores_count_the_pins_and_leave_out_what_the_origin_rule_holds_back(chosen,
                                                                                 reseed):
    pinned = _stores({**chosen, "pins": {BIRYANI: {"1": BONELESS}}})
    plain = _stores(chosen)
    assert pinned["products_total"] == plain["products_total"]     # thighs out, boneless in
    _sql("DELETE FROM store_products WHERE product_id = :p AND store_id = 1", p=BONELESS)
    near = {s["id"]: s for s in _stores({**chosen, "pins": {BIRYANI: {"1": BONELESS}}})["stores"]}
    assert near[1]["sells"] == pinned["products_total"] - 1
    _us_evidence(TUNA)
    held = _stores({**chosen, "settings": {"exclude_origin": ["United States"]}})
    assert held["products_total"] == plain["products_total"] - 1


def test_every_stores_address_is_given_and_synthetic_says_they_are_demo_stores(chosen,
                                                                               monkeypatch):
    """The console labels the addresses of synthetic stores as demo ones; the calendar export
    keeps its own policy and still never writes them."""
    from pantry_planner import calendar_api, config

    table = {r["id"]: r["address"] for r in _sql("SELECT id, address FROM stores")}
    assert all(table.values())
    demo = _stores(chosen)
    assert demo["synthetic"] is True and calendar_api._addresses() is None
    assert {s["id"]: s["address"] for s in demo["stores"]} == table
    monkeypatch.setenv("STORES_SYNTHETIC", "false")
    config.settings.cache_clear()
    try:
        real = _stores(chosen)
        assert calendar_api._addresses() is not None
    finally:
        monkeypatch.delenv("STORES_SYNTHETIC")
        config.settings.cache_clear()
    assert real["synthetic"] is False
    assert {s["id"]: s["address"] for s in real["stores"]} == table


def test_the_stores_answer_the_plans_own_shopping_point_and_no_limit(chosen):
    body = _stores({**chosen, "settings": {"lat": 49.25, "lon": -123.1}})
    assert (body["origin"], body["max_km"]) == ({"lat": 49.25, "lon": -123.1}, None)
    assert all(s["in_range"] for s in body["stores"])
    dist = [s["distance_km"] for s in body["stores"]]
    assert dist == sorted(dist)


def test_the_stores_route_refuses_what_the_schedule_refuses(chosen, monkeypatch):
    from pantry_planner import limits

    bad = {**chosen, "recipes": {**chosen["recipes"], "lib:nope": {
        "ref": {"key": "lib:nope", "slug": "nope"}, "wanted": 1}}}
    r = client().post("/mealplan/stores", json={"draft": bad})
    assert r.status_code == 422 and r.json()["detail"]["error"] == "unknown_recipe_key"
    big = {"draft": {**chosen, "padding": "x" * 300_000}}
    assert client().post("/mealplan/stores", json=big).status_code == 413
    assert limits.LIMITS["/mealplan/stores"] == limits.Limit(per_minute=120, burst=120)
    monkeypatch.setitem(limits.LIMITS, "/mealplan/stores", limits.Limit(per_minute=60,
                                                                        burst=1))
    assert client().post("/mealplan/stores", json={"draft": chosen}).status_code == 200
    assert client().post("/mealplan/stores", json={"draft": chosen}).status_code == 429


# ─── What the origin rule left out; picks that follow the evidence ─

US = ["United States"]
SAUCES = ("lib:tomato_penne", "lib:spaghetti_bolognese", "lib:beef_rice_bowl")
BEEF_LEAN, BEEF_MEDIUM, BEEF_EXTRA_LEAN = 22, 35, 36     # the catalog's three ground beefs


def _us_label(pid: int, verbatim: str) -> None:
    """A package label read as a full claim of the United States, in its own words."""
    from pantry_planner import db

    db.save_origin_evidence([dict(
        product_id=pid, source="label-photo", source_ref=f"{pid}-label.jpg",
        claim_type="product-of", verbatim=verbatim, ingredient_origin="United States",
        manufactured_in="United States", confidence="high", importer_only=False, note="",
        observed_at="")])


def _resolved_draft(keys, settings: dict | None = None, **knobs) -> dict:
    """Library recipes and demo starters, two dinners each, resolved in demo mode with
    `knobs` (as the console resolves them), under the plan's `settings`."""
    from pantry_planner.mealplan.models import RecipeRef
    from pantry_planner.mealplan.resolve import resolve_one

    recipes, resolved = {}, {}
    for key in keys:
        kind, _, name = key.partition(":")
        ref = RecipeRef(key=key, **({"slug": name} if kind == "lib" else {"starter": name}))
        recipes[key] = {"ref": ref.model_dump(mode="json"), "wanted": 2, "slot": "dinner"}
        resolved[key] = resolve_one(ref, **knobs).model_dump(mode="json")
    return {"start_date": "2026-10-09", "days": 7, "recipes": recipes, "resolved": resolved,
            "settings": settings or {}}


def _named(line: dict) -> list[tuple[int, str, str]]:
    return [(h["product_id"], h["claim"], h["field"]) for h in line["held_back"]]


def test_each_line_names_what_the_origin_rule_left_out_in_the_plans_order(reseed):
    """Ground beef: the plan's first choice is evidenced as from the United States, then the
    choice it makes instead. Both lines buying ground beef name both, in that order, with
    their evidence in a shopper's words and the offer each would have been bought at, and buy
    a third. Canned tomatoes: the first choice, by a label claim with no text of its own.
    Each product is counted once, though four lines name them; a plan that excludes nothing
    names nothing."""
    first = _resolved("lib:beef_rice_bowl")[1]
    _us_label(first, "Product of USA")
    second = _resolved("lib:beef_rice_bowl", exclude=US)[1]
    _us_evidence(second)
    tomatoes = _resolved("lib:tomato_penne")[2]
    _us_label(tomatoes, "")
    draft = _resolved_draft(SAUCES, {"exclude_origin": US}, exclude=US)
    body = bask(draft)
    lines = {(r["key"], ln["ingredient"]): ln for r in body["recipes"] for ln in r["lines"]}
    for key in ("lib:beef_rice_bowl", "lib:spaghetti_bolognese"):
        beef = lines[key, "Ground Beef"]
        assert beef["available"] and beef["product"]["id"] not in (first, second)
        assert _named(beef) == [(first, "Product of USA", "ingredient_origin"),
                                (second, "Made in United States", "manufactured_in")]
        for h in beef["held_back"]:
            offer = _cheapest_offer(h["product_id"])
            assert (h["product"], h["country"], h["store"], h["price"]) == (
                PRODUCTS[h["product_id"]]["name"], "United States", offer["store_name"],
                offer["price"])
    for key in ("lib:tomato_penne", "lib:spaghetti_bolognese"):
        assert _named(lines[key, "Canned Tomatoes"]) == [
            (tomatoes, "From United States", "ingredient_origin")]
    others = [ln for (_key, name), ln in lines.items()
              if name not in ("Ground Beef", "Canned Tomatoes")]
    assert others and all(ln["held_back"] == [] for ln in others)
    assert body["origin_held_back"] == 3
    plain = bask({**draft, "settings": {}})
    assert plain["origin_held_back"] == 0
    assert all(ln["held_back"] == [] for r in plain["recipes"] for ln in r["lines"])


def test_a_line_names_at_most_three_and_only_what_a_store_that_is_on_sells(reseed):
    """Every yellow onion evidenced as from the United States: the bolognese's Yellow Onion
    buys nothing and names the first three of the four, the plan's first choice first, while
    origin_held_back counts all four. One sold only at stores the shopper turned off is
    neither named nor counted."""
    onions = [pid for pid, p in PRODUCTS.items()
              if {"yellow", "onion"} <= set(p["name"].lower().replace("onions", "onion").split())]
    first = _resolved("lib:spaghetti_bolognese")[3]
    assert first in onions and len(onions) == 4
    for pid in onions:
        _us_label(pid, "Grown in USA")
    draft = _resolved_draft(["lib:spaghetti_bolognese"], {"exclude_origin": US}, exclude=US)

    def onion(body: dict) -> dict:
        (line,) = [ln for r in body["recipes"] for ln in r["lines"]
                   if ln["ingredient"] == "Yellow Onion"]
        return line

    body = bask(draft)
    line = onion(body)
    assert (line["product"], line["unavailable_kind"]) == (None, "origin_excluded")
    named = [h["product_id"] for h in line["held_back"]]
    assert len(named) == 3 and named[0] == first and set(named) < set(onions)
    assert body["origin_held_back"] == 4
    _sql("DELETE FROM store_products WHERE product_id = :p AND store_id NOT IN (:a, :b)",
         p=first, a=OFF[0], b=OFF[1])
    off = bask({**draft, "settings": {"exclude_origin": US, "stores_off": OFF}})
    assert sorted(h["product_id"] for h in onion(off)["held_back"]) == sorted(
        set(onions) - {first})
    assert off["origin_held_back"] == 3


def test_a_pinned_product_the_rule_now_holds_back_is_named_first_on_its_line(reseed):
    """Pinned, then evidenced as from the United States: the Ingredients stage keeps the pin
    and shows its line held back, naming it first and then what else the rule dropped, while
    the schedule refuses the pin until the shopper answers it there."""
    draft = _resolved_draft(["lib:beef_rice_bowl"])
    planned = draft["resolved"]["lib:beef_rice_bowl"]["lines"][0]["product_id"]
    pin, other = [p for p in (BEEF_EXTRA_LEAN, BEEF_MEDIUM, BEEF_LEAN) if p != planned][:2]
    draft["pins"] = {"lib:beef_rice_bowl": {"1": pin}}
    before = bask(draft)
    (beef,) = [ln for r in before["recipes"] for ln in r["lines"] if ln["line_no"] == 1]
    assert (beef["product"]["id"], beef["pinned"], beef["available"]) == (pin, True, True)
    _us_evidence(pin, other)
    ruled = {**draft, "settings": {"exclude_origin": US}}
    body = bask(ruled)
    (beef,) = [ln for r in body["recipes"] for ln in r["lines"] if ln["line_no"] == 1]
    assert (beef["product"]["id"], beef["pinned"], beef["available"],
            beef["unavailable_kind"]) == (pin, True, False, "origin_excluded")
    assert [h["product_id"] for h in beef["held_back"]] == [pin, other]
    assert (body["origin_held_back"], body["unavailable_count"]) == (2, 1)
    assert _stores(ruled)["products_total"] == _stores(draft)["products_total"] - 1
    r = client().post("/mealplan/schedule", json=ruled)
    assert (r.status_code, r.json()["detail"]["error"]) == (422, "pin_invalid")


def test_a_planned_product_the_rule_now_holds_back_is_never_bought(reseed):
    """Resolved, then evidenced as made in the United States: the Ingredients stage keeps the
    planned product on its line, held back, for the shopper to answer, and until then no trip
    of either strategy buys it. The schedule says so in a decide warning that names the
    product, the evidence and the line, with the remedy that checks the recipe's products
    again. Without the rule the trips buy it as before, and so they do under the rule while
    nothing evidences it (only positive evidence holds a product back)."""
    from tests.mealplan_fixtures import schedule

    key = "lib:beef_rice_bowl"
    draft = _resolved_draft([key])
    planned = draft["resolved"][key]["lines"][0]["product_id"]
    ruled = {**draft, "settings": {"exclude_origin": US}}

    def bought(body: dict) -> set[int]:
        return {ln["product"]["id"] for res in body["strategies"] for t in res["trips"]
                for ln in t["lines"]}

    def held(body: dict) -> list[dict]:
        return [w for w in body["warnings"] if w["code"] == "origin_excluded"]

    before = schedule(ruled)
    assert planned in bought(before) and held(before) == []
    _us_evidence(planned)
    body = schedule(ruled)
    assert planned not in bought(body)
    assert bought(body) == bought(before) - {planned}
    (w,) = held(body)
    assert (w["level"], w["recipe_key"], w["product_id"], w["remedies"]) == (
        "decide", key, planned, [{"op": "resolve", "recipe_key": key}])
    assert sorted(w["meal_ids"]) == sorted(m["id"] for m in body["meals"]
                                           if m["recipe_key"] == key and m["date"])
    assert ("is held back by the origin rule: it is evidenced as United States (made or "
            'packed there), which this plan excludes ("Made in United States")') in w["message"]
    assert "The trips buy nothing for Ground Beef until" in w["message"]
    (beef,) = [ln for r in bask(ruled)["recipes"] for ln in r["lines"] if ln["line_no"] == 1]
    assert (beef["product"]["id"], beef["available"], beef["unavailable_kind"]) == (
        planned, False, "origin_excluded")
    unruled = schedule(draft)
    assert planned in bought(unruled) and held(unruled) == []


def test_a_line_matched_by_words_names_the_products_sharing_the_most_words(reseed, by_words):
    """Matched by words (names no seed knows, with the embedder down), a line's candidates are
    the products sharing the most words with it. A shopper's recipe resolved before the
    evidence: Yellow Onion names its own product, the yellow onion, and not the green onions
    the rule also holds back, which share one word with it; Diced Tomatoes buys the petite
    ones and names the canned diced tomatoes the rule dropped."""
    _changed()          # what is cached was matched by meaning
    yellow, green, diced, petite = 12, 142, 15, 165
    for pid in (yellow, green, diced):
        _us_label(pid, "Product of USA")
    recipe = doc_recipe("my:onions", "Onion Salsa", 2, [
        ("Yellow Onion", 200, "g", yellow), ("Green Onions", 50, "g", green),
        ("Diced Tomatoes", 400, "g", petite)], wanted=2)
    body = bask({**doc_draft({"my:onions": recipe}), "settings": {"exclude_origin": US}})
    named = {ln["ingredient"]: [h["product_id"] for h in ln["held_back"]]
             for r in body["recipes"] for ln in r["lines"]}
    assert named == {"Yellow Onion": [yellow], "Green Onions": [green],
                     "Diced Tomatoes": [diced]}
    assert body["origin_held_back"] == 3


def test_picks_follow_new_evidence_with_no_cache_cleared(reseed):
    """Evidence reaches the database from another process (an ingest from the CLI) and no
    cache is cleared: under the exclusion the resolve and the catalog pick again, so the
    recipe resolved again buys another product instead of leaving its line to answer."""
    ref = {"key": "lib:beef_rice_bowl", "slug": "beef_rice_bowl"}
    rules = {"exclude_origin": US}

    def resolve() -> dict:
        r = client().post("/mealplan/resolve", json={"recipes": [ref], **rules})
        assert r.status_code == 200, r.text
        return r.json()["resolved"][0]

    def beef(body: dict) -> dict:
        return next(ln for r in body["recipes"] for ln in r["lines"]
                    if ln["ingredient"] == "Ground Beef")

    def catalog_pick() -> int:
        return beef({"recipes": [by_key(cat(settings=rules))[ref["key"]]]})["product_id"]

    before = resolve()
    pick = before["lines"][0]["product_id"]
    assert resolve()["cached"] and catalog_pick() == pick
    _us_evidence(pick)                       # written to the database only, as an ingest does
    draft = {"start_date": "2026-10-09", "days": 7, "settings": rules,
             "recipes": {ref["key"]: {"ref": ref, "wanted": 2, "slot": "dinner"}},
             "resolved": {ref["key"]: before}}
    stale = beef(bask(draft))
    assert (stale["unavailable_kind"], _named(stale)[0][0]) == ("origin_excluded", pick)
    after = resolve()
    now = after["lines"][0]["product_id"]
    assert not after["cached"] and now not in (pick, None)
    assert catalog_pick() == now
    fresh = beef(bask({**draft, "resolved": {ref["key"]: after}}))
    assert fresh["available"] and fresh["product"]["id"] == now
    assert [h["product_id"] for h in fresh["held_back"]] == [pick]


def test_what_the_rule_left_out_calls_no_model_and_is_the_same_live(reseed, monkeypatch):
    """Live, the resolve's selector is a model, but what the rule left out of each line is
    the planner's retrieval and ranking with no model, as the catalog's picks are: the same
    in demo mode and live, for a library recipe and a starter."""
    from pantry_planner import config, flow, llm, selector
    from pantry_planner.mealplan import catalog
    from pantry_planner.nlsearch import planner, query_parser
    from pantry_planner.router import classifier

    _us_label(_resolved("lib:beef_rice_bowl")[1], "Product of USA")
    _us_label(_resolved("starter:tomato_pilaf")[3], "Grown in USA")
    draft = _resolved_draft(["lib:beef_rice_bowl", "starter:tomato_pilaf"],
                            {"exclude_origin": US}, exclude=US)

    def boom(*_a, **_k):
        raise AssertionError("the basket called a model")

    for mod, name in ((llm, "forced_tool_call"), (selector, "forced_tool_call"),
                      (classifier, "forced_tool_call"), (query_parser, "forced_tool_call"),
                      (selector, "call_selector"), (flow, "call_selector"),
                      (query_parser, "parse_input"), (planner, "parse_input")):
        monkeypatch.setattr(mod, name, boom)
    catalog.clear_cache()
    demo = bask(draft)
    catalog.clear_cache()
    config.set_runtime_overrides(demo_mode=False)
    try:
        assert not config.settings().demo_mode
        live = bask(draft)
    finally:
        config.clear_runtime_overrides()
    assert demo["origin_held_back"] == 2
    del demo["latency_ms"], live["latency_ms"]
    assert live == demo


def test_the_evidence_is_read_once_a_request_and_only_under_an_origin_rule(chosen,
                                                                          monkeypatch):
    """The evidence's version is one small query: a catalog, a basket and a resolve under an
    origin rule read it once each, and nothing reads it without one."""
    from pantry_planner import db

    reads = []
    real = db.origin_evidence_version
    monkeypatch.setattr(db, "origin_evidence_version", lambda: reads.append(1) or real())
    refs = [{"key": f"starter:{k}", "starter": k} for k in COUNTS]
    for rules in ({"exclude_origin": US}, {"preference": ["Canada"]}):
        reads.clear()
        cat(settings=rules)
        bask({**chosen, "settings": rules})
        r = client().post("/mealplan/resolve", json={"recipes": refs[:2], **rules})
        assert r.status_code == 200
        assert len(reads) == 3, rules
    reads.clear()
    cat()
    bask(chosen)
    assert client().post("/mealplan/resolve", json={"recipes": refs[:2]}).status_code == 200
    assert reads == []

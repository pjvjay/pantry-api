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
    _sql("DELETE FROM store_products WHERE product_id IN (:a, :b) AND store_id <> :far",
         a=TUNA, b=CHEDDAR, far=FAR_STORE)
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


def test_stores_prices_and_shares_come_from_the_offers_the_trips_read(chosen):
    """Every product once, its packs for every meal, its store the trip optimizer's for that
    one basket over the in-range offers; each line's cost its share of the packs by need."""
    from pantry_planner import tripopt
    from pantry_planner.config import settings
    from pantry_planner.nlsearch.sql_builder import build_price_matrix_sql

    body = bask(chosen)
    needs = _needs(chosen)
    buying: dict[int, list[tuple[str, int]]] = {}
    for r in body["recipes"]:
        for ln in r["lines"]:
            buying.setdefault(ln["product"]["id"], []).append((r["key"], ln["line_no"]))
    packs = {pid: max(1, math.ceil(float(sum(needs[c][0] for c in cs))
                                   / PRODUCTS[pid]["unit_qty"] - 1e-9))
             for pid, cs in buying.items()}
    cfg = settings()
    sql, params = build_price_matrix_sql(sorted(buying), cfg.default_lat, cfg.default_lon)
    rows = _sql(sql, **params)
    best = next(o for o in tripopt.optimize_trips(
        rows, [(pid, PRODUCTS[pid]["name"]) for pid in sorted(buying)],
        home_lat=cfg.default_lat, home_lon=cfg.default_lon,
        cost_per_km=cfg.travel_cost_per_km, packs=packs) if o.recommended)
    store_of = {it.product_id: it.store_name for it in best.items}
    row = {(r["store_name"], r["product_id"]): r for r in rows}
    cost = {pid: round(row[store_of[pid], pid]["price"] * packs[pid], 2) for pid in buying}
    assert body["est_total"] == pytest.approx(sum(cost.values()), abs=0.001)
    for r in body["recipes"]:
        for ln in r["lines"]:
            pid = ln["product"]["id"]
            at = row[store_of[pid], pid]
            assert ln["store"] == {"name": store_of[pid], "distance_km": round(
                math.sqrt(at["dist_km2"]), 1)}
            assert ln["price"] == at["price"]
            mine = needs[r["key"], ln["line_no"]][0]
            share = cost[pid] * float(mine / sum(needs[c][0] for c in buying[pid]))
            assert ln["cost_for_need"] == pytest.approx(share, abs=0.006)
    assert sum(ln["cost_for_need"] for r in body["recipes"] for ln in r["lines"]) == \
        pytest.approx(body["est_total"], abs=0.02)


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
    assert line["origin"] == {"country": "United States", "claim": "United States: processed "
                              "there, ingredients may be imported", "status": "evidenced",
                              "preferred_rank": None}
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
    assert origin[18] == {"country": "Canada", "claim": "Canada: origin", "status": "evidenced",
                          "preferred_rank": 2}
    assert origin[SALT] == {"country": None, "claim": "Origin not checked", "status": "unknown",
                            "preferred_rank": None}
    eggs = bask(starter_draft({"scrambled_eggs_on_toast": 1}))
    (egg,) = [ln for r in eggs["recipes"] for ln in r["lines"] if ln["product"]["id"] == 60]
    assert egg["origin"]["status"] == "conflicting" and egg["origin"]["country"] is None


def test_an_unresolved_recipe_lists_no_lines_and_unknown_amounts_have_no_cost():
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
    assert (salmon["need"], salmon["cost_for_need"], salmon["available"]) == (
        "amount unknown", None, True)
    assert salmon["price"] is not None
    (rice,) = got["my:k"]["lines"]
    assert rice["need"] == "400 g" and rice["cost_for_need"] is not None
    # the total sums the lines with a cost; the budget is the draft's
    assert body["est_total"] == pytest.approx(rice["cost_for_need"], abs=0.001)
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

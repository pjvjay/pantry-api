"""The alternatives ranking (alternatives.rank_alternatives): what else could fill a planned
line, in the planner's order, and what choosing it would do to the trip.

No LLM: plans are made in DEMO_MODE. Expected values come from the seed file, a re-price
(flow.reprice, tested on its own in test_reprice) or origins.filter_pool, the plan's own
origin filter, never from the ranking itself.
"""
from __future__ import annotations

import os
import random
import statistics
import time

import pytest

HOME = {"lat": 49.28, "lon": -123.12}
MIXED = ("Mixed\n- 500g penne\n- 2 cloves garlic\n- 1 can crushed tomatoes\n"
         "- 1 cup light brown sugar\n- 1 tsp cumin powder\n- 200g red onion\n")
_TIER = {"same": 0, "other": 1, "outside": 2}
_FIT = {"covers": 0, "unknown": 1, "short": 2}


@pytest.fixture(scope="module", autouse=True)
def seeded_db(tmp_path_factory):
    os.environ["DB_URL"] = (os.environ.get("PANTRY_TEST_DB_URL")
                            or f"sqlite:///{tmp_path_factory.mktemp('db') / 'alts.db'}")
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


@pytest.fixture(autouse=True)
def clean_evidence():
    from sqlalchemy.orm import Session

    from pantry_planner.db import ProductOriginEvidenceRow, engine

    with Session(engine()) as s:
        s.query(ProductOriginEvidenceRow).delete()
        s.commit()
    yield


def _evidence(pid, country, *, claim="made-in", ref=None, source="label-photo"):
    return dict(product_id=pid, source=source, source_ref=ref or f"{pid}.jpg",
                claim_type=claim, verbatim=f"{'Product of' if claim == 'product-of' else 'Made in'}"
                                           f" {country}" if country else "",
                ingredient_origin=country if claim == "product-of" else "",
                manufactured_in=country, confidence="high", importer_only=False, note="",
                observed_at="")


def _save(rows):
    from pantry_planner import db

    db.save_origin_evidence(rows)


def _id(name: str) -> int:
    from tests.seed_catalog import SEED_PRODUCTS

    return next(p["id"] for p in SEED_PRODUCTS if p["name"] == name)


def _line(plan, name):
    return next(ln for ln in plan.basis.lines if ln.name == name)


def _rank(plan, name, limit=25):
    from pantry_planner.alternatives import rank_alternatives

    return rank_alternatives(plan.basis, _line(plan, name).line_no, limit=limit)


def _by_name(ranking):
    return {it.product: it for it in ranking.items}


# ─── The key invariant: a row's trip is a real re-price ──────

def _located_plans():
    from pantry_planner import db, flow

    plans = [flow.run(r.slug, max_km=12, **HOME) for r in db.load_all_recipes()]
    plans.append(flow.run_nl(MIXED))
    return plans


def test_every_rows_trip_total_is_what_reprice_charges_to_the_cent():
    from pantry_planner import flow
    from pantry_planner.alternatives import rank_alternatives
    from pantry_planner.models import Pin

    checked = 0
    for plan in _located_plans():
        for ln in plan.basis.lines:
            if ln.product_id is None:
                continue
            ranking = rank_alternatives(plan.basis, ln.line_no)
            assert ranking.lines and ln.line_no in ranking.lines
            for it in ranking.items:
                assert it.trip is not None, (plan.recipe_slug, ln.name, it.product)
                again = flow.reprice(plan.basis, [Pin(line_no=n, product_id=it.product_id)
                                                  for n in ranking.lines])
                trip = next(o for o in again.trip_options if o.recommended)
                assert round(trip.total_cost * 100) == round(it.trip.total * 100), \
                    (plan.recipe_slug, ln.name, it.product)
                assert trip.stores == it.trip.stores
                if it.current:
                    assert it.trip.delta == 0.0 and it.trip.stops_delta == 0
                checked += 1
    assert checked > 100


def test_the_cart_pick_is_always_listed_and_flagged():
    from pantry_planner import flow

    plan = flow.run_nl(MIXED)
    for ln in plan.basis.lines:
        ranking = _rank(plan, ln.name, limit=1)
        current = [it for it in ranking.items if it.current]
        assert [it.product_id for it in current] == [ln.product_id]
        # In demo mode the plan's pick is the ranking's first, or its row says why not.
        assert current[0].rank == 1 or current[0].rank_reason.startswith("Below #")


# ─── Order ───────────────────────────────────────────────────

def _ingredient_word(line_name):
    """The line's last word that is not a colour, form or descriptor word."""
    from pantry_planner.nlsearch.planner import WEAK_SUGGESTION_WORDS
    from pantry_planner.nlsearch.units import tokens

    strong = [t for t in tokens(line_name) if t not in WEAK_SUGGESTION_WORDS]
    return strong[-1] if strong else ""


def _closeness(line_name, product):
    """semantic_key's shared words and fresh test, then: does the product's name have the
    line's ingredient word, and is the product mainly that word."""
    from pantry_planner.nlsearch.units import head_noun, semantic_key, tokens

    overlap, fresh, last = semantic_key(line_name, product)
    word = _ingredient_word(line_name)
    if not word:
        return overlap, fresh, 0, last
    return (overlap, fresh, int(word not in tokens(product.name)),
            int(head_noun(product.name) != word))


def _key_prefix(ranking, line_name, catalog, use_pref, prefs):
    out = []
    for it in ranking.items:
        out.append((_TIER[it.tier], _closeness(line_name, catalog[it.product_id]),
                    _FIT[it.pack_fit], prefs.get(it.product_id, 0) if use_pref else 0,
                    round(it.trip.total * 100) if it.trip else float("inf")))
    return out


def test_rows_never_go_up_on_tier_words_pack_fit_preference_then_trip():
    from pantry_planner import db, flow, origins

    _save([_evidence(_id("Penne Rigate 500g"), "Italy", claim="product-of"),
           _evidence(_id("Crushed Tomatoes Canned 796ml"), "Italy", claim="product-of"),
           _evidence(_id("Canned Diced Tomatoes"), "Canada")])
    catalog = {p.id: p for p in db.load_all_products()}
    plans = [*_located_plans(), flow.run_nl(MIXED, preference=["Italy", "Canada"])]
    for plan in plans:
        use_pref = bool(plan.basis.preference)
        resolved = origins.resolve_all() if use_pref else {}
        prefs = {pid: origins.preference_rank(o, plan.basis.preference)
                 for pid, o in resolved.items()}
        for ln in plan.basis.lines:
            if ln.product_id is None:
                continue
            ranking = _rank(plan, ln.name)
            keys = _key_prefix(ranking, ln.name, catalog, use_pref, prefs)
            assert [it.rank for it in ranking.items] == list(range(1, len(keys) + 1))
            assert keys == sorted(keys), (plan.recipe_slug, ln.name)


def test_a_line_ending_in_its_ingredient_word_keeps_the_demo_selectors_order():
    """The demo selector picks by units.semantic_key. Where the line's last word is its
    ingredient word, the ranking only breaks semantic_key's ties, so within a tier it never
    puts a product the selector thinks less close above a closer one."""
    from pantry_planner import db
    from pantry_planner.nlsearch.units import semantic_key, tokens

    catalog = {p.id: p for p in db.load_all_products()}
    checked = 0
    for plan in _located_plans():
        for ln in plan.basis.lines:
            toks = tokens(ln.name)
            if ln.product_id is None or not toks or _ingredient_word(ln.name) != toks[-1]:
                continue
            keys = [(_TIER[it.tier], semantic_key(ln.name, catalog[it.product_id]))
                    for it in _rank(plan, ln.name).items]
            assert keys == sorted(keys), (plan.recipe_slug, ln.name)
            checked += 1
    assert checked > 30


def test_a_line_ending_in_a_form_word_ranks_by_its_ingredient_word():
    """'cumin powder' is about cumin: another form of cumin ranks above powders that are not
    cumin, and no row is explained by 'powder'."""
    from pantry_planner import flow

    items = _rank(flow.run_nl(MIXED), "cumin powder").items
    rows = {it.product: it for it in items}
    seeds = rows["Cumin Seeds 100g"]
    assert seeds.rank < rows["Curry Powder 100g"].rank
    assert seeds.rank < rows["Chinese Five-Spice Powder 50g"].rank
    assert next(it for it in items if it.rank == seeds.rank + 1).rank_reason.endswith(
        "its name does not mention cumin.")
    assert not any("powder" in it.rank_reason for it in items)


def test_a_closer_but_dearer_product_ranks_above_a_cheaper_less_close_one():
    from pantry_planner import flow

    ranking = _rank(flow.run("grilled_cheese", max_km=12, **HOME), "Butter")
    rows = _by_name(ranking)
    salted, peanut = rows["Butter Salted 454g"], rows["Canadian Peanut Butter Cream"]
    assert salted.rank < peanut.rank
    assert salted.trip.total > peanut.trip.total


def test_rating_breaks_only_exact_cent_ties():
    """Rows in rating order but out of trip or cost order would mean rating outranked
    money; rating may only decide between rows equal to the cent on both."""
    for plan in _located_plans():
        for ln in plan.basis.lines:
            if ln.product_id is None:
                continue
            items = _rank(plan, ln.name).items
            for a, b in zip(items, items[1:], strict=False):
                if "rated lower" in b.rank_reason or "no reviews" in b.rank_reason:
                    assert a.trip.total == b.trip.total
                    assert a.cost_for_need == b.cost_for_need


def test_each_row_says_why_it_sits_below_the_one_above():
    from pantry_planner import flow

    items = _rank(flow.run_nl(MIXED), "crushed tomatoes").items
    assert items[0].rank_reason == "Ranked first: nothing ranks above it."
    assert all(it.rank_reason.startswith(f"Below #{it.rank - 1}: ") for it in items[1:])
    first_other = next(it for it in items if it.tier == "other")
    assert first_other.rank_reason.endswith("not the same ingredient.")


# ─── Match levels ────────────────────────────────────────────

def test_match_levels_and_tiers():
    from pantry_planner import flow

    plan = flow.run_nl(MIXED)
    sugar = _by_name(_rank(plan, "light brown sugar"))
    assert (sugar["Brown Sugar 1kg"].match, sugar["Brown Sugar 1kg"].tier) == ("generic", "same")
    # the head word: another sugar is related, not the same ingredient
    assert (sugar["Granulated Sugar 1kg"].match,
            sugar["Granulated Sugar 1kg"].tier) == ("related", "other")
    cumin = _by_name(_rank(plan, "cumin powder"))
    assert cumin["Cumin Ground 100g"].match == "form"
    # the head word skips the form word: cumin seeds relate, baking powder does not
    assert cumin["Cumin Seeds 100g"].match == "related"
    assert "Baking Powder 225g" not in cumin or cumin["Baking Powder 225g"].match == "substitute"


def test_a_thin_pool_adds_same_aisle_substitutes():
    from pantry_planner import flow
    from tests.seed_catalog import SEED_PRODUCTS

    ranking = _rank(flow.run_nl(MIXED), "garlic")
    same = [it for it in ranking.items if it.tier == "same"]
    subs = [it for it in ranking.items if it.match == "substitute"]
    assert len(same) < 3 and subs
    aisle = {p["id"]: p["subcategory"] for p in SEED_PRODUCTS}
    garlic_aisle = aisle[_id("Fresh Garlic")]
    assert all(aisle[it.product_id] == garlic_aisle for it in subs)
    assert all(it.reasons[0].text.startswith("Same aisle") for it in subs)


def test_a_product_appears_once_at_its_strictest_level():
    from pantry_planner import flow

    plan = flow.run_nl(MIXED)
    for ln in plan.basis.lines:
        ids = [it.product_id for it in _rank(plan, ln.name).items]
        assert len(ids) == len(set(ids))
    # Crushed Tomatoes match "crushed tomatoes" exactly and its head word "tomato" too
    tomatoes = _by_name(_rank(plan, "crushed tomatoes"))
    assert tomatoes["Crushed Tomatoes Canned 796ml"].match == "exact"


def test_one_options_query_and_at_most_one_substitute_query_per_call():
    from sqlalchemy import event

    from pantry_planner import db, flow
    from pantry_planner.alternatives import rank_alternatives

    plan = flow.run_nl(MIXED)
    statements: list[str] = []

    def record(_conn, _cursor, statement, *_a):
        statements.append(statement)

    eng = db.engine()
    event.listen(eng, "before_cursor_execute", record)
    try:
        for ln in plan.basis.lines:
            statements.clear()
            rank_alternatives(plan.basis, ln.line_no)
            assert sum("ing_counts(ing_no, ntok)" in st for st in statements) == 1, ln.name
            assert sum(":sub_subcat" in st or "p.subcategory = " in st
                       for st in statements if "ing_counts" not in st) <= 1, ln.name
            assert len(statements) <= 6, (ln.name, len(statements))
            for st in statements:
                compact = " ".join(st.split())
                assert "IN ()" not in compact and "VALUES ()" not in compact
    finally:
        event.remove(eng, "before_cursor_execute", record)


# ─── Honesty: unknown stays unknown ──────────────────────────

def test_an_unchecked_origin_is_never_a_country():
    from pantry_planner import flow

    for it in _rank(flow.run_nl(MIXED), "penne").items:
        assert (it.origin.status, it.origin.country, it.origin.label) == \
            ("unknown", "", "Origin not checked")
        assert any(r.code == "origin" and r.tone == "unknown" for r in it.reasons)


def test_conflicting_evidence_and_a_demo_label_photo_are_named():
    from pantry_planner import flow

    rigate, gf = _id("Penne Rigate 500g"), _id("Gluten-Free Penne 340g")
    _save([_evidence(rigate, "Italy", claim="product-of", ref="demo/penne-rigate-500g.jpg"),
           _evidence(gf, "United States", claim="product-of"),
           _evidence(gf, "Canada", claim="product-of", source="open-food-facts", ref="0001")])
    rows = _by_name(_rank(flow.run_nl(MIXED), "penne"))
    o = rows["Penne Rigate 500g"].origin
    assert (o.status, o.country, o.claim, o.verbatim, o.demo) == \
        ("resolved", "Italy", "full", "Product of Italy", True)
    assert any(r.text == 'Italy: origin: "Product of Italy" (demo label photo)'
               for r in rows["Penne Rigate 500g"].reasons)
    c = rows["Gluten-Free Penne 340g"].origin
    assert (c.status, c.country, c.label) == ("conflicting", "", "Sources disagree on origin")


def test_no_reviews_is_no_rating_not_zero_stars():
    from sqlalchemy import text
    from sqlalchemy.orm import Session

    from pantry_planner import db, flow

    gf = _id("Gluten-Free Penne 340g")
    plan = flow.run_nl(MIXED)
    try:
        with Session(db.engine()) as s:
            s.execute(text("DELETE FROM reviews WHERE product_id = :p"), {"p": gf})
            s.commit()
        rows = _by_name(_rank(plan, "penne"))
        assert rows["Gluten-Free Penne 340g"].rating is None
        assert any(r.text == "No reviews" and r.tone == "unknown"
                   for r in rows["Gluten-Free Penne 340g"].reasons)
        rated = rows["Penne Rigate 500g"].rating
        assert rated is not None and rated.count > 0 and rated.synthetic is True
    finally:
        db.seed_from_json()


def test_pack_fit_is_unknown_when_units_differ_or_no_amount_is_given():
    from pantry_planner import flow

    plan = flow.run_nl(MIXED)
    sugar = _by_name(_rank(plan, "light brown sugar"))["Brown Sugar 1kg"]
    # 1 cup is 250 ml; the bag is sold by weight: no conversion is guessed
    assert (sugar.pack_fit, sugar.cost_for_need) == ("unknown", None)
    assert any(r.text == "Pack in g, recipe in ml" for r in sugar.reasons)
    penne = _by_name(_rank(plan, "penne"))
    assert penne["Penne Rigate 500g"].pack_fit == "covers"
    assert penne["Gluten-Free Penne 340g"].pack_fit == "short"
    # 500 g of penne takes two 340 g bags: the cost of the need, not of the cart's one pack
    gf = penne["Gluten-Free Penne 340g"]
    assert gf.cost_for_need == round(2 * gf.offer.price, 2)
    classic = flow.run("tomato_penne", max_km=12, **HOME)
    for it in _rank(classic, classic.basis.lines[0].name).items:
        assert (it.pack_fit, it.cost_for_need) == ("unknown", None)
        assert any(r.text == "Recipe gives no amount" for r in it.reasons)


def test_the_data_note_follows_offers_synthetic(monkeypatch):
    from pantry_planner import config, flow

    plan = flow.run_nl(MIXED)
    assert _rank(plan, "penne").data_note == \
        "Store prices, stock at every store and reviews are demo data."
    monkeypatch.setenv("OFFERS_SYNTHETIC", "false")
    config.settings.cache_clear()
    try:
        ranking = _rank(plan, "penne")
        assert ranking.data_note == ""
        assert all(it.rating is None or it.rating.synthetic is False for it in ranking.items)
    finally:
        monkeypatch.delenv("OFFERS_SYNTHETIC")
        config.settings.cache_clear()


# ─── Exclusion parity ────────────────────────────────────────

def test_held_back_is_exactly_what_the_plans_origin_filter_drops():
    """Random evidence and random exclusions: everything held back is what filter_pool (the
    plan's own filter) drops, and nothing it drops is ever ranked."""
    from pantry_planner import db, flow, origins
    from pantry_planner.nlsearch import PlanAborted

    rng = random.Random(7)
    catalog = {p.id: p for p in db.load_all_products()}
    countries = ["United States", "China", "Mexico", "Italy", "Canada"]
    rows = []
    for pid in rng.sample(sorted(catalog), 70):
        rows.append(_evidence(pid, rng.choice(countries), claim=rng.choice(
            ["product-of", "made-in"])))
        if rng.random() < 0.2:      # a second, disagreeing source
            rows.append(_evidence(pid, rng.choice(countries), source="open-food-facts",
                                  ref=f"off-{pid}"))
    _save(rows)
    resolved = origins.resolve_all()
    checked = 0
    for _ in range(6):
        exclude = rng.sample(countries, rng.randint(1, 2))
        try:
            plan = flow.run_nl(MIXED, exclude=exclude, allow_partial=True)
        except PlanAborted:
            continue
        for ln in plan.basis.lines:
            if ln.product_id is None:
                continue
            ranking = _rank(plan, ln.name)
            ranked = {it.product_id for it in ranking.items}
            held = {h.product_id for h in ranking.held_back}
            assert not ranked & held
            _kept, dropped = origins.filter_pool(
                [catalog[i] for i in ranked | held], exclude=exclude, origins=resolved)
            assert {p.id for p, _c, _f in dropped} == held, (exclude, ln.name)
            for h in ranking.held_back:
                assert h.country in exclude
            checked += 1
    assert checked >= 10


def test_a_classic_plan_within_5_km_never_prices_at_the_far_store():
    """Red onion taken off the three stores within 5 km: MegaSave Richmond (about 14 km out)
    still sells it, so it is counted as unavailable, never offered."""
    from sqlalchemy import text
    from sqlalchemy.orm import Session

    from pantry_planner import db, flow

    red = _id("Red Onion")
    try:
        with Session(db.engine()) as s:
            s.execute(text("DELETE FROM store_products WHERE product_id = :p AND store_id <> 4"),
                      {"p": red})
            s.commit()
        plan = flow.run("chicken_curry", max_km=5, **HOME)
        onion = next(ln for ln in plan.basis.lines if "onion" in ln.name.lower())
        ranking = _rank(plan, onion.name)
        assert red not in {it.product_id for it in ranking.items}
        assert ranking.unavailable >= 1
        assert all(it.offer.store != "MegaSave Richmond" for it in ranking.items)
        assert all(it.offer.distance_km is not None and it.offer.distance_km <= 5
                   for it in ranking.items)
    finally:
        db.seed_from_json()


# ─── Merges ──────────────────────────────────────────────────

def test_choosing_another_lines_product_says_it_merges_and_reprice_agrees():
    from pantry_planner import flow
    from pantry_planner.models import Pin

    plan = flow.run_nl("Onions\n- 300g yellow onion\n- 200g red onion\n")
    yellow = _id("Yellow Onion")
    row = _by_name(_rank(plan, "red onion"))["Yellow Onion"]
    assert row.trip.merges_with_line == 1
    again = flow.reprice(plan.basis, [Pin(line_no=2, product_id=yellow)])
    (li,) = again.line_items
    assert li.also_lines == [2] and li.packs == row.packs == 3
    assert any("already bought for line 1" in r.text for r in row.reasons)


def test_a_shared_purchase_is_ranked_for_every_line_it_covers():
    from pantry_planner import flow
    from pantry_planner.alternatives import rank_alternatives

    plan = flow.run_nl("Onions\n- 300g yellow onion\n- 100g yellow onions\n")
    (shared,) = plan.line_items
    assert shared.also_lines == [2]
    ranking = rank_alternatives(plan.basis, 2)
    assert ranking.lines == [1, 2] and ranking.need == "400 g"
    assert ranking.ingredient == "yellow onion + yellow onions"
    current = next(it for it in ranking.items if it.current)
    assert current.pack_fit == "covers" and current.packs == shared.packs


# ─── Bounds and speed ────────────────────────────────────────

def test_an_unplanned_line_and_a_bad_limit_are_refused():
    from pantry_planner import flow
    from pantry_planner.alternatives import BasisError, PinError, rank_alternatives

    plan = flow.run_nl(MIXED)
    with pytest.raises(PinError, match="line 42 is not a planned line; planned: 1, 2, 3"):
        rank_alternatives(plan.basis, 42)
    with pytest.raises(BasisError, match="limit must be 1..25"):
        rank_alternatives(plan.basis, 1, limit=26)


@pytest.mark.skipif(bool(os.environ.get("PANTRY_TEST_DB_URL")),
                    reason="the timing guard is for SQLite in demo mode")
def test_a_twenty_line_recipe_ranks_one_line_in_under_300_ms():
    from pantry_planner import flow
    from pantry_planner.alternatives import rank_alternatives

    lines = ["500g penne", "2 cloves garlic", "1 can crushed tomatoes", "400g spaghetti",
             "1 cup light brown sugar", "1 tsp cumin powder", "2 yellow onions",
             "500g ground beef", "1 cup milk", "2 tbsp olive oil", "1 tsp salt",
             "1 tsp black pepper", "200g cheddar cheese", "1 red bell pepper", "2 carrots",
             "1 cup basmati rice", "2 eggs", "1 tbsp butter", "1 tsp paprika",
             "1 tbsp soy sauce"]
    plan = flow.run_nl("Big\n" + "\n".join(f"- {x}" for x in lines))
    assert plan.ingredient_count == 20
    runs = []
    for _ in range(3):
        t0 = time.perf_counter()
        rank_alternatives(plan.basis, 1)
        runs.append(time.perf_counter() - t0)
    assert statistics.median(runs) < 0.300, runs


# ─── The MCP tools ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_the_mcp_tools_rank_and_reprice_from_a_plans_basis():
    from pantry_planner.mcp_server import server

    res = await server.call_tool("plan_from_text", {"recipe_text": MIXED, "basis": True})
    plan = res.structured_content["summary"]
    basis = plan["basis"]
    ranked = (await server.call_tool("rank_alternatives",
                                     {"basis": basis, "line_no": 1, "limit": 3})
              ).structured_content
    assert ranked["line_no"] == 1 and len([i for i in ranked["items"] if not i["current"]]) <= 3
    assert sum(1 for i in ranked["items"] if i["current"]) == 1
    same = (await server.call_tool("reprice_plan", {"basis": basis})).structured_content
    timing = ("llm_cost_usd", "latency_ms", "llm_calls", "burr_run", "pipeline")
    assert {k: v for k, v in same["summary"].items() if k not in timing} == \
        {k: v for k, v in plan.items() if k not in timing}
    other = next(i for i in ranked["items"] if not i["current"])
    swapped = (await server.call_tool("reprice_plan", {
        "basis": basis, "pins": [{"line_no": 1, "product_id": other["product_id"]}]})
               ).structured_content["summary"]
    assert swapped["basis"]["pins"] == [{"line_no": 1, "product_id": other["product_id"]}]
    assert any(n.startswith("line 1 (penne): chosen by the shopper") for n in swapped["notes"])
    assert swapped["trip"]["total_cost"] == other["trip"]["total"]


@pytest.mark.asyncio
async def test_the_mcp_tools_refuse_with_the_reason():
    from mcp.server.mcpserver.exceptions import ToolError

    from pantry_planner.mcp_server import server

    basis = (await server.call_tool("plan_from_text", {"recipe_text": MIXED, "basis": True})
             ).structured_content["summary"]["basis"]
    with pytest.raises(ToolError, match="line 9 is not a planned line"):
        await server.call_tool("rank_alternatives", {"basis": basis, "line_no": 9})
    with pytest.raises(ToolError, match="unknown product id 999999 in pins"):
        await server.call_tool("reprice_plan", {
            "basis": basis, "pins": [{"line_no": 1, "product_id": 999999}]})
    with pytest.raises(ToolError, match="Unrecognised country"):
        await server.call_tool("reprice_plan", {
            "basis": {**basis, "exclude_origin": ["Atlantis"]}})
    with pytest.raises(ToolError):
        await server.call_tool("reprice_plan", {
            "basis": basis, "pins": [{"line_no": 1, "product_id": 1}] * 41})


# ─── The REST twins ──────────────────────────────────────────

def _client():
    from fastapi.testclient import TestClient

    from pantry_planner.api import app

    return TestClient(app)


def test_rest_twins_rank_and_reprice_like_the_functions():
    from pantry_planner import flow
    from pantry_planner.alternatives import rank_alternatives

    plan = flow.run_nl(MIXED)
    basis = plan.basis.model_dump(mode="json")
    c = _client()
    resp = c.post("/plan/alternatives", json={"basis": basis, "line_no": 1, "limit": 4})
    assert resp.status_code == 200, resp.text
    assert resp.json() == rank_alternatives(plan.basis, 1, 4).model_dump(mode="json")
    resp = c.post("/plan/reprice", json={"basis": basis})
    assert resp.status_code == 200, resp.text
    again = resp.json()
    assert again["total_cost"] == plan.total_cost and again["basis"]["pins"] == []
    assert again["routing_strategy"] == "reprice" and again["total_llm_cost_usd"] == 0.0
    other = next(i for i in c.post("/plan/alternatives", json={
        "basis": basis, "line_no": 1}).json()["items"] if not i["current"])
    swapped = c.post("/plan/reprice", json={
        "basis": basis, "pins": [{"line_no": 1, "product_id": other["product_id"]}]}).json()
    trip = next(o for o in swapped["trip_options"] if o["recommended"])
    assert trip["total_cost"] == other["trip"]["total"]


def test_rest_twins_answer_bad_input_with_a_code_and_the_reason():
    from pantry_planner import flow

    plan = flow.run_nl(MIXED)
    basis = plan.basis.model_dump(mode="json")
    c = _client()
    resp = c.post("/plan/alternatives", json={"basis": basis, "line_no": 9})
    assert resp.status_code == 422
    assert resp.json()["detail"]["error"] == "line_not_planned"
    assert "line 9 is not a planned line" in resp.json()["detail"]["detail"]
    resp = c.post("/plan/reprice", json={"basis": basis,
                                         "pins": [{"line_no": 1, "product_id": 999999}]})
    assert resp.status_code == 422 and resp.json()["detail"]["error"] == "invalid_pin"
    long = {**basis, "recipe_name": "x" * 201}
    resp = c.post("/plan/reprice", json={"basis": long})
    assert resp.status_code == 422 and resp.json()["detail"]["error"] == "invalid_basis"
    assert c.post("/plan/reprice", json={
        "basis": {**basis, "exclude_origin": ["Atlantis"]}}).status_code == 422
    assert c.post("/plan/reprice", json={
        "basis": basis, "pins": [{"line_no": 1, "product_id": 1}] * 41}).status_code == 422
    # a product the catalog no longer has: plan again
    gone = {**basis, "lines": [{**ln, "product_id": 999999} if ln["line_no"] == 1 else ln
                               for ln in basis["lines"]]}
    for path, body in (("/plan/reprice", {"basis": gone}),
                       ("/plan/alternatives", {"basis": gone, "line_no": 1})):
        resp = c.post(path, json=body)
        assert resp.status_code == 409 and resp.json()["detail"]["error"] == "stale_basis"


def test_rest_twins_have_their_own_sixty_a_minute_buckets(monkeypatch):
    from pantry_planner import flow, limits

    now = [1000.0]
    monkeypatch.setattr(limits, "monotonic", lambda: now[0])
    basis = flow.run_nl("P\n- 500g penne\n").basis.model_dump(mode="json")
    c = _client()
    for n in range(60):
        assert c.post("/plan/reprice", json={"basis": basis}).status_code == 200, n
    resp = c.post("/plan/reprice", json={"basis": basis})
    assert resp.status_code == 429 and resp.headers["Retry-After"] == "1"
    assert c.post("/plan/alternatives", json={"basis": basis, "line_no": 1}).status_code == 200

"""Re-pricing a finished plan from its basis (flow.reprice) and the pin checks it runs.

No LLM: plans are made in DEMO_MODE, and the re-price makes no call of either kind (the
tests patch both parsers and the selector to raise). Expected values come from the plan
being re-priced, the seed file or plain arithmetic, never from the code under test.
"""
from __future__ import annotations

import os

import pytest

TIMING = ("llm_cost_usd", "latency_ms", "llm_calls", "burr_run", "pipeline")
HOME = {"lat": 49.28, "lon": -123.12}
PASTA = "Garlic Pasta (serves 2)\n- 500g penne\n- 2 cloves garlic\n- 1 can crushed tomatoes\n"
ONIONS = "Onions\n- 300g yellow onion\n- 200g red onion\n"
US = ["United States"]


@pytest.fixture(scope="module", autouse=True)
def seeded_db(tmp_path_factory):
    os.environ["DB_URL"] = (os.environ.get("PANTRY_TEST_DB_URL")
                            or f"sqlite:///{tmp_path_factory.mktemp('db') / 'reprice.db'}")
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


def _mark(ids, country="United States"):
    from pantry_planner import db

    db.save_origin_evidence([
        dict(product_id=pid, source="label-photo", source_ref=f"{pid}.jpg", claim_type="made-in",
             verbatim=f"Made in {country}", ingredient_origin="", manufactured_in=country,
             confidence="high", importer_only=False, note="", observed_at="")
        for pid in ids])


def _id(name: str) -> int:
    from tests.seed_catalog import SEED_PRODUCTS

    return next(p["id"] for p in SEED_PRODUCTS if p["name"] == name)


def _summary(plan) -> dict:
    from pantry_planner.mcp_server import _summarize_plan

    out = _summarize_plan(plan).model_dump()
    for k in TIMING:
        out.pop(k)
    return out


def _accounted(plan) -> int:
    return (sum(1 + len(li.also_lines) for li in plan.line_items)
            + len(plan.not_stocked) + len(plan.out_of_range) + len(plan.skipped))


# ─── The round trip: no pins reproduces the plan ─────────────

def _plans():
    from pantry_planner import db, flow

    for r in db.load_all_recipes():
        yield f"{r.slug} (no location)", flow.run(r.slug)
        yield f"{r.slug} (5 km)", flow.run(r.slug, max_km=5, **HOME)
    yield "nl", flow.run_nl(PASTA)
    yield "nl within 10 km, no dairy", flow.run_nl(PASTA + "Notes: within 10 km, no dairy\n")
    yield "nl partial", flow.run_nl(PASTA + "- 1 tbsp unobtainium flakes\n- 2 cups water\n",
                                    allow_partial=True)


def test_a_reprice_with_no_pins_reproduces_every_plan():
    """Lines, packs, the trip and each line's trip store and price, the total, coverage and
    the left-out lists: everything but the timings. The library plans with a location
    break offer ties by distance and the NL plans by store id, so both are covered."""
    from pantry_planner import flow

    seen = 0
    for label, plan in _plans():
        again = flow.reprice(plan.basis)
        assert _summary(again) == _summary(plan), label
        assert again.servings == plan.servings, label
        assert again.basis.pins == [], label
        seen += 1
    assert seen >= 17


def test_round_trip_holds_with_an_origin_exclusion_and_coverage():
    from pantry_planner import flow

    _mark([_id("Gluten-Free Penne 340g"), _id("Canned Diced Tomatoes")])
    for plan in (flow.run_nl(PASTA, exclude=US, allow_partial=True),
                 flow.run("tomato_penne", exclude=US, max_km=10, **HOME),
                 flow.run_nl(PASTA, preference=["Italy"])):
        assert plan.origin_coverage is not None
        assert _summary(flow.reprice(plan.basis)) == _summary(plan)


def test_reprice_calls_no_parser_and_no_selector(monkeypatch):
    from pantry_planner import demomode, flow, selector
    from pantry_planner.nlsearch import planner

    plan = flow.run_nl(PASTA)

    def boom(*_a, **_k):
        raise AssertionError("an LLM boundary was called")

    for mod, name in ((demomode, "parse_recipe"), (demomode, "select_products"),
                      (planner, "parse_input"), (selector, "call_selector"),
                      (flow, "call_selector")):
        monkeypatch.setattr(mod, name, boom)
    again = flow.reprice(plan.basis)
    assert again.total_llm_cost_usd == 0.0 and again.routing_strategy == "reprice"


# ─── Pins ────────────────────────────────────────────────────

def test_a_pin_is_bought_and_named_as_the_shoppers_choice():
    from pantry_planner import flow
    from pantry_planner.models import Pin

    plan = flow.run_nl(PASTA)
    penne = next(ln for ln in plan.basis.lines if ln.name == "penne")
    gf = _id("Gluten-Free Penne 340g")
    assert penne.product_id != gf
    again = flow.reprice(plan.basis, [Pin(line_no=penne.line_no, product_id=gf)])
    li = next(li for li in again.line_items if li.line_no == penne.line_no)
    assert (li.product_id, li.confidence, li.model_used) == (gf, 1.0, "shopper")
    assert li.reasoning.startswith("chosen by the shopper in the cart (was Penne Rigate 500g)")
    assert "substitution" not in li.reasoning          # same ingredient
    assert again.basis.pins == [Pin(line_no=penne.line_no, product_id=gf)]
    assert again.total_cost == round(sum(x.price for x in again.line_items), 2)
    notes = _summary(again)["notes"]
    assert (f"line {penne.line_no} (penne): chosen by the shopper, Gluten-Free Penne 340g "
            "(was Penne Rigate 500g)") in notes
    assert _accounted(again) == again.ingredient_count


def test_pinning_the_planners_pick_is_no_pin_and_undoes_a_swap():
    from pantry_planner import flow
    from pantry_planner.models import Pin

    plan = flow.run_nl(PASTA)
    penne = next(ln for ln in plan.basis.lines if ln.name == "penne")
    same = flow.reprice(plan.basis, [Pin(line_no=penne.line_no, product_id=penne.product_id)])
    assert same.basis.pins == [] and _summary(same) == _summary(plan)
    swapped = flow.reprice(plan.basis, [Pin(line_no=penne.line_no,
                                            product_id=_id("Gluten-Free Penne 340g"))])
    undone = flow.reprice(swapped.basis, [Pin(line_no=penne.line_no,
                                              product_id=penne.product_id)])
    assert undone.basis.pins == [] and _summary(undone) == _summary(plan)


def test_a_product_from_another_aisle_is_named_a_substitution():
    from pantry_planner import flow
    from pantry_planner.models import Pin

    plan = flow.run_nl(ONIONS)
    again = flow.reprice(plan.basis, [Pin(line_no=2, product_id=_id("Zucchini"))])
    li = next(li for li in again.line_items if li.line_no == 2)
    assert li.reasoning.endswith("; substitution")
    assert "line 2 (red onion): substitution — Zucchini" in _summary(again)["notes"]


def test_choosing_another_lines_product_makes_one_purchase_with_summed_packs():
    """Red onion swapped for the yellow onion line 1 already buys: one purchase covering
    both lines, with packs for 300 g + 200 g of a 200 g onion."""
    import math

    from pantry_planner import flow
    from pantry_planner.models import Pin

    plan = flow.run_nl(ONIONS)
    yellow = _id("Yellow Onion")
    assert [li.product_id for li in plan.line_items if li.line_no == 1] == [yellow]
    again = flow.reprice(plan.basis, [Pin(line_no=2, product_id=yellow)])
    (li,) = again.line_items
    assert (li.line_no, li.also_lines, li.product_id) == (1, [2], yellow)
    assert li.packs == math.ceil((300 + 200) / 200)
    assert (li.need_qty, li.need_uom) == (500.0, "g")
    assert _accounted(again) == again.ingredient_count == 2


def test_pins_on_a_classic_plan_keep_every_line_accounted_for():
    from pantry_planner import flow
    from pantry_planner.models import Pin

    plan = flow.run("chicken_curry", max_km=10, **HOME)
    onion = next(ln for ln in plan.basis.lines if "onion" in ln.name.lower())
    again = flow.reprice(plan.basis, [Pin(line_no=onion.line_no, product_id=_id("Red Onion"))])
    assert _accounted(again) == again.ingredient_count == plan.ingredient_count
    assert all(li.store_name for li in again.line_items)


# ─── What a pin may not be ───────────────────────────────────

def test_an_excluded_origin_pin_names_the_country_and_the_field():
    from pantry_planner import flow
    from pantry_planner.alternatives import PinError
    from pantry_planner.models import Pin

    gf = _id("Gluten-Free Penne 340g")
    _mark([gf])
    plan = flow.run_nl(PASTA, exclude=US)
    penne = next(ln for ln in plan.basis.lines if ln.name == "penne")
    with pytest.raises(PinError, match=r"Gluten-Free Penne 340g is evidenced as United States "
                                       r"\(made or packed there\), which this plan excludes"):
        flow.reprice(plan.basis, [Pin(line_no=penne.line_no, product_id=gf)])


def test_a_pin_sold_only_beyond_max_km_names_the_nearest_offer():
    """Gluten-free penne taken off the three stores within 5 km: only MegaSave Richmond
    (about 14 km out) still sells it, and the error says so with its price there."""
    from sqlalchemy import text
    from sqlalchemy.orm import Session

    from pantry_planner import db, flow
    from pantry_planner.alternatives import PinError
    from pantry_planner.models import Pin

    gf = _id("Gluten-Free Penne 340g")
    plan = flow.run_nl(PASTA + "Notes: within 5 km\n")
    penne = next(ln for ln in plan.basis.lines if ln.name == "penne")
    try:
        with Session(db.engine()) as s:
            price = s.execute(text("SELECT price FROM store_products WHERE product_id = :p "
                                   "AND store_id = 4"), {"p": gf}).scalar_one()
            s.execute(text("DELETE FROM store_products WHERE product_id = :p AND store_id <> 4"),
                      {"p": gf})
            s.commit()
        with pytest.raises(PinError) as exc:
            flow.reprice(plan.basis, [Pin(line_no=penne.line_no, product_id=gf)])
        msg = str(exc.value)
        assert "no store within 5 km sells it" in msg
        assert "the nearest offer is MegaSave Richmond" in msg and f"${price:.2f}" in msg
    finally:
        db.seed_from_json()


def test_unknown_lines_products_and_too_many_pins_are_refused():
    from pantry_planner import flow
    from pantry_planner.alternatives import PinError
    from pantry_planner.models import Pin

    plan = flow.run_nl(PASTA)
    with pytest.raises(PinError, match=r"line 9 is not a planned line; planned: 1, 2, 3"):
        flow.reprice(plan.basis, [Pin(line_no=9, product_id=_id("Penne Rigate 500g"))])
    with pytest.raises(PinError, match="unknown product id 999999 in pins"):
        flow.reprice(plan.basis, [Pin(line_no=1, product_id=999999)])
    with pytest.raises(PinError, match="at most 40 pins, got 41"):
        flow.reprice(plan.basis, [Pin(line_no=1, product_id=1)] * 41)


def test_a_product_that_is_no_option_says_why():
    from pantry_planner import flow
    from pantry_planner.alternatives import PinError
    from pantry_planner.models import Pin

    plan = flow.run_nl(PASTA + "Notes: no dairy\n")
    garlic = next(ln for ln in plan.basis.lines if ln.name == "garlic")
    with pytest.raises(PinError, match=r"Whole Milk 1L is not an option for line 2 \(garlic\): "
                                       r"it contains dairy, which this plan excludes"):
        flow.reprice(plan.basis, [Pin(line_no=garlic.line_no, product_id=_id("Whole Milk 1L"))])
    with pytest.raises(PinError, match=r"it does not match the words of 'penne'"):
        flow.reprice(plan.basis, [Pin(line_no=1, product_id=_id("Fresh Ginger"))])


def test_a_basis_out_of_bounds_is_refused_before_any_query():
    from pantry_planner import flow
    from pantry_planner.alternatives import BasisError

    plan = flow.run_nl(PASTA)
    long = plan.basis.model_copy(update={"recipe_name": "x" * 201})
    with pytest.raises(BasisError, match="recipe_name is longer than 200"):
        flow.reprice(long)
    country = plan.basis.model_copy(update={"exclude_origin": ["Atlantis"]})
    with pytest.raises(BasisError, match="Unrecognised country"):
        flow.reprice(country)

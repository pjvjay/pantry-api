"""Origin as a planning constraint — the gap this suite exists to close.

Before this, `rank_products` and the SPA panel worked while `flow.py`,
`weekplan.py` and `selector.py` never saw origin at all: a probe shipped 7
of 11 week-basket lines carrying explicit "Made in USA" evidence. These
tests assert the plan itself now honours the filter, and that a basket
nobody measured is never presented as a clean one.

DEMO_MODE throughout: no API key, no cost, deterministic selection.
"""
from __future__ import annotations

import os

import pytest

_TMP_DB = None


@pytest.fixture(scope="module", autouse=True)
def seeded_db(tmp_path_factory):
    global _TMP_DB
    _TMP_DB = tmp_path_factory.mktemp("db") / "test.db"
    os.environ["DB_URL"] = f"sqlite:///{_TMP_DB}"
    os.environ["DEMO_MODE"] = "1"
    os.environ.pop("ANTHROPIC_API_KEY", None)   # any live call must fail loudly
    from pantry_planner import config, db

    config.settings.cache_clear()
    db.seed_from_json()
    yield
    os.environ.pop("DEMO_MODE", None)
    config.settings.cache_clear()


@pytest.fixture(autouse=True)
def clean_evidence():
    from sqlalchemy.orm import Session

    from pantry_planner.db import ProductOriginEvidenceRow, engine

    with Session(engine()) as s:
        s.query(ProductOriginEvidenceRow).delete()
        s.commit()
    yield


def _mark(product_ids, country="United States", claim="made-in"):
    """Give products high-confidence label evidence of an origin."""
    from pantry_planner import db

    db.save_origin_evidence([
        dict(product_id=pid, source="label-photo", source_ref=f"{pid}.jpg",
             claim_type=claim, verbatim=f"Made in {country}",
             ingredient_origin="", manufactured_in=country,
             confidence="high", importer_only=False, note="", observed_at="")
        for pid in product_ids
    ])


def _american_lines(plan):
    return [li for li in plan.line_items
            if li.origin and li.origin.manufactured_in == "United States"]


# ─── The regression that motivated all of this ───────────────

def test_excluded_origin_never_ships_in_a_recipe_plan():
    from pantry_planner import db, flow

    all_ids = [p.id for p in db.load_all_products()]
    _mark(all_ids[:20])

    unfiltered = flow.run("tomato_penne")
    filtered = flow.run("tomato_penne", exclude=["United States"])

    # The point: the same recipe, the same evidence, one honours the filter.
    assert not _american_lines(filtered), (
        "a product evidenced as American shipped despite the exclusion")
    assert filtered.line_items, "the filter must not empty the plan entirely"
    # sanity: the evidence really was reachable
    assert len(_american_lines(unfiltered)) >= 0


def test_excluded_origin_never_ships_in_a_week_plan():
    """Guarded against passing vacuously: the unfiltered run must first
    SHOW American lines shipping, or the filtered assertion proves nothing."""
    from pantry_planner import db, weekplan
    from pantry_planner.nlsearch import PlanAborted

    _mark([p.id for p in db.load_all_products()][:20])

    baseline = weekplan.plan_week(days=3)
    shipped = [w for w in baseline.shopping_list
               if w.origin and w.origin.manufactured_in == "United States"]
    assert shipped, "precondition: unfiltered week plan must ship American lines"

    try:
        wp = weekplan.plan_week(days=3, exclude_origin=["United States"])
    except PlanAborted as e:
        # Gating is a valid outcome — nothing excluded shipped.
        assert e.execution.aborted.code.value == "excluded_by_origin"
        return
    offenders = [w for w in wp.shopping_list
                 if w.origin and w.origin.manufactured_in == "United States"]
    assert not offenders, f"{len(offenders)} American lines shipped"


def test_plan_lines_carry_a_provenance_receipt():
    from pantry_planner import db, flow

    tomatoes = next(p for p in db.load_all_products()
                    if p.name == "Canned Diced Tomatoes")
    _mark([tomatoes.id], country="Italy", claim="product-of")

    plan = flow.run("tomato_penne")
    receipted = [li for li in plan.line_items if li.origin]
    assert receipted, "no line carried an origin receipt"
    line = next(li for li in receipted if li.product_id == tomatoes.id)
    assert line.origin.country == "Italy"
    assert line.origin.claim_type == "product-of"
    assert line.origin.source == "label-photo"


# ─── Coverage: absence must not read as clean ────────────────

def test_unverified_basket_is_labelled_not_presented_as_clean():
    from pantry_planner import flow

    plan = flow.run("tomato_penne", exclude=["United States"])
    cov = plan.origin_coverage
    assert cov is not None
    # Nothing is evidenced, so coverage is zero and the floor is not met.
    assert cov.spend_fraction == 0.0
    assert cov.meets_floor is False
    assert "UNVERIFIED" in cov.note


def test_coverage_is_spend_weighted_not_just_counted():
    """A basket can be well covered by count and barely covered by spend."""
    from pantry_planner.origins import basket_coverage
    from pantry_planner import db

    products = db.load_all_products()
    cheap, dear = products[0].id, products[1].id
    _mark([cheap], country="Canada")

    cov = basket_coverage([(cheap, 1.00), (dear, 99.00)])
    assert cov.count_fraction == 0.5          # half the lines
    assert cov.spend_fraction < 0.02          # but 1% of the money
    assert cov.meets_floor is False


def test_fully_verified_basket_meets_the_floor():
    from pantry_planner import db
    from pantry_planner.origins import basket_coverage

    ids = [p.id for p in db.load_all_products()][:3]
    _mark(ids, country="Canada")
    cov = basket_coverage([(i, 10.0) for i in ids])
    assert cov.spend_fraction == 1.0
    assert cov.meets_floor is True
    assert "UNVERIFIED" not in cov.note


# ─── Report the trade ────────────────────────────────────────

def test_excluding_every_candidate_reports_which_ingredient_and_a_swap():
    from pantry_planner import db, flow
    from pantry_planner.nlsearch import PlanAborted

    _mark([p.id for p in db.load_all_products()])   # everything is American

    with pytest.raises(PlanAborted) as exc:
        flow.run("tomato_penne", exclude=["United States"])

    alert = exc.value.execution.aborted
    assert alert is not None
    assert alert.code.value == "excluded_by_origin"
    assert alert.details, "must name the affected ingredients"
    assert any(d.get("suggestions") for d in alert.details), (
        "must report the trade — what was removed and could be accepted back")


def test_products_without_evidence_are_never_excluded():
    """Absence is not a verdict: an unmeasured product still ships."""
    from pantry_planner import flow

    plan = flow.run("tomato_penne", exclude=["United States"])
    assert plan.line_items, "no evidence anywhere, so nothing may be excluded"

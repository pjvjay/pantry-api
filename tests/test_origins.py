"""Provenance tests — no LLM calls, no network.

Origin is evidence-driven now, so these exercise the real path: write
evidence rows, resolve them, rank against a caller-supplied preference.
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
    from pantry_planner import config, db

    config.settings.cache_clear()
    db.seed_from_json()
    yield
    config.settings.cache_clear()


@pytest.fixture(autouse=True)
def clean_evidence():
    """Each test starts with an empty evidence table."""
    from sqlalchemy.orm import Session

    from pantry_planner.db import ProductOriginEvidenceRow, engine

    with Session(engine()) as s:
        s.query(ProductOriginEvidenceRow).delete()
        s.commit()
    yield


def _ev(product_id=1, **kw):
    base = dict(product_id=product_id, source="open-food-facts",
                claim_type="made-in", verbatim="", ingredient_origin="",
                manufactured_in="", confidence="medium", importer_only=False,
                note="", source_ref="", observed_at="")
    base.update(kw)
    return base


def _product(pid=1):
    from pantry_planner import db

    return next(p for p in db.load_all_products() if p.id == pid)


# ─── Country matching ────────────────────────────────────────

def test_word_boundary_stops_us_matching_australia():
    """The naive substring test that 'us' passes inside 'Australia'."""
    from pantry_planner.origins import country_matches

    assert not country_matches("Australia", "United States")
    assert not country_matches("Austria", "us")
    assert country_matches("Made in USA", "United States")
    assert country_matches("Product of the US", "United States")


def test_india_does_not_match_indiana():
    from pantry_planner.origins import country_matches

    assert not country_matches("Indiana, USA", "India")
    assert country_matches("Indiana, USA", "United States")


def test_us_state_names_resolve_to_united_states():
    """Measured case: Lindt Excellence 70% is 'New Hampshire, Stratham'."""
    from pantry_planner.origins import country_matches

    assert country_matches("New Hampshire, Stratham", "United States")
    assert country_matches("Vermont", "United States")


def test_georgia_is_not_treated_as_a_us_state():
    """Georgia is also a country — an ambiguous name must not fire."""
    from pantry_planner.origins import country_matches

    assert not country_matches("Georgia", "United States")


def test_aliases_and_subnational_canada():
    from pantry_planner.origins import country_matches

    assert country_matches("Quebec, Canada", "Canada")
    assert country_matches("British Columbia", "Canada")
    assert country_matches("Made in England", "United Kingdom")


# ─── Evidence resolution ─────────────────────────────────────

def test_agreeing_sources_resolve():
    from pantry_planner import db
    from pantry_planner.origins import resolve_all

    db.save_origin_evidence([
        _ev(manufactured_in="Canada", claim_type="product-of",
            verbatim="Product of Canada", confidence="high"),
    ])
    o = resolve_all([1])[1]
    assert o.status == "resolved"
    assert o.claim_type == "product-of"
    assert o.verbatim == "Product of Canada"


def test_disagreeing_sources_are_conflicting_not_resolved():
    """A conflict is not a country; nothing may pick a winner."""
    from pantry_planner import db
    from pantry_planner.origins import resolve_all

    db.save_origin_evidence([
        _ev(manufactured_in="Italy"),
        _ev(manufactured_in="Greece"),
    ])
    o = resolve_all([1])[1]
    assert o.status == "conflicting"
    assert "Greece" in o.manufactured_in and "Italy" in o.manufactured_in


def test_no_evidence_is_unknown_not_foreign():
    from pantry_planner.origins import resolve_all

    assert resolve_all([1])[1].status == "unknown"


def test_importer_only_evidence_never_resolves():
    """An 'Imported by ...' address is not a country of origin."""
    from pantry_planner import db
    from pantry_planner.origins import resolve_all

    db.save_origin_evidence([
        _ev(manufactured_in="Canada", importer_only=True,
            verbatim="Imported by Acme, Toronto"),
    ])
    assert resolve_all([1])[1].status == "unknown"


def test_full_claim_outranks_processing_claim_as_representative():
    from pantry_planner import db
    from pantry_planner.origins import resolve_all

    db.save_origin_evidence([
        _ev(manufactured_in="Canada", claim_type="prepared-in",
            confidence="high"),
        _ev(manufactured_in="Canada", claim_type="product-of",
            confidence="medium", verbatim="Product of Canada"),
    ])
    assert resolve_all([1])[1].claim_type == "product-of"


# ─── Ranking ─────────────────────────────────────────────────

def test_product_of_outranks_made_in_for_same_country():
    from pantry_planner import db
    from pantry_planner.origins import rank_products

    db.save_origin_evidence([
        _ev(product_id=1, manufactured_in="Canada", claim_type="product-of"),
        _ev(product_id=2, manufactured_in="Canada", claim_type="made-in"),
    ])
    r = rank_products([_product(1), _product(2)], preference=["Canada"])
    assert [x.product_id for x in r.ranked] == [1, 2]
    assert r.ranked[0].rank < r.ranked[1].rank
    assert "ingredients may be imported" in r.ranked[1].tier_label


def test_made_in_canada_with_us_ingredients_is_excluded():
    """The Kraft case: American peanuts, Canadian processing.

    A filter reading only the manufacturing country would pass this.
    """
    from pantry_planner import db
    from pantry_planner.origins import rank_products

    db.save_origin_evidence([
        _ev(product_id=1, manufactured_in="Canada",
            ingredient_origin="United States", claim_type="made-in",
            verbatim="Made in Canada from imported ingredients"),
    ])
    r = rank_products([_product(1)], preference=["Canada"],
                      exclude=["United States"])
    assert not r.ranked
    assert len(r.excluded) == 1
    assert r.excluded[0].matched_field == "ingredient_origin"
    assert r.excluded[0].excluded_country == "United States"


def test_exclusion_never_removes_a_product_for_lacking_evidence():
    from pantry_planner.origins import rank_products

    r = rank_products([_product(1)], preference=["Canada"],
                      exclude=["United States"])
    assert not r.excluded
    assert [u.reason for u in r.unranked] == ["no_evidence"]


def test_unranked_never_enters_the_ranking():
    from pantry_planner import db
    from pantry_planner.origins import rank_products

    db.save_origin_evidence([_ev(product_id=1, manufactured_in="Canada")])
    products = [_product(1), _product(2), _product(3)]
    r = rank_products(products, preference=["Canada"])
    assert len(r.ranked) == 1
    assert r.counts["unranked"] == 2
    ranked_ids = {x.product_id for x in r.ranked}
    assert not ranked_ids & {u.product_id for u in r.unranked}
    assert "not evidence" in r.coverage_note


def test_preference_order_is_respected():
    from pantry_planner import db
    from pantry_planner.origins import rank_products

    db.save_origin_evidence([
        _ev(product_id=1, manufactured_in="Mexico", claim_type="product-of"),
        _ev(product_id=2, manufactured_in="Canada", claim_type="product-of"),
    ])
    r = rank_products([_product(1), _product(2)],
                      preference=["Canada", "Mexico"])
    assert [x.product_id for x in r.ranked] == [2, 1]


def test_unpreferred_country_still_ranks_last_not_excluded():
    from pantry_planner import db
    from pantry_planner.origins import rank_products

    db.save_origin_evidence([
        _ev(product_id=1, manufactured_in="Italy", claim_type="product-of"),
    ])
    r = rank_products([_product(1)], preference=["Canada"])
    assert r.ranked[0].tier_label == "Other country"
    assert not r.excluded


def test_conflicting_products_are_held_out_of_ranking():
    from pantry_planner import db
    from pantry_planner.origins import rank_products

    db.save_origin_evidence([
        _ev(product_id=1, manufactured_in="Italy"),
        _ev(product_id=1, manufactured_in="Greece"),
    ])
    r = rank_products([_product(1)], preference=["Italy"])
    assert not r.ranked
    assert [u.reason for u in r.unranked] == ["conflicting"]


# ─── Triage ──────────────────────────────────────────────────

def test_triage_suggests_unresolved_products_only():
    from pantry_planner import db
    from pantry_planner.origins import triage_candidates

    products = db.load_all_products()
    before = {c["product_id"] for c in triage_candidates(products)}
    assert before, "seeded catalog should have triage candidates"

    pid = next(iter(before))
    db.save_origin_evidence([
        _ev(product_id=pid, manufactured_in="Canada", claim_type="product-of"),
    ])
    after = {c["product_id"] for c in triage_candidates(products)}
    assert pid not in after


def test_triage_returns_hints_never_countries():
    from pantry_planner import db
    from pantry_planner.origins import triage_candidates

    for c in triage_candidates(db.load_all_products()):
        assert set(c) == {"product_id", "product_name", "reason", "status"}

"""The origin eval must discriminate, not just run.

An eval that reports the same number regardless of the data underneath is
worse than no eval: it looks like measurement. These tests assert it moves
in the right direction, and that it catches the specific failure it exists
to catch — a brand-nationality guess resolving to a country.
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
    from sqlalchemy.orm import Session

    from pantry_planner.db import ProductOriginEvidenceRow, engine

    with Session(engine()) as s:
        s.query(ProductOriginEvidenceRow).delete()
        s.commit()
    yield


def _ev(pid, mfg, ing="", claim="made-in"):
    return dict(product_id=pid, source="label-photo", source_ref=f"{pid}.jpg",
                claim_type=claim, verbatim=f"Made in {mfg}",
                ingredient_origin=ing, manufactured_in=mfg,
                confidence="high", importer_only=False, note="", observed_at="")


def _run():
    from evals.origin_eval import load_cases, score, score_exclusion
    from pantry_planner.origins import resolve_all

    cases = load_cases()
    resolved = resolve_all([c["product_id"] for c in cases])
    return (cases, score(cases, resolved),
            score_exclusion(cases, resolved, "United States"))


def test_reports_zero_coverage_on_an_empty_corpus():
    """A low number here is a statement about the data, not the resolver."""
    _cases, s, _us = _run()
    assert s["coverage"] == 0.0
    assert s["resolved_count"] == 0


def test_coverage_and_accuracy_rise_with_evidence():
    from pantry_planner import db

    db.save_origin_evidence([
        _ev(8, "India", "India", "product-of"),
        _ev(19, "Canada", "Canada", "product-of"),
        _ev(43, "Italy", "Italy", "product-of"),
    ])
    _cases, s, _us = _run()
    assert s["coverage"] > 0.0
    assert s["accuracy"] == 1.0, "correct evidence must score as correct"


def test_the_split_case_is_caught_by_the_exclusion_filter():
    """American ingredients, Canadian processing — recall must catch it."""
    from pantry_planner import db

    db.save_origin_evidence([_ev(5, "Canada", "United States")])
    _cases, _s, us = _run()
    assert us["recall"] == 1.0
    assert not us["missed"]


def test_a_brand_nationality_guess_is_penalised_by_holdout():
    """Cadbury is a British brand that manufactures on several continents.
    Resolving it to the UK is the failure this eval exists to detect."""
    from pantry_planner import db

    _cases, before, _ = _run()
    db.save_origin_evidence([_ev(3, "United Kingdom")])
    _cases, after, _ = _run()
    assert after["holdout"] < before["holdout"], (
        "resolving a should-be-held-out product must lower the holdout score")


def test_conflicting_sources_count_as_correctly_held_out():
    from pantry_planner import db

    db.save_origin_evidence([_ev(16, "Italy"), _ev(16, "Greece")])
    _cases, s, _us = _run()
    assert s["holdout"] == 1.0, "a conflict correctly refuses to resolve"

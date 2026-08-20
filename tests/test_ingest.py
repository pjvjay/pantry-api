"""Ingest tests — fixtures mirror the claude-chrome-container output schemas.

No container, no network: the container is a batch producer of files, so
the integration surface is exactly these file formats.
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
def clean_tables():
    from sqlalchemy.orm import Session

    from pantry_planner.db import (
        PriceObservationRow,
        ProductOriginEvidenceRow,
        engine,
    )

    with Session(engine()) as s:
        s.query(ProductOriginEvidenceRow).delete()
        s.query(PriceObservationRow).delete()
        s.commit()
    yield


# ─── Product matching ────────────────────────────────────────

def test_match_threshold_rejects_weak_matches():
    """The container's 0.5 rule — the one that stopped 'bananas'
    resolving to a Moroccan yogurt."""
    from pantry_planner import db
    from pantry_planner.ingest import match_product

    products = db.load_all_products()
    assert match_product("Spaghetti Pasta 500g", products) is not None
    assert match_product("Yogurt Bnine Banana Morocco", products) is None


# ─── ./origin --json ─────────────────────────────────────────

ORIGIN_JSON = {
    "products": [
        {"product": "Spaghetti Pasta 500g", "claim_type": "made-in",
         "ingredient_origin": None, "manufactured_in": "Italy",
         "verbatim": "Made in Italy", "confidence": "medium", "note": None},
        {"product": "Cheddar Cheese Block 300g", "claim_type": "conflicting",
         "ingredient_origin": None, "manufactured_in": None,
         "verbatim": None, "confidence": "low",
         "note": "three records disagree"},
        {"product": "Nonexistent Imported Widget", "claim_type": "product-of",
         "ingredient_origin": "Peru", "manufactured_in": "Peru",
         "verbatim": "Product of Peru", "confidence": "high", "note": None},
    ],
    "usage": {"input_tokens": 3180, "output_tokens": 1335},
    "model": "claude-opus-5",
}


def test_ingest_origin_json_writes_evidence_and_reports_unmatched():
    from pantry_planner.ingest import ingest_origin_json

    result = ingest_origin_json(ORIGIN_JSON)
    assert result["written"] == 2
    assert result["unmatched"] == ["Nonexistent Imported Widget"]
    assert result["model"] == "claude-opus-5"


def test_conflicting_record_is_stored_but_never_resolves():
    from pantry_planner.ingest import ingest_origin_json
    from pantry_planner.origins import resolve_all

    ingest_origin_json(ORIGIN_JSON)
    resolved = resolve_all()
    spaghetti = next(o for o in resolved.values()
                     if o.product_name == "Spaghetti Pasta 500g")
    cheddar = next(o for o in resolved.values()
                   if o.product_name == "Cheddar Cheese Block 300g")
    assert spaghetti.status == "resolved" and spaghetti.manufactured_in == "Italy"
    assert cheddar.status == "unknown"     # stored, but carries no origin


# ─── ./label --json ──────────────────────────────────────────

LABEL_JSON = {
    "labels": [
        {"file": "photos/rice.jpg", "cached": False, "photo": "rice.jpg",
         "product": "Basmati Rice 2kg", "verbatim": "Product of India",
         "claim_type": "product-of", "country": "India",
         "importer_only": False, "confidence": "high", "note": None},
        {"file": "photos/blank.jpg", "cached": False, "photo": "blank.jpg",
         "product": "Frozen Broccoli 500g", "verbatim": None,
         "claim_type": "none", "country": None, "importer_only": False,
         "confidence": "low",
         "note": "This image does not show food product packaging"},
        {"file": "photos/tuna.jpg", "cached": True, "photo": "tuna.jpg",
         "product": "Canned Tuna Flaked 170g",
         "verbatim": "Imported by Acme Foods, Toronto ON",
         "claim_type": "packaged-in", "country": "Canada",
         "importer_only": True, "confidence": "low", "note": None},
    ]
}


def test_label_full_claim_populates_both_origin_fields():
    """'Product of India' asserts ingredient content, not just processing."""
    from pantry_planner.ingest import ingest_label_json
    from pantry_planner.origins import resolve_all

    ingest_label_json(LABEL_JSON)
    rice = next(o for o in resolve_all().values()
                if o.product_name == "Basmati Rice 2kg")
    assert rice.status == "resolved"
    assert rice.ingredient_origin == "India"
    assert rice.manufactured_in == "India"
    assert rice.verbatim == "Product of India"
    assert rice.source == "label-photo"


def test_label_with_no_declaration_is_recorded_but_carries_no_origin():
    from pantry_planner.ingest import ingest_label_json
    from pantry_planner.origins import resolve_all

    result = ingest_label_json(LABEL_JSON)
    assert "Frozen Broccoli 500g" in result["no_claim"]
    broccoli = next(o for o in resolve_all().values()
                    if o.product_name == "Frozen Broccoli 500g")
    assert broccoli.status == "unknown"


def test_importer_only_label_never_becomes_an_origin():
    from pantry_planner import db
    from pantry_planner.ingest import ingest_label_json
    from pantry_planner.origins import rank_products

    ingest_label_json(LABEL_JSON)
    tuna = next(p for p in db.load_all_products()
                if p.name == "Canned Tuna Flaked 170g")
    r = rank_products([tuna], preference=["Canada"])
    assert not r.ranked
    assert r.unranked[0].reason == "no_evidence"


# ─── ./grocery markdown ──────────────────────────────────────

GROCERY_MD = """\
Here are the results.

| Item | Store | Branch | Product | Size | Price | UnitPrice | Link |
|---|---|---|---|---|---|---|---|
| rice | superstore | Burnaby | Basmati Rice 2kg | 2kg | $12.99 | $0.65/100g | http://x/1 |
| rice | walmart | Campbell River | Long Grain White Rice 1kg | 1kg | $4.49 | — | http://x/2 |
| shrimp | superstore | Burnaby | Seaquest Pacific White Shrimp | 340g | $14.99 | — | http://x/3 |

## Notes

Walmart blocked after 3 searches; results partial.
"""


def test_parse_markdown_table_tolerates_surrounding_prose():
    from pantry_planner.ingest import parse_markdown_table

    rows = parse_markdown_table(GROCERY_MD)
    assert len(rows) == 3
    assert rows[0]["product"] == "Basmati Rice 2kg"
    assert rows[0]["branch"] == "Burnaby"


def test_grocery_ingest_keeps_unmatched_listings():
    """An unmatched listing is a real market observation; dropping it
    would hide how often catalog matching fails."""
    from pantry_planner import db
    from pantry_planner.ingest import ingest_grocery_markdown

    result = ingest_grocery_markdown(GROCERY_MD, run_id="run-1")
    assert result["written"] == 3
    assert result["matched"] == 2      # shrimp is not in the catalog
    rows = db.load_price_observations()
    unmatched = [r for r in rows if r.product_id is None]
    assert len(unmatched) == 1
    assert "Shrimp" in unmatched[0].product_name
    assert unmatched[0].branch == "Burnaby"


def test_price_parsing_keeps_raw_text():
    from pantry_planner import db
    from pantry_planner.ingest import ingest_grocery_markdown, parse_price

    assert parse_price("$12.99") == 12.99
    assert parse_price("$10.00 - $14.00") == 10.00   # range: first number
    assert parse_price("—") is None

    ingest_grocery_markdown(GROCERY_MD, run_id="run-1")
    row = next(r for r in db.load_price_observations()
               if r.product_name == "Basmati Rice 2kg")
    assert row.price == 12.99
    assert row.price_text == "$12.99"   # the range is never lost


# ─── Resolved-summary refresh ────────────────────────────────

def test_refresh_writes_summary_rows_with_status_counts():
    from pantry_planner import db
    from pantry_planner.ingest import ingest_label_json, refresh_resolved

    ingest_label_json(LABEL_JSON)
    result = refresh_resolved()
    assert result["by_status"]["resolved"] >= 1
    assert result["by_status"]["unknown"] >= 1

    summaries = db.load_resolved_origins()
    rice = next(r for r in summaries.values()
                if r.manufactured_in == "India")
    assert rice.status == "resolved"
    assert rice.claim_type == "product-of"


# ─── Regressions found by adversarial review ─────────────────

def test_single_token_query_does_not_match_an_unrelated_product():
    """'Milk 2L' reduces to {milk} and covered 100% of itself against
    'Milk Chocolate Cadbury', attaching a milk label to a chocolate bar."""
    from pantry_planner import db
    from pantry_planner.ingest import match_product

    products = db.load_all_products()
    assert match_product("Milk 2L", products) is None
    assert match_product("Cheese", products) is None
    # genuine multi-token matches still resolve
    assert match_product("Whole Milk", products).name == "Whole Milk 1L"
    assert match_product("Basmati Rice 2kg", products).name == "Basmati Rice 2kg"


def test_ambiguous_names_decline_rather_than_guess():
    from pantry_planner import db
    from pantry_planner.ingest import match_product

    products = [p for p in db.load_all_products() if "Ground Beef" in p.name]
    assert len(products) > 1
    assert match_product("Ground Beef", products) is None


def test_reingesting_the_same_file_is_a_noop():
    """evidence_count is surfaced as corroboration, so duplicates would
    overstate how well-supported a provenance claim is."""
    from pantry_planner.ingest import ingest_origin_json

    first = ingest_origin_json(ORIGIN_JSON)
    second = ingest_origin_json(ORIGIN_JSON)
    assert first["written"] == 2
    assert second["written"] == 0

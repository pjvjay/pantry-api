"""The reality check: the catalog against a real store's shelf."""
from __future__ import annotations

import json
import os

import pytest

US = "United States"


@pytest.fixture(scope="module", autouse=True)
def seeded_db(tmp_path_factory):
    db_file = tmp_path_factory.mktemp("db") / "test.db"
    os.environ["DB_URL"] = os.environ.get("PANTRY_TEST_DB_URL") or f"sqlite:///{db_file}"
    os.environ["DEMO_MODE"] = "1"
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


def _mark(fragment, country=US, exact=False):
    from pantry_planner import db

    db.save_origin_evidence([
        dict(product_id=p.id, source="label-photo", source_ref=f"{p.id}.jpg",
             claim_type="product-of", verbatim=f"Product of {country}",
             ingredient_origin=country, manufactured_in=country, confidence="high",
             importer_only=False, note="", observed_at="")
        for p in db.load_all_products()
        if (p.name.lower() == fragment if exact else fragment in p.name.lower())])


def _obs(**kw):
    row = {"ingredient": "Yellow Onion", "store": "Real Canadian Superstore", "branch": "test",
           "product": "Yellow Onions, 3 lb Bag", "size": "1.36 kg", "qty": 1360, "uom": "g",
           "price": 3.47, "regular_price": None, "origin": None, "origin_source": None,
           "observed_at": "2026-10-07", "link": ""}
    return {**row, **kw}


def test_unit_prices_per_kg_l_and_each():
    from evals.reality_eval import unit_price

    assert unit_price(3.47, 1360, "g") == pytest.approx((2.551, "kg"), rel=1e-3)
    assert unit_price(1.99, 796, "ml")[1] == "L"
    assert unit_price(4.0, 12, "each") == pytest.approx((1 / 3, "each"))
    assert unit_price(None, 100, "g") is None and unit_price(1.0, None, "g") is None


def test_an_ingredient_with_only_excluded_candidates_fails():
    from evals.reality_eval import catalog_rows

    _mark("garlic")
    [row] = catalog_rows(["Garlic"], exclusions=[US])
    assert row.candidates and not row.without[US]
    assert row.failures == [f"no candidate without {US}"]


def test_one_excluded_pack_among_several_is_fine():
    from evals.reality_eval import catalog_rows

    _mark("yellow onion", exact=True)        # the loose onion, not the bags
    [row] = catalog_rows(["Yellow Onion"], exclusions=[US])
    assert len(row.candidates) > 1 and row.without[US] and not row.failures


def test_price_far_from_the_reference_is_flagged():
    from evals.reality_eval import catalog_rows, compare

    rows = catalog_rows(["Yellow Onion"], exclusions=[])
    compare(rows, [_obs(price=40.0)])
    assert any("the reference" in f for f in rows[0].failures)
    rows = catalog_rows(["Yellow Onion"], exclusions=[])
    compare(rows, [_obs()])
    assert rows[0].failures == []


def test_an_origin_the_store_sells_and_the_catalog_lacks_is_flagged():
    from evals.reality_eval import catalog_rows, compare

    rows = catalog_rows(["Yellow Onion"], exclusions=[])
    compare(rows, [_obs(origin="Peru", origin_source="label")])
    assert rows[0].failures == ["the reference store sells it from Peru; the catalog does not"]
    _mark("yellow onions 3lb", country="Peru")
    rows = catalog_rows(["Yellow Onion"], exclusions=[])
    compare(rows, [_obs(origin="Peru", origin_source="label")])
    assert rows[0].failures == []


def test_the_shipped_basket_loads_and_names_library_ingredients():
    from evals.reality_eval import library_ingredients, load_basket

    rows = load_basket()
    assert rows
    assert {r["ingredient"] for r in rows} <= set(library_ingredients())


def test_a_row_missing_a_field_is_refused(tmp_path):
    from evals.reality_eval import load_basket

    path = tmp_path / "basket.json"
    path.write_text(json.dumps({"observations": [_obs(price=None)]}))
    with pytest.raises(ValueError, match="price"):
        load_basket(path)


def test_report_lists_what_to_capture(tmp_path):
    from evals.reality_eval import main

    out = tmp_path / "reality.md"
    assert main(["--out", str(out)]) == 0
    text = out.read_text()
    assert "## Per ingredient" in text and "## To capture" in text
    assert "| Yellow Onion | 4 | 4 |" in text

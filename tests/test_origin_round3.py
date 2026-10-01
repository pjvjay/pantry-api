"""Round-three findings, each asserted as the RIGHT output.

The headline: head-noun (relaxed) matches are not candidates. Unioning them
into the candidate set let "Basmati Rice" survive on Gluten-Free Penne and
ship it — 13 of 13 such shapes on the seeded catalog. They are alternatives
the gate may OFFER, stated as a trade, never silently substituted.
"""
from __future__ import annotations

import os

import pytest

_TMP_DB = None
US = ["United States"]
TEXT = "Garlic Pasta (serves 2)\n- 500g penne\n- 2 cloves garlic\n- 1 can crushed tomatoes\n"


@pytest.fixture(scope="module", autouse=True)
def seeded_db(tmp_path_factory):
    global _TMP_DB
    _TMP_DB = tmp_path_factory.mktemp("db") / "test.db"
    os.environ["DB_URL"] = os.environ.get("PANTRY_TEST_DB_URL") or f"sqlite:///{_TMP_DB}"
    os.environ["DEMO_MODE"] = "1"
    os.environ.pop("ANTHROPIC_API_KEY", None)
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


def _mark(ids, country="United States", claim="made-in", source="label-photo"):
    from pantry_planner import db

    db.save_origin_evidence([
        dict(product_id=pid, source=source, source_ref=f"{pid}.jpg", claim_type=claim,
             verbatim=f"Made in {country}", ingredient_origin="", manufactured_in=country,
             confidence="high", importer_only=False, note="", observed_at="")
        for pid in ids])


def _ids_named(fragment):
    from pantry_planner import db

    return [p.id for p in db.load_all_products() if fragment in p.name.lower()]


def _pools(recipe):
    from pantry_planner import flow
    from pantry_planner.config import settings

    cfg = settings()
    return flow._ingredient_pools(recipe, cfg.default_lat, cfg.default_lon)


# ─── B: relaxed matches are alternatives, never candidates ────────────

def _under_gate_shapes():
    """(recipe, ingredient) pairs with direct candidates AND relaxed survivors."""
    from pantry_planner import db

    out = []
    for recipe in db.load_all_recipes():
        for k, pool in _pools(recipe).items():
            if pool["direct"] and pool["relaxed"]:
                out.append((recipe, recipe.ingredients[k].name,
                            {p.id for p in pool["direct"]},
                            {p.id for p in pool["relaxed"]}))
    return out


def test_seeded_catalog_has_under_gate_shapes():
    assert len(_under_gate_shapes()) >= 10, "precondition for the tests below"


def test_excluding_every_direct_candidate_gates_even_when_relaxed_matches_survive():
    """The 13-of-13 defect: before, each of these shipped a non-candidate."""
    from sqlalchemy.orm import Session

    from pantry_planner import flow
    from pantry_planner.db import ProductOriginEvidenceRow, engine
    from pantry_planner.nlsearch import PlanAborted

    checked = 0
    for recipe, ing_name, direct, _relaxed in _under_gate_shapes():
        with Session(engine()) as s:
            s.query(ProductOriginEvidenceRow).delete()
            s.commit()
        _mark(direct)
        with pytest.raises(PlanAborted) as exc:
            flow.run(recipe.slug, exclude=US)
        alert = exc.value.execution.aborted
        assert alert.code.value == "excluded_by_origin"
        names = {d["name"] for d in alert.details}
        assert ing_name in names, f"{recipe.slug}: gate did not name {ing_name!r}"
        checked += 1
    assert checked >= 10


def test_gate_offers_surviving_alternatives_as_a_stated_trade():
    from pantry_planner import db, flow
    from pantry_planner.nlsearch import PlanAborted

    recipe, ing_name, direct, relaxed = _under_gate_shapes()[0]
    _mark(direct)
    with pytest.raises(PlanAborted) as exc:
        flow.run(recipe.slug, exclude=US)
    detail = next(d for d in exc.value.execution.aborted.details if d["name"] == ing_name)
    offered = [s for s in detail["suggestions"] if s.startswith("still available")]
    assert offered, "the gate must offer the alternatives it is holding"
    names = {p.name for p in db.load_all_products() if p.id in relaxed}
    assert any(n in s for s in offered for n in names)


def test_every_plan_line_is_one_of_its_own_direct_candidates():
    from pantry_planner import db, flow

    _mark([p.id for p in db.load_all_products()][::3])
    for recipe in db.load_all_recipes():
        pools = _pools(recipe)
        direct = {recipe.ingredients[k].name: {p.id for p in v["direct"]} for k, v in pools.items()}
        try:
            plan = flow.run(recipe.slug, exclude=US)
        except Exception:
            continue                                     # gated: covered above
        for li in plan.line_items:
            if direct.get(li.ingredient_name):
                assert li.product_id in direct[li.ingredient_name], (
                    f"{recipe.slug}: {li.ingredient_name!r} -> {li.product_name!r}")


# ─── cap-8 residual on the NL path and the week planner ──────────────

def _nine_garlic_twins():
    from sqlalchemy.orm import Session

    from pantry_planner.db import ProductRow, ProductTermRow, StoreProductRow, StoreRow, engine

    ids = []
    with Session(engine()) as s:
        store_id = s.query(StoreRow.id).first()[0]
        for i in range(9):
            pid = 9100 + i
            s.add(ProductRow(id=pid, name=f"Garlic Bulb Pack {i}", description="fresh garlic",
                             price=1.00 + i * 0.10, category="produce", subcategory="vegetables"))
            s.add(StoreProductRow(store_id=store_id, product_id=pid, price=1.00 + i * 0.10))
            for term in ("garlic", "bulb", "pack", "fresh"):
                s.add(ProductTermRow(term=term, product_id=pid))
            ids.append(pid)
        s.commit()
    return ids


def _drop_twins(ids):
    from sqlalchemy.orm import Session

    from pantry_planner.db import ProductRow, ProductTermRow, StoreProductRow, engine

    with Session(engine()) as s:
        s.query(ProductTermRow).filter(ProductTermRow.product_id.in_(ids)).delete()
        s.query(StoreProductRow).filter(StoreProductRow.product_id.in_(ids)).delete()
        s.query(ProductRow).filter(ProductRow.id.in_(ids)).delete()
        s.commit()


def test_nl_path_finds_the_ninth_garlic_past_the_cheapest_eight():
    from pantry_planner import flow

    ids = _nine_garlic_twins()
    try:
        _mark(ids[:8] + _ids_named("fresh garlic"))
        plan = flow.run_nl(TEXT, exclude=US)
        garlic = next(li for li in plan.line_items if li.ingredient_name == "garlic")
        assert garlic.product_id == ids[8]
    finally:
        _drop_twins(ids)


def test_week_planner_finds_the_ninth_garlic_past_the_cheapest_eight():
    from pantry_planner import weekplan

    ids = _nine_garlic_twins()
    try:
        _mark(ids[:8] + _ids_named("fresh garlic"))
        wp = weekplan.plan_week(days=7, exclude_origin=US)
        garlic_notes = [n for n in wp.notes if "Garlic" in n and "excluded origin" in n]
        assert not garlic_notes, f"recipes wrongly skipped: {garlic_notes}"
        assert ids[8] in {w.product_id for w in wp.shopping_list}
    finally:
        _drop_twins(ids)


# ─── D: aliases must not fire inside other countries' names ──────────

@pytest.mark.parametrize("evidence,country", [
    ("Made in the People's Republic of China", "Taiwan"),
    ("Made in the People’s Republic of China", "Taiwan"),
    ("Lao Gan Ma chili crisp", "Laos"),
    ("Macedonia, Greece", "North Macedonia"),
    ("Democratic Republic of the Congo", "Republic of the Congo"),
    ("Republic of the Congo", "Democratic Republic of the Congo"),
    ("Republic of China (Taiwan)", "China"),
])
def test_no_cross_country_false_match(evidence, country):
    from pantry_planner.origins import country_matches

    assert not country_matches(evidence, country)


@pytest.mark.parametrize("evidence,country", [
    ("Made in Taiwan", "Taiwan"), ("People's Republic of China", "China"),
    ("Vientiane, Laos", "Laos"), ("Skopje, North Macedonia", "North Macedonia"),
    ("Kinshasa, DR Congo", "Democratic Republic of the Congo"),
    ("Brazzaville, Republic of the Congo", "Republic of the Congo"),
])
def test_true_matches_still_hold(evidence, country):
    from pantry_planner.origins import country_matches

    assert country_matches(evidence, country)


# ─── E: ambiguity is reported, not silently resolved to both ──────────

def test_bare_congo_is_ambiguous_and_both_congos_are_accepted_by_name():
    from pantry_planner.origins import validate_countries

    assert validate_countries(["Congo"]) == {
        "Congo": ["Republic of the Congo", "Democratic Republic of the Congo"]}
    assert validate_countries(["Republic of the Congo", "DR Congo", "Congo-Brazzaville"]) == {}
    assert "Macedonia" in validate_countries(["Macedonia"])


def test_previously_rejected_legitimate_spellings_are_accepted():
    from pantry_planner.origins import validate_countries

    ok = ["Bosnia & Herzegovina", "Trinidad & Tobago", "Antigua & Barbuda", "St. Lucia",
          "St Kitts and Nevis", "St Vincent", "Moldova, Republic of",
          "Tanzania, United Republic of", "Iran, Islamic Republic of",
          "Venezuela, Bolivarian Republic of", "Bolivia, Plurinational State of",
          "Micronesia, Federated States of", "Kyrgyz Republic", "Republic of Ireland",
          "Deutschland", "Italia", "España", "Brasil", "Curaçao", "Bermuda", "Guam",
          "Macau", "Macao"]
    assert validate_countries(ok) == {}


# ─── H: the week gate's message and details name the same ingredients ─

def test_week_gate_message_and_details_agree_when_more_than_twelve_are_emptied():
    from pantry_planner import db, weekplan
    from pantry_planner.nlsearch import PlanAborted

    _mark([p.id for p in db.load_all_products()])
    with pytest.raises(PlanAborted) as exc:
        weekplan.plan_week(days=3, exclude_origin=US)
    alert = exc.value.execution.aborted
    detail_names = [d["name"] for d in alert.details]
    for n in detail_names:
        assert n in alert.message, f"details name {n!r} but the message does not"


# ─── engine: one Engine per URL ──────────────────────────────────────

def test_engine_is_cached_per_url():
    from pantry_planner import db

    assert db.engine() is db.engine()

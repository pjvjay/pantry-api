"""The meal plan's seed files: the labelled demo products, the cited storage times and the
demo starter recipes.

Expected values come from the seed files read directly, never from the loaders under test.
"""
from __future__ import annotations

import json

from pantry_planner.db import SEEDS_DIR


def _load(name: str):
    return json.loads((SEEDS_DIR / name).read_text(encoding="utf-8"))


# ─── Demo products ───────────────────────────────────────────

def test_the_demo_products_are_labelled_and_appear_in_the_catalog_byte_for_byte():
    demo = _load("demo_products.json")
    assert demo["synthetic"] is True and demo["label"] == "synthetic demo product"
    assert [p["id"] for p in demo["products"]] == [166, 167, 168, 169]

    def rows(name: str) -> list[str]:
        text = (SEEDS_DIR / name).read_text(encoding="utf-8")
        return [ln.strip().rstrip(",") for ln in text.splitlines()
                if ln.strip().startswith('{"id": ')]

    catalog = rows("products.json")
    for row in rows("demo_products.json"):
        assert row in catalog, row
    by_id = {p["id"]: p for p in _load("products.json")}
    for p in demo["products"]:
        assert by_id[p["id"]] == p


# ─── Cited storage times (seeds/shelf_life.json) ─────────────

SHELF = _load("shelf_life.json")


def test_every_rule_cites_a_source_with_a_url_and_a_page_date():
    sources = {s["id"]: s for s in SHELF["sources"]}
    for s in sources.values():
        assert s["url"].startswith("https://") and s["page_date"] and s["credit"], s["id"]
    for r in [*SHELF["rules"], *SHELF["thaw_rules"]]:
        assert r["source"] in sources and r["verbatim"].strip(), r["id"]
    for r in SHELF["rules"]:
        if r["days_min"] is not None:
            assert 0 <= r["days_min"] <= r["days_max"], r["id"]
        # A time in months or years carries no day count: it only means "longer than a plan".
        if (r.get("stated") or {}).get("unit") in {"month", "year"}:
            assert r["days_min"] is None and r["days_max"] is None, r["id"]


def test_every_catalog_product_is_listed_once_and_every_mapping_names_real_rules():
    catalog = {p["id"] for p in _load("products.json")}
    listed = [p["product_id"] for p in SHELF["products"]] + [
        u["product_id"] for u in SHELF["unmapped"]]
    assert sorted(listed) == sorted(catalog)
    rule_ids = {r["id"] for r in [*SHELF["rules"], *SHELF["thaw_rules"]]}
    for p in SHELF["products"]:
        for ids in p["rules"].values():
            assert set(ids) <= rule_ids, p["product_id"]


def test_an_unmapped_product_has_no_number():
    from pantry_planner.mealplan import shelf

    for pid in (24, 12):          # Mozzarella Shredded 200g, Yellow Onion: no cited row
        ps = shelf.for_product(pid)
        assert (ps.mapped, ps.fridge_days, ps.fridge, ps.freezer) == (False, None, (), ())
        assert ps.reason
    assert shelf.thaw_lead(shelf.for_product(24), 1.0) is None


def test_plans_use_the_lower_bound_and_the_shorter_of_two_rows():
    from pantry_planner.mealplan import shelf

    rules = {r["id"]: r for r in SHELF["rules"]}
    assert rules["fs-poultry-pieces-fridge"]["verbatim"] == "1 to 2 days"
    assert shelf.for_product(10).fridge_days == 1      # chicken breast: the lower bound
    two = shelf.ProductShelf(product_id=0, mapped=True, storage_class="chilled_or_fresh",
                             fridge=("fs-shrimp-fridge", "fs-poultry-pieces-fridge"))
    assert two.fridge_days == min(rules["fs-shrimp-fridge"]["days_min"],
                                  rules["fs-poultry-pieces-fridge"]["days_min"])


def test_thaw_leads_come_from_the_cited_thaw_rows():
    import math

    from pantry_planner.mealplan import shelf

    thaw = {r["id"]: r for r in SHELF["thaw_rules"]}
    per_kg = thaw["fsis-thaw-fridge-large"]["lead"]["per_kg_derived"]
    chicken = shelf.for_product(11)
    small = shelf.thaw_lead(chicken, 0.7)
    assert (small.days, small.rule_ids) == (1, ("fsis-thaw-fridge-small",))
    large = shelf.thaw_lead(chicken, 3.0)
    assert (large.days, large.rule_ids) == (math.ceil(3.0 / per_kg), ("fsis-thaw-fridge-large",))
    unknown = shelf.thaw_lead(chicken, None)
    assert (unknown.days, unknown.weight_known) == (1, False)


def test_only_products_with_a_freezer_time_in_months_or_years_are_frozen():
    from pantry_planner.mealplan import shelf

    assert shelf.for_product(11).freezable                 # 9 months
    assert not shelf.for_product(60).freezable             # eggs: "Do not freeze in shell"
    assert shelf.for_product(158).bought_frozen            # frozen shrimp


def test_get_shelf_life_quotes_cited_rows_and_gives_no_number_for_unknowns(tmp_path_factory):
    import os

    from fastapi.testclient import TestClient

    from pantry_planner import config, db
    from pantry_planner.api import app

    os.environ["DB_URL"] = (os.environ.get("PANTRY_TEST_DB_URL")
                            or f"sqlite:///{tmp_path_factory.mktemp('db') / 'shelf.db'}")
    config.settings.cache_clear()
    db.seed_from_json()
    body = TestClient(app).get("/shelf-life", params={"product_id": [10, 24]}).json()
    assert [s["id"] for s in body["sources"]] == [s["id"] for s in SHELF["sources"]]
    chicken, mozzarella = body["products"]
    assert chicken["status"] == "cited"
    assert [r["verbatim"] for r in chicken["rules"]["fridge"]] == ["1 to 2 days"]
    assert mozzarella["status"] == "unknown" and mozzarella["rules"] == {}
    assert body["rules_of_use"] == SHELF["rules_of_use"]

    def numbers(x):
        if isinstance(x, dict):
            return [n for k, v in x.items() if k != "product_id" for n in numbers(v)]
        if isinstance(x, list):
            return [n for v in x for n in numbers(v)]
        return [x] if isinstance(x, int | float) and not isinstance(x, bool) else []

    assert numbers(mozzarella) == []

"""The meal plan's seed files: the labelled demo products, the cited storage times and the
demo starter recipes.

Expected values come from the seed files read directly, never from the loaders under test.
"""
from __future__ import annotations

import json

import pytest

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


# ─── Slots and diets (seeds/mealplan_starters.json) ──────────

STARTERS = _load("mealplan_starters.json")
LIBRARY = {r["slug"]: r for r in _load("recipes.json")}


def _all_meta() -> dict[str, dict]:
    """recipe key -> {slot, diets, contains}, from the file as written."""
    out = {f"starter:{s['key']}": {k: s[k] for k in ("slot", "diets", "contains")}
           for s in STARTERS["starters"]}
    out.update({f"lib:{slug}": m for slug, m in STARTERS["library_meta"].items()})
    return out


def test_every_starter_and_library_recipe_has_a_slot_diets_and_contains():
    from pantry_planner.mealplan import diets

    meta = _all_meta()
    assert set(STARTERS["library_meta"]) == set(LIBRARY)
    for key, m in meta.items():
        assert m["slot"] in {"breakfast", "lunch", "dinner", "snack"}, key
        assert m["diets"] == [d for d in ("vegan", "vegetarian", "pescatarian")
                              if d in m["diets"]], key
        assert m["contains"] == sorted(set(m["contains"])), key
        assert set(m["contains"]) <= set(diets.CONTAINS), key


def test_every_starter_and_library_recipe_names_a_wikipedia_article_for_its_picture():
    """image_wiki is the English Wikipedia article whose lead image the console shows: a title
    as Wikipedia writes it (spaces, a capital first letter, none of the characters no title
    has, at most 255 bytes). That each is an article with a lead image was checked by hand
    against Wikipedia's API; a test never calls it."""
    titles = {f"starter:{s['key']}": s["image_wiki"] for s in STARTERS["starters"]}
    titles.update({f"lib:{slug}": m["image_wiki"] for slug, m in STARTERS["library_meta"].items()})
    assert len(titles) == len(STARTERS["starters"]) + len(LIBRARY)
    for key, title in titles.items():
        assert isinstance(title, str) and title == " ".join(title.split()) != "", key
        assert title[0].isupper() and "_" not in title, key
        assert not set(title) & set("#<>[]|{}"), key
        assert len(title.encode()) <= 255, key


def test_the_wizard_has_enough_recipes_in_each_section_for_each_diet():
    meta = _all_meta()
    for slot, at_least in (("breakfast", 5), ("lunch", 5), ("dinner", 8)):
        section = [m for m in meta.values() if m["slot"] == slot]
        assert len(section) >= at_least, slot
        for diet in ("vegan", "vegetarian", "pescatarian"):
            assert any(diet in m["diets"] for m in section), (slot, diet)
        # and a pescatarian choice that is fish, not only the vegetarian ones
        assert any("fish" in m["contains"] for m in section), slot


@pytest.fixture(scope="module")
def demo_db(tmp_path_factory):
    from tests.mealplan_fixtures import done_db, use_db

    use_db(tmp_path_factory, "seeds_diets")
    yield
    done_db()


def test_each_recipes_diets_and_contains_are_those_of_the_products_it_resolves_to(demo_db):
    """In demo mode (resolve_one, no LLM), every line a recipe buys resolves to a catalog
    product in its pack's unit, and the file's diets and contains are what those products' tags
    and the lines' names say: diets.classify for the rule, and the tags and the name guard
    themselves for what does not depend on it."""
    from pantry_planner import db
    from pantry_planner.mealplan import diets
    from pantry_planner.mealplan.models import RecipeRef
    from pantry_planner.mealplan.resolve import resolve_one
    from pantry_planner.nlsearch.units import is_non_purchase

    catalog = {p["id"]: p for p in _load("products.json")}
    products = {p.id: p for p in db.load_all_products()}
    for key, m in _all_meta().items():
        kind, _, name = key.partition(":")
        ref = (RecipeRef(key=key, starter=name) if kind == "starter"
               else RecipeRef(key=key, slug=name))
        r = resolve_one(ref)
        assert r.status == "ok" and not r.not_stocked and not r.out_of_range, key
        for ln in r.lines:
            assert (ln.product_id is None) == is_non_purchase(ln.name), (key, ln.name)
            # in its pack's unit, so the catalog prices it and a trip counts its packs
            assert ln.product_id is None or ln.need_uom == catalog[ln.product_id]["unit_uom"], (
                key, ln.name)
        lines = [(ln.name, ln.product_id) for ln in r.lines if ln.product_id is not None]
        assert diets.classify((n, products[pid]) for n, pid in lines) == (
            m["diets"], m["contains"])
        tags = {t for _n, pid in lines for t in catalog[pid]["dietary_tags"].split(",") if t}
        named = {t for n, _pid in lines for t in diets.name_contains(n)}
        contains = set(m["contains"])
        assert (tags - {"meat"}) | named <= contains, key
        assert ("meat" in tags) <= bool({"meat", "fish"} & contains), key
        assert ("vegan" in m["diets"]) == (not contains & {"meat", "fish", "dairy", "egg",
                                                           "honey"}), key
        assert ("vegan" in m["diets"]) <= (not tags & {"meat", "dairy", "egg"}), key
        assert ("vegetarian" in m["diets"]) == ("meat" not in tags | named
                                               and "fish" not in contains), key
        assert ("pescatarian" in m["diets"]) == ("meat" not in contains), key


def test_seafood_is_read_as_fish_and_every_other_meat_as_meat():
    from pantry_planner.mealplan import diets
    from pantry_planner.models import Product

    by_id = {p["id"]: p for p in _load("products.json")}

    def tags(pid: int) -> set[str]:
        return diets.product_contains(Product(**by_id[pid]))

    # The seafood aisle (salmon, shrimp, white fish), and fish sold in the pantry aisles
    # (canned tuna, fish sauce, oyster sauce).
    for pid in (47, 158, 161, 48, 94, 77):
        assert "fish" in tags(pid) and "meat" not in tags(pid), by_id[pid]["name"]
    # Chicken, beef, chicken stock, pork and pepperoni.
    for pid in (10, 22, 57, 156, 167):
        assert "meat" in tags(pid) and "fish" not in tags(pid), by_id[pid]["name"]
    assert diets.fits(["dairy", "fish", "gluten"], "vegetarian", ["gluten", "dairy"]) == (
        False, "contains fish, gluten and dairy")
    assert diets.fits(["meat"], "pescatarian", []) == (False, "contains meat")
    assert diets.fits(["dairy"], "any", ["nuts"]) == (True, None)


def test_a_lines_name_says_what_its_product_tags_leave_out():
    """The PB&J's White Chocolate is bought as a chocolate with no dairy tag: its name makes
    the sandwich dairy, so it is vegetarian and not vegan. A plant word cancels the guard."""
    from pantry_planner.mealplan import diets

    cases = {"White Chocolate": {"dairy"}, "Milk Chocolate": {"dairy"}, "Dark Chocolate": set(),
             "Whole Milk": {"dairy"}, "Coconut Milk": set(), "Oat Milk": set(),
             "Peanut Butter and Jelly Jam": {"peanuts", "nuts"}, "Butter": {"dairy"},
             "Ghee": {"dairy"},
             "Paneer": {"dairy"}, "Plain Yogurt": {"dairy"}, "Cheddar Cheese": {"dairy"},
             "Eggs": {"egg"}, "Eggplant": set(), "Canned Tuna": {"fish"},
             "Anchovies": {"fish"}, "Fish Sauce": {"fish"}, "Raw Shrimp": {"fish", "shellfish"},
             "Chicken Stock": {"meat"}, "Pepperoni": {"meat"}, "Vegan Sausage": set(),
             "Beefsteak Tomatoes": set(), "Liquid Honey": {"honey"},
             # allergens a name says whatever word comes before it
             "Toasted Sesame Oil": {"sesame"}, "Dijon Mustard": {"mustard"},
             "Almond Milk": {"tree_nuts", "nuts"}, "Oyster Sauce": {"fish", "shellfish"}}
    for name, want in cases.items():
        assert diets.name_contains(name) == want, name
    assert diets.classify([("Liquid Honey", None)]) == (
        ["vegetarian", "pescatarian"], ["honey"])
    assert STARTERS["library_meta"]["pbj_sandwich"] == {
        "slot": "lunch", "diets": ["vegetarian", "pescatarian"],
        "contains": ["dairy", "gluten", "nuts", "peanuts"],
        "image_wiki": "Peanut butter and jelly sandwich"}


def test_the_default_allies_are_countries_the_origin_rules_know():
    from pantry_planner.mealplan.models import DEFAULT_ALLIES, DEFAULT_EXCLUDE_ORIGIN
    from pantry_planner.origins import validate_countries

    assert DEFAULT_ALLIES == (
        "United Kingdom", "France", "Germany", "Italy", "Spain", "Netherlands", "Ireland",
        "Denmark", "Norway", "Sweden", "Australia", "New Zealand", "Japan", "South Korea",
        "Mexico")
    assert validate_countries(["Canada", *DEFAULT_ALLIES, *DEFAULT_EXCLUDE_ORIGIN]) == {}

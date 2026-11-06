"""seeds/allergens.json: every product's priority allergens, reviewed, and read by every rule
that reads contains-tags (diets.product_contains)."""
from __future__ import annotations

import json

from pantry_planner.db import SEEDS_DIR

PRODUCTS = json.loads((SEEDS_DIR / "products.json").read_text(encoding="utf-8"))
FILE = json.loads((SEEDS_DIR / "allergens.json").read_text(encoding="utf-8"))


def _product(pid: int):
    from pantry_planner.models import Product

    row = next(p for p in PRODUCTS if p["id"] == pid)
    return Product(**{k: row[k] for k in ("id", "name", "description", "price", "category",
                                          "subcategory", "dietary_tags", "unit_size",
                                          "unit_qty", "unit_uom")})


def _id(name: str) -> int:
    return next(p["id"] for p in PRODUCTS if p["name"] == name)


def test_every_product_was_reviewed_with_known_tags_and_a_basis():
    from pantry_planner.mealplan.diets import ALLERGENS

    assert {int(k) for k in FILE["products"]} == {p["id"] for p in PRODUCTS}
    for pid, row in FILE["products"].items():
        name = next(p["name"] for p in PRODUCTS if p["id"] == int(pid))
        assert row["name"] == name, pid
        assert set(row["contains"]) <= set(ALLERGENS), name
        assert set(row["contains"].values()) <= {"what", "recipe", "typical"}, name


def test_the_file_never_drops_an_allergen_the_catalog_tags():
    for p in PRODUCTS:
        mine = set(FILE["products"][str(p["id"])]["contains"])
        for tag in {t for t in p["dietary_tags"].split(",") if t} - {"meat"}:
            assert (mine & {"peanuts", "tree_nuts"}) if tag == "nuts" else tag in mine, (
                p["name"], tag)


def test_sesame_mustard_and_shellfish_are_tagged_where_the_catalog_has_no_word_for_them():
    from pantry_planner.mealplan.diets import product_contains

    assert "sesame" in product_contains(_product(_id("Toasted Sesame Oil 250ml")))
    assert "sesame" in product_contains(_product(_id("Sesame Seeds 200g")))
    assert "mustard" in product_contains(_product(_id("Dijon Mustard 250ml")))
    assert "mustard" in product_contains(_product(_id("Black Mustard Seeds 100g")))
    assert "shellfish" in product_contains(_product(_id("Raw Shrimp 340g")))
    assert {"peanuts", "nuts"} <= product_contains(_product(_id("Peanut Oil 1L")))
    assert {"tree_nuts", "nuts"} <= product_contains(_product(_id("Almond Milk 946ml")))


def test_a_swap_that_adds_sesame_or_mustard_is_held_back_and_named():
    from pantry_planner.alternatives import rule

    canola, sesame = _product(_id("Canola Oil 1L")), _product(_id("Toasted Sesame Oil 250ml"))
    why, tags, words = rule(sesame, canola, forbid=set())
    assert (why, tags) == ("adds_tags", ["sesame"])
    assert words == "contains sesame, which Canola Oil 1L does not"
    assert rule(canola, sesame, forbid=set()) is None
    held = rule(_product(_id("Dijon Mustard 250ml")), None, forbid={"mustard"})
    assert held is not None and held[0] == "diet"


def test_an_avoid_list_of_nuts_still_leaves_out_peanuts_and_tree_nuts():
    from pantry_planner.alternatives import rule

    for name in ("Peanut Oil 1L", "Almond Milk 946ml"):
        held = rule(_product(_id(name)), None, forbid={"nuts"})
        assert held is not None and held[0] == "diet", name

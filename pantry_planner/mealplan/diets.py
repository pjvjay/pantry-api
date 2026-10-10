"""Which diets a recipe satisfies and what it contains, read from the products it buys.

A product's contains-tags are its own dietary_tags (dairy, egg, gluten, meat, nuts, soy),
except that a seafood product's "meat" is read as "fish": the catalog tags fish as meat, and a
pescatarian eats fish. A product is seafood when its aisle is seafood or its name or
description names a fish or shellfish (FISH_WORDS), since canned tuna and fish sauce sit in
the pantry aisles.

A recipe contains every tag of every product its lines buy, and satisfies:
- vegan: none of meat, fish, dairy or egg;
- vegetarian: no meat or fish;
- pescatarian: no meat.
Nothing else is claimed. Honey and the like carry no tag in the catalog, so a recipe is only as
vegan as its products' tags say.

The demo starters and the library recipes carry these as metadata in
seeds/mealplan_starters.json (each starter's slot, diets and contains, and library_meta for the
library, whose recipes.json pantry-db shares byte for byte); tests/test_mealplan_seeds.py checks
them against the products their lines resolve to in demo mode. A shopper's own recipe is
classified the same way when every line it buys resolves to a product, and is unknown
otherwise.
"""
from __future__ import annotations

from collections.abc import Iterable

from ..models import Product
from ..nlsearch.units import tokens
from .models import RecipeRef

DIETS = ("vegan", "vegetarian", "pescatarian")
# What each diet leaves out, and the order a refusal names it in.
EXCLUDES = {"vegan": ("meat", "fish", "dairy", "egg"), "vegetarian": ("meat", "fish"),
            "pescatarian": ("meat",)}
CONTAINS = ("dairy", "egg", "fish", "gluten", "meat", "nuts", "soy")
FISH_WORDS = frozenset({"fish", "seafood", "salmon", "tuna", "cod", "anchovy", "sardine",
                        "mackerel", "trout", "shrimp", "prawn", "crab", "lobster", "oyster",
                        "mussel", "clam", "scallop", "squid"})


def product_contains(p: Product) -> set[str]:
    """The product's tags, its meat read as fish when it is seafood."""
    tags = {t.strip() for t in (p.dietary_tags or "").split(",") if t.strip()}
    if "meat" in tags and (p.subcategory == "seafood"
                           or FISH_WORDS & set(tokens(f"{p.name} {p.description}"))):
        tags = (tags - {"meat"}) | {"fish"}
    return tags


def classify(products: Iterable[Product]) -> tuple[list[str], list[str]]:
    """(diets, contains) of a recipe that buys these products: diets in DIETS order, contains
    sorted."""
    contains = set()
    for p in products:
        contains |= product_contains(p)
    diets = [d for d in DIETS if not contains & set(EXCLUDES[d])]
    return diets, sorted(contains)


def fits(contains: list[str], diet: str, avoid: list[str]) -> tuple[bool, str | None]:
    """(whether a classified recipe fits the shopper's diet and avoid list, why not): "contains
    meat", "contains fish and dairy". The diet's own tags are named first, in EXCLUDES order,
    then the avoided ones in the shopper's order."""
    offending = [t for t in EXCLUDES.get(diet, ()) if t in contains]
    offending += [t for t in avoid if t in contains and t not in offending]
    if not offending:
        return True, None
    words = (offending[0] if len(offending) == 1
             else ", ".join(offending[:-1]) + " and " + offending[-1])
    return False, f"contains {words}"


def seed_meta(ref: RecipeRef) -> dict | None:
    """{slot, diets, contains} of a demo starter or library recipe, from the starters file;
    None for a recipe the file does not describe (the shopper's own)."""
    from .resolve import starters_file

    data = starters_file()
    if ref.starter is not None:
        s = next((s for s in data["starters"] if s["key"] == ref.starter), None)
        return None if s is None else {k: s[k] for k in ("slot", "diets", "contains")}
    if ref.slug is not None:
        return data["library_meta"].get(ref.slug)
    return None

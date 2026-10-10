"""Which diets a recipe satisfies and what it contains, read from the products it buys.

A product's contains-tags are its own dietary_tags (dairy, egg, gluten, meat, nuts, soy),
except that a seafood product's "meat" is read as "fish": the catalog tags fish as meat, and a
pescatarian eats fish. A product is seafood when its aisle is seafood or its name or
description names a fish or shellfish (FISH_WORDS), since canned tuna and fish sauce sit in
the pantry aisles.

A recipe contains every tag of every product its lines buy, and what its lines' own names say
(name_contains): a "White Chocolate" line bought as a chocolate with no dairy tag is still
dairy, "Eggs" are egg, "Canned Tuna" is fish, "Chicken Stock" is meat, and honey is "honey",
which no vegan eats. A plant word right before such a word cancels it ("peanut butter",
"coconut milk", "vegan sausage"). A recipe satisfies:
- vegan: none of meat, fish, dairy, egg or honey;
- vegetarian: no meat or fish;
- pescatarian: no meat.
Nothing else is claimed: a recipe is only as vegan as its products' tags and its lines' words
say.

The demo starters and the library recipes carry these as metadata in
seeds/mealplan_starters.json (each starter's slot, diets and contains, and library_meta for the
library, whose recipes.json pantry-db shares byte for byte); tests/test_mealplan_seeds.py checks
them against the products their lines resolve to in demo mode and the lines' names. A shopper's
own recipe is
classified the same way when every line it buys resolves to a product, and is
unknown otherwise.
"""
from __future__ import annotations

from collections.abc import Iterable

from ..models import Product
from ..nlsearch.units import tokens
from .models import RecipeRef

DIETS = ("vegan", "vegetarian", "pescatarian")
# What each diet leaves out, and the order a refusal names it in.
EXCLUDES = {"vegan": ("meat", "fish", "dairy", "egg", "honey"),
            "vegetarian": ("meat", "fish"), "pescatarian": ("meat",)}
CONTAINS = ("dairy", "egg", "fish", "gluten", "honey", "meat", "nuts", "soy")
FISH_WORDS = frozenset({"fish", "seafood", "salmon", "tuna", "cod", "anchovy", "anchovie",
                        "sardine", "mackerel", "trout", "shrimp", "prawn", "crab", "lobster",
                        "oyster", "mussel", "clam", "scallop", "squid"})
# The words of a line's name that say what it contains (as units.tokens reads them: lower
# case, a plural's s dropped). Chocolate is dairy only after "white" or "milk".
NAME_WORDS = {
    "dairy": frozenset({"milk", "cheese", "butter", "cream", "yogurt", "yoghurt", "ghee",
                        "paneer", "mozzarella", "cheddar", "parmesan", "ricotta",
                        "buttermilk"}),
    "egg": frozenset({"egg"}),
    "fish": FISH_WORDS,
    "meat": frozenset({"chicken", "beef", "pork", "bacon", "ham", "pepperoni", "sausage",
                       "lamb", "turkey"}),
    "honey": frozenset({"honey"}),
}
DAIRY_CHOCOLATE = frozenset({"white", "milk"})
# A word right before a guarded one that makes it a plant food.
PLANT_WORDS = frozenset({"peanut", "almond", "cashew", "coconut", "oat", "soy", "rice", "cocoa",
                         "shea", "vegan", "plant", "veggie", "apple"})


def product_contains(p: Product) -> set[str]:
    """The product's tags, its meat read as fish when it is seafood."""
    tags = {t.strip() for t in (p.dietary_tags or "").split(",") if t.strip()}
    if "meat" in tags and (p.subcategory == "seafood"
                           or FISH_WORDS & set(tokens(f"{p.name} {p.description}"))):
        tags = (tags - {"meat"}) | {"fish"}
    return tags


def name_contains(name: str) -> set[str]:
    """What a line's name says it contains, whatever its product's tags say."""
    words = tokens(name)
    out = set()
    for i, w in enumerate(words):
        before = words[i - 1] if i else ""
        if before in PLANT_WORDS:
            continue
        out |= {tag for tag, guarded in NAME_WORDS.items() if w in guarded}
        if w == "chocolate" and before in DAIRY_CHOCOLATE:
            out.add("dairy")
    return out


def classify(lines: Iterable[tuple[str, Product | None]]) -> tuple[list[str], list[str]]:
    """(diets, contains) of a recipe whose lines (name, the product bought for it) these are:
    the products' tags and the names' words, diets in DIETS order, contains sorted."""
    contains = set()
    for name, p in lines:
        contains |= name_contains(name)
        if p is not None:
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

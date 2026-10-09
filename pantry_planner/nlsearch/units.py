"""PRE-PROCESSING: unit normalization + ingredient tokenization.

Canonical units are g | ml | each. Unknown units ("pinch", "clove")
return None and the size predicate is simply skipped for that ingredient.
"""
from __future__ import annotations

import re

# multiplier to the canonical unit
# Plurals are listed: the parser keeps the unit as written ("2 tablespoons"),
# and an unknown unit silently drops the size need.
_WEIGHT = {"g": 1, "gram": 1, "grams": 1, "kg": 1000, "kgs": 1000, "kilogram": 1000,
           "kilograms": 1000, "lb": 454, "lbs": 454, "pound": 454, "pounds": 454,
           "oz": 28, "ounce": 28, "ounces": 28}
_VOLUME = {"ml": 1, "milliliter": 1, "millilitre": 1, "milliliters": 1,
           "millilitres": 1, "l": 1000, "liter": 1000, "litre": 1000,
           "liters": 1000, "litres": 1000, "cup": 250, "cups": 250,
           "tbsp": 15, "tbsps": 15, "tablespoon": 15, "tablespoons": 15,
           "tsp": 5, "tsps": 5, "teaspoon": 5, "teaspoons": 5,
           "can": 400, "cans": 400}   # "1 can" ≈ standard 400ml
_COUNT = {"each", "whole", "piece", "pieces", "dozen"}

_STOPWORDS = {"fresh", "of", "the", "a", "an", "large", "small", "medium",
              "to", "taste", "optional", "some"}


def normalize_quantity(quantity: float | None, unit: str | None) -> tuple[float, str] | None:
    """('225', 'g') -> (225, 'g'); ('2', 'cups') -> (500, 'ml'); unknown -> None."""
    if quantity is None or unit is None:
        return None
    u = unit.strip().lower()
    if u in _WEIGHT:
        return quantity * _WEIGHT[u], "g"
    if u in _VOLUME:
        return quantity * _VOLUME[u], "ml"
    if u in _COUNT:
        return quantity * (12 if u == "dozen" else 1), "each"
    return None


# Plurals the two suffix rules below get wrong. A recipe writes "1 bay leaf"
# and the shelf says "Bay Leaves 10g"; plain s-stripping made that "leave",
# so the two never met. A general -ves -> -f rule is NOT the fix: it would
# turn olives, cloves and chives into olif, clof and chif. Lowercase words.
IRREGULAR_PLURALS = {"leaves": "leaf", "loaves": "loaf", "halves": "half"}


def stem(token: str) -> str:
    """tomatoes -> tomato, onions -> onion, leaves -> leaf. Deliberately
    naive: two suffix rules plus a short irregular list."""
    if token in IRREGULAR_PLURALS:
        return IRREGULAR_PLURALS[token]
    if token.endswith("oes") and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and not token.endswith("ss") and len(token) > 3:
        return token[:-1]
    return token


def tokens(name: str) -> list[str]:
    """Match tokens for an ingredient name: lowercase, stemmed, stopwords out."""
    raw = re.findall(r"[a-zA-Z]+", name.lower())
    return [stem(t) for t in raw if t not in _STOPWORDS and len(t) > 1]


# ─── what a product is indexed under ─────────────────────────
# A product's match terms (the product_terms index) are the tokens of its
# name plus its description, because the description is where the catalog
# writes the words recipes use and the name does not: "scallions" on Green
# Onions, "garbanzo beans" on Chickpeas, "coriander leaves" on Cilantro.
# A word the description NEGATES is the opposite of a match term: "Canned
# crushed tomatoes, no salt added" must never answer a recipe's "salt". So
# "no X", "no X added" and "without X" are cut before tokenizing. "X-free"
# is left alone: "dairy-free" is what a "dairy-free cheese" shopper asks for.
# KEEP-IN-SYNC: pantry-db's scripts/gen-seed-sql.py applies the same rule.
_NEGATED = re.compile(r"\b(?:no|without)[\s-]+[a-z]+(?:[\s-]+added)?\b", re.IGNORECASE)


def index_text(name: str, description: str | None = "") -> str:
    """The text a product is matched under: its name plus its description
    with negated phrases removed ("no salt added")."""
    return f"{name} {_NEGATED.sub(' ', description or '')}"


# ─── form equivalence ────────────────────────────────────────
# Shelf labels and recipes name the same purchase form two ways: a recipe's
# "cumin powder" is the shelf's "Cumin Ground 100g". The planner tries this
# swap only after the exact match found nothing, and labels the line
# match="form", so "chili powder" still matches Chili Powder 100g exactly.
FORM_EQUIVALENTS = {"powder": "ground", "powdered": "ground", "ground": "powder"}


def equivalent_tokens(toks: list[str]) -> list[str]:
    """`toks` with each form word swapped for its equivalent: [cumin, powder]
    -> [cumin, ground]. [] when no token has an equivalent."""
    out: list[str] = []
    changed = False
    for t in toks:
        e = FORM_EQUIVALENTS.get(t, t)
        changed = changed or e != t
        if e not in out:
            out.append(e)
    return out if changed else []


# ─── non-purchases ───────────────────────────────────────────
# Lines a recipe lists that nobody buys. The planner drops them before
# retrieval: they are never priced and never offered to the selector (which
# once "bought" a recipe's water as a second can of chickpeas), and the plan
# names them in `skipped`. Whole ingredient names only, compared as tokens()
# output: "coconut water", "sparkling water" and "rose water" are products
# and are NOT matched by "water".
NON_PURCHASES = frozenset({
    ("water",), ("hot", "water"), ("boiling", "water"), ("cold", "water"),
    ("warm", "water"), ("lukewarm", "water"), ("tap", "water"),
    ("ice", "water"), ("iced", "water"), ("ice",), ("ice", "cube"),
})


def is_non_purchase(name: str) -> bool:
    """True for "water", "boiling water", "ice cubes", ...; see NON_PURCHASES."""
    return tuple(tokens(name)) in NON_PURCHASES


# ─── generic matching: the third retrieval level ─────────────
# A recipe names "light brown sugar"; the shelf says "Brown Sugar 1kg". When
# neither the strict match (every token, purchase form included) nor the
# form-relaxed one finds a product, the planner retries without these
# descriptor words and labels the line match="generic".
#
# A small FIXED vocabulary on purpose: every word here is one a recipe adds
# and a shelf label routinely omits, and none of them names the product
# itself. The parser keeps such words in the ingredient name when the recipe
# writes them ("boneless skinless chicken thigh"), so the exact level finds
# the product that has them and this level is the fallback when none does.
# It applies to tokens() OUTPUT, so entries are lowercase, stemmed tokens.
# "fresh", "large", "small" and "medium" are already stopwords there; they
# are listed so the vocabulary reads complete. Two-word descriptors are
# matched as phrases ("bone-in" is bone + in, "low-sodium" is low + sodium),
# so "sodium" or "in" alone is never dropped.
DESCRIPTORS = frozenset({
    "light", "dark", "toasted", "roasted", "ground", "whole", "dried", "fresh",
    "frozen", "boneless", "skinless", "large", "small", "medium", "extra",
    "chopped", "sliced", "minced", "diced", "raw", "organic", "smoked",
    "unsalted",
})
DESCRIPTOR_PHRASES = (("low", "sodium"), ("reduced", "sodium"), ("bone", "in"),
                      ("skin", "on"))


def generic_tokens(name: str) -> list[str]:
    """tokens(name) without descriptor words: "light soy sauce" -> soy, sauce.

    Returns [] when the generic level does not apply: nothing was dropped
    (it would only repeat the relaxed match) or nothing would remain (a bare
    descriptor must never match the whole catalog)."""
    toks = tokens(name)
    out: list[str] = []
    dropped = False
    i = 0
    while i < len(toks):
        if tuple(toks[i:i + 2]) in DESCRIPTOR_PHRASES:
            dropped, i = True, i + 2
            continue
        if toks[i] in DESCRIPTORS:
            dropped = True
        elif toks[i] not in out:            # token-AND counts distinct terms
            out.append(toks[i])
        i += 1
    return out if dropped else []


# ─── head nouns ──────────────────────────────────────────────
# The purchase forms the parser may split out of a name (query_parser's
# ALLOWED_FORMS; demomode reads them off the front of a line).
PURCHASE_FORMS = frozenset({"canned", "frozen", "dried", "ground", "smoked", "pickled",
                            "powdered"})

# Words in a product name that are not what the product IS: pack and size
# words, purchase forms, descriptors, the brand words storeseed reads from
# names, and the words this catalog's names put AFTER the noun ("Ground Beef
# Lean", "Butter Salted", "Cheddar Cheese Block", "Bell Pepper Trio"). What
# is left ends in the product's head noun.
_NOT_HEAD = (DESCRIPTORS | PURCHASE_FORMS
             | {"pack", "kg", "ml", "lb", "oz", "dozen", "count", "bag", "box",
                "jar", "bottle", "cadbury", "nestle",
                "lean", "salted", "block", "shredded", "wedge", "flaked", "family",
                "crown", "trio", "rigate"})


def head_noun(name: str) -> str | None:
    """The noun a name is about: "Flour Tortillas 10-Pack" -> tortilla,
    "All-Purpose Flour 2.5kg" -> flour, "Coriander Ground 100g" -> coriander,
    "Chicken Thighs Bone-In" -> thigh. None when nothing is left."""
    toks = tokens(name)
    kept: list[str] = []
    i = 0
    while i < len(toks):
        if tuple(toks[i:i + 2]) in DESCRIPTOR_PHRASES:
            i += 2
            continue
        if toks[i] not in _NOT_HEAD:
            kept.append(toks[i])
        i += 1
    return kept[-1] if kept else None


# ─── semantic closeness: the single ranking key ──────────────
# How well a product answers an ingredient, before price or origin are
# looked at. The demo-mode selector picks by it, and the alternatives
# ranking orders a line's candidates by it, so the two can never disagree
# about which product is the closer match.

def semantic_key(ingredient: str, product) -> tuple[int, int, int]:
    """(-shared tokens, fresh miss, head-noun miss) for `product` (anything
    with .name and .description) against an ingredient name; lower is
    closer. Token overlap is over the product's index_text. "fresh" is a
    tokenizer stopword (half the catalog says it), yet "fresh coriander" is
    cilantro, not ground coriander, so it is read raw. The head-noun key
    breaks overlap ties the way a shopper would: for "flour", All-Purpose
    Flour (about flour) beats Flour Tortillas (about tortillas)."""
    toks = tokens(ingredient)
    want = toks[-1] if toks else None
    fresh = "fresh" in ingredient.lower().split()
    return (-len(set(toks) & set(tokens(index_text(product.name, product.description)))),
            0 if not fresh or "fresh" in f"{product.name} {product.description}".lower() else 1,
            0 if want is not None and head_noun(product.name) == want else 1)

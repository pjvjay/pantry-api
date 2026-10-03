"""PRE-PROCESSING: unit normalization + ingredient tokenization.

Canonical units are g | ml | each. Unknown units ("pinch", "clove")
return None and the size predicate is simply skipped for that ingredient.
"""
from __future__ import annotations

import re

# multiplier to the canonical unit
_WEIGHT = {"g": 1, "gram": 1, "grams": 1, "kg": 1000, "kilogram": 1000,
           "lb": 454, "lbs": 454, "pound": 454, "pounds": 454,
           "oz": 28, "ounce": 28, "ounces": 28}
_VOLUME = {"ml": 1, "milliliter": 1, "millilitre": 1, "l": 1000, "liter": 1000,
           "litre": 1000, "cup": 250, "cups": 250, "tbsp": 15, "tablespoon": 15,
           "tsp": 5, "teaspoon": 5, "can": 400}   # "1 can" ≈ standard 400ml
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


def stem(token: str) -> str:
    """tomatoes -> tomato, onions -> onion. Deliberately naive."""
    if token.endswith("oes") and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and not token.endswith("ss") and len(token) > 3:
        return token[:-1]
    return token


def tokens(name: str) -> list[str]:
    """Match tokens for an ingredient name: lowercase, stemmed, stopwords out."""
    raw = re.findall(r"[a-zA-Z]+", name.lower())
    return [stem(t) for t in raw if t not in _STOPWORDS and len(t) > 1]


# ─── generic matching: the third retrieval level ─────────────
# A recipe names "light brown sugar"; the shelf says "Brown Sugar 1kg". When
# neither the strict match (every token, purchase form included) nor the
# form-relaxed one finds a product, the planner retries without these
# descriptor words and labels the line match="generic".
#
# A small FIXED vocabulary on purpose: every word here is one a recipe adds
# and a shelf label routinely omits, and none of them names the product
# itself. It applies to tokens() OUTPUT, so entries are lowercase, stemmed
# tokens. tokens() itself is untouched — pantry-db's seed generator keeps a
# copy of it and the parity test guards that copy. "fresh", "large",
# "small" and "medium" are already stopwords there; they are listed so the
# vocabulary reads complete. Two-word descriptors are matched as phrases,
# so "sodium" alone is never dropped.
DESCRIPTORS = frozenset({
    "light", "dark", "toasted", "roasted", "ground", "whole", "dried", "fresh",
    "frozen", "boneless", "skinless", "large", "small", "medium", "extra",
    "chopped", "sliced", "minced", "diced", "raw", "organic",
})
DESCRIPTOR_PHRASES = (("low", "sodium"), ("reduced", "sodium"))


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

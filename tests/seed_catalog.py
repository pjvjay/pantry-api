"""Catalog facts read straight from seeds/products.json.

Tests that need a catalog-wide number (how many products, how many in an
aisle) take it from the file the seed loader reads, never from the database
the code under test wrote and never from a literal. The catalog is a copy of
pantry-db's seed and grows with it (pjvjay/pantry-api#22); a count derived
here follows it, while a loader that drops or duplicates a row still fails.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

SEED_FILE = Path(__file__).resolve().parents[1] / "seeds" / "products.json"
SEED_PRODUCTS: list[dict] = json.loads(SEED_FILE.read_text(encoding="utf-8"))
CATALOG = len(SEED_PRODUCTS)


def count(category: str, subcategory: str | None = None) -> int:
    """Products in a category (and subcategory, when given)."""
    return sum(1 for p in SEED_PRODUCTS if p["category"] == category
               and (subcategory is None or p.get("subcategory") == subcategory))


def categories() -> dict[str, dict[str, int]]:
    """{category: {subcategory: count}} as the categories resource reports it."""
    out: dict[str, dict[str, int]] = {}
    for (cat, sub), n in Counter(
            (p["category"] or "", p.get("subcategory") or "") for p in SEED_PRODUCTS).items():
        out.setdefault(cat, {})[sub] = n
    return out


def ids_mentioning(word: str) -> set[int]:
    """Ids whose name or description contains `word` as a whole word — a
    plain-text reading of the seed, independent of the tokenizer."""
    rx = re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE)
    return {p["id"] for p in SEED_PRODUCTS
            if rx.search(f"{p['name']} {p.get('description', '')}")}

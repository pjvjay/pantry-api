"""Synthetic store/brand/review/terms generation for the local dev DB.

KEEP-IN-SYNC: this is the same deterministic algorithm as pantry-db's
scripts/gen-seed-sql.py (which renders it into seed.sql for the Postgres
migration Job). The store list, brand derivation, ±15% price variance and
review generation are duplicated verbatim; the tokenizer is NOT duplicated
here — we import the real one (nlsearch.units.tokens), and the test-suite
parity test guards pantry-db's copy of it. Change either side → change both.
"""
from __future__ import annotations

import hashlib
import random

from .nlsearch.units import tokens

# Reference shopping location (49.28, -123.12) — three stores within
# 10 km, one at ~14 km so the distance constraint is demonstrable.
STORES = [
    (1, "Pantry Mart Downtown", 49.2820, -123.1180, "833 Granville St, Vancouver"),
    (2, "GreenLeaf Grocers Kitsilano", 49.2680, -123.1550, "2301 W 4th Ave, Vancouver"),
    (3, "ValueFoods East Van", 49.2620, -123.0700, "1605 Commercial Dr, Vancouver"),
    (4, "MegaSave Richmond", 49.1550, -123.1350, "4800 No. 3 Rd, Richmond"),
]

BRANDS_IN_NAME = ["Cadbury", "Nestles"]
HOUSE_BRANDS = ["PantryCo", "Fraser Farms", "Maple Ridge",
                "Coastline Foods", "Golden Gate"]

REVIEW_COMMENTS = [
    "Great value.", "Would buy again.", "Just okay.", "Family favourite.",
    "Quality varies by batch.", "Fresh and tasty.",
    "A bit pricey for what it is.", "Solid staple.",
]


def brand_for(product: dict) -> str:
    if product.get("brand"):
        return product["brand"]
    for b in BRANDS_IN_NAME:
        if b.lower() in product["name"].lower():
            return b
    return HOUSE_BRANDS[product["id"] % len(HOUSE_BRANDS)]


def store_price(store_id: int, product_id: int, base: float) -> float:
    rng = random.Random(f"sp:{store_id}:{product_id}")
    return round(base * (1 + rng.uniform(-0.15, 0.15)), 2)


def _brand_quality(brand: str) -> float:
    h = int(hashlib.sha256(brand.encode()).hexdigest(), 16) % 1000
    return 3.0 + 1.8 * (h / 999)


def reviews_for(product: dict, brand: str) -> list[tuple[int, str, str]]:
    rng = random.Random(f"rev:{product['id']}")
    mean = _brand_quality(brand)
    out = []
    for _ in range(rng.randint(2, 8)):
        rating = max(1, min(5, round(rng.gauss(mean, 0.7))))
        comment = rng.choice(REVIEW_COMMENTS)
        created = f"2026-{rng.randint(1, 6):02d}-{rng.randint(1, 28):02d}"
        out.append((rating, comment, created))
    return out


def product_terms(product: dict) -> list[str]:
    text = f"{product['name']} {product.get('description', '')}"
    return sorted(set(tokens(text)))

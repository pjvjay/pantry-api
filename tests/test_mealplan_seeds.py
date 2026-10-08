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

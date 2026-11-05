"""The committed matching seeds (seeds/product_profiles.json, product_embeddings.json,
ingredient_types.json): every product has a profile and a vector made from its current row,
every profile's embedded sentence is the template of its fields (hand fixes included), the
catalog's contains-tags are never part of a profile, and every seed ingredient name is mapped.
Read from the files themselves, with no database and no model."""
from __future__ import annotations

import json
import math

from tests.seed_catalog import SEED_PRODUCTS


def test_every_product_has_a_profile_and_a_vector_made_from_its_row():
    from pantry_planner.matching import seeds

    s = seeds.load()
    assert set(s.profiles) == {p["id"] for p in SEED_PRODUCTS} == set(s.vectors)
    assert (s.model, s.dims) == ("embeddinggemma", 256)
    for p in SEED_PRODUCTS:
        prof = s.profiles[p["id"]]
        assert prof.hash == seeds.product_hash(p), p["name"]
        assert prof.product_type == prof.product_type.lower().strip()
        assert prof.taxonomy[-1] == prof.product_type
        assert set(prof.diet) <= {"vegan", "vegetarian", "pescatarian"}
        v = s.vectors[p["id"]]
        assert len(v) == 256 and abs(math.sqrt(sum(x * x for x in v)) - 1) < 1e-3


def test_a_profile_text_is_the_template_of_its_fields_and_never_carries_contains_tags():
    """Fixes made by hand after review change the fields; the embedded sentence follows them,
    and the catalog's contains-tags are never part of a profile."""
    from pantry_planner.matching import seeds

    data = json.loads((seeds.SEEDS_DIR / "product_profiles.json").read_text())
    for p in data["products"]:
        assert p["profile_text"] == seeds.profile_text(p), p["name"]
        assert "dietary_tags" not in p and "contains" not in p
        for fix in p["fixes"]:
            assert p[fix["field"]] == fix["to"] and fix["why"], p["name"]


def test_every_seed_ingredient_name_is_mapped_with_a_query_vector():
    from pantry_planner.matching import seeds
    from pantry_planner.nlsearch.units import is_non_purchase

    names = [i["name"] if isinstance(i, dict) else i
             for r in json.loads((seeds.SEEDS_DIR / "recipes.json").read_text())
             for i in r["ingredients"]]
    names += [ln["name"] for s in json.loads((seeds.SEEDS_DIR / "mealplan_starters.json")
                                             .read_text())["starters"] for ln in s["lines"]]
    known = seeds.load().ingredients
    types = seeds.load().types()
    assert seeds.seed_ingredient_names() == list(dict.fromkeys(
        n for n in names if not is_non_purchase(n)))
    for name in names:
        if is_non_purchase(name):
            continue
        hit = known[seeds.ingredient_key(name)]
        assert len(hit.vector) == 256
        assert set(hit.exact_types) & types or set(hit.substitute_types) & types, name

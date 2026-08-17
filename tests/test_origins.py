"""Country-of-origin resolver tests — no LLM calls, no network.

The heuristic tier and the DB cache run for real against a seeded
SQLite DB; the LLM tier is exercised through a monkeypatched
call_origin_resolver so tier ordering and caching are verified without
an API key.
"""
from __future__ import annotations

import os

import pytest

_TMP_DB = None


@pytest.fixture(scope="module", autouse=True)
def seeded_db(tmp_path_factory):
    global _TMP_DB
    _TMP_DB = tmp_path_factory.mktemp("db") / "test.db"
    os.environ["DB_URL"] = f"sqlite:///{_TMP_DB}"
    from pantry_planner import config, db

    config.settings.cache_clear()
    db.seed_from_json()
    yield
    config.settings.cache_clear()


def _product(id=1, name="Mystery Item", brand="", category="pantry",
             subcategory="", description=""):
    from pantry_planner.models import Product

    return Product(id=id, name=name, description=description, price=1.0,
                   category=category, subcategory=subcategory, brand=brand)


# ─── Tier 1: heuristics ──────────────────────────────────────

def test_keyword_rules():
    from pantry_planner.origins import heuristic_origin

    cases = {
        "Basmati Rice 2kg": "India",
        "Dark Chocolate Cadbury": "United Kingdom",
        "Canola Oil 1L": "Canada",
        "Canadian Peanut Butter Cream": "Canada",
        "Garam Masala 100g": "India",
        "Passata Strained Tomatoes 680ml": "Italy",
    }
    for name, country in cases.items():
        o = heuristic_origin(_product(name=name))
        assert o is not None, name
        assert o.country == country, name
        assert o.source == "heuristic"
        assert 0.6 <= o.confidence <= 1.0


def test_keyword_rule_beats_subcategory_default():
    # Parmesan is in the "cheese" subcategory (default: Canada) but the
    # keyword rule must win with Italy.
    from pantry_planner.origins import heuristic_origin

    o = heuristic_origin(_product(name="Parmesan Wedge 200g",
                                  category="dairy", subcategory="cheese"))
    assert o is not None and o.country == "Italy"


def test_subcategory_defaults():
    from pantry_planner.origins import heuristic_origin

    o = heuristic_origin(_product(name="Roma Tomato", category="produce",
                                  subcategory="vegetables"))
    assert o is not None and o.country == "Canada"

    o = heuristic_origin(_product(name="Whole Milk 1L", category="dairy",
                                  subcategory="milk"))
    assert o is not None and o.country == "Canada"


def test_plant_based_dairy_not_claimed_by_dairy_default():
    # Oat/almond/vegan items sit in dairy subcategories but are processed
    # goods — the supply-management rationale doesn't apply.
    from pantry_planner.origins import heuristic_origin

    assert heuristic_origin(_product(name="Oat Milk 1L", subcategory="milk")) is None
    assert heuristic_origin(_product(name="Vegan Cheese Shreds 200g",
                                     subcategory="cheese")) is None


def test_no_rule_returns_none():
    from pantry_planner.origins import heuristic_origin

    assert heuristic_origin(_product(name="Canned Diced Tomatoes",
                                     subcategory="canned")) is None


# ─── resolve_origins tiering ─────────────────────────────────

def test_resolve_without_llm_marks_unruled_unknown():
    from pantry_planner import db
    from pantry_planner.origins import resolve_origins

    products = db.load_all_products()
    origins = resolve_origins(products, allow_llm=False)

    assert len(origins) == len(products)
    assert [o.product_id for o in origins] == [p.id for p in products]
    by_source = {o.source for o in origins}
    assert "llm" not in by_source and "cache" not in by_source
    unknown = [o for o in origins if o.country == "Unknown"]
    resolved = [o for o in origins if o.country != "Unknown"]
    assert resolved, "heuristics should cover most of the catalog"
    assert all(o.confidence == 0.0 for o in unknown)


def test_llm_tier_called_once_then_cached(monkeypatch):
    """Unresolved products go to the (mocked) LLM exactly once; a second
    resolve reads the cache instead of calling again."""
    from pantry_planner import db, origins
    from pantry_planner.models import ProductOrigin

    calls = []

    def fake_resolver(products, *, model):
        calls.append(len(products))
        return [
            ProductOrigin(product_id=p.id, product_name=p.name,
                          country="Testlandia", confidence=0.5,
                          source="llm", reasoning="mocked")
            for p in products
        ], 0.001

    monkeypatch.setattr(origins, "call_origin_resolver", fake_resolver)
    # allow_llm requires a non-empty key; settings is cached so patch the
    # cached instance's field via env + cache_clear.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from pantry_planner import config
    config.settings.cache_clear()

    products = db.load_all_products()
    first = origins.resolve_origins(products, allow_llm=True)
    assert calls, "LLM tier should have been invoked"
    assert all(o.country != "Unknown" for o in first)
    llm_resolved = [o for o in first if o.source == "llm"]
    assert len(llm_resolved) == calls[0]

    second = origins.resolve_origins(products, allow_llm=True)
    assert len(calls) == 1, "second resolve must hit the cache, not the LLM"
    cached = [o for o in second if o.source == "cache"]
    assert len(cached) == len(llm_resolved)
    assert all(o.country == "Testlandia" for o in cached)

    config.settings.cache_clear()


def test_search_by_origin_matches_case_insensitive_substring():
    from pantry_planner.origins import search_by_origin

    italy = search_by_origin("italy")
    assert italy, "seeded catalog has Italian products"
    assert all("Italy" == o.country for _, o in italy)

    uk = search_by_origin("kingdom")
    assert {p.name for p, _ in uk} == {"Dark Chocolate Cadbury",
                                       "Milk Chocolate Cadbury"}

    assert search_by_origin("Atlantis") == []


def test_search_by_origin_default_is_free(monkeypatch):
    from pantry_planner import origins

    def boom(*a, **k):
        raise AssertionError("search_by_origin must not call the LLM by default")

    monkeypatch.setattr(origins, "call_origin_resolver", boom)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from pantry_planner import config
    config.settings.cache_clear()
    origins.search_by_origin("canada")
    config.settings.cache_clear()


# ─── LLM tier response hygiene ───────────────────────────────

def test_llm_resolver_ignores_hallucinated_ids(monkeypatch):
    """call_origin_resolver drops product_ids not present in the input."""
    from types import SimpleNamespace

    from pantry_planner import origins

    tool_block = SimpleNamespace(
        type="tool_use",
        input={"origins": [
            {"product_id": 1, "country": "Canada", "confidence": 0.8,
             "reasoning": "ok"},
            {"product_id": 999, "country": "Atlantis", "confidence": 0.9,
             "reasoning": "hallucinated"},
        ]},
    )
    resp = SimpleNamespace(
        content=[tool_block],
        usage=SimpleNamespace(input_tokens=100, output_tokens=50))

    class FakeClient:
        def __init__(self, **kw):
            self.messages = SimpleNamespace(create=lambda **kw: resp)

    monkeypatch.setattr(origins, "Anthropic", FakeClient)
    got, cost = origins.call_origin_resolver(
        [_product(id=1, name="Thing")], model="claude-haiku-4-5-20251001")
    assert [o.product_id for o in got] == [1]
    assert got[0].country == "Canada"
    assert cost > 0

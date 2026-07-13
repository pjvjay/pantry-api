"""
Tests for the routing signal computations.

These tests don't call the Anthropic API — they exercise the Phase A
deterministic signals (which have no LLM) and mock the classifier call
for Phase B / Phase C coverage.
"""
from __future__ import annotations

from pantry_planner.models import (
    PhaseAMetrics,
    PhaseBMetrics,
    Product,
    Recipe,
    RecipeIngredient,
)
from pantry_planner.router.decision import decide
from pantry_planner.router.deterministic import char_trigrams, compute_phase_a, jaccard


def test_trigrams_basic():
    assert char_trigrams("cat") == {"cat"}
    assert "che" in char_trigrams("cheddar")
    assert len(char_trigrams("")) == 0


def test_jaccard_edges():
    assert jaccard(set(), set()) == 0.0
    assert jaccard({"a", "b"}, {"a", "b"}) == 1.0
    assert 0.0 < jaccard({"a", "b"}, {"b", "c"}) < 1.0


def test_phase_a_high_similarity():
    """Recipe with obvious matches should score high mean similarity."""
    recipe = Recipe(
        slug="test",
        name="Test",
        ingredients=[RecipeIngredient(line_no=1, name="Wheat Bread")],
    )
    products = [
        Product(id=1, name="Wheat Bread Loaf", description="whole grain wheat", price=3.0),
        Product(id=2, name="White Bread Loaf", description="basic white loaf", price=2.0),
    ]
    sig = compute_phase_a(recipe, products)
    assert sig.ingredient_count == 1
    assert sig.mean_max_similarity > 0.3  # obvious match should score well
    assert sig.count_below_0_3 == 0


def test_phase_a_low_similarity():
    """Recipe with no obvious matches should flag ingredients."""
    recipe = Recipe(
        slug="test",
        name="Test",
        ingredients=[RecipeIngredient(line_no=1, name="Quinoa")],
    )
    products = [
        Product(id=1, name="Wheat Bread", description="loaf", price=3.0),
    ]
    sig = compute_phase_a(recipe, products)
    assert sig.count_below_0_3 == 1


def test_phase_c_routes_easy_task_to_haiku():
    """Low complexity → Haiku."""
    phase_a = PhaseAMetrics(
        ingredient_count=3,
        mean_max_similarity=0.7,
        min_max_similarity=0.6,
        count_below_0_3=0,
        category_density=2.0,
    )
    phase_b = PhaseBMetrics(
        match_confidence_1_to_10=9,
        cost_complexity_1_to_10=2,
        ambiguous_ingredients=[],
        confidence_in_own_estimate_1_to_10=8,
        reasoning="all clear",
    )
    model, complexity, reason = decide(phase_a, phase_b)
    assert "haiku" in model.lower()
    assert complexity < 0.3


def test_phase_c_routes_hard_task_to_sonnet():
    """High complexity → escalation model."""
    phase_a = PhaseAMetrics(
        ingredient_count=15,
        mean_max_similarity=0.15,
        min_max_similarity=0.05,
        count_below_0_3=10,
        category_density=8.0,
    )
    phase_b = PhaseBMetrics(
        match_confidence_1_to_10=2,
        cost_complexity_1_to_10=9,
        ambiguous_ingredients=[
            {"name": f"ing{i}", "why": "unclear"} for i in range(10)
        ],
        confidence_in_own_estimate_1_to_10=6,
        reasoning="many unknowns",
    )
    model, complexity, reason = decide(phase_a, phase_b)
    assert complexity > 0.6
    assert "sonnet" in model.lower() or "haiku" not in model.lower()

"""
Phase A of the three-phase router: cheap deterministic signals.

Everything here runs in <10ms with zero API cost. The signals are
proxies for problem difficulty — not the answer, just enough to inform
routing.

Character-trigram Jaccard is a rough surface-similarity measure. Think of
it as ROUGE-N at the character level. It won't tell you WHICH product
matches best (LLMs do that) — but it tells you when the surface features
alone are enough (a very-confident match exists) vs. when the LLM has
work to do.
"""
from __future__ import annotations

from statistics import mean

from ..models import PhaseAMetrics, Product, Recipe


# ─── Similarity primitives ───────────────────────────────────

def char_trigrams(s: str, n: int = 3) -> set[str]:
    """Character n-grams of s. Lowercased, stripped."""
    s = s.lower().strip()
    if len(s) < n:
        return {s} if s else set()
    return {s[i:i + n] for i in range(len(s) - n + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity: |A ∩ B| / |A ∪ B|."""
    if not (a or b):
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


# ─── The Phase A computation ─────────────────────────────────

def compute_phase_a(recipe: Recipe, products: list[Product]) -> PhaseAMetrics:
    """
    For each ingredient:
        max_similarity = max over products of Jaccard(ingredient_grams, product_grams)

    Aggregate into a per-run signal.
    """
    max_sims: list[float] = []
    for ing in recipe.ingredients:
        ing_grams = char_trigrams(ing.name)
        best = 0.0
        for p in products:
            product_grams = char_trigrams(f"{p.name} {p.description}")
            score = jaccard(ing_grams, product_grams)
            if score > best:
                best = score
        max_sims.append(best)

    # Category density: mean number of products per category the recipe
    # touches. Higher = more decisions per category = harder.
    category_counts: dict[str, int] = {}
    for p in products:
        if p.category:
            category_counts[p.category] = category_counts.get(p.category, 0) + 1

    ing_categories = {ing.category for ing in recipe.ingredients if ing.category}
    density_values = [category_counts.get(c, 0) for c in ing_categories]
    category_density = mean(density_values) if density_values else 0.0

    return PhaseAMetrics(
        ingredient_count=len(recipe.ingredients),
        mean_max_similarity=mean(max_sims) if max_sims else 0.0,
        min_max_similarity=min(max_sims) if max_sims else 0.0,
        count_below_0_3=sum(1 for s in max_sims if s < 0.3),
        category_density=category_density,
    )

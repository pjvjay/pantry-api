"""
Phase C of the three-phase router: weighted-sum thresholding.

Combines Phase A (deterministic signals) and Phase B (classifier output)
into a single complexity score in [0, 1] and threshold-selects a model.

Weights are hyperparameters. Tune with an eval sweep — the current
values were chosen by inspection and haven't been swept yet.
"""
from __future__ import annotations

from ..config import HAIKU, SONNET
from ..models import PhaseAMetrics, PhaseBMetrics


# ─── Weights ──────────────────────────────────────────────────
# Sum = 1.0. Rebalance based on eval results.
W_INGREDIENT_COUNT = 0.10   # more ingredients → more decisions
W_SIMILARITY_GAP   = 0.25   # low surface similarity → LLM has work to do
W_AMBIGUOUS_FRAC   = 0.25   # LLM-flagged ambiguity per ingredient
W_MATCH_UNCERTAIN  = 0.30   # classifier's overall assessment
W_COST_COMPLEXITY  = 0.10   # cost-vs-suitability difficulty

# Retrieval-aware weights — NL2SQL path only. Applied as a gated blend:
#   complexity = base * (1 - W_RETRIEVAL_TOTAL) + retrieval_terms
# so the classic path (has_retrieval=False) scores exactly as before.
W_POOL_SIZE        = 0.05   # big per-ingredient pools → more to judge
W_ZERO_HIT         = 0.10   # widening ladder fired → retrieval was guessing
W_VALUE_DISAGREE   = 0.05   # cheapest != best unit value → real reasoning needed
W_RETRIEVAL_TOTAL  = W_POOL_SIZE + W_ZERO_HIT + W_VALUE_DISAGREE

# ─── Thresholds ───────────────────────────────────────────────
# Initial values (0.30 / 0.65) over-escalated on the current golden set:
# 4/5 recipes routed to Sonnet when Haiku handled them at 100% in cascade.
# Bumping the Haiku band up to 0.50 pulls those cases back where they
# belong. Sonnet-tier band unchanged for now.
#
# History:
#   v0.1 (initial):   THRESHOLD_HAIKU=0.30, THRESHOLD_SONNET=0.65
#                     → 4/5 recipes over-routed to Sonnet on eval set.
#   v0.2 (current):   THRESHOLD_HAIKU=0.50, THRESHOLD_SONNET=0.75
#                     → expected: PBJ still Sonnet (real ambiguity), rest Haiku.
THRESHOLD_HAIKU = 0.50
THRESHOLD_SONNET = 0.75


def decide(
    phase_a: PhaseAMetrics,
    phase_b: PhaseBMetrics,
    *,
    escalation_model: str = SONNET,
) -> tuple[str, float, str]:
    """
    Returns (model, complexity_score, reason).

    complexity_score is in [0, 1] — higher = harder task.
    """
    # Normalize each signal to [0, 1]. Higher = harder.
    n_ings_norm      = min(phase_a.ingredient_count / 15, 1.0)
    similarity_gap   = 1.0 - phase_a.mean_max_similarity
    ambiguous_frac   = (
        len(phase_b.ambiguous_ingredients) / max(phase_a.ingredient_count, 1)
    )
    match_uncertain  = (10 - phase_b.match_confidence_1_to_10) / 10
    cost_complexity  = phase_b.cost_complexity_1_to_10 / 10

    complexity = (
        W_INGREDIENT_COUNT * n_ings_norm     +
        W_SIMILARITY_GAP   * similarity_gap  +
        W_AMBIGUOUS_FRAC   * ambiguous_frac  +
        W_MATCH_UNCERTAIN  * match_uncertain +
        W_COST_COMPLEXITY  * cost_complexity
    )

    if phase_a.has_retrieval:
        # Normalize retrieval signals to [0, 1]; blend without disturbing
        # the classic-path scale (see weight comment above).
        pool_norm = min(phase_a.mean_pool_size / 8.0, 1.0)      # 8 = per-ingredient LIMIT
        zero_norm = min(
            phase_a.zero_hit_ingredients / max(phase_a.ingredient_count, 1), 1.0)
        complexity = complexity * (1 - W_RETRIEVAL_TOTAL) + (
            W_POOL_SIZE      * pool_norm +
            W_ZERO_HIT       * zero_norm +
            W_VALUE_DISAGREE * phase_a.value_disagreement
        )

    if complexity < THRESHOLD_HAIKU:
        return HAIKU, complexity, (
            f"low complexity ({complexity:.2f}) → Haiku sufficient"
        )
    elif complexity < THRESHOLD_SONNET:
        return escalation_model, complexity, (
            f"moderate complexity ({complexity:.2f}) → escalation model"
        )
    else:
        return escalation_model, complexity, (
            f"high complexity ({complexity:.2f}) → escalation model "
            "(would enable thinking mode if configured)"
        )

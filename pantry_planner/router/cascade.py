"""
Cascade router: start cheap, escalate on low confidence.

    preselect: return the default (Haiku) — no pre-scoring
    escalate:  re-run any selection whose confidence < THRESHOLD with Sonnet

The two-tier escalation (Haiku → Sonnet only) matches the ARCHITECTURE.md
default. A third tier (Sonnet-with-thinking) would slot in here if
ENABLE_THINKING_ON_ESCALATION is set — see settings for the flag.
"""
from __future__ import annotations

from ..config import settings
from ..models import EscalationDecision, PreselectResult, Product, Recipe, SelectorResult


class CascadeRouter:
    name = "cascade"

    def preselect_model(
        self, recipe: Recipe, products: list[Product], retrieval_stats=None
    ) -> PreselectResult:
        # retrieval_stats is unused: cascade routes by post-hoc confidence.
        cfg = settings()
        return PreselectResult(
            model=cfg.selector_model_default,
            complexity_score=None,
            phase_a=None,
            phase_b=None,
            routing_cost_usd=0.0,   # cascade doesn't spend anything to route
            reason="cascade: start with default model, escalate on low confidence",
        )

    def should_escalate(
        self, initial_result: SelectorResult
    ) -> EscalationDecision:
        cfg = settings()
        threshold = cfg.confidence_threshold

        low_conf = [
            s for s in initial_result.selections if s.confidence < threshold
        ]

        if not low_conf:
            return EscalationDecision(
                escalate=False,
                reason=(
                    f"cascade: all {len(initial_result.selections)} selections "
                    f"met threshold {threshold:.2f}"
                ),
            )

        return EscalationDecision(
            escalate=True,
            ingredients_to_rerun=[s.line_no for s in low_conf],
            escalation_model=cfg.selector_model_escalation,
            reason=(
                f"cascade: {len(low_conf)}/{len(initial_result.selections)} "
                f"selections below threshold {threshold:.2f} — "
                f"escalating to {cfg.selector_model_escalation}"
            ),
        )

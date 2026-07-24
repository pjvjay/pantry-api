"""
Three-phase router: predicts model BEFORE the main selector call.

Phase A — deterministic signals (Jaccard, category density) — 0 cost, <10ms
Phase B — meta-cognitive Haiku classifier — one cheap LLM call
Phase C — weighted-sum threshold picks the model

Never escalates after the fact — the whole point is that the model was
chosen upfront.
"""
from __future__ import annotations

from ..config import settings
from ..models import EscalationDecision, PreselectResult, Product, Recipe, SelectorResult
from .classifier import call_classifier
from .decision import decide
from .deterministic import compute_phase_a


class ThreePhaseRouter:
    name = "three_phase"

    def preselect_model(
        self, recipe: Recipe, products: list[Product], retrieval_stats=None
    ) -> PreselectResult:
        # Phase A — cheap deterministic signals (+ NL2SQL pool stats when present)
        phase_a = compute_phase_a(recipe, products, retrieval_stats=retrieval_stats)

        # Phase B — meta-cognitive classifier
        phase_b = call_classifier(recipe, products)

        # Phase C — weighted-sum thresholding
        model, complexity, reason = decide(
            phase_a, phase_b,
            escalation_model=settings().selector_model_escalation,
        )

        return PreselectResult(
            model=model,
            complexity_score=complexity,
            phase_a=phase_a,
            phase_b=phase_b,
            routing_cost_usd=phase_b.cost_usd,
            reason=reason,
        )

    def should_escalate(
        self, initial_result: SelectorResult
    ) -> EscalationDecision:
        # Three-phase never escalates — it picked the model upfront.
        return EscalationDecision(
            escalate=False,
            reason="three_phase: model was selected upfront by Phase C",
        )

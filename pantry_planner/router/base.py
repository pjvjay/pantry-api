"""
Router protocol — the shared interface for both routing strategies.

The Burr flow calls two hooks:
    preselect_model  — before the main selector call, decide which model to use
    should_escalate  — after the main call, decide if a re-run is warranted

Cascade and ThreePhase both implement this. Which one is used is a config
knob (ROUTING_STRATEGY env var). See ARCHITECTURE.md for the tradeoffs.
"""
from __future__ import annotations

from typing import Protocol

from ..models import PreselectResult, EscalationDecision, Recipe, Product, SelectorResult


class Router(Protocol):
    """The strategy interface. Both routers satisfy this."""

    name: str  # "cascade" | "three_phase" — for tracing

    def preselect_model(
        self,
        recipe: Recipe,
        products: list[Product],
        retrieval_stats=None,
    ) -> PreselectResult:
        """Pick the model the main selector should use.

        Called before the main selector call. May itself make LLM calls
        (three-phase does; cascade doesn't) — the cost of doing so is
        reported in `routing_cost_usd` so we can compare strategies
        end-to-end.
        """
        ...

    def should_escalate(
        self,
        initial_result: SelectorResult,
    ) -> EscalationDecision:
        """Decide whether to re-run any ingredients with a stronger model.

        Called after the main selector call. Cascade returns True for
        low-confidence selections; three-phase always returns False (it
        picked the right model upfront).
        """
        ...

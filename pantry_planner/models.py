"""
Data shapes used across the pipeline. Pydantic for validation on the
API boundary; plain dataclasses inside would also work.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


# ─── Domain models ────────────────────────────────────────────

class RecipeIngredient(BaseModel):
    line_no: int
    name: str
    category: str | None = None  # e.g. "bread", "chocolate", "spread"


class Recipe(BaseModel):
    slug: str
    name: str
    servings: int = 1
    ingredients: list[RecipeIngredient]


class Product(BaseModel):
    id: int
    name: str
    description: str
    price: float                 # CAD
    category: str | None = None


# ─── Selector I/O ─────────────────────────────────────────────

class Selection(BaseModel):
    """One chosen product for one ingredient."""
    line_no: int
    product_id: int
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str = ""


class SelectorResult(BaseModel):
    """Structured output from a single call to the selector LLM."""
    selections: list[Selection]
    total_cost: float
    model_used: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    cost_usd: float = 0.0


# ─── Router I/O ───────────────────────────────────────────────

class PhaseAMetrics(BaseModel):
    ingredient_count: int
    mean_max_similarity: float
    min_max_similarity: float
    count_below_0_3: int
    category_density: float


class PhaseBMetrics(BaseModel):
    match_confidence_1_to_10: int = Field(..., ge=1, le=10)
    cost_complexity_1_to_10: int = Field(..., ge=1, le=10)
    ambiguous_ingredients: list[dict] = Field(default_factory=list)
    confidence_in_own_estimate_1_to_10: int = Field(..., ge=1, le=10)
    reasoning: str = ""
    # Metadata about the classifier call itself:
    cost_usd: float = 0.0
    latency_ms: int = 0


class PreselectResult(BaseModel):
    """What preselect_model() returns."""
    model: str
    complexity_score: float | None = None       # 0..1, or None (cascade)
    phase_a: PhaseAMetrics | None = None
    phase_b: PhaseBMetrics | None = None
    routing_cost_usd: float = 0.0               # cost of the router itself
    reason: str = ""                            # human-readable summary


class EscalationDecision(BaseModel):
    """What should_escalate() returns."""
    escalate: bool
    ingredients_to_rerun: list[int] = Field(default_factory=list)  # line_no values
    escalation_model: str = ""
    reason: str = ""


# ─── Final plan ───────────────────────────────────────────────

class PlanLineItem(BaseModel):
    line_no: int
    ingredient_name: str
    product_id: int
    product_name: str
    product_description: str
    price: float
    confidence: float
    reasoning: str
    model_used: str


class ShoppingPlan(BaseModel):
    recipe_slug: str
    recipe_name: str
    line_items: list[PlanLineItem]
    total_cost: float
    routing_strategy: str
    preselected_model: str
    escalated: bool
    total_llm_cost_usd: float
    total_latency_ms: int

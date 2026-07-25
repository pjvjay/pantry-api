"""The query-plan formalism: retrieval is an explicit sequence of templated
DB queries with abort gates, not one opaque SQL string.

Deterministic by design — the LLM never composes a plan. `build_plan` is a
pure function of the parse (+ location), `execute_plan` (planner.py) runs
the steps in order and evaluates each gate. Every step's SQL, row count,
timing, and outcome are captured so the API can return the full trace.

Imports only pydantic/stdlib on purpose: models.py embeds StepResult and
PlanAlert in the API response, so this module must sit below both models.py
and the rest of nlsearch in the import graph.
"""
from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class StepKind(str, Enum):
    existence = "existence"      # t1 — do we stock every ingredient at all?
    options = "options"          # t2 — offers per ingredient under constraints
    statistics = "statistics"    # t3 — per-brand price/rating stats for the pools
    lookup = "lookup"            # t4 — extra lookups (substitutes for thin pools)


class GateCode(str, Enum):
    missing_ingredients = "missing_ingredients"                    # t1 abort
    unavailable_within_constraints = "unavailable_within_constraints"  # t2 abort
    budget_infeasible = "budget_infeasible"                        # t2 abort (floor)


class QueryStep(BaseModel):
    id: str                          # "t1_existence", "t2_options", "t4_lookup_3"
    kind: StepKind
    template: str                    # named template in sql_builder's registry
    params_summary: dict = Field(default_factory=dict)   # display copy of bindings
    gate: GateCode | None = None     # abort condition evaluated on this step


class QueryPlan(BaseModel):
    """Deterministic order: t1 -> t2 -> t3; t4 steps are appended during
    execution when a pool comes back thin (data-driven, never model-driven)."""
    steps: list[QueryStep]


class StepResult(BaseModel):
    step_id: str
    kind: StepKind
    label: str = ""                  # human summary: "6 ingredients stocked"
    sql_display: str = ""            # params inlined — display only
    row_count: int = 0
    duration_ms: int = 0
    outcome: Literal["ok", "aborted", "skipped"] = "ok"


class PlanAlert(BaseModel):
    """User-facing abort payload — which gate fired, for what, and what to try."""
    stage: str                       # step id that aborted
    code: GateCode
    message: str
    details: list[dict] = Field(default_factory=list)   # per-ingredient: name, reason, suggestions[]


class PlanExecution(BaseModel):
    steps: list[StepResult] = Field(default_factory=list)
    aborted: PlanAlert | None = None

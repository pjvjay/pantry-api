"""Natural-language recipe → staged SQL query plan (constrained NL2SQL).

Pipeline (NL2SQL_Handbook taxonomy):
  PRE-PROCESSING   units.py (normalization), vocab.py (schema linking)
  TRANSLATION      query_parser.py (multi-shot semantic parse)
                   + sql_builder.py (named templates, pattern-menu compilation)
  POST-PROCESSING  query_parser.validate_parsed (clamping),
                   plan.py + planner.py (query-plan execution + abort gates)

The executor symbols are exported lazily: models.py imports the plan
formalism (StepResult/PlanAlert) from here, while planner.py imports
models back — eager re-exports would make that a cycle.
"""
from .plan import (GateCode, PlanAlert, PlanExecution, QueryPlan,  # noqa: F401
                   QueryStep, StepKind, StepResult)

_PLANNER_EXPORTS = {"run_query_plan", "execute_plan", "build_plan",
                    "PlanRunResult", "PlanAborted", "UnparseableRecipe",
                    "build_recipe"}


def __getattr__(name: str):
    if name in _PLANNER_EXPORTS:
        from . import planner
        return getattr(planner, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

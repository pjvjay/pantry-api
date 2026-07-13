"""
Burr state machine — the whole pipeline as a 6-action graph.

    load_recipe
      → load_products
      → preselect_model         (router.preselect_model)
      → select_products         (main LLM call using preselected model)
      → check_escalation        (router.should_escalate)
      → escalate_if_needed      (conditional; re-runs subset if cascade said so)
      → build_plan

Both routers use the same graph. The only difference is which nodes
"do work":
  * cascade: preselect is a no-op (returns Haiku); check_escalation may
    fire; escalate_if_needed makes the second call.
  * three_phase: preselect calls the classifier (real LLM cost);
    check_escalation always returns False; escalate_if_needed is skipped.
"""
from __future__ import annotations

from typing import Any

from burr.core import Application, ApplicationBuilder, State, action, expr

from . import db
from .config import get_router, settings
from .models import (
    EscalationDecision,
    PlanLineItem,
    PreselectResult,
    Product,
    Recipe,
    SelectorResult,
    ShoppingPlan,
)
from .selector import call_selector, merge_selections
from .tracing import llm_span, make_tracker


# ─── Actions ─────────────────────────────────────────────────

@action(reads=[], writes=["recipe"])
def load_recipe(state: State, recipe_slug: str) -> tuple[dict, State]:
    recipe = db.load_recipe(recipe_slug)
    result = {"ingredient_count": len(recipe.ingredients)}
    return result, state.update(recipe=recipe)


@action(reads=[], writes=["products"])
def load_products(state: State) -> tuple[dict, State]:
    products = db.load_all_products()
    result = {"product_count": len(products)}
    return result, state.update(products=products)


@action(reads=["recipe", "products"], writes=["preselect_result"])
def preselect_model(state: State) -> tuple[dict, State]:
    router = get_router()
    recipe: Recipe = state["recipe"]
    products: list[Product] = state["products"]

    preselect: PreselectResult = router.preselect_model(recipe, products)

    result = {
        "router": router.name,
        "chosen_model": preselect.model,
        "complexity_score": preselect.complexity_score,
        "routing_cost_usd": preselect.routing_cost_usd,
        "reason": preselect.reason,
    }
    return result, state.update(preselect_result=preselect)


@action(reads=["recipe", "products", "preselect_result"], writes=["initial_result"])
def select_products(state: State) -> tuple[dict, State]:
    recipe: Recipe = state["recipe"]
    products: list[Product] = state["products"]
    preselect: PreselectResult = state["preselect_result"]

    result: SelectorResult = call_selector(
        recipe.ingredients,
        products,
        model=preselect.model,
        enable_thinking=False,
    )

    span = llm_span(
        step="select_products",
        model=result.model_used,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cost_usd=result.cost_usd,
        latency_ms=result.latency_ms,
    )
    return {"llm_call": span, "n_selections": len(result.selections)}, \
        state.update(initial_result=result)


@action(reads=["initial_result"], writes=["escalation_decision"])
def check_escalation(state: State) -> tuple[dict, State]:
    router = get_router()
    decision: EscalationDecision = router.should_escalate(state["initial_result"])
    return {
        "escalate": decision.escalate,
        "n_ingredients_to_rerun": len(decision.ingredients_to_rerun),
        "reason": decision.reason,
    }, state.update(escalation_decision=decision)


@action(
    reads=["recipe", "products", "initial_result", "escalation_decision"],
    writes=["final_result"],
)
def escalate_if_needed(state: State) -> tuple[dict, State]:
    """Only runs when escalation_decision.escalate == True (routed via graph)."""
    recipe: Recipe = state["recipe"]
    products: list[Product] = state["products"]
    initial: SelectorResult = state["initial_result"]
    decision: EscalationDecision = state["escalation_decision"]

    # Re-run just the flagged ingredients through the escalation model.
    flagged_ingredients = [
        i for i in recipe.ingredients if i.line_no in decision.ingredients_to_rerun
    ]

    escalated = call_selector(
        flagged_ingredients,
        products,
        model=decision.escalation_model,
        enable_thinking=settings().enable_thinking_on_escalation,
    )

    merged = merge_selections(initial, escalated, decision.ingredients_to_rerun)

    span = llm_span(
        step="escalate",
        model=escalated.model_used,
        input_tokens=escalated.input_tokens,
        output_tokens=escalated.output_tokens,
        cost_usd=escalated.cost_usd,
        latency_ms=escalated.latency_ms,
    )
    return {"llm_call": span, "n_reran": len(flagged_ingredients)}, \
        state.update(final_result=merged)


@action(reads=["initial_result", "final_result"], writes=["final_result"])
def skip_escalation(state: State) -> tuple[dict, State]:
    """No-op — hoists initial_result into final_result when we don't escalate."""
    return {"escalated": False}, state.update(final_result=state["initial_result"])


@action(
    reads=["recipe", "products", "final_result", "preselect_result", "escalation_decision"],
    writes=["plan"],
)
def build_plan(state: State) -> tuple[dict, State]:
    recipe: Recipe = state["recipe"]
    products_by_id = {p.id: p for p in state["products"]}
    final: SelectorResult = state["final_result"]
    preselect: PreselectResult = state["preselect_result"]
    decision: EscalationDecision = state["escalation_decision"]
    ingredients_by_line = {i.line_no: i for i in recipe.ingredients}

    line_items: list[PlanLineItem] = []
    total_cost = 0.0
    for s in final.selections:
        prod = products_by_id.get(s.product_id)
        ing = ingredients_by_line.get(s.line_no)
        if prod is None or ing is None:
            # Defensive: the LLM referenced a product_id or line_no we
            # don't have. Skip; downstream can flag/re-run.
            continue
        line_items.append(PlanLineItem(
            line_no=s.line_no,
            ingredient_name=ing.name,
            product_id=prod.id,
            product_name=prod.name,
            product_description=prod.description,
            price=prod.price,
            confidence=s.confidence,
            reasoning=s.reasoning,
            model_used=final.model_used,
        ))
        total_cost += prod.price

    plan = ShoppingPlan(
        recipe_slug=recipe.slug,
        recipe_name=recipe.name,
        line_items=line_items,
        total_cost=round(total_cost, 2),
        routing_strategy=get_router().name,
        preselected_model=preselect.model,
        escalated=decision.escalate,
        total_llm_cost_usd=round(
            preselect.routing_cost_usd + final.cost_usd, 6
        ),
        total_latency_ms=final.latency_ms,
    )
    return {"total_cost": plan.total_cost, "n_line_items": len(plan.line_items)}, \
        state.update(plan=plan)


# ─── Application builder ─────────────────────────────────────

def build_application(recipe_slug: str) -> Application:
    """Construct the Burr Application for one run.

    Conditional transitions:
      * check_escalation → escalate_if_needed  if escalation_decision.escalate
      * check_escalation → skip_escalation     otherwise
    """
    return (
        ApplicationBuilder()
        .with_actions(
            load_recipe.bind(recipe_slug=recipe_slug),
            load_products,
            preselect_model,
            select_products,
            check_escalation,
            escalate_if_needed,
            skip_escalation,
            build_plan,
        )
        .with_transitions(
            ("load_recipe", "load_products"),
            ("load_products", "preselect_model"),
            ("preselect_model", "select_products"),
            ("select_products", "check_escalation"),
            (
                "check_escalation",
                "escalate_if_needed",
                expr("escalation_decision.escalate == True"),
            ),
            (
                "check_escalation",
                "skip_escalation",
                expr("escalation_decision.escalate == False"),
            ),
            ("escalate_if_needed", "build_plan"),
            ("skip_escalation", "build_plan"),
        )
        .with_entrypoint("load_recipe")
        .with_tracker(make_tracker())
        .with_identifiers(app_id=f"run-{recipe_slug}")
        .build()
    )


def run(recipe_slug: str) -> ShoppingPlan:
    """Run the pipeline end-to-end. Returns the final ShoppingPlan."""
    app = build_application(recipe_slug)
    _action, _result, state = app.run(halt_after=["build_plan"])
    return state["plan"]

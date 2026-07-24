"""
The main selector — one Anthropic call, structured output via tool use.

Given a recipe + a subset of products, returns per-item choices with
per-item confidence. The confidence is what the cascade router keys off.

We use tool_choice to force the model to call submit_plan — this
guarantees valid structured output. No JSON parsing.
"""
from __future__ import annotations

import json
import time

from anthropic import Anthropic

from .config import estimate_cost_usd, settings
from .models import Product, Recipe, RecipeIngredient, Selection, SelectorResult
from .prompts import SELECTOR_SYSTEM, SELECTOR_TOOL


def _serialize_products(products: list[Product]) -> list[dict]:
    out = []
    for p in products:
        d = {
            "id": p.id,
            "name": p.name,
            "description": p.description,
            "price": p.price,
            "category": p.category,
        }
        # NL2SQL-era attributes — omitted when absent to keep tokens lean
        if p.subcategory:
            d["subcategory"] = p.subcategory
        if p.dietary_tags:
            d["contains"] = p.dietary_tags
        if p.unit_size:
            d["unit_size"] = p.unit_size
        out.append(d)
    return out


def _serialize_ingredients(ingredients: list[RecipeIngredient]) -> list[dict]:
    return [
        {"line_no": i.line_no, "name": i.name, "category": i.category}
        for i in ingredients
    ]


def call_selector(
    ingredients: list[RecipeIngredient],
    products: list[Product],
    *,
    model: str,
    enable_thinking: bool = False,
    constraints: dict | None = None,
) -> SelectorResult:
    """Make one main-selector call. Returns structured selections.

    `constraints` (NL2SQL path) carries binding shopping constraints —
    budget, quantities needed, preferences — see SELECTOR_SYSTEM rule 6.
    """
    cfg = settings()
    client = Anthropic(api_key=cfg.anthropic_api_key)

    payload: dict = {
        "recipe_ingredients": _serialize_ingredients(ingredients),
        "available_products": _serialize_products(products),
        "objective": "cost",
    }
    if constraints:
        payload["constraints"] = constraints
    user_msg = json.dumps(payload, indent=2)

    kwargs: dict = {
        "model": model,
        "max_tokens": 4096,
        "system": SELECTOR_SYSTEM,
        "tools": [SELECTOR_TOOL],
        "tool_choice": {"type": "tool", "name": "submit_plan"},
        "messages": [{"role": "user", "content": user_msg}],
    }

    # Thinking is opt-in — the escalation model may or may not use it.
    if enable_thinking:
        kwargs["thinking"] = {"type": "enabled", "budget_tokens": 4000}

    t0 = time.perf_counter()
    resp = client.messages.create(**kwargs)
    latency_ms = int((time.perf_counter() - t0) * 1000)

    tool_block = next((b for b in resp.content if b.type == "tool_use"), None)
    if tool_block is None:
        raise ValueError(f"Selector didn't call the tool. Response: {resp.content!r}")

    args = tool_block.input
    selections = [
        Selection(
            line_no=s["line_no"],
            product_id=s["product_id"],
            confidence=s["confidence"],
            reasoning=s["reasoning"],
        )
        for s in args["selections"]
    ]

    return SelectorResult(
        selections=selections,
        total_cost=float(args["total_cost"]),
        model_used=model,
        input_tokens=resp.usage.input_tokens,
        output_tokens=resp.usage.output_tokens,
        latency_ms=latency_ms,
        cost_usd=estimate_cost_usd(model, resp.usage.input_tokens, resp.usage.output_tokens),
    )


def merge_selections(
    base: SelectorResult,
    escalated: SelectorResult,
    escalated_line_nos: list[int],
) -> SelectorResult:
    """Overlay escalated selections onto the base result. Used by cascade.

    Only ingredients that were re-run overwrite the base; the rest stay.
    Cost and latency accumulate across both calls.
    """
    by_line = {s.line_no: s for s in base.selections}
    for s in escalated.selections:
        if s.line_no in escalated_line_nos:
            by_line[s.line_no] = s

    merged_selections = [by_line[k] for k in sorted(by_line.keys())]

    # Recompute total_cost from actual selections (in case escalation
    # changed which product is chosen — new price may differ).
    return SelectorResult(
        selections=merged_selections,
        # total_cost recomputed downstream once products are joined back in
        total_cost=base.total_cost,   # placeholder; build_plan recomputes
        model_used=f"{base.model_used}+{escalated.model_used}",
        input_tokens=base.input_tokens + escalated.input_tokens,
        output_tokens=base.output_tokens + escalated.output_tokens,
        latency_ms=base.latency_ms + escalated.latency_ms,
        cost_usd=base.cost_usd + escalated.cost_usd,
    )

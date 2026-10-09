"""
The main selector — one LLM call, structured output via tool use.

Given a recipe + a subset of products, returns per-item choices with
per-item confidence. The confidence is what the cascade router keys off.

We force the model to call submit_plan (llm.forced_tool_call) — this
guarantees structured output. The model spec picks the provider
("gemini:<model>" or an Anthropic name).
"""
from __future__ import annotations

import json
import time

from .config import estimate_cost_usd, settings
from .llm import forced_tool_call
from .models import Product, RecipeIngredient, Selection, SelectorResult
from .prompts import SELECTOR_SYSTEM, SELECTOR_TOOL

# Extended thinking on escalation (Anthropic only; Gemini ignores it)
THINKING_BUDGET_TOKENS = 4000


def _serialize_products(products: list[Product],
                        origins_by_id: dict | None = None) -> list[dict]:
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
        # Query-plan path: each candidate is pinned to its best store offer
        if p.brand:
            d["brand"] = p.brand
        if p.store_name:
            d["store"] = p.store_name
            d["price"] = p.store_price          # the offer price is what's charged
            d["distance_km"] = p.distance_km
        if p.substitute:
            d["substitute"] = True              # t4 alternative, not a direct match
        # Provenance, when known. Products positively evidenced as coming
        # from an excluded country are filtered out before they reach here;
        # this is shown so the model can PREFER on origin and explain why,
        # and so it never has to guess at a country it was not told.
        origin = origins_by_id.get(p.id) if origins_by_id else None
        if origin is not None and origin.status == "resolved":
            d["origin"] = {
                "country": origin.country,
                "claim": origin.claim_type,
                "ingredient_origin": origin.ingredient_origin or None,
            }
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
    origins_by_id: dict | None = None,
    preference: list[str] | None = None,
    substitutes: dict[int, list[int]] | None = None,
) -> SelectorResult:
    """Make one main-selector call. Returns structured selections.

    `constraints` (NL2SQL path) carries binding shopping constraints —
    budget, quantities needed, preferences — see SELECTOR_SYSTEM rule 6.

    `substitutes` (NL2SQL path): recipe line_no -> the product ids that
    line's own pool holds only as t4 substitutes. The demo selector keeps
    them out of that line's choice only. The live payload is unchanged: a
    product is flagged "substitute" there when it substitutes for every
    line that retrieved it (planner.union_of_pools).
    """
    cfg = settings()
    if cfg.demo_mode:                       # public demo: no API key, no cost
        from . import demomode
        return demomode.select_products(ingredients, products, model=model,
                                        enable_thinking=enable_thinking,
                                        constraints=constraints,
                                        origins_by_id=origins_by_id,
                                        preference=preference,
                                        substitutes=substitutes)
    payload: dict = {
        "recipe_ingredients": _serialize_ingredients(ingredients),
        "available_products": _serialize_products(products, origins_by_id),
        "objective": "cost",
    }
    if constraints:
        payload["constraints"] = constraints
    user_msg = json.dumps(payload, indent=2)

    t0 = time.perf_counter()
    reply = forced_tool_call(
        model=model,
        system=SELECTOR_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
        tool=SELECTOR_TOOL,
        max_tokens=4096,
        # Thinking is opt-in — the escalation model may or may not use it.
        thinking_budget=THINKING_BUDGET_TOKENS if enable_thinking else None,
    )
    latency_ms = int((time.perf_counter() - t0) * 1000)

    args = reply.input
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
        model_used=model,               # the spec: "gemini:<model>" or the Anthropic name
        input_tokens=reply.input_tokens,
        output_tokens=reply.output_tokens,
        latency_ms=latency_ms,
        cost_usd=estimate_cost_usd(model, reply.input_tokens, reply.output_tokens),
        http=[reply.http] if reply.http else [],
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
        http=[*base.http, *escalated.http],
    )

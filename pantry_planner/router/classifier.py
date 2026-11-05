"""
Phase B of the three-phase router: meta-cognitive Haiku classifier.

Design notes:
 - We only pass the catalog SUMMARY (category counts), NOT the individual
   products. This prevents the classifier from doing the matching itself.
 - The prompt frames the model as an "LLM systems expert" giving a
   REASONABLE estimate for a GENERAL matching LLM — not for itself. This
   third-person framing reduces overconfidence bias empirically.
 - We record `confidence_in_own_estimate` as a second-order calibration
   signal we can plot against actual selector accuracy on the eval set.
"""
from __future__ import annotations

import json
import time
from collections import Counter

from ..config import estimate_cost_usd, settings
from ..llm import forced_tool_call
from ..models import PhaseBMetrics, Product, Recipe
from ..prompts import CLASSIFIER_SYSTEM, CLASSIFIER_TOOL


def _catalog_summary(products: list[Product]) -> dict:
    """Compact description of the catalog — NOT the products themselves.

    The classifier can reason about problem shape without being able to
    do the matching directly.
    """
    by_category = Counter(p.category or "uncategorized" for p in products)
    prices = [p.price for p in products]
    return {
        "total_products": len(products),
        "categories": [
            {"category": cat, "count": count}
            for cat, count in sorted(by_category.items(), key=lambda x: -x[1])
        ],
        "price_range": {
            "min": min(prices) if prices else 0.0,
            "max": max(prices) if prices else 0.0,
            "median": sorted(prices)[len(prices) // 2] if prices else 0.0,
        },
    }


def call_classifier(recipe: Recipe, products: list[Product]) -> PhaseBMetrics:
    """One CLASSIFIER_MODEL call (Haiku by default; a "gemini:<model>" spec
    routes to Gemini). Meta-cognitive prompt. Structured output via tool use."""
    cfg = settings()
    if cfg.demo_mode:                       # public demo: fixed triage, no call
        from .. import demomode
        return PhaseBMetrics(**demomode.triage())
    user_msg = json.dumps({
        "recipe": {
            "name": recipe.name,
            "ingredients": [
                {"line_no": i.line_no, "name": i.name, "category": i.category}
                for i in recipe.ingredients
            ],
        },
        "catalog_summary": _catalog_summary(products),
    }, indent=2)

    t0 = time.perf_counter()
    reply = forced_tool_call(
        model=cfg.classifier_model,
        max_tokens=1024,
        system=CLASSIFIER_SYSTEM,
        tool=CLASSIFIER_TOOL,
        messages=[{"role": "user", "content": user_msg}],
    )
    latency_ms = int((time.perf_counter() - t0) * 1000)

    args = reply.input
    return PhaseBMetrics(
        match_confidence_1_to_10=args["match_confidence_1_to_10"],
        cost_complexity_1_to_10=args["cost_complexity_1_to_10"],
        ambiguous_ingredients=args["ambiguous_ingredients"],
        confidence_in_own_estimate_1_to_10=args["confidence_in_own_estimate_1_to_10"],
        reasoning=args["reasoning"],
        cost_usd=estimate_cost_usd(
            cfg.classifier_model, reply.input_tokens, reply.output_tokens
        ),
        latency_ms=latency_ms,
    )

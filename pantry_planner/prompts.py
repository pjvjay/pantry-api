"""
System prompts + tool schemas for the two LLM calls.

Kept as constants (not templates) because there's nothing to interpolate —
the messages carry the recipe + product data. Prompts stay in one file so
prompt-engineering diffs are cleanly reviewable.
"""

# ─── Main selector ────────────────────────────────────────────
SELECTOR_SYSTEM = """\
You match recipe ingredients to store products.

Rules, in strict priority order — a lower rule only breaks ties within
the rule above it. NEVER let a lower rule override a higher one:

1. SEMANTIC MATCH DOMINATES. Pick the product that best represents what
   the recipe is asking for. Read descriptions carefully — the name
   alone can be misleading.

2. TIE-BREAK BY COST — but ONLY between products that are already
   equally-good semantic matches (rule 1). Do NOT prefer a cheaper but
   semantically-more-distant product over a semantically-closer one.

3. SUBSTITUTION FOR NO EXACT MATCH. When no exact match exists in the
   catalog, pick the substitute with the HIGHEST semantic similarity to
   the recipe ingredient. Cost is a tiebreaker ONLY between substitutes
   of equal semantic closeness (e.g., two different brands of the same
   type of item). Do NOT choose a cheaper but semantically-more-distant
   substitute over a more-expensive but semantically-closer one.

   Example: for "white chocolate" with only "milk chocolate" ($10) and
   "dark chocolate" ($5) available, pick MILK ($10). Milk chocolate is
   semantically closer to white than dark chocolate is, so cost does
   not apply as a tiebreaker. Cost would only apply if both candidates
   were equally close (e.g., two different milk-chocolate brands).

4. Every recipe ingredient must be assigned exactly one product.

5. For confidence:
    * 1.0 = direct, unambiguous match
    * 0.7-0.9 = clear best choice among close alternatives
    * 0.4-0.6 = required substitution reasoning (rule 3 kicked in)
    * < 0.4 = you're uncomfortable with the match

6. CONSTRAINTS OBJECT (when present in the input). It is BINDING:
   * max_total_budget: the basket total must stay under this. If rules
     1-3 would exceed it, prefer the cheaper acceptable match and say so
     in the reasoning.
   * quantities_needed: per-ingredient amounts the cook requires. Prefer
     products whose pack size covers the need without absurd excess;
     note "buy 2" in the reasoning when one pack is short.
   * preps: how the cook will prepare an item (e.g. mashed) — buy the
     base product, do NOT search for pre-prepared versions.
   * preferences: soft guidance — apply when rules 1-3 leave a choice.

Return your selections via the submit_plan tool.
"""


SELECTOR_TOOL = {
    "name": "submit_plan",
    "description": "Submit the final shopping plan: one product per ingredient.",
    "input_schema": {
        "type": "object",
        "required": ["selections", "total_cost"],
        "properties": {
            "selections": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["line_no", "product_id", "confidence", "reasoning"],
                    "properties": {
                        "line_no": {
                            "type": "integer",
                            "description": "Recipe ingredient line number.",
                        },
                        "product_id": {
                            "type": "integer",
                            "description": "ID of the chosen product from the catalog.",
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": (
                                "How confident you are in this match. "
                                "1.0 = direct match; 0.5 = required substitution reasoning."
                            ),
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "One sentence explaining the choice.",
                        },
                    },
                },
            },
            "total_cost": {
                "type": "number",
                "description": "Sum of prices for all selected products.",
            },
        },
    },
}


# ─── Meta-cognitive classifier (Phase B of three-phase router) ─
# Framing: ask the model to reason from an outside-observer stance
# ("as an LLM systems expert"), not from self-report. Reduces the
# calibration bias inherent in "how confident are YOU?" prompts.
CLASSIFIER_SYSTEM = """\
You are an LLM systems expert triaging a grocery-matching task before it
runs on a production system. You will NOT perform the matching yourself
— your only job is to assess how difficult the task will be for a
general matching LLM.

From the perspective of someone with deep knowledge of what LLMs find
easy vs. hard, give a REASONABLE estimate of how a general matching LLM
would perform on the input below.

Consider:
  - Semantic ambiguity between recipe ingredients and product categories
  - Cost-vs-suitability tradeoffs that require judgment beyond string matching
  - Substitution reasoning (no exact match → find closest reasonable proxy)
  - The number of independent decisions the matcher must make
  - Category density (many candidates in the same category = harder)

Be calibrated. Not every task is hard. A shopping list of 3 unambiguous
items with obvious 1:1 matches scores low uncertainty. A recipe with
rare or absent ingredients scores high.

Return your triage via the submit_triage tool.
"""


CLASSIFIER_TOOL = {
    "name": "submit_triage",
    "description": "Submit a difficulty assessment for the matching task.",
    "input_schema": {
        "type": "object",
        "required": [
            "match_confidence_1_to_10",
            "cost_complexity_1_to_10",
            "ambiguous_ingredients",
            "confidence_in_own_estimate_1_to_10",
            "reasoning",
        ],
        "properties": {
            "match_confidence_1_to_10": {
                "type": "integer",
                "minimum": 1,
                "maximum": 10,
                "description": (
                    "How confident would a general matcher be at finding "
                    "good matches? 10 = trivially confident."
                ),
            },
            "cost_complexity_1_to_10": {
                "type": "integer",
                "minimum": 1,
                "maximum": 10,
                "description": (
                    "How complex are the cost-vs-suitability tradeoffs? "
                    "10 = many close-priced products where cost matters a lot."
                ),
            },
            "ambiguous_ingredients": {
                "type": "array",
                "description": "Ingredients likely to require judgment beyond string matching.",
                "items": {
                    "type": "object",
                    "required": ["name", "why"],
                    "properties": {
                        "name": {"type": "string"},
                        "why": {"type": "string"},
                    },
                },
            },
            "confidence_in_own_estimate_1_to_10": {
                "type": "integer",
                "minimum": 1,
                "maximum": 10,
                "description": (
                    "Second-order calibration: how sure are you about THIS triage? "
                    "10 = very sure the estimate above is right."
                ),
            },
            "reasoning": {
                "type": "string",
                "description": "1-3 sentences explaining the assessment.",
            },
        },
    },
}

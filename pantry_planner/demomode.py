"""DEMO_MODE: deterministic stand-ins for the two LLM boundaries.

The public demo (Hugging Face Space) runs with no ANTHROPIC_API_KEY —
these functions replace exactly the two places the pipeline talks to
Claude, and nothing else:

  * parse_recipe()    stands in for nlsearch.query_parser.parse_input
  * select_products() stands in for selector.call_selector
  * triage()          stands in for the three_phase Phase B classifier

Everything between the boundaries — the staged query plan, abort gates,
brand statistics, split-trip and weekly optimizers — is deterministic
code that runs identically in both modes. Responses are labeled
model_used="demo-deterministic" and the /health endpoint reports
demo_mode so the UI can say so honestly.

The stand-ins are deliberately simple (regex recipe parse; token-overlap
+ price product ranking): good enough to drive the demo, visibly not
the point of the project.
"""
from __future__ import annotations

import re

from .models import Product, RecipeIngredient, Selection, SelectorResult
from .nlsearch import lineparse
from .nlsearch.schemas import Constraints, IngredientSpec, ParsedInput, RecipeSpec
from .nlsearch.units import semantic_key

_TAGS = ("dairy", "gluten", "meat", "nuts", "egg", "soy")


def parse_recipe(text_input: str, *, model: str | None = None) -> ParsedInput:
    """Regex recipe parse: title, servings, '- qty unit name' ingredient
    lines, and budget / distance / dietary constraints from a Notes line.
    Same failure contract as the real parser: empty parse, never raises.

    Each line is read by nlsearch.lineparse.parse_line, which is consistent
    with the live parser on what to buy (see there); this function decides
    which lines are ingredient lines and reads the rest of the recipe."""
    lines = [ln.strip() for ln in text_input.splitlines() if ln.strip()]
    title = lines[0].split("(")[0].strip() if lines else "Pasted recipe"
    stated = lineparse.parse_servings(text_input)
    servings = stated if stated is not None else 1

    ingredients: list[IngredientSpec] = []
    notes = ""
    for ln in lines[1:]:
        if ln.lower().startswith(("notes:", "note:")):
            notes = ln.split(":", 1)[1]
            continue
        if not ln.startswith(("-", "*", "•")):
            continue
        p = lineparse.parse_line(ln)
        ingredients.append(IngredientSpec(name=p.name, form=p.form, quantity=p.quantity,
                                          unit=p.unit, prep=p.prep))

    cons = Constraints()
    scope = notes or text_input
    if m := re.search(r"under\s*\$\s*(\d+(?:\.\d+)?)", scope, re.IGNORECASE):
        cons.max_total_budget = float(m.group(1))
    if m := re.search(r"within\s+(\d+(?:\.\d+)?)\s*km", scope, re.IGNORECASE):
        cons.max_distance_km = float(m.group(1))
    for tag in _TAGS:
        if re.search(rf"no {tag}|{tag}[- ]free", scope, re.IGNORECASE):
            cons.exclude_tags.append(tag)
    return ParsedInput(recipe=RecipeSpec(title=title, servings=servings,
                                         servings_stated=stated is not None,
                                         ingredients=ingredients),
                       constraints=cons, cost_usd=0.0, latency_ms=0)


def _preference_rank(origin, preference: list[str]) -> int:
    from .origins import preference_rank
    return preference_rank(origin, preference)


def select_products(ingredients: list[RecipeIngredient],
                    products: list[Product], *, model: str,
                    enable_thinking: bool = False,
                    constraints: dict | None = None,
                    origins_by_id: dict | None = None,
                    preference: list[str] | None = None,
                    substitutes: dict[int, list[int]] | None = None) -> SelectorResult:
    """Token-overlap, then "fresh" when the ingredient says it, then head
    noun, then origin preference, then offer price; direct matches before
    t4 substitutes. Confidence is fixed at 0.9
    — above the cascade threshold, so demo mode never triggers a (would-be)
    escalation.

    A t4 substitute is a substitute for the line whose thin pool fetched it,
    not for every line: `substitutes` (line_no -> product ids, the NL and
    spec paths) keeps each line's own substitutes out of that line's choice
    only, so Fresh Ginger, fetched as a same-aisle substitute for garlic, is
    still the pick for ginger. Without it (the library path, where nothing
    is a substitute) the products' own flags are read.

    The head-noun key breaks overlap ties the way a shopper would: for
    "flour", All-Purpose Flour (about flour) beats Flour Tortillas (about
    tortillas, though cheaper); for "mustard", Dijon Mustard beats Black
    Mustard Seeds. Preference applies strictly AFTER semantic match, the
    same priority the live selector prompt gives it: it only breaks ties
    between equally-good matches, never trades correctness for origin."""
    origins_by_id = origins_by_id or {}
    preference = [c for c in (preference or []) if c.strip()]
    selections: list[Selection] = []
    for ing in ingredients:
        if substitutes is None:
            pool = [p for p in products if not p.substitute] or products
        else:
            subs = set(substitutes.get(ing.line_no, ()))
            pool = [p for p in products if p.id not in subs] or products
        if not pool:
            continue
        # semantic_key is the ranking the alternatives share: overlap, then
        # "fresh", then head noun.
        pick = min(pool, key=lambda p: (
            *semantic_key(ing.name, p),
            _preference_rank(origins_by_id.get(p.id), preference),
            p.store_price if p.store_price is not None else p.price,
            p.id))
        why = "demo mode: highest token overlap, then head noun, then cheapest offer"
        if preference:
            why = ("demo mode: highest token overlap, then head noun, then origin "
                   f"preference ({', '.join(preference)}), then cheapest offer")
        selections.append(Selection(
            line_no=ing.line_no, product_id=pick.id, confidence=0.9,
            reasoning=why))
    return SelectorResult(selections=selections, total_cost=0.0,
                          model_used="demo-deterministic",
                          input_tokens=0, output_tokens=0,
                          latency_ms=0, cost_usd=0.0)


def triage() -> dict:
    """Fixed Phase B verdict for the three_phase router in demo mode."""
    return {
        "match_confidence_1_to_10": 8,
        "cost_complexity_1_to_10": 3,
        "ambiguous_ingredients": [],
        "confidence_in_own_estimate_1_to_10": 9,
        "reasoning": "demo mode: fixed triage, no classifier call",
    }

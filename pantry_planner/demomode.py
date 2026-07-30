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
from .nlsearch.schemas import Constraints, IngredientSpec, ParsedInput, RecipeSpec
from .nlsearch.units import tokens

_QTY = re.compile(
    r"^(\d+(?:\.\d+)?)\s*(g|kg|ml|l|lb|oz|cups?|cans?|cloves?|dozen)?\s+",
    re.IGNORECASE)
_FORMS = {"canned", "frozen", "dried", "ground", "smoked", "pickled", "powdered"}
_TAGS = ("dairy", "gluten", "meat", "nuts", "egg", "soy")


def parse_recipe(text_input: str, *, model: str | None = None) -> ParsedInput:
    """Regex recipe parse: title, servings, '- qty unit name' ingredient
    lines, and budget / distance / dietary constraints from a Notes line.
    Same failure contract as the real parser: empty parse, never raises."""
    lines = [ln.strip() for ln in text_input.splitlines() if ln.strip()]
    title = lines[0].split("(")[0].strip() if lines else "Pasted recipe"
    m = re.search(r"serves\s+(\d+)", text_input, re.IGNORECASE)
    servings = int(m.group(1)) if m else 1

    ingredients: list[IngredientSpec] = []
    notes = ""
    for ln in lines[1:]:
        if ln.lower().startswith(("notes:", "note:")):
            notes = ln.split(":", 1)[1]
            continue
        if not ln.startswith(("-", "*", "•")):
            continue
        item = ln.lstrip("-*• ").strip()
        qty = unit = None
        if qm := _QTY.match(item):
            qty = float(qm.group(1))
            unit = (qm.group(2) or "each").lower()
            item = item[qm.end():]
        words = [w for w in item.lower().split() if w not in {"of", "fresh"}]
        form = None
        if words and words[0] in _FORMS:
            form = words.pop(0)
        if unit and unit.startswith("can"):
            form, unit = "canned", "can"
        name = " ".join(words).strip() or item
        ingredients.append(IngredientSpec(name=name, form=form,
                                          quantity=qty, unit=unit))

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
                                         ingredients=ingredients),
                       constraints=cons, cost_usd=0.0, latency_ms=0)


def select_products(ingredients: list[RecipeIngredient],
                    products: list[Product], *, model: str,
                    enable_thinking: bool = False,
                    constraints: dict | None = None) -> SelectorResult:
    """Token-overlap + offer-price ranking, direct matches before t4
    substitutes. Confidence is fixed at 0.9 — above the cascade threshold,
    so demo mode never triggers a (would-be) escalation call."""
    selections: list[Selection] = []
    for ing in ingredients:
        toks = set(tokens(ing.name))
        pool = [p for p in products if not p.substitute] or products
        if not pool:
            continue
        pick = min(pool, key=lambda p: (
            -len(toks & set(tokens(f"{p.name} {p.description}"))),
            p.store_price if p.store_price is not None else p.price,
            p.id))
        selections.append(Selection(
            line_no=ing.line_no, product_id=pick.id, confidence=0.9,
            reasoning="demo mode: highest token overlap, then cheapest offer"))
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

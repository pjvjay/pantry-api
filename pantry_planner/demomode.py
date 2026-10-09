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
from .nlsearch.units import head_noun, index_text, tokens

# A leading amount: "2", "1.5", "1/4", "1 1/2", "1½", "¾", and a range's
# lower end ("1 to 3", "1-3") — then an optional unit. Units normalize_quantity
# knows become a size need; the rest (inch, pinch, bunch, ...) only count.
_NUM = r"(?:\d+(?:\.\d+)?(?:\s+\d+/\d+|\s*[½¼¾⅓⅔⅛])?|\d+/\d+|[½¼¾⅓⅔⅛])"
_QTY = re.compile(
    rf"^({_NUM})(?:\s*(?:-|–|to)\s*{_NUM})?\s*"
    r"(g|grams?|kg|kilograms?|ml|millilit(?:er|re)s?|l|lit(?:er|re)s?|lbs?|pounds?"
    r"|oz|ounces?|cups?|cans?|cloves?|dozen|tbsps?|tablespoons?|tsps?|teaspoons?"
    r"|inch(?:es)?|thumbs?|pinch(?:es)?|bunch(?:es)?|sprigs?|slices?|pieces?|heads?"
    r"|stalks?|handfuls?)?\.?(?=\s|$)\s*",
    re.IGNORECASE)
_FRACTIONS = {"½": 0.5, "¼": 0.25, "¾": 0.75, "⅓": 1 / 3, "⅔": 2 / 3, "⅛": 0.125}
_FORMS = {"canned", "frozen", "dried", "ground", "smoked", "pickled", "powdered"}
# What the COOK does, kept out of the name as the live parser keeps it out
# ("1 cup chopped cilantro" buys cilantro). Purchase descriptors — boneless,
# skinless, whole, unsalted, smoked, low-sodium — stay in the name, as the
# live parser keeps them: they decide which product to buy. "crushed" is not
# here: Crushed Tomatoes and Crushed Red Pepper Flakes are products.
_PREP = {"chopped", "sliced", "minced", "diced", "shredded", "grated", "mashed",
         "peeled", "cubed", "julienned", "finely", "thinly", "roughly", "freshly",
         "fine"}
# "5 garlic cloves", "2 thyme sprigs": the count word comes after the name
_TRAILING_COUNT = {"clove", "cloves", "sprig", "sprigs", "stalk", "stalks"}
_TAGS = ("dairy", "gluten", "meat", "nuts", "egg", "soy")


def _amount(text: str) -> float:
    """"1 1/2" -> 1.5, "1½" -> 1.5, "3/4" -> 0.75, "2" -> 2.0."""
    total = 0.0
    for part in re.findall(r"\d+/\d+|\d+(?:\.\d+)?|[½¼¾⅓⅔⅛]", text):
        if part in _FRACTIONS:
            total += _FRACTIONS[part]
        elif "/" in part:
            num, den = part.split("/")
            total += int(num) / int(den) if int(den) else 0.0
        else:
            total += float(part)
    return round(total, 4)


def _strip_notes(item: str) -> tuple[str, str]:
    """(item without parenthetical notes, the text after its first comma):
    "garlic cloves (, thinly sliced)" -> ("garlic cloves", "");
    "onion, finely chopped" -> ("onion", "finely chopped")."""
    prev = None
    while prev != item:                          # innermost groups first
        prev, item = item, re.sub(r"\([^()]*\)", " ", item)
    item = item.replace("(", " ").replace(")", " ")
    head, _, rest = item.partition(",")
    return " ".join(head.split()), " ".join(rest.split())


def parse_recipe(text_input: str, *, model: str | None = None) -> ParsedInput:
    """Regex recipe parse: title, servings, '- qty unit name' ingredient
    lines, and budget / distance / dietary constraints from a Notes line.
    Same failure contract as the real parser: empty parse, never raises.

    Consistent with the live parser on what to buy: parenthetical notes and
    the text after a comma are dropped, prep words go to `prep`, a leading
    purchase form goes to `form`, and purchase descriptors stay in the name."""
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
        raw = ln.lstrip("-*• ").strip()
        item, after_comma = _strip_notes(raw)
        qty = unit = None
        if qm := _QTY.match(item):
            qty = _amount(qm.group(1))
            unit = (qm.group(2) or "each").lower().rstrip(".")
            item = item[qm.end():]
            if not item.strip() and qm.group(2):     # "2 cloves": the unit IS the item
                item, unit = qm.group(2), "each"
        words = [w for w in item.lower().split() if w != "of"]
        prep = [w for w in words if w in _PREP] + ([after_comma] if after_comma else [])
        words = [w for w in words if w not in _PREP]
        if unit == "each" and len(words) > 1 and words[-1] in _TRAILING_COUNT:
            unit = words.pop()
        form = None
        if words and words[0] in _FORMS:
            form = words.pop(0)
        if unit and unit.startswith("can"):
            form, unit = "canned", "can"
        name = " ".join(words).strip() or item.strip() or raw
        ingredients.append(IngredientSpec(name=name, form=form, quantity=qty, unit=unit,
                                          prep=" ".join(prep) or None))

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


def _preference_rank(origin, preference: list[str]) -> int:
    from .origins import preference_rank
    return preference_rank(origin, preference)


def select_products(ingredients: list[RecipeIngredient],
                    products: list[Product], *, model: str,
                    enable_thinking: bool = False,
                    constraints: dict | None = None,
                    origins_by_id: dict | None = None,
                    preference: list[str] | None = None) -> SelectorResult:
    """Token-overlap, then "fresh" when the ingredient says it, then head
    noun, then origin preference, then offer price; direct matches before
    t4 substitutes. Confidence is fixed at 0.9
    — above the cascade threshold, so demo mode never triggers a (would-be)
    escalation.

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
        toks = set(tokens(ing.name))
        want = tokens(ing.name)[-1] if tokens(ing.name) else None
        # "fresh" is a tokenizer stopword (half the catalog says it), yet
        # "fresh coriander" is cilantro, not ground coriander: read it raw.
        fresh = "fresh" in ing.name.lower().split()
        pool = [p for p in products if not p.substitute] or products
        if not pool:
            continue
        pick = min(pool, key=lambda p: (
            -len(toks & set(tokens(index_text(p.name, p.description)))),
            0 if not fresh or "fresh" in f"{p.name} {p.description}".lower() else 1,
            0 if want is not None and head_noun(p.name) == want else 1,
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

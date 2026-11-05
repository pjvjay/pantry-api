"""TRANSLATION: multi-shot constrained semantic parse of a pasted recipe.

One LLM call with a forced tool (llm.forced_tool_call, same idiom as
selector.py; NL2SQL_MODEL picks the provider) returns the two-part parse:
RecipeSpec (what to cook) + Constraints (how to shop).
The model never writes SQL — it parameterizes a fixed pattern menu, and
validate_parsed() clamps every value against the live DB vocabulary.

Failure mode: any error returns an empty parse with `error` set — the
caller decides how to degrade. Never raises.
"""
from __future__ import annotations

import time

from ..config import estimate_cost_usd, settings
from ..llm import forced_tool_call
from .schemas import ParsedInput

ALLOWED_TAGS = {"dairy", "gluten", "meat", "nuts", "egg", "soy"}
MAX_INGREDIENTS = 40
ALLOWED_FORMS = {"canned", "frozen", "dried", "ground", "smoked", "pickled", "powdered"}

PARSER_SYSTEM = """\
You parse a pasted recipe (plus any inline shopping notes) for a grocery
planner. Extract TWO things via the submit_parse tool:

1. recipe — title, servings, and the ingredient list. Normalize each
   ingredient:
   * name: the PRODUCT to buy — no quantities, no units, no prep words.
     KEEP every word the recipe wrote that changes WHICH product is bought:
     boneless, skinless, bone-in, whole, unsalted, salted, smoked,
     low-sodium, extra-virgin, light/dark, toasted, and the variety
     (Sichuan, basmati, Kashmiri). "1 lb boneless skinless chicken thighs"
     -> name "boneless skinless chicken thigh", NOT "chicken thigh": a
     bone-in thigh is a different product. Never add a word the recipe did
     not write. When a form IS the product, keep the compound name
     ("tomato sauce", "ground beef").
   * form: a PURCHASE form only if the shelf product differs:
     canned/frozen/dried/ground/smoked/pickled/powdered.
     "canned tomatoes" -> name "tomato", form "canned";
     "ground Sichuan peppercorns" -> name "Sichuan peppercorn", form
     "ground". Never drop a written form: ground and whole peppercorns are
     different products.
   * prep: what the COOK does (mashed/diced/shredded/minced/sliced/chopped)
     — record it, but it must NOT stay in name. "mashed potatoes" -> name
     "potato", prep "mashed" (you buy fresh potatoes to mash).
   * quantity + unit exactly as written ("400g" -> 400, "g").
   * category_hint: your best guess from the known vocabulary below.
   * water and ice: list them like any other line (name "water"); the
     planner knows they are never bought.

2. constraints — budget/dietary/scope filters from the notes or title:
   * "no dairy" (dietary) -> exclude_tags ["dairy"]  (strictest reading)
   * "no cheese" (a product kind) -> exclude_subcategories ["cheese"]
   * "skip the dairy aisle" -> exclude_categories ["dairy"]
   * budgets: "under $30" -> max_total_budget 30
   * travel limits: "only stores within 10km" -> max_distance_km 10
   * anything left that expresses preference -> soft_text

Known vocabulary:
{vocab}
Allowed dietary tags: dairy, gluten, meat, nuts, egg, soy.

EXAMPLES (input -> key outputs):

A) "Spaghetti Bolognese (serves 4)\\n- 400g spaghetti\\n- 500g ground beef
   \\n- 1 yellow onion\\n- 2 cloves garlic\\n- 1 can crushed tomatoes
   \\n- olive oil\\nNotes: under $30 please, no pork, only stores within 10km."
   -> title "Spaghetti Bolognese", servings 4; ingredients include
      {{name "spaghetti", quantity 400, unit "g", category_hint "pasta"}},
      {{name "ground beef", quantity 500, unit "g", category_hint "beef"}},
      {{name "tomato", form "canned", quantity 1, unit "can", category_hint "canned"}},
      {{name "olive oil", category_hint "oil"}};
      constraints {{max_total_budget 30, max_distance_km 10, soft_text "no pork"}}

B) "Shepherd's pie for 6: 2 lb ground lamb, mashed potatoes (about 1kg),
   frozen peas, 2 cups shredded cheddar. We're gluten-free."
   -> ingredients include {{name "ground lamb", quantity 2, unit "lb"}},
      {{name "potato", prep "mashed", quantity 1, unit "kg"}},
      {{name "pea", form "frozen"}},
      {{name "cheddar", prep "shredded", quantity 2, unit "cups",
        category_hint "cheese"}};
      constraints {{exclude_tags ["gluten"]}}

C) "quick veggie stir fry — broccoli, bell peppers, soy sauce, rice.
   dairy free and keep it cheap"
   -> servings null (unstated); ingredients broccoli/bell pepper/soy sauce/rice;
      constraints {{exclude_tags ["dairy"], soft_text "keep it cheap"}}

D) "Tomato soup: 700ml passata (ground tomatoes), 1 onion, cream 250ml.
   No canned stuff."
   -> {{name "passata", quantity 700, unit "ml", category_hint "sauce"}}
      (ground tomatoes = passata, a strained product — NOT form "ground");
      {{name "cream", quantity 250, unit "ml", category_hint "milk"}};
      constraints {{exclude_subcategories ["canned"]}}

E) "Baking day! 2.5kg all-purpose flour, dozen eggs, 454g butter,
   dark chocolate 200g. Nut-free household, max $5 per item."
   -> {{name "all-purpose flour", quantity 2.5, unit "kg"}},
      {{name "egg", quantity 1, unit "dozen"}}, ...;
      constraints {{exclude_tags ["nuts"], max_item_price 5}}

F) "Grilled cheese x2 — bread, cheddar, butter" (no notes)
   -> three ingredients, no constraints at all (all fields null/empty).

G) "Mala chicken (serves 4)\\n- 1 lb boneless skinless chicken thighs, cubed
   \\n- 1 tsp ground Sichuan peppercorns\\n- 2 tsp Sichuan peppercorns
   \\n- 3/4 cup water"
   -> {{name "boneless skinless chicken thigh", prep "cubed", quantity 1,
        unit "lb", category_hint "poultry"}},
      {{name "Sichuan peppercorn", form "ground", quantity 1, unit "tsp"}},
      {{name "Sichuan peppercorn", quantity 2, unit "tsp"}},
      {{name "water", quantity 0.75, unit "cup"}}
"""

_ING_SCHEMA = {
    "type": "object",
    "required": ["name"],
    "properties": {
        "name": {"type": "string"},
        "form": {"type": ["string", "null"]},
        "prep": {"type": ["string", "null"]},
        "quantity": {"type": ["number", "null"]},
        "unit": {"type": ["string", "null"]},
        "category_hint": {"type": ["string", "null"]},
    },
}

PARSER_TOOL = {
    "name": "submit_parse",
    "description": "Submit the parsed recipe and shopping constraints.",
    "input_schema": {
        "type": "object",
        "required": ["recipe", "constraints"],
        "properties": {
            "recipe": {
                "type": "object",
                "required": ["title", "servings", "ingredients"],
                "properties": {
                    "title": {"type": "string"},
                    # null when the recipe does not say: never a guessed 1
                    "servings": {"type": ["integer", "null"]},
                    "ingredients": {"type": "array", "items": _ING_SCHEMA},
                },
            },
            "constraints": {
                "type": "object",
                "properties": {
                    "max_item_price": {"type": ["number", "null"]},
                    "max_total_budget": {"type": ["number", "null"]},
                    "max_distance_km": {"type": ["number", "null"]},
                    "exclude_tags": {"type": "array", "items": {"type": "string"}},
                    "require_tags": {"type": "array", "items": {"type": "string"}},
                    "categories": {"type": "array", "items": {"type": "string"}},
                    "exclude_categories": {"type": "array", "items": {"type": "string"}},
                    "subcategories": {"type": "array", "items": {"type": "string"}},
                    "exclude_subcategories": {"type": "array", "items": {"type": "string"}},
                    "soft_text": {"type": "string"},
                },
            },
        },
    },
}


def parse_input(text_input: str, *, model: str | None = None) -> ParsedInput:
    """One extractor call. Never raises — see module docstring."""
    from .vocab import db_vocab, prompt_block

    cfg = settings()
    try:
        t0 = time.perf_counter()
        reply = forced_tool_call(
            model=model or cfg.nl2sql_model,
            max_tokens=2048,
            temperature=0.0,   # structured extraction: same input, same parse
            system=PARSER_SYSTEM.format(vocab=prompt_block(db_vocab())),
            tool=PARSER_TOOL,
            messages=[{"role": "user", "content": text_input}],
        )
        latency_ms = int((time.perf_counter() - t0) * 1000)
        parsed = ParsedInput.model_validate(reply.input)
        parsed.cost_usd = estimate_cost_usd(
            reply.model, reply.input_tokens, reply.output_tokens)
        parsed.latency_ms = latency_ms
        parsed.http = reply.http
        return parsed
    except Exception as e:  # noqa: BLE001 — housing failure mode: degrade, don't raise
        return ParsedInput(error=str(e)[:200])


def validate_parsed(parsed: ParsedInput, vocab: dict[str, str]) -> ParsedInput:
    """POST-PROCESSING: deterministic clamp against the live vocabulary.

    Every scope value resolves to its true level regardless of which list the
    model put it in; unknown values are dropped and surfaced in `ignored`.
    """
    c = parsed.constraints

    def clamp_tags(values: list[str]) -> list[str]:
        kept = []
        for v in (t.strip().lower() for t in values):
            (kept.append(v) if v in ALLOWED_TAGS else parsed.ignored.append(v))
        return kept

    def resolve_levels(*lists: list[str]) -> tuple[list[str], list[str]]:
        cats, subs = [], []
        for v in (x.strip().lower() for lst in lists for x in lst):
            level = vocab.get(v)
            if level == "category":
                cats.append(v)
            elif level == "subcategory":
                subs.append(v)
            else:
                parsed.ignored.append(v)
        return cats, subs

    c.exclude_tags = clamp_tags(c.exclude_tags)
    c.require_tags = clamp_tags(c.require_tags)
    c.categories, c.subcategories = resolve_levels(c.categories, c.subcategories)
    c.exclude_categories, c.exclude_subcategories = resolve_levels(
        c.exclude_categories, c.exclude_subcategories)

    for bound, upper in (("max_item_price", 10_000), ("max_total_budget", 10_000),
                         ("max_distance_km", 500)):
        v = getattr(c, bound)
        if v is not None and not (0 < v < upper):
            setattr(c, bound, None)
            parsed.ignored.append(f"{bound}={v}")

    for ing in parsed.recipe.ingredients:
        if ing.form:
            f = ing.form.strip().lower()
            if f in ALLOWED_FORMS:
                ing.form = f
            else:                              # unknown form -> extra name token
                ing.name, ing.form = f"{f} {ing.name}", None
        if ing.category_hint and ing.category_hint.strip().lower() not in vocab:
            ing.category_hint = None
    if parsed.recipe.servings is None:
        parsed.recipe.servings_stated = False
    parsed.recipe.servings = max(1, parsed.recipe.servings or 1)
    # Public-endpoint bound: cap the plan at 40 ingredients (keeps the
    # VALUES rowsets well under SQLite's parameter limit). Surfaced, not
    # silent — the dropped count lands in the interpretation chips and the
    # dropped ingredients, by name, on the plan's `skipped` list.
    if len(parsed.recipe.ingredients) > MAX_INGREDIENTS:
        parsed.over_cap = parsed.recipe.ingredients[MAX_INGREDIENTS:]
        parsed.recipe.ingredients = parsed.recipe.ingredients[:MAX_INGREDIENTS]
        parsed.ignored.append(f"{len(parsed.over_cap)} ingredients over the "
                              f"{MAX_INGREDIENTS}-ingredient cap")
    return parsed

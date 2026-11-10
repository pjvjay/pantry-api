"""What each placed meal needs: the recipe's line amounts scaled to the meal's servings.

A meal serves its own `servings`, or the household's. A line's need is its amount in
canonical units (g, ml, each) times meal servings / recipe servings, kept as a Fraction so
three meals of a recipe for 4 cooked for 2 add up to exactly one and a half batches. A need
is unknown, never guessed, when the line states no measurable amount (amount_unknown) or the
recipe does not say how many it serves and the shopper has not answered (needs_servings).

The amounts come from the recipe itself (the library's amounts, the starter file, the doc in
the draft); only the product per line comes from the draft's resolve result, overridden by
the shopper's pins, and every product id is checked against the catalog.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from ..models import Product, RecipeDoc
from ..nlsearch.units import normalize_quantity
from .models import Meal, MealPlanDraft, MealPlanError

USABLE = {"ok", "needs_servings"}


@dataclass(frozen=True)
class Need:
    """One line of one placed meal, bought as one product."""
    meal_id: str
    recipe_key: str
    title: str
    day: int
    slot: str
    line_no: int
    ingredient: str
    product_id: int
    qty: Fraction | None
    uom: str | None
    unknown: str | None          # None, 'amount_unknown' or 'needs_servings'

    @property
    def as_pack_need(self) -> tuple[float, str] | None:
        return None if self.qty is None else (float(self.qty), self.uom)


def line_products(draft: MealPlanDraft, key: str, doc: RecipeDoc,
                  products: dict[int, Product]) -> dict[int, int]:
    """line_no -> product_id for one recipe: the resolve result's choice, replaced by the
    shopper's pin. A pin on a line the recipe does not have, or naming a product the catalog
    does not have, is 422 pin_invalid; a resolved product the catalog no longer has is 422
    stale_product."""
    out: dict[int, int] = {}
    resolved = draft.resolved.get(key)
    if resolved is not None and resolved.status in USABLE:
        for ln in resolved.lines:
            if ln.product_id is None:
                continue
            if ln.product_id not in products:
                raise MealPlanError("stale_product",
                                    f"product {ln.product_id} (for {key} line {ln.line_no}) "
                                    "is no longer in the catalog; resolve the recipe again",
                                    product_id=ln.product_id, recipe_key=key)
            out[ln.line_no] = ln.product_id
    line_nos = {ln.line_no for ln in doc.lines}
    for raw, pid in sorted((draft.pins.get(key) or {}).items()):
        if not raw.isdigit() or int(raw) not in line_nos or pid not in products:
            raise MealPlanError("pin_invalid", f"pin {key} line {raw} -> product {pid} names "
                                "a line or product that does not exist", recipe_key=key)
        out[int(raw)] = pid
    return out


def recipe_servings(draft: MealPlanDraft, key: str, doc: RecipeDoc) -> int | None:
    return doc.servings or draft.recipes[key].servings or draft.recipes[key].ref.servings


def meal_servings(draft: MealPlanDraft, meal: Meal) -> int:
    return meal.servings or draft.prefs.household_servings


def needs_for_meal(draft: MealPlanDraft, meal: Meal, day: int, slot: str, doc: RecipeDoc,
                   chosen: dict[int, int]) -> list[Need]:
    yields = recipe_servings(draft, meal.recipe_key, doc)
    scale = None if yields is None else Fraction(meal_servings(draft, meal), yields)
    out = []
    for ln in doc.lines:
        pid = chosen.get(ln.line_no)
        if pid is None:
            continue
        canon = normalize_quantity(ln.quantity, ln.unit)
        if scale is None:
            qty, uom, unknown = None, None, "needs_servings"
        elif canon is None:
            qty, uom, unknown = None, None, "amount_unknown"
        else:
            qty, uom, unknown = Fraction(repr(canon[0])) * scale, canon[1], None
        out.append(Need(meal_id=meal.id, recipe_key=meal.recipe_key, title=doc.title,
                        day=day, slot=slot, line_no=ln.line_no, ingredient=ln.name,
                        product_id=pid, qty=qty, uom=uom, unknown=unknown))
    return out


def fmt_qty(qty: float | Fraction | None, uom: str | None) -> str:
    """'800 g', '1.5 l' is never written: grams and millilitres as given, counts bare."""
    if qty is None:
        return "amount unknown"
    v = round(float(qty), 1)
    text = f"{v:g}"
    return text if uom == "each" else f"{text} {uom}"

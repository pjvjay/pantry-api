"""RecipeDoc in, planning spec or display text out.

to_spec is how a reviewed recipe reaches the planner: each line's name,
quantity and unit go in exactly as the shopper confirmed them, with no parse,
so what the shopper reviewed is what gets planned (flow.run_spec). The
selector still picks the products; only the reading of the recipe is fixed.

to_recipe_text renders a doc in the pasted-recipe text format for the run
trace, the recipe-link skill's step 2 and the chat transcript. It is never a
planning input: a model reading it back could change an amount.
"""
from __future__ import annotations

from .models import RecipeDoc
from .nlsearch.schemas import IngredientSpec, RecipeSpec

# The units that are a can: the demo parser reads "1 can tomatoes" as a
# canned purchase in the same way (nlsearch.lineparse), so a reviewed line
# with unit "can" asks the catalog for the canned product too.
_CAN_UNITS = {"can", "cans"}


class UnconfirmedLines(ValueError):  # noqa: N818 — named like PlanAborted, UnparseableRecipe
    """A line the shopper has not confirmed yet (a video transcription not
    ticked). Planning it would plan something nobody reviewed: REST 422
    unconfirmed_lines, naming the lines."""

    def __init__(self, line_nos: list[int]):
        self.line_nos = line_nos
        super().__init__("lines not confirmed yet: " + ", ".join(map(str, line_nos))
                         + "; confirm or remove them before planning")


def to_spec(doc: RecipeDoc) -> RecipeSpec:
    """The doc as the planner's RecipeSpec, line for line. Raises
    UnconfirmedLines when any line is unconfirmed.

    servings is the doc's, or 1 to plan one batch when it is not stated; then
    servings_stated is False and the plan reports servings None."""
    unconfirmed = [ln.line_no for ln in doc.lines if not ln.confirmed]
    if unconfirmed:
        raise UnconfirmedLines(unconfirmed)
    return RecipeSpec(
        title=doc.title,
        servings=doc.servings or 1,
        servings_stated=doc.servings is not None,
        ingredients=[
            IngredientSpec(name=ln.name, quantity=ln.quantity, unit=ln.unit,
                           form="canned" if ln.unit.strip().lower() in _CAN_UNITS else None,
                           prep=ln.note or None)
            for ln in doc.lines],
    )


def _amount(quantity: float | None, unit: str) -> str:
    if quantity is None:
        return ""
    return f"{quantity:g} {unit} " if unit else f"{quantity:g} "


def to_recipe_text(doc: RecipeDoc) -> str:
    """'Title (serves N)' then one '- 400 g spaghetti, note' line per ingredient:
    for people and the trace, never for planning."""
    head = f"{doc.title} (serves {doc.servings})" if doc.servings else doc.title
    lines = [f"- {_amount(ln.quantity, ln.unit)}{ln.name}" + (f", {ln.note}" if ln.note else "")
             for ln in doc.lines]
    return "\n".join([head, *lines]) + "\n"

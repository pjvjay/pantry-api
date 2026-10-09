"""The shopper's own product for a recipe line, checked as the chat cart checks a swap.

A meal plan's recipe is resolved once (resolve.py): the planner picks a product per line,
under the plan's shopping point and origin rules. The browser keeps the picks (the draft's
`resolved`) and the shopper's pins (`pins[recipe_key][line_no]`); the server trusts neither
and re-reads every fact. A pin is checked by alternatives.validate_pins, the one check the
chat cart's swap goes through: the line is one the plan bought a product for, the product is
one of its candidates, the plan's origin exclusion does not hold it back, and a store in
range sells it.

validate_pins reads a PlanBasis: what the recipe was planned from. meal_basis rebuilds it on
the server from the recipe itself (resolve.recipe_doc) exactly as resolve planned it (a
library recipe by its ingredient names, as flow.run plans it; every RecipeDoc by its reviewed
lines, as flow.run_spec plans it), with the products the resolve chose and the plan's own
shopping point and origin rules from the draft's settings. Nothing in it comes from a basis
the browser sends.
"""
from __future__ import annotations

from ..models import BasisLine, Pin, PlanBasis, Product, RecipeDoc
from ..nlsearch.schemas import Constraints
from ..recipe_doc import UnconfirmedLines, to_spec
from .models import MealPlanDraft, MealPlanError
from .needs import USABLE


def location(draft: MealPlanDraft) -> tuple[float, float, float | None]:
    """Where the plan shops: the draft's point (or the server's), and its distance limit."""
    from .schedule import _location

    return _location(draft)


def meal_basis(draft: MealPlanDraft, key: str, doc: RecipeDoc,
               picks: dict[int, int] | None = None) -> PlanBasis:
    """The PlanBasis the recipe `key` was resolved from: its lines as the planner planned
    them, the product chosen for each (`picks`, line_no -> product_id: the resolve's choice
    unless the caller passes the plan's effective picks), the plan's shopping point and its
    origin rules. Lines the resolve bought nothing for carry no product, so they are not
    planned lines."""
    ref = draft.recipes[key].ref
    resolved = draft.resolved.get(key)
    chosen = ({ln.line_no: ln.product_id for ln in resolved.lines if ln.product_id is not None}
              if resolved is not None else {})
    if picks is not None:
        chosen = picks
    levels = {ln.line_no: ln.match for ln in resolved.lines} if resolved is not None else {}
    if ref.slug is not None:
        # flow.run plans a library recipe by its ingredient names, with no amounts.
        lines = [BasisLine(line_no=ln.line_no, name=ln.name, level=levels.get(ln.line_no)
                           or "exact", product_id=chosen.get(ln.line_no))
                 for ln in doc.lines]
        path = "library"
    else:
        # flow.run_spec plans the reviewed lines as recipe_doc.to_spec gives them.
        try:
            specs = to_spec(doc).ingredients
        except UnconfirmedLines as e:
            # A recipe with unconfirmed lines is never resolved; this one changed since.
            raise MealPlanError("stale_product", f"{doc.title} has lines that are not "
                                "confirmed; check its products again", recipe_key=key) from e
        lines = [BasisLine(line_no=ln.line_no, name=sp.name, form=sp.form, prep=sp.prep,
                           quantity=sp.quantity, unit=sp.unit,
                           level=levels.get(ln.line_no) or "exact",
                           product_id=chosen.get(ln.line_no))
                 for ln, sp in zip(doc.lines, specs, strict=True)]
        path = "spec"
    lat, lon, max_km = location(draft)
    s = draft.settings
    return PlanBasis(path=path, recipe_slug=key, recipe_name=doc.title, lines=lines,
                     constraints=Constraints(), lat=lat, lon=lon, max_km=max_km,
                     exclude_origin=list(s.exclude_origin), preference=list(s.preference),
                     origin_requested=bool(s.exclude_origin or s.preference),
                     ingredient_count=len(doc.lines), servings=doc.servings)


def check_pins(draft: MealPlanDraft, docs: dict[str, RecipeDoc],
               products: dict[int, Product]) -> None:
    """Every pin of every resolved recipe through alternatives.validate_pins; the first that
    fails is 422 pin_invalid, naming the recipe, the line and the reason. A recipe not
    resolved yet (or whose resolve failed) buys nothing, so its pins wait until it is."""
    from .. import alternatives

    for key, lines in sorted(draft.pins.items()):
        resolved = draft.resolved.get(key)
        if not lines or key not in docs or resolved is None or resolved.status not in USABLE:
            continue
        basis = meal_basis(draft, key, docs[key])
        pins = [Pin(line_no=int(raw), product_id=pid) for raw, pid in sorted(lines.items())]
        try:
            alternatives.validate_pins(basis, pins, catalog=products)
        except alternatives.PinError as e:
            raise MealPlanError("pin_invalid", f"{docs[key].title}: {e}", recipe_key=key,
                                line_no=e.line_no) from e

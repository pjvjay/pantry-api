"""Nutrition per portion eaten, computed by code from a recipe's amounts. No LLM.

A line's grams come from the recipe (quantity and unit), never from the packs bought, and its
nutrients from one reference food per ingredient: published values per 100 g for a generic
food (pantry-db 0008), never a product's label. Then

    value = per_100g x grams / 100 x portions / servings

for each nutrient the reference food published. The rules that keep the numbers honest:

- Unknown is never zero. A line with no amount, no weight, or no reference food is not
  counted and is named; a nutrient the reference did not publish is absent, not 0. A total
  that misses anything is a lower bound (status "at_least"), and one with nothing counted
  has amount None ("unknown").
- No density is assumed, not even water's. A millilitre or count line becomes grams only
  through a measure the source publishes for that food (pantry-db 0009), quoted in the
  line's receipt. A "can" has no stated size, so it is not converted either.
- Coverage is reported by line count and by mass against NUTRITION_MIN_COVERAGE; below it a
  meal is "below_floor" and its totals are minimums.
- amounts_basis says where the counted amounts came from. demo_amounts is True when any of
  them are demo house amounts (the library and the demo starters), and every surface that
  shows a number shows the "demo amounts" badge.

The reference is looked up by ingredient key: units.tokens() of the line's name joined by
spaces ("chicken thigh"), then the generic key (descriptors dropped, only when one was), then
the head noun. The first key with a map row wins, so "ground beef" never falls back to a
bare "beef" when it has its own row. A shorter key is tried only when every word it leaves
out names a cut, size or preparation (SAME_FOOD_WORDS): a row is reviewed for one food, and
"peanut butter", "coconut water" or "garlic salt" are other foods than its "butter",
"water" or "salt". Such a line has no reference and is named, so its meal's totals are
minimums. A row with match_kind 'none' (reviewed, nothing fits) stops the lookup, and is
reported differently from a key nobody reviewed.
"""
from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable

from .config import settings
from .db import NUTRIENT_KEYS, NutrientMeasure, Reference
from .models import (
    DayMeal,
    DayNutrition,
    MealNutrition,
    MissingLine,
    NutrientTotal,
    NutritionCoverage,
    NutritionLine,
    NutritionSource,
    NutritionTarget,
    PeriodNutrition,
    PeriodTotal,
    Recipe,
    RecipeDoc,
    RecipeLine,
    RecipeNutrition,
    TargetCheck,
)
from .nlsearch.units import generic_tokens, head_noun, is_non_purchase, normalize_quantity, tokens

NUTRIENTS: tuple[str, ...] = NUTRIENT_KEYS
UNITS = {"energy_kcal": "kcal", "protein_g": "g", "fat_g": "g", "satfat_g": "g",
         "carbohydrate_g": "g", "fibre_g": "g", "sugars_g": "g", "sodium_mg": "mg"}
HEADLINE = ("energy_kcal", "protein_g")

NOT_DEPLOYED = "nutrition not shown: nutrition tables not deployed (pantry-db 0008)"
DINNER_ONLY = "dinner only: other meals are not planned"
DEMO_AMOUNTS = "demo_house_amounts"
DEMO_BADGE_TITLE = "Recipe amounts are demo house amounts, not from a published recipe"
DISCLAIMER = ("Reference values for generic foods, not product labels; raw ingredients summed, "
              "cooking losses not counted. Not medical or dietary advice.")

# Units with a quantity but no weight a source could give: never converted.
_NO_AMOUNT_UNITS = {"pinch", "pinches", "dash", "dashes", "to taste", "handful", "handfuls"}
_CAN_UNITS = {"can", "cans"}
_SIZE_WORDS = {"pee", "wee", "small", "medium", "large", "extra", "jumbo"}


# ─── Keys and reference lookup ───────────────────────────────

# The words a shorter key may leave out of a name. Each names how the food is cut or
# prepared, or that it was grown organically or had its bone or skin taken off, and none
# changes what 100 g of the edible food holds. tokens() already drops "fresh", "large",
# "small" and "medium". Words that do change it (dried, smoked, lean, salted, unsalted,
# whole, light, frozen, toasted, low sodium, bone-in, and any word that names another food,
# such as the "peanut" of "peanut butter") are left out on purpose, so a name with one of
# them needs a map row of its own.
SAME_FOOD_WORDS = frozenset({
    "chopped", "sliced", "minced", "diced", "grated", "shredded", "cubed", "peeled",
    "trimmed", "organic", "boneless", "skinless",
})


def nutrition_key(name: str) -> str:
    """The ingredient key: 'Chicken Thighs' -> 'chicken thigh'."""
    return " ".join(tokens(name))


def _same_food(name: str, key: str) -> bool:
    """True when every word of the name that `key` leaves out is in SAME_FOOD_WORDS."""
    dropped = Counter(tokens(name)) - Counter(key.split())
    return all(w in SAME_FOOD_WORDS for w in dropped)


def candidate_keys(name: str) -> list[str]:
    """The keys tried for a name, in order: full, generic (only when a descriptor was
    dropped), head noun. A shorter key is kept only when the words it drops leave the food
    the same ('boneless skinless chicken thighs' -> 'chicken thigh', never 'peanut butter'
    -> 'butter'). Duplicates and empty keys are left out."""
    full = nutrition_key(name)
    out: list[str] = [full] if full else []
    for key in (" ".join(generic_tokens(name)), head_noun(name) or ""):
        if key and key not in out and _same_food(name, key):
            out.append(key)
    return out


def keys_for(names: Iterable[str]) -> set[str]:
    """Every key the lines might need, for db.load_reference."""
    return {k for n in names for k in candidate_keys(n)}


def lookup(name: str, ref: Reference):
    """(the key that matched, its map entry), or (the full key, None) when no key has a row."""
    for key in candidate_keys(name):
        if key in ref.map:
            return key, ref.map[key]
    return nutrition_key(name), None


# ─── Grams ───────────────────────────────────────────────────

def _g(x: float) -> str:
    return f"{x:.4g}" if x < 10 else f"{x:.1f}".rstrip("0").rstrip(".")


def _words(*texts: str) -> set[str]:
    return {w for t in texts for w in re.findall(r"[a-z]+", (t or "").lower())}


def _volume_measure(measures: list[NutrientMeasure], ml: float) -> NutrientMeasure | None:
    vols = [m for m in measures if m.volume_ml]
    if not vols:
        return None
    return min(vols, key=lambda m: (abs(m.volume_ml - ml), m.volume_ml, m.measure))


def _count_measure(measures: list[NutrientMeasure], line: RecipeLine
                   ) -> tuple[NutrientMeasure | None, str]:
    """The source's weight for one of this food, and '' or why none can be chosen. A size
    the line names ("large eggs") picks that size; with no size named, a single measure is
    used and several sizes are left unconverted rather than guessed."""
    each = [m for m in measures if m.volume_ml is None and m.verbatim.strip().startswith("1 ")]
    if not each:
        return None, "the source publishes no weight for one of this food"
    said = _words(line.name, line.note, line.text) & _SIZE_WORDS
    if said:
        hits = [m for m in each if _words(m.verbatim) & _SIZE_WORDS == said]
        if len(hits) == 1:
            return hits[0], ""
    elif len(each) == 1:
        return each[0], ""
    sizes = ", ".join(f"{m.verbatim} {m.grams:g} g" for m in each)
    return None, f"the source lists several sizes ({sizes}) and the recipe does not say which"


def grams_of(line: RecipeLine, ref: Reference | None, ref_id: str | None
             ) -> tuple[float | None, str, str]:
    """(grams in the recipe, status, conversion or reason). status is 'ok', 'no_quantity' or
    'no_conversion'. Weight units need no reference; a volume or count needs the reference
    food's own measure, so with ref_id None it reports 'needs_reference'."""
    if line.quantity is None:
        return None, "no_quantity", "the recipe states no amount"
    unit = line.unit.strip().lower()
    if unit in _NO_AMOUNT_UNITS:
        return None, "no_quantity", f"'{line.unit}' is not an amount"
    if unit in _CAN_UNITS:
        return None, "no_conversion", "a can's size is not stated, so its weight is unknown"
    canon = normalize_quantity(line.quantity, line.unit)
    if canon is None:
        why = "no unit" if not unit else f"unit '{line.unit}' has no weight"
        return None, "no_conversion", f"{why}, so the line's weight is unknown"
    qty, uom = canon
    if uom == "g":
        written = f"{_g(line.quantity)} {line.unit}"
        return qty, "ok", (f"{written} as written" if unit in {"g", "gram", "grams"}
                           else f"{written} = {_g(qty)} g")
    if ref_id is None:
        return None, "needs_reference", ""
    if not ref.measures_deployed:
        return None, "no_conversion", ("no source measures are deployed (pantry-db 0009), "
                                       "and no density is assumed")
    measures = ref.measures.get(ref_id, [])
    if uom == "ml":
        m = _volume_measure(measures, qty)
        if m is None:
            return None, "no_conversion", ("the source publishes no volume measure for this "
                                           "food, and no density is assumed")
        grams = qty * m.grams / m.volume_ml
        return grams, "ok", (f"{_g(qty)} ml at the source's measure '{m.verbatim}' = "
                             f"{m.grams:g} g ({m.grams / m.volume_ml:.4f} g/ml): "
                             f"{_g(grams)} g")
    m, why = _count_measure(measures, line)
    if m is None:
        return None, "no_conversion", why
    grams = qty * m.grams
    return grams, "ok", f"{_g(qty)} x the source's '{m.verbatim}' ({m.grams:g} g) = {_g(grams)} g"


# ─── One line ────────────────────────────────────────────────

def line_receipt(line: RecipeLine, ref: Reference, factor: float) -> NutritionLine:
    """A line's receipt. Status, in order: excluded (water or ice with no reviewed reference
    food), no_quantity, no_conversion (no weight that needs no reference), no_reference,
    no_conversion (no source measure for a volume or count), else counted."""
    key, entry = lookup(line.name, ref)
    base = {"line_no": line.line_no, "ingredient": line.name, "quantity": line.quantity,
            "unit": line.unit, "key": key, "amount_basis": line.amount_basis}
    if entry is not None:
        base.update(match_kind=entry.match_kind, match_note=entry.note)
    food = ref.foods.get(entry.ref_id) if entry is not None and entry.ref_id else None
    if food is not None:
        base.update(ref_id=food.ref_id, ref_description=food.description,
                    state_note=food.state_note, source=food.source,
                    absent=[n for n in NUTRIENTS if n not in food.per_100g])
    if entry is None and is_non_purchase(line.name):
        return NutritionLine(**base, grams=None, status="excluded",
                             reason="water and ice are not counted")
    grams, status, how = grams_of(line, ref, None if food is None else food.ref_id)
    if status in {"no_quantity", "no_conversion"}:
        return NutritionLine(**base, grams=grams, status=status, reason=how)
    if food is None:
        if entry is not None and entry.match_kind == "none":
            reason = "reviewed: no reference food fits" + (f" ({entry.note})" if entry.note else "")
        elif entry is not None:
            reason = f"its reference food {entry.ref_id} is not loaded"
        else:
            reason = f"no reference food has been chosen for '{key}'"
        return NutritionLine(**base, grams=grams, status="no_reference", reason=reason)
    values = {n: round(food.per_100g[n] * grams / 100 * factor, 3)
              for n in NUTRIENTS if n in food.per_100g}
    return NutritionLine(**base, grams=round(grams, 3), status="counted", conversion=how,
                         values=values)


# ─── One meal ────────────────────────────────────────────────

def _total(nutrient: str, amount: float | None, counted: int, total: int,
           gaps: list[str]) -> NutrientTotal:
    complete = total > 0 and counted == total
    return NutrientTotal(
        amount=None if amount is None else round(amount, 1), unit=UNITS[nutrient],
        status="complete" if complete else ("unknown" if amount is None else "at_least"),
        complete=complete, lines_counted=counted, lines_total=total, gaps=gaps)


def _pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def meal_nutrition(doc: RecipeDoc, ref: Reference, *, portions: float = 1,
                   servings: int | None = None, floor: float | None = None) -> MealNutrition:
    """Nutrition of `portions` servings of a recipe. servings defaults to the doc's; when
    neither says, the basis is the whole recipe (per_recipe) and portions do not apply."""
    servings = servings if servings is not None else doc.servings
    floor = settings().nutrition_min_coverage if floor is None else floor
    basis = "per_serving" if servings else "per_recipe"
    factor = portions / servings if servings else 1.0
    lines = [line_receipt(ln, ref, factor) for ln in doc.lines]

    in_scope = [ln for ln in lines if ln.status != "excluded"]
    counted = [ln for ln in in_scope if ln.status == "counted"]
    totals = {}
    for n in NUTRIENTS:
        have = [ln for ln in counted if n in ln.values]
        amount = sum(ln.values[n] for ln in have) if have else None
        gaps = [ln.ingredient for ln in in_scope if n not in ln.values]
        totals[n] = _total(n, amount, len(have), len(in_scope), gaps)

    weighed = [ln for ln in in_scope if ln.grams is not None]
    grams_weighed = sum(ln.grams for ln in weighed)
    grams_known = sum(ln.grams for ln in counted)
    count_fraction = len(counted) / len(in_scope) if in_scope else None
    mass_fraction = grams_known / grams_weighed if grams_weighed > 0 else None
    meets = (count_fraction is not None and count_fraction >= floor
             and (mass_fraction is None or mass_fraction >= floor))
    missing = [MissingLine(line_no=ln.line_no, ingredient=ln.ingredient, reason=ln.status,
                           detail=ln.reason) for ln in in_scope if ln.status != "counted"]
    cov_note = (f"{len(counted)} of {len(in_scope)} ingredients counted"
                + (f", {_pct(mass_fraction)} by weight" if mass_fraction is not None else ""))
    if not meets:
        cov_note = (f"Nutrition known for only {len(counted)} of {len(in_scope)} ingredients "
                    "- totals are minimums")
    coverage = NutritionCoverage(
        lines_total=len(in_scope), lines_counted=len(counted),
        count_fraction=None if count_fraction is None else round(count_fraction, 4),
        grams_weighed=round(grams_weighed, 1), grams_known=round(grams_known, 1),
        mass_fraction=None if mass_fraction is None else round(mass_fraction, 4),
        lines_mass_unknown=len(in_scope) - len(weighed), floor=floor, meets_floor=meets,
        note=cov_note)

    status = ("below_floor" if not meets
              else "complete" if all(t.complete for t in totals.values()) else "incomplete")
    amounts_basis = sorted({ln.amount_basis for ln in counted if ln.amount_basis})
    demo = DEMO_AMOUNTS in amounts_basis
    head = ("For the whole recipe: it does not say how many it serves." if basis == "per_recipe"
            else f"Per serving (the recipe serves {servings})." if portions == 1
            else f"For {_g(portions)} servings (the recipe serves {servings}).")
    parts = [head, cov_note + "."]
    if missing:
        parts.append("Not counted: " + "; ".join(f"{m.ingredient} ({m.detail})" for m in missing)
                     + ".")
    if demo:
        parts.append(DEMO_BADGE_TITLE + ".")
    return MealNutrition(
        basis=basis, servings=servings, portions=portions, totals=totals, status=status,
        coverage=coverage, missing=missing, lines=lines,
        source_ids=sorted({ln.source for ln in counted if ln.source}),
        amounts_basis=amounts_basis, demo_amounts=demo, note=" ".join(parts))


# ─── Days and periods ────────────────────────────────────────

def day_meal(meal_id: str, recipe_key: str, title: str, slot: str,
             mn: MealNutrition) -> DayMeal:
    return DayMeal(meal_id=meal_id, recipe_key=recipe_key, title=title, slot=slot,
                   basis=mn.basis, status=mn.status, totals=mn.totals,
                   amounts_basis=mn.amounts_basis, demo_amounts=mn.demo_amounts)


def day_totals(meals: list[DayMeal], *, meals_counted: list[str], all_meals_planned: bool,
               date=None, note: str | None = None,
               targets: dict[str, NutritionTarget] | None = None) -> DayNutrition:
    """One person's day: each meal's per-serving values added up. A total is complete only
    when every counted slot is planned and every meal's total is complete. A meal of a recipe
    whose servings are unknown (per_recipe) adds nothing and leaves every total incomplete."""
    per_serving = [m for m in meals if m.basis == "per_serving"]
    unknown_servings = [m.title for m in meals if m.basis != "per_serving"]
    totals = {}
    for n in NUTRIENTS:
        amounts = [m.totals[n].amount for m in per_serving if m.totals[n].amount is not None]
        complete = (all_meals_planned and bool(meals) and not unknown_servings
                    and all(m.totals[n].complete for m in per_serving))
        gaps = [f"{m.title}: {g}" for m in per_serving for g in m.totals[n].gaps]
        gaps += [f"{t}: servings unknown" for t in unknown_servings]
        amount = round(sum(amounts), 1) if amounts else None
        totals[n] = NutrientTotal(
            amount=amount, unit=UNITS[n], complete=complete,
            status="complete" if complete else ("unknown" if amount is None else "at_least"),
            lines_counted=sum(m.totals[n].lines_counted for m in per_serving),
            lines_total=sum(m.totals[n].lines_total for m in per_serving), gaps=gaps)
    amounts_basis = sorted({b for m in meals for b in m.amounts_basis})
    complete = all(t.complete for t in totals.values())
    if note is None:
        note = ("every meal counted" if complete else
                "not every slot is planned" if not all_meals_planned else
                "some ingredients or servings are unknown, so totals are minimums")
    day = DayNutrition(date=date, meals=meals, meals_counted=meals_counted,
                       all_meals_planned=all_meals_planned, totals=totals, complete=complete,
                       amounts_basis=amounts_basis, demo_amounts=DEMO_AMOUNTS in amounts_basis,
                       note=note)
    if targets:
        day.targets = check_targets(totals, targets)
    return day


def period_nutrition(days: list[DayNutrition], slots_counted: list[str]) -> PeriodNutrition:
    """The plan's days together. A day counts as complete only when all its totals are."""
    done = [d for d in days if d.complete]
    average = None
    if done:
        average = {n: round(sum(d.totals[n].amount for d in done) / len(done), 1)
                   for n in NUTRIENTS}
    lower = {}
    for n in NUTRIENTS:
        amounts = [d.totals[n].amount for d in days if d.totals[n].amount is not None]
        lower[n] = PeriodTotal(amount=round(sum(amounts), 1) if amounts else None,
                               complete=bool(days) and all(d.totals[n].complete for d in days))
    amounts_basis = sorted({b for d in days for b in d.amounts_basis})
    note = (f"{len(done)} of {len(days)} days complete"
            + ("; averages are over complete days only" if done
               else "; no day is complete, so there is no daily average")
            + f"; slots counted: {', '.join(slots_counted)}.")
    return PeriodNutrition(
        days_total=len(days), days_complete=len(done),
        incomplete_days=[d.date for d in days if not d.complete and d.date is not None],
        per_day_average_over_complete_days=average, lower_bound_total=lower,
        slots_counted=list(slots_counted), amounts_basis=amounts_basis,
        demo_amounts=DEMO_AMOUNTS in amounts_basis, note=note)


# ─── Targets ─────────────────────────────────────────────────

def check_targets(totals: dict[str, NutrientTotal], targets: dict[str, NutritionTarget]
                  ) -> dict[str, TargetCheck]:
    """A verdict only where the data proves it. A lower bound above a max is 'over' and one
    reaching a min is 'met', complete or not; 'within' and 'short' need a complete total
    (a dinner-only day never has one). Anything else is 'unknown'."""
    out = {}
    for n, t in sorted(targets.items()):
        total = totals[n]
        amount, complete = total.amount, total.complete
        check = TargetCheck(amount=amount, complete=complete)
        if t.max is not None:
            check.max = ("over" if amount is not None and amount > t.max
                         else "within" if complete and amount is not None else "unknown")
        if t.min is not None:
            check.min = ("met" if amount is not None and amount >= t.min
                         else "short" if complete else "unknown")
        out[n] = check
    return out


# ─── Sources ─────────────────────────────────────────────────

def sources(ref: Reference, ids: Iterable[str] | None = None) -> list[NutritionSource]:
    """The sources behind these ids (all when None), with their licence and attribution."""
    wanted = None if ids is None else set(ids)
    return [NutritionSource(**s) for k, s in sorted(ref.sources.items())
            if wanted is None or k in wanted]


# ─── Library recipes ─────────────────────────────────────────

def library_reference(recipes: list[Recipe]) -> Reference | None:
    """The reference rows these library recipes need; None when the tables are missing."""
    from . import db

    return db.load_reference(keys_for(ing.name for r in recipes for ing in r.ingredients))


def recipe_nutrition(recipe: Recipe, amounts: dict | None, ref: Reference | None, *,
                     portions: float = 1, with_sources: bool = False) -> RecipeNutrition:
    """A library recipe's nutrition from its demo house amounts (`amounts` is its line_no ->
    LineAmount map, None when 0007 is missing). nutrition None with a note when the reference
    tables are missing."""
    from .recipe_doc import library_doc

    base = {"slug": recipe.slug, "name": recipe.name, "servings": recipe.servings}
    if ref is None:
        return RecipeNutrition(**base, nutrition=None, note=NOT_DEPLOYED)
    mn = meal_nutrition(library_doc(recipe, amounts), ref, portions=portions)
    return RecipeNutrition(**base, nutrition=mn, note=DISCLAIMER,
                           sources=sources(ref, mn.source_ids) if with_sources else [])

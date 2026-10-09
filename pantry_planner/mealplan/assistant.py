"""plan_meals: one or two weeks of meals drafted for the assistant, never approved by it.

An agent (the demo hub's Assistant, or any MCP client) names counted dishes, "3 Pepperoni
Pizza + 2 Chicken Fried Rice ... in 2 weeks", and gets back a draft for the shopper's Meal
plan: the meals spread over the days, the trips the schedule suggests with their totals, and
the ops a console applies to the plan it holds. Nothing is saved and nothing is approved: the
plan lives in the shopper's browser, and only the shopper approves a trip, in the Meal plan.

Count semantics are the Meal plan's: a count is that many meal occasions, each serving the
household (prefs.household_servings, default 2).

Which recipe a dish is, in order: a recipe key ('lib:<slug>', 'starter:<key>', or the key of
one of the shopper's docs sent in `my_recipe_docs`), a library slug, a demo starter key, then
its name, through Quick add's matcher (selection.match_name, called as it is). Only an exact
or plural match is placed. An alias or fuzzy match ("chicken briyani" for Chicken Biryani) is
returned as a proposal for the shopper to accept or refuse, never placed, and so is every dish
in `proposed`, which an app fills from its own parse of the shopper's words. A name that fits
several recipes, or none, is listed as unmatched with the recipes it could be. A dish with
`lines` is one the assistant writes: its lines are read with the paste parser (no LLM), their
amounts labelled written_by_assistant.

The draft is the browser's MealPlanDraft with the new meals placed where the schedule's even
spread put them (place.place); existing meals in `current` never move. Trips, totals, warnings
and nutrition come from schedule.compute on that draft, so they are what the Meal plan will
show once the shopper applies it.
"""
from __future__ import annotations

import datetime as dt
import hashlib
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from ..models import MAX_DOC_LINES, RecipeDoc, RecipeLine, RecipeSource
from .models import (
    MAX_DAYS,
    MAX_MEALS,
    MAX_RECIPES,
    SLOTS,
    DraftRecipe,
    Meal,
    MealPlanDraft,
    MealPlanError,
    MealSchedule,
    PlanSettings,
    Prefs,
    RecipeRef,
    Slot,
)

MAX_DISHES = 12
MAX_COUNT = 28
MAX_WARNINGS = 8
NEVER_APPROVES = ("Trips are suggestions: only the shopper approves them, in the Meal plan. "
                  "Prices and stock are demo data.")


class PlanMealsError(ValueError):
    """A request plan_meals cannot draft: the MCP tool's ToolError, REST 422."""


# ─── What the agent sends ────────────────────────────────────

class DishIn(BaseModel):
    """One dish and how many meals of it. `recipe` names it: a recipe key, a library slug, a
    demo starter key, or its title. With `lines` it is a dish you write: `recipe` is its title,
    each line one ingredient with its amount ("400 g chicken thighs"), and `servings` how many
    the recipe serves."""
    recipe: str = Field(min_length=1, max_length=200)
    count: int = Field(ge=1, le=MAX_COUNT)
    slot: Slot | None = None
    servings: int | None = Field(default=None, ge=1, le=20)
    lines: list[Annotated[str, Field(max_length=300)]] | None = Field(default=None,
                                                                      max_length=MAX_DOC_LINES)


class ProposedDish(BaseModel):
    """A dish the app read from the shopper's words but may not place without them: an alias
    or fuzzy match. `input` is what the shopper wrote."""
    recipe_key: str = Field(min_length=1, max_length=100)
    count: int = Field(ge=1, le=MAX_COUNT)
    slot: Slot | None = None
    input: str = Field(default="", max_length=300)
    how: Literal["alias", "fuzzy"] = "fuzzy"


class ContextRecipe(BaseModel):
    """A recipe in the shopper's current plan. kind: library, starter or doc (any other key)."""
    key: str = Field(min_length=1, max_length=100)
    title: str = Field(min_length=1, max_length=200)
    kind: str = Field(default="doc", max_length=20)
    slot: Slot = "dinner"


class ContextTrip(BaseModel):
    date: dt.date
    strategy: str | None = Field(default=None, max_length=20)


class MealPlanContext(BaseModel):
    """The shopper's current Meal plan, as compact as an app can send it: its window, its
    meals and recipes, the dates of approved trips (approvals stay the console's), and its
    household settings. A doc recipe's lines travel in `my_recipe_docs`."""
    start_date: dt.date
    days: int = Field(ge=1, le=MAX_DAYS)
    rev: int = Field(default=0, ge=0)
    meals: list[Meal] = Field(default_factory=list, max_length=MAX_MEALS)
    approved_trips: list[ContextTrip] = Field(default_factory=list, max_length=MAX_DAYS)
    recipes: list[ContextRecipe] = Field(default_factory=list, max_length=MAX_RECIPES)
    prefs: Prefs | None = None


# ─── What comes back ─────────────────────────────────────────

class PlannedMeal(BaseModel):
    """A meal of the drafted plan. new: drafted now (not in the shopper's plan yet)."""
    date: dt.date
    slot: Slot
    title: str
    recipe_key: str
    new: bool


class AddedDish(BaseModel):
    """A dish placed in the draft. how: key (named by its key or slug), exact or plural (by
    its title) or written (lines you wrote)."""
    recipe_key: str
    title: str
    label: str
    how: str
    count: int
    placed: int
    slot: Slot
    input: str = ""


class Proposal(BaseModel):
    """A dish waiting for the shopper's yes: `op` is what the console applies on "Use" (the
    meals then join the plan and are spread over free slots)."""
    input: str
    recipe_key: str
    title: str
    label: str
    how: str
    count: int
    slot: Slot
    question: str
    op: dict[str, Any]


class Unmatched(BaseModel):
    """A dish no recipe matched, or several did: never guessed. candidates: recipes it could
    be, for the shopper to choose."""
    input: str
    reason: str
    candidates: list[dict[str, str]] = Field(default_factory=list)


class TripBrief(BaseModel):
    """A suggested trip of the recommended strategy, priced from demo data. total_cost None:
    no line of it has a price."""
    date: dt.date
    stores: list[str]
    items: int
    total_cost: float | None
    total_is_floor: bool
    reason: str


class OtherStrategy(BaseModel):
    name: str
    trips: int
    total_cost: float | None
    total_is_floor: bool


class MealPlanSummary(BaseModel):
    """The drafted plan, lean enough for a model to read. `meals` lists every meal of the
    plan by date (new marks the drafted ones); `added` the dishes placed; `proposals` the
    dishes that wait for the shopper's yes; `unmatched` the names no single recipe fits;
    `unplaced` meals with no free slot. `trips` are the recommended strategy's suggested
    trips, `other_strategy` the alternative in brief. `nutrition` is one line to quote as
    it stands. `ops` are what a console applies, in order, as one undo step."""
    kind: Literal["mealplan"] = "mealplan"
    start_date: dt.date
    days: int
    base_rev: int | None
    household_servings: int
    meals: list[PlannedMeal]
    added: list[AddedDish]
    proposals: list[Proposal]
    unmatched: list[Unmatched]
    unplaced: list[dict[str, Any]]
    strategy: str
    trips: list[TripBrief]
    other_strategy: OtherStrategy | None
    total_cost: float | None
    total_is_floor: bool
    nutrition: str
    warnings: list[str]
    notes: list[str]
    ops: list[dict[str, Any]]


class MealPlanResult(BaseModel):
    summary: MealPlanSummary
    draft: MealPlanDraft               # the whole plan, for an app that holds none
    full: MealSchedule | None = None   # with verbose=true: the schedule behind the summary


# ─── Recipes the dishes can be ───────────────────────────────

def _candidates(docs: list[RecipeDoc]):
    from .. import db
    from . import selection
    from .resolve import starters_file

    return selection.candidates(
        [(r.slug, r.name) for r in db.load_all_recipes()], starters_file()["starters"],
        [{"key": d.key, "title": d.title} for d in docs])


def _ref_for(key: str, docs: dict[str, RecipeDoc]) -> RecipeRef | None:
    if key.startswith("lib:"):
        return RecipeRef(key=key, slug=key[4:])
    if key.startswith("starter:"):
        return RecipeRef(key=key, starter=key[len("starter:"):])
    doc = docs.get(key)
    return RecipeRef(key=key, doc=doc) if doc is not None else None


def _written_doc(dish: DishIn) -> RecipeDoc:
    """A dish the assistant wrote, as a RecipeDoc: its lines read with the paste parser, its
    amounts the assistant's. The key is a hash of what was written, so the same dish written
    again is the same recipe, and two different ones never share a key."""
    from ..models import MAX_LINE_NAME, MAX_LINE_QUANTITY
    from ..nlsearch import lineparse

    lines = []
    for text in (t.strip() for t in dish.lines or []):
        if not lineparse.without_bullet(text):
            continue
        p = lineparse.parse_line(text)
        qty, unit = p.quantity, p.unit or ""
        if qty is not None and qty > MAX_LINE_QUANTITY:
            qty, unit = None, ""
        lines.append(RecipeLine(line_no=len(lines) + 1, text=text,
                                name=p.doc_name[:MAX_LINE_NAME].rstrip(), quantity=qty,
                                unit=unit, note=p.prep or "",
                                amount_basis="written_by_assistant"))
    if not lines:
        raise PlanMealsError(f"{dish.recipe!r} has no ingredient lines to plan")
    digest = hashlib.sha256("\n".join([dish.recipe, str(dish.servings)]
                                      + [ln.text for ln in lines]).encode()).hexdigest()[:12]
    return RecipeDoc(key=f"asst:{digest}", title=dish.recipe[:200], servings=dish.servings,
                     servings_stated=dish.servings is not None,
                     servings_basis="source" if dish.servings else None, lines=lines,
                     source=RecipeSource(kind="assistant", method="agent_written",
                                         label="written by the assistant"))


def _by_key(name: str, by_key: dict):
    """The recipe a dish names by its key, library slug or starter key; None when the name
    is a title to match."""
    name = name.strip()
    return next((by_key[k] for k in (name, f"lib:{name}", f"starter:{name}") if k in by_key),
                None)


def _question(input_text: str, title: str, label: str) -> str:
    return f"{input_text} → {title} ({label})?"


# ─── The draft ───────────────────────────────────────────────

def _default_days(counts: dict[str, int]) -> int:
    """A week when every slot's meals fit in one, else two."""
    from .place import CAPACITY

    return 7 if all(n <= CAPACITY[s] * 7 for s, n in counts.items()) else MAX_DAYS


def plan_meals(dishes: list[DishIn] | None = None, proposed: list[ProposedDish] | None = None,
               *, days: int | None = None, start_date: dt.date | None = None,
               current: MealPlanContext | None = None,
               my_recipe_docs: list[RecipeDoc] | None = None,
               household_servings: int | None = None, lat: float | None = None,
               lon: float | None = None, max_km: float | None = None,
               verbose: bool = False) -> MealPlanResult:
    """The draft (see the module docstring). PlanMealsError for a request with nothing to
    plan, more recipes or meals than a plan holds, or a current plan whose own meals
    overfill a slot."""
    from . import schedule, selection
    from .resolve import resolve_one

    dishes, proposed = list(dishes or []), list(proposed or [])
    if not dishes and not proposed:
        raise PlanMealsError("No dishes: name each dish with how many meals of it, for example "
                             '[{"recipe": "Pepperoni Pizza", "count": 3}].')
    if len(dishes) > MAX_DISHES or len(proposed) > MAX_DISHES:
        raise PlanMealsError(f"At most {MAX_DISHES} dishes at a time.")
    docs = {d.key: d for d in my_recipe_docs or []}
    cands = _candidates(list(docs.values()))
    by_key = {c.key: c for c in cands}
    warnings: list[str] = []
    notes: list[str] = []

    # The shopper's plan as it is.
    ctx_recipes = {r.key: r for r in (current.recipes if current else [])}
    refs: dict[str, RecipeRef] = {}
    titles: dict[str, str] = {}
    labels: dict[str, str] = {}
    slots: dict[str, str] = {}
    for key, r in ctx_recipes.items():
        ref = _ref_for(key, docs)
        if ref is None:
            # Its lines were not sent: its meals keep their places and buy nothing here.
            ref = RecipeRef(key=key, doc=RecipeDoc(
                key=key, title=r.title, lines=[],
                source=RecipeSource(kind="pasted", method="paste")))
            warnings.append(f"{r.title}: its lines were not sent, so its meals keep their "
                            "places but buy nothing in this draft.")
        refs[key], titles[key], slots[key] = ref, r.title, r.slot
        labels[key] = by_key[key].label if key in by_key else "my recipe"
    meals: list[Meal] = []
    for m in current.meals if current else []:
        if m.recipe_key not in ctx_recipes:
            warnings.append(f"A meal of {m.recipe_key!r} was left out: the plan sent has no "
                            "such recipe.")
            continue
        meals.append(m)

    # The dishes: placed ones (added), proposals and unmatched names.
    added: dict[str, AddedDish] = {}
    proposals: list[Proposal] = []
    unmatched: list[Unmatched] = []
    written: dict[str, RecipeDoc] = {}

    def propose(input_text: str, key: str, how: str, count: int, slot: str | None) -> None:
        c = by_key[key]
        s = slot or c.slot
        ref = _ref_for(key, docs)
        proposals.append(Proposal(
            input=input_text, recipe_key=key, title=c.title, label=c.label, how=how,
            count=count, slot=s, question=_question(input_text, c.title, c.label),
            op={"op": "add_recipe", "recipe_key": key, "title": c.title, "slot": s,
                "count": count, "spread": True,
                "ref": None if ref is None else ref.model_dump(mode="json",
                                                               exclude_none=True)}))

    def add(key: str, title: str, label: str, how: str, count: int, slot: str,
            input_text: str) -> None:
        if key in added:
            added[key].count += count
            return
        added[key] = AddedDish(recipe_key=key, title=title, label=label, how=how,
                               count=count, placed=0, slot=slot, input=input_text)

    for dish in dishes:
        if dish.lines:
            doc = _written_doc(dish)
            written[doc.key] = doc
            docs[doc.key] = doc
            add(doc.key, doc.title, "written by the assistant", "written", dish.count,
                dish.slot or "dinner", dish.recipe)
            if doc.servings is None:
                warnings.append(f"{doc.title}: say how many it serves; until then its amounts "
                                "stay unknown.")
            continue
        c = _by_key(dish.recipe, by_key)
        if c is not None:
            add(c.key, c.title, c.label, "key", dish.count, dish.slot or c.slot, dish.recipe)
            continue
        found, how = selection.match_name(dish.recipe, cands)
        if len(found) == 1 and how in ("exact", "plural"):
            c = found[0].candidate
            add(c.key, c.title, c.label, how, dish.count, dish.slot or c.slot, dish.recipe)
        elif len(found) == 1:
            propose(dish.recipe, found[0].candidate.key, how, dish.count, dish.slot)
        else:
            listed = ([m.candidate for m in found] if found
                      else selection.contains_all(dish.recipe, cands))
            unmatched.append(Unmatched(
                input=dish.recipe,
                reason=("several recipes fit" if len(listed) > 1 else
                        "no recipe of that name" if not listed else "not the same name"),
                candidates=[{"recipe_key": c.key, "title": c.title, "label": c.label}
                            for c in listed[:5]]))
    for p in proposed:
        if p.recipe_key not in by_key:
            unmatched.append(Unmatched(input=p.input or p.recipe_key,
                                       reason="no such recipe here"))
            continue
        if p.recipe_key in added:
            continue                      # placed already, by the dish that named it
        propose(p.input or by_key[p.recipe_key].title, p.recipe_key, p.how, p.count, p.slot)

    for key, a in added.items():
        if key in refs:
            continue
        ref = RecipeRef(key=key, doc=written[key]) if key in written else _ref_for(key, docs)
        if ref is None:
            raise PlanMealsError(f"{a.title}: its lines were not sent ({key}).")
        refs[key], titles[key], labels[key], slots[key] = ref, a.title, a.label, a.slot
    if len(refs) > MAX_RECIPES:
        raise PlanMealsError(f"A plan holds {MAX_RECIPES} recipes; this one would have "
                             f"{len(refs)}.")

    # The window: the current plan's start, never shortened; a new plan from start_date.
    want_days = days
    start = current.start_date if current else (start_date or dt.date.today()
                                                 + dt.timedelta(days=1))
    if current is not None:
        n_days = max(current.days, want_days or 0)
        if want_days is not None and want_days < current.days:
            notes.append(f"Your plan runs {current.days} days; it is not shortened to "
                         f"{want_days}.")
    else:
        load: dict[str, int] = {}
        for a in added.values():
            load[a.slot] = load.get(a.slot, 0) + a.count
        n_days = want_days or _default_days(load)
    prefs = current.prefs if current and current.prefs else Prefs()
    if household_servings is not None:
        prefs = prefs.model_copy(update={"household_servings": household_servings})
    for a in added.values():
        if a.slot not in prefs.slots_on:
            prefs = prefs.model_copy(update={"slots_on": [s for s in SLOTS
                                                           if s in prefs.slots_on
                                                           or s == a.slot]})
            notes.append(f"{a.title} is a {a.slot}: that slot is switched on in the draft.")

    have = {k: sum(1 for m in meals if m.recipe_key == k) for k in refs}
    total = len(meals) + sum(a.count for a in added.values())
    if total > MAX_MEALS:
        raise PlanMealsError(f"A plan holds {MAX_MEALS} meals; this one would have {total}.")
    keep = [m for m in meals if m.date is None or 0 <= (m.date - start).days < n_days]
    if len(keep) < len(meals):
        warnings.append(f"{len(meals) - len(keep)} meal(s) outside the plan's dates were left "
                        "out.")
    draft = MealPlanDraft(
        rev=current.rev if current else 0, start_date=start, days=n_days, prefs=prefs,
        recipes={k: DraftRecipe(ref=refs[k], slot=slots[k],
                                wanted=have[k] + (added[k].count if k in added else 0))
                 for k in sorted(refs)},
        meals=keep, settings=PlanSettings(lat=lat, lon=lon, max_km=max_km))
    draft.resolved = {k: resolve_one(refs[k], lat=lat, lon=lon, max_km=max_km)
                      for k in sorted(refs)}
    try:
        sched = schedule.compute(draft)
    except MealPlanError as e:
        raise PlanMealsError(f"{e.code}: {e.detail}") from e
    return _result(draft, sched, current, added, proposals, unmatched, labels, warnings,
                   notes, start_date_given=start_date, verbose=verbose)


def _result(draft: MealPlanDraft, sched: MealSchedule, current: MealPlanContext | None,
            added: dict[str, AddedDish], proposals: list[Proposal],
            unmatched: list[Unmatched], labels: dict[str, str], warnings: list[str],
            notes: list[str], *, start_date_given: dt.date | None,
            verbose: bool) -> MealPlanResult:
    old_ids = {m.id for m in current.meals} if current else set()
    placed_meals = [m for m in sched.meals if m.date is not None]
    new_placed = [m for m in placed_meals if m.id not in old_ids]
    for m in new_placed:
        if m.recipe_key in added:
            added[m.recipe_key].placed += 1
    # The ops a console applies to the plan it holds, in order.
    ops: list[dict[str, Any]] = []
    if current is None or current.days != draft.days:
        ops.append({"op": "set_window", "start_date": draft.start_date.isoformat(),
                    "days": draft.days})
    ctx_keys = {r.key for r in current.recipes} if current else set()
    for key, a in added.items():
        if key in ctx_keys:
            ops.append({"op": "add_meals", "recipe_key": key, "title": a.title,
                        "count": a.count})
        else:
            ops.append({"op": "add_recipe", "recipe_key": key, "title": a.title,
                        "slot": a.slot, "count": a.count,
                        "ref": draft.recipes[key].ref.model_dump(mode="json",
                                                                 exclude_none=True)})
    for m in sorted(new_placed, key=lambda m: (m.date, SLOTS.index(m.slot), m.id)):
        ops.append({"op": "place", "recipe_key": m.recipe_key, "title": m.title,
                    "date": m.date.isoformat(), "slot": m.slot})

    unplaced: dict[str, dict[str, Any]] = {}
    for m in sched.meals:
        if m.date is None and m.id not in old_ids and not m.pinned:
            u = unplaced.setdefault(m.recipe_key, {"title": m.title, "count": 0,
                                                   "reason": f"no free {m.slot} slot in "
                                                             f"{draft.days} days"})
            u["count"] += 1

    rec = next(s for s in sched.strategies if s.name == sched.recommended_strategy)
    other = next((s for s in sched.strategies if s.name != rec.name), None)
    trips = [TripBrief(date=t.date, stores=t.stores, items=len(t.lines),
                       total_cost=t.total_cost, total_is_floor=t.total_is_floor,
                       reason=t.reason) for t in rec.trips]
    plan_warnings = [w.message for w in sched.warnings
                     if w.level in ("must_fix", "decide")
                     and w.strategy in (None, rec.name)]
    if current is not None and current.approved_trips:
        n = len(current.approved_trips)
        plan_warnings.insert(0, f"Your plan has {n} approved trip(s); new meals change what "
                                "they buy, so the Meal plan will show them for review.")
    all_warnings = (warnings + plan_warnings)[:MAX_WARNINGS]
    if len(warnings) + len(plan_warnings) > MAX_WARNINGS:
        all_warnings[-1] = (f"... and {len(warnings) + len(plan_warnings) - MAX_WARNINGS + 1} "
                            "more in the Meal plan.")
    notes = [*notes, NEVER_APPROVES]
    if start_date_given is None and current is None:
        notes.append(f"The plan starts {draft.start_date.isoformat()} (tomorrow).")
    summary = MealPlanSummary(
        start_date=draft.start_date, days=draft.days,
        base_rev=current.rev if current else None,
        household_servings=draft.prefs.household_servings,
        meals=[PlannedMeal(date=m.date, slot=m.slot, title=m.title, recipe_key=m.recipe_key,
                           new=m.id not in old_ids)
               for m in sorted(placed_meals, key=lambda m: (m.date, SLOTS.index(m.slot),
                                                            m.id))],
        added=list(added.values()), proposals=proposals, unmatched=unmatched,
        unplaced=list(unplaced.values()), strategy=rec.name, trips=trips,
        other_strategy=None if other is None else OtherStrategy(
            name=other.name, trips=len(other.trips), total_cost=other.total_cost,
            total_is_floor=other.total_is_floor),
        total_cost=rec.total_cost, total_is_floor=rec.total_is_floor,
        nutrition=nutrition_line(sched), warnings=all_warnings, notes=notes, ops=ops)
    # The draft as the console would hold it once applied: every placed meal on its date.
    by_id = {m.id: m for m in sched.meals}
    meals = []
    for m in draft.meals:
        pm = by_id.get(m.id)
        meals.append(m if pm is None or m.date is not None
                     else m.model_copy(update={"date": pm.date, "slot": pm.slot}))
    seen = {m.id for m in meals}
    for pm in sched.meals:
        if pm.id not in seen:
            meals.append(Meal(id=pm.id, recipe_key=pm.recipe_key, date=pm.date, slot=pm.slot,
                              pinned=pm.date is None))
    out_draft = draft.model_copy(update={"meals": meals})
    return MealPlanResult(summary=summary, draft=out_draft, full=sched if verbose else None)


def nutrition_line(sched: MealSchedule) -> str:
    """One line a model can quote as it stands, with "at least" where a total is a floor and
    "(demo amounts)" whenever the amounts are demo house amounts."""
    from ..nutrition import NOT_DEPLOYED

    p = sched.period_nutrition
    if p is None:
        return NOT_DEPLOYED
    demo = " (demo amounts)" if p.demo_amounts else ""
    avg = p.per_day_average_over_complete_days
    if p.days_complete and avg and avg.get("energy_kcal") is not None:
        protein = avg.get("protein_g")
        return (f"nutrition per person per day{demo}: {p.days_complete} of {p.days_total} days "
                f"complete; {avg['energy_kcal']:,.0f} kcal"
                + (f" and {protein:,.0f} g protein" if protein is not None else "")
                + " on average over the complete days")
    kcal = p.lower_bound_total.get("energy_kcal")
    if kcal is None or kcal.amount is None:
        return f"nutrition{demo}: unknown for these meals"
    return (f"nutrition per person{demo}: no day is complete (a day needs a meal in every "
            f"slot that is on: {', '.join(p.slots_counted)}), so there is no daily average; "
            f"the period's meals add up to {'' if kcal.complete else 'at least '}"
            f"{kcal.amount:,.0f} kcal")

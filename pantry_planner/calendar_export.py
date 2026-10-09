"""Calendar export: the approved meal plan as all-day calendar events, built by code. No LLM, no
network call, no credentials.

One builder, several outlets (PLAN.md C7). /mealplan/schedule emits an ApprovedSchedule: the
trips the shopper approved, with their shopping lists; every meal placed on the board; and the
freeze and thaw reminders of the approved trips, each with its source. The console posts it
back to /calendar/preview (the events, each with a Google add-event link) or /calendar/ics (an
RFC 5545 file to import). Nothing is stored and nothing is fetched: a Google link is built
here and only ever opened by the shopper.

The rules every event keeps:

- all-day: DTSTART;VALUE=DATE with no time and no zone, so an event lands on its plan date in
  every time zone and on both sides of 2026-11-01 (timed events are decision D9);
- identity: the UID hashes the plan id with the item (a trip's strategy and date, a meal's id,
  a reminder's kind, meal or trip and product), so exporting again gives the same UIDs; the
  SEQUENCE is the plan's revision, which only counts up;
- a trip's DESCRIPTION is its shopping list (list_text), then "Store hours: unknown", the
  cited reason for its date and the demo-data line. Its LOCATION is "<store> (demo store)":
  the seeded street addresses belong to fictional stores and never appear while
  STORES_SYNTHETIC is on (the default);
- a cook event's DESCRIPTION is the recipe's lines as written, the servings, and nutrition per
  serving, with the "demo amounts" badge whenever those amounts are demo house amounts;
- a reminder needs a source: cited rows of shelf_life.json (rule ids, whose text and page are
  read here, never taken from the request) or the shopper's own setting. A reminder without
  one is refused (422), never exported as a bare instruction;
- a trip that needs review, or whose product is no longer stocked, is refused (409): an export
  never carries a list the shopper did not approve as it stands.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import html
import math
from collections.abc import Callable, Collection
from fractions import Fraction
from typing import Annotated, Any, Literal
from urllib.parse import quote, urlencode

from pydantic import BaseModel, Field, model_validator

from .mealplan import shelf
from .mealplan.models import (
    SLOTS,
    Action,
    MealPlanDraft,
    PlacedMeal,
    Slot,
    StrategyResult,
    Trip,
    TripLine,
)
from .mealplan.needs import recipe_servings
from .mealplan.place import day_label
from .models import MealNutrition, NutrientTotal, RecipeDoc, RecipeLine
from .nutrition import DEMO_BADGE_TITLE, NOT_DEPLOYED

CALENDAR_NAME = "Pantry plan"
PRODID = "-//pantry-planner//Meal plan calendar export 1//EN"
UID_DOMAIN = "pantry-planner"
GOOGLE_TEMPLATE = "https://calendar.google.com/calendar/render"
# Google's add-event link carries the description in its URL, so a long shopping list is cut
# and points at the .ics, which always has all of it.
GOOGLE_DETAILS_MAX = 1000
GOOGLE_CUT = "\n... cut here: the full text is in the .ics export."
STORE_HOURS = "Store hours: unknown, check before you go."
DEMO_STORE = "demo store"
DEMO_BADGE = "demo amounts"
MAX_TEXT = 60_000
# Generous: the 512 KB body cap is the real bound, and the emission must never fail a schedule.
MAX_ITEMS = 5000

EventKind = Literal["trip", "cook", "freeze", "thaw"]
Include = Literal["trips", "cooks", "reminders"]
INCLUDE_ALL: tuple[Include, ...] = ("trips", "cooks", "reminders")
_KIND_ORDER = {"trip": 0, "freeze": 1, "thaw": 2, "cook": 3}
_CATEGORY = {"trip": "Groceries", "cook": "Cooking", "freeze": "Reminder", "thaw": "Reminder"}

Text = Annotated[str, Field(max_length=MAX_TEXT)]
Short = Annotated[str, Field(max_length=300)]


class CalendarExportError(Exception):
    """An ApprovedSchedule the export refuses: 409 for a trip that is not approved as it stands
    (not_approved, needs_review, no_longer_stocked), 422 for one it cannot read (unknown_rule,
    nothing_to_export). REST {error: code, detail, item_id?, date?}."""

    def __init__(self, status: int, code: str, detail: str, **extra: Any):
        self.status, self.code, self.detail, self.extra = status, code, detail, extra
        super().__init__(f"{code}: {detail}")


# ─── The contract: ApprovedSchedule ──────────────────────────

class ReminderSource(BaseModel):
    """What a reminder rests on. cited: rows of shelf_life.json named by rule_ids (the export
    reads their words, source and page itself). your_setting: the shopper's own setting, named
    in `setting`. There is no third kind: a reminder with no source is not exported."""
    basis: Literal["cited", "your_setting"]
    rule_ids: list[Annotated[str, Field(min_length=1, max_length=80)]] = Field(
        default_factory=list, max_length=10)
    setting: str | None = Field(default=None, max_length=200)

    @model_validator(mode="after")
    def _named(self) -> ReminderSource:
        if self.basis == "cited" and not self.rule_ids:
            raise ValueError("a cited reminder names the rule ids it rests on")
        if self.basis == "your_setting" and not (self.setting or "").strip():
            raise ValueError("a reminder from your setting names that setting")
        return self


class TripReason(BaseModel):
    """Why the trip falls when it does, built by code from its lines: the tightest cited fridge
    time first, then a cited thaw, then the shopper's buy-ahead setting. basis none: no storage
    time limits it."""
    text: Text
    basis: Literal["cited", "your_setting", "none"]
    rule_ids: list[Short] = Field(default_factory=list, max_length=10)
    url: str | None = Field(default=None, max_length=2000)
    page_date: str | None = Field(default=None, max_length=40)


class ScheduleTrip(BaseModel):
    """An approved trip as the export reads it. id is the engine's trip id; item_id is its
    calendar identity. not_stocked names products with no offer in range any more."""
    item_id: Short
    id: Short
    strategy: Literal["fresh", "fewest_trips"]
    date: dt.date
    status: Literal["suggested", "approved", "needs_review"]
    stores: list[Short] = Field(default_factory=list, max_length=50)
    reason: TripReason
    list_text: Text
    not_stocked: list[Short] = Field(default_factory=list, max_length=MAX_ITEMS)
    total_cost: float
    total_is_floor: bool
    price_delta: float | None = None
    fingerprint: str = Field(default="", max_length=64)


class CookNutrition(BaseModel):
    """One serving of the recipe (or the whole recipe when it does not say how many it serves),
    as /mealplan/schedule computed it. demo_amounts: the "demo amounts" badge goes with every
    number."""
    basis: Literal["per_serving", "per_recipe"]
    servings: int | None = None
    status: Literal["complete", "incomplete", "below_floor"]
    totals: dict[str, NutrientTotal]
    demo_amounts: bool
    coverage_note: Text = ""


class ScheduleCook(BaseModel):
    """A meal placed on the board. servings is how many it feeds (the household's number unless
    the meal sets its own); recipe_servings is how many the recipe makes (None: not stated and
    not answered). ingredients are the recipe's lines as written."""
    item_id: Short
    meal_id: Short
    recipe_key: Short
    title: Short
    date: dt.date
    slot: Slot
    servings: int = Field(ge=1, le=100)
    servings_set_by: Literal["household", "meal"] = "household"
    recipe_servings: int | None = Field(default=None, ge=1, le=100)
    recipe_servings_basis: Literal["source", "your_setting"] | None = None
    occurrence: int = Field(default=1, ge=1)
    of: int = Field(default=1, ge=1)
    ingredients: list[Short] = Field(default_factory=list, max_length=60)
    label: Short | None = None
    nutrition: CookNutrition | None = None
    nutrition_note: Short = ""


class ScheduleReminder(BaseModel):
    """Freeze on arrival (the trip's date) or move to the fridge to thaw. source is required:
    a request without one is a 422."""
    item_id: Short
    kind: Literal["freeze", "thaw"]
    date: dt.date
    product_id: int
    product: Short
    text: Text
    trip_id: Short | None = None
    meal_id: Short | None = None
    source: ReminderSource


class Blocked(BaseModel):
    """Why the schedule cannot be exported yet, one per trip."""
    item_id: str
    code: Literal["not_approved", "needs_review", "no_longer_stocked"]
    date: dt.date
    message: str


class ApprovedSchedule(BaseModel):
    """What the calendar export reads (PLAN.md 3.9), emitted by /mealplan/schedule as
    approved_schedule and posted back unchanged. plan_id and rev are the draft's: the UIDs hash
    the first, SEQUENCE is the second. exportable is False while `blocked` names a trip; the
    export checks the trips again itself. actions are the engine's shop, freeze, thaw and cook
    actions for the approved trips, kept for consoles that read them."""
    v: Literal[1] = 1
    plan_id: str = Field(default="", max_length=64)
    rev: int = Field(default=0, ge=0)
    start_date: dt.date
    days: int = Field(default=14, ge=1, le=14)
    calendar_name: Short = CALENDAR_NAME
    trips: list[ScheduleTrip] = Field(default_factory=list, max_length=28)
    cooks: list[ScheduleCook] = Field(default_factory=list, max_length=56)
    reminders: list[ScheduleReminder] = Field(default_factory=list, max_length=MAX_ITEMS)
    actions: list[dict] = Field(default_factory=list, max_length=MAX_ITEMS)
    exportable: bool = False
    blocked: list[Blocked] = Field(default_factory=list, max_length=28)
    synthetic_notice: Text = ""

    @model_validator(mode="after")
    def _unique_items(self) -> ApprovedSchedule:
        ids = [x.item_id for x in (*self.trips, *self.cooks, *self.reminders)]
        if len(ids) != len(set(ids)):
            raise ValueError("item ids must be unique")
        return self


# ─── From a meal schedule ────────────────────────────────────

def _first_meal_date(ln: TripLine) -> dt.date:
    return min(m.date for m in ln.for_meals)


def trip_reason(trip: Trip, buy_ahead: int) -> TripReason:
    """The cited reason for a trip's date, from its lines. An approved trip's own `reason`
    says only that the shopper approved it, so the storage time behind the date is found
    again here: the line with the shortest cited fridge time, for the last meal it is bought
    for (the meal that keeps the trip from being earlier)."""
    lines = [ln for ln in trip.lines if ln.for_meals]
    fridge = [ln for ln in lines if ln.shelf_life.status == "cited" and ln.storage == "fridge"
              and ln.shelf_life.days_planned is not None and ln.shelf_life.verbatim
              and ln.shelf_life.rule_ids]
    if fridge:
        ln = min(fridge, key=lambda x: (x.shelf_life.days_planned, _first_meal_date(x),
                                        x.product.name))
        meal = max(ln.for_meals, key=lambda m: (m.date, SLOTS.index(m.slot), m.meal_id))
        info = ln.shelf_life
        return TripReason(
            text=(f"{meal.title} on {day_label(meal.date)} needs {ln.product.name}, which keeps "
                  f"{info.verbatim[0]} in the fridge ({info.source}, {info.rule_ids[0]})."),
            basis="cited", rule_ids=info.rule_ids[:1], url=info.url, page_date=info.page_date)
    frozen = [ln for ln in lines if ln.shelf_life.status == "cited" and ln.storage == "freezer"
              and ln.shelf_life.rule_ids]
    if frozen:
        ln = min(frozen, key=lambda x: (_first_meal_date(x), x.product.name))
        meal = min(ln.for_meals, key=lambda m: (m.date, SLOTS.index(m.slot), m.meal_id))
        info = ln.shelf_life
        return TripReason(
            text=(f"{meal.title} on {day_label(meal.date)} needs {ln.product.name}, kept frozen "
                  f"and thawed in the fridge ({info.source}, {', '.join(info.rule_ids)})."),
            basis="cited", rule_ids=info.rule_ids[:10], url=info.url, page_date=info.page_date)
    setting = [ln for ln in lines if ln.shelf_life.status == "your_setting"]
    if setting:
        ln = min(setting, key=lambda x: (_first_meal_date(x), x.product.name))
        meal = max(ln.for_meals, key=lambda m: (m.date, SLOTS.index(m.slot), m.meal_id))
        return TripReason(
            text=(f"{meal.title} on {day_label(meal.date)} needs {ln.product.name}, which has no "
                  f"cited storage time; your setting buys it at most {buy_ahead} day(s) ahead."),
            basis="your_setting")
    return TripReason(text="No storage time limits this trip's date; it is the date you "
                           "approved.", basis="none")


def _blocked(trip: ScheduleTrip) -> Blocked | None:
    when = day_label(trip.date)
    if trip.status == "suggested":
        return Blocked(item_id=trip.item_id, code="not_approved", date=trip.date,
                       message=f"The trip on {when} is a suggestion: approve it first.")
    if trip.not_stocked:
        return Blocked(item_id=trip.item_id, code="no_longer_stocked", date=trip.date,
                       message=(f"{', '.join(trip.not_stocked)} on your approved trip {when} "
                                "has no offer in range any more: choose another product or "
                                "leave it off, then approve the trip again."))
    if trip.status == "needs_review":
        return Blocked(item_id=trip.item_id, code="needs_review", date=trip.date,
                       message=(f"The trip on {when} changed since you approved it: review it "
                                "and approve it again."))
    return None


def _cook_nutrition(mn: MealNutrition | None) -> CookNutrition | None:
    if mn is None:
        return None
    return CookNutrition(basis=mn.basis, servings=mn.servings, status=mn.status,
                         totals=mn.totals, demo_amounts=mn.demo_amounts,
                         coverage_note=mn.coverage.note)


def _ingredient(line: RecipeLine) -> str:
    text = " ".join(line.text.split())
    if text:
        return text
    qty = "" if line.quantity is None else f"{line.quantity:g} "
    unit = f"{line.unit} " if line.unit and line.unit != "each" else ""
    return f"{qty}{unit}{line.name}".strip()


def _unique(item_id: str, seen: set[str]) -> str:
    out, n = item_id, 2
    while out in seen:
        out, n = f"{item_id}-{n}", n + 1
    seen.add(out)
    return out


def approved_schedule(draft: MealPlanDraft, results: list[StrategyResult],
                      placed: list[PlacedMeal], docs: dict[str, RecipeDoc],
                      per_recipe: dict[str, MealNutrition], *, notice: str) -> dict | None:
    """The ApprovedSchedule for a computed schedule, as JSON (MealSchedule.approved_schedule):
    the draft's approved trips as the schedule now has them, their freeze and thaw reminders,
    and a cook event for every placed meal. notice is the schedule's demo-data line. None when
    there is nothing to export."""
    by_name = {r.name: r for r in results}
    seen: set[str] = set()
    trips: list[ScheduleTrip] = []
    reminders: list[ScheduleReminder] = []
    actions: list[dict] = []
    for appr in sorted(draft.trips, key=lambda t: (t.date, t.strategy)):
        res = by_name[appr.strategy]
        trip = next((t for t in res.trips if t.date == appr.date), None)
        if trip is None:
            continue
        trips.append(ScheduleTrip(
            item_id=_unique(f"trip-{trip.id}", seen), id=trip.id, strategy=res.name,
            date=trip.date, status=trip.status, stores=trip.stores,
            reason=trip_reason(trip, draft.prefs.buy_ahead_days), list_text=trip.list_text,
            not_stocked=trip.not_stocked, total_cost=trip.total_cost,
            total_is_floor=trip.total_is_floor, price_delta=trip.price_delta,
            fingerprint=trip.fingerprint))
        names = {ln.product.id: ln.product.name for ln in trip.lines}
        meals = {m.meal_id for ln in trip.lines for m in ln.for_meals}
        for a in res.actions:
            if a.trip_id == trip.id or (a.kind == "cook" and a.meal_id in meals):
                actions.append(a.model_dump(mode="json"))
            if a.trip_id != trip.id or a.kind not in ("freeze", "thaw") or a.product_id is None:
                continue
            source = _reminder_source(a, draft.prefs.thaw_reminder)
            if source is None:
                continue      # no source, no reminder: the export never carries a bare one
            owner = a.meal_id if a.kind == "thaw" else trip.id
            reminders.append(ScheduleReminder(
                item_id=_unique(f"{a.kind}-{owner}-{a.product_id}", seen), kind=a.kind,
                date=a.date, product_id=a.product_id,
                product=names.get(a.product_id, f"product {a.product_id}"), text=a.text,
                trip_id=trip.id, meal_id=a.meal_id, source=source))

    own = {m.id: m for m in draft.meals}
    dated = [m for m in placed if m.date is not None]
    totals: dict[str, int] = {}
    for m in dated:
        totals[m.recipe_key] = totals.get(m.recipe_key, 0) + 1
    counted: dict[str, int] = {}
    cooks: list[ScheduleCook] = []
    for m in sorted(dated, key=lambda m: (m.date, SLOTS.index(m.slot), m.id)):
        counted[m.recipe_key] = counted.get(m.recipe_key, 0) + 1
        doc = docs[m.recipe_key]
        rs = recipe_servings(draft, m.recipe_key, doc)
        basis = (doc.servings_basis or "source") if doc.servings else (
            "your_setting" if rs is not None else None)
        resolved = draft.resolved.get(m.recipe_key)
        mn = per_recipe.get(m.recipe_key)
        cooks.append(ScheduleCook(
            item_id=_unique(f"cook-{m.id}", seen), meal_id=m.id, recipe_key=m.recipe_key,
            title=m.title, date=m.date, slot=m.slot, servings=m.servings,
            servings_set_by="meal" if own.get(m.id) and own[m.id].servings else "household",
            recipe_servings=rs, recipe_servings_basis=basis,
            occurrence=counted[m.recipe_key], of=totals[m.recipe_key],
            ingredients=[_ingredient(ln) for ln in doc.lines],
            label=(resolved.label if resolved else None) or doc.source.label,
            nutrition=_cook_nutrition(mn),
            nutrition_note="" if mn is not None or per_recipe else NOT_DEPLOYED))

    if not trips and not cooks:
        return None
    blocked = [b for b in (_blocked(t) for t in trips) if b is not None]
    return ApprovedSchedule(
        plan_id=draft.id, rev=draft.rev, start_date=draft.start_date, days=draft.days,
        trips=trips, cooks=cooks, reminders=reminders, actions=actions,
        exportable=not blocked, blocked=blocked, synthetic_notice=notice,
    ).model_dump(mode="json")


def _reminder_source(a: Action, thaw_reminder: str) -> ReminderSource | None:
    """A freeze or thaw action's source: its cited rows, or the shopper's setting it follows.
    None when it names neither."""
    if a.basis == "cited" and a.rule_ids:
        return ReminderSource(basis="cited", rule_ids=a.rule_ids)
    if a.basis == "your_setting":
        setting = (f"thaw reminders: {thaw_reminder.replace('_', ' ')}" if a.kind == "thaw"
                   else "freeze on arrival, set by you")
        return ReminderSource(basis="your_setting", setting=setting)
    return None


# ─── Events ──────────────────────────────────────────────────

class CalendarEvent(BaseModel):
    """One all-day event. end is exclusive (the day after), as DTEND;VALUE=DATE and Google's
    dates= both read it. labels are the console's chips (store hours unknown, demo store, demo
    amounts, cited, your setting)."""
    uid: str
    item_id: str
    kind: EventKind
    date: dt.date
    end: dt.date
    all_day: Literal[True] = True
    title: str
    location: str | None
    description: str
    categories: list[str]
    sequence: int
    labels: list[str]
    google_url: str


class CalendarPreview(BaseModel):
    """POST /calendar/preview: the events the .ics would hold, in date order."""
    calendar_name: str
    all_day: Literal[True] = True
    events: list[CalendarEvent]
    counts: dict[str, int]
    filename: str
    notes: list[str]


IMPORT_NOTES = [
    "Every event is all-day, on its plan date in any time zone.",
    "Import the .ics into a new calendar named Pantry plan: to replace an export later, delete "
    "that calendar and import the new file.",
    "A Google add-event link adds one event to your own calendar; it cannot set reminders or "
    "update an event later.",
    "Store hours are not known: check before you go. Stores, prices and stock are demo data.",
]


def plan_key(s: ApprovedSchedule) -> str:
    """The plan's identity for UIDs: its id, or its start date for a plan saved without one."""
    return s.plan_id or f"plan-{s.start_date.isoformat()}"


def event_uid(key: str, item_id: str) -> str:
    """A stable UID: the same plan and item give the same UID on every export."""
    digest = hashlib.sha256(f"{key}\n{item_id}".encode()).hexdigest()[:32]
    return f"{digest}@{UID_DOMAIN}"


def check_exportable(s: ApprovedSchedule) -> None:
    """Raise 409 for the first trip (by date) that is not approved as it stands."""
    for trip in sorted(s.trips, key=lambda t: (t.date, t.item_id)):
        b = _blocked(trip)
        if b is not None:
            raise CalendarExportError(409, b.code, b.message, item_id=b.item_id,
                                      date=b.date.isoformat())


def location_for(stores: list[str], addresses: dict[str, str] | None) -> str | None:
    """'<store> (demo store)' for each stop. A street address only for a store a deployment
    says is real (STORES_SYNTHETIC off and an address on file)."""
    if not stores:
        return None
    if addresses is None:
        return "; ".join(f"{s} ({DEMO_STORE})" for s in stores)
    return "; ".join(f"{s}, {addresses[s]}" if addresses.get(s) else s for s in stores)


def _round_half_up(x: float, places: int = 0) -> float:
    f = 10 ** places
    return math.floor(abs(x) * f + 0.5) / f * (1 if x >= 0 else -1)


_NUTRIENT_WORDS = {"energy_kcal": "energy", "protein_g": "protein", "fat_g": "fat",
                   "satfat_g": "saturated fat", "carbohydrate_g": "carbohydrate",
                   "fibre_g": "fibre", "sugars_g": "sugars", "sodium_mg": "sodium"}


def nutrient_text(key: str, total: NutrientTotal | None) -> str:
    """'642 kcal', '≥ 35 g protein' or 'sodium unknown': never 0 for a figure that is not
    there (the console's nutritionFormat, in the same words)."""
    word = _NUTRIENT_WORDS[key]
    if total is None or total.status == "unknown" or total.amount is None:
        return "kcal unknown" if key == "energy_kcal" else f"{word} unknown"
    amount = total.amount
    if total.unit == "g" and abs(amount) < 10:
        number = f"{_round_half_up(amount, 1):g}"
    else:
        number = f"{int(_round_half_up(amount)):,}"
    at_least = "≥ " if total.status == "at_least" else ""
    if key == "energy_kcal":
        return f"{at_least}{number} kcal"
    return f"{at_least}{number} {total.unit} {word}"


def _share(meal: int, recipe: int) -> str:
    f = Fraction(meal, recipe)
    whole, rest = divmod(f.numerator, f.denominator)
    if rest == 0:
        return f"{whole} times each amount"
    part = f"{rest}/{f.denominator}"
    return f"{part} of each amount" if whole == 0 else f"{whole} {part} times each amount"


def _servings_lines(c: ScheduleCook) -> list[str]:
    who = "your household setting" if c.servings_set_by == "household" else "set for this meal"
    people = "person" if c.servings == 1 else "people"
    out = [f"{c.slot.capitalize()} for {c.servings} {people} ({who})."]
    if c.recipe_servings is None:
        out.append("The recipe does not say how many it serves: the amounts below are for the "
                   "whole recipe.")
    else:
        yours = " (your answer)" if c.recipe_servings_basis == "your_setting" else ""
        if c.recipe_servings == c.servings:
            out.append(f"The recipe makes {c.recipe_servings} servings{yours}: use the amounts "
                       "as given.")
        else:
            out.append(f"The recipe makes {c.recipe_servings} servings{yours}; for "
                       f"{c.servings}, use {_share(c.servings, c.recipe_servings)}.")
    return out


def _nutrition_lines(c: ScheduleCook) -> list[str]:
    n = c.nutrition
    if n is None:
        return [f"Nutrition: {c.nutrition_note or 'unknown'}."]
    head = ("Nutrition per serving" if n.basis == "per_serving"
            else "Nutrition for the whole recipe")
    badge = f" ({DEMO_BADGE})" if n.demo_amounts else ""
    values = ", ".join(nutrient_text(k, n.totals.get(k)) for k in _NUTRIENT_WORDS)
    out = [f"{head}{badge}: {values}."]
    if n.coverage_note:
        out.append(n.coverage_note.rstrip(".") + ".")
    if n.demo_amounts:
        out.append(DEMO_BADGE_TITLE + ".")
    out.append("Reference values for generic foods, raw ingredients summed; not dietary advice.")
    return out


def cook_description(c: ScheduleCook) -> str:
    out = _servings_lines(c)
    out.append("")
    out.append("Ingredients, as the recipe gives them:")
    out += [f"- {x}" for x in c.ingredients] or ["- (the recipe lists none)"]
    out.append("")
    out += _nutrition_lines(c)
    if c.label:
        out.append(f"Recipe: {c.label}.")
    if c.of > 1:
        out.append(f"Meal {c.occurrence} of {c.of} of {c.title} in this plan.")
    return "\n".join(out)


def trip_description(t: ScheduleTrip, notice: str) -> str:
    out = [t.list_text.rstrip("\n"), "", STORE_HOURS, f"Why this date: {t.reason.text}"]
    if t.reason.url:
        dated = f", page dated {t.reason.page_date}" if t.reason.page_date else ""
        out.append(f"Source: {t.reason.url}{dated}.")
    if t.price_delta:
        sign = "+" if t.price_delta >= 0 else "-"
        out.append(f"Price changed since you approved it: {sign}${abs(t.price_delta):.2f} "
                   "(demo prices).")
    out.append(f"Demo data: {notice or 'stores, prices and stock are synthetic.'}")
    return "\n".join(out)


def reminder_description(r: ScheduleReminder, rules: Callable[[str], dict],
                         trips: dict[str, ScheduleTrip]) -> str:
    out = [r.text]
    if r.source.basis == "cited":
        for rid in r.source.rule_ids:
            row = rules(rid)
            page = shelf.source(row["source"])
            out.append(f"Source: {shelf.short_source(row['source'])}, rule {rid}: "
                       f"\"{row['verbatim']}\" ({page['url']}, page dated {page['page_date']}).")
    else:
        out.append(f"Source: your setting ({r.source.setting}).")
    trip = trips.get(r.trip_id or "")
    if trip is not None:
        out.append(f"For your shopping trip on {day_label(trip.date)}.")
    return "\n".join(out)


def _check_rules(s: ApprovedSchedule, rules: Callable[[str], dict]) -> None:
    for r in s.reminders:
        for rid in r.source.rule_ids:
            try:
                rules(rid)
            except KeyError:
                raise CalendarExportError(
                    422, "unknown_rule", f"Reminder {r.item_id} cites {rid!r}, which is not a "
                    "row of the storage-time sources.", item_id=r.item_id) from None


def build_events(s: ApprovedSchedule, *, include: Collection[str] = INCLUDE_ALL,
                 addresses: dict[str, str] | None = None,
                 rules: Callable[[str], dict] = shelf.rule) -> list[CalendarEvent]:
    """The events for an approved schedule, in date order. Raises CalendarExportError: 409
    while a trip is not approved as it stands, 422 for a reminder citing an unknown rule.
    addresses is None while the stores are synthetic (LOCATION '<store> (demo store)')."""
    check_exportable(s)
    _check_rules(s, rules)
    key = plan_key(s)
    events: list[tuple[tuple, CalendarEvent]] = []

    def add(item_id: str, kind: EventKind, date: dt.date, title: str, location: str | None,
            description: str, labels: list[str], slot: int = 0) -> None:
        end = date + dt.timedelta(days=1)
        events.append(((date, _KIND_ORDER[kind], slot, item_id), CalendarEvent(
            uid=event_uid(key, item_id), item_id=item_id, kind=kind, date=date, end=end,
            title=title, location=location, description=description,
            categories=[s.calendar_name, _CATEGORY[kind]], sequence=s.rev, labels=labels,
            google_url=google_template_url(title, date, end, description, location))))

    if "trips" in include:
        for t in s.trips:
            stores = ", ".join(t.stores)
            labels = ["store hours unknown", "demo prices"]
            if addresses is None and t.stores:
                labels.insert(1, DEMO_STORE)
            if t.price_delta:
                labels.append("price changed")
            add(t.item_id, "trip", t.date, f"Groceries: {stores}" if stores else "Groceries",
                location_for(t.stores, addresses), trip_description(t, s.synthetic_notice),
                labels)
    if "reminders" in include:
        trips = {t.id: t for t in s.trips}
        for r in s.reminders:
            title = (f"Freeze on arrival: {r.product}" if r.kind == "freeze"
                     else f"Thaw: move {r.product} to the fridge")
            add(r.item_id, r.kind, r.date, title, None, reminder_description(r, rules, trips),
                ["cited"] if r.source.basis == "cited" else ["your setting"])
    if "cooks" in include:
        for c in s.cooks:
            title = f"Cook: {c.title} ({c.slot}" + (f", {c.occurrence} of {c.of})"
                                                     if c.of > 1 else ")")
            labels = []
            if c.nutrition is not None and c.nutrition.demo_amounts:
                labels.append(DEMO_BADGE)
            if c.nutrition is None:
                labels.append("nutrition not shown")
            if c.recipe_servings is None:
                labels.append("servings unknown")
            add(c.item_id, "cook", c.date, title, None, cook_description(c), labels,
                SLOTS.index(c.slot))
    return [e for _, e in sorted(events, key=lambda x: x[0])]


def preview(s: ApprovedSchedule, events: list[CalendarEvent]) -> CalendarPreview:
    counts = {k: sum(1 for e in events if e.kind == k) for k in _KIND_ORDER}
    return CalendarPreview(calendar_name=s.calendar_name, events=events, counts=counts,
                           filename=ics_filename(s, events), notes=IMPORT_NOTES)


def ics_filename(s: ApprovedSchedule, events: list[CalendarEvent]) -> str:
    first = min((e.date for e in events), default=s.start_date)
    return f"pantry-plan-{first.isoformat()}.ics"


# ─── RFC 5545 ────────────────────────────────────────────────

def escape_text(value: str) -> str:
    """A TEXT value (RFC 5545 3.3.11): backslash, semicolon, comma and newline escaped; other
    control characters, which TEXT may not hold, dropped."""
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = "".join(ch for ch in value if ch in "\n\t" or (ord(ch) >= 0x20 and ord(ch) != 0x7F))
    return (value.replace("\\", "\\\\").replace(";", "\\;").replace(",", "\\,")
            .replace("\n", "\\n"))


def fold(line: str, limit: int = 75) -> str:
    """Fold a content line at `limit` octets (RFC 5545 3.1): each continuation starts with one
    space, which counts toward its own line's octets. A break never falls inside a character's
    UTF-8 sequence. CRLF between the parts; none at the end."""
    parts: list[str] = []
    current: list[str] = []
    size = 0
    for ch in line:
        n = len(ch.encode("utf-8"))
        if size + n > limit:
            parts.append("".join(current))
            current, size = [" "], 1
        current.append(ch)
        size += n
    parts.append("".join(current))
    return "\r\n".join(parts)


def _ics_date(d: dt.date) -> str:
    return d.strftime("%Y%m%d")


def render_ics(events: list[CalendarEvent], *, calendar_name: str = CALENDAR_NAME,
               now: dt.datetime | None = None) -> str:
    """An RFC 5545 calendar of all-day events: CRLF line endings, lines folded at 75 octets,
    DTSTART/DTEND as VALUE=DATE (no TZID, so no VTIMEZONE is needed), a stable UID and the
    plan's revision as SEQUENCE. DTSTAMP is when the file was made (UTC)."""
    stamp = (now or dt.datetime.now(dt.UTC)).astimezone(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    lines = ["BEGIN:VCALENDAR", "VERSION:2.0", f"PRODID:{PRODID}", "CALSCALE:GREGORIAN",
             f"X-WR-CALNAME:{escape_text(calendar_name)}"]
    for e in events:
        lines += ["BEGIN:VEVENT", f"UID:{e.uid}", f"DTSTAMP:{stamp}", f"SEQUENCE:{e.sequence}",
                  f"DTSTART;VALUE=DATE:{_ics_date(e.date)}",
                  f"DTEND;VALUE=DATE:{_ics_date(e.end)}", f"SUMMARY:{escape_text(e.title)}"]
        if e.location:
            lines.append(f"LOCATION:{escape_text(e.location)}")
        lines += [f"DESCRIPTION:{escape_text(e.description)}",
                  f"CATEGORIES:{','.join(escape_text(c) for c in e.categories)}",
                  "TRANSP:TRANSPARENT", "STATUS:CONFIRMED", "END:VEVENT"]
    lines.append("END:VCALENDAR")
    return "".join(fold(ln) + "\r\n" for ln in lines)


# ─── Google add-event links ──────────────────────────────────

def google_template_url(title: str, start: dt.date, end: dt.date, details: str,
                        location: str | None) -> str:
    """Google Calendar's add-event page for one all-day event (dates=YYYYMMDD/YYYYMMDD, end
    exclusive). No key, no account on our side: the shopper opens it while signed in. The
    format is community-documented, not an official API. details are HTML-escaped and cut at
    GOOGLE_DETAILS_MAX characters with a pointer to the .ics."""
    if len(details) > GOOGLE_DETAILS_MAX:
        details = details[:GOOGLE_DETAILS_MAX - len(GOOGLE_CUT)].rstrip() + GOOGLE_CUT
    params = [("action", "TEMPLATE"), ("text", title),
              ("dates", f"{_ics_date(start)}/{_ics_date(end)}"),
              ("details", html.escape(details, quote=False))]
    if location:
        params.append(("location", location))
    return f"{GOOGLE_TEMPLATE}?{urlencode(params, quote_via=quote, safe='/')}"

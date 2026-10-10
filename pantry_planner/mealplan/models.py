"""Meal-plan shapes: the draft the browser holds and the schedule the server computes from it.

The draft (MealPlanDraft) is the shopper's plan, kept in the browser under
`pantry.mealplan.v1`; the server is stateless and trusts only ids from it. Every fact a
schedule shows (names, sizes, prices, storage times) is re-read by id on each call.

Count semantics: a recipe's `wanted` count is that many meal occasions, each serving
`prefs.household_servings` people unless the meal says otherwise. Needs scale by
meal servings / recipe servings.
"""
from __future__ import annotations

import datetime as dt
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, model_validator

from ..models import (
    AlternativeRanking,
    DayNutrition,
    DroppedIngredient,
    MatchLevel,
    MealNutrition,
    NutritionTargets,
    PeriodNutrition,
    RecipeDoc,
    TripOption,
)

Slot = Literal["breakfast", "lunch", "dinner", "snack"]
SLOTS: tuple[str, ...] = ("breakfast", "lunch", "dinner", "snack")
Strategy = Literal["fresh", "fewest_trips"]
STRATEGIES: tuple[str, ...] = ("fresh", "fewest_trips")
# The shopper's diet, and the products' contains-tags they can choose to avoid.
Diet = Literal["any", "pescatarian", "vegetarian", "vegan"]
Avoid = Literal["dairy", "gluten", "nuts", "soy", "egg"]
# Where a purchase is kept: from the product's storage class in shelf_life.json (the file's
# own label), or the freezer when the plan freezes it on arrival.
Storage = Literal["fridge", "freezer", "pantry"]

MAX_MEALS = 56
MAX_RECIPES = 12
MAX_DAYS = 14
MAX_SERVINGS = 20
MAX_DRAFT_BYTES = 256 * 1024
# Packs on one trip line, as the draft sends them back (an override, or an approved trip's
# snapshot). The engine never computes more: MAX_LINE_QUANTITY x 20 servings x 56 meals of
# a 2-pack is about 6e8. The bound keeps packs x price a finite float; JSON's integers have
# no limit, and one past 1e308 would fail the price sum with a 500.
MAX_PACKS = 10_000_000_000
# A line's price at approval: MAX_PACKS packs at $1,000 each. Finite, so the delta is too.
MAX_LINE_PRICE = 1e13
# Pins: at most this many lines of one recipe (alternatives.MAX_PINS, the chat cart's own
# bound), each a line number of at most 4 digits and a product id that fits the database's
# integer. A product id past that would fail the query, not the check.
MAX_PINS_PER_RECIPE = 40
MAX_PRODUCT_ID = 2**31 - 1
# Origin rules for the plan: as many countries as /plan/spec and /mealplan/resolve take.
MAX_COUNTRIES = 50
# Stores the shopper may turn off in one plan.
MAX_STORES_OFF = 50
# The grocery budget for the whole plan, in dollars.
MAX_BUDGET = 10_000


class MealPlanError(ValueError):
    """A draft the schedule cannot be computed for. REST 422 {error: code, detail}.

    Codes: slot_capacity, unknown_recipe_key, stale_product, invalid_dates, pin_invalid,
    version."""

    def __init__(self, code: str, detail: str, **extra: Any):
        self.code, self.detail, self.extra = code, detail, extra
        super().__init__(f"{code}: {detail}")


# ─── Recipes ─────────────────────────────────────────────────

class RecipeRef(BaseModel):
    """Which recipe: a library slug, a demo starter key or a RecipeDoc (pasted, imported or
    written by the assistant). The key is 'lib:<slug>', 'starter:<key>' or the doc's own key.
    `servings` is the shopper's answer when the recipe does not say how many it serves."""
    key: str = Field(min_length=1, max_length=100)
    slug: str | None = Field(default=None, max_length=100)
    starter: str | None = Field(default=None, max_length=100)
    doc: RecipeDoc | None = None
    servings: int | None = Field(default=None, ge=1, le=100)

    @model_validator(mode="after")
    def _one_source(self) -> RecipeRef:
        given = [x for x in (self.slug, self.starter, self.doc) if x is not None]
        if len(given) != 1:
            raise ValueError("a recipe ref names exactly one of slug, starter or doc")
        want = (f"lib:{self.slug}" if self.slug is not None
                else f"starter:{self.starter}" if self.starter is not None
                else self.doc.key)
        if self.key != want:
            raise ValueError(f"recipe ref key {self.key!r} should be {want!r}")
        return self


class ResolvedLine(BaseModel):
    """One recipe line and the product the pipeline chose for it (None: nothing chosen, the
    line is on not_stocked, out_of_range or skipped). need_qty/need_uom are the line's
    amount in canonical units for one batch of the recipe, None when unknown."""
    line_no: int
    name: str
    quantity: float | None = None
    unit: str = ""
    need_qty: float | None = None
    need_uom: str | None = None
    product_id: int | None = None
    product_name: str | None = None
    match: MatchLevel | None = None
    amount_basis: str | None = None


ResolveStatus = Literal["ok", "needs_servings", "unconfirmed_lines", "not_found",
                        "unparseable", "aborted", "llm_error"]


class ResolvedRecipe(BaseModel):
    """A recipe after its slow resolve: its lines and the products chosen. A recipe that
    fails keeps a status of its own and never fails the plan. `needs_servings`: the recipe
    does not say how many it serves and the shopper has not answered, so its amounts stay
    unknown. `servings_basis` is 'source' or 'your_setting' (the shopper's answer)."""
    key: str
    title: str
    status: ResolveStatus
    message: str = ""
    servings: int | None = None
    servings_basis: Literal["source", "your_setting"] | None = None
    label: str | None = None
    lines: list[ResolvedLine] = Field(default_factory=list)
    not_stocked: list[DroppedIngredient] = Field(default_factory=list)
    out_of_range: list[DroppedIngredient] = Field(default_factory=list)
    skipped: list[DroppedIngredient] = Field(default_factory=list)
    llm_cost_usd: float = 0.0
    model_used: str = ""
    burr_run: str = ""
    cached: bool = False


class DraftRecipe(BaseModel):
    """A recipe in the tray: how many meals of it are wanted and in which slot. `servings`
    is how many the recipe serves, the shopper's answer to needs_servings (never a meal's
    head count: that is Meal.servings)."""
    ref: RecipeRef
    wanted: int = Field(default=0, ge=0, le=28)
    slot: Slot = "dinner"
    servings: int | None = Field(default=None, ge=1, le=100)


# ─── The draft ───────────────────────────────────────────────

class Meal(BaseModel):
    """One meal occasion. A meal with a date, or pinned, never moves; a pinned meal with no
    date stays in the tray. servings None means the household's."""
    id: str = Field(min_length=1, max_length=120)
    recipe_key: str = Field(min_length=1, max_length=100)
    date: dt.date | None = None
    slot: Slot | None = None
    servings: int | None = Field(default=None, ge=1, le=MAX_SERVINGS)
    pinned: bool = False


class Prefs(BaseModel):
    """Household settings. shop_weekdays are ISO weekdays minus one (Monday 0 .. Sunday 6).
    buy_ahead_days is the shopper's own limit for a perishable with no cited storage time.
    thaw_reminder is used only where no cited thaw time applies. budget is the groceries for
    the whole plan (None: no budget); diet and avoid choose the recipes the catalog offers.
    All three have defaults, so a draft saved before them still loads."""
    household_servings: int = Field(default=2, ge=1, le=MAX_SERVINGS)
    slots_on: list[Slot] = Field(default_factory=lambda: list(SLOTS), min_length=1)
    shop_weekdays: list[int] = Field(default_factory=lambda: list(range(7)), min_length=1,
                                     max_length=7)
    max_trips: int | None = Field(default=None, ge=1, le=MAX_DAYS)
    buy_ahead_days: int = Field(default=7, ge=0, le=MAX_DAYS)
    thaw_reminder: Literal["evening_before", "morning_of"] = "evening_before"
    allow_freezer: bool = True
    strategy: Strategy = "fresh"
    budget: float | None = Field(default=None, gt=0, le=MAX_BUDGET, allow_inf_nan=False)
    diet: Diet = "any"
    avoid: list[Avoid] = Field(default_factory=list, max_length=5)

    @model_validator(mode="after")
    def _weekdays(self) -> Prefs:
        if any(not 0 <= d <= 6 for d in self.shop_weekdays):
            raise ValueError("shop_weekdays are 0 (Monday) to 6 (Sunday)")
        return self


class SnapshotLine(BaseModel):
    """One line of a trip as the shopper approved it. Price is kept for the delta only: it
    is not part of the fingerprint."""
    product_id: int
    packs: int | None = Field(default=None, ge=0, le=MAX_PACKS)
    storage: Storage
    price_at_approval: float | None = Field(default=None, ge=0, le=MAX_LINE_PRICE,
                                            allow_inf_nan=False)


class ApprovedTrip(BaseModel):
    date: dt.date
    fingerprint: str = Field(min_length=64, max_length=64)
    strategy: Strategy
    snapshot: list[SnapshotLine] = Field(default_factory=list, max_length=200)


Country = Annotated[str, Field(min_length=1, max_length=100)]
LineNo = Annotated[str, Field(min_length=1, max_length=4)]
ProductId = Annotated[int, Field(ge=1, le=MAX_PRODUCT_ID)]
StoreId = Annotated[int, Field(ge=1, le=MAX_PRODUCT_ID)]
# The shopper's own product for a recipe line: recipe_key -> {line_no: product_id}.
PinMap = dict[Annotated[str, Field(min_length=1, max_length=100)],
              Annotated[dict[LineNo, ProductId], Field(max_length=MAX_PINS_PER_RECIPE)]]

# The meal plan wizard's default origin rules, which the shopper edits in its first stage:
# products evidenced as from the United States are excluded, and the preference is Canada,
# then these allies. A preference only orders; it never leaves a product out, since most
# products have no origin evidence at all.
DEFAULT_EXCLUDE_ORIGIN = ("United States",)
DEFAULT_ALLIES = ("United Kingdom", "France", "Germany", "Italy", "Spain", "Netherlands",
                  "Ireland", "Denmark", "Norway", "Sweden", "Australia", "New Zealand", "Japan",
                  "South Korea", "Mexico")


class PlanSettings(BaseModel):
    """Where the household shops: the stores in range and their prices. Defaults are the
    server's reference point and every store. exclude_origin and preference are the plan's
    origin rules, as /mealplan/resolve takes them: the console resolves with them, and every
    pin and every Options list is checked against them (a product evidenced as from an
    excluded country is held back). address is the shopper's own words for lat/lon (an
    address they searched, or "your current location"): kept with the plan for the console to
    show, and never read here, so changing it alone changes nothing. stores_off are the store
    ids the shopper turned off: no offer of theirs is priced or put on a trip (the catalog,
    the basket, the schedule's trips and the Options lists skip them), while resolve ignores
    them, so turning a store off or on never resolves a recipe again."""
    lat: float | None = Field(default=None, ge=-90, le=90)
    lon: float | None = Field(default=None, ge=-180, le=180)
    max_km: float | None = Field(default=None, ge=0.5, le=100)
    address: str | None = Field(default=None, max_length=200)
    exclude_origin: list[Country] = Field(default_factory=list, max_length=MAX_COUNTRIES)
    preference: list[Country] = Field(default_factory=list, max_length=MAX_COUNTRIES)
    stores_off: list[StoreId] = Field(default_factory=list, max_length=MAX_STORES_OFF)

    @model_validator(mode="after")
    def _known_countries(self) -> PlanSettings:
        from ..origins import validate_countries

        unknown = validate_countries([*self.exclude_origin, *self.preference])
        if unknown:
            raise ValueError("unrecognised country name(s): "
                             + ", ".join(repr(k) for k in unknown))
        return self


class MealPlanDraft(BaseModel):
    """The browser's plan (`pantry.mealplan.v1`), sent whole to /mealplan/schedule.

    pins: recipe_key -> {line_no (as a string): product_id}, the shopper's own product for a
    line, checked on every call as a chat cart's swap is (pins.check_pins). packs_override:
    '<YYYY-MM-DD>:<product_id>' -> packs on that trip's line, 0 for a line the shopper
    dismissed. storage_overrides: product_id (as a string) -> 'fridge' (never
    freeze) or 'freezer' (freeze on arrival). dismissed_dates are never suggested as trips;
    fixed_dates always are. nutrition_targets are the shopper's own daily targets, kept in the
    browser: each day gets a verdict only where its totals prove one."""
    v: int = 1
    id: str = Field(default="", max_length=64)
    rev: int = Field(default=0, ge=0)
    start_date: dt.date
    days: int = Field(default=MAX_DAYS, ge=1, le=MAX_DAYS)
    prefs: Prefs = Field(default_factory=Prefs)
    recipes: dict[str, DraftRecipe] = Field(default_factory=dict, max_length=MAX_RECIPES)
    resolved: dict[str, ResolvedRecipe] = Field(default_factory=dict, max_length=MAX_RECIPES)
    meals: list[Meal] = Field(default_factory=list, max_length=MAX_MEALS)
    pins: PinMap = Field(default_factory=dict, max_length=MAX_RECIPES)
    trips: list[ApprovedTrip] = Field(default_factory=list, max_length=MAX_DAYS)
    dismissed_dates: list[dt.date] = Field(default_factory=list, max_length=MAX_DAYS)
    fixed_dates: list[dt.date] = Field(default_factory=list, max_length=MAX_DAYS)
    packs_override: dict[str, Annotated[int, Field(le=MAX_PACKS)]] = Field(
        default_factory=dict, max_length=200)
    storage_overrides: dict[str, Literal["fridge", "freezer"]] = Field(
        default_factory=dict, max_length=200)
    settings: PlanSettings = Field(default_factory=PlanSettings)
    nutrition_targets: NutritionTargets = Field(default_factory=dict)

    @model_validator(mode="after")
    def _unique_meal_ids(self) -> MealPlanDraft:
        ids = [m.id for m in self.meals]
        if len(ids) != len(set(ids)):
            raise ValueError("meal ids must be unique")
        return self

    def day(self, i: int) -> dt.date:
        return self.start_date + dt.timedelta(days=i)

    def index(self, d: dt.date) -> int:
        return (d - self.start_date).days


# ─── The schedule ────────────────────────────────────────────

class PlacedMeal(BaseModel):
    """A meal where the schedule put it. placed_by: 'you' (it had a date) or 'spread' (the
    schedule placed it; the console stores it back). date None: unplaced."""
    id: str
    recipe_key: str
    title: str
    date: dt.date | None
    slot: Slot
    servings: int
    pinned: bool
    placed_by: Literal["you", "spread"] | None


class ShelfLifeInfo(BaseModel):
    """What a line's storage time rests on. status: cited (rows of shelf_life.json, quoted
    verbatim), your_setting (no cited time; the shopper's buy-ahead limit) or unknown (no cited
    time and none planned: shelf-stable or bought frozen, by the file's own storage class).
    days_planned is the lower bound the plan used, never shown without the verbatim text."""
    status: Literal["cited", "your_setting", "unknown"]
    verbatim: list[str] = Field(default_factory=list)
    rule_ids: list[str] = Field(default_factory=list)
    source: str | None = None
    url: str | None = None
    page_date: str | None = None
    days_planned: int | None = None
    note: str = ""


class LineProduct(BaseModel):
    id: int
    name: str
    unit_size: str
    category: str | None
    demo_product: bool


class MealRef(BaseModel):
    meal_id: str
    recipe_key: str
    title: str
    date: dt.date
    slot: Slot


class TripLine(BaseModel):
    """One product bought on one trip, for one or more meals. packs None: the amount is
    unknown (packs_basis says why), and so is the price. price is the assigned store's offer
    times packs (demo prices). leftover is packs x size minus the need, when both are known;
    leftover_until only when the storage time is cited."""
    product: LineProduct
    category: str | None
    packs: int | None
    packs_basis: Literal["computed", "your_setting", "amount_unknown", "needs_servings"]
    need_qty: float | None
    need_uom: str | None
    leftover_qty: float | None
    leftover_until: dt.date | None
    storage: Storage
    freeze_on_arrival: bool = False
    shelf_life: ShelfLifeInfo
    for_meals: list[MealRef]
    store: str | None
    price: float | None
    price_at_approval: float | None = None
    price_delta: float | None = None
    stocked: bool


class TripDiff(BaseModel):
    """How a trip differs from what the shopper approved: lines added, removed and with
    changed packs or storage, and the same as short text ('+1 Whole Milk 1L')."""
    added: list[dict] = Field(default_factory=list)
    removed: list[dict] = Field(default_factory=list)
    changed: list[dict] = Field(default_factory=list)
    text: list[str] = Field(default_factory=list)


class Trip(BaseModel):
    """One shopping trip. total_cost sums the lines that have a price; total_is_floor is True
    when a line has none, and total_cost is None when the trip has lines and not one of them
    is priced, since an unknown is never shown as $0.00."""
    id: str
    date: dt.date
    status: Literal["suggested", "approved", "needs_review"]
    reason: str
    lines: list[TripLine]
    dismissed: list[LineProduct] = Field(default_factory=list)
    stores: list[str]
    recommended: TripOption | None
    frontier: list[TripOption]
    not_stocked: list[str]
    total_cost: float | None
    total_is_floor: bool
    price_delta: float | None = None
    fingerprint: str
    diff: TripDiff | None = None
    list_text: str


class Action(BaseModel):
    """A dated thing to do: shop, freeze (on the trip date), thaw (move to the fridge) or
    cook. basis: cited (rule_ids name the rows) or your_setting."""
    kind: Literal["shop", "freeze", "thaw", "cook"]
    date: dt.date
    text: str
    trip_id: str | None = None
    meal_id: str | None = None
    product_id: int | None = None
    rule_ids: list[str] = Field(default_factory=list)
    basis: Literal["cited", "your_setting"] | None = None


class PlanWarning(BaseModel):
    """level: must_fix, should_fix, decide or note. strategy None: about the plan whatever the
    strategy. remedies are edits the console can apply (at most 3); the engine never applies
    them."""
    level: Literal["must_fix", "should_fix", "decide", "note"]
    code: str
    message: str
    strategy: Strategy | None = None
    remedies: list[dict] = Field(default_factory=list, max_length=3)
    meal_ids: list[str] = Field(default_factory=list)
    product_id: int | None = None
    trip_date: dt.date | None = None
    recipe_key: str | None = None


class StrategyResult(BaseModel):
    """total_cost sums the trips' known totals, None when no line on any trip is priced."""
    name: Strategy
    recommended: bool
    trips: list[Trip]
    actions: list[Action]
    total_cost: float | None
    total_is_floor: bool
    warning_counts: dict[str, int]


class DayOut(BaseModel):
    """One day of the board. nutrition is what one person eats that day (one serving of each
    meal), None when the nutrition tables are not deployed."""
    date: dt.date
    weekday: str
    meal_ids: list[str]
    nutrition: DayNutrition | None = None


class Coverage(BaseModel):
    """How much of the plan rests on cited facts. Counts are over the distinct products the
    plan buys and over the needs (meal x line) it computes."""
    products: int
    freshness_cited: int
    freshness_your_setting: int
    freshness_unknown: int
    needs: int
    amounts_known: int
    nutrition: Literal["computed", "not_deployed"] = "not_deployed"


class MealSchedule(BaseModel):
    v: Literal[1] = 1
    rev: int
    start_date: dt.date
    meals: list[PlacedMeal]
    unplaced: list[str]
    strategies: list[StrategyResult]
    recommended_strategy: Strategy
    warnings: list[PlanWarning]
    days: list[DayOut]
    period_nutrition: PeriodNutrition | None = None
    # recipe_key -> per-serving nutrition with every line's receipt; days carry the totals
    recipe_nutrition: dict[str, MealNutrition] = Field(default_factory=dict)
    coverage: Coverage
    approved_schedule: dict | None
    sources: list[dict]
    synthetic_notice: str


# ─── Options for a trip line ─────────────────────────────────

class CoveredLine(BaseModel):
    """A recipe line a trip line's purchase is for. Choosing another product pins every one
    of them (pins[recipe_key][line_no]); a pin equal to planner_product_id, the resolve's own
    pick, is the same as no pin."""
    recipe_key: str
    title: str
    line_no: int
    ingredient: str
    planner_product_id: int | None
    pinned_product_id: int | None = None


class TripLineOptions(BaseModel):
    """POST /mealplan/alternatives: the products that could take the place of one trip
    line's product, ranked by alternatives.rank_alternatives for every recipe line the
    purchase covers, under the plan's shopping point and origin rules.

    Each ranked row's trip is the plan re-scheduled with that product pinned, under the same
    strategy and with no LLM: trip.total and trip.delta are the total of the strategy's
    trips (as the Shop band sums them) and its change, trip.stores the stops of the trip
    that buys it for these meals, trip.buys_at the store and pack price there; packs and
    cost_for_need are what that trip line then buys and charges. stocked is False when no
    store in range sells the line's product any more (it is then not among the rows).

    trip_date None: no trip is planned yet (the wizard's Ingredients stage). The ranking is
    for every line of the plan that buys the product, each row's trip is None, its packs are
    what those meals need and cost_for_need those packs at its cheapest offer in range;
    plan_total is None. `lines` still names every covered line, so choosing pins them all."""
    v: Literal[1] = 1
    rev: int
    strategy: Strategy
    trip_date: dt.date | None
    product_id: int
    product: str
    stocked: bool
    pinned: bool
    lines: list[CoveredLine]
    plan_total: float | None
    ranking: AlternativeRanking


# ─── The wizard's catalog and basket ─────────────────────────

class WikiImage(BaseModel):
    """A demo starter's or library recipe's picture: the lead image of this English Wikipedia
    article (the starters file's image_wiki), which the console shows through the demo hub
    (GET /hub/images/wiki?title=), credited to Wikipedia."""
    kind: Literal["wiki"] = "wiki"
    title: str


class UrlImage(BaseModel):
    """A shopper's own recipe's picture: the link to the preview image of the page it was
    imported from (its doc's source.image_url), which the console shows through the hub's
    guarded image proxy, credited to the site."""
    kind: Literal["url"] = "url"
    url: str


RecipeImage = Annotated[WikiImage | UrlImage, Field(discriminator="kind")]


class CatalogLine(BaseModel):
    """One line a recipe buys, for costing a selection by whole packs: the product the
    resolve picks in demo mode, one pack's price at its cheapest store in range (None when no
    store in range sells it, or no product was found), the pack's size and unit, and the
    line's need for one meal for the household in that unit (None when unknown or in another
    unit). held_back: the origin rule left the line without a product."""
    ingredient: str
    product_id: int | None
    pack_price: float | None
    pack_qty: float | None
    pack_uom: str | None
    need_per_meal: float | None
    held_back: bool


class CatalogRecipe(BaseModel):
    """One recipe the Meals stage offers. diets and contains are empty when they are not
    known (a shopper's recipe whose lines do not all have a product): a known recipe that
    satisfies no diet always contains something. est_cost_per_meal is one meal for the
    household, each priced line's share of its pack price; None when no line is priced.
    unavailable names the ingredients with no offer in range; held_back those of them the
    plan's origin exclusion left out (every candidate in range is evidenced as from an excluded
    country). image is its picture (catalog.image), None when it has none."""
    key: str
    ref: RecipeRef
    title: str
    image: RecipeImage | None
    slot: Slot
    diets: list[str]
    contains: list[str]
    servings: int | None
    fits: bool
    why_not: str | None
    est_cost_per_meal: float | None
    lines_total: int
    lines_priced: int
    unavailable: list[str]
    held_back: list[str]
    lines: list[CatalogLine]


class Catalog(BaseModel):
    """POST /mealplan/catalog. est_basis says in one sentence how the estimates are made, and
    how a selection is costed from the recipes' lines: per product, the total need (meals x
    need_per_meal) over pack_qty rounded up, plus one pack for each line whose need or pack
    size is unknown, at pack_price; a line with no pack_price is not bought. /mealplan/basket's
    est_total counts the same way."""
    v: Literal[1] = 1
    recipes: list[CatalogRecipe]
    est_basis: str
    latency_ms: int


class BasketProduct(BaseModel):
    """labels are the product's contains-tags (diets.product_contains)."""
    id: int
    name: str
    size: str
    labels: list[str]
    says_organic: bool
    demo_product: bool


class BasketStore(BaseModel):
    name: str
    distance_km: float | None


class BasketOrigin(BaseModel):
    """status: evidenced (the evidence resolves to a country), conflicting or unknown. claim is
    the evidence in plain words: "From Thailand", "Made in Canada, ingredients may be
    imported", "Origin unclear: sources disagree", "Origin unknown".
    preferred_rank is the place (1 first) of the plan's preference entry it matches."""
    country: str | None
    claim: str
    status: Literal["evidenced", "unknown", "conflicting"]
    preferred_rank: int | None


class BasketLine(BaseModel):
    """One ingredient of one recipe and the product the plan buys for it. need is the
    recipe's amount for all its meals; price is one pack at the store; cost_for_need is this
    recipe's share of what the whole packs of that product cost: by need for the lines whose
    need is known, one pack for a line whose need is not (None when unavailable). shared_with
    names the other recipes buying the same product. unavailable_kind says why an unavailable
    line is: origin_excluded, stores_off (sold in range only at stores the shopper turned
    off), out_of_range or not_stocked (None when it is available, or no product was chosen
    for another reason)."""
    line_no: int
    ingredient: str
    need: str
    product: BasketProduct | None
    store: BasketStore | None
    price: float | None
    cost_for_need: float | None
    origin: BasketOrigin
    pinned: bool
    shared_with: list[str]
    available: bool
    unavailable_kind: Literal["origin_excluded", "stores_off", "not_stocked",
                              "out_of_range"] | None
    why_unavailable: str | None


class BasketRecipe(BaseModel):
    """status is the resolve's (ok, needs_servings, ...) or unresolved; a recipe that buys
    nothing yet lists no lines. image is its picture, as the catalog gives it."""
    key: str
    title: str
    image: RecipeImage | None
    status: str
    meals: int
    lines: list[BasketLine]


class PlanStore(BaseModel):
    """One store for the Ingredients stage's store list. address only where the stores are
    real (STORES_SYNTHETIC off), as the calendar export's policy is; distance_km from the
    plan's shopping point; in_range within its max_km; off when the shopper turned it off;
    sells how many of the basket's products it stocks (None while the plan buys nothing)."""
    id: int
    name: str
    address: str | None
    distance_km: float
    in_range: bool
    off: bool
    sells: int | None


class PlanStores(BaseModel):
    """POST /mealplan/stores: every store, nearest first, and how many products the basket
    buys (products_total: the draft's chosen products, pins included, less any the origin
    exclusion holds back)."""
    v: Literal[1] = 1
    stores: list[PlanStore]
    products_total: int


class Basket(BaseModel):
    """POST /mealplan/basket: every chosen recipe's ingredients, each with its product, store,
    price and origin. est_total is what the whole packs cost, each product bought once for
    every meal at its cheapest store in range, by the catalog's rule (catalog.whole_packs):
    the summed need over the pack size rounded up, and at least one pack (one in all) when a
    line's need or the pack size is unknown. Trips that buy a product on two dates can cost more."""
    v: Literal[1] = 1
    rev: int
    recipes: list[BasketRecipe]
    unavailable_count: int
    est_total: float | None
    budget: float | None
    latency_ms: int

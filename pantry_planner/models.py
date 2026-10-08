"""
Data shapes used across the pipeline. Pydantic for validation on the
API boundary; plain dataclasses inside would also work.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from .nlsearch.plan import StepPhase, StepResult  # import-safe: plan.py is pydantic-only
from .nlsearch.schemas import Constraints  # import-safe: schemas.py is pydantic-only

# ─── Domain models ────────────────────────────────────────────

class RecipeIngredient(BaseModel):
    line_no: int
    name: str
    category: str | None = None  # e.g. "bread", "chocolate", "spread"


class Recipe(BaseModel):
    slug: str
    name: str
    servings: int = 1
    ingredients: list[RecipeIngredient]


class Product(BaseModel):
    id: int
    name: str
    description: str
    price: float                 # CAD
    category: str | None = None
    # 0002_product_attributes (NL2SQL search) — defaults keep old data valid
    subcategory: str | None = None
    dietary_tags: str = ""       # comma list of what it CONTAINS: "dairy,gluten"
    unit_size: str = ""          # display: "450g"
    unit_qty: float | None = None  # canonical amount: 450
    unit_uom: str = ""           # canonical uom: g|ml|each
    # 0003_stores_reviews_terms (query-plan retrieval) — a candidate is a
    # product pinned to its cheapest in-range store offer
    brand: str = ""
    store_name: str = ""
    store_price: float | None = None   # offer price at store_name
    distance_km: float | None = None   # store distance from the request point
    substitute: bool = False           # t4 same-subcategory alternative


class OriginEvidence(BaseModel):
    """One observation about one product from one source.

    A product may carry several of these, and they may disagree — that is
    the point. Nothing here is reconciled; `verbatim` is the exact source
    wording and is never paraphrased.
    """
    product_id: int
    source: str                    # open-food-facts | label-photo | retailer-pdp | guess
    source_ref: str = ""           # barcode, photo filename or URL
    claim_type: str = "unknown"    # product-of | made-in | prepared-in | ...
    verbatim: str = ""
    ingredient_origin: str = ""    # where the inputs came from
    manufactured_in: str = ""      # where it was processed
    confidence: str = "low"        # high | medium | low
    importer_only: bool = False    # an importer address is NOT an origin
    note: str = ""
    observed_at: str = ""


class ProductOrigin(BaseModel):
    """Resolved provenance summary for one product.

    `status` is what ranking keys off:
      "resolved"      — usable evidence
      "conflicting"   — sources disagree; deliberately not resolved
      "unknown"       — a source answered and published nothing
      "lookup_failed" — no source answered; evidence of nothing
      "guess"         — name/brand inference only, never rankable
    """
    product_id: int
    product_name: str = ""
    status: str = "unknown"
    claim_type: str = "unknown"
    country: str = ""              # best single country, display only
    ingredient_origin: str = ""
    manufactured_in: str = ""
    verbatim: str = ""
    confidence: str = "low"
    source: str = ""
    note: str = ""
    evidence_count: int = 0
    # Per-field claim attribution. Fields are merged across evidence rows,
    # so a single claim_type cannot describe both: one source may assert
    # "Product of Italy" about the ingredients while another says only
    # "Made in Canada" about the processing. Ranking reads the claim that
    # belongs to the field it actually matched.
    ingredient_claim: str = ""
    manufactured_claim: str = ""
    # Countries named anywhere in this product's evidence, including rows
    # too weak to resolve (importer addresses, conflicting records). Used
    # to warn rather than to rank — never treated as provenance.
    seen_countries: list[str] = Field(default_factory=list)


class OriginReceipt(BaseModel):
    """The provenance behind one chosen plan line.

    Carried on the line item so a basket can be audited without a second
    lookup: which country, on what claim, from which source, how sure.
    """
    status: str = "unknown"
    country: str = ""
    claim_type: str = ""
    ingredient_origin: str = ""
    manufactured_in: str = ""
    source: str = ""
    confidence: str = ""
    verbatim: str = ""


class OriginCoverage(BaseModel):
    """How much of a basket's provenance is actually known.

    Reported BOTH count-weighted and spend-weighted, because they diverge
    hard: a basket can be 20% covered by count and 4% by spend when the one
    verified line is the cheapest item in it. Spend-weighting is the honest
    denominator for "how much of this purchase did we actually check".

    `meets_floor` is false when coverage is below ORIGIN_MIN_COVERAGE. A
    plan below the floor is not wrong, but it must not be presented as
    clean: with no floor, missing data becomes a competitive advantage,
    because the cheapest candidate is usually the one nobody measured.
    """
    lines_total: int = 0
    lines_known: int = 0
    lines_excluded_origin: int = 0
    count_fraction: float = 0.0
    spend_total: float = 0.0
    spend_known: float = 0.0
    spend_fraction: float = 0.0
    meets_floor: bool = True
    floor: float = 0.0
    note: str = ""


class RankedProduct(BaseModel):
    """A product placed against a caller-supplied country preference."""
    product_id: int
    product_name: str
    price: float = 0.0
    rank: int                      # 0 = most preferred
    tier_label: str
    origin: ProductOrigin
    matched_country: str = ""      # which preference entry it matched
    matched_field: str = ""        # ingredient_origin | manufactured_in


class ExcludedProduct(BaseModel):
    """Positively evidenced as coming from an excluded country."""
    product_id: int
    product_name: str
    price: float = 0.0
    excluded_country: str
    matched_field: str             # which field carried the match
    claim_type: str = ""
    verbatim: str = ""
    confidence: str = "low"


class UnrankedProduct(BaseModel):
    """Held out of the ranking. Never treated as foreign or as domestic."""
    product_id: int
    product_name: str
    price: float = 0.0
    reason: str                    # no_evidence | conflicting | lookup_failed | guess_only
    detail: str = ""


class OriginRanking(BaseModel):
    """Result of ranking a set of products against a preference order.

    The buckets are kept apart on purpose. Merging `unranked` into the
    ranked list would present "nobody published this" as a provenance
    verdict, and with coverage as thin as it is that would be the
    majority of any real catalog.
    """
    preference: list[str] = Field(default_factory=list)
    exclude: list[str] = Field(default_factory=list)
    ranked: list[RankedProduct] = Field(default_factory=list)
    excluded: list[ExcludedProduct] = Field(default_factory=list)
    unranked: list[UnrankedProduct] = Field(default_factory=list)
    counts: dict[str, int] = Field(default_factory=dict)
    coverage_note: str = ""


# ─── Selector I/O ─────────────────────────────────────────────

class Selection(BaseModel):
    """One chosen product for one ingredient."""
    line_no: int
    product_id: int
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str = ""


class SelectorResult(BaseModel):
    """Structured output from a single call to the selector LLM."""
    selections: list[Selection]
    total_cost: float
    model_used: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    cost_usd: float = 0.0
    # One httptrace record per LLM call behind this result (Gemini only): where the time went.
    http: list[dict] = Field(default_factory=list)


class LlmCallTrace(BaseModel):
    """One LLM call's time, phase by phase (httptrace.py). `server_ms` is the provider's own
    processing time as its front end reports it (Google's server-timing header), when it
    does: `waiting for Google` far above it was spent before the request reached Google."""
    step: str
    model: str
    total_ms: int
    attempts: int = 1
    status: int | None = None
    server_ms: int | None = None
    phases: list[StepPhase] = Field(default_factory=list)


# ─── Router I/O ───────────────────────────────────────────────

class PhaseAMetrics(BaseModel):
    ingredient_count: int
    mean_max_similarity: float
    min_max_similarity: float
    count_below_0_3: int
    category_density: float
    # Retrieval-aware signals (NL2SQL path only; has_retrieval gates the
    # extra decision weights so the classic path scores exactly as before)
    has_retrieval: bool = False
    mean_pool_size: float = 0.0        # avg candidates per ingredient
    zero_hit_ingredients: int = 0      # needed t1's form-relaxed match
    value_disagreement: float = 0.0    # pools where cheapest != best unit value


class PhaseBMetrics(BaseModel):
    match_confidence_1_to_10: int = Field(..., ge=1, le=10)
    cost_complexity_1_to_10: int = Field(..., ge=1, le=10)
    ambiguous_ingredients: list[dict] = Field(default_factory=list)
    confidence_in_own_estimate_1_to_10: int = Field(..., ge=1, le=10)
    reasoning: str = ""
    # Metadata about the classifier call itself:
    cost_usd: float = 0.0
    latency_ms: int = 0


class PreselectResult(BaseModel):
    """What preselect_model() returns."""
    model: str
    complexity_score: float | None = None       # 0..1, or None (cascade)
    phase_a: PhaseAMetrics | None = None
    phase_b: PhaseBMetrics | None = None
    routing_cost_usd: float = 0.0               # cost of the router itself
    reason: str = ""                            # human-readable summary


class EscalationDecision(BaseModel):
    """What should_escalate() returns."""
    escalate: bool
    ingredients_to_rerun: list[int] = Field(default_factory=list)  # line_no values
    escalation_model: str = ""
    reason: str = ""


# ─── Split-trip optimizer (4A) ────────────────────────────────

class TripItem(BaseModel):
    """One basket line under a trip option's store assignment."""
    product_id: int
    product_name: str
    store_name: str
    price: float


class TripOption(BaseModel):
    """One point on the stops-vs-cost frontier: shop these stores, pay this."""
    stores: list[str]
    basket_cost: float
    travel_km: float
    travel_cost: float
    total_cost: float                  # basket + travel
    savings_vs_one_stop: float = 0.0   # against the best single-store trip
    recommended: bool = False
    items: list[TripItem] = Field(default_factory=list)  # recommended option only


# ─── Final plan ───────────────────────────────────────────────

# How retrieval matched a line's ingredient to the catalog:
#   exact   — every word of the ingredient as written (the purchase form
#             included); always the case on the classic seeded-recipe path,
#             whose ingredient names are used verbatim
#   form    — only after dropping the purchase form ("powdered tomato" ->
#             tomato products)
#   generic — only after dropping descriptor words (units.DESCRIPTORS:
#             light/dark/toasted/ground/...): "light brown sugar" -> Brown Sugar.
#             On the week path, a line rescued by the head-noun fallback.
MatchLevel = Literal["exact", "form", "generic"]


class DroppedIngredient(BaseModel):
    """An ingredient a partial plan (allow_partial) left out instead of
    aborting: what it was, why, and what to try instead."""
    ingredient: str
    reason: str
    suggestions: list[str] = Field(default_factory=list)


class PlanLineItem(BaseModel):
    """One PURCHASE. Recipe lines that resolve to the same product are one
    purchase: `line_no` is the first of them, `also_lines` the rest,
    `ingredient_name` names every one ("ground Sichuan peppercorn + Sichuan
    peppercorn"), and `packs` is how many packs their summed need takes
    (1 unless the need is known in the pack's unit and one pack is short)."""
    line_no: int
    ingredient_name: str
    product_id: int
    product_name: str
    product_description: str
    price: float                       # the charged price: store offer x packs
    confidence: float
    reasoning: str
    model_used: str
    # Query-plan path: which store the price comes from
    store_name: str = ""
    store_price: float | None = None
    # Provenance of this line, when origin evidence exists for it
    origin: OriginReceipt | None = None
    match: MatchLevel = "exact"        # the loosest level among the lines it covers
    also_lines: list[int] = Field(default_factory=list)
    packs: int = 1
    # The summed need of every line this purchase covers, in canonical units,
    # when every one of them is known in the same unit; None otherwise (an
    # unstated amount, or a teaspoon and a gram on one purchase).
    need_qty: float | None = None
    need_uom: Literal["g", "ml", "each"] | None = None


# ─── Plan basis: what a plan was made from ───────────────────
# Everything needed to re-price or rank a finished plan without re-running the
# parse or the selector: its lines as planned, the products chosen, and the
# constraints and location it was planned under. Read-only and recomputed
# from the plan; whoever holds it (the hub, the browser's meal-plan draft)
# sends it back, and pins are validated against it on the server.

class BasisLine(BaseModel):
    """One planned recipe line: the ingredient as planned (name, form, prep,
    quantity and unit exactly as the parse or the reviewed recipe gave them),
    how it matched the catalog, and the product chosen (None when the
    selector chose nothing valid, or the product had no offer in range)."""
    line_no: int
    name: str
    form: str | None = None
    prep: str | None = None
    quantity: float | None = None
    unit: str | None = None
    level: MatchLevel = "exact"
    product_id: int | None = None
    confidence: float | None = None


class Pin(BaseModel):
    """The shopper's own choice of product for one line."""
    line_no: int
    product_id: int


class PlanBasis(BaseModel):
    """`path`: library (a seeded recipe), nl (pasted text through the
    parser) or spec (reviewed lines, planned with no parse). The left-out
    lists and the interpretation are the plan's own. `origin_dropped` counts
    the products the origin exclusion removed."""
    v: Literal[1] = 1
    path: Literal["library", "nl", "spec"]
    recipe_slug: str
    recipe_name: str
    lines: list[BasisLine]
    constraints: Constraints = Field(default_factory=Constraints)
    lat: float | None = None
    lon: float | None = None
    max_km: float | None = None
    exclude_origin: list[str] = Field(default_factory=list)
    preference: list[str] = Field(default_factory=list)
    origin_requested: bool = False
    origin_dropped: int = 0
    interpretation: list[str] = Field(default_factory=list)
    not_stocked: list[DroppedIngredient] = Field(default_factory=list)
    out_of_range: list[DroppedIngredient] = Field(default_factory=list)
    skipped: list[DroppedIngredient] = Field(default_factory=list)
    ingredient_count: int = 0
    pins: list[Pin] = Field(default_factory=list)


class ShoppingPlan(BaseModel):
    recipe_slug: str
    recipe_name: str
    line_items: list[PlanLineItem]
    total_cost: float
    routing_strategy: str
    preselected_model: str
    escalated: bool
    total_llm_cost_usd: float
    total_latency_ms: int
    # NL2SQL path extras (empty on the classic /plan/{slug} path)
    interpretation: list[str] = Field(default_factory=list)
    plan_trace: list[StepResult] = Field(default_factory=list)
    candidate_count: int = 0
    # Every LLM call the plan made, phase by phase (both paths; empty in demo mode).
    llm_calls: list[LlmCallTrace] = Field(default_factory=list)
    # The Burr run that traced this plan, step by step (its app id in the Burr UI), and how long
    # each of its steps took: [{step, ms, error}], in order.
    burr_run: str = ""
    pipeline: list[dict] = Field(default_factory=list)
    # Split-trip optimizer: stops-vs-cost frontier for the chosen basket
    trip_options: list[TripOption] = Field(default_factory=list)
    # Provenance of the basket as a whole. Only computed when the caller
    # asked an origin question (exclude or preference); otherwise None, so a
    # plan nobody asked about origin for is not stamped UNVERIFIED.
    origin_coverage: OriginCoverage | None = None
    # not_requested | verified | unverified — the one field to read before
    # describing a basket as clean. "unverified" means coverage is below the
    # floor, not that anything excluded shipped.
    origin_status: str = "not_requested"
    # Partial plans (allow_partial=True): ingredients left out instead of
    # aborting. not_stocked = the catalog has no match at all (t1, NL path);
    # out_of_range = stocked, but no offer within the distance/price/diet
    # constraints (t2), or every candidate is evidenced as coming from an
    # excluded country (either path; its suggestions name the removed
    # products and the alternatives left). skipped = never planned, whatever allow_partial says:
    # water and ice (never bought), lines past the 40-ingredient cap, and
    # lines the selector returned no valid product for. total_cost,
    # origin_coverage and trip_options cover the planned line_items only.
    # ingredient_count is how many ingredients the recipe asked for, before
    # anything was dropped: the lines covered by line_items (line_no plus
    # also_lines) + not_stocked + out_of_range + skipped == ingredient_count.
    not_stocked: list[DroppedIngredient] = Field(default_factory=list)
    out_of_range: list[DroppedIngredient] = Field(default_factory=list)
    skipped: list[DroppedIngredient] = Field(default_factory=list)
    ingredient_count: int = 0
    # How many the recipe serves, when it says; None when it does not (the
    # planner still plans one batch, but never claims it serves 1).
    servings: int | None = None
    # What the plan was made from (see PlanBasis); set on every plan.
    basis: PlanBasis | None = None


# ─── Recipe documents: one schema for every recipe the shopper reviews ──
# A library recipe, a demo starter, a pasted or imported ingredient list and
# a dish the assistant wrote all reach the planner as a RecipeDoc. Its lines
# are what the shopper saw and confirmed; recipe_doc.to_spec plans them as
# given, with no parse. Only ingredient lines and a link back are kept: a
# recipe's method text is never stored or shown. Mirrored in the frontend's
# types.ts.

MAX_DOC_LINES = 60
# The bounds a RecipeDoc is validated against. POST /recipes/parse-lines keeps
# its output inside them, so what it returns can be planned as it stands.
MAX_LINE_NAME = 200
MAX_SERVINGS = 100

AmountBasis = Literal["stated_by_source", "demo_house_amounts", "parsed_from_your_paste",
                      "transcribed_confirmed_by_you", "written_by_assistant"]


class LineEvidence(BaseModel):
    """Where a line's amount came from: a verbatim quote, a video timestamp."""
    quote: str | None = Field(default=None, max_length=300)
    at: str | None = Field(default=None, pattern=r"^\d{1,3}:\d{2}$")   # mm:ss


class RecipeLine(BaseModel):
    """One ingredient line. `text` is the line as the source wrote it; `name`,
    `quantity` and `unit` are what gets planned. quantity None means the
    source did not say; unit "" means no unit. `confirmed` is False only for
    a line transcribed from a video until the shopper ticks it."""
    line_no: int = Field(ge=1)
    text: str = Field(max_length=300)
    name: str = Field(min_length=1, max_length=MAX_LINE_NAME)
    quantity: float | None = Field(default=None, ge=0)
    unit: str = Field(default="", max_length=40)
    note: str = Field(default="", max_length=300)
    evidence: LineEvidence | None = None
    confirmed: bool = True
    amount_basis: AmountBasis


class RecipeSource(BaseModel):
    """Where the recipe came from, for the link back and the labels."""
    kind: Literal["library", "starter", "pasted", "web", "youtube", "assistant"]
    method: Literal["db", "seed", "paste", "jsonld", "microdata", "youtube_description",
                    "youtube_linked_page", "gemini_video", "agent_written"]
    url: str | None = Field(default=None, max_length=2000)
    site: str | None = Field(default=None, max_length=200)
    page_title: str | None = Field(default=None, max_length=300)
    author: str | None = Field(default=None, max_length=200)
    channel: str | None = Field(default=None, max_length=200)
    retrieved_at: str | None = Field(default=None, max_length=40)
    extractor: str | None = Field(default=None, max_length=100)
    model: str | None = Field(default=None, max_length=100)
    label: str | None = Field(default=None, max_length=100)


class RecipeDoc(BaseModel):
    """A recipe as the shopper reviews it. `key` names it: 'lib:<slug>',
    'starter:<key>', 'my:<uuid>', 'imp:<n>' (chat import) or 'asst:<n>'.
    servings None means not stated, never silently 1; servings_basis says
    whose number it is ('source', or 'your_setting' once the shopper
    answered). Lines are numbered 1..n in order (a client that removes a
    line renumbers the rest), so a plan's line_no is the doc's line_no."""
    v: Literal[1] = 1
    key: str = Field(min_length=1, max_length=100)
    title: str = Field(min_length=1, max_length=200)
    servings: int | None = Field(default=None, ge=1, le=MAX_SERVINGS)
    servings_stated: bool = False
    servings_basis: Literal["source", "your_setting"] | None = None
    yield_text: str = Field(default="", max_length=200)
    lines: list[RecipeLine] = Field(default_factory=list, max_length=MAX_DOC_LINES)
    source: RecipeSource
    warnings: list[str] = Field(default_factory=list, max_length=20)

    @model_validator(mode="after")
    def _numbered_in_order(self) -> RecipeDoc:
        got = [ln.line_no for ln in self.lines]
        if got != list(range(1, len(got) + 1)):
            raise ValueError(f"lines must be numbered 1..{len(got)} in order, got {got}")
        return self


# ─── Weekly menu optimizer (5A) ───────────────────────────────

class WeekItem(BaseModel):
    """One line of the merged week shopping list. A product shared by
    several dinners appears ONCE, with every recipe that uses it."""
    product_id: int
    product_name: str
    store_name: str
    price: float
    used_by: list[str]                 # recipe names
    origin: OriginReceipt | None = None


class DayPlan(BaseModel):
    recipe_slug: str
    recipe_name: str
    line_items: list[PlanLineItem]
    day_cost: float                    # this dinner priced standalone


class WeekPlan(BaseModel):
    days: list[DayPlan]
    shopping_list: list[WeekItem]      # merged; shared products counted once
    total_cost: float                  # merged basket total
    standalone_cost: float             # sum of day costs (no sharing)
    overlap_savings: float             # standalone - merged
    budget: float | None = None
    notes: list[str] = Field(default_factory=list)
    plan_trace: list[StepResult] = Field(default_factory=list)
    trip_options: list[TripOption] = Field(default_factory=list)
    origin_coverage: OriginCoverage | None = None
    origin_status: str = "not_requested"
    total_llm_cost_usd: float = 0.0

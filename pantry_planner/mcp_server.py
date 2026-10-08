"""
MCP server — exposes the pantry pipeline to MCP clients (Claude
Desktop, Claude Code, any agent) as tools, resources and prompts.

One server definition, two transports:
  stdio  — the `pantry-mcp` console script (see [project.scripts]);
           what Claude Desktop / `claude mcp add` launch locally.
  HTTP   — `http_app()` is mounted into the FastAPI app (api.py), so
           the deployed instance serves Streamable HTTP at /mcp.

Tools call the service layer in-process (flow.py, db.py, origins.py) —
no HTTP hop — and return the existing Pydantic models, which the MCP
SDK converts into structured output schemas automatically.

stdio discipline: nothing in this module may print() to stdout — the
JSON-RPC stream lives there. The SDK logs to stderr.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Annotated

from mcp.server.auth.middleware.auth_context import get_access_token
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ResourceNotFoundError, ToolError
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field, model_serializer

from .llm import LLMError
from .models import (
    MAX_DOC_LINES,
    AmountBasis,
    DroppedIngredient,
    LlmCallTrace,
    MatchLevel,
    OriginCoverage,
    OriginRanking,
    PlanBasis,
    PlanLineItem,
    Product,
    ProductOrigin,
    Recipe,
    RecipeLine,
    RecipeSource,
    ShoppingPlan,
    TripOption,
    WeekPlan,
)

# Tool annotations are hints for the client, not security: a read tool
# never writes, and nothing here reaches outside the seeded catalog, so
# open_world is False everywhere. Plan tools are read-only too but not
# idempotent — the LLM selector can pick differently between calls.
_READ = ToolAnnotations(read_only_hint=True, open_world_hint=False)
_PLAN = ToolAnnotations(read_only_hint=True, open_world_hint=False,
                        idempotent_hint=False)
# Write tools: they add to a review queue or copy a reviewed claim into
# evidence; nothing is ever deleted. Submitting is idempotent (the queue
# dedupes the same reading), reviewing is not (a second review errors).
_SUBMIT = ToolAnnotations(read_only_hint=False, destructive_hint=False,
                          open_world_hint=False, idempotent_hint=True)
_REVIEW = ToolAnnotations(read_only_hint=False, destructive_hint=False,
                          open_world_hint=False, idempotent_hint=False)

# REST-parity input bounds (api.py): the MCP endpoint is as public as the
# REST one, so a paste, a list or a page size is capped the same way.
MAX_TEXT = 8000
MAX_LIST = 50
MAX_SEARCH = 200
MAX_IDS = 200
# Every value origins.resolve_origin can stamp on ProductOrigin.status.
_ORIGIN_STATUSES = {"resolved", "conflicting", "unknown", "lookup_failed", "guess"}

server = MCPServer(
    name="pantry-planner",
    version="0.1.0",
    instructions=(
        "Grocery planning over a seeded Canadian store catalog. Browse "
        "recipes and products for free; plan_recipe / plan_from_text run "
        "the full LLM pipeline (slow, costs API credits) and return a "
        "priced shopping plan. Provenance tools rank products by where "
        "they come from against a country preference you supply, using "
        "ingested evidence only - never a guess. Origin coverage is "
        "partial, so always report how many products were unranked. "
        "Resources: pantry://countries lists the country spellings the "
        "server accepts, pantry://origins/coverage says how thin the "
        "evidence is, pantry://recipes and pantry://catalog/categories "
        "are the library and the aisle list."
    ),
)


# ─── Slim response shapes (token-lean summaries) ─────────────

class RecipeSummary(BaseModel):
    slug: str
    name: str
    servings: int
    ingredient_count: int


class ProductSummary(BaseModel):
    id: int
    name: str
    brand: str
    category: str | None
    subcategory: str | None
    price: float
    unit_size: str
    dietary_tags: str


class PipelineStatus(BaseModel):
    status: str
    routing_strategy: str
    default_model: str
    escalation_model: str
    confidence_threshold: float
    db: str
    anthropic_key_configured: bool
    demo_mode: bool
    mcp_auth: str               # "required" (tokens configured) | "anonymous"
    write_tools: str            # "enabled" | "disabled" for THIS caller
    # Added with Gemini support: the other two model specs ("gemini:<model>"
    # or an Anthropic name) and whether a Gemini key is present.
    classifier_model: str
    nl2sql_model: str
    gemini_key_configured: bool


class TriageCandidate(BaseModel):
    """A product worth photographing next. A hint, never an origin."""
    product_id: int
    product_name: str
    reason: str
    status: str


class Offer(BaseModel):
    store: str
    price: float
    distance_km: float


class ProductMatch(BaseModel):
    """One lookup hit, priced the way the planner prices a candidate: the
    cheapest in-range store offer, not the catalog list price."""
    id: int
    name: str
    brand: str
    size: str
    category: str | None
    subcategory: str | None
    store: str
    price: float
    distance_km: float
    origin_status: str          # from the resolved evidence; "unknown" is common
    origin_country: str         # "" when nothing is resolved


class ProductSearch(BaseModel):
    query: str
    tokens: list[str]
    match: str                  # "direct" | "generic" | "relaxed" | "none"
    total: int                  # hits before `limit` trimmed the list
    items: list[ProductMatch]
    note: str


class EvidenceSummary(BaseModel):
    source: str
    source_ref: str
    claim_type: str
    ingredient_origin: str
    manufactured_in: str
    verbatim: str
    confidence: str
    importer_only: bool
    observed_at: str


class ProductDetail(BaseModel):
    id: int
    name: str
    brand: str
    description: str
    category: str | None
    subcategory: str | None
    dietary_tags: str
    unit_size: str
    unit_qty: float | None
    unit_uom: str
    list_price: float
    offers: list[Offer]                 # every store, cheapest first
    origin: ProductOrigin               # read `status` before `country`
    evidence: list[EvidenceSummary]     # every row, including ones the resolver ignored
    # Origin readings awaiting review; null when the review-queue table is not
    # deployed yet (the API can roll before pantry-db's 0006 migration).
    pending_submissions: int | None
    review_count: int
    avg_rating: float | None


class Submission(BaseModel):
    """One origin reading in the review queue. `status` is "pending" until
    a reviewer acts; only "approved" has become evidence (`evidence_id`)."""
    id: int
    product_id: int
    product_name: str
    status: str
    claim_type: str
    country: str
    ingredient_origin: str
    manufactured_in: str
    verbatim: str
    confidence: str
    importer_only: bool
    note: str
    source_ref: str
    submitted_by: str
    submitted_at: str
    reviewed_by: str
    reviewed_at: str
    review_note: str
    evidence_id: int | None
    duplicate: bool = False     # True when this call inserted nothing


class SubmissionPage(BaseModel):
    items: list[Submission]
    total: int
    next_offset: int | None


class ProductPage(BaseModel):
    items: list[ProductSummary]
    total: int                  # matches before the page was cut
    next_offset: int | None     # None on the last page


class OriginPage(BaseModel):
    items: list[ProductOrigin]
    total: int                  # rows matching ids/search AND status
    by_status: dict[str, int]   # over the ids/search set BEFORE the status filter
    next_offset: int | None


# ─── Token-lean plan results ──────────────────────────────────
# A plan's full shape (ShoppingPlan / WeekPlan) carries the retrieval
# trace, every trip option and the selector's reasoning per line: 3-19k
# chars on the demo seed, almost all of it debugging context an agent
# never reads. The summary keeps what a shopper acts on — the lines, the
# price, the provenance verdict — and `full` is attached only on request.

class LeanLine(BaseModel):
    """One purchase. `match` says how the ingredient met the catalog:
    "exact" — every word as written; "form" — only once the purchase form
    was swapped for its equivalent ("cumin powder" -> Cumin Ground 100g) or
    dropped ("powdered tomato" -> tomato products); "generic" — only once
    descriptor words were dropped too ("light brown sugar" -> Brown Sugar
    1kg; the vocabulary is fixed: light, dark, toasted, roasted, ground,
    whole, dried, fresh, frozen, boneless, skinless, bone-in, skin-on,
    large, small, medium, extra, chopped, sliced, minced, diced, raw,
    organic, smoked, unsalted, low/reduced sodium). Every generic line is
    also named in the summary's notes. `line_no` is the ingredient's
    position in the recipe as written. When several recipe lines chose the
    same product it is bought ONCE: `also_lines` lists the other line
    numbers, `ingredient` names every one of them ("ground Sichuan
    peppercorn + Sichuan peppercorn"), and `packs` is how many packs their
    summed need takes (`price` is for all of them); notes say so."""
    line_no: int
    ingredient: str
    product_id: int
    product: str
    brand: str
    size: str
    store: str
    price: float
    confidence: float
    origin_country: str         # "" unless the line's origin resolved
    origin_status: str          # the receipt status, or "none" when there is no receipt
    match: MatchLevel
    also_lines: list[int] = Field(default_factory=list)
    packs: int = 1


class PlanLine(LeanLine):
    """A recipe plan's line, with where the recommended `trip` buys it and its price there:
    the store to send the shopper to ("" and None when the plan has no trip). `store` and
    `price` are the line's cheapest offer in range, which the trip may skip to save a stop."""
    trip_store: str = ""
    trip_price: float | None = None


class Coverage(BaseModel):
    spend_fraction: float       # the binding measure: what the money did
    count_fraction: float
    meets_floor: bool
    lines_known: int
    lines_total: int
    lines_excluded_origin: int


class TripLine(BaseModel):
    """One basket line at the store the trip buys it from; the product's
    name is on the matching `lines` / `shopping_list` entry."""
    product_id: int
    store: str
    price: float


class Trip(BaseModel):
    """The recommended point on the stops-vs-cost frontier, priced as ITS
    OWN basket: `items` are the per-line prices at the stores the trip
    visits, `basket_cost` their sum, `total_cost` basket plus travel. The
    summary's `lines` and `total_cost` price every line at its cheapest
    in-range store regardless of how many stops that implies — a different
    basket, so the two totals legitimately differ. Report the one you mean."""
    stores: list[str]
    basket_cost: float
    travel_cost: float
    total_cost: float
    savings_vs_one_stop: float
    items: list[TripLine]


class PlanSummary(BaseModel):
    """`total_cost` is the sum of `lines`, each at its cheapest in-range
    store; `trip` is one realistic shopping trip priced at its own stores
    (see Trip). They answer different questions and need not agree.

    Both describe the PLANNED lines only, and so does `coverage`. A partial
    plan (plan_from_text with allow_partial=true) lists what it left out:
    `not_stocked` — the catalog has nothing matching the ingredient at all;
    `out_of_range` — it is stocked, but no offer passes the distance
    (lat/lon/max_km), price or diet constraints, and `reason` names the
    nearest offer outside them and the limit it breaks ("beyond the 5 km
    limit", "over the $5.00 per-item price cap"). `skipped` is never
    planned, with or without allow_partial: water and ice (never bought),
    lines past the 40-ingredient cap, and lines the selector returned no
    product for. Each entry is {ingredient, reason, suggestions}. All three
    lists are always present and empty when nothing was left out; when
    something was, `notes` says "planned K of N ingredients". Every recipe
    ingredient is in exactly one place: a line (by `line_no` or
    `also_lines`), not_stocked, out_of_range or skipped."""
    recipe_slug: str
    recipe_name: str
    total_cost: float           # planned lines only
    origin_status: str          # not_requested | verified | unverified
    coverage: Coverage | None   # only when an origin question was asked; planned lines only
    lines: list[PlanLine]
    trip: Trip | None           # None when the planner produced no trip options
    notes: list[str]            # interpretation, planned K of N, generic matches,
                                # substitutions, coverage warnings
    not_stocked: list[DroppedIngredient]
    out_of_range: list[DroppedIngredient]
    skipped: list[DroppedIngredient]
    llm_cost_usd: float
    latency_ms: int
    # Where the plan's LLM time went, call by call and phase by phase (connect, TLS, waiting
    # for Google, ...). For people and trace views: an agent can ignore it.
    llm_calls: list[LlmCallTrace] = Field(default_factory=list)
    # The Burr run that traced this plan, step by step (its app id in the Burr UI), and each of
    # its steps' time in ms, in order ({step: ms}; the full plan has errors too). For people and
    # trace views: an agent can ignore them.
    burr_run: str = ""
    pipeline: dict[str, float] = Field(default_factory=dict)
    # Only with basis=true: what the plan was made from, for re-pricing and
    # ranking alternatives without re-planning. Absent from the result
    # otherwise, so a summary without it is byte for byte what it was.
    basis: PlanBasis | None = None

    @model_serializer(mode="wrap")
    def _omit_absent_basis(self, handler):
        data = handler(self)
        if self.basis is None:
            data.pop("basis", None)
        return data


class PlanResult(BaseModel):
    summary: PlanSummary
    full: ShoppingPlan | None = None    # populated only with verbose=True


class WeekDay(BaseModel):
    recipe_slug: str
    recipe_name: str
    day_cost: float
    lines: list[LeanLine]


class WeekListItem(BaseModel):
    product_id: int
    product: str
    store: str
    price: float
    used_by: list[str]          # recipe names sharing this product
    origin_country: str
    origin_status: str


class WeekSummary(BaseModel):
    days: list[WeekDay]
    shopping_list: list[WeekListItem]   # merged; a shared product appears once
    total_cost: float
    standalone_cost: float
    overlap_savings: float
    budget: float | None
    origin_status: str
    coverage: Coverage | None
    trip: Trip | None
    notes: list[str]


class WeekResult(BaseModel):
    summary: WeekSummary
    full: WeekPlan | None = None        # populated only with verbose=True


def _check_countries(*lists) -> None:
    """Unknown country names are an error the agent must see, not a silent
    no-op filter that reports success. Suggests the closest spellings."""
    from .origins import validate_countries

    names = [n for lst in lists for n in (lst or [])]
    unknown = validate_countries(names)
    if unknown:
        parts = [f"{k!r} (did you mean: {', '.join(v) or 'no close match'})"
                 for k, v in unknown.items()]
        raise ToolError("Unrecognised country name(s): " + "; ".join(parts)
                        + ". Use a country name or common alias such as "
                          "'United States', 'USA' or 'Canada'.")


def _principal(ctx: Context | None) -> tuple[str, str]:
    """(label, transport) for the caller of the current tool.

    No request object means the tool is running in the operator's own
    process — stdio, or an in-process call — and is trusted as "local".
    Over HTTP the label is the bearer token's, or "anonymous" when the
    endpoint runs without MCP_AUTH_TOKENS. Headers are never read here:
    they are client input, not an identity assertion.
    """
    try:
        req = ctx.request_context.request if ctx is not None else None
    except ValueError:          # "Context is not available outside of a request"
        req = None
    if req is None:
        return ("local", "stdio")
    tok = get_access_token()
    return (tok.client_id, "http") if tok is not None else ("anonymous", "http")


def _require_write(ctx: Context | None) -> str:
    """The label to record as submitter/reviewer, or a ToolError when the
    caller is anonymous over HTTP: an open endpoint must not be able to
    fill the review queue."""
    label, transport = _principal(ctx)
    if transport == "http" and label == "anonymous":
        raise ToolError(
            "Submissions are disabled on this endpoint: the operator has not "
            "configured MCP_AUTH_TOKENS. Run the server over stdio, or ask the "
            "operator for a token.")
    return label


def _affected(alert) -> str:
    """Each ingredient a gate names, with what the planner found to buy instead (the removed
    products, the alternatives still available), so an agent can offer the shopper the trade
    rather than only report the failure."""
    if not alert or not alert.details:
        return ""
    parts = []
    for d in alert.details:
        options = [str(o) for o in d.get("suggestions") or []]
        parts.append(str(d.get("name", "?"))
                     + (f" (options: {'; '.join(options)})" if options else ""))
    return f" Affected: {', '.join(parts)}."


# Gates a partial plan gets past: allow_partial plans the rest and reports these ingredients.
PARTIAL_CODES = ("missing_ingredients", "unavailable_within_constraints", "excluded_by_origin")


def _partial_hint(alert, allow_partial: bool) -> str:
    """Only when a partial plan would price something: with every ingredient missing (or out
    of range, or excluded) the retry is a second paid parse certain to fail the same way."""
    if (not allow_partial and alert is not None and alert.code.value in PARTIAL_CODES
            and (alert.partial_would_plan or 0) > 0):
        return " Retry with allow_partial=true to plan the rest."
    return ""


def _gate_message(e, allow_partial: bool = False) -> str:
    """One shape for every gate abort an agent can hit."""
    alert = e.execution.aborted
    steps = ", ".join(f"{s.step_id}:{s.outcome}" for s in e.execution.steps)
    code = alert.code.value if alert else "unknown"
    msg = alert.message if alert else "plan aborted"
    return (f"Plan aborted before product selection — {code}: {msg}{_affected(alert)}"
            f"{_partial_hint(alert, allow_partial)}"
            + (f" Steps: {steps}." if steps else ""))


def _llm_message(e: LLMError) -> str:
    """An LLM failure as a tool error. The SDK masks any other exception as
    "Error executing tool", which would hide the one line that says what to
    fix (the key, the model spec, the quota)."""
    return f"LLM call failed — {e}"


def _product_summary(p) -> ProductSummary:
    return ProductSummary(
        id=p.id, name=p.name, brand=p.brand, category=p.category,
        subcategory=p.subcategory, price=p.price, unit_size=p.unit_size,
        dietary_tags=p.dietary_tags)


def _location(lat: float | None, lon: float | None) -> tuple[float, float]:
    """The request point, defaulting to the planner's own default."""
    from .config import settings

    cfg = settings()
    return (cfg.default_lat if lat is None else lat,
            cfg.default_lon if lon is None else lon)


def _planner_candidates(name: str, lat: float, lon: float, *,
                        generic: bool = False) -> list:
    """Exactly how the planner finds candidates for one ingredient
    (flow._ingredient_pools): token-AND over the product_terms index, each
    product pinned to its cheapest in-range store, cheapest first, uncapped.
    A hit here is a candidate there; a miss here is a miss there.
    `generic` matches at the planner's generic level (descriptors dropped)."""
    from sqlalchemy import text
    from sqlalchemy.orm import Session

    from . import db
    from .nlsearch.planner import _row_to_product
    from .nlsearch.schemas import Constraints, IngredientSpec
    from .nlsearch.sql_builder import build_options_sql

    sql, params = build_options_sql(
        Constraints(), [IngredientSpec(name=name)], relaxed=set(),
        lat=lat, lon=lon, per_ingredient_limit=10_000,
        generic={0} if generic else None)
    with Session(db.engine()) as s:
        return [_row_to_product(r) for r in s.execute(text(sql), params).mappings()]


def _page(total: int, offset: int, page_len: int) -> int | None:
    """Offset of the next page, or None when this one was the last."""
    end = offset + page_len
    return end if end < total else None


def _products_by_id(ids: set[int]) -> dict[int, Product]:
    """Catalog rows for the products a plan chose (brand and pack size are
    not carried on a plan line)."""
    from . import db

    if not ids:
        return {}
    return {p.id: p for p in db.load_all_products() if p.id in ids}


def _lean_line(li: PlanLineItem, products: dict[int, Product]) -> LeanLine:
    p = products.get(li.product_id)
    o = li.origin
    return LeanLine(
        line_no=li.line_no, ingredient=li.ingredient_name,
        product_id=li.product_id, product=li.product_name,
        brand=p.brand if p else "", size=p.unit_size if p else "",
        store=li.store_name, price=li.price, confidence=li.confidence,
        origin_country=o.country if o else "",
        origin_status=o.status if o else "none", match=li.match,
        also_lines=list(li.also_lines), packs=li.packs)


def _coverage(c: OriginCoverage | None) -> Coverage | None:
    if c is None:
        return None
    return Coverage(spend_fraction=c.spend_fraction, count_fraction=c.count_fraction,
                    meets_floor=c.meets_floor, lines_known=c.lines_known,
                    lines_total=c.lines_total,
                    lines_excluded_origin=c.lines_excluded_origin)


def _trip(options: list[TripOption]) -> Trip | None:
    best = next((t for t in options if t.recommended), None)
    if best is None:
        return None
    return Trip(stores=best.stores, basket_cost=best.basket_cost,
                travel_cost=best.travel_cost, total_cost=best.total_cost,
                savings_vs_one_stop=best.savings_vs_one_stop,
                items=[TripLine(product_id=i.product_id, store=i.store_name, price=i.price)
                       for i in best.items])


def _substitution_notes(lines: list[PlanLineItem], prefix: str = "") -> list[str]:
    """One note per line the selector itself flagged as a substitution."""
    return [f"{prefix}line {li.line_no} ({li.ingredient_name}): substitution — "
            f"{li.product_name}"
            for li in lines if "substitut" in li.reasoning.lower()]


def _generic_notes(lines: list[PlanLineItem], prefix: str = "") -> list[str]:
    """Name every line that matched only after dropping descriptor words, so
    a generic match is never reported as if it were the ingredient asked."""
    return [f"{prefix}{li.ingredient_name} matched generically: {li.product_name}"
            for li in lines if li.match == "generic"]


def _planned_count(lines: list[PlanLineItem]) -> int:
    """Recipe lines the purchases cover — a shared purchase covers several."""
    return sum(1 + len(li.also_lines) for li in lines)


def _partial_note(plan: ShoppingPlan) -> list[str]:
    left_out = len(plan.not_stocked) + len(plan.out_of_range) + len(plan.skipped)
    if not left_out:
        return []
    return [f"planned {_planned_count(plan.line_items)} of {plan.ingredient_count} "
            f"ingredients: {len(plan.not_stocked)} not stocked, "
            f"{len(plan.out_of_range)} out of range, {len(plan.skipped)} skipped "
            "(see not_stocked / out_of_range / skipped); total_cost covers the "
            "planned lines only"]


def _shared_notes(lines: list[PlanLineItem]) -> list[str]:
    """Name every purchase that covers more than one recipe line."""
    out = []
    for li in lines:
        if not li.also_lines:
            continue
        nums = ", ".join(str(n) for n in [li.line_no, *li.also_lines])
        out.append(f"lines {nums} ({li.ingredient_name}) share one purchase: "
                   f"{li.product_name}"
                   + (f", {li.packs} packs for their combined quantity"
                      if li.packs > 1 else ", bought once"))
    return out


def _floor_note(c: OriginCoverage | None) -> list[str]:
    if c is None or c.meets_floor:
        return []
    return [f"coverage below floor: origin known for {c.lines_known} of "
            f"{c.lines_total} lines ({c.spend_fraction:.0%} of spend, floor "
            f"{c.floor:.0%}); unknown is not foreign, but do not call this "
            "basket clean"]


def _summarize_plan(plan: ShoppingPlan, basis: bool = False) -> PlanSummary:
    products = _products_by_id({li.product_id for li in plan.line_items})
    notes = (list(plan.interpretation)
             + _partial_note(plan)
             + _shared_notes(plan.line_items)
             + _generic_notes(plan.line_items)
             + _substitution_notes(plan.line_items)
             + _floor_note(plan.origin_coverage))
    trip = _trip(plan.trip_options)
    on_trip = {i.product_id: i for i in trip.items} if trip else {}
    lines = []
    for li in plan.line_items:
        stop = on_trip.get(li.product_id)
        lines.append(PlanLine(**_lean_line(li, products).model_dump(),
                              trip_store=stop.store if stop else "",
                              trip_price=stop.price if stop else None))
    return PlanSummary(
        recipe_slug=plan.recipe_slug, recipe_name=plan.recipe_name,
        total_cost=plan.total_cost, origin_status=plan.origin_status,
        coverage=_coverage(plan.origin_coverage),
        lines=lines, trip=trip, notes=notes,
        not_stocked=list(plan.not_stocked), out_of_range=list(plan.out_of_range),
        skipped=list(plan.skipped),
        llm_cost_usd=plan.total_llm_cost_usd, latency_ms=plan.total_latency_ms,
        llm_calls=plan.llm_calls, burr_run=plan.burr_run,
        pipeline={s["step"]: s["ms"] for s in plan.pipeline if s.get("ms") is not None},
        basis=plan.basis if basis else None)


def _summarize_week(plan: WeekPlan) -> WeekSummary:
    products = _products_by_id({li.product_id for d in plan.days for li in d.line_items})
    notes = list(plan.notes)
    for d in plan.days:
        notes += _generic_notes(d.line_items, prefix=f"{d.recipe_name}: ")
        notes += _substitution_notes(d.line_items, prefix=f"{d.recipe_name}: ")
    notes += _floor_note(plan.origin_coverage)
    return WeekSummary(
        days=[WeekDay(recipe_slug=d.recipe_slug, recipe_name=d.recipe_name,
                      day_cost=d.day_cost,
                      lines=[_lean_line(li, products) for li in d.line_items])
              for d in plan.days],
        shopping_list=[
            WeekListItem(product_id=w.product_id, product=w.product_name,
                         store=w.store_name, price=w.price, used_by=w.used_by,
                         origin_country=w.origin.country if w.origin else "",
                         origin_status=w.origin.status if w.origin else "none")
            for w in plan.shopping_list],
        total_cost=plan.total_cost, standalone_cost=plan.standalone_cost,
        overlap_savings=plan.overlap_savings, budget=plan.budget,
        origin_status=plan.origin_status, coverage=_coverage(plan.origin_coverage),
        trip=_trip(plan.trip_options), notes=notes)


# ─── Catalog / recipe tools (free — DB reads only) ───────────

@server.tool(title="List recipes", annotations=_READ)
def list_recipes() -> list[RecipeSummary]:
    """List all seeded recipes (slug, name, servings, ingredient count).
    Use the slug with get_recipe or plan_recipe. Free — no LLM calls."""
    from sqlalchemy import func
    from sqlalchemy.orm import Session

    from .db import RecipeIngredientRow, RecipeRow, engine

    with Session(engine()) as s:
        counts: dict[str, int] = {
            slug: n
            for slug, n in s.query(RecipeIngredientRow.recipe_slug,
                                   func.count(RecipeIngredientRow.line_no))
            .group_by(RecipeIngredientRow.recipe_slug)
            .all()
        }
        rows = s.query(RecipeRow).all()
        # str()/int() casts: db.py uses legacy Column declarations, so
        # attributes type as Column[...] under mypy.
        return [
            RecipeSummary(slug=str(r.slug), name=str(r.name),
                          servings=int(r.servings),
                          ingredient_count=counts.get(str(r.slug), 0))
            for r in rows
        ]


@server.tool(title="Get a recipe", annotations=_READ)
def get_recipe(slug: str) -> Recipe:
    """Fetch one seeded recipe with its full ingredient list.
    Free — no LLM calls."""
    from . import db

    try:
        return db.load_recipe(slug)
    except ValueError as e:
        raise ToolError(f"{e}. Call list_recipes for valid slugs.") from e


@server.tool(title="List products", annotations=_READ)
def list_products(search: Annotated[str | None, Field(max_length=MAX_SEARCH)] = None,
                  category: Annotated[str | None, Field(max_length=MAX_SEARCH)] = None,
                  limit: Annotated[int, Field(ge=1, le=200)] = 50,
                  offset: Annotated[int, Field(ge=0)] = 0) -> ProductPage:
    """Page through the store catalog, ordered by id. `search` is a
    case-insensitive substring over name, brand, category and subcategory;
    `category` is an exact (case-insensitive) category OR subcategory such
    as "pantry", "dairy" or "pasta" — a value that is neither is an error
    naming the vocabulary (pantry://catalog/categories has the full tree),
    never a silent empty page. `total` counts every match and
    `next_offset` is null on the last page. For an ingredient lookup the
    way the planner sees it, prefer find_product. Free — no LLM calls."""
    from . import db

    catalog = sorted(db.load_all_products(), key=lambda p: p.id)
    products = catalog
    if search:
        needle = search.lower()
        products = [
            p for p in products
            if needle in f"{p.name} {p.brand} {p.category} {p.subcategory}".lower()
        ]
    if category:
        wanted = category.strip().lower()
        categories = sorted({(p.category or "").lower() for p in catalog} - {""})
        subcategories = sorted({(p.subcategory or "").lower() for p in catalog} - {""})
        if wanted not in categories and wanted not in subcategories:
            raise ToolError(
                f"Unknown category {category!r}. Categories: {', '.join(categories)}. "
                f"Subcategories: {', '.join(subcategories)}. "
                "Read pantry://catalog/categories for the tree with counts.")
        products = [p for p in products
                    if (p.category or "").lower() == wanted
                    or (p.subcategory or "").lower() == wanted]
    page = products[offset:offset + limit]
    return ProductPage(items=[_product_summary(p) for p in page], total=len(products),
                       next_offset=_page(len(products), offset, len(page)))


# ─── Product lookup (free — DB reads only) ───────────────────

@server.tool(title="Find a product", annotations=_READ)
def find_product(query: Annotated[str, Field(min_length=1, max_length=MAX_SEARCH)],
                 limit: Annotated[int, Field(ge=1, le=25)] = 8,
                 lat: float | None = None, lon: float | None = None) -> ProductSearch:
    """Look up catalog products exactly the way the planner does, so a hit
    here is a planning candidate there. `query` is an ingredient name
    ("basmati rice", "gluten free penne"); every significant word must
    match (lowercase, stemmed, "fresh"/"of" ignored — see `tokens`). Items
    come back cheapest first, each priced at its cheapest store near
    lat/lon (the default location when omitted). `total` is the hit count
    before `limit` trims. Free — no LLM calls.

    Read `match` before `items`: "direct" — every token matched;
    "generic" — nothing matched every token, but dropping descriptor words
    (light, dark, toasted, ground, whole, dried, ...) did: "light brown
    sugar" finds Brown Sugar 1kg. These ARE planning candidates — the planner
    selects them and labels the line match="generic"; "relaxed" — not
    even that, so these are same-aisle alternatives on the last word only,
    which the planner would OFFER as substitutes but never select; "none" —
    nothing at all (try list_products(search=...) for substring matching),
    or a non-purchase such as water or ice, which the planner never buys.
    An empty result is an answer, not an error. `origin_status`/
    `origin_country` come from ingested evidence; "unknown" is the common
    case."""
    from . import origins
    from .nlsearch.units import generic_tokens, is_non_purchase, tokens

    lat, lon = _location(lat, lon)
    toks = tokens(query)
    if is_non_purchase(query):
        # planner parity: a plan skips these before retrieval, so no product
        # is ever a candidate for them ("in water" on a can is not water)
        return ProductSearch(query=query, tokens=toks, match="none", total=0, items=[],
                             note=(f"{query!r} is never bought: plans skip water and "
                                   "ice and list them under summary.skipped."))
    hits = _planner_candidates(query, lat, lon) if toks else []
    generic = generic_tokens(query)
    if hits:
        match = "direct"
        note = (f"{len(hits)} product(s) match every token in {query!r}; "
                "each is priced at its cheapest store in range.")
    elif generic and (hits := _planner_candidates(query, lat, lon, generic=True)):
        match = "generic"
        note = (f"No product matches every token in {query!r}; these match "
                f"{' '.join(generic)!r} once descriptor words are dropped. The "
                "planner selects them as a generic match and says so in its notes.")
    else:
        if len(toks) > 1:
            hits = _planner_candidates(toks[-1], lat, lon)
        if hits:
            match = "relaxed"
            note = (f"No product matches every token in {query!r}. These are "
                    f"same-aisle alternatives matching only {toks[-1]!r}; the "
                    "planner would offer them as substitutes, never select them.")
        else:
            match = "none"
            note = (f"No catalog product matches {query!r}. Try "
                    "list_products(search=...) for substring matching over "
                    "name, brand, category and subcategory.")
    total = len(hits)
    kept = hits[:limit]
    resolved = origins.resolve_all([p.id for p in kept]) if kept else {}
    items = []
    for p in kept:
        o = resolved.get(p.id)
        items.append(ProductMatch(
            id=p.id, name=p.name, brand=p.brand, size=p.unit_size,
            category=p.category, subcategory=p.subcategory,
            store=p.store_name,
            price=p.store_price if p.store_price is not None else p.price,
            distance_km=p.distance_km if p.distance_km is not None else 0.0,
            origin_status=o.status if o else "unknown",
            origin_country=o.country if o else ""))
    return ProductSearch(query=query, tokens=toks, match=match, total=total,
                         items=items, note=note)


@server.tool(title="Get a product", annotations=_READ)
def get_product(product_id: int, lat: float | None = None,
                lon: float | None = None) -> ProductDetail:
    """Everything the catalog knows about one product: every store offer
    (cheapest first, distance from lat/lon or the default location), the
    resolved origin summary, the evidence rows behind it — including rows
    the resolver ignored, such as importer-only addresses — the number of
    origin readings still awaiting review, and review stats. Read
    `origin.status` before `origin.country`: only "resolved" is usable.
    Get ids from find_product. Free — no LLM calls."""
    from sqlalchemy import func, text
    from sqlalchemy.orm import Session

    from . import db, ingest, origins
    from .nlsearch.sql_builder import DIST_EXPR, _location_params

    lat, lon = _location(lat, lon)
    params: dict = {"pid": product_id}
    _location_params(params, lat, lon)
    # Same distance expression as the planner's retrieval, so the km an
    # agent sees here are the km the plan was priced on.
    offer_sql = (
        f"SELECT s.name AS store, sp.price AS price, {DIST_EXPR} AS dist_km2 "
        "FROM store_products sp JOIN stores s ON s.id = sp.store_id "
        "WHERE sp.product_id = :pid ORDER BY sp.price ASC, s.id ASC")
    with Session(db.engine()) as s:
        row = s.get(db.ProductRow, product_id)
        if row is None:
            raise ToolError(f"Unknown product id {product_id}. "
                            "Use find_product to look one up.")
        offers = [
            Offer(store=str(r["store"]), price=float(r["price"]),
                  distance_km=round(math.sqrt(max(float(r["dist_km2"]), 0.0)), 1))
            for r in s.execute(text(offer_sql), params).mappings()
        ]
        n_reviews, avg = (
            s.query(func.count(db.ReviewRow.id), func.avg(db.ReviewRow.rating))
            .filter(db.ReviewRow.product_id == product_id).one())
        evidence = [
            EvidenceSummary(
                source=str(e.source), source_ref=str(e.source_ref or ""),
                claim_type=str(e.claim_type or "unknown"),
                ingredient_origin=str(e.ingredient_origin or ""),
                manufactured_in=str(e.manufactured_in or ""),
                verbatim=str(e.verbatim or ""), confidence=str(e.confidence or "low"),
                importer_only=bool(e.importer_only), observed_at=str(e.observed_at or ""))
            for e in db.load_origin_evidence([product_id]).get(product_id, [])
        ]
        # Runtime values of legacy Column attributes; the explicit None
        # checks keep a NULL from becoming the string "None".
        category, subcategory, unit_qty = row.category, row.subcategory, row.unit_qty
        return ProductDetail(
            id=int(row.id), name=str(row.name), brand=str(row.brand or ""),
            description=str(row.description or ""),
            category=str(category) if category is not None else None,
            subcategory=str(subcategory) if subcategory else None,
            dietary_tags=str(row.dietary_tags or ""),
            unit_size=str(row.unit_size or ""),
            unit_qty=float(unit_qty) if unit_qty is not None else None,
            unit_uom=str(row.unit_uom or ""), list_price=float(row.price),
            offers=offers, origin=origins.resolve_all([product_id])[product_id],
            evidence=evidence,
            pending_submissions=ingest.pending_submission_count(product_id),
            review_count=int(n_reviews),
            avg_rating=float(avg) if avg is not None else None)


# ─── Planning tools (SLOW — run the LLM pipeline) ────────────

@server.tool(title="Plan a recipe", annotations=_PLAN)
def plan_recipe(slug: str,
                exclude_origin: Annotated[list[str] | None, Field(max_length=MAX_LIST)] = None,
                preference: Annotated[list[str] | None, Field(max_length=MAX_LIST)] = None,
                verbose: bool = False,
                lat: Annotated[float | None, Field(ge=-90, le=90)] = None,
                lon: Annotated[float | None, Field(ge=-180, le=180)] = None,
                max_km: Annotated[float | None, Field(ge=0.5, le=100)] = None,
                allow_partial: bool = False,
                basis: bool = False) -> PlanResult:
    """Run the full shopping-plan pipeline for a seeded recipe: an LLM
    matches every ingredient to the best-value product, with a model
    router escalating hard cases. SLOW (10-60s) and costs LLM API credits
    unless the server runs in demo mode. Get slugs from list_recipes first.

    Pass the shopper's location to get stores: any of lat/lon/max_km makes
    the plan store-aware. Each line is then priced at its cheapest store
    within max_km (0.5-100) of lat/lon (the server's default point when
    omitted; any distance when max_km is omitted), `summary.trip` is the
    recommended store split, and a chosen product no store in range sells
    goes to `summary.out_of_range` (naming the nearest offer) instead of
    being priced. Without a location, lines carry catalog prices and no
    store.

    `exclude_origin` drops candidates positively evidenced as coming from
    those countries (e.g. ["United States"]) before the model ever sees
    them — never candidates that merely lack evidence. When that leaves an
    ingredient with no candidate the call fails naming it, with the removed
    products and the alternatives still available; with allow_partial=true
    the rest is planned and the ingredient goes to `summary.out_of_range`
    with those options. `preference` is soft
    guidance. Read `summary.origin_status` and `summary.coverage`
    (`spend_fraction`, `meets_floor`) before describing a basket as clean,
    because unverified lines are not verified-clean lines; `summary.notes`
    carries substitutions and the coverage warning. `summary` is the
    token-lean result; `verbose=True` also attaches `full` (the complete
    ShoppingPlan with per-line reasoning and every trip option).
    `basis=true` adds `summary.basis`, what the plan was made from, for a
    client that re-prices or ranks alternatives later; an agent never needs
    it."""
    from . import flow
    from .nlsearch import PlanAborted

    _check_countries(exclude_origin, preference)
    try:
        plan = flow.run(slug, exclude=exclude_origin, preference=preference,
                        lat=lat, lon=lon, max_km=max_km, allow_partial=allow_partial)
        return PlanResult(summary=_summarize_plan(plan, basis),
                          full=plan if verbose else None)
    except ValueError as e:
        raise ToolError(f"{e}. Call list_recipes for valid slugs.") from e
    except PlanAborted as e:
        raise ToolError(_gate_message(e, allow_partial)) from e
    except LLMError as e:
        raise ToolError(_llm_message(e)) from e


@server.tool(title="Plan from recipe text", annotations=_PLAN)
def plan_from_text(recipe_text: Annotated[str, Field(max_length=MAX_TEXT)],
                   lat: float | None = None, lon: float | None = None,
                   exclude_origin: Annotated[list[str] | None, Field(max_length=MAX_LIST)] = None,
                   preference: Annotated[list[str] | None, Field(max_length=MAX_LIST)] = None,
                   verbose: bool = False,
                   max_km: Annotated[float | None, Field(ge=0.5, le=100)] = None,
                   allow_partial: bool = False,
                   basis: bool = False) -> PlanResult:
    """Plan a shopping basket from PASTED RECIPE TEXT — include the full
    ingredient list (quantities optional) and any shopping notes
    (budget, dietary exclusions). Parses the text, runs a staged SQL
    retrieval plan, then the LLM selector. SLOW (10-60s) and costs real
    Claude API credits.

    "Nearby" is lat/lon plus max_km: lat/lon set the shopping location
    (the server's default point when omitted) and max_km (0.5-100) is how
    far a store may be. max_km overrides any distance written in the text;
    with neither, every store counts, however far.

    allow_partial=true plans what the catalog can supply instead of
    failing on the first gap — use it for a real recipe (e.g. one fetched
    from a link): ingredients the catalog does not stock go to
    `summary.not_stocked`, ingredients stocked but with no offer within
    the distance/price/diet constraints go to `summary.out_of_range` (the
    reason names the nearest offer outside them and the limit it breaks),
    and the rest is priced. `total_cost`, `coverage` and `trip` then cover
    the planned lines only and `notes` says "planned K of N ingredients" —
    report the dropped ingredients too, never only the priced basket. With
    allow_partial false (the default) any such gap is an error naming the
    ingredients; when nothing at all can be planned it is an error either
    way, and so is a budget the cheapest basket exceeds (budget_infeasible).
    Water and ice are never bought: they, and anything past the
    40-ingredient cap, go to `summary.skipped` either way.

    Each line's `match` is "exact", "form" (purchase form swapped or
    dropped: "cumin powder" -> Cumin Ground 100g) or "generic" (descriptor
    words such as light/dark/toasted dropped: "light brown sugar" -> Brown
    Sugar 1kg); generic lines are named in `notes`. Recipe lines that chose
    the same product are one purchase (`also_lines`, `packs`), priced once.
    `summary.notes` starts with how the text was interpreted; `trip` is the
    recommended store split. `verbose=True` attaches `full` with the
    retrieval `plan_trace` and every trip option. `basis=true` adds
    `summary.basis` (see plan_recipe); an agent never needs it."""
    _check_countries(exclude_origin, preference)
    from . import flow
    from .nlsearch import PlanAborted, UnparseableRecipe

    try:
        plan = flow.run_nl(recipe_text, lat=lat, lon=lon,
                           exclude=exclude_origin, preference=preference,
                           max_km=max_km, allow_partial=allow_partial)
        return PlanResult(summary=_summarize_plan(plan, basis),
                          full=plan if verbose else None)
    except UnparseableRecipe as e:
        raise ToolError(
            "Couldn't find an ingredient list in that text. Paste a recipe "
            "with its ingredients (quantities optional), e.g.:\n"
            "Spaghetti Bolognese (serves 4)\n"
            "- 400g spaghetti\n- 500g ground beef\n- 1 can crushed tomatoes\n"
            "Notes: under $30, no dairy") from e
    except PlanAborted as e:
        alert = e.execution.aborted
        steps = ", ".join(
            f"{s.step_id}:{s.outcome}" for s in e.execution.steps)
        raise ToolError(
            f"Plan aborted before product selection — "
            f"{alert.code.value if alert else 'gate'}: "
            f"{(alert.message if alert else 'constraint infeasible').rstrip('.')}."
            f"{_affected(alert)}{_partial_hint(alert, allow_partial)} (steps: {steps})") from e
    except LLMError as e:
        raise ToolError(_llm_message(e)) from e


class LineIn(BaseModel):
    """One reviewed ingredient line for plan_from_lines: planned exactly as given."""
    name: Annotated[str, Field(min_length=1, max_length=MAX_SEARCH)]
    quantity: Annotated[float | None, Field(ge=0)] = None
    unit: Annotated[str, Field(max_length=40)] = ""
    note: Annotated[str, Field(max_length=300)] = ""
    text: Annotated[str, Field(max_length=300)] = ""
    confirmed: bool = True
    # Who stated the amount; the hub passes the reviewed doc's own. A client
    # writing lines itself is an assistant writing them.
    amount_basis: AmountBasis = "written_by_assistant"


@server.tool(title="Plan from reviewed lines", annotations=_PLAN)
def plan_from_lines(doc_key: Annotated[str, Field(min_length=1, max_length=100)],
                    lines: Annotated[list[LineIn] | None,
                                     Field(max_length=MAX_DOC_LINES)] = None,
                    title: Annotated[str | None, Field(max_length=200)] = None,
                    servings: Annotated[int | None, Field(ge=1, le=100)] = None,
                    lat: Annotated[float | None, Field(ge=-90, le=90)] = None,
                    lon: Annotated[float | None, Field(ge=-180, le=180)] = None,
                    max_km: Annotated[float | None, Field(ge=0.5, le=100)] = None,
                    exclude_origin: Annotated[list[str] | None,
                                              Field(max_length=MAX_LIST)] = None,
                    preference: Annotated[list[str] | None, Field(max_length=MAX_LIST)] = None,
                    allow_partial: bool = False,
                    basis: bool = False,
                    verbose: bool = False) -> PlanResult:
    """Plan a recipe the shopper has already reviewed, line by line, exactly as
    reviewed: each line's name, quantity and unit are planned as given, with no
    re-reading of the recipe. Name the recipe by `doc_key` (for example
    "imp:1", the key of a recipe the shopper imported); the app fills in its
    reviewed lines. A client without such an app passes `lines` itself (at
    most 60, each {name, quantity, unit, note}), with `title` and `servings`
    when known. The selector still picks the products, so this is SLOW and
    costs LLM credits unless the server runs in demo mode.

    Location, origin and allow_partial work as in plan_from_text. Every line
    must be confirmed; an unconfirmed one is an error naming it. The result
    is shaped like plan_from_text's: read `summary.notes` and the left-out
    lists the same way."""
    from . import flow
    from .models import RecipeDoc
    from .nlsearch import PlanAborted, UnparseableRecipe
    from .recipe_doc import UnconfirmedLines, to_recipe_text, to_spec

    if not lines:
        raise ToolError(f"No lines for {doc_key!r}: plan_from_lines plans reviewed lines, "
                        "which the app fills in for a doc_key it holds. Without it, pass "
                        "`lines` (at most 60) or use plan_from_text for recipe text.")
    _check_countries(exclude_origin, preference)
    doc = RecipeDoc(
        key=doc_key, title=title or doc_key, servings=servings,
        servings_stated=servings is not None,
        lines=[RecipeLine(line_no=i, text=ln.text or ln.name, name=ln.name,
                          quantity=ln.quantity, unit=ln.unit, note=ln.note,
                          confirmed=ln.confirmed, amount_basis=ln.amount_basis)
               for i, ln in enumerate(lines, start=1)],
        source=RecipeSource(kind="assistant", method="agent_written"))
    try:
        plan = flow.run_spec(to_spec(doc), lat=lat, lon=lon, exclude=exclude_origin,
                             preference=preference, max_km=max_km,
                             allow_partial=allow_partial, display_text=to_recipe_text(doc))
        return PlanResult(summary=_summarize_plan(plan, basis),
                          full=plan if verbose else None)
    except UnconfirmedLines as e:
        raise ToolError(f"{e}.") from e
    except UnparseableRecipe as e:
        raise ToolError("Nothing to plan: every line was empty.") from e
    except PlanAborted as e:
        raise ToolError(_gate_message(e, allow_partial)) from e
    except LLMError as e:
        raise ToolError(_llm_message(e)) from e


@server.tool(title="Plan a week of dinners", annotations=_PLAN)
def plan_week(days: Annotated[int, Field(ge=1, le=14)] = 5,
              max_total_budget: float | None = None,
              exclude_tags: Annotated[list[str] | None, Field(max_length=MAX_LIST)] = None,
              lat: float | None = None, lon: float | None = None,
              max_distance_km: float | None = None,
              exclude_origin: Annotated[list[str] | None, Field(max_length=MAX_LIST)] = None,
              preference: Annotated[list[str] | None, Field(max_length=MAX_LIST)] = None,
              verbose: bool = False) -> WeekResult:
    """Plan `days` dinners from the recipe library under an optional
    budget, rewarding ingredient overlap (a shared product is bought
    once). `summary` has the per-day lines, the merged shopping list
    (each product once, with `used_by`), overlap savings, the recommended
    trip and notes (skipped recipes, substitutions, coverage warnings).
    exclude_tags filters out recipes containing those dietary tags (e.g.
    ["dairy", "gluten"]). `verbose=True` attaches `full` with the retrieval
    trace and every trip option. SLOW (runs the LLM selector per day) and
    costs real Claude API credits — roughly one plan_recipe per day."""
    _check_countries(exclude_origin, preference)
    from . import weekplan
    from .nlsearch import PlanAborted

    try:
        plan = weekplan.plan_week(
            days=days, max_total_budget=max_total_budget,
            exclude_tags=exclude_tags or [], lat=lat, lon=lon,
            max_distance_km=max_distance_km,
            exclude_origin=exclude_origin, preference=preference)
        return WeekResult(summary=_summarize_week(plan), full=plan if verbose else None)
    except PlanAborted as e:
        alert = e.execution.aborted
        raise ToolError(
            f"Week plan aborted — "
            f"{alert.code.value if alert else 'gate'}: "
            f"{alert.message if alert else 'constraint infeasible'}{_affected(alert)}") from e
    except LLMError as e:
        raise ToolError(_llm_message(e)) from e


# --- Provenance tools ----------------------------------------
# Origin here is evidence, never inference. Records are ingested from the
# companion claude-chrome-container tooling (Open Food Facts lookups and
# package-label photo reads); nothing in this server guesses a country.

@server.tool(title="Get product origins", annotations=_READ)
def get_product_origins(product_ids: Annotated[list[int] | None, Field(max_length=MAX_IDS)] = None,
                        search: Annotated[str | None, Field(max_length=MAX_SEARCH)] = None,
                        status: str | None = None,
                        limit: Annotated[int, Field(ge=1, le=200)] = 50,
                        offset: Annotated[int, Field(ge=0)] = 0) -> OriginPage:
    """Resolved country-of-origin evidence per product - by ids, by a name
    `search` filter, or the whole catalog - one page at a time, ordered by
    product id. Free, no LLM calls.

    `by_status` counts EVERY product the ids/search selected, before the
    `status` filter and before paging, so one call answers "161 products:
    2 resolved, 159 unknown"; then `status="resolved"` pages through just
    those. Read `status` before using `country`: only "resolved" carries
    usable evidence. "unknown" means no source published an origin,
    "conflicting" means sources disagreed and no winner was picked,
    "lookup_failed" means a source did not answer, and "guess" is a
    name-based hint that must not be treated as provenance. Coverage is
    thin and biased - most Canadian products resolve to "unknown" - so
    absence is never evidence of foreign origin."""
    from . import db, origins

    if status is not None and status not in _ORIGIN_STATUSES:
        raise ToolError(f"Unknown origin status {status!r}. Allowed: "
                        + ", ".join(sorted(_ORIGIN_STATUSES)) + ".")
    products = sorted(db.load_all_products(), key=lambda p: p.id)
    if product_ids is not None:
        wanted = set(product_ids)
        products = [p for p in products if p.id in wanted]
        missing = wanted - {p.id for p in products}
        if missing:
            raise ToolError(
                f"Unknown product ids: {sorted(missing)}. "
                "Call list_products for valid ids.")
    elif search:
        needle = search.lower()
        products = [p for p in products
                    if needle in f"{p.name} {p.brand} {p.category}".lower()]
    resolved = origins.resolve_all([p.id for p in products]) if products else {}
    rows = [resolved[p.id] for p in products if p.id in resolved]
    by_status: dict[str, int] = {}
    for o in rows:
        by_status[o.status] = by_status.get(o.status, 0) + 1
    if status is not None:
        rows = [o for o in rows if o.status == status]
    page = rows[offset:offset + limit]
    return OriginPage(items=page, total=len(rows), by_status=by_status,
                      next_offset=_page(len(rows), offset, len(page)))


@server.tool(title="Rank products by origin", annotations=_READ)
def rank_products_by_origin(
        preference: Annotated[list[str] | None, Field(max_length=MAX_LIST)] = None,
        exclude: Annotated[list[str] | None, Field(max_length=MAX_LIST)] = None,
        search: Annotated[str | None, Field(max_length=MAX_SEARCH)] = None,
) -> OriginRanking:
    """Rank catalog products by where they come from, against a country
    preference YOU supply. Free, no LLM calls.

    `preference` is ordered, most-preferred first (e.g. ["Canada",
    "Mexico"]). `exclude` filters out products with positive evidence of
    those origins (e.g. ["United States"]) - a product is never excluded
    merely for lacking evidence.

    Exclusion checks ingredient origin as well as manufacturing origin,
    because "Made in Canada" legally permits imported ingredients: peanut
    butter made in Canada from American peanuts matches an exclusion of the
    United States, which is intended.

    Within a preferred country, a full origin claim ("Product of Canada",
    >=98% domestic content) outranks a processing claim ("Made in Canada",
    ingredients may be imported).

    The result keeps `ranked`, `excluded` and `unranked` separate. Report
    the `unranked` count - those products have no published origin, and
    showing only the ranked list would imply a coverage this data does not
    have."""
    _check_countries(preference, exclude)
    from . import db, origins

    products = db.load_all_products()
    if search:
        needle = search.lower()
        products = [p for p in products
                    if needle in f"{p.name} {p.brand} {p.category}".lower()]
    return origins.rank_products(
        products, preference=preference or [], exclude=exclude or [])


@server.tool(title="Origin triage", annotations=_READ)
def origin_triage() -> list[TriageCandidate]:
    """Products whose origin is unresolved and which are worth reading a
    package label for. Free, no LLM calls.

    These are hints derived from names and categories, NOT origins - they
    say "go check this", never "this is from X". Seafood and fresh produce
    appear often because their origin is not published online at all; only
    the printed label carries it."""
    from . import db, origins

    return [TriageCandidate(**c)
            for c in origins.triage_candidates(db.load_all_products())]


# ─── Origin submissions (the write path) ─────────────────────
# An agent that has read a package label can submit what it saw. The
# submission is a queue entry, not evidence: nothing it says reaches the
# resolver or the planner until a reviewer approves it, at which point it
# is COPIED into product_origin_evidence as source "agent-label". Over HTTP
# the caller must hold a configured bearer token; over stdio the operator's
# own process is trusted.

@server.tool(title="Submit an origin reading", annotations=_SUBMIT)
def submit_origin_evidence(product_id: int, claim_type: str,
                           country: Annotated[str, Field(max_length=80)],
                           verbatim: Annotated[str, Field(min_length=3, max_length=500)],
                           confidence: str = "medium", importer_only: bool = False,
                           note: Annotated[str, Field(max_length=1000)] = "",
                           source_ref: Annotated[str, Field(max_length=300)] = "",
                           ctx: Context = None) -> Submission:  # type: ignore[assignment]
    """Submit a country-of-origin reading from a package label for review.
    Call find_product first to get the `product_id`. Free — no LLM calls.

    `verbatim` is the EXACT printed wording, character for character —
    never a paraphrase, never a translation. A reviewer checks it against
    the claim, so "Product of U.S.A." must not become "Made in USA".

    `claim_type` says what the wording legally asserts:
      product-of / grown-in / farmed-in / harvested-in / caught-in — a FULL
        claim: all or virtually all (>=98%) of the content is from that
        country. Fills ingredient_origin and manufactured_in.
      made-in / prepared-in / packaged-in — a PROCESSING claim: the last
        substantial transformation happened there; the ingredients may
        well be imported. Fills manufactured_in only.
    An "Imported by …" or "Distributed by …" address is NOT an origin: set
    `importer_only=True` so it is recorded but never ranked as provenance.
    `confidence`: high = the full declaration is legible; medium = partial
    or cropped; low = inferred from a flag, address or fragment.

    Submissions are PENDING and change nothing — not the product's origin,
    not any plan — until a reviewer approves them with
    review_origin_submission. Resubmitting the same reading returns the
    existing entry (`duplicate=True`) rather than queueing it twice; if it
    was rejected you get the rejection and its note. Validation failures
    name the field and the allowed values; unknown country spellings come
    back with suggestions."""
    from . import ingest

    label = _require_write(ctx)
    _check_countries([country])
    try:
        return Submission(**ingest.submit_origin({
            "product_id": product_id, "claim_type": claim_type, "country": country,
            "verbatim": verbatim, "confidence": confidence,
            "importer_only": importer_only, "note": note, "source_ref": source_ref,
        }, submitted_by=label))
    except ValueError as e:
        raise ToolError(str(e)) from e


@server.tool(title="List origin submissions", annotations=_READ)
def list_origin_submissions(status: str | None = "pending",
                            limit: Annotated[int, Field(ge=1, le=200)] = 50,
                            offset: Annotated[int, Field(ge=0)] = 0) -> SubmissionPage:
    """Page through the origin review queue, oldest first. `status` is
    "pending" (default), "approved", "rejected" or null for every status.
    `next_offset` is null on the last page. Free — no LLM calls."""
    from . import ingest

    try:
        items, total = ingest.list_submissions(status, limit=limit, offset=offset)
    except ValueError as e:
        raise ToolError(str(e)) from e
    end = offset + len(items)
    return SubmissionPage(items=[Submission(**d) for d in items], total=total,
                          next_offset=end if end < total else None)


@server.tool(title="Review an origin submission", annotations=_REVIEW)
def review_origin_submission(submission_id: int, decision: str,
                             note: Annotated[str, Field(max_length=1000)] = "",
                             ctx: Context = None) -> Submission:  # type: ignore[assignment]
    """Approve or reject a pending submission. `decision` is "approve" or
    "reject"; a rejection needs a `note` (>= 3 chars) saying why, so the
    next reader can tell a blurry photo from a wrong claim. Approval copies
    the reading into the evidence table as source "agent-label", refreshes
    the product's resolved origin, and from then on plan tools honour it.
    Review each one against get_product first: does the verbatim wording
    support the claim type and the country? A submission can be reviewed
    once; reviewing it again is an error naming its status."""
    from . import ingest

    label = _require_write(ctx)
    try:
        return Submission(**ingest.review_submission(
            submission_id, decision, reviewed_by=label, note=note))
    except ValueError as e:
        raise ToolError(str(e)) from e


# ─── Status ──────────────────────────────────────────────────

@server.tool(title="Pipeline status", annotations=_READ)
def pipeline_status(ctx: Context = None) -> PipelineStatus:  # type: ignore[assignment]
    """Active configuration: routing strategy, models (each a spec:
    "gemini:<model>" or an Anthropic name), confidence threshold, DB
    target, which LLM keys are configured, and whether this caller may
    submit origin readings (`write_tools`). Reflects runtime overrides
    from POST /settings/runtime. Free — no LLM calls."""
    from .config import redact_db_url, settings

    cfg = settings()
    label, transport = _principal(ctx)
    return PipelineStatus(
        status="ok",
        routing_strategy=cfg.routing_strategy,
        default_model=cfg.selector_model_default,
        escalation_model=cfg.selector_model_escalation,
        confidence_threshold=cfg.confidence_threshold,
        db=redact_db_url(cfg.db_url),
        anthropic_key_configured=bool(cfg.anthropic_api_key),
        demo_mode=cfg.demo_mode,
        mcp_auth="required" if cfg.mcp_auth_tokens else "anonymous",
        write_tools=("disabled" if transport == "http" and label == "anonymous"
                     else "enabled"),
        classifier_model=cfg.classifier_model,
        nl2sql_model=cfg.nl2sql_model,
        gemini_key_configured=bool(cfg.gemini_api_key),
    )


# ─── Resources (reference data an agent reads once per session) ──
# Each resource is a thin wrapper over the same code the tools use, so it
# can never disagree with them. A template function that cannot find its
# instance raises ResourceNotFoundError: the SDK passes that message
# through to the client, whereas any other exception is masked as
# "Error creating resource from template <uri>" and the hint is lost.

_JSON = "application/json"


@server.resource("pantry://recipes", name="recipes", title="Recipe library",
                 mime_type=_JSON)
def recipes_resource() -> list[RecipeSummary]:
    """Every seeded recipe: slug, name, servings, ingredient count. The
    slug is what plan_recipe and pantry://recipes/{slug} take."""
    return list_recipes()


@server.resource("pantry://recipes/{slug}", name="recipe", title="One recipe",
                 mime_type=_JSON)
def recipe_resource(slug: str) -> Recipe:
    """A seeded recipe with its ordered ingredient list."""
    from . import db

    try:
        return db.load_recipe(slug)
    except ValueError as e:
        raise ResourceNotFoundError(
            f"{e}. Read pantry://recipes or call list_recipes for valid slugs.") from e


@server.resource("pantry://catalog/categories", name="catalog-categories",
                 title="Catalog categories", mime_type=_JSON)
def categories_resource() -> dict[str, dict[str, int]]:
    """{category: {subcategory: product count}} over the whole catalog —
    the vocabulary list_products(category=...) accepts, with the size of
    each aisle, from one GROUP BY."""
    from sqlalchemy import func
    from sqlalchemy.orm import Session

    from .db import ProductRow, engine

    with Session(engine()) as s:
        rows = (s.query(ProductRow.category, ProductRow.subcategory,
                        func.count(ProductRow.id))
                .group_by(ProductRow.category, ProductRow.subcategory)
                .order_by(ProductRow.category, ProductRow.subcategory).all())
    out: dict[str, dict[str, int]] = {}
    for category, subcategory, n in rows:
        out.setdefault(str(category or ""), {})[str(subcategory or "")] = int(n)
    return out


@server.resource("pantry://countries", name="countries", title="Country spellings",
                 mime_type=_JSON)
def countries_resource() -> dict[str, object]:
    """How to spell a country this server accepts. `canonical` is the
    sorted list of names every country folds to; `aliases` maps a canonical
    name to the other forms that fold to it ("usa", "U.S.A", state names);
    `ambiguous` lists inputs that are rejected with guidance ("korea" —
    which one?). Any name in `canonical` or `aliases` passes the country
    validation on exclude_origin / preference / submit_origin_evidence."""
    from . import origins

    canonical = sorted({origins._title(origins.canonical_country(n))
                        for n in origins.KNOWN_COUNTRIES})
    aliases = {origins._title(canon): sorted(forms)
               for canon, forms in origins._ALIASES.items()}
    return {"canonical": canonical, "aliases": aliases,
            "ambiguous": dict(origins.AMBIGUOUS)}


@server.resource("pantry://origins/coverage", name="origin-coverage",
                 title="Origin coverage", mime_type=_JSON)
def origin_coverage_resource() -> dict[str, object]:
    """How much of the catalog has a provenance at all: product count,
    resolved-origin counts by status, evidence rows, the review queue by
    status (null, with a note, when the review-queue table is not deployed
    yet), and the coverage floor a plan must reach before its basket may be
    called verified. Read this before describing coverage."""
    from sqlalchemy import func
    from sqlalchemy.orm import Session

    from . import db, ingest, origins
    from .config import settings

    with Session(db.engine()) as s:
        products = int(s.query(func.count(db.ProductRow.id)).scalar() or 0)
        evidence_rows = int(s.query(func.count(db.ProductOriginEvidenceRow.id)).scalar() or 0)
    by_status: dict[str, int] = {}
    for o in origins.resolve_all().values():
        by_status[o.status] = by_status.get(o.status, 0) + 1
    # None (not zeros) when the review-queue table is not deployed yet: the
    # resource must still answer the coverage question it exists for.
    queue = ingest.submission_counts_by_status()
    return {
        "products": products,
        "by_status": by_status,
        "evidence_rows": evidence_rows,
        "submissions": queue,
        "submissions_note": "" if queue is not None else ingest.QUEUE_MISSING,
        "floor": settings().origin_min_coverage,
    }


# ─── Prompts (the protocols a client can hand its model) ─────
# A prompt is the operator's wording for a multi-tool task, so the agent
# follows the same discipline every time: look the id up first, report the
# coverage numbers verbatim, transcribe a label rather than paraphrase it.

@server.prompt(title="Plan a dinner",
               description="Plan a priced basket for a recipe and report its "
                           "provenance honestly.")
def plan_dinner(recipe: str, exclude_origin: str = "", budget: str = "") -> str:
    """Plan a priced shopping basket for `recipe` (a seeded slug or pasted
    recipe text), optionally excluding origin countries and under a
    budget."""
    exclude = (f"Exclude products evidenced as coming from: {exclude_origin}. "
               "Check each spelling against the pantry://countries resource "
               "and pass them as the exclude_origin list. "
               if exclude_origin.strip() else "")
    budget_line = (f"The budget is {budget}: compare it with summary.total_cost "
                   "and say plainly whether the basket is under it. "
                   if budget.strip() else "")
    return (
        f"Plan dinner for: {recipe}.\n\n"
        "Steps:\n"
        "1. Call list_recipes. If the request names a seeded recipe, use its "
        "slug with plan_recipe; otherwise pass the text to plan_from_text "
        "(for a full recipe from elsewhere, with allow_partial=true). "
        "If an ingredient is in doubt, call find_product for it first and "
        "read `match` — \"relaxed\" hits are alternatives the planner would "
        "only offer, not candidates.\n"
        f"2. {exclude}{budget_line}Call the plan tool once; it is slow and "
        "costs credits.\n"
        "3. Report summary.total_cost, every line (ingredient → product, store, "
        "price; say so when its `match` is \"generic\") and the recommended "
        "trip. Then name every entry in summary.not_stocked, "
        "summary.out_of_range and summary.skipped with its reason: when any "
        "is non-empty the priced basket is not the whole recipe.\n"
        "4. Report summary.origin_status and summary.coverage.spend_fraction "
        "VERBATIM as numbers, with coverage.lines_known of coverage.lines_total. "
        "Never describe the basket as clean, verified or free of a country "
        "unless origin_status is \"verified\" and coverage.meets_floor is true: "
        "a line with no evidence is unknown, not foreign and not domestic.\n"
        "5. Name every entry in summary.notes (interpretation, substitutions, "
        "the coverage-floor warning) rather than summarising them away.\n"
        "If the tool returns an error about an origin gate, report the gate's "
        "message and the affected ingredients; do not retry with the exclusion "
        "silently dropped."
    )


@server.prompt(title="Read a package label",
               description="Transcribe a country-of-origin declaration from a "
                           "package and submit it for review.")
def read_label(product: str) -> str:
    """The vision protocol for turning a label photo into a pending origin
    submission for `product`."""
    return (
        f"You are reading the package label of: {product}.\n\n"
        "1. Call find_product with the product name to get its `id`; if "
        "`match` is not \"direct\", ask which hit is the one in the photo "
        "before continuing.\n"
        "2. Find the country-of-origin declaration on the label and "
        "transcribe it EXACTLY as printed into `verbatim`: same words, same "
        "punctuation, same capitalisation, no translation, no paraphrase. "
        "\"Product of U.S.A.\" must not become \"Made in USA\".\n"
        "3. Classify `claim_type` from the wording alone: product-of / "
        "grown-in / farmed-in / harvested-in / caught-in are FULL claims (all "
        "or virtually all content from that country); made-in / prepared-in / "
        "packaged-in are PROCESSING claims (processed there, ingredients may be "
        "imported). Set `country` to the country the wording names, spelled as "
        "in the pantry://countries resource.\n"
        "4. An \"Imported by\", \"Distributed by\" or \"Packed for\" address is "
        "NOT an origin. Record it with importer_only=true so a reviewer sees "
        "it, and never present it as the product's origin.\n"
        "5. Set `confidence`: high = the full declaration is legible in the "
        "photo; medium = partial, cropped or a fragment; low = inferred from a "
        "flag, an address or a maple leaf rather than read.\n"
        "6. Call submit_origin_evidence with product_id, claim_type, country, "
        "verbatim, confidence, importer_only and a `note` on what you could "
        "and could not see. If it returns duplicate=true or status "
        "\"rejected\", report that instead of resubmitting.\n"
        "7. Tell the user the submission is PENDING: it changes nothing — not "
        "the product's origin, not any plan — until a reviewer approves it."
    )


@server.prompt(title="Review origin submissions",
               description="Work the pending origin-submission queue with "
                           "auditable decisions.")
def review_submissions() -> str:
    """The reviewer protocol: check each pending submission against the
    product and approve or reject it with a note a later reader can audit."""
    return (
        "Review the pending origin submissions.\n\n"
        "1. Call list_origin_submissions (status \"pending\"); page with "
        "next_offset until it is null.\n"
        "2. For each submission call get_product with its product_id and "
        "compare: does the product match what the verbatim wording describes? "
        "Does the wording support the claim_type (\"Product of\" is a full "
        "claim; \"Made in\" is processing-only)? Does it name the stated "
        "country? Is an \"Imported by\" address marked importer_only? Does "
        "existing evidence on the product contradict it?\n"
        "3. Call review_origin_submission with decision \"approve\" when the "
        "verbatim wording supports the claim type and the country, or "
        "\"reject\" otherwise. Always give a `note` that a later reader can "
        "audit without the photo: quote the wording and say what it does or "
        "does not establish (\"blurry\" and \"wrong claim type\" are "
        "different rejections).\n"
        "4. Approval copies the reading into evidence as source \"agent-label\" "
        "and the product's origin resolves from it immediately; report which "
        "products changed status, and leave anything you cannot verify pending."
    )


# ─── Transports ──────────────────────────────────────────────

def http_app():
    """Streamable HTTP ASGI app, mounted by api.py. Stateless + plain
    JSON responses: no session affinity or SSE needed behind nginx.
    DNS-rebinding protection is off because the app sits behind a
    reverse proxy whose Host header is the public hostname.

    Bearer auth is layered on by mcp_auth.install: with MCP_AUTH_TOKENS
    set every request needs a token (401 otherwise); unset, the endpoint
    is anonymous and the write tools refuse."""
    from mcp.server.transport_security import TransportSecuritySettings

    from .mcp_auth import install

    return install(server.streamable_http_app(
        streamable_http_path="/mcp",
        json_response=True,
        stateless_http=True,
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=False),
    ))


def main() -> None:
    """stdio entry point (the pantry-mcp console script).

    MCP clients launch this from an arbitrary cwd (Claude Desktop uses
    "/"), so Burr tracking goes to a home-dir default and .env loading
    is best-effort. Env must be final before the first tool call —
    settings() is lru_cached."""
    os.environ.setdefault(
        "BURR_TRACKING_DIR", str(Path.home() / ".pantry-planner" / "burr"))
    from dotenv import load_dotenv

    from .config import validate_startup

    load_dotenv()
    validate_startup()
    server.run("stdio")


if __name__ == "__main__":
    main()

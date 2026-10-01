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
from pydantic import BaseModel, Field

from .models import (
    OriginCoverage,
    OriginRanking,
    PlanLineItem,
    Product,
    ProductOrigin,
    Recipe,
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
    match: str                  # "direct" | "relaxed" | "none"
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
    pending_submissions: int            # origin readings awaiting review
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


class Coverage(BaseModel):
    spend_fraction: float       # the binding measure: what the money did
    count_fraction: float
    meets_floor: bool
    lines_known: int
    lines_total: int
    lines_excluded_origin: int


class Trip(BaseModel):
    """The recommended point on the stops-vs-cost frontier."""
    stores: list[str]
    total_cost: float
    savings_vs_one_stop: float


class PlanSummary(BaseModel):
    recipe_slug: str
    recipe_name: str
    total_cost: float
    origin_status: str          # not_requested | verified | unverified
    coverage: Coverage | None   # only when an origin question was asked
    lines: list[LeanLine]
    trip: Trip | None           # None when the planner produced no trip options
    notes: list[str]            # interpretation, substitutions, coverage warnings
    llm_cost_usd: float
    latency_ms: int


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


def _gate_message(e) -> str:
    """One shape for every gate abort an agent can hit."""
    alert = e.execution.aborted
    steps = ", ".join(f"{s.step_id}:{s.outcome}" for s in e.execution.steps)
    detail = ""
    if alert and alert.details:
        names = ", ".join(str(d.get("name", "?")) for d in alert.details)
        detail = f" Affected: {names}."
    code = alert.code.value if alert else "unknown"
    msg = alert.message if alert else "plan aborted"
    return (f"Plan aborted before product selection — {code}: {msg}{detail}"
            + (f" Steps: {steps}." if steps else ""))


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


def _planner_candidates(name: str, lat: float, lon: float) -> list:
    """Exactly how the planner finds candidates for one ingredient
    (flow._ingredient_pools): token-AND over the product_terms index, each
    product pinned to its cheapest in-range store, cheapest first, uncapped.
    A hit here is a candidate there; a miss here is a miss there."""
    from sqlalchemy import text
    from sqlalchemy.orm import Session

    from . import db
    from .nlsearch.planner import _row_to_product
    from .nlsearch.schemas import Constraints, IngredientSpec
    from .nlsearch.sql_builder import build_options_sql

    sql, params = build_options_sql(
        Constraints(), [IngredientSpec(name=name)], relaxed=set(),
        lat=lat, lon=lon, per_ingredient_limit=10_000)
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
        origin_status=o.status if o else "none")


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
    return Trip(stores=best.stores, total_cost=best.total_cost,
                savings_vs_one_stop=best.savings_vs_one_stop)


def _substitution_notes(lines: list[PlanLineItem], prefix: str = "") -> list[str]:
    """One note per line the selector itself flagged as a substitution."""
    return [f"{prefix}line {li.line_no} ({li.ingredient_name}): substitution — "
            f"{li.product_name}"
            for li in lines if "substitut" in li.reasoning.lower()]


def _floor_note(c: OriginCoverage | None) -> list[str]:
    if c is None or c.meets_floor:
        return []
    return [f"coverage below floor: origin known for {c.lines_known} of "
            f"{c.lines_total} lines ({c.spend_fraction:.0%} of spend, floor "
            f"{c.floor:.0%}); unknown is not foreign, but do not call this "
            "basket clean"]


def _summarize_plan(plan: ShoppingPlan) -> PlanSummary:
    products = _products_by_id({li.product_id for li in plan.line_items})
    notes = (list(plan.interpretation)
             + _substitution_notes(plan.line_items)
             + _floor_note(plan.origin_coverage))
    return PlanSummary(
        recipe_slug=plan.recipe_slug, recipe_name=plan.recipe_name,
        total_cost=plan.total_cost, origin_status=plan.origin_status,
        coverage=_coverage(plan.origin_coverage),
        lines=[_lean_line(li, products) for li in plan.line_items],
        trip=_trip(plan.trip_options), notes=notes,
        llm_cost_usd=plan.total_llm_cost_usd, latency_ms=plan.total_latency_ms)


def _summarize_week(plan: WeekPlan) -> WeekSummary:
    products = _products_by_id({li.product_id for d in plan.days for li in d.line_items})
    notes = list(plan.notes)
    for d in plan.days:
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
    `category` is an exact (case-insensitive) category such as "pantry",
    "dairy", "produce" or "meat". `total` counts every match and
    `next_offset` is null on the last page. For an ingredient lookup the
    way the planner sees it, prefer find_product. Free — no LLM calls."""
    from . import db

    products = sorted(db.load_all_products(), key=lambda p: p.id)
    if search:
        needle = search.lower()
        products = [
            p for p in products
            if needle in f"{p.name} {p.brand} {p.category} {p.subcategory}".lower()
        ]
    if category:
        wanted = category.strip().lower()
        products = [p for p in products if (p.category or "").lower() == wanted]
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
    "relaxed" — nothing matched all tokens, so these are same-aisle
    alternatives on the last word only, which the planner would OFFER as
    substitutes but never select; "none" — nothing at all (try
    list_products(search=...) for substring matching). An empty result is
    an answer, not an error. `origin_status`/`origin_country` come from
    ingested evidence; "unknown" is the common case."""
    from . import origins
    from .nlsearch.units import tokens

    lat, lon = _location(lat, lon)
    toks = tokens(query)
    hits = _planner_candidates(query, lat, lon) if toks else []
    if hits:
        match = "direct"
        note = (f"{len(hits)} product(s) match every token in {query!r}; "
                "each is priced at its cheapest store in range.")
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
                verbose: bool = False) -> PlanResult:
    """Run the full shopping-plan pipeline for a seeded recipe: an LLM
    matches every ingredient to the best-value product, with a model
    router escalating hard cases. SLOW (10-60s) and costs real Claude
    API credits. Get slugs from list_recipes first.

    `exclude_origin` drops candidates positively evidenced as coming from
    those countries (e.g. ["United States"]) before the model ever sees
    them — never candidates that merely lack evidence. `preference` is soft
    guidance. Read `summary.origin_status` and `summary.coverage`
    (`spend_fraction`, `meets_floor`) before describing a basket as clean,
    because unverified lines are not verified-clean lines; `summary.notes`
    carries substitutions and the coverage warning. `summary` is the
    token-lean result; `verbose=True` also attaches `full` (the complete
    ShoppingPlan with per-line reasoning and every trip option)."""
    from . import flow
    from .nlsearch import PlanAborted

    _check_countries(exclude_origin, preference)
    try:
        plan = flow.run(slug, exclude=exclude_origin, preference=preference)
        return PlanResult(summary=_summarize_plan(plan), full=plan if verbose else None)
    except ValueError as e:
        raise ToolError(f"{e}. Call list_recipes for valid slugs.") from e
    except PlanAborted as e:
        raise ToolError(_gate_message(e)) from e


@server.tool(title="Plan from recipe text", annotations=_PLAN)
def plan_from_text(recipe_text: Annotated[str, Field(max_length=MAX_TEXT)],
                   lat: float | None = None, lon: float | None = None,
                   exclude_origin: Annotated[list[str] | None, Field(max_length=MAX_LIST)] = None,
                   preference: Annotated[list[str] | None, Field(max_length=MAX_LIST)] = None,
                   verbose: bool = False) -> PlanResult:
    """Plan a shopping basket from PASTED RECIPE TEXT — include the full
    ingredient list (quantities optional) and any shopping notes
    (budget, dietary exclusions); lat/lon optionally set the shopping
    location. Parses the text, runs a staged SQL retrieval plan, then
    the LLM selector. SLOW (10-60s) and costs real Claude API credits.
    `summary.notes` starts with how the text was interpreted; `trip` is
    the recommended store split. `verbose=True` attaches `full` with the
    retrieval `plan_trace` and every trip option."""
    _check_countries(exclude_origin, preference)
    from . import flow
    from .nlsearch import PlanAborted, UnparseableRecipe

    try:
        plan = flow.run_nl(recipe_text, lat=lat, lon=lon,
                           exclude=exclude_origin, preference=preference)
        return PlanResult(summary=_summarize_plan(plan), full=plan if verbose else None)
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
        detail = ""
        if alert and alert.details:
            names = ", ".join(str(d.get("name", "?")) for d in alert.details)
            detail = f" Affected: {names}."
        raise ToolError(
            f"Plan aborted before product selection — "
            f"{alert.code.value if alert else 'gate'}: "
            f"{alert.message if alert else 'constraint infeasible'}."
            f"{detail} (steps: {steps})") from e


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
            f"{alert.message if alert else 'constraint infeasible'}") from e


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
    `status` filter and before paging, so one call answers "62 products:
    2 resolved, 60 unknown"; then `status="resolved"` pages through just
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
    """Active configuration: routing strategy, models, confidence
    threshold, DB target, and whether this caller may submit origin
    readings (`write_tools`). Free — no LLM calls."""
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
    status, and the coverage floor a plan must reach before its basket
    may be called verified. Read this before describing coverage."""
    from sqlalchemy import func
    from sqlalchemy.orm import Session

    from . import db, origins
    from .config import settings

    with Session(db.engine()) as s:
        products = int(s.query(func.count(db.ProductRow.id)).scalar() or 0)
        evidence_rows = int(s.query(func.count(db.ProductOriginEvidenceRow.id)).scalar() or 0)
        queue: dict[str, int] = {
            str(status): int(n)
            for status, n in s.query(db.OriginSubmissionRow.status,
                                     func.count(db.OriginSubmissionRow.id))
            .group_by(db.OriginSubmissionRow.status).all()}
    by_status: dict[str, int] = {}
    for o in origins.resolve_all().values():
        by_status[o.status] = by_status.get(o.status, 0) + 1
    return {
        "products": products,
        "by_status": by_status,
        "evidence_rows": evidence_rows,
        "submissions": {st: int(queue.get(st, 0))
                        for st in ("pending", "approved", "rejected")},
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
        "slug with plan_recipe; otherwise pass the text to plan_from_text. "
        "If an ingredient is in doubt, call find_product for it first and "
        "read `match` — \"relaxed\" hits are alternatives the planner would "
        "only offer, not candidates.\n"
        f"2. {exclude}{budget_line}Call the plan tool once; it is slow and "
        "costs credits.\n"
        "3. Report summary.total_cost, every line (ingredient → product, store, "
        "price) and the recommended trip.\n"
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

    load_dotenv()
    server.run("stdio")


if __name__ == "__main__":
    main()

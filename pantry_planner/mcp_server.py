"""
MCP server — exposes the pantry pipeline to MCP clients (Claude
Desktop, Claude Code, any agent) as tools.

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

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field

from .models import OriginRanking, ProductOrigin, Recipe, ShoppingPlan, WeekPlan

# Tool annotations are hints for the client, not security: a read tool
# never writes, and nothing here reaches outside the seeded catalog, so
# open_world is False everywhere. Plan tools are read-only too but not
# idempotent — the LLM selector can pick differently between calls.
_READ = ToolAnnotations(read_only_hint=True, open_world_hint=False)
_PLAN = ToolAnnotations(read_only_hint=True, open_world_hint=False,
                        idempotent_hint=False)

# REST-parity input bounds (api.py): the MCP endpoint is as public as the
# REST one, so a paste, a list or a page size is capped the same way.
MAX_TEXT = 8000
MAX_LIST = 50
MAX_SEARCH = 200
MAX_IDS = 200

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
        "partial, so always report how many products were unranked."
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
                  ) -> list[ProductSummary]:
    """List the store catalog. Optional `search` filters by
    case-insensitive substring over name, brand, category and
    subcategory (recommended — the full catalog is long).
    Free — no LLM calls."""
    from . import db

    products = db.load_all_products()
    if search:
        needle = search.lower()
        products = [
            p for p in products
            if needle in f"{p.name} {p.brand} {p.category} {p.subcategory}".lower()
        ]
    return [_product_summary(p) for p in products]


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

    from . import db, origins
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
            pending_submissions=0,  # wired to origin_submissions when that table lands
            review_count=int(n_reviews),
            avg_rating=float(avg) if avg is not None else None)


# ─── Planning tools (SLOW — run the LLM pipeline) ────────────

@server.tool(title="Plan a recipe", annotations=_PLAN)
def plan_recipe(slug: str,
                exclude_origin: Annotated[list[str] | None, Field(max_length=MAX_LIST)] = None,
                preference: Annotated[list[str] | None, Field(max_length=MAX_LIST)] = None,
                ) -> ShoppingPlan:
    """Run the full shopping-plan pipeline for a seeded recipe: an LLM
    matches every ingredient to the best-value product, with a model
    router escalating hard cases. SLOW (10-60s) and costs real Claude
    API credits. Get slugs from list_recipes first.

    `exclude_origin` drops candidates positively evidenced as coming from
    those countries (e.g. ["United States"]) before the model ever sees
    them — never candidates that merely lack evidence. `preference` is soft
    guidance. The plan carries per-line provenance and `origin_coverage`:
    read its `spend_fraction` and `meets_floor` before describing a basket
    as clean, because unverified lines are not verified-clean lines."""
    from . import flow
    from .nlsearch import PlanAborted

    _check_countries(exclude_origin, preference)
    try:
        return flow.run(slug, exclude=exclude_origin, preference=preference)
    except ValueError as e:
        raise ToolError(f"{e}. Call list_recipes for valid slugs.") from e
    except PlanAborted as e:
        raise ToolError(_gate_message(e)) from e


@server.tool(title="Plan from recipe text", annotations=_PLAN)
def plan_from_text(recipe_text: Annotated[str, Field(max_length=MAX_TEXT)],
                   lat: float | None = None, lon: float | None = None,
                   exclude_origin: Annotated[list[str] | None, Field(max_length=MAX_LIST)] = None,
                   preference: Annotated[list[str] | None, Field(max_length=MAX_LIST)] = None,
                   ) -> ShoppingPlan:
    """Plan a shopping basket from PASTED RECIPE TEXT — include the full
    ingredient list (quantities optional) and any shopping notes
    (budget, dietary exclusions); lat/lon optionally set the shopping
    location. Parses the text, runs a staged SQL retrieval plan, then
    the LLM selector. SLOW (10-60s) and costs real Claude API credits."""
    _check_countries(exclude_origin, preference)
    from . import flow
    from .nlsearch import PlanAborted, UnparseableRecipe

    try:
        return flow.run_nl(recipe_text, lat=lat, lon=lon,
                           exclude=exclude_origin, preference=preference)
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
              ) -> WeekPlan:
    """Plan `days` dinners from the recipe library under an optional
    budget, rewarding ingredient overlap (a shared product is bought
    once). Returns per-day plans, the merged shopping list, overlap
    savings, and store-split trip options. exclude_tags filters out
    recipes containing those dietary tags (e.g. ["dairy", "gluten"]).
    SLOW (runs the LLM selector per day) and costs real Claude API
    credits — roughly one plan_recipe per day planned."""
    _check_countries(exclude_origin, preference)
    from . import weekplan
    from .nlsearch import PlanAborted

    try:
        return weekplan.plan_week(
            days=days, max_total_budget=max_total_budget,
            exclude_tags=exclude_tags or [], lat=lat, lon=lon,
            max_distance_km=max_distance_km,
            exclude_origin=exclude_origin, preference=preference)
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
                        ) -> list[ProductOrigin]:
    """Resolved country-of-origin evidence per product - by ids, by a name
    `search` filter, or the whole catalog. Free, no LLM calls.

    Read `status` before using `country`: only "resolved" carries usable
    evidence. "unknown" means no source published an origin, "conflicting"
    means sources disagreed and no winner was picked, "lookup_failed" means
    a source did not answer, and "guess" is a name-based hint that must not
    be treated as provenance. Coverage is thin and biased - most Canadian
    products resolve to "unknown" - so absence is never evidence of foreign
    origin."""
    from . import db, origins

    products = db.load_all_products()
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
    resolved = origins.resolve_all([p.id for p in products])
    return [resolved[p.id] for p in products if p.id in resolved]


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


# ─── Status ──────────────────────────────────────────────────

@server.tool(title="Pipeline status", annotations=_READ)
def pipeline_status() -> PipelineStatus:
    """Active configuration: routing strategy, models, confidence
    threshold, DB target. Free — no LLM calls."""
    from .config import redact_db_url, settings

    cfg = settings()
    return PipelineStatus(
        status="ok",
        routing_strategy=cfg.routing_strategy,
        default_model=cfg.selector_model_default,
        escalation_model=cfg.selector_model_escalation,
        confidence_threshold=cfg.confidence_threshold,
        db=redact_db_url(cfg.db_url),
        anthropic_key_configured=bool(cfg.anthropic_api_key),
        demo_mode=cfg.demo_mode,
    )


# ─── Transports ──────────────────────────────────────────────

def http_app():
    """Streamable HTTP ASGI app, mounted by api.py. Stateless + plain
    JSON responses: no session affinity or SSE needed behind nginx.
    DNS-rebinding protection is off because the app sits behind a
    reverse proxy whose Host header is the public hostname."""
    from mcp.server.transport_security import TransportSecuritySettings

    return server.streamable_http_app(
        streamable_http_path="/mcp",
        json_response=True,
        stateless_http=True,
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=False),
    )


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

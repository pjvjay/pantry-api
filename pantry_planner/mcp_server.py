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

import os
from pathlib import Path

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import BaseModel

from .models import OriginRanking, ProductOrigin, Recipe, ShoppingPlan, WeekPlan

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


def _product_summary(p) -> ProductSummary:
    return ProductSummary(
        id=p.id, name=p.name, brand=p.brand, category=p.category,
        subcategory=p.subcategory, price=p.price, unit_size=p.unit_size,
        dietary_tags=p.dietary_tags)


# ─── Catalog / recipe tools (free — DB reads only) ───────────

@server.tool()
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


@server.tool()
def get_recipe(slug: str) -> Recipe:
    """Fetch one seeded recipe with its full ingredient list.
    Free — no LLM calls."""
    from . import db

    try:
        return db.load_recipe(slug)
    except ValueError as e:
        raise ToolError(f"{e}. Call list_recipes for valid slugs.") from e


@server.tool()
def list_products(search: str | None = None) -> list[ProductSummary]:
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


# ─── Planning tools (SLOW — run the LLM pipeline) ────────────

@server.tool()
def plan_recipe(slug: str, exclude_origin: list[str] | None = None,
                preference: list[str] | None = None) -> ShoppingPlan:
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

    try:
        return flow.run(slug, exclude=exclude_origin, preference=preference)
    except ValueError as e:
        raise ToolError(f"{e}. Call list_recipes for valid slugs.") from e


@server.tool()
def plan_from_text(recipe_text: str, lat: float | None = None,
                   lon: float | None = None,
                   exclude_origin: list[str] | None = None,
                   preference: list[str] | None = None) -> ShoppingPlan:
    """Plan a shopping basket from PASTED RECIPE TEXT — include the full
    ingredient list (quantities optional) and any shopping notes
    (budget, dietary exclusions); lat/lon optionally set the shopping
    location. Parses the text, runs a staged SQL retrieval plan, then
    the LLM selector. SLOW (10-60s) and costs real Claude API credits."""
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


@server.tool()
def plan_week(days: int = 5, max_total_budget: float | None = None,
              exclude_tags: list[str] | None = None,
              lat: float | None = None, lon: float | None = None,
              max_distance_km: float | None = None,
              exclude_origin: list[str] | None = None,
              preference: list[str] | None = None) -> WeekPlan:
    """Plan `days` dinners from the recipe library under an optional
    budget, rewarding ingredient overlap (a shared product is bought
    once). Returns per-day plans, the merged shopping list, overlap
    savings, and store-split trip options. exclude_tags filters out
    recipes containing those dietary tags (e.g. ["dairy", "gluten"]).
    SLOW (runs the LLM selector per day) and costs real Claude API
    credits — roughly one plan_recipe per day planned."""
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

@server.tool()
def get_product_origins(product_ids: list[int] | None = None,
                        search: str | None = None) -> list[ProductOrigin]:
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


@server.tool()
def rank_products_by_origin(preference: list[str] | None = None,
                            exclude: list[str] | None = None,
                            search: str | None = None) -> OriginRanking:
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
    from . import db, origins

    products = db.load_all_products()
    if search:
        needle = search.lower()
        products = [p for p in products
                    if needle in f"{p.name} {p.brand} {p.category}".lower()]
    return origins.rank_products(
        products, preference=preference or [], exclude=exclude or [])


@server.tool()
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

@server.tool()
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

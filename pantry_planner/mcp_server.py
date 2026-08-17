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

from .models import ProductOrigin, Recipe, ShoppingPlan, WeekPlan

server = MCPServer(
    name="pantry-planner",
    version="0.1.0",
    instructions=(
        "Grocery planning over a seeded Canadian store catalog. Browse "
        "recipes and products for free; plan_recipe / plan_from_text run "
        "the full LLM pipeline (slow, costs API credits) and return a "
        "priced shopping plan. Origin tools resolve where products come "
        "from — heuristics first, one cached LLM call for the rest."
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


class OriginMatch(BaseModel):
    """A product paired with its resolved country of origin."""
    product_id: int
    product_name: str
    brand: str
    category: str | None
    price: float
    country: str
    confidence: float
    source: str


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
def plan_recipe(slug: str) -> ShoppingPlan:
    """Run the full shopping-plan pipeline for a seeded recipe: an LLM
    matches every ingredient to the best-value product, with a model
    router escalating hard cases. SLOW (10-60s) and costs real Claude
    API credits. Get slugs from list_recipes first."""
    from . import flow

    try:
        return flow.run(slug)
    except ValueError as e:
        raise ToolError(f"{e}. Call list_recipes for valid slugs.") from e


@server.tool()
def plan_from_text(recipe_text: str, lat: float | None = None,
                   lon: float | None = None) -> ShoppingPlan:
    """Plan a shopping basket from PASTED RECIPE TEXT — include the full
    ingredient list (quantities optional) and any shopping notes
    (budget, dietary exclusions); lat/lon optionally set the shopping
    location. Parses the text, runs a staged SQL retrieval plan, then
    the LLM selector. SLOW (10-60s) and costs real Claude API credits."""
    from . import flow
    from .nlsearch import PlanAborted, UnparseableRecipe

    try:
        return flow.run_nl(recipe_text, lat=lat, lon=lon)
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
              max_distance_km: float | None = None) -> WeekPlan:
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
            max_distance_km=max_distance_km)
    except PlanAborted as e:
        alert = e.execution.aborted
        raise ToolError(
            f"Week plan aborted — "
            f"{alert.code.value if alert else 'gate'}: "
            f"{alert.message if alert else 'constraint infeasible'}") from e


# ─── Country-of-origin tools ─────────────────────────────────

@server.tool()
def get_product_origins(product_ids: list[int] | None = None,
                        search: str | None = None,
                        allow_llm: bool = True) -> list[ProductOrigin]:
    """Resolve the country of origin for products — by ids, by a name
    `search` filter, or the whole catalog when neither is given.
    Cheapest tier wins: DB cache, then free deterministic rules, then
    ONE batch Haiku call for the remainder (cached afterwards, so the
    catalog costs at most one small call ever). Set allow_llm=false to
    guarantee zero cost (unruled products come back "Unknown")."""
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
        products = [
            p for p in products
            if needle in f"{p.name} {p.brand} {p.category} {p.subcategory}".lower()
        ]
    return origins.resolve_origins(products, allow_llm=allow_llm)


@server.tool()
def search_products_by_origin(country: str,
                              allow_llm: bool = False) -> list[OriginMatch]:
    """Find catalog products from a given country ("Canada", "Italy",
    "India", ...; case-insensitive substring). Free by default — only
    cached and rule-based origins are searched. Set allow_llm=true to
    first resolve the rest of the catalog with one batch Haiku call."""
    from . import origins

    matches = origins.search_by_origin(country, allow_llm=allow_llm)
    return [
        OriginMatch(
            product_id=p.id, product_name=p.name, brand=p.brand,
            category=p.category, price=p.price, country=o.country,
            confidence=o.confidence, source=o.source)
        for p, o in matches
    ]


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

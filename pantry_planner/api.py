"""
Thin FastAPI wrapper — turns the Burr flow into an HTTP endpoint.

Deliberately minimal: this repo is about the pipeline design, not the
web layer. Every route delegates to pantry_planner.flow or pantry_planner.db.
"""
from __future__ import annotations

import contextlib
import os

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from . import db, flow
from .config import settings
from .models import (
    OriginRanking,
    Product,
    ProductOrigin,
    Recipe,
    ShoppingPlan,
    WeekPlan,
)

# The MCP Streamable-HTTP endpoint rides this app at /mcp (mounted at
# the bottom of the file). MCP_HTTP_ENABLED=false turns it off — the
# endpoint shares the API's no-auth posture, and plan tools spend
# Anthropic credits.
MCP_HTTP_ENABLED = os.environ.get("MCP_HTTP_ENABLED", "true").lower() != "false"


@contextlib.asynccontextmanager
async def _lifespan(app: FastAPI):
    # Starlette never runs a mounted sub-app's lifespan, so the MCP
    # session manager must be driven from here — without it every /mcp
    # request 500s even though everything imports cleanly.
    if MCP_HTTP_ENABLED:
        from .mcp_server import server as mcp_server

        async with mcp_server.session_manager.run():
            yield
    else:
        yield


app = FastAPI(
    title="pantry-planner",
    version="0.1.0",
    description=(
        "Match recipe ingredients to store products with an LLM-driven pipeline. "
        "Toggle routing strategy via ROUTING_STRATEGY env var."
    ),
    lifespan=_lifespan,
)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "routing_strategy": settings().routing_strategy,
        "default_model": settings().selector_model_default,
        "escalation_model": settings().selector_model_escalation,
        "confidence_threshold": settings().confidence_threshold,
        "demo_mode": settings().demo_mode,
    }


@app.get("/recipes", response_model=list[Recipe])
def list_recipes() -> list[Recipe]:
    return db.load_all_recipes()


@app.get("/recipes/{slug}", response_model=Recipe)
def get_recipe(slug: str) -> Recipe:
    try:
        return db.load_recipe(slug)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/products", response_model=list[Product])
def list_products() -> list[Product]:
    return db.load_all_products()


class NLPlanRequest(BaseModel):
    """The FULL pasted recipe text (+ optional inline shopping notes).
    lat/lon: optional shopping location for the distance constraint;
    defaults to the configured reference point."""

    recipe_text: str = Field(max_length=8000)   # public endpoint: bound the paste
    lat: float | None = None
    lon: float | None = None
    # Provenance: exclude removes candidates positively evidenced as coming
    # from these countries; preference is soft guidance to the selector.
    exclude_origin: list[str] = Field(default_factory=list, max_length=50)
    preference: list[str] = Field(default_factory=list, max_length=50)


@app.post("/plan/nl", response_model=ShoppingPlan)
def plan_nl(req: NLPlanRequest) -> ShoppingPlan:
    """NL2SQL path: parse a pasted recipe, execute the staged query plan
    (existence → options → brand stats → lookups), route, select. Returns
    the plan + interpretation + the full per-step SQL trace. A gate abort
    returns 409 with the alert and the trace up to the failed step."""
    from .nlsearch import PlanAborted, UnparseableRecipe

    try:
        return flow.run_nl(req.recipe_text, lat=req.lat, lon=req.lon,
                           exclude=req.exclude_origin,
                           preference=req.preference)
    except UnparseableRecipe:
        raise HTTPException(status_code=422, detail=(
            "Couldn't find an ingredient list in that text. Paste a recipe "
            "with its ingredients (quantities optional), e.g.:\n"
            "Spaghetti Bolognese (serves 4)\n"
            "- 400g spaghetti\n- 500g ground beef\n- 1 can crushed tomatoes\n"
            "Notes: under $30, no dairy"))
    except PlanAborted as e:
        raise HTTPException(status_code=409, detail=e.execution.model_dump(mode="json"))


class WeekPlanRequest(BaseModel):
    """Plan `days` dinners from the recipe library under an optional budget.
    Deterministic menu selection (marginal-cost greedy over one batched
    retrieval); the LLM only does the per-day product mapping."""

    days: int = Field(5, ge=1, le=14)
    max_total_budget: float | None = None
    exclude_tags: list[str] = []
    lat: float | None = None
    lon: float | None = None
    max_distance_km: float | None = None
    exclude_origin: list[str] = Field(default_factory=list, max_length=50)
    preference: list[str] = Field(default_factory=list, max_length=50)


@app.post("/plan/week", response_model=WeekPlan)
def plan_week(req: WeekPlanRequest) -> WeekPlan:
    """5A: weekly menu optimizer. Rewards ingredient overlap exactly (a
    shared product costs $0 marginal), gates on the cheapest-basket floor
    before any LLM spend, and prices the merged basket's store split."""
    from . import weekplan
    from .nlsearch import PlanAborted

    try:
        return weekplan.plan_week(
            days=req.days, max_total_budget=req.max_total_budget,
            exclude_tags=req.exclude_tags, lat=req.lat, lon=req.lon,
            max_distance_km=req.max_distance_km,
            exclude_origin=req.exclude_origin, preference=req.preference)
    except PlanAborted as e:
        raise HTTPException(status_code=409, detail=e.execution.model_dump(mode="json"))


@app.post("/plan/{slug}", response_model=ShoppingPlan)
def plan_recipe(slug: str, exclude_origin: list[str] | None = Query(default=None),
                preference: list[str] | None = Query(default=None)) -> ShoppingPlan:
    """Run the pipeline for one recipe. Returns the shopping plan.

    `exclude_origin` removes candidates positively evidenced as coming from
    those countries — never candidates that merely lack evidence. The
    returned plan carries per-line provenance and a spend-weighted coverage
    figure saying how much of the basket was actually checked."""
    try:
        return flow.run(slug, exclude=exclude_origin, preference=preference)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ─── Provenance ──────────────────────────────────────────────

class RankRequest(BaseModel):
    """Rank the catalog against a country preference the CALLER supplies.

    preference: ordered, most-preferred first (e.g. ["Canada", "Mexico"]).
    exclude:    countries to filter out. A product is only ever excluded on
                positive evidence — never for lacking any.
    """

    preference: list[str] = Field(default_factory=list, max_length=50)
    exclude: list[str] = Field(default_factory=list, max_length=50)
    search: str | None = Field(default=None, max_length=200)
    product_ids: list[int] | None = None


@app.get("/origins", response_model=list[ProductOrigin])
def list_origins(search: str | None = None) -> list[ProductOrigin]:
    """Resolved provenance per product, from ingested evidence only."""
    from . import origins

    products = db.load_all_products()
    if search:
        needle = search.lower()
        products = [p for p in products
                    if needle in f"{p.name} {p.brand} {p.category}".lower()]
    resolved = origins.resolve_all([p.id for p in products])
    return [resolved[p.id] for p in products if p.id in resolved]


@app.post("/origins/rank", response_model=OriginRanking)
def rank_by_origin(req: RankRequest) -> OriginRanking:
    """Rank products by provenance against the caller's preference order.

    Returns four things, kept apart on purpose: ranked, excluded,
    unranked (no evidence / conflicting / lookup failed) and the counts.
    Unverified products are never folded into the ranking — with coverage
    as thin as it is, that would present silence as a verdict.
    """
    from . import origins

    products = db.load_all_products()
    if req.product_ids is not None:
        wanted = set(req.product_ids)
        products = [p for p in products if p.id in wanted]
    elif req.search:
        needle = req.search.lower()
        products = [p for p in products
                    if needle in f"{p.name} {p.brand} {p.category}".lower()]
    return origins.rank_products(
        products, preference=req.preference, exclude=req.exclude)


@app.get("/origins/triage")
def origin_triage() -> list[dict]:
    """Products worth photographing next. Hints, never origins."""
    from . import origins

    return origins.triage_candidates(db.load_all_products())


# MCP Streamable HTTP — mounted last so the REST routes above keep
# priority; the sub-app serves exactly /mcp (public:
# https://<host>/pantry/api/mcp). Mounting also instantiates the
# session manager that _lifespan drives.
if MCP_HTTP_ENABLED:
    from .mcp_server import http_app

    app.mount("/", http_app())

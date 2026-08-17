"""
Thin FastAPI wrapper — turns the Burr flow into an HTTP endpoint.

Deliberately minimal: this repo is about the pipeline design, not the
web layer. Every route delegates to pantry_planner.flow or pantry_planner.db.
"""
from __future__ import annotations

import contextlib
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import db, flow
from .config import settings
from .models import Product, Recipe, ShoppingPlan, WeekPlan

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


@app.post("/plan/nl", response_model=ShoppingPlan)
def plan_nl(req: NLPlanRequest) -> ShoppingPlan:
    """NL2SQL path: parse a pasted recipe, execute the staged query plan
    (existence → options → brand stats → lookups), route, select. Returns
    the plan + interpretation + the full per-step SQL trace. A gate abort
    returns 409 with the alert and the trace up to the failed step."""
    from .nlsearch import PlanAborted, UnparseableRecipe

    try:
        return flow.run_nl(req.recipe_text, lat=req.lat, lon=req.lon)
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
            max_distance_km=req.max_distance_km)
    except PlanAborted as e:
        raise HTTPException(status_code=409, detail=e.execution.model_dump(mode="json"))


@app.post("/plan/{slug}", response_model=ShoppingPlan)
def plan_recipe(slug: str) -> ShoppingPlan:
    """Run the pipeline for one recipe. Returns the shopping plan."""
    try:
        return flow.run(slug)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# MCP Streamable HTTP — mounted last so the REST routes above keep
# priority; the sub-app serves exactly /mcp (public:
# https://<host>/pantry/api/mcp). Mounting also instantiates the
# session manager that _lifespan drives.
if MCP_HTTP_ENABLED:
    from .mcp_server import http_app

    app.mount("/", http_app())

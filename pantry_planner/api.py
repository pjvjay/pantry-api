"""
Thin FastAPI wrapper — turns the Burr flow into an HTTP endpoint.

Deliberately minimal: this repo is about the pipeline design, not the
web layer. Every route delegates to pantry_planner.flow or pantry_planner.db.
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from . import db, flow
from .config import settings
from .models import Product, Recipe, ShoppingPlan, WeekPlan

app = FastAPI(
    title="pantry-planner",
    version="0.1.0",
    description=(
        "Match recipe ingredients to store products with an LLM-driven pipeline. "
        "Toggle routing strategy via ROUTING_STRATEGY env var."
    ),
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

    recipe_text: str
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

    days: int = 5
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

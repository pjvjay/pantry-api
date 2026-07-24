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
from .models import Product, Recipe, ShoppingPlan

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
    }


@app.get("/recipes", response_model=list[Recipe])
def list_recipes() -> list[Recipe]:
    from .db import RecipeRow, engine
    from sqlalchemy.orm import Session

    with Session(engine()) as s:
        rows = s.query(RecipeRow).all()
        return [db.load_recipe(r.slug) for r in rows]


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
    """The FULL pasted recipe text (+ optional inline shopping notes)."""

    recipe_text: str


@app.post("/plan/nl", response_model=ShoppingPlan)
def plan_nl(req: NLPlanRequest) -> ShoppingPlan:
    """NL2SQL path: parse a pasted recipe, retrieve narrowed candidates via
    templated SQL, route, select. Returns the plan + interpretation + the
    executed SQL for transparency."""
    from .nlsearch import UnparseableRecipe

    try:
        return flow.run_nl(req.recipe_text)
    except UnparseableRecipe:
        raise HTTPException(status_code=422, detail=(
            "Couldn't find an ingredient list in that text. Paste a recipe "
            "with its ingredients (quantities optional), e.g.:\n"
            "Spaghetti Bolognese (serves 4)\n"
            "- 400g spaghetti\n- 500g ground beef\n- 1 can crushed tomatoes\n"
            "Notes: under $30, no dairy"))


@app.post("/plan/{slug}", response_model=ShoppingPlan)
def plan_recipe(slug: str) -> ShoppingPlan:
    """Run the pipeline for one recipe. Returns the shopping plan."""
    try:
        return flow.run(slug)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

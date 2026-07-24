"""Typed contracts for the NL2SQL stage."""
from __future__ import annotations

from pydantic import BaseModel, Field

from ..models import Product, Recipe


class IngredientSpec(BaseModel):
    name: str                              # base or compound product name: "tomato sauce"
    form: str | None = None                # PURCHASE form ("canned") — a required match token
    prep: str | None = None                # cook's prep ("mashed") — display/selector only
    quantity: float | None = None
    unit: str | None = None
    category_hint: str | None = None       # extractor's guess; vocab-clamped later


class RecipeSpec(BaseModel):
    title: str = "Untitled recipe"
    servings: int = 1
    ingredients: list[IngredientSpec] = Field(default_factory=list)


class Constraints(BaseModel):
    max_item_price: float | None = None
    max_total_budget: float | None = None
    exclude_tags: list[str] = Field(default_factory=list)
    require_tags: list[str] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)
    exclude_categories: list[str] = Field(default_factory=list)
    subcategories: list[str] = Field(default_factory=list)
    exclude_subcategories: list[str] = Field(default_factory=list)
    soft_text: str = ""


class ParsedInput(BaseModel):
    recipe: RecipeSpec = Field(default_factory=RecipeSpec)
    constraints: Constraints = Field(default_factory=Constraints)
    ignored: list[str] = Field(default_factory=list)   # values dropped by validation
    cost_usd: float = 0.0
    latency_ms: int = 0
    error: str | None = None

    def display_lines(self) -> list[str]:
        """The UI's 'Interpreted as:' chips."""
        out = [
            f"{self.recipe.title} · serves {self.recipe.servings} · "
            f"{len(self.recipe.ingredients)} ingredients"
        ]
        c = self.constraints
        if c.max_total_budget is not None:
            out.append(f"budget ≤ ${c.max_total_budget:.2f}")
        if c.max_item_price is not None:
            out.append(f"per item ≤ ${c.max_item_price:.2f}")
        out += [f"no {t}" for t in c.exclude_tags]
        out += [f"only {t}" for t in c.require_tags]
        out += [f"skip {s}" for s in (*c.exclude_categories, *c.exclude_subcategories)]
        if c.soft_text:
            out.append(f"prefer: {c.soft_text}")
        out += [f"ignored: {v}" for v in self.ignored]
        return out


class RetrievalStats(BaseModel):
    """Per-ingredient pool statistics — feed Phase A of the model router."""

    pool_sizes: list[int] = Field(default_factory=list)
    zero_hit_ingredients: int = 0          # needed the widening ladder
    value_disagreement: float = 0.0        # fraction of pools: cheapest != best unit value
    catalog_size: int = 0

    @property
    def mean_pool_size(self) -> float:
        return sum(self.pool_sizes) / len(self.pool_sizes) if self.pool_sizes else 0.0

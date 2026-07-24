"""Orchestration: parse → clamp → compile → execute → widen → stats."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from sqlalchemy import text
from sqlalchemy.orm import Session

from .. import db
from ..models import Product, Recipe, RecipeIngredient
from .query_parser import parse_input, validate_parsed
from .schemas import ParsedInput, RecipeSpec, RetrievalStats
from .sql_builder import build_fallback_sql, build_retrieval_sql, inline_for_display
from .vocab import db_vocab, subcat_parents


class UnparseableRecipe(Exception):
    """No ingredient list could be extracted — nothing to plan."""


@dataclass
class RetrievalResult:
    recipe: Recipe
    products: list[Product]                    # deduped union of all pools
    pools: dict[int, list[Product]]            # ingredient_no -> candidates
    stats: RetrievalStats
    parsed: ParsedInput
    sql_display: str


def _slugify(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_") or "adhoc_recipe"


def build_recipe(spec: RecipeSpec) -> Recipe:
    """The plan's Recipe comes from the pasted text, not the DB."""
    if not spec.ingredients:
        raise UnparseableRecipe()
    return Recipe(
        slug=_slugify(spec.title),
        name=spec.title,
        servings=spec.servings,
        ingredients=[
            RecipeIngredient(line_no=i + 1, name=ing.name, category=ing.category_hint)
            for i, ing in enumerate(spec.ingredients)
        ],
    )


def _row_to_product(row) -> Product:
    return Product(id=row["id"], name=row["name"], description=row["description"],
                   price=row["price"], category=row["category"],
                   subcategory=row["subcategory"] or None,
                   dietary_tags=row["dietary_tags"] or "",
                   unit_size=row["unit_size"] or "",
                   unit_qty=row["unit_qty"], unit_uom=row["unit_uom"] or "")


def _value_disagreement(pools: dict[int, list[Product]]) -> float:
    """Per-pool: does the cheapest candidate differ from the best unit-value one?"""
    checked = disagreed = 0
    for pool in pools.values():
        sized = [p for p in pool if p.unit_qty]
        if len(sized) < 2:
            continue
        checked += 1
        cheapest = min(sized, key=lambda p: p.price)
        best_value = min(sized, key=lambda p: p.price / p.unit_qty)
        if cheapest.id != best_value.id:
            disagreed += 1
    return disagreed / checked if checked else 0.0


def retrieve(text_input: str, *, parsed: ParsedInput | None = None) -> RetrievalResult:
    """Full retrieval. `parsed` injectable for tests (skips the LLM call)."""
    parsed = validate_parsed(parsed or parse_input(text_input), db_vocab())
    recipe = build_recipe(parsed.recipe)            # raises UnparseableRecipe -> 422
    ingredients = parsed.recipe.ingredients

    sql, params = build_retrieval_sql(parsed.constraints, ingredients)
    pools: dict[int, list[Product]] = {}
    with Session(db.engine()) as s:
        for row in s.execute(text(sql), params).mappings():
            pools.setdefault(row["ingredient_no"], []).append(_row_to_product(row))

        # POST-PROCESSING: execution-guided widening for zero-hit ingredients
        missed = [(n, ing) for n, ing in enumerate(ingredients) if not pools.get(n)]
        if missed:
            fb_sql, fb_params = build_fallback_sql(
                parsed.constraints, missed, subcat_parents())
            rung_rows: dict[int, dict[int, list[Product]]] = {}
            for row in s.execute(text(fb_sql), fb_params).mappings():
                rung_rows.setdefault(row["ingredient_no"], {}) \
                         .setdefault(row["rung"], []).append(_row_to_product(row))
            for n, by_rung in rung_rows.items():
                pools[n] = by_rung.get(1) or by_rung.get(2) or []
        catalog_size = s.execute(text("SELECT COUNT(*) FROM products")).scalar_one()

    seen: set[int] = set()
    products = [p for pool in pools.values() for p in pool
                if p.id not in seen and not seen.add(p.id)]

    stats = RetrievalStats(
        pool_sizes=[len(pools.get(n, [])) for n in range(len(ingredients))],
        zero_hit_ingredients=len(missed),
        value_disagreement=_value_disagreement(pools),
        catalog_size=catalog_size,
    )
    return RetrievalResult(recipe=recipe, products=products, pools=pools,
                           stats=stats, parsed=parsed,
                           sql_display=inline_for_display(sql, params))

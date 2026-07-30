"""The query-plan executor: build_plan (deterministic) + execute_plan.

Replaces the old retrieve.py orchestration. Retrieval is now the formal
sequence t1 existence -> t2 options -> t3 brand stats (-> t4 substitute
lookups, appended when a pool comes back thin), with hard abort gates:

    t1  missing_ingredients            we don't stock it at all
    t2  unavailable_within_constraints stocked, but not within distance/budget
    t2  budget_infeasible              cheapest possible basket > budget

On abort, PlanAborted carries the full PlanExecution (every step's SQL,
rows, timing) plus a PlanAlert saying which ingredients failed, why, and
what to try instead. The API maps it to a 409.
"""
from __future__ import annotations

import math
import re
import time
from collections import Counter
from dataclasses import dataclass, field

from sqlalchemy import text
from sqlalchemy.orm import Session

from .. import db
from ..models import Product, Recipe, RecipeIngredient
from .plan import (GateCode, PlanAlert, PlanExecution, QueryPlan, QueryStep,
                   StepKind, StepResult)
from .query_parser import parse_input, validate_parsed
from .schemas import ParsedInput, RecipeSpec, RetrievalStats
from .sql_builder import (THIN_POOL, build_existence_sql, build_options_sql,
                          build_stats_sql, build_substitute_sql,
                          build_suggestions_sql, inline_for_display)
from .vocab import db_vocab


class UnparseableRecipe(Exception):
    """No ingredient list could be extracted — nothing to plan."""


class PlanAborted(Exception):
    """A gate fired. `.execution.aborted` is the user-facing PlanAlert."""

    def __init__(self, execution: PlanExecution):
        self.execution = execution
        super().__init__(execution.aborted.message if execution.aborted else "aborted")


@dataclass
class PlanRunResult:
    recipe: Recipe
    products: list[Product]                    # deduped union of all pools
    pools: dict[int, list[Product]]            # ingredient_no -> candidates
    stats: RetrievalStats
    parsed: ParsedInput
    execution: PlanExecution
    brand_stats: dict[str, list[dict]] = field(default_factory=dict)  # ing name -> per-brand rows
    # resolved shopping location — downstream steps (trip optimizer) reuse it
    lat: float = 0.0
    lon: float = 0.0
    max_km: float | None = None


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


def _row_to_product(row, *, substitute: bool = False) -> Product:
    return Product(id=row["id"], name=row["name"], description=row["description"],
                   price=row["price"], category=row["category"],
                   subcategory=row["subcategory"] or None,
                   dietary_tags=row["dietary_tags"] or "",
                   unit_size=row["unit_size"] or "",
                   unit_qty=row["unit_qty"], unit_uom=row["unit_uom"] or "",
                   brand=row["brand"] or "",
                   store_name=row["store_name"] or "",
                   store_price=row["store_price"],
                   distance_km=round(math.sqrt(max(row["dist_km2"], 0.0)), 1),
                   substitute=substitute)


def _value_disagreement(pools: dict[int, list[Product]]) -> float:
    """Per-pool: does the cheapest candidate differ from the best unit-value one?"""
    checked = disagreed = 0
    for pool in pools.values():
        sized = [p for p in pool if p.unit_qty and p.store_price is not None]
        if len(sized) < 2:
            continue
        checked += 1
        cheapest = min(sized, key=lambda p: p.store_price)
        best_value = min(sized, key=lambda p: p.store_price / p.unit_qty)
        if cheapest.id != best_value.id:
            disagreed += 1
    return disagreed / checked if checked else 0.0


# ─── plan composition (deterministic — pure function of the parse) ───

def build_plan(parsed: ParsedInput, *, max_km: float | None) -> QueryPlan:
    c = parsed.constraints
    n_ing = len(parsed.recipe.ingredients)
    steps = [
        QueryStep(id="t1_existence", kind=StepKind.existence,
                  template="existence_probe",
                  params_summary={"ingredients": n_ing},
                  gate=GateCode.missing_ingredients),
        QueryStep(id="t2_options", kind=StepKind.options,
                  template="options_single_pass",
                  params_summary={k: v for k, v in {
                      "ingredients": n_ing,
                      "max_km": max_km,
                      "max_item_price": c.max_item_price,
                      "max_total_budget": c.max_total_budget,
                  }.items() if v is not None},
                  gate=GateCode.unavailable_within_constraints),
        QueryStep(id="t3_statistics", kind=StepKind.statistics,
                  template="brand_stats",
                  params_summary={"group_by": "brand"}),
    ]
    # t4 lookup steps are appended during execution when a pool is thin —
    # data-driven, never model-driven.
    return QueryPlan(steps=steps)


# ─── execution ───────────────────────────────────────────────

def _timed(session: Session, sql: str, params: dict) -> tuple[list, int]:
    t0 = time.perf_counter()
    rows = list(session.execute(text(sql), params).mappings())
    return rows, int((time.perf_counter() - t0) * 1000)


def _abort(execution: PlanExecution, alert: PlanAlert) -> None:
    execution.aborted = alert
    raise PlanAborted(execution)


def _brand_regroup(pools: dict[int, list[Product]],
                   ingredients, stat_rows: list) -> dict[str, list[dict]]:
    """t3 rows (per product) -> per ingredient, per brand aggregates."""
    by_id = {r["product_id"]: r for r in stat_rows}
    out: dict[str, list[dict]] = {}
    for n, pool in pools.items():
        groups: dict[str, list] = {}
        for p in pool:
            r = by_id.get(p.id)
            if r is not None:
                groups.setdefault(p.brand or "unbranded", []).append(r)
        brand_rows = []
        for brand, rows in sorted(groups.items()):
            rated = [r for r in rows if r["avg_rating"] is not None]
            brand_rows.append({
                "brand": brand,
                "options": len(rows),
                "avg_price": round(sum(r["avg_price"] for r in rows) / len(rows), 2),
                "min_price": round(min(r["min_price"] for r in rows), 2),
                "avg_rating": (round(sum(r["avg_rating"] for r in rated) / len(rated), 1)
                               if rated else None),
                "review_count": sum(r["review_count"] or 0 for r in rows),
            })
        out[ingredients[n].name] = brand_rows
    return out


def execute_plan(parsed: ParsedInput, plan: QueryPlan, *,
                 lat: float, lon: float, max_km: float | None) -> PlanRunResult:
    recipe = build_recipe(parsed.recipe)        # raises UnparseableRecipe -> 422
    ingredients = parsed.recipe.ingredients
    c = parsed.constraints
    execution = PlanExecution()

    with Session(db.engine()) as s:
        # ── t1: existence probe ──
        sql, params = build_existence_sql(ingredients)
        rows, ms = _timed(s, sql, params)
        counts = {r["ing_no"]: (r["strict_matches"], r["relaxed_matches"]) for r in rows}
        missing = [n for n in range(len(ingredients))
                   if counts.get(n, (0, 0))[1] == 0]
        relaxed = {n for n, (strict, rel) in counts.items() if strict == 0 and rel > 0}
        step = StepResult(
            step_id="t1_existence", kind=StepKind.existence,
            sql_display=inline_for_display(sql, params),
            row_count=len(rows), duration_ms=ms,
            outcome="aborted" if missing else "ok",
            label=(f"{len(ingredients) - len(missing)}/{len(ingredients)} ingredients stocked"
                   + (f" ({len(relaxed)} via form relaxation)" if relaxed else "")))
        execution.steps.append(step)
        if missing:
            details = []
            for n in missing:
                ing = ingredients[n]
                suggestions: list[str] = []
                if ing.category_hint:
                    sg_sql, sg_params = build_suggestions_sql(ing.category_hint.lower())
                    suggestions = [f"{r['name']} (${r['price']:.2f})"
                                   for r in s.execute(text(sg_sql), sg_params).mappings()]
                details.append({"name": ing.name,
                                "reason": "not in the catalog",
                                "suggestions": suggestions})
            _abort(execution, PlanAlert(
                stage="t1_existence", code=GateCode.missing_ingredients,
                message=("We don't stock: "
                         + ", ".join(ingredients[n].name for n in missing)
                         + ". Remove or substitute them and retry."),
                details=details))

        # ── t2: options under constraints ──
        sql, params = build_options_sql(c, ingredients, relaxed, lat, lon, max_km)
        rows, ms = _timed(s, sql, params)
        pools: dict[int, list[Product]] = {}
        for row in rows:
            pools.setdefault(row["ing_no"], []).append(_row_to_product(row))
        empty = [n for n in range(len(ingredients)) if not pools.get(n)]
        step = StepResult(
            step_id="t2_options", kind=StepKind.options,
            sql_display=inline_for_display(sql, params),
            row_count=len(rows), duration_ms=ms,
            outcome="aborted" if empty else "ok",
            label=f"{len(rows)} offers across {len(pools)} ingredient pools")
        execution.steps.append(step)
        if empty:
            # Constraint attribution: same template, distance + price caps
            # stripped. Anything that reappears was priced/located out, not
            # missing (t1 already proved it exists).
            relaxed_c = c.model_copy(update={"max_item_price": None})
            probe_sql, probe_params = build_options_sql(
                relaxed_c, ingredients, relaxed, lat, lon, None)
            probe_pools: dict[int, list[Product]] = {}
            for row in s.execute(text(probe_sql), probe_params).mappings():
                probe_pools.setdefault(row["ing_no"], []).append(_row_to_product(row))
            details = []
            for n in empty:
                alt = probe_pools.get(n)
                if alt:
                    best = alt[0]
                    reason = (f"available only outside the constraints — e.g. "
                              f"{best.name} at {best.store_name} "
                              f"(${best.store_price:.2f}, {best.distance_km} km)")
                else:
                    reason = "no offer passes the dietary/category constraints"
                details.append({"name": ingredients[n].name, "reason": reason,
                                "suggestions": []})
            _abort(execution, PlanAlert(
                stage="t2_options", code=GateCode.unavailable_within_constraints,
                message=("No options within the current constraints for: "
                         + ", ".join(ingredients[n].name for n in empty)
                         + ". Widen the distance/budget or relax a filter."),
                details=details))

        # Basket feasibility floor — pure Python over the fetched pools.
        if c.max_total_budget is not None:
            floor = sum(min(p.store_price for p in pool) for pool in pools.values())
            if floor > c.max_total_budget:
                execution.steps[-1].outcome = "aborted"
                _abort(execution, PlanAlert(
                    stage="t2_options", code=GateCode.budget_infeasible,
                    message=(f"Cheapest possible basket is ${floor:.2f} — over the "
                             f"${c.max_total_budget:.2f} budget. Raise the budget or "
                             "trim the recipe."),
                    details=[{"name": ingredients[n].name,
                              "reason": f"cheapest option ${min(p.store_price for p in pool):.2f}",
                              "suggestions": []}
                             for n, pool in sorted(pools.items())]))

        # ── t3: per-brand statistics over the pooled products ──
        pool_ids = sorted({p.id for pool in pools.values() for p in pool})
        sql, params = build_stats_sql(pool_ids)
        stat_rows, ms = _timed(s, sql, params)
        brand_stats = _brand_regroup(pools, ingredients, stat_rows)
        n_groups = sum(len(v) for v in brand_stats.values())
        execution.steps.append(StepResult(
            step_id="t3_statistics", kind=StepKind.statistics,
            sql_display=inline_for_display(sql, params),
            row_count=len(stat_rows), duration_ms=ms,
            label=f"{n_groups} brand groups across {len(pool_ids)} products"))

        # ── t4: substitute lookups for thin pools (data-driven) ──
        for n in sorted(pools):
            pool = pools[n]
            if len(pool) >= THIN_POOL:
                continue
            subcats = Counter(p.subcategory for p in pool if p.subcategory)
            if not subcats:
                continue
            sub = subcats.most_common(1)[0][0]
            sql, params = build_substitute_sql(
                c, sub, [p.id for p in pool], lat, lon, max_km)
            rows, ms = _timed(s, sql, params)
            subs = [_row_to_product(r, substitute=True) for r in rows]
            pools[n] = [*pool, *subs]
            plan.steps.append(QueryStep(
                id=f"t4_lookup_{n}", kind=StepKind.lookup,
                template="substitute_lookup",
                params_summary={"ingredient": ingredients[n].name, "subcategory": sub}))
            execution.steps.append(StepResult(
                step_id=f"t4_lookup_{n}", kind=StepKind.lookup,
                sql_display=inline_for_display(sql, params),
                row_count=len(rows), duration_ms=ms,
                label=f"{len(subs)} substitutes for {ingredients[n].name} ({sub})"))

        catalog_size = s.execute(text("SELECT COUNT(*) FROM products")).scalar_one()

    seen: set[int] = set()
    products = [p for pool in pools.values() for p in pool
                if p.id not in seen and not seen.add(p.id)]
    stats = RetrievalStats(
        pool_sizes=[len(pools.get(n, [])) for n in range(len(ingredients))],
        zero_hit_ingredients=len(relaxed),
        value_disagreement=_value_disagreement(pools),
        catalog_size=catalog_size,
    )
    return PlanRunResult(recipe=recipe, products=products, pools=pools,
                         stats=stats, parsed=parsed, execution=execution,
                         brand_stats=brand_stats, lat=lat, lon=lon, max_km=max_km)


def run_query_plan(text_input: str, *, parsed: ParsedInput | None = None,
                   lat: float | None = None, lon: float | None = None) -> PlanRunResult:
    """Full NL2SQL retrieval. `parsed` injectable for tests (skips the LLM)."""
    from ..config import settings

    cfg = settings()
    parsed = validate_parsed(parsed or parse_input(text_input), db_vocab())
    max_km = parsed.constraints.max_distance_km
    plan = build_plan(parsed, max_km=max_km)
    return execute_plan(parsed, plan,
                        lat=lat if lat is not None else cfg.default_lat,
                        lon=lon if lon is not None else cfg.default_lon,
                        max_km=max_km)

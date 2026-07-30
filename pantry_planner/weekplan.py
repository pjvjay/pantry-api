"""5A: deterministic weekly menu optimizer over the seeded recipe library.

"Plan N dinners under $X" without an LLM composing anything:

  w1  ONE single-pass retrieval prices every ingredient of EVERY library
      recipe (the VALUES/product_terms shape scales linearly, so the whole
      library costs one round trip),
  w2  head-noun fallback re-match for pools the strict token-AND left
      empty; recipes still missing an ingredient are dropped (with a note
      — e.g. "no dairy" correctly knocks out grilled cheese),
  w3  greedy menu selection by MARGINAL basket cost: an ingredient whose
      cheapest product is already in the basket costs $0 more, so
      ingredient overlap is rewarded exactly, not heuristically. The
      budget gate fires on the menu's cheapest-possible floor BEFORE any
      LLM spend (PlanAborted -> 409, same contract as /plan/nl),
  then the existing selector (one Haiku call per dinner) makes the final
  per-day mapping, the shopping list is merged (shared products counted
  once), and the 4A trip optimizer prices the merged basket's store split.
"""
from __future__ import annotations

import time

from sqlalchemy import text
from sqlalchemy.orm import Session

from . import db, tripopt
from .config import settings
from .models import (DayPlan, PlanLineItem, Recipe, WeekItem, WeekPlan)
from .nlsearch.plan import GateCode, PlanAlert, PlanExecution, StepKind, StepResult
from .nlsearch.planner import PlanAborted
from .nlsearch.query_parser import ALLOWED_TAGS
from .nlsearch.schemas import Constraints, IngredientSpec
from .nlsearch.sql_builder import (build_options_sql, build_price_matrix_sql,
                                   inline_for_display)
from .nlsearch.units import tokens
from .selector import call_selector


def _timed(session: Session, sql: str, params: dict) -> tuple[list, int]:
    t0 = time.perf_counter()
    rows = list(session.execute(text(sql), params).mappings())
    return rows, int((time.perf_counter() - t0) * 1000)


def _batched_pools(session: Session, specs: list[IngredientSpec], c: Constraints,
                   lat: float, lon: float, max_km: float | None,
                   execution: PlanExecution, step_id: str, label_prefix: str) -> dict[int, list]:
    from .nlsearch.planner import _row_to_product

    sql, params = build_options_sql(c, specs, relaxed=set(), lat=lat, lon=lon,
                                    max_km=max_km)
    rows, ms = _timed(session, sql, params)
    pools: dict[int, list] = {}
    for row in rows:
        pools.setdefault(row["ing_no"], []).append(_row_to_product(row))
    execution.steps.append(StepResult(
        step_id=step_id, kind=StepKind.options,
        sql_display=inline_for_display(sql, params),
        row_count=len(rows), duration_ms=ms,
        label=f"{label_prefix}: {len(rows)} offers across {len(pools)} pools "
              f"({len(specs)} ingredients, one pass)"))
    return pools


def plan_week(*, days: int = 5, max_total_budget: float | None = None,
              exclude_tags: list[str] | None = None,
              lat: float | None = None, lon: float | None = None,
              max_distance_km: float | None = None) -> WeekPlan:
    cfg = settings()
    lat = lat if lat is not None else cfg.default_lat
    lon = lon if lon is not None else cfg.default_lon
    notes: list[str] = []
    execution = PlanExecution()

    tags = [t for t in (exclude_tags or []) if t in ALLOWED_TAGS]
    dropped_tags = sorted(set(exclude_tags or []) - set(tags))
    if dropped_tags:
        notes.append(f"ignored unknown dietary tags: {', '.join(dropped_tags)}")
    c = Constraints(exclude_tags=tags)

    recipes = db.load_all_recipes()
    if days > len(recipes):
        notes.append(f"only {len(recipes)} recipes in the library — planning "
                     f"{len(recipes)} dinners instead of {days}")
        days = len(recipes)

    # ── w1: one single-pass retrieval for the WHOLE library ──
    flat: list[tuple[int, int]] = []          # global ing_no -> (recipe_idx, line_idx)
    specs: list[IngredientSpec] = []
    for ri, r in enumerate(recipes):
        for li, ing in enumerate(r.ingredients):
            flat.append((ri, li))
            specs.append(IngredientSpec(name=ing.name))

    with Session(db.engine()) as s:
        pools = _batched_pools(s, specs, c, lat, lon, max_distance_km,
                               execution, "w1_options", "library retrieval")

        # ── w2: head-noun fallback for strict-AND misses ──
        empty = [g for g in range(len(specs)) if not pools.get(g)]
        if empty:
            fb_specs = []
            fb_map = []
            for g in empty:
                toks = tokens(specs[g].name)
                if toks:
                    fb_specs.append(IngredientSpec(name=toks[-1]))
                    fb_map.append(g)
            if fb_specs:
                fb_pools = _batched_pools(s, fb_specs, c, lat, lon,
                                          max_distance_km, execution,
                                          "w2_fallback", "head-noun fallback")
                for i, g in enumerate(fb_map):
                    if fb_pools.get(i):
                        pools[g] = fb_pools[i]

        # regroup per recipe; drop recipes that still miss an ingredient
        by_recipe: dict[int, dict[int, list]] = {}
        for g, (ri, li) in enumerate(flat):
            by_recipe.setdefault(ri, {})[li] = pools.get(g, [])
        candidates: list[tuple[Recipe, dict[int, list]]] = []
        for ri, r in enumerate(recipes):
            missing = [r.ingredients[li].name
                       for li, pool in by_recipe[ri].items() if not pool]
            if missing:
                notes.append(f"skipped {r.name}: no match for "
                             f"{', '.join(missing)} under current constraints")
            else:
                candidates.append((r, by_recipe[ri]))
        if len(candidates) < days:
            days = len(candidates)
            notes.append(f"only {days} recipes remain feasible")
        if days == 0:
            execution.aborted = PlanAlert(
                stage="w2_fallback", code=GateCode.unavailable_within_constraints,
                message="No library recipe is feasible under the current "
                        "constraints. Relax a dietary filter or the distance.",
                details=[])
            raise PlanAborted(execution)

        # ── w3: greedy menu pick by marginal basket cost ──
        t0 = time.perf_counter()
        cheapest: list[dict[int, tuple[int, float]]] = []  # per candidate: line -> (pid, price)
        for _r, rpools in candidates:
            cheapest.append({li: (pool[0].id, pool[0].store_price or pool[0].price)
                             for li, pool in rpools.items()})
        picked: list[int] = []
        union: dict[int, float] = {}                       # product_id -> price
        remaining = list(range(len(candidates)))
        for _day in range(days):
            def marginal(ci: int) -> tuple[float, int]:
                new_cost, shared = 0.0, 0
                seen_new: set[int] = set()
                for pid, price in cheapest[ci].values():
                    if pid in union or pid in seen_new:
                        shared += 1
                    else:
                        seen_new.add(pid)
                        new_cost += price
                return new_cost, shared

            best = min(remaining,
                       key=lambda ci: (marginal(ci)[0], -marginal(ci)[1],
                                       candidates[ci][0].slug))
            picked.append(best)
            remaining.remove(best)
            for pid, price in cheapest[best].values():
                union.setdefault(pid, price)
        floor = round(sum(union.values()), 2)
        menu_names = [candidates[ci][0].name for ci in picked]
        execution.steps.append(StepResult(
            step_id="w3_menu", kind=StepKind.lookup,
            duration_ms=int((time.perf_counter() - t0) * 1000),
            row_count=len(picked),
            label=f"menu: {', '.join(menu_names)} · cheapest-basket floor ${floor:.2f}"))

        # budget gate on the floor — BEFORE any LLM spend
        if max_total_budget is not None and floor > max_total_budget:
            execution.steps[-1].outcome = "aborted"
            execution.aborted = PlanAlert(
                stage="w3_menu", code=GateCode.budget_infeasible,
                message=(f"Cheapest possible {days}-dinner basket is ${floor:.2f} "
                         f"— over the ${max_total_budget:.2f} budget."),
                details=[{"name": candidates[ci][0].name,
                          "reason": f"adds ${marginal_cost:.2f} at cheapest",
                          "suggestions": []}
                         for ci, marginal_cost in
                         [(ci, sum(p for _pid, p in cheapest[ci].values()))
                          for ci in picked]])
            raise PlanAborted(execution)

        # ── selector per dinner (the existing LLM boundary) ──
        day_plans: list[DayPlan] = []
        llm_cost = 0.0
        for ci in picked:
            recipe, rpools = candidates[ci]
            seen: set[int] = set()
            products = [p for pool in rpools.values() for p in pool
                        if p.id not in seen and not seen.add(p.id)]
            result = call_selector(recipe.ingredients, products,
                                   model=cfg.selector_model_default)
            llm_cost += result.cost_usd
            by_id = {p.id: p for p in products}
            by_line = {i.line_no: i for i in recipe.ingredients}
            items, day_cost = [], 0.0
            for sel in result.selections:
                prod, ing = by_id.get(sel.product_id), by_line.get(sel.line_no)
                if prod is None or ing is None:
                    continue
                charged = prod.store_price if prod.store_price is not None else prod.price
                items.append(PlanLineItem(
                    line_no=sel.line_no, ingredient_name=ing.name,
                    product_id=prod.id, product_name=prod.name,
                    product_description=prod.description, price=charged,
                    confidence=sel.confidence, reasoning=sel.reasoning,
                    model_used=result.model_used,
                    store_name=prod.store_name, store_price=prod.store_price))
                day_cost += charged
            day_plans.append(DayPlan(recipe_slug=recipe.slug,
                                     recipe_name=recipe.name,
                                     line_items=items,
                                     day_cost=round(day_cost, 2)))

        # ── merged shopping list (shared products counted once) ──
        merged: dict[int, WeekItem] = {}
        for dp in day_plans:
            for li in dp.line_items:
                if li.product_id in merged:
                    if dp.recipe_name not in merged[li.product_id].used_by:
                        merged[li.product_id].used_by.append(dp.recipe_name)
                else:
                    merged[li.product_id] = WeekItem(
                        product_id=li.product_id, product_name=li.product_name,
                        store_name=li.store_name, price=li.price,
                        used_by=[dp.recipe_name])
        shopping = sorted(merged.values(), key=lambda w: -len(w.used_by))
        total = round(sum(w.price for w in shopping), 2)
        standalone = round(sum(dp.day_cost for dp in day_plans), 2)
        if max_total_budget is not None and total > max_total_budget:
            notes.append(f"selector's semantic picks land at ${total:.2f}, over "
                         f"budget — floor was feasible; consider a higher budget")

        # ── t5: 4A trip optimizer on the merged basket ──
        basket = [(w.product_id, w.product_name) for w in shopping]
        sql, params = build_price_matrix_sql(
            sorted({pid for pid, _ in basket}), lat, lon, max_distance_km)
        rows, ms = _timed(s, sql, params)
        trip_options = tripopt.optimize_trips(
            rows, basket, home_lat=lat, home_lon=lon,
            cost_per_km=cfg.travel_cost_per_km)
        best = next((o for o in trip_options if o.recommended), None)
        execution.steps.append(StepResult(
            step_id="t5_trip_optimizer", kind=StepKind.lookup,
            sql_display=inline_for_display(sql, params),
            row_count=len(rows), duration_ms=ms,
            label=(f"{len(trip_options)} trip options · best: "
                   f"{len(best.stores)} stop(s), ${best.total_cost:.2f} total")
                  if best else "no trip options"))

    return WeekPlan(days=day_plans, shopping_list=shopping, total_cost=total,
                    standalone_cost=standalone,
                    overlap_savings=round(standalone - total, 2),
                    budget=max_total_budget, notes=notes,
                    plan_trace=execution.steps, trip_options=trip_options,
                    total_llm_cost_usd=round(llm_cost, 6))

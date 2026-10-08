"""
Burr state machine — the whole pipeline as one action graph.

    load_recipe (or parse_and_retrieve on the NL path)
      → load_products
      → preselect_model         (router.preselect_model)
      → select_products         (main LLM call using preselected model)
      → check_escalation        (router.should_escalate)
      → escalate_if_needed      (conditional; re-runs subset if cascade said so)
      → optimize_trips          (4A: deterministic split-trip optimizer, NL path)
      → build_plan

Both routers use the same graph. The only difference is which nodes
"do work":
  * cascade: preselect is a no-op (returns Haiku); check_escalation may
    fire; escalate_if_needed makes the second call.
  * three_phase: preselect calls the classifier (real LLM cost);
    check_escalation always returns False; escalate_if_needed is skipped.
"""
from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from burr.core import Application, ApplicationBuilder, State, action, expr

from . import db
from .config import get_router, settings
from .models import (
    BasisLine,
    DroppedIngredient,
    EscalationDecision,
    PlanBasis,
    PlanLineItem,
    PreselectResult,
    Product,
    Recipe,
    RecipeIngredient,
    Selection,
    SelectorResult,
    ShoppingPlan,
)
from .packs import pack_count
from .selector import call_selector, merge_selections
from .tracing import StepTimer, llm_call_trace, llm_span, llm_step, make_tracker

# ─── Purchases: selections -> what is actually bought ────────
# Two recipe lines can resolve to one product (mala chicken's "ground
# Sichuan peppercorns" and "Sichuan peppercorns" both became the whole 50 g
# bag). Pricing each selection separately charged that bag twice in
# total_cost and listed it twice on the trip. A purchase is one product,
# bought once for every line that chose it — more than one pack only when
# the lines' summed need, known in the pack's own unit, exceeds one pack.

_MATCH_ORDER = {"exact": 0, "form": 1, "generic": 2}
NO_SELECTION = "the selector returned no product for this line"


@dataclass
class Purchase:
    product: Product
    lines: list[tuple[Selection, RecipeIngredient]] = field(default_factory=list)
    packs: int = 1

    @property
    def unit_price(self) -> float:
        p = self.product
        return p.store_price if p.store_price is not None else p.price


def _packs(product: Product, needs: list[tuple[float, str] | None]) -> int:
    """Packs a shared purchase takes: the summed need over the pack size,
    rounded up — or 1 when any line's need is unknown or in another unit
    (a teaspoon of peppercorns against a 50 g bag says nothing about bags),
    and 1 for a single line, as chat plans have always bought."""
    return pack_count(product.unit_qty, product.unit_uom, needs, min_lines=2) or 1


def group_purchases(selections: list[Selection], products_by_id: dict[int, Product],
                    ingredients_by_line: dict[int, RecipeIngredient],
                    needs: dict[int, tuple[float, str] | None] | None = None,
                    ) -> tuple[list[Purchase], list[tuple[RecipeIngredient, str]]]:
    """(purchases in recipe order, lines with no valid selection + why).

    The first valid selection per line counts; a repeat for the same line is
    ignored, and a product_id that is not a candidate is no selection."""
    needs = needs or {}
    purchases: dict[int, Purchase] = {}
    chosen: set[int] = set()
    invalid: dict[int, str] = {}
    for sel in selections:
        ing = ingredients_by_line.get(sel.line_no)
        if ing is None or sel.line_no in chosen:
            continue
        prod = products_by_id.get(sel.product_id)
        if prod is None:
            invalid.setdefault(sel.line_no, f"the selector named product {sel.product_id}, "
                                            "which is not a candidate for this line")
            continue
        chosen.add(sel.line_no)
        purchases.setdefault(prod.id, Purchase(product=prod)).lines.append((sel, ing))
    for pu in purchases.values():
        pu.lines.sort(key=lambda t: t[0].line_no)
        pu.packs = _packs(pu.product, [needs.get(sel.line_no) for sel, _ in pu.lines])
    ordered = sorted(purchases.values(), key=lambda pu: pu.lines[0][0].line_no)
    unselected = [(ing, invalid.get(line, NO_SELECTION))
                  for line, ing in sorted(ingredients_by_line.items()) if line not in chosen]
    return ordered, unselected


def _needs(parsed) -> dict[int, tuple[float, str] | None]:
    """recipe line_no -> canonical need (qty, g|ml|each) from the NL parse."""
    from .nlsearch.units import normalize_quantity

    if parsed is None:
        return {}
    return {i + 1: normalize_quantity(ing.quantity, ing.unit)
            for i, ing in enumerate(parsed.recipe.ingredients)}


def _selector_constraints(parsed, brand_stats: dict | None = None,
                          preference: list | None = None,
                          planned: set[str] | None = None) -> dict | None:
    """Condense the NL parse into the binding-constraints object the
    selector prompt understands. On the classic path only the origin
    preference (if any) is carried. `planned` limits the per-ingredient
    hints to the ingredients still in the plan (a partial plan dropped the
    rest; the selector is never told about them)."""
    if parsed is None:
        return {"origin_preference": list(preference)} if preference else None
    c = parsed.constraints
    ings = [i for i in parsed.recipe.ingredients if planned is None or i.name in planned]
    quantities = {
        ing.name: f"{ing.quantity:g} {ing.unit}"
        for ing in ings
        if ing.quantity is not None and ing.unit
    }
    out = {
        "max_total_budget": c.max_total_budget,
        "preferences": c.soft_text or None,
        "quantities_needed": quantities or None,
        "preps": {i.name: i.prep for i in ings if i.prep} or None,
        # t3's per-ingredient brand aggregates (options/avg price/avg rating/
        # review count) — context for the final mapping, not a hard rule
        "brand_statistics": brand_stats or None,
        # ordered, most-preferred first; soft guidance for rule 7
        "origin_preference": list(preference) if preference else None,
    }
    out = {k: v for k, v in out.items() if v is not None}
    return out or None


# ─── Actions ─────────────────────────────────────────────────

@action(reads=[], writes=["recipe", "location", "plan_path"])
def load_recipe(state: State, recipe_slug: str,
                location: dict | None = None) -> tuple[dict, State]:
    """`location` ({lat, lon, max_km}) makes the plan store-aware: each chosen product is
    priced at its cheapest store in range and the trip optimizer splits the basket."""
    recipe = db.load_recipe(recipe_slug)
    result = {"ingredient_count": len(recipe.ingredients)}
    return result, state.update(recipe=recipe, location=location, plan_path="library")


@action(reads=["recipe"], writes=["recipe", "products", "origins", "origin_dropped", "preference",
                                 "origin_requested", "out_of_range", "ingredient_count",
                                 "exclude"])
def load_products(state: State, exclude: list | None = None,
                  preference: list | None = None,
                  allow_partial: bool = False) -> tuple[dict, State]:
    """With `allow_partial`, an ingredient the origin exclusion emptied leaves the recipe and
    goes to out_of_range with what to buy instead; the rest is planned."""
    products = db.load_all_products()
    recipe = state.get("recipe")
    cfg = settings()
    pools = _ingredient_pools(recipe, cfg.default_lat, cfg.default_lon) if recipe else {}
    names = {i: ing.name for i, ing in enumerate(recipe.ingredients)} if recipe else {}
    kept, dropped, origins, _pools, excluded = apply_origin_constraint(
        products, pools, names, exclude=exclude, preference=preference,
        allow_partial=allow_partial)
    count = len(recipe.ingredients) if recipe else 0
    if excluded:
        recipe = recipe.model_copy(update={"ingredients": [
            ing for i, ing in enumerate(recipe.ingredients) if i not in excluded]})
    result = {"product_count": len(kept), "origin_dropped": len(dropped),
              "excluded_by_origin": len(excluded)}
    return result, state.update(recipe=recipe, products=kept, origins=origins,
                                origin_dropped=dropped,
                                preference=list(preference or []),
                                exclude=list(exclude or []),
                                origin_requested=bool(exclude or preference),
                                out_of_range=list(excluded.values()),
                                ingredient_count=count)


@action(reads=[], writes=["recipe", "products", "parsed_input", "location",
                          "plan_trace", "brand_stats", "retrieval_stats",
                          "origins", "origin_dropped", "preference",
                          "origin_requested", "not_stocked", "out_of_range",
                          "skipped", "pool_hints", "match_levels",
                          "ingredient_count", "exclude", "plan_path"])
def parse_and_retrieve(state: State, recipe_text: str,
                       lat: float | None = None,
                       lon: float | None = None,
                       exclude: list | None = None,
                       preference: list | None = None,
                       max_km: float | None = None,
                       allow_partial: bool = False,
                       parsed=None) -> tuple[dict, State]:
    """NL2SQL entrypoint: pasted recipe text → query-plan execution
    (t1 existence → t2 options → t3 brand stats → t4 lookups) producing an
    ad-hoc Recipe + store-priced candidate pools. Replaces load_recipe +
    load_products on the NL path; downstream actions consume the same state
    keys. Gate aborts raise nlsearch.PlanAborted → API 409. `max_km`
    overrides the parsed distance; with `allow_partial` the recipe in state
    holds only the ingredients still planned, and the drops ride along.

    `parsed` (a ParsedInput) is a recipe the shopper already reviewed
    (run_spec): it is planned as given, with no parse call of either kind,
    and `recipe_text` is only what the trace and the plan display."""
    from . import nlsearch

    # With an exclusion active, retrieve wider than the usual cheapest-8 so
    # the 9th-cheapest non-excluded product is still there after filtering;
    # the pool is trimmed back to the normal size below.
    from .nlsearch.sql_builder import PER_INGREDIENT_LIMIT
    limit = 10_000 if exclude else PER_INGREDIENT_LIMIT
    r = nlsearch.run_query_plan(recipe_text, parsed=parsed, lat=lat, lon=lon,
                                per_ingredient_limit=limit, max_km=max_km,
                                allow_partial=allow_partial)
    names = {i: ing.name for i, ing in enumerate(r.recipe.ingredients)}
    kept, dropped, origins, kept_pools, excluded = apply_origin_constraint(
        r.products, r.pools, names, exclude=exclude, preference=preference,
        allow_partial=allow_partial)
    # A partial plan leaves out what the exclusion emptied, as it does what is not stocked.
    recipe = r.recipe if not excluded else r.recipe.model_copy(update={"ingredients": [
        ing for i, ing in enumerate(r.recipe.ingredients) if i not in excluded]})
    if exclude and kept_pools:
        def price(p):
            return p.store_price if p.store_price is not None else p.price
        seen: set[int] = set()
        kept = []
        for key, (direct, alts) in kept_pools.items():
            if key in excluded:
                continue
            for p in sorted(direct, key=price)[:PER_INGREDIENT_LIMIT] + alts:
                if p.id not in seen:
                    seen.add(p.id)
                    kept.append(p)
    result = {
        "ingredient_count": len(r.recipe.ingredients),
        "product_count": len(kept),
        "origin_dropped": len(dropped),
        "excluded_by_origin": len(excluded),
        "plan_steps": [s.step_id for s in r.execution.steps],
        "parse_cost_usd": r.parsed.cost_usd,
    }
    # Each line's best candidates, named on `skipped` if the selector then
    # returns no product for the line.
    kept_ids = {p.id for p in kept}
    pool_hints = {
        ing.line_no: [f"{p.name} (${p.store_price if p.store_price is not None else p.price:.2f})"
                      for p in r.pools.get(i, []) if p.id in kept_ids and not p.substitute][:3]
        for i, ing in enumerate(r.recipe.ingredients) if i not in excluded}
    return result, state.update(
        recipe=recipe, products=kept, origins=origins,
        origin_dropped=dropped, preference=list(preference or []),
        exclude=list(exclude or []), plan_path="nl" if parsed is None else "spec",
        origin_requested=bool(exclude or preference),
        parsed_input=r.parsed,
        location={"lat": r.lat, "lon": r.lon, "max_km": r.max_km},
        plan_trace=[*(_parse_step(r.parsed) if parsed is None else [_reviewed_step(r.parsed)]),
                    *r.execution.steps],
        brand_stats=r.brand_stats,
        retrieval_stats=r.stats, not_stocked=r.not_stocked,
        out_of_range=[*r.out_of_range, *excluded.values()], skipped=r.skipped,
        pool_hints=pool_hints,
        match_levels=r.match_levels, ingredient_count=r.ingredient_count)


@action(reads=["recipe", "products"], writes=["preselect_result"])
def preselect_model(state: State) -> tuple[dict, State]:
    router = get_router()
    recipe: Recipe = state["recipe"]
    products: list[Product] = state["products"]

    preselect: PreselectResult = router.preselect_model(
        recipe, products, retrieval_stats=state.get("retrieval_stats"))

    result = {
        "router": router.name,
        "chosen_model": preselect.model,
        "complexity_score": preselect.complexity_score,
        "routing_cost_usd": preselect.routing_cost_usd,
        "reason": preselect.reason,
    }
    return result, state.update(preselect_result=preselect)


@action(reads=["recipe", "products", "preselect_result", "origins", "preference",
               "plan_trace"],
        writes=["initial_result", "plan_trace"])
def select_products(state: State) -> tuple[dict, State]:
    recipe: Recipe = state["recipe"]
    products: list[Product] = state["products"]
    preselect: PreselectResult = state["preselect_result"]

    parsed = state.get("parsed_input")   # NL path only
    result: SelectorResult = call_selector(
        recipe.ingredients,
        products,
        model=preselect.model,
        enable_thinking=False,
        constraints=_selector_constraints(parsed, state.get("brand_stats"),
                                          preference=state.get("preference"),
                                          planned={i.name for i in recipe.ingredients}),
        origins_by_id=state.get("origins") or {},
        preference=state.get("preference") or [],
    )

    span = llm_span(
        step="select_products",
        model=result.model_used,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cost_usd=result.cost_usd,
        latency_ms=result.latency_ms,
        http=result.http[-1] if result.http else None,
    )
    from . import metrics as _m
    _m.record_llm(span)
    return {"llm_call": span, "n_selections": len(result.selections)}, \
        state.update(initial_result=result,
                     plan_trace=_with_llm_step(state, span))


@action(reads=["initial_result"], writes=["escalation_decision"])
def check_escalation(state: State) -> tuple[dict, State]:
    router = get_router()
    decision: EscalationDecision = router.should_escalate(state["initial_result"])
    return {
        "escalate": decision.escalate,
        "n_ingredients_to_rerun": len(decision.ingredients_to_rerun),
        "reason": decision.reason,
    }, state.update(escalation_decision=decision)


@action(
    reads=["recipe", "products", "initial_result", "escalation_decision", "plan_trace"],
    writes=["final_result", "plan_trace"],
)
def escalate_if_needed(state: State) -> tuple[dict, State]:
    """Only runs when escalation_decision.escalate == True (routed via graph)."""
    recipe: Recipe = state["recipe"]
    products: list[Product] = state["products"]
    initial: SelectorResult = state["initial_result"]
    decision: EscalationDecision = state["escalation_decision"]

    # Re-run just the flagged ingredients through the escalation model.
    flagged_ingredients = [
        i for i in recipe.ingredients if i.line_no in decision.ingredients_to_rerun
    ]

    escalated = call_selector(
        flagged_ingredients,
        products,
        model=decision.escalation_model,
        enable_thinking=settings().enable_thinking_on_escalation,
    )

    merged = merge_selections(initial, escalated, decision.ingredients_to_rerun)

    span = llm_span(
        step="escalate",
        model=escalated.model_used,
        input_tokens=escalated.input_tokens,
        output_tokens=escalated.output_tokens,
        cost_usd=escalated.cost_usd,
        latency_ms=escalated.latency_ms,
        http=escalated.http[-1] if escalated.http else None,
    )
    return {"llm_call": span, "n_reran": len(flagged_ingredients)}, \
        state.update(final_result=merged,
                     plan_trace=_with_llm_step(state, span))


@action(reads=["initial_result", "final_result"], writes=["final_result"])
def skip_escalation(state: State) -> tuple[dict, State]:
    """No-op — hoists initial_result into final_result when we don't escalate."""
    return {"escalated": False}, state.update(final_result=state["initial_result"])


@action(reads=["final_result", "products", "recipe", "parsed_input", "location",
               "plan_trace"],
        writes=["trip_options", "plan_trace", "store_offers", "nearest_offers"])
def optimize_trips(state: State) -> tuple[dict, State]:
    """4A: deterministic split-trip optimizer over the chosen basket.
    Prices every PURCHASE (group_purchases: a product shared by two lines
    once, times its packs) at every in-range store (one templated query),
    then enumerates store subsets with exact travel loops — no LLM. Runs
    whenever the plan has a location: always on the NL path, and on the
    classic path when the caller gave one. Also records each product's
    cheapest in-range offer (the classic path prices its lines with it) and,
    for a product no store in range sells, the nearest offer anywhere."""
    import time as _time

    from sqlalchemy import text as _text
    from sqlalchemy.orm import Session as _Session

    from . import db, tripopt
    from .nlsearch.plan import StepKind, StepResult
    from .nlsearch.sql_builder import build_price_matrix_sql, inline_for_display

    loc = state.get("location")
    final: SelectorResult = state["final_result"]
    products_by_id = {p.id: p for p in state["products"]}
    recipe: Recipe = state["recipe"]
    purchases, _ = group_purchases(final.selections, products_by_id,
                                   {i.line_no: i for i in recipe.ingredients},
                                   _needs(state.get("parsed_input")))
    basket = [(pu.product.id, pu.product.name) for pu in purchases]
    if loc is None or not basket:
        # Burr requires every declared write; pass the trace through untouched.
        return {"skipped": True}, state.update(
            trip_options=[], plan_trace=state.get("plan_trace") or [],
            store_offers={}, nearest_offers={})

    sql, params = build_price_matrix_sql(
        sorted({pid for pid, _ in basket}), loc["lat"], loc["lon"], loc["max_km"])
    t0 = _time.perf_counter()
    with _Session(db.engine()) as s:
        rows = list(s.execute(_text(sql), params).mappings())
    duration_ms = int((_time.perf_counter() - t0) * 1000)
    offers = _best_offers(rows, key=lambda r: (r["price"], r["dist_km2"]))
    # A product no store in range sells (possible only on the classic path, whose selector
    # does not see distance) is left out of the trip, and its nearest offer anywhere named.
    missing = sorted({pid for pid, _ in basket} - offers.keys())
    nearest: dict[int, dict] = {}
    if missing:
        sql2, params2 = build_price_matrix_sql(missing, loc["lat"], loc["lon"], None)
        with _Session(db.engine()) as s:
            nearest = _best_offers(list(s.execute(_text(sql2), params2).mappings()),
                                   key=lambda r: (r["dist_km2"], r["price"]))
    basket = [(pid, name) for pid, name in basket if pid in offers]

    options = tripopt.optimize_trips(
        rows, basket, home_lat=loc["lat"], home_lon=loc["lon"],
        cost_per_km=settings().travel_cost_per_km,
        packs={pu.product.id: pu.packs for pu in purchases})
    best = next((o for o in options if o.recommended), None)
    label = (f"{len(options)} trip options · best: {len(best.stores)} stop(s), "
             f"${best.total_cost:.2f} total"
             + (f" (saves ${best.savings_vs_one_stop:.2f} vs one stop)"
                if best.savings_vs_one_stop > 0 else "")) if best else "no options"
    step = StepResult(step_id="t5_trip_optimizer", kind=StepKind.lookup,
                      sql_display=inline_for_display(sql, params),
                      row_count=len(rows), duration_ms=duration_ms, label=label)
    return {"n_options": len(options)}, state.update(
        trip_options=options,
        plan_trace=[*(state.get("plan_trace") or []), step],
        store_offers=offers, nearest_offers=nearest)


def _with_llm_step(state: State, span: dict) -> list:
    """The plan trace with this LLM call appended (a demo-mode call, instant and untraced,
    adds nothing)."""
    trace = list(state.get("plan_trace") or [])
    return trace + [llm_step(span)] if span.get("http") or span["latency_ms"] else trace


def _reviewed_step(parsed):
    """The trace's parse_input step for reviewed lines: no parse ran, and the trace says so
    rather than leaving the reader to wonder where the interpretation came from."""
    from .nlsearch.plan import StepKind, StepResult

    n = len(parsed.recipe.ingredients) + len(parsed.over_cap)
    return StepResult(step_id="parse_input", kind=StepKind.llm, outcome="skipped",
                      label=f"skipped: reviewed lines ({n} planned as given, no parse)")


def _parse_step(parsed) -> list:
    """The recipe parse's LLM call as the plan trace's first step (none in demo mode)."""
    if parsed is None or not parsed.http:
        return []
    return [llm_step(llm_span(step="parse_input", model=parsed.http.get("model", ""),
                              input_tokens=0, output_tokens=0, cost_usd=parsed.cost_usd,
                              latency_ms=parsed.latency_ms, http=parsed.http))]


def _llm_calls(parsed, initial: SelectorResult | None, final: SelectorResult) -> list:
    """Every traced LLM call behind the plan, phase by phase: the parse (NL path), the
    selection and, after an escalation, the re-run (merge_selections appends its trace after
    the selection's; without one, final_result IS initial_result)."""
    first = initial.http if initial is not None else []
    return ([llm_call_trace("parse_input", parsed.http)] if parsed and parsed.http else []) \
        + [llm_call_trace("select_products", h) for h in first] \
        + [llm_call_trace("escalate", h) for h in final.http[len(first):]]


def _best_offers(rows, *, key) -> dict[int, dict]:
    """product_id -> the offer that sorts first by `key`: {store, price, dist_km}."""
    best: dict[int, dict] = {}
    for r in sorted(rows, key=key):
        best.setdefault(r["product_id"], {
            "store": r["store_name"], "price": r["price"],
            "dist_km": round(math.sqrt(max(r["dist_km2"], 0.0)), 1)})
    return best


@action(
    reads=["recipe", "products", "initial_result", "final_result", "preselect_result",
           "escalation_decision", "trip_options", "parsed_input", "origins",
           "origin_requested", "origin_dropped", "match_levels", "not_stocked",
           "out_of_range", "skipped", "pool_hints", "ingredient_count",
           "plan_trace", "location", "store_offers", "nearest_offers", "exclude",
           "preference", "plan_path"],
    writes=["plan"],
)
def build_plan(state: State) -> tuple[dict, State]:
    recipe: Recipe = state["recipe"]
    products_by_id = {p.id: p for p in state["products"]}
    final: SelectorResult = state["final_result"]
    preselect: PreselectResult = state["preselect_result"]
    decision: EscalationDecision = state["escalation_decision"]
    ingredients_by_line = {i.line_no: i for i in recipe.ingredients}

    from . import origins as origins_mod
    origins_map = state.get("origins") or {}
    # NL path: how t1 matched each line's ingredient. The classic path uses
    # its seeded ingredient names verbatim, which is what "exact" means.
    match_levels = state.get("match_levels") or {}

    parsed = state.get("parsed_input")   # NL path only
    location = state.get("location")
    offers = state.get("store_offers") or {}
    if parsed is None and location is not None:
        # Classic path with a shopping location: each chosen product priced at its cheapest
        # store in range, as plan_from_text's lines are.
        products_by_id = {
            pid: p.model_copy(update={"store_name": offers[pid]["store"],
                                      "store_price": offers[pid]["price"]})
            if pid in offers else p
            for pid, p in products_by_id.items()}
    needs = _needs(parsed)
    purchases, unselected = group_purchases(final.selections, products_by_id,
                                            ingredients_by_line, needs)
    unreachable: list[DroppedIngredient] = []
    if parsed is None and location is not None:
        # Reported, never silently dropped: a chosen product no store in range sells leaves
        # the priced basket (and the trip) for out_of_range, naming the nearest offer.
        nearest = state.get("nearest_offers") or {}
        within = (f"within {location['max_km']:g} km of the shopping location"
                  if location.get("max_km") is not None else "at any store")
        for pu in [pu for pu in purchases if pu.product.id not in offers]:
            near = nearest.get(pu.product.id)
            unreachable.append(DroppedIngredient(
                ingredient=" + ".join(ing.name for _, ing in pu.lines),
                reason=(f"{pu.product.name} has no offer {within}"
                        + (f"; the nearest is {near['store']}, {near['dist_km']:.1f} km away, "
                           f"at ${near['price']:.2f}" if near else "; no store sells it"))))
        purchases = [pu for pu in purchases if pu.product.id in offers]
        if unreachable and not purchases:
            # Nothing left to price: an error, as plan_from_text's range gate is, naming the
            # nearest offer for every product.
            from .nlsearch.plan import GateCode, PlanAlert, PlanExecution
            from .nlsearch.planner import PlanAborted

            raise PlanAborted(PlanExecution(
                steps=state.get("plan_trace") or [],
                aborted=PlanAlert(
                    stage="t5_trip_optimizer", code=GateCode.unavailable_within_constraints,
                    message=(f"No store {within} sells any of the chosen products. Widen "
                             "max_km or move the location."),
                    details=[{"name": d.ingredient, "reason": d.reason, "suggestions": []}
                             for d in unreachable],
                    partial_would_plan=0)))
    line_items: list[PlanLineItem] = []
    for pu in purchases:
        prod = pu.product
        sels = [sel for sel, _ in pu.lines]
        first, others = sels[0], sels[1:]
        reasoning = first.reasoning + "".join(
            f" | line {sel.line_no}: {sel.reasoning}" for sel in others)
        line_items.append(PlanLineItem(
            line_no=first.line_no,
            ingredient_name=" + ".join(ing.name for _, ing in pu.lines),
            product_id=prod.id,
            product_name=prod.name,
            product_description=prod.description,
            price=round(pu.unit_price * pu.packs, 2),
            confidence=min(sel.confidence for sel in sels),
            reasoning=reasoning,
            model_used=final.model_used,
            store_name=prod.store_name,
            store_price=prod.store_price,
            origin=origins_mod.origin_receipt(origins_map.get(prod.id)),
            # the loosest level among the lines, so a generic one is never hidden
            match=max((match_levels.get(sel.line_no, "exact") for sel in sels),
                      key=_MATCH_ORDER.__getitem__),
            also_lines=[sel.line_no for sel in others],
            packs=pu.packs,
            **_need(needs, sels),
        ))
    total_cost = sum(li.price for li in line_items)
    # A line the selector left without a valid product is reported, never
    # silently dropped: every ingredient lands somewhere on the plan.
    pool_hints = state.get("pool_hints") or {}
    skipped = list(state.get("skipped") or []) + [
        DroppedIngredient(ingredient=ing.name, reason=why,
                          suggestions=pool_hints.get(ing.line_no, []))
        for ing, why in unselected]

    # Coverage is an answer to an origin question. Computing it for every
    # plan stamped "UNVERIFIED" on baskets nobody asked about, which made the
    # label noise instead of signal.
    coverage = None
    origin_status = "not_requested"
    if state.get("origin_requested"):
        coverage = origins_mod.basket_coverage(
            [(li.product_id, li.price) for li in line_items],
            origins=origins_map or None,
            excluded_lines=len(state.get("origin_dropped") or []))
        origin_status = "verified" if coverage.meets_floor else "unverified"

    plan = ShoppingPlan(
        recipe_slug=recipe.slug,
        recipe_name=recipe.name,
        line_items=line_items,
        total_cost=round(total_cost, 2),
        routing_strategy=get_router().name,
        preselected_model=preselect.model,
        escalated=decision.escalate,
        origin_coverage=coverage,
        origin_status=origin_status,
        total_llm_cost_usd=round(
            preselect.routing_cost_usd + final.cost_usd
            + (parsed.cost_usd if parsed else 0.0), 6
        ),
        total_latency_ms=final.latency_ms,
        interpretation=parsed.display_lines() if parsed else [],
        plan_trace=state.get("plan_trace") or [],
        candidate_count=len(state["products"]) if parsed else 0,
        llm_calls=_llm_calls(parsed, state.get("initial_result"), final),
        trip_options=state.get("trip_options") or [],
        not_stocked=state.get("not_stocked") or [],
        out_of_range=list(state.get("out_of_range") or []) + unreachable,
        skipped=skipped,
        ingredient_count=state.get("ingredient_count") or len(recipe.ingredients),
        servings=(recipe.servings if parsed is None
                  else parsed.recipe.servings if parsed.recipe.servings_stated else None),
    )
    plan.basis = _basis(state, plan, purchases)
    return {"total_cost": plan.total_cost, "n_line_items": len(plan.line_items)}, \
        state.update(plan=plan)


def _need(needs: dict[int, tuple[float, str] | None], sels: list[Selection]) -> dict:
    """need_qty/need_uom for a purchase: the summed need of its lines when every one is known
    in the same canonical unit, else both None."""
    ns = [needs.get(sel.line_no) for sel in sels]
    if not ns or any(n is None for n in ns) or len({n[1] for n in ns if n}) != 1:
        return {"need_qty": None, "need_uom": None}
    return {"need_qty": round(sum(n[0] for n in ns if n), 4), "need_uom": ns[0][1]}


def _basis(state: State, plan: ShoppingPlan, purchases: list[Purchase]) -> PlanBasis:
    """The plan's PlanBasis: its planned lines as the parse (or the reviewed recipe) gave
    them, the product each one is bought as, and what it was planned under. A line whose
    product left the basket (no valid selection, no offer in range) has product_id None."""
    from .nlsearch.schemas import Constraints

    parsed = state.get("parsed_input")
    specs = parsed.recipe.ingredients if parsed is not None else []
    levels = state.get("match_levels") or {}
    bought = {li.product_id for li in plan.line_items}
    chosen = {sel.line_no: (pu.product.id, sel.confidence)
              for pu in purchases if pu.product.id in bought for sel, _ in pu.lines}
    lines = []
    for ing in state["recipe"].ingredients:
        spec = specs[ing.line_no - 1] if 0 < ing.line_no <= len(specs) else None
        product_id, confidence = chosen.get(ing.line_no, (None, None))
        lines.append(BasisLine(
            line_no=ing.line_no, name=spec.name if spec else ing.name,
            form=spec.form if spec else None, prep=spec.prep if spec else None,
            quantity=spec.quantity if spec else None, unit=spec.unit if spec else None,
            level=levels.get(ing.line_no, "exact"),
            product_id=product_id, confidence=confidence))
    loc = state.get("location") or {}
    return PlanBasis(
        path=state.get("plan_path") or ("library" if parsed is None else "nl"),
        recipe_slug=plan.recipe_slug, recipe_name=plan.recipe_name, lines=lines,
        constraints=parsed.constraints if parsed is not None else Constraints(),
        lat=loc.get("lat"), lon=loc.get("lon"), max_km=loc.get("max_km"),
        exclude_origin=list(state.get("exclude") or []),
        preference=list(state.get("preference") or []),
        origin_requested=bool(state.get("origin_requested")),
        origin_dropped=len(state.get("origin_dropped") or []),
        interpretation=list(plan.interpretation),
        not_stocked=list(plan.not_stocked), out_of_range=list(plan.out_of_range),
        skipped=list(plan.skipped), ingredient_count=plan.ingredient_count)


# ─── Application builder ─────────────────────────────────────

# ─── Origin constraint ───────────────────────────────────────
# Applied AFTER retrieval and BEFORE selection, PER INGREDIENT. The first
# version gated only when the entire candidate pool was empty, so when the
# exclusion removed every garlic the planner quietly mapped "Garlic" to
# whatever was left — a 200 with the wrong product. A plan line must come
# from its own ingredient's candidates or not be made at all.

def _ingredient_pools(recipe, lat, lon):
    """Per-ingredient term-matched candidates, keyed by ingredient index.

    The classic path has no retrieval step of its own (the selector sees the
    whole catalog), so this borrows the week planner's single-pass retrieval
    purely to know WHICH products are candidates for WHICH ingredient. That
    is what lets the gate name the ingredient the exclusion emptied.
    """
    from sqlalchemy import text
    from sqlalchemy.orm import Session

    from .nlsearch.planner import _row_to_product
    from .nlsearch.schemas import Constraints, IngredientSpec
    from .nlsearch.sql_builder import build_options_sql
    from .nlsearch.units import tokens

    specs = [IngredientSpec(name=i.name) for i in recipe.ingredients]
    if not specs:
        return {}

    def fetch(spec_list):
        # Uncapped: the gate asks "is ANY candidate left?", so a cheapest-8
        # shortlist is wrong — the 9th-cheapest non-excluded product is a
        # perfectly good answer and must not trigger a 409.
        sql, params = build_options_sql(Constraints(), spec_list, relaxed=set(),
                                        lat=lat, lon=lon,
                                        per_ingredient_limit=10_000)
        out: dict[int, list] = {}
        with Session(db.engine()) as s:
            for row in s.execute(text(sql), params).mappings():
                out.setdefault(row["ing_no"], []).append(_row_to_product(row))
        return out

    direct = fetch(specs)
    # Head-noun matches are offered as ALTERNATIVES when the gate fires —
    # never counted as candidates. Unioning them into the candidate set let
    # "Basmati Rice" survive on Gluten-Free Penne and ship it with a 200.
    heads = []
    for s in specs:
        toks = tokens(s.name)
        heads.append(IngredientSpec(name=toks[-1]) if len(toks) > 1 else None)
    head_specs = [h for h in heads if h is not None]
    relaxed_by_g: dict[int, list] = {}
    if head_specs:
        fetched = fetch(head_specs)
        gi = [g for g, h in enumerate(heads) if h is not None]
        for i, g in enumerate(gi):
            seen = {p.id for p in direct.get(g, [])}
            relaxed_by_g[g] = [p for p in fetched.get(i, []) if p.id not in seen]
    return {g: {"direct": direct.get(g, []), "relaxed": relaxed_by_g.get(g, [])}
            for g in range(len(specs)) if direct.get(g) or relaxed_by_g.get(g)}


def apply_origin_constraint(products, pools, names, *, exclude, preference,
                            allow_partial: bool = False):
    """Filter `products` on origin and gate per ingredient.

    products : what the selector will be shown (filtered copy returned)
    pools    : ingredient key -> candidate list, used ONLY for gating
    names    : ingredient key -> display name

    Returns (kept_products, dropped, origins, kept_pools, excluded). `dropped`
    is deduplicated by product. An ingredient whose pool had candidates and
    lost all of them is the trade the shopper is being asked to make, stated
    rather than silently made for them, with the removed products and the
    same-aisle alternatives still available as suggestions. By default that
    raises PlanAborted(excluded_by_origin) naming EVERY such ingredient. With
    `allow_partial`, and anything else left to plan, they come back instead in
    `excluded` (ingredient key -> DroppedIngredient) for the plan's
    out_of_range, and the caller plans the rest.

    Products with no evidence are kept: absence is not a verdict. The
    coverage figure on the finished plan is what stops that leniency from
    reading as a clean basket.
    """
    from . import origins as origins_mod
    from .nlsearch.plan import GateCode, PlanAlert, PlanExecution
    from .nlsearch.planner import PlanAborted

    def _members(pool):
        if isinstance(pool, dict):
            return list(pool.get("direct", [])) + list(pool.get("relaxed", []))
        return list(pool)

    pool_ids = {p.id for pool in pools.values() for p in _members(pool)}
    all_ids = sorted(pool_ids | {p.id for p in products})
    # Resolved unconditionally: receipts and coverage are worth carrying
    # even with no filter — a basket you can audit afterwards is the point.
    origins = origins_mod.resolve_all(all_ids)
    if not exclude:
        return list(products), [], origins, {}, {}

    kept_products, dropped = origins_mod.filter_pool(
        products, exclude=exclude, origins=origins)
    dropped_by_id = {p.id: (p, c, f) for p, c, f in dropped}

    def split(pool):
        """(direct candidates, alternatives). Classic pools are dicts; NL pools
        are flat lists where t4 substitutes carry substitute=True."""
        if isinstance(pool, dict):
            return list(pool.get("direct", [])), list(pool.get("relaxed", []))
        return [p for p in pool if not p.substitute], [p for p in pool if p.substitute]

    emptied: list[tuple[Any, str, list, list]] = []
    kept_pools: dict = {}
    for key, pool in pools.items():
        direct, alternatives = split(pool)
        kept_direct, dropped_direct = origins_mod.filter_pool(
            direct, exclude=exclude, origins=origins)
        kept_alt, dropped_alt = origins_mod.filter_pool(
            alternatives, exclude=exclude, origins=origins)
        for p, c, f in dropped_direct + dropped_alt:
            dropped_by_id.setdefault(p.id, (p, c, f))
        kept_pools[key] = (kept_direct, kept_alt)
        if direct and not kept_direct:
            # Every candidate for THIS ingredient is gone. Whatever remains
            # in the pool is a same-aisle alternative, not the ingredient —
            # offered below as a stated trade, never silently substituted.
            emptied.append((key, names.get(key, str(key)), dropped_direct, kept_alt))
    if not emptied:
        return kept_products, list(dropped_by_id.values()), origins, kept_pools, {}

    def price(p):
        return p.store_price if p.store_price is not None else p.price

    def known_origin(p):
        o = origins.get(p.id)
        return f", {o.country}" if o is not None and o.status == "resolved" and o.country else ""

    def reason(removed):
        countries = sorted({country for _, country, _ in removed})
        return (f"all {len(removed)} candidate(s) are evidenced as coming from "
                f"{', '.join(countries)}, which this plan excludes")

    def suggestions(removed, alts):
        return [
            f"{p.name} (${price(p):.2f}) — {country} via {field}"
            for p, country, field in removed[:5]
        ] + [
            f"still available, not a direct match: {p.name} (${price(p):.2f}{known_origin(p)})"
            for p in sorted(alts, key=price)[:3]
        ]

    plannable = len(names or pools) - len(emptied)
    if allow_partial and plannable > 0:
        excluded = {key: DroppedIngredient(ingredient=name, reason=reason(removed),
                                           suggestions=suggestions(removed, alts))
                    for key, name, removed, alts in emptied}
        return (kept_products, list(dropped_by_id.values()), origins, kept_pools,
                excluded)
    affected = ", ".join(name for _, name, _, _ in emptied)
    raise PlanAborted(PlanExecution(steps=[], aborted=PlanAlert(
        stage="origin_filter", code=GateCode.excluded_by_origin,
        message=(f"Excluding {', '.join(exclude)} left no candidate for: "
                 f"{affected}. Relax the exclusion, or accept one of the "
                 f"removed products listed per ingredient."),
        details=[{"name": name, "reason": reason(removed),
                  "suggestions": suggestions(removed, alts)}
                 for _, name, removed, alts in emptied],
        partial_would_plan=max(plannable, 0))))


def build_application(recipe_slug: str | None = None,
                      recipe_text: str | None = None,
                      lat: float | None = None,
                      lon: float | None = None,
                      exclude: list | None = None,
                      preference: list | None = None,
                      max_km: float | None = None,
                      allow_partial: bool = False,
                      hooks: list | None = None) -> Application:
    """Construct the Burr Application for one run (`hooks`: Burr lifecycle hooks, e.g. the
    StepTimer that times each action).

    Two entry variants sharing the router/selector/plan tail:
      * classic (recipe_slug): load_recipe → load_products → …
      * NL2SQL (recipe_text):  parse_and_retrieve → …  (recipe + narrowed
        products both come from the pasted text)

    Conditional transitions:
      * check_escalation → escalate_if_needed  if escalation_decision.escalate
      * check_escalation → skip_escalation     otherwise
    """
    if (recipe_slug is None) == (recipe_text is None):
        raise ValueError("provide exactly one of recipe_slug / recipe_text")

    shared_tail = [
        ("preselect_model", "select_products"),
        ("select_products", "check_escalation"),
        (
            "check_escalation",
            "escalate_if_needed",
            expr("escalation_decision.escalate == True"),
        ),
        (
            "check_escalation",
            "skip_escalation",
            expr("escalation_decision.escalate == False"),
        ),
        ("escalate_if_needed", "optimize_trips"),
        ("skip_escalation", "optimize_trips"),
        ("optimize_trips", "build_plan"),
    ]
    common_actions = [preselect_model, select_products, check_escalation,
                      escalate_if_needed, skip_escalation, optimize_trips,
                      build_plan]

    if recipe_text is not None:
        builder = (
            ApplicationBuilder()
            .with_actions(parse_and_retrieve.bind(recipe_text=recipe_text,
                                                  lat=lat, lon=lon,
                                                  exclude=exclude,
                                                  preference=preference,
                                                  max_km=max_km,
                                                  allow_partial=allow_partial),
                          *common_actions)
            .with_transitions(("parse_and_retrieve", "preselect_model"),
                              *shared_tail)
            .with_entrypoint("parse_and_retrieve")
            .with_identifiers(app_id=_run_id("nl"))
        )
    else:
        # A store-aware classic plan when the caller gives any of lat/lon/max_km; lat/lon
        # default to the server's shopping point and max_km None means any distance.
        location = None
        if lat is not None or lon is not None or max_km is not None:
            cfg = settings()
            location = {"lat": cfg.default_lat if lat is None else lat,
                        "lon": cfg.default_lon if lon is None else lon, "max_km": max_km}
        builder = (
            ApplicationBuilder()
            .with_actions(load_recipe.bind(recipe_slug=recipe_slug, location=location),
                          load_products.bind(exclude=exclude,
                                             preference=preference,
                                             allow_partial=allow_partial),
                          *common_actions)
            .with_transitions(("load_recipe", "load_products"),
                              ("load_products", "preselect_model"),
                              *shared_tail)
            .with_entrypoint("load_recipe")
            .with_identifiers(app_id=_run_id(recipe_slug))
        )
    builder = builder.with_tracker(make_tracker())
    return (builder.with_hooks(*hooks) if hooks else builder).build()


def _run_id(name: str) -> str:
    """One Burr run per plan call, so the Burr UI lists each call on its own (newest first)
    instead of appending every plan of a recipe to one ever-growing run:
    run-<recipe or nl>-<local time>-<6 hex>."""
    return f"run-{name}-{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"


def run(recipe_slug: str, *, exclude: list | None = None,
        preference: list | None = None, lat: float | None = None,
        lon: float | None = None, max_km: float | None = None,
        allow_partial: bool = False) -> ShoppingPlan:
    """Run the classic pipeline end-to-end. Returns the final ShoppingPlan.

    `exclude` removes candidates positively evidenced as coming from those
    countries; `preference` is passed to the selector as soft guidance.
    Any of `lat`/`lon`/`max_km` makes the plan store-aware (stores per line,
    trip options); without them it is priced from the catalog as before.
    `allow_partial` plans the rest when the exclusion leaves an ingredient
    with no candidate, reporting it on `out_of_range` with what to buy
    instead (a gate still fires when nothing would remain).
    """
    timer = StepTimer()
    app = build_application(recipe_slug=recipe_slug, exclude=exclude,
                            preference=preference, lat=lat, lon=lon, max_km=max_km,
                            allow_partial=allow_partial, hooks=[timer])
    _action, _result, state = app.run(halt_after=["build_plan"])
    return state["plan"].model_copy(update={"burr_run": app.uid, "pipeline": timer.steps})


def run_nl(recipe_text: str, lat: float | None = None,
           lon: float | None = None, *, exclude: list | None = None,
           preference: list | None = None, max_km: float | None = None,
           allow_partial: bool = False) -> ShoppingPlan:
    """Run the NL2SQL pipeline on pasted recipe text.

    Raises nlsearch.UnparseableRecipe when no ingredient list is found
    (API → 422 with guidance) and nlsearch.PlanAborted when a query-plan
    gate fires (API → 409 with the PlanAlert + trace). `max_km` overrides
    any distance stated in the text; `allow_partial` plans the stocked,
    in-range ingredients and reports the rest on the plan's `not_stocked` /
    `out_of_range` instead of aborting (a gate still fires when nothing
    would remain).
    """
    timer = StepTimer()
    app = build_application(recipe_text=recipe_text, lat=lat, lon=lon,
                            exclude=exclude, preference=preference,
                            max_km=max_km, allow_partial=allow_partial, hooks=[timer])
    _action, _result, state = app.run(halt_after=["build_plan"])
    return state["plan"].model_copy(update={"burr_run": app.uid, "pipeline": timer.steps})

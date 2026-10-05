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
from dataclasses import dataclass, field
from typing import Any

from burr.core import Application, ApplicationBuilder, State, action, expr

from . import db
from .config import get_router, settings
from .models import (
    DroppedIngredient,
    EscalationDecision,
    PlanLineItem,
    PreselectResult,
    Product,
    Recipe,
    RecipeIngredient,
    Selection,
    SelectorResult,
    ShoppingPlan,
)
from .selector import call_selector, merge_selections
from .tracing import llm_span, make_tracker

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
    (a teaspoon of peppercorns against a 50 g bag says nothing about bags)."""
    if (len(needs) < 2 or not product.unit_qty
            or any(n is None or n[1] != product.unit_uom for n in needs)):
        return 1
    total = sum(n[0] for n in needs if n is not None)
    return max(1, math.ceil(total / product.unit_qty - 1e-9))


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

@action(reads=[], writes=["recipe"])
def load_recipe(state: State, recipe_slug: str) -> tuple[dict, State]:
    recipe = db.load_recipe(recipe_slug)
    result = {"ingredient_count": len(recipe.ingredients)}
    return result, state.update(recipe=recipe)


@action(reads=["recipe"], writes=["products", "origins", "origin_dropped", "preference",
                                 "origin_requested"])
def load_products(state: State, exclude: list | None = None,
                  preference: list | None = None) -> tuple[dict, State]:
    products = db.load_all_products()
    recipe = state.get("recipe")
    cfg = settings()
    pools = _ingredient_pools(recipe, cfg.default_lat, cfg.default_lon) if recipe else {}
    names = {i: ing.name for i, ing in enumerate(recipe.ingredients)} if recipe else {}
    kept, dropped, origins, _pools = apply_origin_constraint(
        products, pools, names, exclude=exclude, preference=preference)
    result = {"product_count": len(kept), "origin_dropped": len(dropped)}
    return result, state.update(products=kept, origins=origins,
                                origin_dropped=dropped,
                                preference=list(preference or []),
                                origin_requested=bool(exclude or preference))


@action(reads=[], writes=["recipe", "products", "parsed_input", "location",
                          "plan_trace", "brand_stats", "retrieval_stats",
                          "origins", "origin_dropped", "preference",
                          "origin_requested", "not_stocked", "out_of_range",
                          "skipped", "pool_hints", "match_levels",
                          "ingredient_count"])
def parse_and_retrieve(state: State, recipe_text: str,
                       lat: float | None = None,
                       lon: float | None = None,
                       exclude: list | None = None,
                       preference: list | None = None,
                       max_km: float | None = None,
                       allow_partial: bool = False) -> tuple[dict, State]:
    """NL2SQL entrypoint: pasted recipe text → query-plan execution
    (t1 existence → t2 options → t3 brand stats → t4 lookups) producing an
    ad-hoc Recipe + store-priced candidate pools. Replaces load_recipe +
    load_products on the NL path; downstream actions consume the same state
    keys. Gate aborts raise nlsearch.PlanAborted → API 409. `max_km`
    overrides the parsed distance; with `allow_partial` the recipe in state
    holds only the ingredients still planned, and the drops ride along."""
    from . import nlsearch

    # With an exclusion active, retrieve wider than the usual cheapest-8 so
    # the 9th-cheapest non-excluded product is still there after filtering;
    # the pool is trimmed back to the normal size below.
    from .nlsearch.sql_builder import PER_INGREDIENT_LIMIT
    limit = 10_000 if exclude else PER_INGREDIENT_LIMIT
    r = nlsearch.run_query_plan(recipe_text, lat=lat, lon=lon,
                                per_ingredient_limit=limit, max_km=max_km,
                                allow_partial=allow_partial)
    names = {i: ing.name for i, ing in enumerate(r.recipe.ingredients)}
    kept, dropped, origins, kept_pools = apply_origin_constraint(
        r.products, r.pools, names, exclude=exclude, preference=preference)
    if exclude and kept_pools:
        def price(p):
            return p.store_price if p.store_price is not None else p.price
        seen: set[int] = set()
        kept = []
        for direct, alts in kept_pools.values():
            for p in sorted(direct, key=price)[:PER_INGREDIENT_LIMIT] + alts:
                if p.id not in seen:
                    seen.add(p.id)
                    kept.append(p)
    result = {
        "ingredient_count": len(r.recipe.ingredients),
        "product_count": len(kept),
        "origin_dropped": len(dropped),
        "plan_steps": [s.step_id for s in r.execution.steps],
        "parse_cost_usd": r.parsed.cost_usd,
    }
    # Each line's best candidates, named on `skipped` if the selector then
    # returns no product for the line.
    kept_ids = {p.id for p in kept}
    pool_hints = {
        ing.line_no: [f"{p.name} (${p.store_price if p.store_price is not None else p.price:.2f})"
                      for p in r.pools.get(i, []) if p.id in kept_ids and not p.substitute][:3]
        for i, ing in enumerate(r.recipe.ingredients)}
    return result, state.update(
        recipe=r.recipe, products=kept, origins=origins,
        origin_dropped=dropped, preference=list(preference or []),
        origin_requested=bool(exclude or preference),
        parsed_input=r.parsed,
        location={"lat": r.lat, "lon": r.lon, "max_km": r.max_km},
        plan_trace=r.execution.steps, brand_stats=r.brand_stats,
        retrieval_stats=r.stats, not_stocked=r.not_stocked,
        out_of_range=r.out_of_range, skipped=r.skipped, pool_hints=pool_hints,
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


@action(reads=["recipe", "products", "preselect_result", "origins", "preference"],
        writes=["initial_result"])
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
    )
    from . import metrics as _m
    _m.record_llm(span)
    return {"llm_call": span, "n_selections": len(result.selections)}, \
        state.update(initial_result=result)


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
    reads=["recipe", "products", "initial_result", "escalation_decision"],
    writes=["final_result"],
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
    )
    return {"llm_call": span, "n_reran": len(flagged_ingredients)}, \
        state.update(final_result=merged)


@action(reads=["initial_result", "final_result"], writes=["final_result"])
def skip_escalation(state: State) -> tuple[dict, State]:
    """No-op — hoists initial_result into final_result when we don't escalate."""
    return {"escalated": False}, state.update(final_result=state["initial_result"])


@action(reads=["final_result", "products", "recipe", "parsed_input", "location",
               "plan_trace"],
        writes=["trip_options", "plan_trace"])
def optimize_trips(state: State) -> tuple[dict, State]:
    """4A: deterministic split-trip optimizer over the chosen basket.
    Prices every PURCHASE (group_purchases: a product shared by two lines
    once, times its packs) at every in-range store (one templated query),
    then enumerates store subsets with exact travel loops — no LLM. NL path
    only; the classic path has no location and passes straight through."""
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
            trip_options=[], plan_trace=state.get("plan_trace") or [])

    sql, params = build_price_matrix_sql(
        sorted({pid for pid, _ in basket}), loc["lat"], loc["lon"], loc["max_km"])
    t0 = _time.perf_counter()
    with _Session(db.engine()) as s:
        rows = list(s.execute(_text(sql), params).mappings())
    duration_ms = int((_time.perf_counter() - t0) * 1000)

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
        plan_trace=[*(state.get("plan_trace") or []), step])


@action(
    reads=["recipe", "products", "final_result", "preselect_result",
           "escalation_decision", "trip_options", "parsed_input", "origins",
           "origin_requested", "origin_dropped", "match_levels", "not_stocked",
           "out_of_range", "skipped", "pool_hints", "ingredient_count",
           "plan_trace"],
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
    purchases, unselected = group_purchases(final.selections, products_by_id,
                                            ingredients_by_line, _needs(parsed))
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
        trip_options=state.get("trip_options") or [],
        not_stocked=state.get("not_stocked") or [],
        out_of_range=state.get("out_of_range") or [],
        skipped=skipped,
        ingredient_count=state.get("ingredient_count") or len(recipe.ingredients),
    )
    return {"total_cost": plan.total_cost, "n_line_items": len(plan.line_items)}, \
        state.update(plan=plan)


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


def apply_origin_constraint(products, pools, names, *, exclude, preference):
    """Filter `products` on origin and gate per ingredient.

    products : what the selector will be shown (filtered copy returned)
    pools    : ingredient key -> candidate list, used ONLY for gating
    names    : ingredient key -> display name

    Returns (kept_products, dropped, origins). `dropped` is deduplicated by
    product. Raises PlanAborted(excluded_by_origin) naming EVERY ingredient
    whose pool had candidates and lost all of them, each with the removed
    products as suggestions — that is the trade the user is being asked to
    make, stated rather than silently made for them.

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
        return list(products), [], origins, {}

    kept_products, dropped = origins_mod.filter_pool(
        products, exclude=exclude, origins=origins)
    dropped_by_id = {p.id: (p, c, f) for p, c, f in dropped}

    def split(pool):
        """(direct candidates, alternatives). Classic pools are dicts; NL pools
        are flat lists where t4 substitutes carry substitute=True."""
        if isinstance(pool, dict):
            return list(pool.get("direct", [])), list(pool.get("relaxed", []))
        return [p for p in pool if not p.substitute], [p for p in pool if p.substitute]

    emptied: list[tuple[str, list, list]] = []
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
            emptied.append((names.get(key, str(key)), dropped_direct, kept_alt))

    if emptied:
        def price(p):
            return p.store_price if p.store_price is not None else p.price
        details = [{
            "name": name,
            "reason": (f"all {len(removed)} candidate(s) are evidenced as "
                       f"coming from an excluded country"),
            "suggestions": [
                f"{p.name} (${price(p):.2f}) — {country} via {field}"
                for p, country, field in removed[:5]
            ] + [
                f"still available, not a direct match: {p.name} (${price(p):.2f})"
                for p in sorted(alts, key=price)[:3]
            ],
        } for name, removed, alts in emptied]
        affected = ", ".join(n for n, _, _ in emptied)
        raise PlanAborted(PlanExecution(steps=[], aborted=PlanAlert(
            stage="origin_filter", code=GateCode.excluded_by_origin,
            message=(f"Excluding {', '.join(exclude)} left no candidate for: "
                     f"{affected}. Relax the exclusion, or accept one of the "
                     f"removed products listed per ingredient."),
            details=details)))
    return kept_products, list(dropped_by_id.values()), origins, kept_pools


def build_application(recipe_slug: str | None = None,
                      recipe_text: str | None = None,
                      lat: float | None = None,
                      lon: float | None = None,
                      exclude: list | None = None,
                      preference: list | None = None,
                      max_km: float | None = None,
                      allow_partial: bool = False) -> Application:
    """Construct the Burr Application for one run.

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
            .with_identifiers(app_id="run-nl")
        )
    else:
        builder = (
            ApplicationBuilder()
            .with_actions(load_recipe.bind(recipe_slug=recipe_slug),
                          load_products.bind(exclude=exclude,
                                             preference=preference),
                          *common_actions)
            .with_transitions(("load_recipe", "load_products"),
                              ("load_products", "preselect_model"),
                              *shared_tail)
            .with_entrypoint("load_recipe")
            .with_identifiers(app_id=f"run-{recipe_slug}")
        )
    return builder.with_tracker(make_tracker()).build()


def run(recipe_slug: str, *, exclude: list | None = None,
        preference: list | None = None) -> ShoppingPlan:
    """Run the classic pipeline end-to-end. Returns the final ShoppingPlan.

    `exclude` removes candidates positively evidenced as coming from those
    countries; `preference` is passed to the selector as soft guidance.
    """
    app = build_application(recipe_slug=recipe_slug, exclude=exclude,
                            preference=preference)
    _action, _result, state = app.run(halt_after=["build_plan"])
    return state["plan"]


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
    app = build_application(recipe_text=recipe_text, lat=lat, lon=lon,
                            exclude=exclude, preference=preference,
                            max_km=max_km, allow_partial=allow_partial)
    _action, _result, state = app.run(halt_after=["build_plan"])
    return state["plan"]

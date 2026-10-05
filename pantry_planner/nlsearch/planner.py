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

allow_partial=True turns the two per-ingredient gates into drops: a t1
miss is reported in `not_stocked`, a t2 miss in `out_of_range`, and the
plan prices what remains. Only when nothing remains does the gate abort as
before. The budget gate is a basket-level verdict and always aborts.

Some lines are never retrieved at all and land in `skipped`, whatever
allow_partial says: non-purchases (water, ice; units.NON_PURCHASES), which
are never priced, and ingredients past the parser's 40-ingredient cap. So
every parsed ingredient is in exactly one of: the planned recipe,
not_stocked, out_of_range, skipped.

Matching runs at four levels, each tried only when the one before found
nothing in the catalog: exact (every token, purchase form included),
equivalent form (powder/ground swapped — "cumin powder" -> Cumin Ground),
form (the form dropped) and generic (descriptor words dropped too — "light
brown sugar" -> brown sugar; units.DESCRIPTORS). The level is reported per
line; the two middle levels both report "form".
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
from ..models import DroppedIngredient, Product, Recipe, RecipeIngredient
from .plan import (GateCode, PlanAlert, PlanExecution, QueryPlan, QueryStep,
                   StepKind, StepResult)
from .query_parser import parse_input, validate_parsed
from .schemas import ParsedInput, RecipeSpec, RetrievalStats
from .sql_builder import (PER_INGREDIENT_LIMIT, THIN_POOL, build_existence_sql,
                          build_options_sql,
                          build_stats_sql, build_substitute_sql,
                          build_suggestions_sql, build_token_suggestions_sql,
                          inline_for_display)
from .units import DESCRIPTORS, PURCHASE_FORMS, head_noun, is_non_purchase, tokens
from .vocab import db_vocab

# Reasons on `skipped` entries.
NOT_BOUGHT = "not bought: water and ice are never priced"
OVER_CAP = "over the 40-ingredient cap: not planned"


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
    # allow_partial drops (empty unless allow_partial=True dropped something)
    not_stocked: list[DroppedIngredient] = field(default_factory=list)
    out_of_range: list[DroppedIngredient] = field(default_factory=list)
    # never retrieved: non-purchases and lines past the 40-ingredient cap
    skipped: list[DroppedIngredient] = field(default_factory=list)
    # recipe line_no -> "exact" | "form" | "generic" for every planned line
    match_levels: dict[int, str] = field(default_factory=dict)
    ingredient_count: int = 0                  # ingredients in the recipe, before drops


def _slugify(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_") or "adhoc_recipe"


def build_recipe(spec: RecipeSpec, keep: list[int] | None = None) -> Recipe:
    """The plan's Recipe comes from the pasted text, not the DB. `keep`
    (positions into spec.ingredients) is what a partial plan still plans;
    line numbers stay those of the recipe as written."""
    if not spec.ingredients:
        raise UnparseableRecipe()
    positions = range(len(spec.ingredients)) if keep is None else keep
    return Recipe(
        slug=_slugify(spec.title),
        name=spec.title,
        servings=spec.servings,
        ingredients=[
            RecipeIngredient(line_no=i + 1, name=spec.ingredients[i].name,
                             category=spec.ingredients[i].category_hint)
            for i in positions
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


def _offer_text(p: Product) -> str:
    return f"{p.name} at {p.store_name} (${p.store_price:.2f}, {p.distance_km} km)"


def _outside(p: Product, c, max_km: float | None) -> str:
    """Which limit an offer breaks — the knob that would bring it in."""
    broken = []
    if max_km is not None and p.distance_km is not None and p.distance_km > max_km:
        broken.append(f"beyond the {max_km:g} km limit")
    if (c.max_item_price is not None and p.store_price is not None
            and p.store_price > c.max_item_price):
        broken.append(f"over the ${c.max_item_price:.2f} per-item price cap")
    return " and ".join(broken)


def _attribute(s: Session, c, ingredients, empty: list[int], relaxed: set[int],
               generic: set[int], lat: float, lon: float,
               max_km: float | None = None,
               equivalent: set[int] | None = None) -> list[dict]:
    """Constraint attribution for t2 misses: the same template with the
    distance and price caps stripped, each product pinned to its NEAREST
    store. Anything that reappears was located or priced out, not missing
    (t1 already proved it exists); the reason names the nearest such offer
    and which limit it breaks (distance, price cap or both), the
    suggestions the next ones. Nothing reappearing means the diet or
    category filters exclude every match."""
    equivalent = equivalent or set()
    probe_ings = [ingredients[n] for n in empty]
    probe_c = c.model_copy(update={"max_item_price": None})
    sql, params = build_options_sql(
        probe_c, probe_ings, {i for i, n in enumerate(empty) if n in relaxed},
        lat, lon, None, per_ingredient_limit=10_000,
        generic={i for i, n in enumerate(empty) if n in generic},
        equivalent={i for i, n in enumerate(empty) if n in equivalent},
        nearest_store=True)
    probe: dict[int, list[Product]] = {}
    for row in s.execute(text(sql), params).mappings():
        probe.setdefault(empty[row["ing_no"]], []).append(_row_to_product(row))
    details = []
    for n in empty:
        alt = sorted(probe.get(n, []),
                     key=lambda p: (p.distance_km, p.store_price, p.id))
        if alt:
            why = _outside(alt[0], c, max_km)
            reason = ("available only outside the constraints — nearest: "
                      + _offer_text(alt[0]) + (f", {why}" if why else ""))
            suggestions = [_offer_text(p) for p in alt[1:4]]
        else:
            reason, suggestions = "no offer passes the dietary/category constraints", []
        details.append({"name": ingredients[n].name, "reason": reason,
                        "suggestions": suggestions})
    return details


# Words too common or too vague to make a product "related" on their own: a
# not_stocked "brown rice" must not be offered Brown Sugar for sharing
# "brown", nor "Kashmiri chili powder" Baking Powder for sharing "powder".
# They still break ties between products that share a strong word.
WEAK_SUGGESTION_WORDS = DESCRIPTORS | PURCHASE_FORMS | {
    "red", "green", "white", "black", "yellow", "orange", "brown", "golden",
    "purple", "sweet", "hot", "mild", "spicy", "plain", "pure", "powder",
}


def _related(s: Session, name: str, catalog_size: int, limit: int = 3) -> list[str]:
    """Products sharing a strong word with a not-stocked ingredient, best
    first: the summed rarity (log N/df) of the strong words they share; then
    whether the product is ABOUT the ingredient's head noun ("brown rice":
    Basmati Rice before Rice Vinegar); then weak shared words; then price.
    [] when no product shares a strong word."""
    toks = list(dict.fromkeys(tokens(name)))
    strong = {t for t in toks if t not in WEAK_SUGGESTION_WORDS}
    if not strong:
        return []
    head = toks[-1]
    sql, params = build_token_suggestions_sql(toks)
    scored: dict[int, list] = {}         # id -> [strong score, head miss, weak hits, price, name]
    for r in s.execute(text(sql), params).mappings():
        row = scored.setdefault(r["id"], [0.0, int(head_noun(r["name"]) != head), 0,
                                          r["price"], r["name"]])
        if r["term"] in strong:
            row[0] += math.log(max(catalog_size, 1) / max(r["df"], 1)) + 1e-6
        else:
            row[2] += 1
    best = sorted((v for v in scored.values() if v[0] > 0),
                  key=lambda v: (-round(v[0], 6), v[1], -v[2], v[3], v[4]))
    return [f"{name_} (${price:.2f})" for *_k, price, name_ in best[:limit]]


def _missing_details(s: Session, ingredients, missing: list[int]) -> list[dict]:
    """not_stocked entries. Suggestions are products related to the
    ingredient by a shared word (see _related); only when nothing shares one
    do they fall back to the cheapest products of the parser's category
    hint, which names an aisle, not the ingredient."""
    catalog_size = s.execute(text("SELECT COUNT(*) FROM products")).scalar_one()
    details = []
    for n in missing:
        ing = ingredients[n]
        suggestions = _related(s, ing.name, catalog_size)
        if not suggestions and ing.category_hint:
            sg_sql, sg_params = build_suggestions_sql(ing.category_hint.lower())
            suggestions = [f"{r['name']} (${r['price']:.2f})"
                           for r in s.execute(text(sg_sql), sg_params).mappings()]
        details.append({"name": ing.name, "reason": "not in the catalog",
                        "suggestions": suggestions})
    return details


def _dropped(details: list[dict]) -> list[DroppedIngredient]:
    return [DroppedIngredient(ingredient=d["name"], reason=d["reason"],
                              suggestions=d["suggestions"]) for d in details]


def execute_plan(parsed: ParsedInput, plan: QueryPlan, *,
                 lat: float, lon: float, max_km: float | None,
                 per_ingredient_limit: int = PER_INGREDIENT_LIMIT,
                 allow_partial: bool = False) -> PlanRunResult:
    build_recipe(parsed.recipe)                 # raises UnparseableRecipe -> 422
    every = parsed.recipe.ingredients
    c = parsed.constraints
    execution = PlanExecution()
    not_stocked: list[DroppedIngredient] = []
    out_of_range: list[DroppedIngredient] = []
    # Never retrieved, whatever allow_partial says: what nobody buys, and
    # what the parser's 40-ingredient cap cut (validate_parsed).
    skipped = [DroppedIngredient(ingredient=ing.name, reason=NOT_BOUGHT)
               for ing in every if is_non_purchase(ing.name)]
    skipped += [DroppedIngredient(ingredient=ing.name, reason=OVER_CAP)
                for ing in parsed.over_cap]
    # positions into the recipe as written that retrieval sees
    buyable = [n for n in range(len(every)) if not is_non_purchase(every[n].name)]
    if not buyable:
        _abort(execution, PlanAlert(
            stage="t1_existence", code=GateCode.missing_ingredients,
            message=("Nothing to buy: " + ", ".join(ing.name for ing in every)
                     + " — water and ice are never bought."),
            details=[{"name": d.ingredient, "reason": d.reason, "suggestions": []}
                     for d in skipped], partial_would_plan=0))

    with Session(db.engine()) as s:
        # ── t1: existence probe (exact -> equivalent form -> form -> generic) ──
        probed = [every[n] for n in buyable]
        sql, params = build_existence_sql(probed)
        rows, ms = _timed(s, sql, params)
        counts = {buyable[r["ing_no"]]: (r["strict_matches"], r["equivalent_matches"],
                                         r["relaxed_matches"], r["generic_matches"])
                  for r in rows if r["ing_no"] >= 0}
        how: dict[int, str] = {}                # recipe position -> term level
        for n in buyable:
            strict, eqv, rel, gen = counts.get(n, (0, 0, 0, 0))
            if strict:
                how[n] = "exact"
            elif eqv:
                how[n] = "equivalent"
            elif rel:
                how[n] = "relaxed"
            elif gen:
                how[n] = "generic"
        level = {n: {"exact": "exact", "generic": "generic"}.get(h, "form")
                 for n, h in how.items()}
        missing = [n for n in buyable if n not in level]
        drop_missing = allow_partial and bool(missing) and len(missing) < len(buyable)
        n_form = sum(1 for v in level.values() if v == "form")
        n_generic = sum(1 for v in level.values() if v == "generic")
        not_bought = len(every) - len(buyable)
        execution.steps.append(StepResult(
            step_id="t1_existence", kind=StepKind.existence,
            sql_display=inline_for_display(sql, params),
            row_count=len(rows), duration_ms=ms,
            outcome="aborted" if missing and not drop_missing else "ok",
            label=(f"{len(buyable) - len(missing)}/{len(buyable)} ingredients stocked"
                   + (f" ({n_form} via form relaxation)" if n_form else "")
                   + (f" ({n_generic} via generic match)" if n_generic else "")
                   + (f"; {not_bought} not bought (water/ice)" if not_bought else "")
                   + (f"; {len(missing)} not stocked, dropped (allow_partial)"
                      if drop_missing else ""))))
        if missing:
            details = _missing_details(s, every, missing)
            if not drop_missing:
                _abort(execution, PlanAlert(
                    stage="t1_existence", code=GateCode.missing_ingredients,
                    message=("We don't stock: "
                             + ", ".join(every[n].name for n in missing)
                             + ". Remove or substitute them and retry."),
                    details=details,
                    partial_would_plan=len(buyable) - len(missing)))
            not_stocked = _dropped(details)

        # `alive`: positions into the recipe as written that are still being
        # planned. Every index below (pools, t2 rows) is into `ingredients`,
        # the alive subset in recipe order.
        alive = [n for n in buyable if n in level]
        ingredients = [every[n] for n in alive]

        def at(kind: str) -> set[int]:
            return {i for i, n in enumerate(alive) if how[n] == kind}
        relaxed, generic, equivalent = at("relaxed"), at("generic"), at("equivalent")

        # ── t2: options under constraints ──
        sql, params = build_options_sql(c, ingredients, relaxed, lat, lon, max_km,
                                        per_ingredient_limit=per_ingredient_limit,
                                        generic=generic, equivalent=equivalent)
        rows, ms = _timed(s, sql, params)
        pools: dict[int, list[Product]] = {}
        for row in rows:
            pools.setdefault(row["ing_no"], []).append(_row_to_product(row))
        empty = [i for i in range(len(ingredients)) if not pools.get(i)]
        drop_empty = allow_partial and bool(empty) and len(empty) < len(ingredients)
        execution.steps.append(StepResult(
            step_id="t2_options", kind=StepKind.options,
            sql_display=inline_for_display(sql, params),
            row_count=len(rows), duration_ms=ms,
            outcome="aborted" if empty and not drop_empty else "ok",
            label=(f"{len(rows)} offers across {len(pools)} ingredient pools"
                   + (f"; {len(empty)} with no offer within the constraints, "
                      "dropped (allow_partial)" if drop_empty else ""))))
        if empty:
            details = _attribute(s, c, ingredients, empty, relaxed, generic, lat, lon,
                                 max_km=max_km, equivalent=equivalent)
            if not drop_empty:
                # Nothing left to plan: the gate aborts as it always has, and
                # anything t1 already dropped is named too, so the alert
                # accounts for every ingredient.
                _abort(execution, PlanAlert(
                    stage="t2_options", code=GateCode.unavailable_within_constraints,
                    message=("No options within the current constraints for: "
                             + ", ".join(ingredients[i].name for i in empty)
                             + ". Widen the distance/budget or relax a filter."
                             + _left_out(not_stocked, [])),
                    details=details + _as_details(not_stocked),
                    partial_would_plan=len(ingredients) - len(empty)))
            out_of_range = _dropped(details)
            keep = [i for i in range(len(ingredients)) if i not in set(empty)]
            pools = {j: pools[i] for j, i in enumerate(keep)}
            alive = [alive[i] for i in keep]
            ingredients = [ingredients[i] for i in keep]

        # Basket feasibility floor — pure Python over the fetched pools.
        if c.max_total_budget is not None:
            floor = sum(min(p.store_price for p in pool) for pool in pools.values())
            if floor > c.max_total_budget:
                execution.steps[-1].outcome = "aborted"
                # A partial plan cannot fix a budget, so this aborts even with
                # allow_partial — and names what was already left out, so the
                # alert still accounts for every ingredient.
                _abort(execution, PlanAlert(
                    stage="t2_options", code=GateCode.budget_infeasible,
                    message=(f"Cheapest possible basket is ${floor:.2f} — over the "
                             f"${c.max_total_budget:.2f} budget. Raise the budget or "
                             "trim the recipe." + _left_out(not_stocked, out_of_range)),
                    details=[{"name": ingredients[n].name,
                              "reason": f"cheapest option ${min(p.store_price for p in pool):.2f}",
                              "suggestions": []}
                             for n, pool in sorted(pools.items())]
                    + _as_details(not_stocked) + _as_details(out_of_range)))

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
            # step ids keep the recipe position, so they read the same
            # whether or not a partial plan dropped earlier ingredients
            plan.steps.append(QueryStep(
                id=f"t4_lookup_{alive[n]}", kind=StepKind.lookup,
                template="substitute_lookup",
                params_summary={"ingredient": ingredients[n].name, "subcategory": sub}))
            execution.steps.append(StepResult(
                step_id=f"t4_lookup_{alive[n]}", kind=StepKind.lookup,
                sql_display=inline_for_display(sql, params),
                row_count=len(rows), duration_ms=ms,
                label=f"{len(subs)} substitutes for {ingredients[n].name} ({sub})"))

        catalog_size = s.execute(text("SELECT COUNT(*) FROM products")).scalar_one()

    seen: set[int] = set()
    products = [p for pool in pools.values() for p in pool
                if p.id not in seen and not seen.add(p.id)]
    stats = RetrievalStats(
        pool_sizes=[len(pools.get(n, [])) for n in range(len(ingredients))],
        # every planned ingredient that needed a looser match than exact
        zero_hit_ingredients=sum(1 for n in alive if level[n] != "exact"),
        value_disagreement=_value_disagreement(pools),
        catalog_size=catalog_size,
    )
    return PlanRunResult(recipe=build_recipe(parsed.recipe, keep=alive),
                         products=products, pools=pools,
                         stats=stats, parsed=parsed, execution=execution,
                         brand_stats=brand_stats, lat=lat, lon=lon, max_km=max_km,
                         not_stocked=not_stocked, out_of_range=out_of_range,
                         skipped=skipped,
                         match_levels={n + 1: level[n] for n in alive},
                         ingredient_count=len(every) + len(parsed.over_cap))


def _as_details(dropped: list[DroppedIngredient]) -> list[dict]:
    return [{"name": d.ingredient, "reason": d.reason, "suggestions": d.suggestions}
            for d in dropped]


def _left_out(not_stocked: list[DroppedIngredient],
              out_of_range: list[DroppedIngredient]) -> str:
    """The alert sentence naming ingredients a gate's predecessors dropped."""
    out = ""
    if not_stocked:
        out += " Not stocked at all: " + ", ".join(d.ingredient for d in not_stocked) + "."
    if out_of_range:
        out += (" No offer within the constraints: "
                + ", ".join(d.ingredient for d in out_of_range) + ".")
    return out


def run_query_plan(text_input: str, *, parsed: ParsedInput | None = None,
                   lat: float | None = None, lon: float | None = None,
                   per_ingredient_limit: int = PER_INGREDIENT_LIMIT,
                   max_km: float | None = None,
                   allow_partial: bool = False) -> PlanRunResult:
    """Full NL2SQL retrieval. `parsed` injectable for tests (skips the LLM).

    `max_km`, when given, is the distance limit and overrides any distance
    the parser read from the text (the effective value is what the
    interpretation shows). `allow_partial` — see the module docstring."""
    from ..config import settings

    cfg = settings()
    if parsed is None:
        if cfg.demo_mode:                   # public demo: deterministic parse
            from .. import demomode
            parsed = demomode.parse_recipe(text_input)
        else:
            parsed = parse_input(text_input)
    parsed = validate_parsed(parsed, db_vocab())
    if max_km is not None:
        stated = parsed.constraints.max_distance_km
        if stated is not None and stated != max_km:
            parsed.ignored.append(f"{stated:g} km from the text (max_km={max_km:g} given)")
        parsed.constraints.max_distance_km = max_km
    max_km = parsed.constraints.max_distance_km
    plan = build_plan(parsed, max_km=max_km)
    return execute_plan(parsed, plan,
                        lat=lat if lat is not None else cfg.default_lat,
                        lon=lon if lon is not None else cfg.default_lon,
                        max_km=max_km,
                        per_ingredient_limit=per_ingredient_limit,
                        allow_partial=allow_partial)

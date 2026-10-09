"""POST /mealplan/schedule: a MealPlanDraft in, a MealSchedule out. Pure code.

No LLM call and no write. The server trusts only ids from the draft: product names, sizes,
categories and prices are re-read from the database on every call, recipe amounts from the
recipe itself (resolve.recipe_doc), storage times from shelf_life.json. Output depends only
on the draft and the data, so the same draft gives byte-identical JSON.

Both strategies are computed every time:
- fresh: no freezing (unless the shopper set a product to the freezer); trips wherever the
  cited fridge times and the shopper's buy-ahead setting need them;
- fewest_trips: products with a cited freezer time and thaw time may be frozen on arrival
  and thawed before the meal, so fewer trips cover the plan.
fresh is recommended unless it has must-fix warnings that fewest_trips does not.
"""
from __future__ import annotations

import datetime as dt
import functools
import json

from sqlalchemy import text
from sqlalchemy.orm import Session

from .. import db, tripopt
from ..config import settings
from ..db import SEEDS_DIR
from ..models import Product, RecipeDoc
from ..nlsearch.sql_builder import build_price_matrix_sql
from . import approved as approved_mod
from . import lists, place, shelf, trips
from . import warnings as warn
from .models import (
    SLOTS,
    STRATEGIES,
    Action,
    ApprovedTrip,
    Coverage,
    DayOut,
    LineProduct,
    MealPlanDraft,
    MealPlanError,
    MealRef,
    MealSchedule,
    PlacedMeal,
    PlanWarning,
    ShelfLifeInfo,
    StrategyResult,
    Trip,
    TripLine,
)
from .needs import USABLE, Need, line_products, meal_servings, needs_for_meal
from .needs import recipe_servings as _recipe_servings
from .place import SHORT, Board, day_label, slot_of
from .resolve import recipe_doc

SYNTHETIC_NOTICE = ("Stores, prices and stock are synthetic demo data; a trip date does not "
                    "change a price or what is stocked.")
DEMO_PRODUCTS_FILE = SEEDS_DIR / "demo_products.json"


@functools.lru_cache(maxsize=1)
def demo_product_ids() -> frozenset[int]:
    data = json.loads(DEMO_PRODUCTS_FILE.read_text(encoding="utf-8"))
    return frozenset(p["id"] for p in data["products"])


# ─── Validation ──────────────────────────────────────────────

def _check_dates(draft: MealPlanDraft) -> None:
    for label, dates in (("fixed date", draft.fixed_dates),
                         ("dismissed date", draft.dismissed_dates),
                         ("approved trip", [t.date for t in draft.trips])):
        for d in dates:
            if not 0 <= draft.index(d) < draft.days:
                raise MealPlanError("invalid_dates", f"{label} {d.isoformat()} is outside the "
                                    f"plan ({draft.start_date.isoformat()}, {draft.days} days)")


def _packs_overrides(draft: MealPlanDraft, products: dict[int, Product]) -> dict:
    """'<date>:<product_id>' -> packs, parsed and checked."""
    out: dict[tuple[int, int], int] = {}
    for raw, packs in draft.packs_override.items():
        date_s, _, pid_s = raw.partition(":")
        try:
            day, pid = draft.index(dt.date.fromisoformat(date_s)), int(pid_s)
        except ValueError as e:
            raise MealPlanError("invalid_dates", f"packs_override key {raw!r} is not "
                                "'<YYYY-MM-DD>:<product_id>'") from e
        if not 0 <= day < draft.days:
            raise MealPlanError("invalid_dates", f"packs_override {raw!r} is outside the plan")
        if pid not in products:
            raise MealPlanError("stale_product", f"packs_override names product {pid}, which "
                                "is not in the catalog", product_id=pid)
        if packs < 0:
            raise MealPlanError("invalid_dates", f"packs_override {raw!r} is negative")
        out[(day, pid)] = packs
    return out


def _storage_overrides(draft: MealPlanDraft, products: dict[int, Product]) -> dict[int, str]:
    out = {}
    for raw, storage in draft.storage_overrides.items():
        if not raw.isdigit() or int(raw) not in products:
            raise MealPlanError("stale_product", f"storage_overrides names product {raw}, "
                                "which is not in the catalog")
        out[int(raw)] = storage
    return out


# ─── Facts ───────────────────────────────────────────────────

def _location(draft: MealPlanDraft) -> tuple[float, float, float | None]:
    cfg = settings()
    s = draft.settings
    return (cfg.default_lat if s.lat is None else s.lat,
            cfg.default_lon if s.lon is None else s.lon, s.max_km)


def _offers(product_ids: list[int], lat: float, lon: float,
            max_km: float | None) -> list[dict]:
    """Every in-range (store, product) offer for these products: one query."""
    sql, params = build_price_matrix_sql(sorted(set(product_ids)), lat, lon, max_km)
    with Session(db.engine()) as s:
        return [dict(r) for r in s.execute(text(sql), params).mappings()]


def _line_product(p: Product) -> LineProduct:
    return LineProduct(id=p.id, name=p.name, unit_size=p.unit_size, category=p.category,
                       demo_product=p.id in demo_product_ids())


def _shelf_info(product: Product, w: trips.Window, storage: str, buy_ahead: int,
                frozen_on_arrival: bool) -> ShelfLifeInfo:
    ps = shelf.for_product(product.id)
    if w.basis == "cited":
        ids = list(ps.freezer if storage == "freezer" else ps.fridge)
        if storage == "freezer":
            ids += list(ps.after_thaw) + list(w.thaw.rule_ids if w.thaw else ())
        rules = [shelf.rule(r) for r in ids]
        first = shelf.source(rules[0]["source"])
        if storage == "freezer":
            note = ("Frozen on arrival and thawed in the fridge for the meal; the chart "
                    "gives its freezer times for quality only." if frozen_on_arrival
                    else "Bought frozen and thawed in the fridge for the meal.")
            days = None
        else:
            days = ps.fridge_days
            note = f"Planned with the lower bound, {days} day(s)."
        if ps.note:
            note += " " + ps.note
        return ShelfLifeInfo(status="cited", verbatim=[r["verbatim"] for r in rules],
                             rule_ids=ids, source=shelf.short_source(first["id"]),
                             url=first["url"], page_date=first["page_date"],
                             days_planned=days, note=note)
    if w.basis == "your_setting":
        return ShelfLifeInfo(status="your_setting", days_planned=buy_ahead,
                             note=f"{ps.reason} Your setting buys it at most {buy_ahead} "
                                  "day(s) before it is eaten.")
    kind = "shelf-stable" if ps.storage_class == "shelf_stable" else "bought frozen"
    return ShelfLifeInfo(status="unknown",
                         note=f"{ps.reason} shelf_life.json classes it as {kind}, so the plan "
                              "sets no limit on when it is bought; no time is claimed.")


# ─── One strategy ────────────────────────────────────────────

class _Ctx:
    """Everything one schedule call shares between the two strategies."""

    def __init__(self, draft: MealPlanDraft, products: dict[int, Product],
                 needs: list[Need], board: Board, cook: dict[str, tuple[int, str, str]]):
        self.draft, self.products, self.needs, self.board, self.cook = (
            draft, products, needs, board, cook)
        self.prefs = draft.prefs
        self.packs_override = _packs_overrides(draft, products)
        self.storage_override = _storage_overrides(draft, products)
        self.fixed = ({draft.index(d) for d in draft.fixed_dates}
                      | {draft.index(t.date) for t in draft.trips})
        dismissed = {draft.index(d) for d in draft.dismissed_dates}
        self.allowed = sorted({i for i in range(draft.days)
                               if draft.day(i).weekday() in draft.prefs.shop_weekdays
                               and i not in dismissed} | self.fixed)
        self.lat, self.lon, self.max_km = _location(draft)
        ids = {n.product_id for n in needs} | {s.product_id for t in draft.trips
                                               for s in t.snapshot}
        self.rows = _offers(sorted(ids & products.keys()), self.lat, self.lon, self.max_km)
        self.stocked = {r["product_id"] for r in self.rows}

    def label(self, day: int) -> str:
        return day_label(self.draft.day(day))


def _reason(ctx: _Ctx, need: Need, w: trips.Window) -> str:
    if w.clo is None:
        return ("The last shopping day before the meal: no shopping day falls inside its "
                "window. " + _reason_body(ctx, need, w))
    return _reason_body(ctx, need, w)


def _reason_body(ctx: _Ctx, need: Need, w: trips.Window) -> str:
    p = ctx.products[need.product_id]
    meal = f"{need.title} on {ctx.label(need.day)}"
    ps = shelf.for_product(p.id)
    if w.basis == "cited" and w.mode in {"fridge", "union"}:
        r = ps.fridge_rules()[0]
        tail = (" or frozen on arrival and thawed" if w.mode == "union" else "")
        return (f"{meal} needs {p.name}, which keeps {r['verbatim']} in the fridge "
                f"({shelf.short_source(r['source'])}, {r['id']}){tail}.")
    if w.basis == "cited":
        lead = f"{w.thaw.days} day(s)" if w.thaw else "no time"
        return (f"{meal} needs {p.name}, kept frozen and thawed in the fridge {lead} ahead "
                f"({', '.join(w.thaw.rule_ids) if w.thaw else 'no thaw row'}).")
    if w.basis == "your_setting":
        return (f"{meal} needs {p.name}, which has no cited storage time; your setting buys "
                f"it at most {ctx.prefs.buy_ahead_days} day(s) ahead.")
    return f"First use of {p.name}: {meal}. No storage limit is planned for it."


def _move_remedy(ctx: _Ctx, need: Need, w: trips.Window, trip_days: list[int]) -> dict | None:
    """Move the meal to the nearest free day whose window, moved with it, holds a trip."""
    age = None
    if w.mode == "fridge":
        age = (shelf.for_product(need.product_id).fridge_days if w.basis == "cited"
               else ctx.prefs.buy_ahead_days)
    for d in place.probe(need.day, 0, ctx.draft.days - 1):
        if d == need.day or not ctx.board.free(d, need.slot):
            continue
        lo, hi = (0 if age is None else d - age), w.hi + (d - need.day)
        if any(lo <= t <= hi for t in trip_days):
            return warn.move_meal(need.meal_id, ctx.draft.day(d), need.slot)
    return None


def _strategy(ctx: _Ctx, name: str) -> tuple[StrategyResult, list[PlanWarning]]:
    draft, products, prefs = ctx.draft, ctx.products, ctx.prefs
    wins = [trips.window(n, products[n.product_id], strategy=name,
                         allow_freezer=prefs.allow_freezer, buy_ahead=prefs.buy_ahead_days,
                         override=ctx.storage_override.get(n.product_id), allowed=ctx.allowed)
            for n in ctx.needs]
    feasible = [i for i, w in enumerate(wins) if w.clo is not None]
    # The reason a trip names is its tightest cited window first, then the shopper's setting.
    rank = {"cited": 0, "your_setting": 1, "unknown": 2}
    trip_days, made = trips.choose_dates([(wins[i].clo, wins[i].chi) for i in feasible],
                                         ctx.fixed, [rank[wins[i].basis] for i in feasible])
    made = {day: feasible[i] for day, i in made.items()}
    made.update(trips.cover_late(wins, trip_days, ctx.allowed))
    asg = trips.assign(ctx.needs, wins, trip_days)
    warnings: list[PlanWarning] = []
    approvals = {t.date: t for t in draft.trips if t.strategy == name}

    # Purchases -> trip lines, per (day, product, storage).
    grouped: dict[tuple[int, int, str], list[trips.Run]] = {}
    for run in asg.runs:
        grouped.setdefault((run.day, run.product_id, run.storage), []).append(run)
    lines_by_day: dict[int, list[TripLine]] = {}
    dismissed_by_day: dict[int, list] = {}
    actions: list[Action] = []
    for (day, pid, storage), runs in sorted(grouped.items()):
        p = products[pid]
        run_needs = [n for r in runs for n in r.needs]
        packs_each = [r.packs(p) for r in runs]
        packs = None if any(x is None for x in packs_each) else sum(packs_each)
        unknown = [n.unknown for n in run_needs if n.unknown]
        basis = ("needs_servings" if "needs_servings" in unknown
                 else "amount_unknown" if packs is None else "computed")
        if (day, pid) in ctx.packs_override:
            packs, basis = ctx.packs_override[(day, pid)], "your_setting"
            if packs == 0:
                dismissed_by_day.setdefault(day, []).append(_line_product(p))
                continue
        need_qty = None if unknown or any(n.qty is None for n in run_needs) else \
            round(float(sum(n.qty for n in run_needs)), 3)
        uoms = {n.uom for n in run_needs}
        need_uom = uoms.pop() if need_qty is not None and len(uoms) == 1 else None
        if need_uom is None:
            need_qty = None
        w0 = runs[0].windows[0]
        frozen_on_arrival = storage == "freezer" and not shelf.for_product(pid).bought_frozen \
            and shelf.for_product(pid).storage_class != "frozen"
        leftover = (round(packs * p.unit_qty - need_qty, 3)
                    if packs is not None and need_qty is not None and p.unit_qty
                    and need_uom == p.unit_uom else None)
        info = _shelf_info(p, w0, storage, prefs.buy_ahead_days, frozen_on_arrival)
        until = (draft.day(day + info.days_planned)
                 if leftover is not None and storage == "fridge" and info.status == "cited"
                 and info.days_planned is not None else None)
        seen: dict[str, MealRef] = {}
        for n in sorted(run_needs, key=lambda n: (n.day, n.meal_id)):
            seen.setdefault(n.meal_id, MealRef(meal_id=n.meal_id, recipe_key=n.recipe_key,
                                               title=n.title, date=draft.day(n.day),
                                               slot=n.slot))
        lines_by_day.setdefault(day, []).append(TripLine(
            product=_line_product(p), category=p.category, packs=packs, packs_basis=basis,
            need_qty=need_qty, need_uom=need_uom, leftover_qty=leftover,
            leftover_until=until, storage=storage, freeze_on_arrival=frozen_on_arrival,
            shelf_life=info, for_meals=list(seen.values()), store=None, price=None,
            stocked=pid in ctx.stocked))
        trip_id = f"{name}-{draft.day(day).isoformat()}"
        if frozen_on_arrival:
            actions.append(Action(
                kind="freeze", date=draft.day(day), trip_id=trip_id, product_id=pid,
                text=f"Freeze {'?' if packs is None else packs} x {p.name} on arrival",
                rule_ids=list(info.rule_ids[:len(shelf.for_product(pid).freezer)]),
                basis="cited"))
        for r in runs:
            for n, w in zip(r.needs, r.windows, strict=True):
                t = trips.thaw_day(n, w)
                if storage != "freezer" or t is None:
                    continue
                actions.append(Action(
                    kind="thaw", date=draft.day(t), trip_id=trip_id, meal_id=n.meal_id,
                    product_id=pid,
                    text=(f"Move {p.name} to the fridge to thaw for {n.title} on "
                          f"{ctx.label(n.day)}: {w.thaw.verbatim}"),
                    rule_ids=list(w.thaw.rule_ids), basis="cited"))

    # Price each trip: tripopt over its stocked lines with known packs.
    out_trips: list[Trip] = []
    for day in trip_days:
        lines = lines_by_day.get(day, [])
        appr = approvals.get(draft.day(day))
        if not lines and appr is None and not dismissed_by_day.get(day):
            continue
        out_trips.append(_price_trip(ctx, name, day, lines, dismissed_by_day.get(day, []),
                                     appr, made, wins, warnings))

    # Needs that could not be bought inside their window.
    for need, w, day in asg.stale:
        _stale_warning(ctx, name, need, w, day, trip_days, warnings)
    for need, w in asg.unbought:
        p = products[need.product_id]
        rem = [warn.add_trip(draft.day(w.hi))] if w.hi >= 0 else []
        mv = _move_remedy(ctx, need, w, trip_days)
        warnings.append(warn.make(
            "must_fix", "meal_before_trip",
            f"{need.title} on {ctx.label(need.day)} needs {p.name}, and no shopping day comes "
            "before it.", strategy=name, remedies=rem + ([mv] if mv else []),
            meal_ids=[need.meal_id], product_id=p.id))
    if prefs.max_trips is not None and len(out_trips) > prefs.max_trips:
        rem = [warn.set_pref("max_trips", len(out_trips))]
        if name == "fresh":
            rem.insert(0, warn.set_strategy("fewest_trips"))
        warnings.append(warn.make(
            "decide", "trip_cap_exceeded",
            f"This plan needs {len(out_trips)} trips; your cap is {prefs.max_trips}.",
            strategy=name, remedies=rem))

    for t in out_trips:
        actions.append(Action(kind="shop", date=t.date, trip_id=t.id,
                              text=f"Shop for {len(t.lines)} item(s)"
                                   + (f" at {', '.join(t.stores)}" if t.stores else "")))
    for n_id, (meal_day, slot, title) in sorted(ctx.cook.items(),
                                                key=lambda kv: (kv[1][0], kv[0])):
        actions.append(Action(kind="cook", date=draft.day(meal_day), meal_id=n_id,
                              text=f"Cook {title} ({slot})"))
    order = {"shop": 0, "freeze": 1, "thaw": 2, "cook": 3}
    actions.sort(key=lambda a: (a.date, order[a.kind], a.trip_id or "", a.meal_id or "",
                                a.product_id or 0, a.text))
    known = [t.total_cost for t in out_trips if t.total_cost is not None]
    unpriced = any(t.lines for t in out_trips) and not any(
        ln.price is not None for t in out_trips for ln in t.lines)
    total = None if unpriced else round(sum(known), 2) + 0.0
    result = StrategyResult(
        name=name, recommended=False, trips=out_trips, actions=actions, total_cost=total,
        total_is_floor=any(t.total_is_floor for t in out_trips), warning_counts={})
    return result, warnings


def _price_trip(ctx: _Ctx, name: str, day: int, lines: list[TripLine], dismissed: list,
                appr: ApprovedTrip | None, made: dict, wins: list,
                warnings: list[PlanWarning]) -> Trip:
    draft, products = ctx.draft, ctx.products
    date = draft.day(day)
    lines.sort(key=lambda ln: (ln.product.name, ln.storage))
    # One basket item per product (fridge and freezer lines of one product are one item).
    combined: dict[int, int] = {}
    for ln in lines:
        if ln.stocked and ln.packs is not None:
            combined[ln.product.id] = combined.get(ln.product.id, 0) + ln.packs
    basket = [(pid, products[pid].name) for pid in sorted(combined)]
    rows = [r for r in ctx.rows if r["product_id"] in combined]
    options = tripopt.optimize_trips(rows, basket, home_lat=ctx.lat, home_lon=ctx.lon,
                                     cost_per_km=settings().travel_cost_per_km,
                                     packs=combined) if basket else []
    best = next((o for o in options if o.recommended), None)
    store_of = {it.product_id: it.store_name for it in best.items} if best else {}
    unit = {(r["store_name"], r["product_id"]): r["price"] for r in ctx.rows}
    for ln in lines:
        pid = ln.product.id
        if not ln.stocked:
            continue
        if ln.packs is not None and pid in store_of:
            ln.store = store_of[pid]
            ln.price = round(unit[(ln.store, pid)] * ln.packs, 2) + 0.0
            continue
        # Packs unknown: listed at its cheapest offer among the trip's stops (or in range),
        # with no price.
        offers = sorted((r["price"], r["store_name"]) for r in ctx.rows
                        if r["product_id"] == pid
                        and (best is None or r["store_name"] in best.stores))
        offers = offers or sorted((r["price"], r["store_name"]) for r in ctx.rows
                                  if r["product_id"] == pid)
        ln.store = offers[0][1] if offers else None
    stores = list(best.stores) if best else sorted({ln.store for ln in lines if ln.store})
    for s in sorted({ln.store for ln in lines if ln.store} - set(stores)):
        stores.append(s)
    # A trip whose lines all lack a price has an unknown total, not $0.00. A trip with no
    # lines at all costs nothing, so its 0.00 is a fact.
    prices = [ln.price for ln in lines if ln.price is not None]
    total = None if lines and not prices else round(sum(prices), 2) + 0.0
    floor = any(ln.price is None for ln in lines)
    fp = approved_mod.trip_fingerprint(date, lines)
    trip_id = f"{name}-{date.isoformat()}"
    status, diff, delta = "suggested", None, None
    if appr is not None:
        delta = approved_mod.apply_prices(appr, lines)
        gone = [ln for ln in lines if not ln.stocked]
        if fp == appr.fingerprint and not gone:
            status = "approved"
        else:
            status = "needs_review"
            diff = approved_mod.diff(appr, lines, {pid: p.name for pid, p in products.items()})
            warnings.append(warn.make(
                "decide", "needs_review",
                f"The trip on {day_label(date)} changed since you approved it"
                + (f": {', '.join(diff.text)}" if diff.text else "") + ".",
                strategy=name, trip_date=date,
                remedies=[warn.approve_trip(date, name)]
                + ([warn.open_options(date, gone[0].product.id)] if gone else [])))
        for ln in gone:
            warnings.append(warn.make(
                "must_fix", "no_longer_stocked",
                f"{ln.product.name} on your approved trip {day_label(date)} has no offer in "
                "range any more.", strategy=name, trip_date=date, product_id=ln.product.id,
                remedies=[warn.open_options(date, ln.product.id),
                          warn.set_packs(date, ln.product.id, 0)]))
        if delta:
            warnings.append(warn.make(
                "note", "price_changed",
                f"Price changed since you approved the trip on {day_label(date)}: "
                f"{approved_mod.money(delta)} (demo prices).", strategy=name, trip_date=date))
    else:
        for ln in lines:
            if not ln.stocked:
                warnings.append(warn.make(
                    "decide", "not_stocked",
                    f"No store in range sells {ln.product.name} (for "
                    f"{', '.join(sorted({m.title for m in ln.for_meals}))}).",
                    strategy=name, trip_date=date, product_id=ln.product.id,
                    remedies=[warn.open_options(date, ln.product.id),
                              warn.set_packs(date, ln.product.id, 0)]))
    for ln in lines:
        if ln.packs_basis == "amount_unknown":
            warnings.append(warn.make(
                "note", "amount_unknown",
                f"{ln.product.name} on {day_label(date)}: the amount needed is not known in "
                "its pack's unit, so the packs to buy are yours to set.", strategy=name,
                trip_date=date, product_id=ln.product.id,
                remedies=[warn.set_packs(date, ln.product.id)]))
    if appr is not None:
        reason = "You approved this trip."
    elif day in made:
        reason = _reason(ctx, ctx.needs[made[day]], wins[made[day]])
    else:
        reason = "Your fixed shopping date."
    return Trip(id=trip_id, date=date, status=status, reason=reason, lines=lines,
                dismissed=sorted(dismissed, key=lambda p: p.name), stores=stores,
                recommended=best, frontier=options,
                not_stocked=sorted(ln.product.name for ln in lines if not ln.stocked),
                total_cost=total, total_is_floor=floor, price_delta=delta, fingerprint=fp,
                diff=diff,
                list_text=lists.list_text(date, name, stores, lines, total, floor))


def _stale_warning(ctx: _Ctx, name: str, need: Need, w: trips.Window, day: int,
                   trip_days: list[int], warnings: list[PlanWarning]) -> None:
    p = ctx.products[need.product_id]
    ps = shelf.for_product(p.id)
    rem = []
    if (w.basis == "cited" and ps.freezable and ps.thaw
            and ctx.storage_override.get(p.id) != "freezer"):
        rem.append(warn.set_storage(p.id, "freezer"))
    rem.append(warn.add_trip(ctx.draft.day(min(w.hi, ctx.draft.days - 1))))
    mv = _move_remedy(ctx, need, w, trip_days)
    if mv:
        rem.append(mv)
    bought = ctx.label(day)
    # Only a fridge window can be missed: freezer and unlimited windows open at the start.
    if w.basis == "cited" and ps.fridge_rules():
        r = ps.fridge_rules()[0]
        warnings.append(warn.make(
            "must_fix", "fridge_window_exceeded",
            f"{need.title} on {ctx.label(need.day)}: {p.name} bought {bought} is past its "
            f"cited fridge time, {r['verbatim']} ({shelf.short_source(r['source'])}, "
            f"{r['id']}).", strategy=name, remedies=rem, meal_ids=[need.meal_id],
            product_id=p.id))
    else:
        rem = rem[-2:] + [warn.set_pref("buy_ahead_days", need.day - day)]
        warnings.append(warn.make(
            "decide", "buy_ahead_exceeded",
            f"{need.title} on {ctx.label(need.day)}: {p.name} bought {bought} is "
            f"{need.day - day} day(s) ahead, more than your setting of "
            f"{ctx.prefs.buy_ahead_days}.", strategy=name, remedies=rem,
            meal_ids=[need.meal_id], product_id=p.id))


# ─── The schedule ────────────────────────────────────────────

def compute(draft: MealPlanDraft) -> MealSchedule:
    """The schedule for a draft. Raises MealPlanError (REST 422) for a draft it cannot be
    computed for."""
    if draft.v != 1:
        raise MealPlanError("version", f"draft version {draft.v}; this server reads 1")
    _check_dates(draft)
    products = {p.id: p for p in db.load_all_products()}
    docs = {key: recipe_doc(dr.ref, dr.servings) for key, dr in sorted(draft.recipes.items())}
    chosen = {key: line_products(draft, key, docs[key], products) for key in docs}
    meals, days = place.place(draft)

    board = Board(days=draft.days, used={})
    placed: list[PlacedMeal] = []
    needs: list[Need] = []
    cook: dict[str, tuple[int, str, str]] = {}
    for m in sorted(meals, key=lambda m: (days[m.id] is None, days[m.id] or 0,
                                          SLOTS.index(slot_of(draft, m)),
                                          place.natural(m.id))):
        slot, day, doc = slot_of(draft, m), days[m.id], docs[m.recipe_key]
        placed.append(PlacedMeal(
            id=m.id, recipe_key=m.recipe_key, title=doc.title,
            date=None if day is None else draft.day(day), slot=slot,
            servings=meal_servings(draft, m), pinned=m.pinned,
            placed_by=None if day is None else ("you" if m.date is not None else "spread")))
        if day is None:
            continue
        board.take(day, slot)
        cook[m.id] = (day, slot, doc.title)
        resolved = draft.resolved.get(m.recipe_key)
        if resolved is not None and resolved.status in USABLE:
            needs.extend(needs_for_meal(draft, m, day, slot, doc, chosen[m.recipe_key]))
    needs.sort(key=lambda n: (n.day, SLOTS.index(n.slot), n.meal_id, n.line_no))

    ctx = _Ctx(draft, products, needs, board, cook)
    warnings = _plan_warnings(draft, placed, docs, needs)
    results = []
    for name in STRATEGIES:
        res, ws = _strategy(ctx, name)
        results.append(res)
        warnings.extend(ws)
    warnings.sort(key=warn.sort_key)
    for res in results:
        res.warning_counts = warn.counts(warnings, res.name)
    fresh, fewest = results
    rec = ("fewest_trips"
           if fresh.warning_counts["must_fix"] > fewest.warning_counts["must_fix"]
           else "fresh")
    for res in results:
        res.recommended = res.name == rec

    return MealSchedule(
        rev=draft.rev, start_date=draft.start_date, meals=placed,
        unplaced=[m.id for m in placed if m.date is None],
        strategies=results, recommended_strategy=rec, warnings=warnings,
        days=[DayOut(date=draft.day(i), weekday=SHORT[draft.day(i).weekday()],
                     meal_ids=[m.id for m in placed if m.date == draft.day(i)])
              for i in range(draft.days)],
        coverage=_coverage(needs),
        approved_schedule=_approved_schedule(draft, results, cook),
        sources=_sources(), synthetic_notice=SYNTHETIC_NOTICE)


def _plan_warnings(draft: MealPlanDraft, placed: list[PlacedMeal],
                   docs: dict[str, RecipeDoc], needs: list[Need]) -> list[PlanWarning]:
    out: list[PlanWarning] = []
    for m in placed:
        if m.date is None and not m.pinned:
            out.append(warn.make(
                "must_fix", "unplaced",
                f"{m.title} ({m.slot}) has no free {m.slot} slot in the plan's "
                f"{draft.days} days.", meal_ids=[m.id],
                remedies=[warn.set_pref("days", 14)] if draft.days < 14 else
                [warn.remove_meal(m.id)]))
    with_meals = sorted({m.recipe_key for m in placed if m.date is not None})
    for key in with_meals:
        resolved = draft.resolved.get(key)
        doc = docs[key]
        if resolved is None or resolved.status not in USABLE:
            out.append(warn.make(
                "decide", "unresolved_recipe",
                f"{doc.title} has not been matched to products"
                + (f" ({resolved.status}: {resolved.message})" if resolved else "")
                + ", so its meals buy nothing yet.", recipe_key=key,
                remedies=[warn.resolve(key)]))
        elif _recipe_servings(draft, key, doc) is None:
            out.append(warn.make(
                "must_fix", "needs_servings",
                f"How many does {doc.title} serve? It does not say, so its amounts, packs and "
                "prices stay unknown until you answer.", recipe_key=key,
                remedies=[warn.set_servings(key)]))
    unknown = sorted({n.product_id for n in needs
                      if not shelf.for_product(n.product_id).mapped})
    if unknown:
        setting = sorted({p for p in unknown
                          if shelf.for_product(p).storage_class == "chilled_or_fresh"})
        out.append(warn.make(
            "note", "shelf_life_unknown",
            f"{len(unknown)} product(s) have no cited storage time: {len(setting)} chilled or "
            f"fresh ones follow your buy-ahead setting ({draft.prefs.buy_ahead_days} days), "
            "and shelf-stable or frozen ones are not limited."))
    return out


def _coverage(needs: list[Need]) -> Coverage:
    pids = sorted({n.product_id for n in needs})
    cited = [p for p in pids if shelf.for_product(p).mapped]
    setting = [p for p in pids if not shelf.for_product(p).mapped
               and shelf.for_product(p).storage_class == "chilled_or_fresh"]
    return Coverage(products=len(pids), freshness_cited=len(cited),
                    freshness_your_setting=len(setting),
                    freshness_unknown=len(pids) - len(cited) - len(setting),
                    needs=len(needs), amounts_known=sum(1 for n in needs if n.qty is not None))


def _approved_schedule(draft: MealPlanDraft, results: list[StrategyResult],
                       cook: dict) -> dict | None:
    """The approved trips as the calendar export reads them, with their freeze, thaw and
    cook actions. exportable is False while any of them needs review."""
    if not draft.trips:
        return None
    by_name = {r.name: r for r in results}
    trips_out, actions = [], []
    for appr in sorted(draft.trips, key=lambda t: (t.date, t.strategy)):
        res = by_name[appr.strategy]
        trip = next((t for t in res.trips if t.date == appr.date), None)
        if trip is None:
            continue
        trips_out.append({"id": trip.id, "strategy": res.name, "date": trip.date.isoformat(),
                          "status": trip.status, "stores": trip.stores,
                          "total_cost": trip.total_cost, "total_is_floor": trip.total_is_floor,
                          "list_text": trip.list_text})
        meals = {m.meal_id for ln in trip.lines for m in ln.for_meals}
        actions += [a.model_dump(mode="json") for a in res.actions
                    if a.trip_id == trip.id or (a.kind == "cook" and a.meal_id in meals)]
    return {"trips": trips_out, "actions": actions,
            "exportable": bool(trips_out) and all(t["status"] == "approved" for t in trips_out)}


def _sources() -> list[dict]:
    out = [{"id": s["id"], "title": s["title"], "publisher": s["publisher"], "url": s["url"],
            "page_date": s["page_date"], "retrieved": s["retrieved"], "credit": s["credit"]}
           for s in shelf.sources()]
    out.append({"id": "demo-data", "title": "Demo data", "publisher": "this demo",
                "url": None, "page_date": None, "retrieved": None,
                "credit": "Store prices and stock, the demo starter recipes, the library's "
                          "house amounts and products 166-169 are synthetic demo data."})
    return out

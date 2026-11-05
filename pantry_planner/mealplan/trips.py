"""Trips: the fewest shopping dates that keep every need inside its window, and what each
trip buys.

Day semantics. Day indexes count from the plan's start (day 0). A purchase on day P with a
cited fridge time of N days (the row's lower bound) may be eaten up to day P + N, so a need
eaten on day d may be bought on days [d - N, d]. A breakfast is eaten before any shop that
day, so it must be bought by d - 1. Windows by what the product's storage rests on:

- cited fridge time: [d - N, d];
- bought frozen, with a cited thaw time: [start, d - thaw lead] (kept frozen, then thawed);
- no cited time, chilled or fresh: [d - buy_ahead_days, d], the shopper's own setting;
- no cited time, shelf-stable or bought frozen by the file's class: [start, d], no limit;
- fewest_trips (and a freezer the shopper allows), for a product whose cited freezer time is
  longer than the plan and which has a cited thaw time: [start, d], bought fresh when a trip
  falls in its fridge window and frozen on arrival otherwise, thawed d - lead.

Windows are clipped to the days the shopper shops (shop_weekdays, less dismissed dates,
plus fixed and approved dates). Choosing the dates is minimum interval stabbing: windows by
right end, and a window no chosen date falls in gets one on its latest allowed day. That
takes the fewest trips (tests check it against brute force) and buys as late as possible.

Assignment, per product in date order: a need joins the product's current purchase when that
trip lies in its window, and its packs are recomputed for the summed need (packs.pack_count,
one line enough). A need that is frozen and thawed for its meal always has a purchase of its
own: the rest of a thawed pack is never planned for later. A need with no allowed day in its
window is bought on the latest trip before it and flagged; with no trip before it at all, it
is not bought and flagged.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..models import Product
from ..packs import pack_count
from . import shelf
from .needs import Need


@dataclass(frozen=True)
class Window:
    """Where a need may be bought: raw [lo, hi] in day indexes, and [clo, chi] its first and
    last allowed shopping days (None: no allowed day in it). mode: fridge, freezer, pantry or
    union (fewest_trips: fridge from fridge_lo on, freezer before). basis: cited,
    your_setting or unknown. thaw_each: a purchase of its own, thawed for this meal."""
    lo: int
    hi: int
    clo: int | None
    chi: int | None
    mode: str
    basis: str
    fridge_lo: int | None = None
    thaw: shelf.ThawLead | None = None
    thaw_each: bool = False


def _kg(product: Product, need: Need) -> float | None:
    """Weight of the packs one need takes, for the thaw time; None when unknown."""
    if product.unit_uom != "g" or not product.unit_qty:
        return None
    packs = pack_count(product.unit_qty, product.unit_uom,
                       [need.as_pack_need], min_lines=1)
    return None if packs is None else packs * product.unit_qty / 1000


def window(need: Need, product: Product, *, strategy: str, allow_freezer: bool,
           buy_ahead: int, override: str | None, allowed: list[int]) -> Window:
    ps = shelf.for_product(need.product_id)
    d = need.day
    buyby = d - 1 if need.slot == "breakfast" else d

    def clip(lo: int, hi: int, mode: str, basis: str, **kw) -> Window:
        lo = max(lo, 0)
        inside = [a for a in allowed if lo <= a <= hi]
        return Window(lo=lo, hi=hi, clo=inside[0] if inside else None,
                      chi=inside[-1] if inside else None, mode=mode, basis=basis, **kw)

    if ps.bought_frozen:
        lead = shelf.thaw_lead(ps, _kg(product, need))
        days = lead.days if lead else 0
        return clip(0, min(buyby, d - days), "freezer", "cited", thaw=lead,
                    thaw_each=lead is not None)
    if ps.mapped and ps.fridge_days is not None:
        f = ps.fridge_days
        lead = shelf.thaw_lead(ps, _kg(product, need)) if ps.freezable else None
        can_freeze = lead is not None and override != "fridge"
        if can_freeze and override == "freezer":
            return clip(0, min(buyby, d - lead.days), "freezer", "cited", thaw=lead,
                        thaw_each=True)
        if can_freeze and strategy == "fewest_trips" and allow_freezer:
            freezer_hi = min(buyby, d - lead.days)
            if d - f <= freezer_hi + 1:
                return clip(0, buyby, "union", "cited", fridge_lo=d - f, thaw=lead,
                            thaw_each=True)
            return clip(0, freezer_hi, "freezer", "cited", thaw=lead, thaw_each=True)
        return clip(d - f, buyby, "fridge", "cited")
    if ps.storage_class == "shelf_stable":
        return clip(0, buyby, "pantry", "unknown")
    if ps.storage_class == "frozen":
        return clip(0, buyby, "freezer", "unknown")
    return clip(d - buy_ahead, buyby, "fridge", "your_setting")


def choose_dates(windows: list[tuple[int, int]], fixed: set[int],
                 rank: list[int] | None = None) -> tuple[list[int], dict]:
    """Minimum interval stabbing. `windows` are (first allowed day, last allowed day);
    `fixed` dates are trips already. Returns (sorted trip days, day -> index of the window
    that made it a trip). Windows ending on the same day all get that day, so their order
    changes nothing but which one is named as the reason: `rank` (lower first), then the
    narrowest."""
    chosen = set(fixed)
    made: dict[int, int] = {}
    rank = rank or [0] * len(windows)
    order = sorted(range(len(windows)),
                   key=lambda i: (windows[i][1], rank[i], -windows[i][0], i))
    for i in order:
        lo, hi = windows[i]
        if any(lo <= t <= hi for t in chosen):
            continue
        chosen.add(hi)
        made[hi] = i
    return sorted(chosen), made


def cover_late(windows: list[Window], trips: list[int], allowed: list[int]) -> dict[int, int]:
    """Trips for needs with no allowed day inside their window: such a need is bought on the
    latest trip before it (and flagged), and when there is none yet, on the latest allowed
    day before it, which becomes a trip. Returns the days added -> the window's index;
    `trips` is extended in place and kept sorted."""
    added: dict[int, int] = {}
    order = sorted((i for i, w in enumerate(windows) if w.clo is None),
                   key=lambda i: (windows[i].hi, i))
    for i in order:
        hi = windows[i].hi
        if any(t <= hi for t in trips):
            continue
        before = [a for a in allowed if a <= hi]
        if before:
            trips.append(before[-1])
            trips.sort()
            added[before[-1]] = i
    return added


@dataclass
class Run:
    """One purchase of one product on one trip, for one or more needs."""
    product_id: int
    day: int
    storage: str
    joinable: bool
    needs: list[Need] = field(default_factory=list)
    windows: list[Window] = field(default_factory=list)

    def packs(self, product: Product) -> int | None:
        return pack_count(product.unit_qty, product.unit_uom,
                          [n.as_pack_need for n in self.needs], min_lines=1)


@dataclass
class Assignment:
    runs: list[Run]
    unbought: list[tuple[Need, Window]]
    stale: list[tuple[Need, Window, int]]       # need, its window, the trip it is bought on


def assign(needs: list[Need], wins: list[Window], trips: list[int]) -> Assignment:
    """Purchases per product in date order (see the module docstring)."""
    runs: list[Run] = []
    current: dict[tuple[int, str], Run] = {}
    unbought: list[tuple[Need, Window]] = []
    stale: list[tuple[Need, Window, int]] = []
    order = sorted(range(len(needs)),
                   key=lambda i: (needs[i].product_id, needs[i].day, needs[i].meal_id,
                                  needs[i].line_no))
    for i in order:
        need, w = needs[i], wins[i]
        if w.clo is None:
            before = [t for t in trips if t <= w.hi]
            if not before:
                unbought.append((need, w))
                continue
            day = before[-1]
            storage = "fridge" if w.mode == "union" else w.mode
            stale.append((need, w, day))
        else:
            run = current.get((need.product_id, _store_kind(w)))
            if run is not None and run.joinable and w.clo <= run.day <= w.chi:
                kept = _storage(w, run.day)
                if kept == run.storage and not (w.thaw_each and kept == "freezer"):
                    run.needs.append(need)
                    run.windows.append(w)
                    continue
            day = max(t for t in trips if w.clo <= t <= w.chi)
            storage = _storage(w, day)
        run = Run(product_id=need.product_id, day=day, storage=storage,
                  joinable=not (w.thaw_each and storage == "freezer"), needs=[need], windows=[w])
        runs.append(run)
        if run.joinable:
            current[(need.product_id, _store_kind(w))] = run
    runs.sort(key=lambda r: (r.day, r.product_id, r.storage, r.needs[0].day,
                             r.needs[0].meal_id))
    return Assignment(runs=runs, unbought=unbought, stale=stale)


def _store_kind(w: Window) -> str:
    return "cold" if w.mode in {"fridge", "union"} else w.mode


def _storage(w: Window, day: int) -> str:
    if w.mode == "union":
        return "fridge" if day >= (w.fridge_lo or 0) else "freezer"
    return w.mode


def thaw_day(need: Need, w: Window) -> int | None:
    """The day a frozen pack moves to the fridge: its thaw lead before the meal."""
    if w.thaw is None:
        return None
    return need.day - w.thaw.days

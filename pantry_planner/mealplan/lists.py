"""The shopping list for one trip, as plain text built by code.

Grouped by store in the order of the trip's recommended stops, then by aisle (the product's
catalog category), each line with packs, size, need and price. Lines with no store in range
come last. The total is "at least" when a line has no price, and "unknown" when no line has
one. The footer says the prices and stock are demo data. The same text is what Copy
list and Print list give, and what a calendar export puts in the trip's description.
"""
from __future__ import annotations

import datetime as dt

from .models import TripLine
from .needs import fmt_qty
from .place import day_label

FOOTER = "Prices and stock are demo data."
NOT_STOCKED = "Not stocked within range"


def _aisle(category: str | None) -> str:
    return (category or "other").replace("_", " ").capitalize()


def _line(ln: TripLine) -> str:
    packs = "? x" if ln.packs is None else f"{ln.packs} x"
    name = ln.product.name + (" (demo product)" if ln.product.demo_product else "")
    size = (f" ({ln.product.unit_size})"
            if ln.product.unit_size and ln.product.unit_size not in ln.product.name else "")
    need = ("amount unknown: the recipe does not say how many it serves"
            if ln.packs_basis == "needs_servings"
            else "amount unknown" if ln.need_qty is None
            else f"need {fmt_qty(ln.need_qty, ln.need_uom)}")
    price = "price unknown" if ln.price is None else f"${ln.price:.2f}"
    extra = ""
    if ln.freeze_on_arrival:
        extra = "; freeze on arrival"
    if ln.packs_basis == "your_setting":
        extra += "; packs set by you"
    return f"  - {packs} {name}{size}: {need}; {price}{extra}"


def list_text(date: dt.date, strategy: str, stores: list[str], lines: list[TripLine],
              total: float | None, floor: bool) -> str:
    out = [f"Shopping trip {day_label(date)} {date.year} ({strategy.replace('_', ' ')})"]
    order = {s: i for i, s in enumerate(stores)}
    groups: dict[str, list[TripLine]] = {}
    for ln in lines:
        groups.setdefault(ln.store or NOT_STOCKED, []).append(ln)
    for store in sorted(groups, key=lambda s: (s == NOT_STOCKED, order.get(s, len(order)), s)):
        out.append("")
        out.append(store)
        by_aisle: dict[str, list[TripLine]] = {}
        for ln in groups[store]:
            by_aisle.setdefault(_aisle(ln.category), []).append(ln)
        for aisle in sorted(by_aisle):
            out.append(f" {aisle}")
            out.extend(_line(ln) for ln in sorted(by_aisle[aisle],
                                                  key=lambda x: (x.product.name, x.storage)))
    out.append("")
    out.append("Total unknown (no line has a price yet)" if total is None
               else f"Total {'at least ' if floor else ''}${total:.2f}")
    out.append(FOOTER)
    return "\n".join(out) + "\n"

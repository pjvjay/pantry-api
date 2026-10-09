"""How many packs a purchase takes: one rule for every planner.

A chat plan (flow._packs), the meal plan and the alternatives ranking all ask
the same question: given a pack size and the needs of the recipe lines that
buy it, how many packs? They differ only in what they do when the answer is
not known. A chat plan buys one pack and says nothing, which is what it has
always done; the meal plan and the alternatives keep the None so they can say
"amount unknown" instead of inventing a count.
"""
from __future__ import annotations

import math


def pack_count(unit_qty: float | None, unit_uom: str,
               needs: list[tuple[float, str] | None], *, min_lines: int) -> int | None:
    """The summed need over the pack size, rounded up; None when unknown.

    Unknown means: fewer than `min_lines` needs, no pack size, any need
    that is unknown or in another unit than the pack (a teaspoon of
    peppercorns against a 50 g bag says nothing about bags), or a count
    that is not a finite number (an inf or NaN need, or two needs near
    1e308 whose sum overflows). A known need always takes at least one pack.

    `min_lines` is the caller's choice of when one line is enough. The chat
    plan passes 2: a single line has always bought one pack there, and
    changing that would change every existing plan. The meal plan passes 1,
    so one 900 g need against a 450 g pack buys 2."""
    if (len(needs) < min_lines or not unit_qty
            or any(n is None or n[1] != unit_uom for n in needs)):
        return None
    count = sum(n[0] for n in needs if n is not None) / unit_qty
    if not math.isfinite(count):
        return None
    return max(1, math.ceil(count - 1e-9))

"""packs.pack_count: one pack rule for chat plans, the meal plan and alternatives.

The chat plan's flow._packs now calls pack_count(min_lines=2) and falls back
to one pack. Its expected values come from the rule as it was written in
flow.py before the lift (copied below as `_packs_before`), checked over a grid
of needs, so the lift cannot change a single existing plan.
"""
from __future__ import annotations

import itertools
import math

import pytest

from pantry_planner.models import Product
from pantry_planner.packs import pack_count


def _packs_before(product: Product, needs) -> int:
    """flow._packs as it was before pack_count existed (feat/origin-alternatives)."""
    if (len(needs) < 2 or not product.unit_qty
            or any(n is None or n[1] != product.unit_uom for n in needs)):
        return 1
    total = sum(n[0] for n in needs if n is not None)
    return max(1, math.ceil(total / product.unit_qty - 1e-9))


def _product(unit_qty, unit_uom="g") -> Product:
    return Product(id=1, name="Penne 450g", description="", price=2.0,
                   unit_qty=unit_qty, unit_uom=unit_uom)


def test_known_needs_round_up_to_whole_packs():
    assert pack_count(450, "g", [(900, "g")], min_lines=1) == 2
    assert pack_count(450, "g", [(901, "g")], min_lines=1) == 3
    assert pack_count(450, "g", [(200, "g"), (250, "g")], min_lines=1) == 1
    assert pack_count(500, "g", [(300, "g"), (300, "g")], min_lines=2) == 2
    # float noise never buys an extra pack: 3 x 0.1 L is 0.30000000000000004
    assert pack_count(0.3, "ml", [(0.1, "ml")] * 3, min_lines=1) == 1


def test_a_known_need_takes_at_least_one_pack():
    assert pack_count(450, "g", [(0, "g")], min_lines=1) == 1
    assert pack_count(450, "g", [(5, "g")], min_lines=1) == 1


@pytest.mark.parametrize("unit_qty, unit_uom, needs", [
    (450, "g", [None]),                         # the line's amount is not stated
    (450, "g", [(900, "g"), None]),             # one of two lines unknown
    (450, "g", [(2, "each")]),                  # a need in another unit
    (450, "g", [(500, "ml")]),                  # volume against a weight pack
    (None, "g", [(900, "g")]),                  # no pack size
    (0, "g", [(900, "g")]),                     # a zero pack size is no pack size
    (450, "g", []),                             # nothing to buy for
    (450, "g", [(math.inf, "g")]),              # not a number of grams anyone can buy
    (450, "g", [(math.nan, "g")]),
    (450, "g", [(1e308, "g"), (1e308, "g")]),   # each finite, the sum past a float
])
def test_unknown_stays_unknown(unit_qty, unit_uom, needs):
    assert pack_count(unit_qty, unit_uom, needs, min_lines=1) is None


def test_a_need_sum_past_a_float_is_unknown():
    """flow._need, the need_qty on a purchase, follows pack_count: two needs that
    are each finite can still sum to inf, which is no amount."""
    from pantry_planner.flow import _need
    from pantry_planner.models import Selection

    sels = [Selection(line_no=n, product_id=1, confidence=1.0) for n in (1, 2)]
    assert _need({1: (1e308, "g"), 2: (1e308, "g")}, sels) == \
        {"need_qty": None, "need_uom": None}
    assert _need({1: (math.nan, "g"), 2: (1.0, "g")}, sels) == \
        {"need_qty": None, "need_uom": None}
    assert _need({1: (200.0, "g"), 2: (250.0, "g")}, sels) == \
        {"need_qty": 450.0, "need_uom": "g"}


def test_min_lines_is_the_callers_choice():
    one = [(900, "g")]
    assert pack_count(450, "g", one, min_lines=1) == 2
    assert pack_count(450, "g", one, min_lines=2) is None
    assert pack_count(450, "g", one * 2, min_lines=2) == 4


NEEDS = [None, (100, "g"), (450, "g"), (451, "g"), (1000, "g"), (2, "each"), (250, "ml")]


@pytest.mark.parametrize("unit_qty, unit_uom", [
    (450, "g"), (1000, "g"), (None, "g"), (0, ""), (6, "each"), (1000, "ml")])
def test_flow_packs_is_unchanged(unit_qty, unit_uom):
    from pantry_planner.flow import _packs

    product = _product(unit_qty, unit_uom)
    for n in range(0, 4):
        for needs in itertools.product(NEEDS, repeat=n):
            assert _packs(product, list(needs)) == _packs_before(product, list(needs)), needs

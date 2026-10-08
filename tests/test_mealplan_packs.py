"""Packs on meal-plan trips: one pack_count rule (one line is enough), servings scaling with
Fractions, unknown amounts kept unknown, the shopper's own pack count.

Pack sizes come from seeds/products.json; amounts from the hand-written docs below.
"""
from __future__ import annotations

import math

import pytest

from tests.mealplan_fixtures import (
    PRODUCTS,
    day,
    doc_draft,
    doc_recipe,
    done_db,
    lines_of,
    schedule,
    use_db,
)


@pytest.fixture(scope="module", autouse=True)
def seeded_db(tmp_path_factory):
    use_db(tmp_path_factory, "packs")
    yield
    done_db()


def _one_meal(entry: tuple[dict, dict], key: str, *, servings: int | None = None,
              **extra) -> dict:
    meal = {"id": "m", "recipe_key": key, "date": day(3).isoformat()}
    if servings is not None:
        meal["servings"] = servings
    return doc_draft({key: entry}, meals=[meal], **extra)


def test_one_need_over_one_pack_buys_enough_packs():
    from pantry_planner.packs import pack_count

    size = PRODUCTS[46]["unit_qty"]                 # Chicken Thighs Boneless 450g
    assert pack_count(size, "g", [(1200, "g")], min_lines=1) == math.ceil(1200 / size) == 3
    entry = doc_recipe("my:t", "Tray Bake", 2, [("Chicken Thighs", 1200, "g", 46)])
    ((_trip, line),) = lines_of(schedule(_one_meal(entry, "my:t")), "fresh", 46)
    assert (line["packs"], line["packs_basis"], line["need_qty"]) == (3, "computed", 1200.0)
    assert line["leftover_qty"] == 3 * size - 1200


def test_chat_plans_still_buy_one_pack_for_a_single_line():
    from pantry_planner.flow import _packs
    from pantry_planner.models import Product

    thighs = Product(id=46, name="t", description="", price=1.0, unit_qty=450, unit_uom="g")
    assert _packs(thighs, [(1200, "g")]) == 1          # min_lines=2: unchanged (D7)
    assert _packs(thighs, [(600, "g"), (600, "g")]) == 3


def test_a_recipe_for_4_cooked_for_2_halves_every_need():
    entry = doc_recipe("my:c", "Curry", 4, [("Chicken Thighs", 900, "g", 46),
                                            ("Basmati Rice", 300, "g", 8)])
    sched = schedule(_one_meal(entry, "my:c"))
    ((_t, thighs),) = lines_of(sched, "fresh", 46)
    ((_t, rice),) = lines_of(sched, "fresh", 8)
    assert (thighs["need_qty"], rice["need_qty"]) == (450.0, 150.0)
    assert thighs["packs"] == 1
    sched = schedule(_one_meal(entry, "my:c", servings=3))
    ((_t, thighs),) = lines_of(sched, "fresh", 46)
    assert thighs["need_qty"] == 900 * 3 / 4


def test_thirds_add_up_exactly():
    # A recipe for 3 eaten by 2, three times: 3 x (2/3) = exactly 2 batches.
    entry = doc_recipe("my:r", "Rice Bowl", 3, [("Basmati Rice", 100, "g", 8)])
    meals = [{"id": f"m{i}", "recipe_key": "my:r", "date": day(i).isoformat()}
             for i in (1, 2, 3)]
    sched = schedule(doc_draft({"my:r": entry}, meals=meals))
    ((_t, rice),) = lines_of(sched, "fresh", 8)
    assert rice["need_qty"] == 200.0


@pytest.mark.parametrize("quantity,unit", [(2, "cloves"), (30, "ml"), (None, "")])
def test_an_unknown_or_mismatched_amount_buys_an_unknown_number_of_packs(quantity, unit):
    # Fresh Garlic is sold by weight: cloves, millilitres or no amount say nothing about it.
    entry = doc_recipe("my:g", "Garlic Bread", 2, [("Garlic", quantity, unit, 13)])
    sched = schedule(_one_meal(entry, "my:g"))
    ((trip, line),) = lines_of(sched, "fresh", 13)
    assert (line["packs"], line["packs_basis"], line["price"]) == (None, "amount_unknown", None)
    # A need in another unit than the pack is still shown as it is; it just says nothing
    # about packs.
    assert (line["need_qty"], line["need_uom"]) == ((30.0, "ml") if unit == "ml"
                                                    else (None, None))
    assert line["leftover_qty"] is None
    assert trip["total_is_floor"] is True
    notes = [w for w in sched["warnings"] if w["code"] == "amount_unknown"
             and w["strategy"] == "fresh"]
    assert notes and notes[0]["level"] == "note"
    assert notes[0]["remedies"][0]["op"] == "set_packs"
    assert "? x Fresh Garlic" in trip["list_text"]


def test_the_shoppers_pack_count_is_used_and_labelled():
    entry = doc_recipe("my:g", "Garlic Bread", 2, [("Garlic", 2, "cloves", 13)])
    draft = _one_meal(entry, "my:g")
    sched = schedule(draft)
    ((trip, _line),) = lines_of(sched, "fresh", 13)
    key = f"{trip['date']}:13"
    sched = schedule({**draft, "packs_override": {key: 2}})
    ((trip, line),) = lines_of(sched, "fresh", 13)
    assert (line["packs"], line["packs_basis"]) == (2, "your_setting")
    assert line["price"] is not None and trip["total_is_floor"] is False
    assert "packs set by you" in trip["list_text"]
    # 0 dismisses the line.
    sched = schedule({**draft, "packs_override": {key: 0}})
    assert lines_of(sched, "fresh", 13) == []

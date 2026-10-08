"""Ranked warnings and the edits that fix them.

Three levels: must_fix (the plan as it stands would buy something too early, before a trip,
or not at all), decide (a choice for the shopper: a trip to re-approve, a product out of
range, a trip cap) and note (unknowns, rolled up). Each warning carries at most three
remedies, edits the console can apply to the draft. The engine computes them and never
applies them.

Remedy ops: move_meal {meal_id, date, slot}, set_storage {product_id, storage}, add_trip
{date}, set_servings {recipe_key}, set_packs {date, product_id, packs?}, open_options
{date, product_id}, resolve {recipe_key}, set_strategy {strategy}, set_pref {field, value},
approve_trip {date, strategy}, remove_meal {meal_id}.
"""
from __future__ import annotations

import datetime as dt

from .models import PlanWarning

LEVELS = ("must_fix", "decide", "note")


def make(level: str, code: str, message: str, *, strategy: str | None = None,
         remedies: list[dict] | None = None, meal_ids: list[str] | None = None,
         product_id: int | None = None, trip_date: dt.date | None = None,
         recipe_key: str | None = None) -> PlanWarning:
    return PlanWarning(level=level, code=code, message=message, strategy=strategy,
                       remedies=(remedies or [])[:3], meal_ids=list(meal_ids or []),
                       product_id=product_id, trip_date=trip_date, recipe_key=recipe_key)


def sort_key(w: PlanWarning) -> tuple:
    return (LEVELS.index(w.level), w.strategy or "", w.code,
            w.trip_date.isoformat() if w.trip_date else "", w.meal_ids, w.product_id or 0,
            w.recipe_key or "", w.message)


def counts(warnings: list[PlanWarning], strategy: str | None = None) -> dict[str, int]:
    """Warnings per level that apply under `strategy`: its own and the plan-wide ones."""
    out = dict.fromkeys(LEVELS, 0)
    for w in warnings:
        if w.strategy is None or w.strategy == strategy:
            out[w.level] += 1
    return out


# ─── Remedies ────────────────────────────────────────────────

def move_meal(meal_id: str, date: dt.date, slot: str) -> dict:
    return {"op": "move_meal", "meal_id": meal_id, "date": date.isoformat(), "slot": slot}


def set_storage(product_id: int, storage: str) -> dict:
    return {"op": "set_storage", "product_id": product_id, "storage": storage}


def add_trip(date: dt.date) -> dict:
    return {"op": "add_trip", "date": date.isoformat()}


def set_servings(recipe_key: str) -> dict:
    return {"op": "set_servings", "recipe_key": recipe_key}


def set_packs(date: dt.date, product_id: int, packs: int | None = None) -> dict:
    op = {"op": "set_packs", "date": date.isoformat(), "product_id": product_id}
    if packs is not None:
        op["packs"] = packs
    return op


def open_options(date: dt.date, product_id: int) -> dict:
    return {"op": "open_options", "date": date.isoformat(), "product_id": product_id}


def resolve(recipe_key: str) -> dict:
    return {"op": "resolve", "recipe_key": recipe_key}


def set_strategy(strategy: str) -> dict:
    return {"op": "set_strategy", "strategy": strategy}


def set_pref(field: str, value) -> dict:
    return {"op": "set_pref", "field": field, "value": value}


def approve_trip(date: dt.date, strategy: str) -> dict:
    return {"op": "approve_trip", "date": date.isoformat(), "strategy": strategy}


def remove_meal(meal_id: str) -> dict:
    return {"op": "remove_meal", "meal_id": meal_id}

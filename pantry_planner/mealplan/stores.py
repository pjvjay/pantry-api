"""POST /mealplan/stores: the stores the shopper can turn off and on in the Ingredients stage.
Pure code: no LLM call and no write.

Every store, nearest first from the plan's shopping point (schedule.shopping_point), measured
as the price matrix measures it; in_range against the plan's max_km; off from
settings.stores_off; and how many of the basket's products each one stocks, the products the
draft buys (basket.purchases: the resolve's picks, the shopper's pins, less what the origin
exclusion holds back), so the shopper sees what turning a store off would cost them.

Each store's street address, lat and lon are the stores table's, for the list and the map, with
synthetic (STORES_SYNTHETIC) saying whether they are the demo's fictional stores, for the
console to label them demo stores. The calendar export keeps its own policy: it never writes a
synthetic store's address (calendar_api._addresses). origin and max_km are the plan's shopping
point and distance limit, the map's centre and range.
"""
from __future__ import annotations

import math
from collections import Counter

from sqlalchemy import text
from sqlalchemy.orm import Session

from .. import db
from ..config import settings
from ..nlsearch.sql_builder import build_store_distances_sql
from . import basket, schedule
from .models import GeoPoint, MealPlanDraft, PlanStore, PlanStores


def stores(draft: MealPlanDraft) -> PlanStores:
    """Raises MealPlanError (REST 422) as /mealplan/schedule does, but for a pin the origin
    exclusion holds back, which basket.purchases keeps and leaves out of products_total."""
    pu = basket.purchases(draft)
    ids = sorted(pu.buying.keys() - pu.held.keys())
    lat, lon, max_km = schedule.shopping_point(draft.settings)
    # Every store's offers for them, whatever its distance or whether it is off.
    stocked = Counter(r["store_id"] for r in schedule._offers(ids, lat, lon, None)) if ids \
        else Counter()
    sql, params = build_store_distances_sql(lat, lon)
    with Session(db.engine()) as s:
        rows = [dict(r) for r in s.execute(text(sql), params).mappings()]
    off = set(draft.settings.stores_off)
    return PlanStores(stores=[PlanStore(
        id=r["store_id"], name=r["store_name"], address=r["address"] or None, lat=r["lat"],
        lon=r["lon"], distance_km=round(math.sqrt(max(r["dist_km2"], 0.0)), 1),
        in_range=max_km is None or r["dist_km2"] <= max_km * max_km,
        off=r["store_id"] in off, sells=stocked[r["store_id"]] if ids else None)
        for r in rows], products_total=len(ids), synthetic=settings().stores_synthetic,
        origin=GeoPoint(lat=lat, lon=lon), max_km=max_km)

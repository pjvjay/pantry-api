"""Cited storage and thaw times: seeds/shelf_life.json, read at runtime.

The file holds the rows of two public pages (FoodSafety.gov's Cold Food Storage Chart and
USDA FSIS's "The Big Thaw"), quoted verbatim, and a map from catalog product to the rows that
apply. Its rules of use are this module's rules:

- Plans use a row's lower bound (days_min); the console shows the verbatim text.
- A time stated in months or years has no day count: it only ever counts as longer than the
  plan's 14 days.
- Where a product cites more than one row for one kind of storage, the shorter time is
  planned and every row is cited.
- Fridge thawing takes at least 24 hours for a small amount, and at least 24 hours per 5 lb
  (2.268 kg, the file's own conversion) for a large item. The rest of a thawed pack is never
  planned for a later meal.

A product with no mapping has no cited time. The planner never makes one up: it uses the
shopper's buy-ahead setting for a chilled or fresh product, and plans no limit for one the
file classes as shelf-stable or bought frozen, saying so on the line.
"""
from __future__ import annotations

import functools
import json
import math
from dataclasses import dataclass

from ..db import SEEDS_DIR

SHELF_FILE = SEEDS_DIR / "shelf_life.json"


@functools.lru_cache(maxsize=1)
def load() -> dict:
    return json.loads(SHELF_FILE.read_text(encoding="utf-8"))


@functools.lru_cache(maxsize=1)
def _rules() -> dict[str, dict]:
    data = load()
    return {r["id"]: r for r in [*data["rules"], *data["thaw_rules"]]}


def sources() -> list[dict]:
    """The cited pages, each with its url, page date, retrieval date and credit line."""
    return [dict(s) for s in load()["sources"]]


def source(source_id: str) -> dict:
    return next(s for s in load()["sources"] if s["id"] == source_id)


def rule(rule_id: str) -> dict:
    return _rules()[rule_id]


@dataclass(frozen=True)
class ProductShelf:
    """What the file says about one product. mapped False: no cited row, and `reason` says
    why. storage_class is the file's own label (chilled_or_fresh, shelf_stable, frozen)."""
    product_id: int
    mapped: bool
    storage_class: str
    bought_state: str | None = None
    fridge: tuple[str, ...] = ()
    freezer: tuple[str, ...] = ()
    after_thaw: tuple[str, ...] = ()
    thaw: tuple[str, ...] = ()
    note: str = ""
    reason: str = ""
    synthetic_product: bool = False

    @property
    def fridge_days(self) -> int | None:
        """The shortest cited fridge time's lower bound, in days; None when there is no
        cited fridge row with a day count."""
        days = [rule(r)["days_min"] for r in self.fridge if rule(r)["days_min"] is not None]
        return min(days) if days else None

    @property
    def bought_frozen(self) -> bool:
        return self.mapped and self.bought_state == "frozen"

    @property
    def freezable(self) -> bool:
        """True when every cited freezer row states a time in months or years, which is
        longer than any plan: freezing on arrival keeps it to the meal. A row like the eggs'
        "Do not freeze in shell" states none, so the product is not frozen."""
        return bool(self.freezer) and all(
            (rule(r).get("stated") or {}).get("unit") in {"month", "year"}
            for r in self.freezer)

    def fridge_rules(self) -> list[dict]:
        return [rule(r) for r in self.fridge]


@functools.lru_cache(maxsize=1)
def _by_product() -> dict[int, ProductShelf]:
    data = load()
    out: dict[int, ProductShelf] = {}
    for p in data["products"]:
        rules = p["rules"]
        out[p["product_id"]] = ProductShelf(
            product_id=p["product_id"], mapped=True, storage_class=p["storage_class"],
            bought_state=p.get("bought_state"),
            fridge=tuple(rules.get("fridge", [])), freezer=tuple(rules.get("freezer", [])),
            after_thaw=tuple(rules.get("after_thaw_fridge", [])),
            thaw=tuple(rules.get("thaw", [])), note=p.get("note", ""),
            synthetic_product=bool(p.get("synthetic_product")))
    for u in data["unmapped"]:
        out[u["product_id"]] = ProductShelf(
            product_id=u["product_id"], mapped=False, storage_class=u["storage_class"],
            reason=u.get("reason", ""), synthetic_product=bool(u.get("synthetic_product")))
    return out


# A product the file does not list at all (a catalog row added after it was written) has
# no cited time and no storage class: it is planned like a chilled one, with the shopper's
# setting, which is the cautious reading.
_NOT_LISTED = "This product is not in shelf_life.json, so its storage time is unknown."


def for_product(product_id: int) -> ProductShelf:
    return _by_product().get(product_id) or ProductShelf(
        product_id=product_id, mapped=False, storage_class="chilled_or_fresh",
        reason=_NOT_LISTED)


@dataclass(frozen=True)
class ThawLead:
    days: int
    rule_ids: tuple[str, ...]
    verbatim: str
    weight_known: bool


def thaw_lead(ps: ProductShelf, kg: float | None) -> ThawLead | None:
    """Days a frozen pack needs in the fridge before the meal, from the cited thaw rows: a
    full day for a small amount; a day per 2.268 kg (5 lb) for a large item, rounded up.
    None when the product has no cited thaw row. With the weight unknown it is the small
    amount's full day, which both rows state as a minimum."""
    if not ps.thaw:
        return None
    large = next((rule(r) for r in ps.thaw if (rule(r).get("lead") or {}).get("per_kg_derived")),
                 None)
    small = next((rule(r) for r in ps.thaw
                  if rule(r).get("lead") and not rule(r)["lead"].get("per_kg_derived")), None)
    if large is not None and kg is not None and kg > large["lead"]["per_kg_derived"]:
        days = math.ceil(kg / large["lead"]["per_kg_derived"] - 1e-9)
        return ThawLead(days=days, rule_ids=(large["id"],), verbatim=large["verbatim"],
                        weight_known=True)
    used = small or large
    if used is None:
        return None
    return ThawLead(days=max(1, math.ceil(used["lead"]["hours"] / 24)), rule_ids=(used["id"],),
                    verbatim=used["verbatim"], weight_known=kg is not None)


def short_source(source_id: str) -> str:
    """'FoodSafety.gov, Cold Food Storage Chart' for a reason line."""
    s = source(source_id)
    publisher = s["publisher"].split(",")[0]
    return f"{publisher}, {s['title']}"


def credit_lines() -> list[str]:
    return [s["credit"] for s in load()["sources"]]

"""Helpers shared by the meal-plan tests: a seeded database per module, drafts built from the
demo starters (resolved in DEMO_MODE, no LLM) or from hand-written docs with chosen products.

Expected values in the tests come from the seed files, the cited rows and plain arithmetic,
never from the engine under test.
"""
from __future__ import annotations

import datetime as dt
import json
import os

from pantry_planner.db import SEEDS_DIR

SENTENCE = ("3 Pepperoni Pizza + 2 Chicken Fried Rice + 3 chicken briyani + 7 mango "
            "milkshakes in 2 weeks")
START = dt.date(2026, 10, 9)          # a Friday, as in the design's worked example
EXAMPLE = {"pepperoni_pizza": 3, "chicken_fried_rice": 2, "chicken_biryani": 3,
           "mango_milkshake": 7}

PRODUCTS = {p["id"]: p for p in json.loads((SEEDS_DIR / "products.json").read_text())}
SHELF = json.loads((SEEDS_DIR / "shelf_life.json").read_text())
STARTERS = {s["key"]: s for s in json.loads((SEEDS_DIR / "mealplan_starters.json")
                                            .read_text())["starters"]}


def use_db(tmp_path_factory, name: str) -> None:
    """Point the app at a fresh seeded database (SQLite, or PANTRY_TEST_DB_URL) in demo mode."""
    os.environ["DB_URL"] = (os.environ.get("PANTRY_TEST_DB_URL")
                            or f"sqlite:///{tmp_path_factory.mktemp('db') / f'{name}.db'}")
    os.environ["DEMO_MODE"] = "1"
    os.environ.pop("ANTHROPIC_API_KEY", None)
    from pantry_planner import config, db
    from pantry_planner.mealplan import resolve
    from pantry_planner.nlsearch import vocab

    config.settings.cache_clear()
    db.seed_from_json()
    vocab.clear_cache()
    resolve.clear_cache()


def done_db() -> None:
    from pantry_planner import config
    from pantry_planner.nlsearch import vocab

    os.environ.pop("DEMO_MODE", None)
    config.settings.cache_clear()
    vocab.clear_cache()


def client():
    from fastapi.testclient import TestClient

    from pantry_planner.api import app

    return TestClient(app)


def day(i: int) -> dt.date:
    return START + dt.timedelta(days=i)


def index(date: str | dt.date) -> int:
    d = dt.date.fromisoformat(date) if isinstance(date, str) else date
    return (d - START).days


def starter_draft(counts: dict[str, int] | None = None, *, slots: dict[str, str] | None = None,
                  **extra) -> dict:
    """A draft of demo starters, resolved in demo mode (resolve_one: run_spec, no LLM)."""
    from pantry_planner.mealplan.models import RecipeRef
    from pantry_planner.mealplan.resolve import resolve_one

    recipes, resolved = {}, {}
    for key, n in (counts or EXAMPLE).items():
        ref = RecipeRef(key=f"starter:{key}", starter=key)
        recipes[ref.key] = {"ref": ref.model_dump(mode="json"), "wanted": n,
                            "slot": (slots or {}).get(key, STARTERS[key]["slot"])}
        resolved[ref.key] = resolve_one(ref).model_dump(mode="json")
    return {"start_date": START.isoformat(), "days": 14, "recipes": recipes,
            "resolved": resolved, **extra}


def doc_recipe(key: str, title: str, servings: int | None,
               lines: list[tuple[str, float | None, str, int]], *, wanted: int = 0,
               slot: str = "dinner") -> tuple[dict, dict]:
    """(recipes entry, resolved entry) for a hand-written doc whose lines buy the given
    product ids: [(name, quantity, unit, product_id)]."""
    doc = {"key": key, "title": title, "servings": servings,
           "servings_stated": servings is not None,
           "lines": [{"line_no": i, "text": f"{q or ''} {u} {n}".strip(), "name": n,
                      "quantity": q, "unit": u, "amount_basis": "parsed_from_your_paste"}
                     for i, (n, q, u, _pid) in enumerate(lines, start=1)],
           "source": {"kind": "pasted", "method": "paste"}}
    entry = {"ref": {"key": key, "doc": doc}, "wanted": wanted, "slot": slot}
    resolved = {"key": key, "title": title,
                "status": "ok" if servings is not None else "needs_servings",
                "servings": servings,
                "lines": [{"line_no": i, "name": n, "quantity": q, "unit": u, "product_id": pid}
                          for i, (n, q, u, pid) in enumerate(lines, start=1)]}
    return entry, resolved


def doc_draft(recipes: dict[str, tuple[dict, dict]], **extra) -> dict:
    return {"start_date": START.isoformat(), "days": 14,
            "recipes": {k: v[0] for k, v in recipes.items()},
            "resolved": {k: v[1] for k, v in recipes.items()}, **extra}


def schedule(draft: dict) -> dict:
    r = client().post("/mealplan/schedule", json=draft)
    assert r.status_code == 200, r.text
    return r.json()


def strategy(sched: dict, name: str) -> dict:
    return next(s for s in sched["strategies"] if s["name"] == name)


def trip_days(sched: dict, name: str) -> list[int]:
    return [index(t["date"]) for t in strategy(sched, name)["trips"]]


def lines_of(sched: dict, name: str, product_id: int) -> list[tuple[dict, dict]]:
    """(trip, line) for every line buying this product."""
    return [(t, ln) for t in strategy(sched, name)["trips"] for ln in t["lines"]
            if ln["product"]["id"] == product_id]


def codes(sched: dict, strategy_name: str | None = None) -> list[str]:
    return [w["code"] for w in sched["warnings"]
            if strategy_name is None or w["strategy"] in (None, strategy_name)]

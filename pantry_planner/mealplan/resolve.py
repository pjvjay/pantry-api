"""Resolve: which products a recipe buys, once per distinct recipe (the slow path).

A recipe enters the meal plan's tray once and is resolved then, never on a drag. Library
recipes go through flow.run (the seeded lines, as /plan/{slug} plans them); every RecipeDoc
(a demo starter, a pasted or imported recipe, a dish the assistant wrote) goes through
flow.run_spec, so its reviewed lines are planned as given with no parse, and only the
product choice is the selector's. Results are cached in process (64 entries) by everything
that can change them.

A recipe that fails keeps a status of its own (needs_servings, unconfirmed_lines, not_found,
unparseable, aborted, llm_error) and never fails the others.

This module also says what a recipe ref is (recipe_doc): the schedule reads the amounts from
here, by ref, rather than trusting amounts the browser sends back.
"""
from __future__ import annotations

import functools
import hashlib
import json
import threading
from collections import OrderedDict

from .. import db, flow
from ..config import settings
from ..db import SEEDS_DIR
from ..models import RecipeDoc, RecipeLine, RecipeSource, ShoppingPlan
from ..nlsearch.units import normalize_quantity
from ..recipe_doc import UnconfirmedLines, library_doc, to_recipe_text, to_spec
from .models import MealPlanError, RecipeRef, ResolvedLine, ResolvedRecipe

STARTERS_FILE = SEEDS_DIR / "mealplan_starters.json"
LIBRARY_LABEL = "demo house amounts"

NEEDS_SERVINGS = "How many does this recipe serve? It does not say, so its amounts stay unknown."


# ─── Demo starters ───────────────────────────────────────────

@functools.lru_cache(maxsize=1)
def starters_file() -> dict:
    return json.loads(STARTERS_FILE.read_text(encoding="utf-8"))


def starter_doc(key: str) -> RecipeDoc:
    """A demo starter as a RecipeDoc ('starter:<key>'), its lines labelled demo house
    amounts. KeyError for an unknown key."""
    s = next((s for s in starters_file()["starters"] if s["key"] == key), None)
    if s is None:
        raise KeyError(key)
    return RecipeDoc(
        key=s["doc_key"], title=s["title"], servings=s["servings"], servings_stated=True,
        servings_basis="source", yield_text=f"Serves {s['servings']}",
        lines=[RecipeLine(line_no=ln["line_no"], text=ln["text"], name=ln["name"],
                          quantity=ln["quantity"], unit=ln["unit"], note=ln["note"],
                          amount_basis=ln["amount_basis"]) for ln in s["lines"]],
        source=RecipeSource(kind="starter", method="seed", label=s["label"]))


def starters() -> list[dict]:
    """GET /mealplan/starters: each starter with its doc, default slot and aliases."""
    out = []
    for s in starters_file()["starters"]:
        doc = starter_doc(s["key"])
        out.append({"key": s["key"], "doc_key": s["doc_key"], "title": s["title"],
                    "servings": s["servings"], "slot": s["slot"], "label": s["label"],
                    "aliases": list(s["aliases"]), "doc": doc.model_dump(mode="json"),
                    "text": to_recipe_text(doc)})
    return out


# ─── What a ref is ───────────────────────────────────────────

def recipe_doc(ref: RecipeRef, answer: int | None = None) -> RecipeDoc:
    """The recipe a ref names, as a RecipeDoc. `answer` (or ref.servings) is the shopper's
    count when the recipe does not say how many it serves: it fills servings with basis
    'your_setting'. MealPlanError unknown_recipe_key for a slug or starter that does not
    exist."""
    if ref.slug is not None:
        try:
            recipe = db.load_recipe(ref.slug)
        except ValueError as e:
            raise MealPlanError("unknown_recipe_key", f"no library recipe {ref.slug!r}",
                                recipe_key=ref.key) from e
        amounts = db.load_line_amounts([ref.slug])
        doc = library_doc(recipe, None if amounts is None else amounts.get(ref.slug, {}))
    elif ref.starter is not None:
        try:
            doc = starter_doc(ref.starter)
        except KeyError as e:
            raise MealPlanError("unknown_recipe_key", f"no demo starter {ref.starter!r}",
                                recipe_key=ref.key) from e
    else:
        doc = ref.doc
    answer = answer if answer is not None else ref.servings
    if doc.servings is None and answer is not None:
        doc = doc.model_copy(update={"servings": answer, "servings_basis": "your_setting"})
    return doc


def label_for(ref: RecipeRef, doc: RecipeDoc) -> str | None:
    if ref.slug is not None:
        return LIBRARY_LABEL
    return doc.source.label


# ─── Resolve ─────────────────────────────────────────────────

_CACHE_SIZE = 64
_cache: OrderedDict[str, ResolvedRecipe] = OrderedDict()
_cache_lock = threading.Lock()


def clear_cache() -> None:
    with _cache_lock:
        _cache.clear()


def _cache_key(ref: RecipeRef, knobs: dict) -> str:
    cfg = settings()
    body = {"ref": ref.model_dump(mode="json"), "knobs": knobs, "demo_mode": cfg.demo_mode,
            "selector": cfg.selector_model_default, "router": cfg.routing_strategy}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def _lines(doc: RecipeDoc, plan: ShoppingPlan | None) -> list[ResolvedLine]:
    chosen = {}
    if plan is not None and plan.basis is not None:
        chosen = {b.line_no: b for b in plan.basis.lines}
    names = {li.product_id: li.product_name for li in plan.line_items} if plan else {}
    out = []
    for ln in doc.lines:
        need = normalize_quantity(ln.quantity, ln.unit)
        b = chosen.get(ln.line_no)
        pid = b.product_id if b is not None else None
        out.append(ResolvedLine(
            line_no=ln.line_no, name=ln.name, quantity=ln.quantity, unit=ln.unit,
            need_qty=None if need is None else round(need[0], 4),
            need_uom=None if need is None else need[1],
            product_id=pid, product_name=names.get(pid) if pid is not None else None,
            match=b.level if b is not None and pid is not None else None,
            amount_basis=ln.amount_basis))
    return out


def resolve_one(ref: RecipeRef, *, lat: float | None = None, lon: float | None = None,
                max_km: float | None = None, exclude: list[str] | None = None,
                preference: list[str] | None = None) -> ResolvedRecipe:
    """Resolve one recipe, from the cache when it can. Never raises for a recipe that
    fails: the failure is its status."""
    from ..llm import LLMError
    from ..nlsearch import PlanAborted, UnparseableRecipe

    knobs = {"lat": lat, "lon": lon, "max_km": max_km, "exclude": list(exclude or []),
             "preference": list(preference or [])}
    key = _cache_key(ref, knobs)
    with _cache_lock:
        hit = _cache.get(key)
        if hit is not None:
            _cache.move_to_end(key)
            return hit.model_copy(update={"cached": True})

    try:
        doc = recipe_doc(ref)
    except MealPlanError as e:
        return ResolvedRecipe(key=ref.key, title=ref.key, status="not_found", message=e.detail)
    base = {"key": ref.key, "title": doc.title, "servings": doc.servings,
            "servings_basis": (doc.servings_basis or "source") if doc.servings else None,
            "label": label_for(ref, doc)}
    try:
        if ref.slug is not None:
            # With none of lat/lon/max_km this is the classic plan, priced from the catalog.
            plan = flow.run(ref.slug, exclude=exclude, preference=preference,
                            lat=lat, lon=lon, max_km=max_km, allow_partial=True)
        else:
            plan = flow.run_spec(to_spec(doc), lat=lat, lon=lon, exclude=exclude,
                                 preference=preference, max_km=max_km, allow_partial=True,
                                 display_text=to_recipe_text(doc))
    except UnconfirmedLines as e:
        return ResolvedRecipe(**base, status="unconfirmed_lines", message=str(e),
                              lines=_lines(doc, None))
    except UnparseableRecipe:
        return ResolvedRecipe(**base, status="unparseable",
                              message="The recipe has no ingredient lines to plan.",
                              lines=_lines(doc, None))
    except PlanAborted as e:
        alert = e.execution.aborted
        return ResolvedRecipe(**base, status="aborted",
                              message=alert.message if alert else "planning stopped",
                              lines=_lines(doc, None))
    except LLMError as e:
        return ResolvedRecipe(**base, status="llm_error", message=str(e),
                              lines=_lines(doc, None))

    out = ResolvedRecipe(
        **base, status="ok" if doc.servings is not None else "needs_servings",
        message="" if doc.servings is not None else NEEDS_SERVINGS,
        lines=_lines(doc, plan), not_stocked=plan.not_stocked,
        out_of_range=plan.out_of_range, skipped=plan.skipped,
        llm_cost_usd=plan.total_llm_cost_usd, model_used=plan.preselected_model,
        burr_run=plan.burr_run)
    with _cache_lock:
        _cache[key] = out
        _cache.move_to_end(key)
        while len(_cache) > _CACHE_SIZE:
            _cache.popitem(last=False)
    return out

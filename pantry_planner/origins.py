"""
Country-of-origin resolution — three tiers, cheapest first.

The design goal is "simple and low-cost": most products resolve for
free, and no product ever pays for more than one LLM call.

  Tier 0 — DB cache (product_origins table). LLM answers are written
           back once and read forever after. Free.
  Tier 1 — deterministic heuristics. Ordered keyword rules over the
           product name/brand, then subcategory defaults tuned to a
           Canadian supermarket. Free, instant, covers most of the
           catalog.
  Tier 2 — one batch Haiku call (forced tool use, submit_origins) for
           whatever the rules didn't cover. Same structured-output
           pattern as selector.py. Cached to the DB afterwards.

Callers that must stay free (e.g. search filters) pass allow_llm=False
and get country="Unknown" for tier-2 products instead of a call.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime

from anthropic import Anthropic

from .config import estimate_cost_usd, settings
from .models import Product, ProductOrigin
from .prompts import ORIGIN_SYSTEM, ORIGIN_TOOL

# ─── Tier 1: deterministic rules ─────────────────────────────
# Keyword rules run first (most specific wins — order matters), then
# subcategory defaults. Keywords match on lowercased "name brand".
# Confidence reflects how safe the inference is for a Canadian store;
# anything we can't state with >= 0.6 is left to the LLM tier.

# (keywords, country, confidence, reasoning)
_KEYWORD_RULES: list[tuple[tuple[str, ...], str, float, str]] = [
    (("canadian",), "Canada", 0.95, "Named Canadian in the product name"),
    (("cadbury",), "United Kingdom", 0.9, "Cadbury is a British chocolate brand"),
    (("basmati",), "India", 0.9, "Basmati rice is grown in India/Pakistan"),
    (("garam masala",), "India", 0.9, "Garam masala is an Indian spice blend"),
    (("turmeric",), "India", 0.8, "India produces most of the world's turmeric"),
    (("parmesan", "parmigiano"), "Italy", 0.85, "Parmesan is an Italian designation"),
    (("passata",), "Italy", 0.85, "Passata is an Italian tomato preparation"),
    (("canola",), "Canada", 0.9, "Canola is a Canadian crop and product"),
    (("atlantic salmon",), "Canada", 0.8, "Atlantic salmon sold in BC is Canadian-farmed"),
    (("soy sauce",), "China", 0.6, "Most supermarket soy sauce is brewed in China"),
    (("coconut milk", "coconut yogurt"), "Thailand", 0.6,
     "Canned coconut products are typically Thai"),
    (("olive oil",), "Italy", 0.6, "Supermarket olive oil is typically Italian-bottled"),
]

# subcategory -> (country, confidence, reasoning)
_SUBCATEGORY_DEFAULTS: dict[str, tuple[str, float, str]] = {
    "vegetables": ("Canada", 0.6, "Fresh produce in Canadian stores is typically domestic"),
    "herbs": ("Canada", 0.6, "Fresh herbs are typically grown domestically"),
    "milk": ("Canada", 0.85, "Canadian dairy is supply-managed and domestic"),
    "butter": ("Canada", 0.85, "Canadian dairy is supply-managed and domestic"),
    "yogurt": ("Canada", 0.8, "Canadian dairy is supply-managed and domestic"),
    "cheese": ("Canada", 0.6, "Everyday cheese in Canadian stores is domestic"),
    "eggs": ("Canada", 0.85, "Canadian eggs are supply-managed and domestic"),
    "poultry": ("Canada", 0.8, "Canadian poultry is supply-managed and domestic"),
    "beef": ("Canada", 0.7, "Most supermarket beef in Canada is domestic"),
    "bread": ("Canada", 0.75, "Supermarket bread is baked locally"),
    "pasta": ("Italy", 0.65, "Dry pasta on Canadian shelves is typically Italian"),
    "baking": ("Canada", 0.7, "Flour sold in Canada is milled from Canadian wheat"),
}

UNKNOWN = "Unknown"


def heuristic_origin(product: Product) -> ProductOrigin | None:
    """Tier 1: return an origin if a deterministic rule covers the
    product, else None (meaning: needs the LLM tier)."""
    haystack = f"{product.name} {product.brand}".lower()
    for keywords, country, confidence, reason in _KEYWORD_RULES:
        if any(k in haystack for k in keywords):
            return ProductOrigin(
                product_id=product.id, product_name=product.name,
                country=country, confidence=confidence,
                source="heuristic", reasoning=reason)
    sub = (product.subcategory or "").lower()
    if sub in _SUBCATEGORY_DEFAULTS:
        country, confidence, reason = _SUBCATEGORY_DEFAULTS[sub]
        # Dairy-aisle plant-based items (oat/almond/coconut/vegan) are
        # processed goods, not supply-managed dairy — leave to the LLM.
        if not any(t in haystack for t in ("oat", "almond", "coconut", "vegan", "gluten-free")):
            return ProductOrigin(
                product_id=product.id, product_name=product.name,
                country=country, confidence=confidence,
                source="heuristic", reasoning=reason)
    return None


# ─── Tier 2: batch LLM fallback ──────────────────────────────

def call_origin_resolver(
    products: list[Product], *, model: str,
) -> tuple[list[ProductOrigin], float]:
    """One batch call for every unresolved product. Returns (origins,
    cost_usd). Same forced-tool-use pattern as selector.call_selector."""
    cfg = settings()
    client = Anthropic(api_key=cfg.anthropic_api_key)

    payload = [
        {
            "id": p.id,
            "name": p.name,
            "brand": p.brand or None,
            "category": p.category,
            "subcategory": p.subcategory,
            "description": p.description,
        }
        for p in products
    ]
    kwargs: dict = {
        "model": model,
        "max_tokens": 4096,
        "system": ORIGIN_SYSTEM,
        "tools": [ORIGIN_TOOL],
        "tool_choice": {"type": "tool", "name": "submit_origins"},
        "messages": [{
            "role": "user",
            "content": json.dumps({"products": payload}, indent=2),
        }],
    }
    resp = client.messages.create(**kwargs)

    tool_block = next((b for b in resp.content if b.type == "tool_use"), None)
    if tool_block is None:
        raise ValueError(f"Origin resolver didn't call the tool. Response: {resp.content!r}")

    by_id = {p.id: p for p in products}
    origins = []
    for o in tool_block.input["origins"]:
        pid = o["product_id"]
        if pid not in by_id:  # ignore hallucinated ids
            continue
        origins.append(ProductOrigin(
            product_id=pid,
            product_name=by_id[pid].name,
            country=o["country"] or UNKNOWN,
            confidence=max(0.0, min(1.0, float(o["confidence"]))),
            source="llm",
            reasoning=o.get("reasoning", ""),
        ))
    cost = estimate_cost_usd(model, resp.usage.input_tokens, resp.usage.output_tokens)
    return origins, cost


# ─── Public API ──────────────────────────────────────────────

def resolve_origins(products: list[Product], *, allow_llm: bool = True) -> list[ProductOrigin]:
    """Resolve countries of origin for the given products, cheapest
    tier first. Order of the input is preserved.

    allow_llm=False guarantees zero API cost: products no cache entry or
    rule covers come back as country="Unknown" with confidence 0.
    """
    from . import db

    resolved: dict[int, ProductOrigin] = {}

    # Tier 0 — cache
    cached = db.load_cached_origins([p.id for p in products])
    names = {p.id: p.name for p in products}
    for pid, row in cached.items():
        # str()/float() casts: db.py uses legacy Column declarations, so
        # attributes type as Column[...] under mypy.
        resolved[pid] = ProductOrigin(
            product_id=pid, product_name=names[pid],
            country=str(row.country), confidence=float(row.confidence),
            source="cache", reasoning=str(row.reasoning))

    # Tier 1 — heuristics
    unresolved: list[Product] = []
    for p in products:
        if p.id in resolved:
            continue
        origin = heuristic_origin(p)
        if origin:
            resolved[p.id] = origin
        else:
            unresolved.append(p)

    # Tier 2 — one batch LLM call, cached for next time
    if unresolved and allow_llm and settings().anthropic_api_key:
        llm_origins, _cost = call_origin_resolver(
            unresolved, model=settings().classifier_model)
        now = datetime.now(UTC).isoformat(timespec="seconds")
        db.save_origins([
            {**o.model_dump(include={"product_id", "country", "confidence",
                                     "source", "reasoning"}),
             "resolved_at": now}
            for o in llm_origins
        ])
        for o in llm_origins:
            resolved[o.product_id] = o

    # Whatever is left is unknowable at zero cost
    return [
        resolved.get(p.id) or ProductOrigin(
            product_id=p.id, product_name=p.name, country=UNKNOWN,
            confidence=0.0, source="heuristic",
            reasoning="No rule matched and LLM resolution was disabled")
        for p in products
    ]


def search_by_origin(
    country: str, *, allow_llm: bool = False,
) -> list[tuple[Product, ProductOrigin]]:
    """All products whose resolved origin matches `country` (case-
    insensitive substring, so "uk"/"United Kingdom" both work). Free by
    default — pass allow_llm=True to also resolve uncached tier-2
    products at the cost of one batch call."""
    from . import db

    products = db.load_all_products()
    origins = resolve_origins(products, allow_llm=allow_llm)
    needle = country.strip().lower()
    by_id = {p.id: p for p in products}
    return [
        (by_id[o.product_id], o)
        for o in origins
        if needle and needle in o.country.lower()
    ]

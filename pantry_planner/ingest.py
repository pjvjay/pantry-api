"""
Ingest for the claude-chrome-container tooling.

That container is a supervised, hard-rate-limited browser agent — Walmart
blocked it after three searches ~18s apart — so it is a batch producer of
files, never a live dependency of this API. It writes; we read.

Three inputs, all already documented and schema-validated on its side:

  ./origin --json   {"products": [{product, claim_type, ingredient_origin,
                                   manufactured_in, verbatim, confidence,
                                   note}], "usage": ..., "model": ...}
  ./label  --json   {"labels":  [{file, cached, photo, product, verbatim,
                                  claim_type, country, importer_only,
                                  confidence, note}]}
  ./grocery         a markdown table:
                    Item | Store | Branch | Product | Size | Price | UnitPrice | Link

Product names in those files are free text from retailer pages and do not
match catalog names exactly, so rows are matched by token overlap using the
same >=0.5 threshold the container applies to Open Food Facts candidates —
the threshold that stopped "bananas" resolving to a Moroccan yogurt. A row
that matches nothing is reported, never silently dropped: for price
observations it is stored with a null product_id, and for evidence it is
counted as unmatched so the miss stays visible.
"""
from __future__ import annotations

import json
import re
import sys
from datetime import UTC, datetime

from .models import Product
from .origins import FULL_CLAIMS

MATCH_THRESHOLD = 0.5


# ─── Matching catalog products to scraped names ──────────────

def _tokens(text: str) -> set[str]:
    return {w for w in re.sub(r"[^a-z0-9 ]", " ", (text or "").lower()).split()
            if len(w) > 2}


def overlap(a: str, b: str) -> float:
    """Shared significant tokens over the smaller token set."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


def match_product(name: str, products: list[Product]) -> Product | None:
    """Best catalog product for a scraped name, or None below threshold."""
    best: Product | None = None
    best_score = 0.0
    for p in products:
        score = max(overlap(name, p.name), overlap(name, f"{p.brand} {p.name}"))
        if score > best_score:
            best, best_score = p, score
    return best if best_score >= MATCH_THRESHOLD else None


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


# ─── ./origin --json ─────────────────────────────────────────

def ingest_origin_json(payload: dict, *, products: list | None = None) -> dict:
    """Turn reconciled origin records into evidence rows.

    `conflicting` and `unknown` records are stored as-is rather than
    discarded: that a source set disagreed is itself worth keeping, and
    resolve_origin() knows not to rank them.
    """
    from . import db

    if products is None:
        products = db.load_all_products()
    observed = _now()
    rows, unmatched = [], []

    for rec in payload.get("products", []):
        name = (rec.get("product") or "").strip()
        product = match_product(name, products)
        if product is None:
            unmatched.append(name)
            continue
        rows.append({
            "product_id": product.id,
            "source": "open-food-facts",
            "source_ref": rec.get("code") or "",
            "claim_type": rec.get("claim_type") or "unknown",
            "verbatim": rec.get("verbatim") or "",
            "ingredient_origin": rec.get("ingredient_origin") or "",
            "manufactured_in": rec.get("manufactured_in") or "",
            "confidence": rec.get("confidence") or "low",
            "importer_only": False,
            "note": rec.get("note") or "",
            "observed_at": observed,
        })

    written = db.save_origin_evidence(rows)
    return {"written": written, "unmatched": unmatched,
            "model": payload.get("model", ""), "usage": payload.get("usage", {})}


# ─── ./label --json ──────────────────────────────────────────

def ingest_label_json(payload: dict, *, products: list | None = None) -> dict:
    """Turn package-photo reads into evidence rows.

    A label is the only source that works for imported goods, so these are
    the highest-value rows in the table. `importer_only` is carried through
    untouched — an "Imported by ..." address is not a country of origin and
    must never be ranked as one.
    """
    from . import db

    if products is None:
        products = db.load_all_products()
    observed = _now()
    rows, unmatched, no_claim = [], [], []

    for rec in payload.get("labels", []):
        name = (rec.get("product") or rec.get("photo") or "").strip()
        claim = rec.get("claim_type") or "none"
        country = (rec.get("country") or "").strip()
        product = match_product(name, products)
        if product is None:
            unmatched.append(name)
            continue
        if claim == "none" or not country:
            # The vision pass correctly returns `none` for a photo with no
            # declaration. Recording it stops the same pack being
            # rephotographed, but it carries no origin.
            no_claim.append(name)
        rows.append({
            "product_id": product.id,
            "source": "label-photo",
            "source_ref": rec.get("file") or rec.get("photo") or "",
            "claim_type": claim,
            "verbatim": rec.get("verbatim") or "",
            # A "Product of X" claim asserts ingredient content too, so it
            # populates both fields; a processing claim only says where the
            # last substantial transformation happened.
            "ingredient_origin": country if claim in FULL_CLAIMS else "",
            "manufactured_in": country,
            "confidence": rec.get("confidence") or "low",
            "importer_only": bool(rec.get("importer_only")),
            "note": rec.get("note") or "",
            "observed_at": observed,
        })

    written = db.save_origin_evidence(rows)
    return {"written": written, "unmatched": unmatched, "no_claim": no_claim}


# ─── ./grocery markdown ──────────────────────────────────────

def parse_markdown_table(text: str) -> list[dict]:
    """Rows of the first markdown table found, keyed by lowercased header.

    Tolerates prose around the table, which the container's reports carry
    (there is always a ## Notes section).
    """
    rows: list[dict] = []
    header: list[str] | None = None
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            if header and rows:
                break          # table ended
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if all(set(c) <= set("-: ") for c in cells if c):
            continue           # separator row
        if header is None:
            header = [c.lower() for c in cells]
            continue
        if len(cells) != len(header):
            continue
        rows.append(dict(zip(header, cells, strict=False)))
    return rows


_PRICE = re.compile(r"\d+(?:[.,]\d+)?")


def parse_price(text: str) -> float | None:
    """First number in a price cell, or None.

    Multi-unit listings print ranges; the raw text is kept alongside so the
    range is not silently collapsed into a single number.
    """
    m = _PRICE.search((text or "").replace(",", ""))
    return float(m.group()) if m else None


def ingest_grocery_markdown(text: str, *, run_id: str = "",
                            products: list | None = None) -> dict:
    """Store scraped price rows.

    Unmatched listings are kept with a null product_id — they are real
    observations of the market even when the catalog has no counterpart,
    and dropping them would hide how often matching fails.
    """
    from . import db

    if products is None:
        products = db.load_all_products()
    observed = _now()
    parsed = parse_markdown_table(text)
    rows, matched = [], 0

    for r in parsed:
        name = r.get("product", "")
        if not name or name == "—":
            continue
        product = match_product(name, products)
        if product is not None:
            matched += 1
        rows.append({
            "product_id": product.id if product else None,
            "item_query": r.get("item", ""),
            "store": r.get("store", ""),
            "branch": r.get("branch", ""),
            "product_name": name,
            "size": r.get("size", ""),
            "price": parse_price(r.get("price", "")),
            "price_text": r.get("price", ""),
            "unit_price": r.get("unitprice", ""),
            "link": r.get("link", ""),
            "observed_at": observed,
            "run_id": run_id,
        })

    written = db.save_price_observations(rows)
    return {"written": written, "matched": matched,
            "unmatched": written - matched}


# ─── Resolved-summary refresh ────────────────────────────────

def refresh_resolved(product_ids: list[int] | None = None) -> dict:
    """Recompute product_origins summaries from the evidence table."""
    from . import db
    from .origins import _conf_float, resolve_all

    resolved = resolve_all(product_ids)
    now = _now()
    db.save_resolved_origins([
        {"product_id": o.product_id, "status": o.status,
         "claim_type": o.claim_type, "country": o.country,
         "ingredient_origin": o.ingredient_origin,
         "manufactured_in": o.manufactured_in, "verbatim": o.verbatim,
         "confidence": _conf_float(o.confidence), "source": o.source,
         "reasoning": o.note, "evidence_count": o.evidence_count,
         "resolved_at": now}
        for o in resolved.values()
    ])
    counts: dict[str, int] = {}
    for o in resolved.values():
        counts[o.status] = counts.get(o.status, 0) + 1
    return {"products": len(resolved), "by_status": counts}


# ─── CLI ─────────────────────────────────────────────────────

def main(argv: list[str]) -> int:
    usage = ("Usage: python -m pantry_planner.ingest "
             "{origin|label|grocery} <file>  [--run-id ID]\n"
             "       python -m pantry_planner.ingest refresh\n\n"
             "Files come from the claude-chrome-container tooling:\n"
             "  ./origin --json > origin.json\n"
             "  ./label  --json photos/*.jpg > labels.json\n"
             "  ./grocery --items \"butter,rice\" > prices.md\n")
    if not argv or argv[0] in {"-h", "--help"}:
        print(usage, file=sys.stderr)
        return 2

    cmd = argv[0]
    if cmd == "refresh":
        print(json.dumps(refresh_resolved(), indent=2))
        return 0

    if len(argv) < 2:
        print(usage, file=sys.stderr)
        return 2
    path = argv[1]
    run_id = ""
    if "--run-id" in argv:
        run_id = argv[argv.index("--run-id") + 1]

    with open(path) as f:
        raw = f.read()

    if cmd == "origin":
        result = ingest_origin_json(json.loads(raw))
    elif cmd == "label":
        result = ingest_label_json(json.loads(raw))
    elif cmd == "grocery":
        result = ingest_grocery_markdown(raw, run_id=run_id)
    else:
        print(usage, file=sys.stderr)
        return 2

    result["resolved"] = refresh_resolved()
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

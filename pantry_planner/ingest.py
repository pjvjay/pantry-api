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

import getpass
import json
import re
import sys
from datetime import UTC, datetime

from .models import Product
from .origins import FULL_CLAIMS, PROCESSING_CLAIMS

# ─── Origin submissions (the MCP write path) ─────────────────
CLAIM_TYPES = sorted(FULL_CLAIMS | PROCESSING_CLAIMS)
CONFIDENCES = ("high", "medium", "low")
SUBMISSION_STATUSES = ("pending", "approved", "rejected")
VERBATIM_MIN, VERBATIM_MAX = 3, 500
NOTE_MAX, SOURCE_REF_MAX, COUNTRY_MAX = 1000, 300, 80
REJECT_NOTE_MIN = 3
# What the submission dedupes on: the same reading of the same label. Not
# confidence or note — a second agent reading the same words with a
# different confidence is the same claim, and the queue should hold it once.
_SUBMISSION_KEY = ("product_id", "claim_type", "country", "verbatim", "importer_only")

MATCH_THRESHOLD = 0.5


# ─── Matching catalog products to scraped names ──────────────

def _stem(word: str) -> str:
    """Crude singular fold, matching the container's own tokenizer.

    Retailer listings and catalogs disagree on number — "Bananas" vs
    "Banana", "Roma Tomatoes" vs "Roma Tomato" — and without this the
    two-shared-token rule drops produce that used to match.
    """
    if len(word) > 3 and word.endswith("es") and word[-3] in "oshxz":
        return word[:-2]          # tomatoes -> tomato, boxes -> box
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _tokens(text: str) -> set[str]:
    return {_stem(w) for w in re.sub(r"[^a-z0-9 ]", " ", (text or "").lower()).split()
            if len(w) > 2}


def overlap(a: str, b: str) -> float:
    """Shared significant tokens over the smaller token set.

    Kept because retailer listings are verbose ("No Name Basmati Rice, 2
    kg" vs a catalog "Basmati Rice 2kg") and a symmetric measure punishes
    that. On its own it is far too permissive — see match_product.
    """
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


def _shared(a: str, b: str) -> int:
    return len(_tokens(a) & _tokens(b))


def match_product(name: str, products: list[Product]) -> Product | None:
    """Best catalog product for a scraped name, or None if unsure.

    Coverage alone is not enough: a single-token query like "Milk 2L"
    reduces to {milk} and covers 100% of itself against "Milk Chocolate
    Cadbury". Attaching a milk label's origin to a chocolate bar is a
    silent, confidently wrong answer, so two further conditions apply:

      * at least TWO significant tokens must be shared whenever both
        names have two or more to give
      * the top score must be unambiguous. Near-ties across different
        products ("Ground Beef Lean" vs "Ground Beef Medium 900g") mean
        the name does not identify one product, and guessing is worse
        than declining.
    """
    scored: list[tuple[float, int, Product]] = []
    for p in products:
        score = max(overlap(name, p.name), overlap(name, f"{p.brand} {p.name}"))
        shared = max(_shared(name, p.name), _shared(name, f"{p.brand} {p.name}"))
        if score < MATCH_THRESHOLD:
            continue
        # Gate on the CATALOG name, not on min(): a one-token query was
        # exempting itself from the very rule meant to catch it, so
        # "Milk 2L" still matched "Milk Chocolate Cadbury".
        if shared < 2 and len(_tokens(p.name)) >= 2:
            continue
        scored.append((score, shared, p))

    if not scored:
        return None
    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    if len(scored) > 1:
        (s0, n0, _), (s1, n1, _) = scored[0], scored[1]
        if abs(s0 - s1) < 1e-9 and n0 == n1:
            return None          # ambiguous — decline rather than guess
    return scored[0][2]


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


# ─── Origin submissions ──────────────────────────────────────
# The MCP server's write path and the CLI share these three functions, so
# a claim is validated, deduplicated and approved the same way whichever
# door it came through. Every ValueError names the field it is about: the
# caller (an agent, usually) has to be told what to fix.

def _submission_dict(row, product_name: str, *, duplicate: bool = False) -> dict:
    return {
        "id": int(row.id), "product_id": int(row.product_id),
        "product_name": product_name, "status": str(row.status),
        "claim_type": str(row.claim_type), "country": str(row.country),
        "ingredient_origin": str(row.ingredient_origin or ""),
        "manufactured_in": str(row.manufactured_in or ""),
        "verbatim": str(row.verbatim), "confidence": str(row.confidence),
        "importer_only": bool(row.importer_only), "note": str(row.note or ""),
        "source_ref": str(row.source_ref or ""),
        "submitted_by": str(row.submitted_by or ""),
        "submitted_at": str(row.submitted_at or ""),
        "reviewed_by": str(row.reviewed_by or ""),
        "reviewed_at": str(row.reviewed_at or ""),
        "review_note": str(row.review_note or ""),
        "evidence_id": int(row.evidence_id) if row.evidence_id is not None else None,
        "duplicate": duplicate,
    }


def _bounded(record: dict, field: str, limit: int) -> str:
    value = str(record.get(field) or "")
    if len(value) > limit:
        raise ValueError(f"{field}: at most {limit} characters, got {len(value)}")
    return value


def _queue_guard(fn):
    """Turn a missing review-queue table into a ValueError the tools already
    map to a clear ToolError, instead of a raw driver error."""
    import functools

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        from sqlalchemy.exc import OperationalError, ProgrammingError

        try:
            return fn(*args, **kwargs)
        except (OperationalError, ProgrammingError) as e:
            if queue_table_missing(e):
                raise ValueError(QUEUE_MISSING) from e
            raise
    return wrapper


QUEUE_MISSING = ("the origin_submissions table is not deployed: apply pantry-db "
                 "migration 0006 before using the review queue")


def queue_table_missing(exc: BaseException) -> bool:
    """True when a DB error means `origin_submissions` does not exist.

    SQLite says "no such table", Postgres raises UndefinedTable ("relation
    ... does not exist"). The API and pantry-db ship through separate CI
    bumps, so an API that knows about the review queue can run against a
    database that does not have it yet.
    """
    text = str(exc).lower()
    return "origin_submissions" in text and (
        "no such table" in text or "does not exist" in text or "undefinedtable" in text)

@_queue_guard
def submit_origin(record: dict, *, submitted_by: str) -> dict:
    """Queue a label reading for review. Returns the row plus `duplicate`.

    Validates every field by name, canonicalises the country ("usa" →
    "United States") and derives the two origin fields exactly as
    ingest_label_json does: a full claim asserts ingredient content too, a
    processing claim only says where it was processed.

    Dedupes on (product, claim, country, verbatim, importer_only): a
    pending twin is returned with duplicate=True; a rejected twin is
    returned as-is (status rejected, review_note) so the agent sees the
    earlier verdict instead of re-queueing the same words; an approved twin
    is returned with duplicate=True because it is already evidence.
    """
    from sqlalchemy.orm import Session

    from . import db
    from .origins import _title, canonical_country, validate_countries

    try:
        product_id = int(record.get("product_id"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise ValueError("product_id: an integer product id is required") from None
    claim_type = str(record.get("claim_type") or "").strip().lower()
    if claim_type not in CLAIM_TYPES:
        raise ValueError(f"claim_type: {claim_type!r} is not one of {', '.join(CLAIM_TYPES)}")
    raw_country = _bounded(record, "country", COUNTRY_MAX).strip()
    if not raw_country:
        raise ValueError("country: a country name is required")
    unknown = validate_countries([raw_country])
    if unknown:
        hints = ", ".join(unknown[raw_country]) or "no close match"
        raise ValueError(f"country: {raw_country!r} is not recognised (did you mean: {hints})")
    country = _title(canonical_country(raw_country))
    verbatim = str(record.get("verbatim") or "").strip()
    if not VERBATIM_MIN <= len(verbatim) <= VERBATIM_MAX:
        raise ValueError(f"verbatim: {VERBATIM_MIN}-{VERBATIM_MAX} characters of the exact "
                         f"printed wording, got {len(verbatim)}")
    confidence = str(record.get("confidence") or "medium").strip().lower()
    if confidence not in CONFIDENCES:
        raise ValueError(f"confidence: {confidence!r} is not one of {', '.join(CONFIDENCES)}")
    note = _bounded(record, "note", NOTE_MAX).strip()
    source_ref = _bounded(record, "source_ref", SOURCE_REF_MAX).strip()
    importer_only = bool(record.get("importer_only"))

    with Session(db.engine()) as s:
        product = s.get(db.ProductRow, product_id)
        if product is None:
            raise ValueError(f"product_id: unknown product id {product_id}")
        product_name = str(product.name)
        twin = (s.query(db.OriginSubmissionRow)
                .filter_by(product_id=product_id, claim_type=claim_type,
                           country=country, verbatim=verbatim,
                           importer_only=importer_only)
                .order_by(db.OriginSubmissionRow.id.desc())
                .first())
        if twin is not None:
            return _submission_dict(twin, product_name, duplicate=True)
        row = db.OriginSubmissionRow(
            product_id=product_id, claim_type=claim_type, country=country,
            ingredient_origin=country if claim_type in FULL_CLAIMS else "",
            manufactured_in=country, verbatim=verbatim, confidence=confidence,
            importer_only=importer_only, note=note, source_ref=source_ref,
            submitted_by=submitted_by, submitted_at=_now(), status="pending")
        s.add(row)
        s.commit()
        s.refresh(row)
        return _submission_dict(row, product_name)


@_queue_guard
def review_submission(submission_id: int, decision: str, *, reviewed_by: str,
                      note: str = "") -> dict:
    """Approve or reject a pending submission.

    Approval copies the claim into product_origin_evidence as source
    "agent-label" (so the resolver treats it like a transcribed label),
    refreshes the product's resolved summary, and records the evidence row's
    id on the submission. The submission itself is never moved or deleted:
    it is the audit trail. A rejection needs a note a later reader can act
    on. Reviewing a non-pending row is an error naming its current status.
    """
    from sqlalchemy.orm import Session

    from . import db

    decision = (decision or "").strip().lower()
    if decision not in {"approve", "reject"}:
        raise ValueError(f"decision: {decision!r} must be 'approve' or 'reject'")
    note = note.strip()
    if len(note) > NOTE_MAX:
        raise ValueError(f"note: at most {NOTE_MAX} characters, got {len(note)}")
    if decision == "reject" and len(note) < REJECT_NOTE_MIN:
        raise ValueError(f"note: a rejection needs a review note of at least "
                         f"{REJECT_NOTE_MIN} characters saying why")

    with Session(db.engine()) as s:
        row = s.get(db.OriginSubmissionRow, submission_id)
        if row is None:
            raise ValueError(f"Unknown submission id {submission_id}")
        if str(row.status) != "pending":
            raise ValueError(f"Submission {submission_id} is already {row.status}"
                             f" (reviewed by {row.reviewed_by or 'unknown'})")
        product = s.get(db.ProductRow, int(row.product_id))
        product_name = str(product.name) if product is not None else ""
        product_id = int(row.product_id)
        evidence_id = None
        if decision == "approve":
            rec = {
                "product_id": product_id, "source": "agent-label",
                "source_ref": str(row.source_ref or "") or f"submission:{submission_id}",
                "claim_type": str(row.claim_type), "verbatim": str(row.verbatim),
                "ingredient_origin": str(row.ingredient_origin or ""),
                "manufactured_in": str(row.manufactured_in or ""),
                "confidence": str(row.confidence),
                "importer_only": bool(row.importer_only),
                "note": (f"{row.note or ''} [submission {submission_id} by "
                         f"{row.submitted_by}; approved by {reviewed_by}]").strip(),
                "observed_at": _now(),
            }
            db.save_origin_evidence([rec])
            refresh_resolved([product_id])
            ev = (s.query(db.ProductOriginEvidenceRow)
                  .filter_by(**{k: rec[k] for k in db._EVIDENCE_KEY})
                  .order_by(db.ProductOriginEvidenceRow.id.desc())
                  .first())
            evidence_id = int(ev.id) if ev is not None else None
        # setattr, as save_resolved_origins does: db.py's legacy Column
        # declarations type the attributes as Column[...] under mypy.
        for field, value in {
            "status": "approved" if decision == "approve" else "rejected",
            "reviewed_by": reviewed_by, "reviewed_at": _now(),
            "review_note": note, "evidence_id": evidence_id,
        }.items():
            setattr(row, field, value)
        s.add(row)
        s.commit()
        s.refresh(row)
        return _submission_dict(row, product_name)


@_queue_guard
def list_submissions(status: str | None = "pending", limit: int = 50,
                     offset: int = 0) -> tuple[list[dict], int]:
    """A page of submissions (oldest first — it is a queue) and the total
    matching `status` (None = every status)."""
    from sqlalchemy import func
    from sqlalchemy.orm import Session

    from . import db

    if status is not None and status not in SUBMISSION_STATUSES:
        raise ValueError(f"status: {status!r} is not one of {', '.join(SUBMISSION_STATUSES)}")
    with Session(db.engine()) as s:
        q = s.query(db.OriginSubmissionRow)
        if status is not None:
            q = q.filter(db.OriginSubmissionRow.status == status)
        total = int(q.with_entities(func.count(db.OriginSubmissionRow.id)).scalar() or 0)
        rows = q.order_by(db.OriginSubmissionRow.id).offset(offset).limit(limit).all()
        ids = {int(r.product_id) for r in rows}
        names: dict[int, str] = {}
        if ids:
            names = {int(p.id): str(p.name) for p in
                     s.query(db.ProductRow).filter(db.ProductRow.id.in_(ids)).all()}
        return [_submission_dict(r, names.get(int(r.product_id), "")) for r in rows], total


def pending_submission_count(product_id: int) -> int | None:
    """Pending review-queue rows for a product, or None when the queue table
    is not deployed yet.

    A product lookup must never fail because the review queue is missing,
    and a missing table must not read as 0 — 0 would claim there is nothing
    to review. None is the honest answer: unknown.
    """
    from sqlalchemy import func
    from sqlalchemy.exc import OperationalError, ProgrammingError
    from sqlalchemy.orm import Session

    from . import db

    try:
        with Session(db.engine()) as s:
            return int(s.query(func.count(db.OriginSubmissionRow.id))
                       .filter(db.OriginSubmissionRow.product_id == product_id,
                               db.OriginSubmissionRow.status == "pending").scalar() or 0)
    except (OperationalError, ProgrammingError) as e:
        if queue_table_missing(e):
            return None
        raise


def submission_counts_by_status() -> dict[str, int] | None:
    """Review-queue size by status, or None when the table is not deployed."""
    from sqlalchemy import func
    from sqlalchemy.exc import OperationalError, ProgrammingError
    from sqlalchemy.orm import Session

    from . import db

    try:
        with Session(db.engine()) as s:
            rows = (s.query(db.OriginSubmissionRow.status, func.count(db.OriginSubmissionRow.id))
                    .group_by(db.OriginSubmissionRow.status).all())
    except (OperationalError, ProgrammingError) as e:
        if queue_table_missing(e):
            return None
        raise
    counts = {str(status): int(n) for status, n in rows}
    return {st: counts.get(st, 0) for st in ("pending", "approved", "rejected")}


# ─── CLI ─────────────────────────────────────────────────────

def _flag(argv: list[str], name: str, default: str) -> str:
    return argv[argv.index(name) + 1] if name in argv and argv.index(name) + 1 < len(argv) \
        else default


def main(argv: list[str]) -> int:
    usage = ("Usage: python -m pantry_planner.ingest "
             "{origin|label|grocery} <file>  [--run-id ID]\n"
             "       python -m pantry_planner.ingest refresh\n"
             "       python -m pantry_planner.ingest submissions [pending|approved|rejected]\n"
             "       python -m pantry_planner.ingest review <id> approve|reject "
             "[--note TEXT] [--by NAME]\n\n"
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
    if cmd == "submissions":
        status = argv[1] if len(argv) > 1 else "pending"
        try:
            items, total = list_submissions(None if status == "all" else status, limit=1000)
        except ValueError as e:
            print(str(e), file=sys.stderr)
            return 2
        print(json.dumps({"items": items, "total": total}, indent=2))
        return 0
    if cmd == "review":
        if len(argv) < 3:
            print(usage, file=sys.stderr)
            return 2
        try:
            out = review_submission(int(argv[1]), argv[2],
                                    reviewed_by=_flag(argv, "--by", getpass.getuser()),
                                    note=_flag(argv, "--note", ""))
        except ValueError as e:
            print(str(e), file=sys.stderr)
            return 2
        print(json.dumps(out, indent=2))
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

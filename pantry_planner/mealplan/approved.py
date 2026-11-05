"""Approved trips: fingerprints, the needs_review diff and price changes since approval.

The shopper approves a trip; the console keeps {date, fingerprint, strategy, snapshot}. The
engine never rewrites it. On every schedule call the trip is recomputed, and:

- the fingerprint is sha256 of the date and the sorted (product_id, packs, storage) lines.
  Price is not in it, so a price change never moves a trip out of approved;
- a different fingerprint makes the trip needs_review, with the diff against the snapshot;
- each line's price_delta is (unit price now - unit price at approval) x packs now (demo
  prices), shown on the line and summed on the trip. The snapshot keeps the line's price for
  all its packs, so the unit price at approval is price_at_approval / packs: a changed pack
  count is a diff, never a price change;
- a line whose product has no offer in range any more makes the trip needs_review whatever
  the fingerprint says (no_longer_stocked).
"""
from __future__ import annotations

import datetime as dt
import hashlib

from .models import ApprovedTrip, TripDiff, TripLine


def fingerprint(date: dt.date, lines: list[tuple[int, int | None, str]]) -> str:
    """sha256 of 'YYYY-MM-DD|pid:packs:storage;...' over the sorted lines (packs '?' when
    unknown)."""
    body = ";".join(f"{pid}:{'?' if packs is None else packs}:{storage}"
                    for pid, packs, storage in sorted(lines, key=lambda x: (x[0], x[2])))
    return hashlib.sha256(f"{date.isoformat()}|{body}".encode()).hexdigest()


def trip_fingerprint(date: dt.date, lines: list[TripLine]) -> str:
    return fingerprint(date, [(ln.product.id, ln.packs, ln.storage) for ln in lines])


def _packs_text(packs: int | None) -> str:
    return "?" if packs is None else str(packs)


def diff(approved: ApprovedTrip, lines: list[TripLine], names: dict[int, str]) -> TripDiff:
    """Lines added, removed or changed since approval, keyed by (product, storage)."""
    before = {(s.product_id, s.storage): s for s in approved.snapshot}
    now = {(ln.product.id, ln.storage): ln for ln in lines}
    out = TripDiff()
    for key in sorted(now.keys() - before.keys()):
        ln = now[key]
        out.added.append({"product_id": key[0], "name": ln.product.name, "packs": ln.packs,
                          "storage": key[1]})
        out.text.append(f"+{_packs_text(ln.packs)} {ln.product.name} ({key[1]})")
    for key in sorted(before.keys() - now.keys()):
        s = before[key]
        name = names.get(key[0], f"product {key[0]}")
        out.removed.append({"product_id": key[0], "name": name, "packs": s.packs,
                            "storage": key[1]})
        out.text.append(f"-{_packs_text(s.packs)} {name} ({key[1]})")
    for key in sorted(now.keys() & before.keys()):
        s, ln = before[key], now[key]
        if s.packs != ln.packs:
            out.changed.append({"product_id": key[0], "name": ln.product.name,
                                "packs_before": s.packs, "packs_after": ln.packs,
                                "storage": key[1]})
            out.text.append(f"{ln.product.name}: {_packs_text(s.packs)} -> "
                            f"{_packs_text(ln.packs)} packs")
    return out


def apply_prices(approved: ApprovedTrip, lines: list[TripLine]) -> float | None:
    """Set price_at_approval and price_delta on each line the snapshot priced; return the
    trip's summed delta (None when no line has one).

    price_at_approval is the line's price for all its packs then, so the unit price then is
    price_at_approval / packs then, and the delta is price now - unit then x packs now:
    (unit now - unit then) x packs now. Buying more or fewer packs at the same prices is no
    price change. A line whose packs are unknown or 0, then or now, has no delta."""
    snap = {(s.product_id, s.storage): s for s in approved.snapshot}
    total, any_delta = 0.0, False
    for ln in lines:
        s = snap.get((ln.product.id, ln.storage))
        ln.price_at_approval = None if s is None else s.price_at_approval
        if s is None or s.price_at_approval is None or not s.packs or ln.price is None \
                or not ln.packs:
            continue
        unit_then = s.price_at_approval / s.packs
        ln.price_delta = round(ln.price - unit_then * ln.packs, 2) + 0.0
        total += ln.price_delta
        any_delta = True
    return round(total, 2) + 0.0 if any_delta else None


def money(delta: float) -> str:
    return f"{'+' if delta >= 0 else '-'}${abs(delta):.2f}"

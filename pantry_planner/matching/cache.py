"""The matcher's model answers, kept in the database (pantry-db 0010_matching_cache): an
ingredient name is described once and an identity string embedded once, then read from here
by every instance, across deploys. In front of the database sits a small in-process memo, so
a name asked twice in one process costs no query either.

Why a database and not a file or a semantic cache: the API has no disk in the deployment and
runs as more than one instance, and an answer for "butter" must never serve "peanut butter",
so only an exact key (a hash of the model, its version and the normalized text) ever hits.

Budget: Gemini's free tier allows 1,000 embedding texts and a few hundred Flash calls a day.
`Budget` counts this process's and every other instance's requests per UTC day in
matching_usage, and `spend` refuses once a kind's daily allowance (MATCHING_DAILY_EMBED_TEXTS,
MATCHING_DAILY_DESCRIBE_CALLS) is used: the caller then falls back (describe.py, embed.py) and
/health says so. A failed request is not cached and not counted twice.

Single flight: two requests for the same uncached key in one process share one model call
(a per-key lock); across instances the first row written wins (INSERT … ON CONFLICT DO
NOTHING), so a race costs at most one duplicate call, never a conflicting answer.

When the tables are missing (a database older than 0010) the memo alone serves, and `state`
says so; nothing fails.
"""
from __future__ import annotations

import collections
import hashlib
import json
import threading
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from .. import db

MEMO_SIZE = 4096
KINDS = ("embed_texts", "describe_calls")


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def today() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d")


def _insert_ignore(session: Session, table, rows: list[dict]) -> None:
    """INSERT … ON CONFLICT DO NOTHING, on SQLite and Postgres alike."""
    if not rows:
        return
    dialect = session.get_bind().dialect.name
    if dialect == "postgresql":
        from sqlalchemy.dialects.postgresql import insert
    else:
        from sqlalchemy.dialects.sqlite import insert
    session.execute(insert(table).values(rows).on_conflict_do_nothing())


class _Memo:
    """A bounded, most-recently-used-last dict, shared by the process."""

    def __init__(self, size: int = MEMO_SIZE):
        self._d: collections.OrderedDict[str, Any] = collections.OrderedDict()
        self._size = size
        self._lock = threading.Lock()

    def get(self, key: str) -> Any | None:
        with self._lock:
            if key in self._d:
                self._d.move_to_end(key)
                return self._d[key]
            return None

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            self._d[key] = value
            self._d.move_to_end(key)
            while len(self._d) > self._size:
                self._d.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._d.clear()

    def __len__(self) -> int:
        return len(self._d)


class Cache:
    """One kind of answer (identities or embeddings): the memo, then the table.

    `table` is the row class, `key_col` its key column name, `to_row` turns (key, value) into
    a row dict and `from_row` a row into the value."""

    def __init__(self, name: str, table, to_row: Callable[[str, Any], dict],
                 from_row: Callable[[Any], Any]):
        self.name = name
        self.table = table
        self.to_row = to_row
        self.from_row = from_row
        self.memo = _Memo()
        self.table_missing = False
        self.hits = self.misses = self.db_hits = 0
        self._locks: dict[str, threading.Lock] = {}
        self._locks_lock = threading.Lock()

    def lock(self, key: str) -> threading.Lock:
        with self._locks_lock:
            return self._locks.setdefault(key, threading.Lock())

    def get_many(self, keys: Iterable[str]) -> dict[str, Any]:
        """key -> value for every key the memo or the table has."""
        keys = list(dict.fromkeys(keys))
        out: dict[str, Any] = {}
        todo = []
        for k in keys:
            v = self.memo.get(k)
            if v is not None:
                out[k] = v
                self.hits += 1
            else:
                todo.append(k)
        if todo and not self.table_missing:
            try:
                with Session(db.engine()) as s:
                    rows = s.execute(select(self.table).where(
                        self.table.key.in_(todo))).scalars().all()
            except Exception as e:  # noqa: BLE001 - an older database serves from memory
                if db._table_missing(e, self.table.__tablename__):
                    self.table_missing = True
                    rows = []
                else:
                    raise
            for r in rows:
                v = self.from_row(r)
                out[r.key] = v
                self.memo.put(r.key, v)
                self.db_hits += 1
        self.misses += len([k for k in todo if k not in out])
        return out

    def put_many(self, items: dict[str, Any]) -> None:
        """Keep every (key, value); a key already there (another instance wrote it) keeps
        its row."""
        for k, v in items.items():
            self.memo.put(k, v)
        if not items or self.table_missing:
            return
        try:
            with Session(db.engine()) as s:
                _insert_ignore(s, self.table, [self.to_row(k, v) for k, v in items.items()])
                s.commit()
        except Exception as e:  # noqa: BLE001
            if db._table_missing(e, self.table.__tablename__):
                self.table_missing = True
            else:
                raise

    def state(self) -> dict:
        return {"memo": len(self.memo), "hits": self.hits, "db_hits": self.db_hits,
                "misses": self.misses, "table": "missing" if self.table_missing else "ok"}


class BudgetExceededError(RuntimeError):
    """A kind's daily allowance of model requests is spent; the message says when it resets."""


class Budget:
    """Model requests per UTC day and kind, counted in matching_usage (every instance) with
    an in-process copy, against the allowances the settings give."""

    def __init__(self, allowances: dict[str, int]):
        self.allowances = allowances
        self._local: dict[tuple[str, str], int] = {}
        self._lock = threading.Lock()
        self.table_missing = False

    def used(self, kind: str) -> int:
        day = today()
        if not self.table_missing:
            try:
                with Session(db.engine()) as s:
                    row = s.get(db.MatchingUsageRow, (day, kind))
                    n = row.count if row is not None else 0
                with self._lock:
                    self._local[(day, kind)] = max(n, self._local.get((day, kind), 0))
                return self._local[(day, kind)]
            except Exception as e:  # noqa: BLE001
                if db._table_missing(e, db.MatchingUsageRow.__tablename__):
                    self.table_missing = True
                else:
                    raise
        with self._lock:
            return self._local.get((day, kind), 0)

    def left(self, kind: str) -> int:
        return max(0, self.allowances.get(kind, 0) - self.used(kind))

    def spend(self, kind: str, n: int) -> None:
        """Count `n` requests of `kind` today, or raise BudgetExceededError before any is made
        when they would pass the allowance (an allowance of 0 means no limit)."""
        allowance = self.allowances.get(kind, 0)
        if allowance and self.used(kind) + n > allowance:
            raise BudgetExceededError(
                f"today's allowance of {allowance} {kind.replace('_', ' ')} is spent "
                f"(resets at 00:00 UTC)")
        day = today()
        with self._lock:
            self._local[(day, kind)] = self._local.get((day, kind), 0) + n
        if self.table_missing:
            return
        try:
            with Session(db.engine()) as s:
                _insert_ignore(s, db.MatchingUsageRow, [{"day": day, "kind": kind, "count": 0}])
                s.execute(update(db.MatchingUsageRow)
                          .where(db.MatchingUsageRow.day == day, db.MatchingUsageRow.kind == kind)
                          .values(count=db.MatchingUsageRow.count + n))
                s.commit()
        except Exception as e:  # noqa: BLE001
            if db._table_missing(e, db.MatchingUsageRow.__tablename__):
                self.table_missing = True
            else:
                raise

    def state(self) -> dict:
        return {kind: {"used": self.used(kind), "allowance": self.allowances.get(kind, 0)}
                for kind in KINDS}


# ─── The two caches ──────────────────────────────────────────

def identity_key(prompt_version: int, model: str, fingerprint: str, normalized: str) -> str:
    raw = f"{prompt_version}|{model}|{fingerprint}|{normalized}"
    return hashlib.sha256(raw.encode()).hexdigest()


def embedding_key(model: str, dims: int, text: str) -> str:
    return hashlib.sha256(f"{model}|{dims}|{text}".encode()).hexdigest()


def _identity_row(key: str, v: dict) -> dict:
    return {"key": key, "name": v["name"], "normalized": v["normalized"],
            "product_type": v["product_type"], "form": v["form"], "variant": v["variant"],
            "look_alikes": json.dumps(list(v["look_alikes"])), "identity": v["identity"],
            "model": v["model"], "prompt_version": v["prompt_version"],
            "catalog_fingerprint": v["fingerprint"], "status": "auto", "created_at": _now()}


def _identity_value(r) -> dict:
    return {"name": r.name, "normalized": r.normalized, "product_type": r.product_type,
            "form": r.form, "variant": r.variant, "look_alikes": tuple(json.loads(r.look_alikes)),
            "identity": r.identity, "model": r.model, "prompt_version": r.prompt_version,
            "fingerprint": r.catalog_fingerprint, "status": r.status}


def _embedding_row(key: str, v: dict) -> dict:
    return {"key": key, "model": v["model"], "dims": v["dims"], "text": v["text"],
            "vector": json.dumps(v["vector"]), "created_at": _now()}


def _embedding_value(r) -> dict:
    return {"model": r.model, "dims": r.dims, "text": r.text, "vector": json.loads(r.vector)}


identities = Cache("identities", db.IngredientIdentityRow, _identity_row, _identity_value)
embeddings = Cache("embeddings", db.EmbeddingCacheRow, _embedding_row, _embedding_value)
_budget: Budget | None = None
_budget_lock = threading.Lock()


def budget() -> Budget:
    global _budget
    with _budget_lock:
        if _budget is None:
            from ..config import settings

            cfg = settings()
            _budget = Budget({"embed_texts": cfg.matching_daily_embed_texts,
                              "describe_calls": cfg.matching_daily_describe_calls})
        return _budget


def reset() -> None:
    """Forget everything held in this process (tests; the tables are untouched)."""
    global _budget
    identities.memo.clear()
    embeddings.memo.clear()
    identities.table_missing = embeddings.table_missing = False
    identities.hits = identities.misses = identities.db_hits = 0
    embeddings.hits = embeddings.misses = embeddings.db_hits = 0
    with _budget_lock:
        _budget = None


def state() -> dict:
    return {"identities": identities.state(), "embeddings": embeddings.state(),
            "budget": budget().state()}

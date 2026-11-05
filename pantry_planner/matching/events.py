"""What shoppers were shown and what they picked, kept for learning to rank later.

Grocery platforms learn their substitute rankings from what shoppers accept and reject. This
catalog has no shopper history yet, so the ranking is a documented weighted sum
(matcher.WEIGHTS); these events are what a learned ranking would one day be trained on.

One JSON object per line, appended to <PANTRY_DATA_DIR>/matching/events.jsonl:

  shown     an Options ranking was served: the ingredient, how it was matched, and every
            row's product id, rank, group and score
  picked    the shopper chose a product for a line (a cart swap, or a meal plan's pin the
            first time it is seen in this process): the ingredient, the product, its rank and
            group in the last ranking shown for that ingredient, and the products ranked
            above it, which it was picked over (passed_over)

Only what matching needs: no shopper, address, location, recipe or plan is written, and the
time is to the minute. MATCHING_EVENTS=0 turns the log off; a data dir that cannot be written
only loses the log, never the request.
"""
from __future__ import annotations

import json
import threading
from collections import OrderedDict
from datetime import UTC, datetime
from pathlib import Path

_lock = threading.Lock()
_last_shown: OrderedDict[str, list[dict]] = OrderedDict()
_picked: OrderedDict[tuple, None] = OrderedDict()
_REMEMBER = 512


def _path() -> Path | None:
    from ..config import settings

    cfg = settings()
    if not cfg.matching_events:
        return None
    return Path(cfg.data_dir).expanduser() / "matching" / "events.jsonl"


def _key(ingredient: str) -> str:
    return " ".join(ingredient.casefold().split())


def _write(event: dict) -> None:
    path = _path()
    if path is None:
        return
    event = {"v": 1, "at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%MZ"), **event}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with _lock, path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, separators=(",", ":")) + "\n")
    except OSError:
        pass


def _remember(store: OrderedDict, key, value) -> None:
    store[key] = value
    store.move_to_end(key)
    while len(store) > _REMEMBER:
        store.popitem(last=False)


def shown(ranking) -> None:
    """An AlternativeRanking was served."""
    items = [{"product_id": it.product_id, "rank": it.rank, "group": it.group,
              "score": it.score, "current": it.current} for it in ranking.items]
    with _lock:
        _remember(_last_shown, _key(ranking.ingredient), items)
    _write({"event": "shown", "ingredient": ranking.ingredient, "matching": ranking.matching,
            "items": items, "filtered": [f.product_id for f in ranking.filtered]})


def picked(ingredient: str, product_id: int, *, was: int | None = None,
           once: bool = False) -> None:
    """The shopper chose `product_id` for an ingredient (in place of `was`). With `once`, a
    pick this process has logged already (a meal plan sends its pins on every request) is
    not logged again."""
    key = (_key(ingredient), product_id, was)
    with _lock:
        if once and key in _picked:
            return
        _remember(_picked, key, None)
        last = list(_last_shown.get(_key(ingredient), []))
    row = next((it for it in last if it["product_id"] == product_id), None)
    _write({"event": "picked", "ingredient": ingredient, "product_id": product_id, "was": was,
            "rank": row["rank"] if row else None, "group": row["group"] if row else None,
            "passed_over": [{"product_id": it["product_id"], "rank": it["rank"],
                             "group": it["group"]}
                            for it in last if row is not None and it["rank"] < row["rank"]]})


def clear() -> None:
    with _lock:
        _last_shown.clear()
        _picked.clear()

"""PRE-PROCESSING: schema linking — the live category/subcategory vocabulary.

One cached DISTINCT query; the resulting map {value -> level} both feeds the
extractor prompt (so the model links to real values) and powers the
level-agnostic clamp in validate_parsed.
"""
from __future__ import annotations

from functools import lru_cache

from sqlalchemy import text
from sqlalchemy.orm import Session

from .. import db


@lru_cache(maxsize=1)
def db_vocab() -> dict[str, str]:
    """{'dairy': 'category', 'cheese': 'subcategory', ...}"""
    vocab: dict[str, str] = {}
    with Session(db.engine()) as s:
        rows = s.execute(text(
            "SELECT DISTINCT category, subcategory FROM products")).all()
    for cat, sub in rows:
        if cat:
            vocab.setdefault(cat.lower(), "category")
        if sub:
            vocab[sub.lower()] = "subcategory"   # subcategory wins on collision
    return vocab


def prompt_block(vocab: dict[str, str]) -> str:
    cats = sorted(v for v, lvl in vocab.items() if lvl == "category")
    subs = sorted(v for v, lvl in vocab.items() if lvl == "subcategory")
    return f"categories: {', '.join(cats)}\nsubcategories: {', '.join(subs)}"


@lru_cache(maxsize=1)
def subcat_parents() -> dict[str, str]:
    """{'cheese': 'dairy', ...} — for the scope-fallback widening rung."""
    parents: dict[str, str] = {}
    with Session(db.engine()) as s:
        rows = s.execute(text(
            "SELECT DISTINCT category, subcategory FROM products")).all()
    for cat, sub in rows:
        if cat and sub:
            parents[sub.lower()] = cat.lower()
    return parents


def clear_cache() -> None:   # tests + after reseeding
    db_vocab.cache_clear()
    subcat_parents.cache_clear()

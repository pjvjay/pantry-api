"""
Measure product matching: today's word matching against matching by meaning, on a golden set
a Gemini judge labelled. Writes evals/reports/substitutes.md.

The golden set (evals/substitutes_golden.json) has, for every distinct ingredient name of the
library and the demo starters, the top candidates of both methods (union, up to GOLDEN_DEPTH
each), each labelled as Amazon's Shopping Queries dataset (ESCI) labels query-product pairs:

  E  exact        the ingredient itself; a shopper accepts it as the item
  S  substitute   not the ingredient, but works instead in the recipe line
  C  complement   goes with it, cannot replace it
  I  irrelevant

scripts/matching_seeds.py golden made the labels, once (cached by the ingredient and its
candidates), from candidate_pools() below. Each method is then scored on its own ranking:

  nDCG@3  the gains of its first three products (E 1, S 0.1, C 0.01, I 0, as the ESCI paper
          weighs them), discounted by log2(rank + 1), over the best three the labels allow
  Hit@1   its first product is labelled E

The rankings are each method's Options order with no recipe amount, no location and no
origin preference, so only matching and price speak:

  words    the word fallback (alternatives' token levels, head word and same-aisle
           substitutes, the another-food tier): tier, closeness, price, catalog id
  meaning  the matcher (matching/matcher.py): group, score, catalog id

Run:  python -m evals.substitutes_eval      (no LLM calls, no network: the labels are
committed; it plans against the seeded database DB_URL names)
"""
from __future__ import annotations

import contextlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
GOLDEN = HERE / "substitutes_golden.json"
REPORTS = HERE / "reports"
GAINS = {"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0}
GOLDEN_DEPTH = 20
K = 3
GOLDEN_ABOUT = (
    "For every distinct ingredient name of the library and the demo starters, the top "
    "candidates of today's word matching and of matching by meaning (union, up to `depth` "
    "each, made by evals/substitutes_eval.candidate_pools), labelled by a Gemini judge "
    "(scripts/matching_seeds.py golden) as ESCI labels query-product pairs. labels: product "
    "id -> E (exact: the ingredient itself), S (substitute: works instead), C (complement: goes "
    "with it), I (irrelevant).")


@contextlib.contextmanager
def _by_words():
    """Every name matched by words while it lasts, as when no seed knows it and the embedder
    is down."""
    from pantry_planner.matching import matcher

    real = matcher.meanings

    def words(items):
        return [matcher.Meaning(name=n, source="words", form=(f or "").lower())
                for n, f in items]

    matcher.meanings = words
    try:
        yield
    finally:
        matcher.meanings = real


def rank_words(name: str, catalog: dict) -> list[int]:
    """Today's word matching for `name`: the Options candidates the word fallback finds, in
    its order with no amount, location or preference (tier, closeness, price, id)."""
    from sqlalchemy.orm import Session

    from pantry_planner import alternatives as alts
    from pantry_planner import db
    from pantry_planner.models import BasisLine, PlanBasis

    basis = PlanBasis(path="spec", recipe_slug="eval", recipe_name="eval",
                      lines=[BasisLine(line_no=1, name=name)])
    with _by_words(), Session(db.engine()) as s:
        found = alts._fetch_pools(s, basis, basis.lines, catalog, {})[1].found
    word = alts._strong_head(name)
    aisle = alts._aisle_of(word, None, found, catalog)

    def key(pid: int) -> tuple:
        p, tier = catalog[pid], alts._TIER[found[pid]]
        if tier == "same" and alts._another_food(p, word, aisle):
            tier = "other"
        return (alts._TIER_RANK[tier], alts.closeness(name, p), p.price, pid)

    return sorted(found, key=key)


def rank_meaning(name: str, catalog: dict) -> list[int]:
    """Matching by meaning for `name`: the matcher's candidates, looser matches included, in
    its order with no amount, location or preference (group, score, id)."""
    from pantry_planner.matching import matcher

    m = matcher.meaning_of(name)
    ranked = matcher.rank_pool(m, list(catalog.values()), catalog=catalog, looser=True)
    return [r.product_id for r in ranked]


def candidate_pools(depth: int = GOLDEN_DEPTH) -> dict[str, list[int]]:
    """name -> the union of both methods' first `depth` products, for the judge."""
    from pantry_planner import db
    from pantry_planner.matching.seeds import seed_ingredient_names

    catalog = {p.id: p for p in db.load_all_products()}
    out = {}
    for name in seed_ingredient_names():
        ids = rank_words(name, catalog)[:depth] + rank_meaning(name, catalog)[:depth]
        out[name] = sorted(set(ids))
    return out


def ndcg(ranked: list[int], labels: dict[int, str], k: int = K) -> float:
    def dcg(gains: list[float]) -> float:
        return sum(g / math.log2(i + 2) for i, g in enumerate(gains[:k]))

    ideal = dcg(sorted((GAINS[v] for v in labels.values()), reverse=True))
    if ideal == 0:
        return 0.0
    return dcg([GAINS.get(labels.get(pid, "I"), 0.0) for pid in ranked]) / ideal


def load_golden() -> dict[str, dict[int, str]]:
    data = json.loads(GOLDEN.read_text(encoding="utf-8"))
    return {name: {int(pid): lab for pid, lab in v["labels"].items()}
            for name, v in data["ingredients"].items()}


def score() -> dict:
    """Both methods on the committed labels: per ingredient, and the means."""
    from pantry_planner import db

    catalog = {p.id: p for p in db.load_all_products()}
    golden = load_golden()
    rows = []
    for name, labels in golden.items():
        row = {"name": name, "ideal": sum(1 for v in labels.values() if v == "E")}
        for method, rank in (("words", rank_words), ("meaning", rank_meaning)):
            ranked = rank(name, catalog)
            row[method] = {"ndcg": ndcg(ranked, labels),
                           "hit": bool(ranked) and labels.get(ranked[0]) == "E",
                           "top": [(pid, labels.get(pid, "?")) for pid in ranked[:K]]}
        rows.append(row)
    n = len(rows) or 1
    mean = {m: {"ndcg": sum(r[m]["ndcg"] for r in rows) / n,
                "hit": sum(r[m]["hit"] for r in rows) / n} for m in ("words", "meaning")}
    return {"rows": rows, "mean": mean, "n": len(rows), "catalog": catalog}


def main() -> int:
    s = score()
    catalog = s["catalog"]
    w, m = s["mean"]["words"], s["mean"]["meaning"]
    lines = [
        "# Substitutes eval: word matching against matching by meaning",
        "",
        f"{s['n']} ingredient names (every distinct one of the library and the demo starters),",
        "each method's first three products scored on labels a Gemini judge gave the union",
        "of both methods' top 20 (evals/substitutes_golden.json; ESCI gains E 1, S 0.1,",
        "C 0.01, I 0). No amount, location or origin preference: only matching and price.",
        "",
        "| Method | nDCG@3 | Hit@1 |",
        "|---|---|---|",
        f"| Words (today's fallback) | {w['ndcg']:.3f} | {w['hit']:.0%} |",
        f"| Meaning (EmbeddingGemma + Gemini seeds) | {m['ndcg']:.3f} | {m['hit']:.0%} |",
        "",
        "## Per ingredient",
        "",
        "| Ingredient | Words nDCG@3 | Words top 3 | Meaning nDCG@3 | Meaning top 3 |",
        "|---|---|---|---|---|",
    ]

    def top(t: list) -> str:
        return "; ".join(f"{catalog[pid].name} ({lab})" for pid, lab in t) or "none"

    for r in sorted(s["rows"], key=lambda r: (r["meaning"]["ndcg"] - r["words"]["ndcg"],
                                              r["name"])):
        lines.append(f"| {r['name']} | {r['words']['ndcg']:.2f} | {top(r['words']['top'])} | "
                     f"{r['meaning']['ndcg']:.2f} | {top(r['meaning']['top'])} |")
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "substitutes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"words   nDCG@3 {w['ndcg']:.3f}  Hit@1 {w['hit']:.0%}")
    print(f"meaning nDCG@3 {m['ndcg']:.3f}  Hit@1 {m['hit']:.0%}")
    print(f"wrote {REPORTS / 'substitutes.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

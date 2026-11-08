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
import random
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


# ─── How sure the means are ──────────────────────────────────

BOOTSTRAP = 10_000
SEED = 20261105


def paired_bootstrap(a: list[float], b: list[float], *, reps: int = BOOTSTRAP,
                     seed: int = SEED) -> dict:
    """95% intervals for the means of `a` and `b` and of b - a, resampling the ingredient
    names (paired: one draw picks the same names for both methods), and the share of draws
    where b does not beat a. Fixed seed, so the report is reproducible."""
    assert len(a) == len(b) and a
    rng = random.Random(seed)
    n = len(a)
    ma, mb, md = [], [], []
    for _ in range(reps):
        idx = [rng.randrange(n) for _ in range(n)]
        sa = sum(a[i] for i in idx) / n
        sb = sum(b[i] for i in idx) / n
        ma.append(sa)
        mb.append(sb)
        md.append(sb - sa)

    def ci(xs: list[float]) -> tuple[float, float]:
        xs = sorted(xs)
        return xs[int(0.025 * (len(xs) - 1))], xs[int(0.975 * (len(xs) - 1))]

    return {"a": ci(ma), "b": ci(mb), "diff": ci(md),
            "diff_mean": sum(b) / n - sum(a) / n,
            "p_not_better": sum(d <= 0 for d in md) / reps}


# ─── When the exact item is missing ──────────────────────────
# What Options are for: the ingredient's own products are out of stock or excluded, so the
# shopper takes the first thing offered instead. Every product the judge labelled E is taken
# out, the hard rule Options apply is applied (no product that adds an allergen or diet tag
# the exact item lacks; alternatives.rule), and the first product left is scored:
#   Sub@1    it is labelled S (works instead in the recipe line)
#   Safe@1   it adds no tag (always true after the rule; reported before it, to show what the
#            rule catches)
#   Offered  anything is left at all
# A product neither method ranked in its top 20 has no label and counts as not S.

GAINS_MISSING = {"S": 1.0, "C": 0.01, "I": 0.0}


def missing_exact(ranked: list[int], labels: dict[int, str], catalog: dict) -> dict:
    from pantry_planner.alternatives import rule

    exact = sorted(pid for pid, lab in labels.items() if lab == "E" and pid in catalog)
    left = [pid for pid in ranked if labels.get(pid) != "E"]
    if not exact:
        return {"applies": False}
    # The product the plan would have bought: this method's first exact product, else the
    # lowest-id one; the rule compares substitutes against what was actually missing.
    original = catalog[next((pid for pid in ranked if pid in exact), exact[0])]
    unsafe_first = bool(left) and rule(catalog[left[0]], original, set()) is not None
    safe = [pid for pid in left if rule(catalog[pid], original, set()) is None]
    first = safe[0] if safe else None
    return {"applies": True, "offered": first is not None,
            "sub": first is not None and labels.get(first) == "S",
            "unsafe_before_rule": unsafe_first,
            "first": (first, labels.get(first, "?")) if first is not None else None}


def score_missing(catalog: dict | None = None) -> dict:
    """Both methods with the exact items gone (missing_exact), per name and as shares."""
    from pantry_planner import db

    catalog = catalog or {p.id: p for p in db.load_all_products()}
    rows = []
    for name, labels in load_golden().items():
        row = {"name": name}
        for method, rank in (("words", rank_words), ("meaning", rank_meaning)):
            row[method] = missing_exact(rank(name, catalog), labels, catalog)
        if row["words"]["applies"]:
            rows.append(row)
    n = len(rows) or 1
    share = {m: {k: sum(r[m][k] for r in rows) / n
                 for k in ("sub", "offered", "unsafe_before_rule")}
             for m in ("words", "meaning")}
    return {"rows": rows, "share": share, "n": len(rows)}


def _uncertainty_lines(s: dict) -> list[str]:
    rows = s["rows"]
    nd = paired_bootstrap([r["words"]["ndcg"] for r in rows], [r["meaning"]["ndcg"] for r in rows])
    ht = paired_bootstrap([float(r["words"]["hit"]) for r in rows],
                          [float(r["meaning"]["hit"]) for r in rows])

    def iv(t: tuple[float, float], pct: bool = False) -> str:
        return f"{t[0]:.0%}–{t[1]:.0%}" if pct else f"{t[0]:.3f}–{t[1]:.3f}"

    return [
        "## How sure these numbers are",
        "",
        f"95% intervals from a paired bootstrap over the {s['n']} names ({BOOTSTRAP:,} draws,",
        "seed fixed). The difference is meaning minus words on the same names.",
        "",
        "| Metric | Words | Meaning | Difference | Draws where meaning is not better |",
        "|---|---|---|---|---|",
        f"| nDCG@3 | {iv(nd['a'])} | {iv(nd['b'])} | {nd['diff_mean']:+.3f} ({iv(nd['diff'])}) "
        f"| {nd['p_not_better']:.1%} |",
        f"| Hit@1 | {iv(ht['a'], True)} | {iv(ht['b'], True)} | {ht['diff_mean']:+.1%} "
        f"({iv(ht['diff'], True)}) | {ht['p_not_better']:.1%} |",
        "",
        "Caveats: the same Gemini family wrote the product profiles and judged the labels, and",
        "the matcher's weights were tuned on this set, so these are optimistic for new names.",
        "",
    ]


def _missing_lines(m: dict, catalog: dict) -> list[str]:
    w, me = m["share"]["words"], m["share"]["meaning"]
    lines = [
        "## When the exact item is missing",
        "",
        f"{m['n']} names with an exact product: every product labelled E removed, Options' hard",
        "rule applied (nothing that adds an allergen or diet tag the exact item lacks), and the",
        "first product left scored. Unlabelled products count as not a substitute.",
        "",
        "| Method | Sub@1 (first offer works instead) | Something offered "
        "| First offer would add an allergen without the rule |",
        "|---|---|---|---|",
        f"| Words | {w['sub']:.0%} | {w['offered']:.0%} | {w['unsafe_before_rule']:.0%} |",
        f"| Meaning | {me['sub']:.0%} | {me['offered']:.0%} | {me['unsafe_before_rule']:.0%} |",
        "",
        "| Ingredient | Words: first offer | Meaning: first offer |",
        "|---|---|---|",
    ]

    def first(x: dict) -> str:
        if not x["first"]:
            return "nothing"
        pid, lab = x["first"]
        return f"{catalog[pid].name} ({lab})"

    for r in sorted(m["rows"], key=lambda r: r["name"]):
        lines.append(f"| {r['name']} | {first(r['words'])} | {first(r['meaning'])} |")
    return lines + [""]


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
        *_uncertainty_lines(s),
        *_missing_lines(score_missing(catalog), catalog),
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
    ms = score_missing(catalog)["share"]
    print(f"exact missing: words Sub@1 {ms['words']['sub']:.0%}, "
          f"meaning Sub@1 {ms['meaning']['sub']:.0%}")
    print(f"wrote {REPORTS / 'substitutes.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Run the pipeline over the golden set. Score each selection. Write a report.

Golden set schema (evals/datasets/pairs.jsonl):
    {
      "recipe_slug": str,
      "expected": [
        {
          "line_no": int,
          "ingredient_name": str,
          "correct_product_ids":   [int],    # any → passing choice
          "preferred_product_ids": [int],    # (optional) top choice among the correct
          "wrong_product_ids":     [int],    # explicit anti-matches
          "why": str,                        # human explanation
        },
        ...
      ]
    }

Scoring, per selected line:
    * chosen ∈ preferred      → 1.00  "perfect"
    * chosen ∈ correct        → 0.70  "correct"  (if preferred set is nonempty; else 1.00)
    * chosen ∈ wrong          → 0.00  "wrong"
    * chosen not in any set   → 0.50  "unspecified"
    * chosen == None (missing) → 0.00  "missing"

Aggregate accuracy = mean of line scores across all recipes.

Run:
    python -m evals.run
    ROUTING_STRATEGY=three_phase python -m evals.run

Report lands in evals/reports/latest.md (and evals/reports/<strategy>.md).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from statistics import mean

from pantry_planner import flow
from pantry_planner.config import settings

GOLDEN = Path(__file__).parent / "datasets" / "pairs.jsonl"
REPORT_DIR = Path(__file__).parent / "reports"


def load_golden() -> list[dict]:
    cases = []
    with GOLDEN.open() as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def score_line(chosen_product_id: int | None, expected: dict) -> tuple[float, str]:
    """Score one selection against the golden expected."""
    if chosen_product_id is None:
        return 0.0, "missing"

    preferred = expected.get("preferred_product_ids") or []
    correct = expected.get("correct_product_ids") or []
    wrong = expected.get("wrong_product_ids") or []

    if preferred and chosen_product_id in preferred:
        return 1.0, "perfect"
    if chosen_product_id in correct:
        # If preferred was specified and chosen wasn't it, we're correct
        # but not preferred → discount slightly.
        return (0.7 if preferred else 1.0), "correct"
    if chosen_product_id in wrong:
        return 0.0, "wrong"
    return 0.5, "unspecified"


def run_case(case: dict) -> dict:
    t0 = time.perf_counter()
    plan = flow.run(case["recipe_slug"])
    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    line_verdicts = []
    for expected_line in case["expected"]:
        line_no = expected_line["line_no"]
        matching = next(
            (li for li in plan.line_items if li.line_no == line_no),
            None,
        )
        chosen_id = matching.product_id if matching else None

        score, verdict = score_line(chosen_id, expected_line)
        line_verdicts.append({
            "line_no": line_no,
            "ingredient_name": expected_line["ingredient_name"],
            "chosen_product_id": chosen_id,
            "chosen_product_name": matching.product_name if matching else "MISSING",
            "chosen_price": matching.price if matching else None,
            "confidence": matching.confidence if matching else None,
            "reasoning": matching.reasoning if matching else "",
            "score": score,
            "verdict": verdict,
            "why_expected": expected_line.get("why", ""),
        })

    line_scores = [v["score"] for v in line_verdicts]
    return {
        "recipe_slug": case["recipe_slug"],
        "recipe_name": plan.recipe_name,
        "avg_score": mean(line_scores) if line_scores else 0.0,
        "line_verdicts": line_verdicts,
        "total_cost": plan.total_cost,
        "total_llm_cost_usd": plan.total_llm_cost_usd,
        "total_latency_ms": elapsed_ms,
        "routing_strategy": plan.routing_strategy,
        "preselected_model": plan.preselected_model,
        "escalated": plan.escalated,
    }


def _emoji(verdict: str) -> str:
    return {
        "perfect": "✅",
        "correct": "☑️",
        "wrong": "❌",
        "unspecified": "❓",
        "missing": "🚫",
    }.get(verdict, "?")


def format_report(results: list[dict], strategy: str) -> str:
    overall = mean(r["avg_score"] for r in results) if results else 0.0
    total_llm = sum(r["total_llm_cost_usd"] for r in results)
    total_latency = sum(r["total_latency_ms"] for r in results)
    n_escalated = sum(1 for r in results if r["escalated"])

    lines = [
        f"# Eval Report — routing strategy: `{strategy}`",
        "",
        f"- **Recipes evaluated:** {len(results)}",
        f"- **Overall accuracy:** {overall:.2%}",
        f"- **Total LLM cost:** ${total_llm:.4f}",
        f"- **Total wall-clock latency:** {total_latency} ms",
        f"- **Escalated (cascade only):** {n_escalated}/{len(results)}",
        "",
        "## Per-recipe results",
        "",
    ]
    for r in results:
        lines.append(f"### {r['recipe_name']} (`{r['recipe_slug']}`)")
        lines.append("")
        lines.append(f"- Accuracy: **{r['avg_score']:.2%}**")
        lines.append(f"- Grocery total: ${r['total_cost']:.2f}")
        lines.append(f"- LLM cost: ${r['total_llm_cost_usd']:.4f}")
        lines.append(f"- Latency: {r['total_latency_ms']} ms")
        lines.append(f"- Preselected: `{r['preselected_model']}`  |  Escalated: {r['escalated']}")
        lines.append("")
        lines.append("| # | Ingredient | Chosen | $ | Conf | Score | Verdict |")
        lines.append("|---|---|---|---|---|---|---|")
        for v in r["line_verdicts"]:
            price = f"${v['chosen_price']:.2f}" if v["chosen_price"] is not None else "—"
            conf = f"{v['confidence']:.2f}" if v["confidence"] is not None else "—"
            lines.append(
                f"| {v['line_no']} | {v['ingredient_name']} | "
                f"{v['chosen_product_name']} `#{v['chosen_product_id']}` | "
                f"{price} | {conf} | {v['score']:.2f} | {_emoji(v['verdict'])} {v['verdict']} |"
            )
        lines.append("")

        # Failure detail — only show where we scored badly
        fails = [v for v in r["line_verdicts"] if v["score"] < 1.0]
        if fails:
            lines.append("**Notes on non-perfect selections:**")
            lines.append("")
            for v in fails:
                lines.append(
                    f"- **{v['ingredient_name']}** — "
                    f"model chose *{v['chosen_product_name']}* "
                    f"(reasoning: _{v['reasoning']}_). "
                    f"Expected: _{v['why_expected']}_"
                )
            lines.append("")

    return "\n".join(lines)


def main() -> int:
    cases = load_golden()
    strategy = settings().routing_strategy

    print(f"Running {len(cases)} cases with strategy={strategy}...")
    results = [run_case(c) for c in cases]

    report = format_report(results, strategy)

    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / "latest.md").write_text(report)
    (REPORT_DIR / f"{strategy}.md").write_text(report)

    print(report)
    print(f"\nReport written to {REPORT_DIR}/latest.md and {REPORT_DIR}/{strategy}.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

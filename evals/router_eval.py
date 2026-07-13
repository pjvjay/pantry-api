"""
Head-to-head comparison of cascade vs three-phase routers.

Runs the same golden set through BOTH strategies in one process,
then writes a comparison table showing per-recipe:
    - which model(s) each strategy chose
    - cost, latency, accuracy for each
    - the winner on each axis

The comparison is the money artifact: this is what shows an interviewer
"I built two routing strategies, measured them, and here's when each wins."

Run:
    python -m evals.router_eval

Report lands in evals/reports/router_comparison.md.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from statistics import mean

# We import lazily below so we can clear the settings() cache between runs.
REPORT_DIR = Path(__file__).parent / "reports"


def _run_strategy(strategy: str) -> list[dict]:
    """Run the whole golden set with the given strategy.

    settings() is @lru_cache'd, so we clear it after mutating the env var.
    """
    os.environ["ROUTING_STRATEGY"] = strategy

    from pantry_planner.config import settings
    settings.cache_clear()

    from evals.run import load_golden, run_case
    return [run_case(c) for c in load_golden()]


def _fmt_money(x: float) -> str:
    return f"${x:.4f}"


def _fmt_ms(ms: int) -> str:
    return f"{ms / 1000:.1f}s"


def _pct(x: float) -> str:
    return f"{x:.0%}"


def _winner(cascade: float, three_phase: float, *, lower_is_better: bool = True) -> str:
    """Return a short marker showing which strategy won on this metric."""
    if abs(cascade - three_phase) / max(cascade, three_phase, 1e-9) < 0.05:
        return "≈"    # within 5% — tie
    if lower_is_better:
        return "🟢 cascade" if cascade < three_phase else "🟠 3-phase"
    return "🟢 cascade" if cascade > three_phase else "🟠 3-phase"


def format_comparison(cascade: list[dict], three_phase: list[dict]) -> str:
    lines = [
        "# Router Comparison — cascade vs three-phase",
        "",
        "Head-to-head over the same golden set. Both strategies ran through",
        "the full pipeline; per-recipe stats below, aggregates at the bottom.",
        "",
        "## Per-recipe results",
        "",
        "| Recipe | Cascade model(s) | Cascade $ | Cascade time | Cascade acc | 3-phase model | 3-phase $ | 3-phase time | 3-phase acc |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    by_slug_c = {r["recipe_slug"]: r for r in cascade}
    by_slug_t = {r["recipe_slug"]: r for r in three_phase}

    for slug in by_slug_c.keys():
        c = by_slug_c[slug]
        t = by_slug_t[slug]

        cascade_models = c["preselected_model"].replace("claude-", "")
        if c["escalated"]:
            cascade_models += " → +sonnet"
        three_phase_model = t["preselected_model"].replace("claude-", "")

        lines.append(
            f"| {c['recipe_name']} | `{cascade_models}` | "
            f"{_fmt_money(c['total_llm_cost_usd'])} | "
            f"{_fmt_ms(c['total_latency_ms'])} | "
            f"{_pct(c['avg_score'])} | "
            f"`{three_phase_model}` | "
            f"{_fmt_money(t['total_llm_cost_usd'])} | "
            f"{_fmt_ms(t['total_latency_ms'])} | "
            f"{_pct(t['avg_score'])} |"
        )

    # Aggregates
    c_cost = sum(r["total_llm_cost_usd"] for r in cascade)
    c_latency = sum(r["total_latency_ms"] for r in cascade)
    c_acc = mean(r["avg_score"] for r in cascade) if cascade else 0.0

    t_cost = sum(r["total_llm_cost_usd"] for r in three_phase)
    t_latency = sum(r["total_latency_ms"] for r in three_phase)
    t_acc = mean(r["avg_score"] for r in three_phase) if three_phase else 0.0

    lines.extend([
        f"| **TOTAL** | — | **{_fmt_money(c_cost)}** | **{_fmt_ms(c_latency)}** | **{_pct(c_acc)}** | — | **{_fmt_money(t_cost)}** | **{_fmt_ms(t_latency)}** | **{_pct(t_acc)}** |",
        "",
        "## Winner per axis",
        "",
        "| Axis | Winner | Cascade | 3-phase | Delta |",
        "|---|---|---|---|---|",
        f"| Cost      | {_winner(c_cost, t_cost)} | {_fmt_money(c_cost)} | {_fmt_money(t_cost)} | {_fmt_money(abs(c_cost - t_cost))} |",
        f"| Latency   | {_winner(c_latency, t_latency)} | {_fmt_ms(c_latency)} | {_fmt_ms(t_latency)} | {_fmt_ms(abs(c_latency - t_latency))} |",
        f"| Accuracy  | {_winner(c_acc, t_acc, lower_is_better=False)} | {_pct(c_acc)} | {_pct(t_acc)} | {abs(c_acc - t_acc):.1%} |",
        "",
        "## Takeaway",
        "",
    ])

    if c_acc >= t_acc and c_cost < t_cost:
        lines.append(
            f"**On this dataset, cascade wins.** Same or better accuracy "
            f"({_pct(c_acc)} vs {_pct(t_acc)}) at "
            f"{c_cost/t_cost:.0%} of the cost and "
            f"{c_latency/t_latency:.0%} of the latency."
        )
        lines.append("")
        lines.append(
            "Three-phase would earn its keep on datasets with more variance — "
            "when Haiku fails often enough that the ~$0.001 classifier call "
            "prevents a $0.02 Sonnet escalation. On this golden set, Haiku "
            "handles most cases first-try, so cascade's opportunistic escalation "
            "is strictly cheaper than pre-classifying every request."
        )
    elif t_acc > c_acc:
        lines.append(
            f"**Three-phase wins on accuracy** ({_pct(t_acc)} vs {_pct(c_acc)}) "
            f"at a {t_cost/c_cost:.1f}× cost premium."
        )
    else:
        lines.append("Neither strategy dominates — inspect per-recipe rows for detail.")

    lines.append("")
    lines.append(f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}_")
    return "\n".join(lines)


def main() -> int:
    print("Running cascade...")
    cascade_results = _run_strategy("cascade")

    print("Running three_phase...")
    three_phase_results = _run_strategy("three_phase")

    report = format_comparison(cascade_results, three_phase_results)

    REPORT_DIR.mkdir(exist_ok=True)
    out = REPORT_DIR / "router_comparison.md"
    out.write_text(report)

    print(report)
    print(f"\nReport written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

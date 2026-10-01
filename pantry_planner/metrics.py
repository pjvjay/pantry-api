"""Prometheus metrics.

The app already computes everything worth alerting on — per-call model,
token counts, cost and latency in selector.py and query_parser.py, gate
aborts in the query planner, origin coverage per basket — and until now
exported none of it. /health told you the process was up. It could not
tell you that plan requests had started aborting, that LLM spend had
spiked, or that origin coverage had collapsed.

Deliberately small: counters and histograms over the boundaries that
actually fail, not a metric per function. Every series here is one a
runbook can act on.
"""
from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

REGISTRY = CollectorRegistry()

# ─── LLM spend ───────────────────────────────────────────────
# Cost is the one that surprises people, so it is a counter you can rate()
# rather than a gauge you have to catch at the right moment.
llm_calls = Counter(
    "pantry_llm_calls_total", "LLM calls made.",
    ["step", "model"], registry=REGISTRY)
llm_cost_usd = Counter(
    "pantry_llm_cost_usd_total", "Estimated USD spent on LLM calls.",
    ["step", "model"], registry=REGISTRY)
llm_tokens = Counter(
    "pantry_llm_tokens_total", "Tokens consumed.",
    ["step", "model", "direction"], registry=REGISTRY)
llm_latency = Histogram(
    "pantry_llm_latency_seconds", "LLM call latency.",
    ["step", "model"], registry=REGISTRY,
    buckets=(0.5, 1, 2, 5, 10, 20, 40, 60, 120))

# ─── Planning outcomes ───────────────────────────────────────
# A gate abort is a normal, correct outcome — but a CHANGE in its rate
# means the catalog or the constraints moved, which is worth waking up for.
plans = Counter(
    "pantry_plans_total", "Plan requests by path and outcome.",
    ["path", "outcome"], registry=REGISTRY)
plan_gates = Counter(
    "pantry_plan_gate_aborts_total", "Plan aborts by gate code.",
    ["code"], registry=REGISTRY)

# ─── Provenance ──────────────────────────────────────────────
# The metric the whole origin feature rests on: if coverage collapses, the
# filter silently stops protecting anyone, because absence is not a verdict
# and unverified products keep shipping.
origin_coverage_spend = Gauge(
    "pantry_origin_coverage_spend_ratio",
    "Spend-weighted origin coverage of the most recent basket.",
    registry=REGISTRY)
origin_excluded = Counter(
    "pantry_origin_excluded_total",
    "Candidates removed because they were evidenced as an excluded origin.",
    registry=REGISTRY)
origin_evidence_rows = Gauge(
    "pantry_origin_evidence_rows", "Origin evidence rows in the database.",
    registry=REGISTRY)


def record_llm(span: dict) -> None:
    """Record one LLM call from the span tracing.py already builds."""
    step = span.get("step", "unknown")
    model = span.get("model", "unknown")
    llm_calls.labels(step, model).inc()
    llm_cost_usd.labels(step, model).inc(float(span.get("cost_usd", 0.0)))
    llm_tokens.labels(step, model, "input").inc(int(span.get("input_tokens", 0)))
    llm_tokens.labels(step, model, "output").inc(int(span.get("output_tokens", 0)))
    llm_latency.labels(step, model).observe(
        float(span.get("latency_ms", 0)) / 1000.0)


def record_plan(path: str, outcome: str, *, gate: str | None = None) -> None:
    plans.labels(path, outcome).inc()
    if gate:
        plan_gates.labels(gate).inc()


def record_coverage(coverage) -> None:
    if coverage is None:
        return
    origin_coverage_spend.set(float(coverage.spend_fraction))
    if coverage.lines_excluded_origin:
        origin_excluded.inc(coverage.lines_excluded_origin)


def refresh_db_gauges() -> None:
    """Sampled at scrape time — cheap counts, not a hot path."""
    from sqlalchemy.orm import Session

    from .db import ProductOriginEvidenceRow, engine

    try:
        with Session(engine()) as s:
            origin_evidence_rows.set(s.query(ProductOriginEvidenceRow).count())
    except Exception:
        # A metrics endpoint must never be the thing that takes the app down.
        pass

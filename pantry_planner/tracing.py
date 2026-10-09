"""
Burr tracing configuration.

Uses Burr's LocalTrackingClient — writes every state transition to a
local SQLite DB that the `burr` CLI can display.

LLM-specific metadata (model, tokens, cost, latency) is attached to
individual actions by convention: any action that makes an LLM call
writes a dict-shaped field to state under the key `llm_call_metadata_<step>`.
That way the Burr UI shows a clear "here's the LLM call and its cost"
section for each traced run.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from burr.lifecycle import PostRunStepHook, PreRunStepHook
from burr.tracking import LocalTrackingClient

APP_NAME = "pantry-planner"
PROJECT_NAME = "pantry-planner"

# The tracking DB lands next to the app by default (volume-mounted in
# docker-compose). BURR_TRACKING_DIR overrides it — the stdio MCP entry
# point sets a home-dir path because MCP clients launch servers with an
# arbitrary (possibly read-only) cwd.
TRACKING_DB_DIR = Path(os.environ.get("BURR_TRACKING_DIR", ".burr"))


class StepTimer(PreRunStepHook, PostRunStepHook):
    """Times every action of one plan call, in order: the plan's `pipeline`, so a client sees
    where pantry spent the call (selection, trip optimizer ...) without opening the Burr UI."""

    def __init__(self) -> None:
        self.steps: list[dict[str, Any]] = []
        self._started: dict[int, float] = {}

    def pre_run_step(self, *, sequence_id: int, **_: Any) -> None:
        self._started[sequence_id] = time.perf_counter()

    def post_run_step(self, *, action: Any, sequence_id: int, exception: Exception | None,
                      **_: Any) -> None:
        started = self._started.pop(sequence_id, None)
        self.steps.append({
            "step": action.name,
            "ms": None if started is None else round((time.perf_counter() - started) * 1000, 1),
            "error": type(exception).__name__ if exception else None})


def make_tracker() -> LocalTrackingClient:
    """Create the tracking client. Each Burr Application should use this."""
    TRACKING_DB_DIR.mkdir(parents=True, exist_ok=True)
    return LocalTrackingClient(
        project=PROJECT_NAME,
        storage_dir=str(TRACKING_DB_DIR),
    )


def llm_span(
    *,
    step: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
    latency_ms: int,
    retry_count: int = 0,
    http: dict | None = None,
) -> dict:
    """
    Shape of the metadata dict that LLM-calling actions write to state.

    Kept as a helper so all LLM actions record the same shape — makes
    the Burr UI (and any downstream analysis) uniform. `http` is the call's
    httptrace record (Gemini): its attempts, phase by phase, so the Burr UI
    shows where the latency went; it also sets retry_count to the HTTP
    retries the call made.
    """
    span = {
        "step": step,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": round(cost_usd, 6),
        "latency_ms": latency_ms,
        "retry_count": retry_count,
    }
    if http:
        span["retry_count"] = max(retry_count, len(http.get("attempts") or []) - 1)
        span["http"] = http
    return span


def _server(http: dict) -> str:
    return "Google" if http.get("provider") == "gemini" else "the server"


def llm_call_trace(step: str, http: dict):
    """The plan's (and the MCP summary's) record of one LLM call."""
    from .httptrace import phase_list
    from .models import LlmCallTrace
    attempts = http.get("attempts") or []
    last = attempts[-1] if attempts else {}
    return LlmCallTrace(step=step, model=str(http.get("model", "")),
                        total_ms=int(http.get("total_ms") or 0), attempts=len(attempts),
                        status=last.get("status"), server_ms=last.get("server_ms"),
                        phases=phase_list(http, _server(http)))


def llm_step(span: dict):
    """The plan trace's step for one LLM call: its phases as the timeline's bar and the
    per-attempt breakdown as the step's detail."""
    from .httptrace import describe, phase_list
    from .nlsearch.plan import StepKind, StepResult
    http = span.get("http")
    label = span["model"]
    if span["input_tokens"] or span["output_tokens"]:
        label += f" · {span['input_tokens']:,} in / {span['output_tokens']:,} out tokens"
    if http:
        server = _server(http)
        phases = phase_list(http, server)
        wait = sum(p["ms"] for p in phases if p["name"] == f"waiting for {server}")
        own = sum(a.get("server_ms") or 0 for a in http.get("attempts") or [])
        label += f" · waiting for {server} {wait / 1000:.1f} s"
        if own:
            label += f" ({server} reports {own / 1000:.1f} s)"
        if len(http.get("attempts") or []) > 1:
            label += f" · {len(http['attempts'])} attempts"
        detail = describe(http, server)
    else:
        phases, detail = [], "No HTTP trace for this call (Anthropic, or demo mode)."
    return StepResult(step_id=f"llm_{span['step']}", kind=StepKind.llm, label=label,
                      sql_display=detail, duration_ms=int(span["latency_ms"]),
                      phases=phases)

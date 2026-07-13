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

from pathlib import Path

from burr.tracking import LocalTrackingClient

APP_NAME = "pantry-planner"
PROJECT_NAME = "pantry-planner"

# The tracking DB lands next to the app. Volume-mounted in docker-compose.
TRACKING_DB_DIR = Path(".burr")
TRACKING_DB_DIR.mkdir(exist_ok=True)


def make_tracker() -> LocalTrackingClient:
    """Create the tracking client. Each Burr Application should use this."""
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
) -> dict:
    """
    Shape of the metadata dict that LLM-calling actions write to state.

    Kept as a helper so all LLM actions record the same shape — makes
    the Burr UI (and any downstream analysis) uniform.
    """
    return {
        "step": step,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": round(cost_usd, 6),
        "latency_ms": latency_ms,
        "retry_count": retry_count,
    }

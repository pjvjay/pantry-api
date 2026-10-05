"""Where a Gemini call's time goes: every attempt timed phase by phase (DNS, connect, TLS,
upload, waiting for Google, download, Google's own server-timing, the wait before a retry),
logged, and carried into the Burr span, the plan trace, the plan and the MCP summary.

No network: Gemini is an httpx.MockTransport (which fires no connection events, so the phases
it cannot see are reported as "other"); the phase arithmetic is checked on a PhaseClock fed
the events httpcore would send.
"""
from __future__ import annotations

import logging
import os
import time

import httpx
import pytest

from tests.llm_fakes import FakeProviders, gemini_reply

KEY = "test-gemini-key-0123456789"
TOOL = {"name": "submit_thing", "description": "Submit the thing.",
        "input_schema": {"type": "object", "properties": {"a": {"type": "integer"}}}}


@pytest.fixture(scope="module", autouse=True)
def seeded_db(tmp_path_factory):
    os.environ["DB_URL"] = (os.environ.get("PANTRY_TEST_DB_URL")
                            or f"sqlite:///{tmp_path_factory.mktemp('db') / 'trace.db'}")
    from pantry_planner import config, db
    from pantry_planner.nlsearch import vocab

    config.settings.cache_clear()
    db.seed_from_json()
    vocab.clear_cache()
    yield
    config.settings.cache_clear()
    vocab.clear_cache()


@pytest.fixture(autouse=True)
def env(monkeypatch):
    from pantry_planner import config, llm

    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    monkeypatch.setenv("SELECTOR_MODEL_DEFAULT", "gemini:gemini-3.1-flash-lite")
    for var in ("ANTHROPIC_API_KEY", "DEMO_MODE", "GEMINI_BASE_URL", "ROUTING_STRATEGY",
                "SELECTOR_MODEL_ESCALATION", "NL2SQL_MODEL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(llm, "_sleep", lambda _s: None)
    config.settings.cache_clear()
    yield
    config.settings.cache_clear()


def _serve(monkeypatch, *responses):
    from pantry_planner import llm

    queue = list(responses)
    monkeypatch.setattr(llm, "_transport", httpx.MockTransport(lambda _r: queue.pop(0)))


def _call():
    from pantry_planner.llm import forced_tool_call

    return forced_tool_call(model="gemini:gemini-test", system="s",
                            messages=[{"role": "user", "content": "hi"}], tool=TOOL,
                            max_tokens=256)


def test_the_clock_turns_connection_events_into_phases():
    from pantry_planner.httptrace import PhaseClock, attempt_record

    clock = PhaseClock()
    t = time.perf_counter() - 5.0           # the request started 5 s ago
    for event, at in [("connection.connect_tcp.started", 0.00),
                      ("connection.connect_tcp.complete", 0.04),
                      ("connection.start_tls.started", 0.04),
                      ("connection.start_tls.complete", 0.10),
                      ("http11.send_request_headers.started", 0.10),
                      ("http11.send_request_headers.complete", 0.11),
                      ("http11.send_request_body.started", 0.11),
                      ("http11.send_request_body.complete", 0.12),
                      ("http11.receive_response_headers.started", 0.12),
                      ("http11.receive_response_headers.complete", 4.92),
                      ("http11.receive_response_body.started", 4.92),
                      ("http11.receive_response_body.complete", 4.95)]:
        clock.marks.setdefault(event.partition(".")[2], t + at)
    response = httpx.Response(200, headers={"server-timing": "gfet4t7; dur=4700"},
                              content=b"{}")
    rec = attempt_record(clock, n=1, started=t, dns_ms=12.0, status=200, response=response)

    assert (rec["connect_ms"], rec["tls_ms"], rec["upload_ms"]) == (40, 60, 20)
    assert (rec["wait_ms"], rec["download_ms"], rec["dns_ms"]) == (4800, 30, 12)
    assert rec["server_ms"] == 4700 and rec["new_connection"] is True
    assert rec["total_ms"] == pytest.approx(5012, abs=50)
    assert rec["other_ms"] == pytest.approx(rec["total_ms"] - 4962, abs=1)


def test_server_timing_takes_the_largest_duration():
    from pantry_planner.httptrace import server_ms

    assert server_ms(httpx.Headers({"server-timing": "gfet4t7; dur=1726"})) == 1726.0
    assert server_ms(httpx.Headers({"server-timing": "a;dur=3, b;dur=12.5"})) == 12.5
    assert server_ms(httpx.Headers({})) is None


def test_every_attempt_and_retry_wait_is_traced_and_logged(monkeypatch, caplog):
    from pantry_planner import httptrace

    _serve(monkeypatch,
           httpx.Response(503, json={"error": {"code": 503, "message": "high demand"}}),
           httpx.Response(200, json=gemini_reply("submit_thing", {"a": 1}),
                          headers={"server-timing": "gfet4t7; dur=850"}))
    with caplog.at_level(logging.INFO, logger="pantry_planner.llm.trace"):
        out = _call()

    trace = out.http
    assert trace["model"] == "gemini:gemini-test" and trace["step"] == "submit_thing"
    assert [a["status"] for a in trace["attempts"]] == [503, 200]
    assert trace["attempts"][0]["backoff_ms"] == 1000          # 1 s before attempt 2
    assert trace["attempts"][1]["server_ms"] == 850
    assert trace["attempts"][0]["request_bytes"] > 0
    names = [p["name"] for p in httptrace.phase_list(trace, "Google")]
    assert "retry wait" in names
    [record] = caplog.records
    assert record.levelno == logging.INFO
    assert "#1 HTTP 503 (503: high demand)" in record.getMessage()
    assert "#2 HTTP 200" in record.getMessage()
    assert "Google reports 0.85 s of its own" in record.getMessage()


def test_a_slow_or_failed_call_is_logged_as_a_warning(monkeypatch, caplog):
    from pantry_planner import httptrace
    from pantry_planner.llm import LLMError

    monkeypatch.setattr(httptrace, "SLOW_CALL_S", 0.0)
    _serve(monkeypatch, httpx.Response(400, json={"error": {"message": "bad"}}))
    with caplog.at_level(logging.INFO, logger="pantry_planner.llm.trace"), \
            pytest.raises(LLMError):
        _call()
    [record] = caplog.records
    assert record.levelno == logging.WARNING and "#1 HTTP 400" in record.getMessage()


def test_the_span_counts_http_retries_and_the_step_shows_the_phases():
    from pantry_planner.tracing import llm_span, llm_step

    http = {"provider": "gemini", "model": "gemini:m", "total_ms": 3200, "attempts": [
        {"n": 1, "status": 503, "total_ms": 150, "wait_ms": 140, "other_ms": 10,
         "backoff_ms": 1000, "server_ms": None},
        {"n": 2, "status": 200, "total_ms": 2050, "connect_ms": 50, "wait_ms": 1990,
         "download_ms": 10, "backoff_ms": 0, "server_ms": 1900}]}
    span = llm_span(step="select_products", model="gemini:m", input_tokens=10,
                    output_tokens=5, cost_usd=0.0, latency_ms=3200, http=http)
    assert span["retry_count"] == 1 and span["http"] is http

    step = llm_step(span)
    assert step.step_id == "llm_select_products" and step.kind.value == "llm"
    assert step.duration_ms == 3200
    assert "waiting for Google 2.1 s (Google reports 1.9 s)" in step.label
    assert "2 attempts" in step.label
    assert [(p.name, p.attempt) for p in step.phases][:3] == [
        ("waiting for Google", 1), ("other (client)", 1), ("retry wait", 1)]
    assert "#2 HTTP 200" in step.sql_display


def test_a_plan_carries_its_llm_calls_into_the_trace_and_the_mcp_summary(monkeypatch):
    from pantry_planner import flow
    from pantry_planner.mcp_server import _summarize_plan

    FakeProviders().install(monkeypatch)
    plan = flow.run("tomato_penne")

    [call] = plan.llm_calls
    assert (call.step, call.model, call.attempts, call.status) == \
        ("select_products", "gemini:gemini-3.1-flash-lite", 1, 200)
    assert plan.plan_trace[0].step_id == "llm_select_products"
    assert plan.plan_trace[0].phases
    assert _summarize_plan(plan).llm_calls == plan.llm_calls


def test_a_demo_mode_plan_adds_no_llm_step(monkeypatch):
    from pantry_planner import config, flow

    monkeypatch.setenv("DEMO_MODE", "1")
    config.settings.cache_clear()
    plan = flow.run("tomato_penne")
    assert plan.llm_calls == [] and plan.plan_trace == []

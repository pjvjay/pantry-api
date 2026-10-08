"""limits.py: a token bucket per client IP and a daily LLM cost ceiling (PLAN.md 3.12).

The limiter's clock is moved by the tests, never slept on. The live path runs on the fake
providers in tests/llm_fakes.py, so "spend" is the estimate for their fixed token counts.
"""
from __future__ import annotations

import os

import pytest

DOC = {"key": "imp:1", "title": "Pasta", "source": {"kind": "pasted", "method": "paste"},
       "lines": [{"line_no": 1, "text": "500g penne", "name": "penne", "quantity": 500,
                  "unit": "g", "amount_basis": "parsed_from_your_paste"}]}
PASTE = {"lines": ["500g penne"]}


@pytest.fixture(scope="module", autouse=True)
def seeded_db(tmp_path_factory):
    os.environ["DB_URL"] = (os.environ.get("PANTRY_TEST_DB_URL")
                            or f"sqlite:///{tmp_path_factory.mktemp('db') / 'limits.db'}")
    os.environ["DEMO_MODE"] = "1"
    os.environ.pop("ANTHROPIC_API_KEY", None)
    from pantry_planner import config, db
    from pantry_planner.nlsearch import vocab

    config.settings.cache_clear()
    db.seed_from_json()
    vocab.clear_cache()
    yield
    for var in ("DEMO_MODE", "TRUSTED_PROXY_HOPS", "LLM_DAILY_COST_CAP_USD"):
        os.environ.pop(var, None)
    config.settings.cache_clear()
    vocab.clear_cache()


@pytest.fixture()
def env(monkeypatch):
    """Set limit env vars for one test; settings are re-read before and after."""
    from pantry_planner import config

    def set_(**values):
        for k, v in values.items():
            monkeypatch.setenv(k, v)
        config.settings.cache_clear()

    yield set_
    monkeypatch.undo()
    config.settings.cache_clear()


@pytest.fixture()
def clock(monkeypatch):
    from pantry_planner import limits

    now = [1000.0]
    monkeypatch.setattr(limits, "monotonic", lambda: now[0])
    return now


def _client():
    from fastapi.testclient import TestClient

    from pantry_planner.api import app

    return TestClient(app)


# ─── Token bucket ────────────────────────────────────────────

def test_over_the_limit_is_a_429_with_retry_after(clock):
    c = _client()
    for n in range(60):
        assert c.post("/recipes/parse-lines", json=PASTE).status_code == 200, n
    resp = c.post("/recipes/parse-lines", json=PASTE)
    assert resp.status_code == 429
    assert resp.headers["Retry-After"] == "1"            # 60/min: one token a second
    # shaped like an LLM failure: a code, and a sentence in `detail` that a client which
    # shows `detail` as it is (the console) shows as written
    assert resp.json() == {"error": "rate_limited",
                           "detail": "Too many requests to /recipes/parse-lines; retry in 1 s."}
    clock[0] += 1.0
    assert c.post("/recipes/parse-lines", json=PASTE).status_code == 200
    assert c.post("/recipes/parse-lines", json=PASTE).status_code == 429


def test_planning_endpoints_allow_ten_a_minute(clock):
    c = _client()
    for n in range(10):
        assert c.post("/plan/spec", json={"doc": DOC}).status_code == 200, n
    resp = c.post("/plan/spec", json={"doc": DOC})
    assert resp.status_code == 429 and resp.headers["Retry-After"] == "6"
    # each endpoint has its own bucket
    assert c.post("/plan/nl", json={"recipe_text": "P\n- 500g penne"}).status_code == 200
    clock[0] += 6.0
    assert c.post("/plan/spec", json={"doc": DOC}).status_code == 200


def test_forwarded_for_is_ignored_unless_proxies_are_trusted(clock, env):
    from pantry_planner import limits

    c = _client()
    for n in range(60):
        hdr = {"X-Forwarded-For": f"203.0.113.{n}"}
        assert c.post("/recipes/parse-lines", json=PASTE, headers=hdr).status_code == 200
    # a fresh forged address is still the same TCP peer
    assert c.post("/recipes/parse-lines", json=PASTE,
                  headers={"X-Forwarded-For": "198.51.100.7"}).status_code == 429

    env(TRUSTED_PROXY_HOPS="1")
    limits.reset()
    # behind one trusted proxy, the rightmost entry is the client; a spoofed left part
    # changes nothing
    for _ in range(60):
        hdr = {"X-Forwarded-For": "1.2.3.4, 203.0.113.9"}
        assert c.post("/recipes/parse-lines", json=PASTE, headers=hdr).status_code == 200
    hdr = {"X-Forwarded-For": "5.6.7.8, 203.0.113.9"}
    assert c.post("/recipes/parse-lines", json=PASTE, headers=hdr).status_code == 429
    hdr = {"X-Forwarded-For": "203.0.113.10"}
    assert c.post("/recipes/parse-lines", json=PASTE, headers=hdr).status_code == 200


def test_client_ip_counts_hops_from_the_right(env):
    from types import SimpleNamespace

    from pantry_planner.limits import client_ip

    def req(xff):
        return SimpleNamespace(client=SimpleNamespace(host="10.0.0.1"),
                               headers={"x-forwarded-for": xff} if xff else {})

    assert client_ip(req("1.1.1.1, 2.2.2.2")) == "10.0.0.1"
    env(TRUSTED_PROXY_HOPS="2")
    assert client_ip(req("9.9.9.9, 1.1.1.1, 2.2.2.2")) == "1.1.1.1"
    assert client_ip(req("1.1.1.1")) == "1.1.1.1"       # fewer entries than hops
    assert client_ip(req("")) == "10.0.0.1"


def test_a_proxy_nobody_trusts_is_logged_once(env, caplog):
    """Behind ingress-nginx or a hosting platform's proxy with TRUSTED_PROXY_HOPS unset, the
    peer is the proxy and every visitor shares its buckets. Nothing in a 429 says why, so the
    first forwarded request says it in the log, once per process."""
    from pantry_planner import limits

    c = _client()
    with caplog.at_level("WARNING", logger="pantry_planner.limits"):
        c.post("/recipes/parse-lines", json=PASTE)       # no header: nothing to say
        assert not caplog.records
        for n in range(3):
            c.post("/recipes/parse-lines", json=PASTE,
                   headers={"X-Forwarded-For": f"198.51.100.{n}"})
        assert len(caplog.records) == 1
        assert "TRUSTED_PROXY_HOPS is 0" in caplog.records[0].getMessage()

        caplog.clear()
        env(TRUSTED_PROXY_HOPS="1")
        limits.reset()
        c.post("/recipes/parse-lines", json=PASTE, headers={"X-Forwarded-For": "198.51.100.9"})
        assert not caplog.records


# ─── Daily LLM cost ceiling ──────────────────────────────────

def test_spend_is_recorded_per_utc_day(monkeypatch):
    from pantry_planner import limits

    day = ["2026-10-08"]
    monkeypatch.setattr(limits, "utc_day", lambda: day[0])
    limits.record_spend(0.25)
    limits.record_spend(0.5)
    limits.record_spend(0.0)
    assert limits.budget()["spent"] == 0.75
    day[0] = "2026-10-09"                               # UTC midnight
    assert limits.budget()["spent"] == 0.0
    limits.record_spend(0.1)
    assert limits.budget()["spent"] == 0.1


def test_every_llm_call_counts_toward_the_ceiling(monkeypatch):
    from pantry_planner import config, limits
    from pantry_planner.config import HAIKU, estimate_cost_usd
    from tests.llm_fakes import FakeProviders

    fakes = FakeProviders().install(monkeypatch)
    config.set_runtime_overrides(demo_mode=False)
    try:
        resp = _client().post("/plan/spec", json={"doc": DOC})
    finally:
        config.clear_runtime_overrides()
    assert resp.status_code == 200, resp.text
    # the fake answers every call with 1000 input and 200 output tokens
    per_call = estimate_cost_usd(HAIKU, 1000, 200)
    assert len(fakes.anthropic) >= 1
    assert limits.budget()["spent"] == pytest.approx(per_call * len(fakes.anthropic))


def test_above_the_ceiling_llm_endpoints_pause_and_the_rest_work(env):
    from pantry_planner import config, limits

    env(LLM_DAILY_COST_CAP_USD="0.01")
    c = _client()
    assert c.get("/health").json()["llm_budget"] == {"spent": 0.0, "cap": 0.01}
    limits.record_spend(0.02)
    assert c.get("/health").json()["llm_budget"] == {"spent": 0.02, "cap": 0.01}

    config.set_runtime_overrides(demo_mode=False)
    try:
        for path, body in (("/plan/spec", {"doc": DOC}),
                           ("/plan/nl", {"recipe_text": "P\n- 500g penne"}),
                           ("/plan/tomato_penne", None), ("/plan/week", {"days": 1})):
            resp = c.post(path, json=body)
            assert resp.status_code == 503, path
            assert resp.json() == {"error": "llm_budget_exhausted", "detail": limits.PAUSED}
        # no LLM call, no pause
        assert c.post("/recipes/parse-lines", json=PASTE).status_code == 200
        assert c.get("/recipes/tomato_penne/doc").status_code == 200
    finally:
        config.clear_runtime_overrides()
    # demo mode makes no LLM call: the demo planner still works
    assert c.post("/plan/spec", json={"doc": DOC}).status_code == 200


@pytest.mark.asyncio
async def test_mcp_plan_tools_honour_the_ceiling(env):
    from mcp.server.mcpserver.exceptions import ToolError

    from pantry_planner import config, limits
    from pantry_planner.mcp_server import server

    env(LLM_DAILY_COST_CAP_USD="0")
    limits.record_spend(0.001)
    config.set_runtime_overrides(demo_mode=False)
    try:
        for tool, args in (("plan_recipe", {"slug": "tomato_penne"}),
                           ("plan_from_text", {"recipe_text": "P\n- 500g penne"}),
                           ("plan_from_lines", {"doc_key": "k", "lines": [{"name": "penne"}]}),
                           ("plan_week", {"days": 1})):
            with pytest.raises(ToolError, match="paused for today"):
                await server.call_tool(tool, args)
    finally:
        config.clear_runtime_overrides()


def test_no_ceiling_by_default_and_bad_values_stop_startup(env):
    from pantry_planner import config, limits

    limits.record_spend(1000.0)
    config.set_runtime_overrides(demo_mode=False)
    try:
        assert limits.llm_paused() is False
        assert limits.budget()["cap"] is None
    finally:
        config.clear_runtime_overrides()
    for var, bad in (("TRUSTED_PROXY_HOPS", "one"), ("TRUSTED_PROXY_HOPS", "-1"),
                     ("LLM_DAILY_COST_CAP_USD", "lots"), ("LLM_DAILY_COST_CAP_USD", "-5")):
        env(**{var: bad})
        with pytest.raises(ValueError, match=var):
            config.validate_startup()
        env(**{var: ""})

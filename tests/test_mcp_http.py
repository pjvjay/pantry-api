"""Bearer auth on /mcp and the write tools over HTTP.

This is the ONE module that opens `TestClient(api.app)`: the streamable
HTTP session manager can be started once per process, so every HTTP-level
/mcp test lives here behind one module-scoped client. Auth policy is read
from settings() on every request, so the tests flip MCP_AUTH_TOKENS and
clear the settings cache without rebuilding the app — exactly what an
operator restarting with a new env does not have to do either.
"""
from __future__ import annotations

import os

import pytest
from sqlalchemy.orm import Session

_TMP_DB = None
_PREV_TOKENS = None
SECRET = "0123456789abcdef0123"
GARLIC = 13
HEADERS = {"Accept": "application/json, text/event-stream",
           "Content-Type": "application/json"}


@pytest.fixture(scope="module", autouse=True)
def seeded_db(tmp_path_factory):
    global _TMP_DB, _PREV_TOKENS
    _TMP_DB = tmp_path_factory.mktemp("db") / "test.db"
    os.environ["DB_URL"] = os.environ.get("PANTRY_TEST_DB_URL") or f"sqlite:///{_TMP_DB}"
    _PREV_TOKENS = os.environ.pop("MCP_AUTH_TOKENS", None)
    from pantry_planner import config, db

    config.settings.cache_clear()
    db.seed_from_json()
    yield
    if _PREV_TOKENS is None:
        os.environ.pop("MCP_AUTH_TOKENS", None)
    else:
        os.environ["MCP_AUTH_TOKENS"] = _PREV_TOKENS
    config.settings.cache_clear()


@pytest.fixture(scope="module")
def client(seeded_db):
    from fastapi.testclient import TestClient

    from pantry_planner import api

    with TestClient(api.app) as c:
        yield c


@pytest.fixture()
def anonymous():
    from pantry_planner import config

    os.environ.pop("MCP_AUTH_TOKENS", None)
    config.settings.cache_clear()
    yield


@pytest.fixture()
def with_tokens():
    from pantry_planner import config

    os.environ["MCP_AUTH_TOKENS"] = f"reviewer:{SECRET}"
    config.settings.cache_clear()
    yield
    os.environ.pop("MCP_AUTH_TOKENS", None)
    config.settings.cache_clear()


@pytest.fixture(autouse=True)
def clean_submissions():
    from pantry_planner.db import OriginSubmissionRow, ProductOriginEvidenceRow, engine

    with Session(engine()) as s:
        s.query(OriginSubmissionRow).delete()
        s.query(ProductOriginEvidenceRow).delete()
        s.commit()
    yield


def _rpc(client, method, params=None, *, headers=None):
    h = dict(HEADERS, **(headers or {}))
    return client.post("/mcp", headers=h, json={
        "jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}})


def _call(client, name, arguments, *, token=None, headers=None):
    h = dict(headers or {})
    if token is not None:
        h["Authorization"] = f"Bearer {token}"
    return _rpc(client, "tools/call", {"name": name, "arguments": arguments}, headers=h)


def _submission_args(**kw):
    args = dict(product_id=GARLIC, claim_type="product-of", country="USA",
                verbatim="Product of USA", confidence="high")
    args.update(kw)
    return args


def _rows():
    from pantry_planner.db import OriginSubmissionRow, engine

    with Session(engine()) as s:
        rows = s.query(OriginSubmissionRow).order_by(OriginSubmissionRow.id).all()
        s.expunge_all()
        return rows


# ─── Anonymous endpoint (MCP_AUTH_TOKENS unset) ──────────────

def test_anonymous_read_tool_works(client, anonymous):
    r = _call(client, "list_recipes", {})
    assert r.status_code == 200, r.text
    result = r.json()["result"]
    assert result["isError"] is False
    assert "pbj_sandwich" in {x["slug"] for x in result["structuredContent"]["result"]}


def test_anonymous_submit_is_refused_naming_the_env_var(client, anonymous):
    r = _call(client, "submit_origin_evidence", _submission_args())
    assert r.status_code == 200, r.text
    result = r.json()["result"]
    assert result["isError"] is True
    assert "MCP_AUTH_TOKENS" in result["content"][0]["text"]
    assert _rows() == []


def test_anonymous_pipeline_status_reports_posture(client, anonymous):
    r = _call(client, "pipeline_status", {})
    status = r.json()["result"]["structuredContent"]
    assert status["mcp_auth"] == "anonymous"
    assert status["write_tools"] == "disabled"


# ─── Tokens configured ───────────────────────────────────────

def test_missing_bearer_is_401_with_challenge(client, with_tokens):
    r = _call(client, "list_recipes", {})
    assert r.status_code == 401
    assert r.headers["www-authenticate"].startswith("Bearer")
    assert r.json()["error"] == "invalid_token"
    assert "Authorization: Bearer" in r.json()["error_description"]


def test_wrong_secret_is_401(client, with_tokens):
    r = _call(client, "list_recipes", {}, token="0123456789abcdef9999")
    assert r.status_code == 401
    r = _call(client, "list_recipes", {}, token=SECRET[:-1])
    assert r.status_code == 401


def test_garbled_authorization_header_is_401_not_500(client, with_tokens):
    r = _call(client, "list_recipes", {}, headers={"Authorization": "Basic zzz"})
    assert r.status_code == 401
    r = _call(client, "list_recipes", {}, headers={"Authorization": "Bearer"})
    assert r.status_code == 401
    r = _call(client, "list_recipes", {}, headers={"Authorization": "Bearer  not a token !"})
    assert r.status_code == 401


def test_right_secret_submits_and_reviews_as_the_token_label(client, with_tokens):
    r = _call(client, "submit_origin_evidence", _submission_args(), token=SECRET)
    assert r.status_code == 200, r.text
    result = r.json()["result"]
    assert result["isError"] is False, result
    sub = result["structuredContent"]
    assert sub["status"] == "pending"
    assert sub["submitted_by"] == "reviewer"
    assert sub["country"] == "United States"
    rows = _rows()
    assert len(rows) == 1 and rows[0].submitted_by == "reviewer"

    r = _call(client, "review_origin_submission",
              {"submission_id": sub["id"], "decision": "approve", "note": "legible"},
              token=SECRET)
    result = r.json()["result"]
    assert result["isError"] is False, result
    assert result["structuredContent"]["reviewed_by"] == "reviewer"
    assert result["structuredContent"]["status"] == "approved"
    row = _rows()[0]
    assert row.reviewed_by == "reviewer" and row.status == "approved"
    assert row.evidence_id is not None

    r = _call(client, "pipeline_status", {}, token=SECRET)
    status = r.json()["result"]["structuredContent"]
    assert status["mcp_auth"] == "required"
    assert status["write_tools"] == "enabled"


def test_tools_list_with_the_right_secret(client, with_tokens):
    r = _rpc(client, "tools/list", headers={"Authorization": f"Bearer {SECRET}"})
    assert r.status_code == 200, r.text
    names = {t["name"] for t in r.json()["result"]["tools"]}
    assert {"submit_origin_evidence", "review_origin_submission",
            "list_origin_submissions", "list_recipes"} <= names


def test_initialize_reports_the_release_not_a_literal(client, anonymous):
    # serverInfo.version is version.app_version(), read when the server is
    # built: the baked release in an image, else git describe or "unknown".
    # It used to be a hard-coded 0.1.0 that no release would ever have moved.
    from pantry_planner.version import app_version

    r = _rpc(client, "initialize", {
        "protocolVersion": "2025-06-18", "capabilities": {},
        "clientInfo": {"name": "pytest", "version": "0"}})
    assert r.status_code == 200, r.text
    info = r.json()["result"]["serverInfo"]
    assert info["name"] == "pantry-planner"
    assert info["version"] == app_version()
    assert info["version"] != "0.1.0"
    assert r.headers["x-pantry-version"] == app_version()


def test_validation_failure_over_http_names_the_field(client, with_tokens):
    r = _call(client, "submit_origin_evidence",
              _submission_args(country="Amerca"), token=SECRET)
    result = r.json()["result"]
    assert result["isError"] is True
    assert "Amerca" in result["content"][0]["text"]
    assert "United States" in result["content"][0]["text"]
    r = _call(client, "submit_origin_evidence",
              _submission_args(claim_type="origin"), token=SECRET)
    text = r.json()["result"]["content"][0]["text"]
    assert "claim_type" in text and "product-of" in text
    assert _rows() == []


# ─── In-process (no request): the operator's own process ─────

@pytest.mark.asyncio
async def test_in_process_call_is_the_trusted_local_operator(anonymous):
    from pantry_planner.mcp_server import server

    res = await server.call_tool("submit_origin_evidence", _submission_args())
    assert res.is_error is False
    assert res.structured_content["submitted_by"] == "local"
    assert _rows()[0].submitted_by == "local"

    res = await server.call_tool("pipeline_status", {})
    assert res.structured_content["write_tools"] == "enabled"

    page = await server.call_tool("list_origin_submissions", {})
    assert page.structured_content["total"] == 1
    assert page.structured_content["next_offset"] is None


# ─── Token parsing ───────────────────────────────────────────

def test_weak_token_fails_startup_without_leaking_it(monkeypatch):
    from pantry_planner.config import Settings

    monkeypatch.setenv("MCP_AUTH_TOKENS", "x:short")
    with pytest.raises(ValueError, match="16") as exc:
        Settings.from_env()
    assert "short" not in str(exc.value)


def test_token_parsing_labels_strips_and_skips_empties():
    from pantry_planner.config import parse_mcp_auth_tokens

    parsed = parse_mcp_auth_tokens(
        " reviewer : 0123456789abcdef0123 ,, 0123456789abcdefXYZW , :0123456789abcdef0000 ")
    assert parsed == (("reviewer", "0123456789abcdef0123"),
                      ("token-2", "0123456789abcdefXYZW"),
                      ("token-3", "0123456789abcdef0000"))
    assert parse_mcp_auth_tokens("") == ()
    assert parse_mcp_auth_tokens(" , ") == ()

"""Runtime settings: demo mode and model specs switch without a restart.

GET /settings/runtime is always readable; POST needs
RUNTIME_SETTINGS_ENABLED=1 (403 otherwise, whatever the body) and rejects a
bad model spec with 422 without changing anything. /health, the MCP
pipeline_status tool and the planner itself all follow the change on the
next request.
"""
from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from tests.llm_fakes import FakeProviders

TEXT = "Garlic pasta (serves 2)\n- 200g spaghetti\n- 3 cloves garlic\n- olive oil\n"
ENV_DEFAULTS = {"selector_default": "claude-haiku-4-5-20251001",
                "selector_escalation": "claude-sonnet-4-6",
                "classifier": "claude-haiku-4-5-20251001",
                "nl2sql": "claude-sonnet-4-6"}


@pytest.fixture(scope="module", autouse=True)
def seeded_db(tmp_path_factory):
    os.environ["DB_URL"] = (os.environ.get("PANTRY_TEST_DB_URL")
                            or f"sqlite:///{tmp_path_factory.mktemp('db') / 'rt.db'}")
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
    from pantry_planner import config

    for var in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "DEMO_MODE", "RUNTIME_SETTINGS_ENABLED",
                "SELECTOR_MODEL_DEFAULT", "SELECTOR_MODEL_ESCALATION", "CLASSIFIER_MODEL",
                "NL2SQL_MODEL", "ROUTING_STRATEGY"):
        monkeypatch.delenv(var, raising=False)
    config.settings.cache_clear()
    yield
    config.settings.cache_clear()          # also drops any runtime override


@pytest.fixture()
def editable(monkeypatch):
    from pantry_planner import config

    monkeypatch.setenv("RUNTIME_SETTINGS_ENABLED", "1")
    config.settings.cache_clear()


@pytest.fixture()
def client():
    from pantry_planner.api import app

    return TestClient(app)


def test_get_reports_the_effective_settings(client, monkeypatch):
    from pantry_planner import config

    monkeypatch.setenv("GEMINI_API_KEY", "k" * 20)
    config.settings.cache_clear()
    assert client.get("/settings/runtime").json() == {
        "demo_mode": False, "models": ENV_DEFAULTS,
        "gemini_key_configured": True, "anthropic_key_configured": False,
        "editable": False,
    }


@pytest.mark.parametrize("body", [
    {"demo_mode": True},
    {"models": {"selector_default": "gpt-4o"}},       # invalid too: 403 comes first
    {},
])
def test_post_is_403_unless_enabled(client, body):
    from pantry_planner import config

    resp = client.post("/settings/runtime", json=body)
    assert resp.status_code == 403
    assert "RUNTIME_SETTINGS_ENABLED=1" in resp.json()["detail"]
    assert config.runtime_overrides() == {}
    assert client.get("/health").json()["demo_mode"] is False


@pytest.mark.parametrize("models", [
    {"selector_default": "gpt-4o"},
    {"nl2sql": ""},
    {"classifier": "gemini:"},
    {"selector_escalation": "anthropic:  "},
    {"selector_default": "gemini:gemini-x", "nl2sql": "llama-3"},   # one bad: none applied
    {"made_up": "gemini:gemini-x"},
])
def test_post_rejects_a_bad_spec_with_422_and_changes_nothing(client, editable, models):
    resp = client.post("/settings/runtime", json={"demo_mode": True, "models": models})
    assert resp.status_code == 422, resp.text
    after = client.get("/settings/runtime").json()
    assert after["models"] == ENV_DEFAULTS and after["demo_mode"] is False


def test_post_rejects_unknown_top_level_fields(client, editable):
    assert client.post("/settings/runtime", json={"demo": True}).status_code == 422


def test_post_applies_models_and_health_and_pipeline_status_follow(client, editable):
    import asyncio

    from pantry_planner.mcp_server import server

    resp = client.post("/settings/runtime", json={"models": {
        "selector_default": " gemini:gemini-3.1-flash-lite ",
        "nl2sql": "anthropic:claude-sonnet-4-6"}})
    assert resp.status_code == 200, resp.text
    assert resp.json()["models"] == {**ENV_DEFAULTS,
                                     "selector_default": "gemini:gemini-3.1-flash-lite",
                                     "nl2sql": "anthropic:claude-sonnet-4-6"}
    assert resp.json()["editable"] is True

    # a later POST touching one field keeps the earlier override
    client.post("/settings/runtime", json={"models": {"classifier": "gemini:gemini-x"}})

    health = client.get("/health").json()
    assert health["default_model"] == "gemini:gemini-3.1-flash-lite"
    assert health["escalation_model"] == "claude-sonnet-4-6"
    assert health["classifier_model"] == "gemini:gemini-x"
    assert health["nl2sql_model"] == "anthropic:claude-sonnet-4-6"
    assert health["gemini_key_configured"] is False

    status = asyncio.run(server.call_tool("pipeline_status", {})).structured_content
    assert status["default_model"] == "gemini:gemini-3.1-flash-lite"
    assert status["classifier_model"] == "gemini:gemini-x"
    assert status["nl2sql_model"] == "anthropic:claude-sonnet-4-6"
    assert status["gemini_key_configured"] is False
    assert status["demo_mode"] is False


def test_toggling_demo_mode_switches_planning_without_a_restart(client, editable,
                                                                monkeypatch):
    """Demo mode on: /health says so and a plan runs on the deterministic
    stand-ins with no key. Off again with Gemini specs: the same request
    goes to Gemini."""
    from pantry_planner import config

    fakes = FakeProviders().install(monkeypatch)
    assert client.get("/health").json()["demo_mode"] is False

    assert client.post("/settings/runtime", json={"demo_mode": True}).json()["demo_mode"]
    assert client.get("/health").json()["demo_mode"] is True
    plan = client.post("/plan/nl", json={"recipe_text": TEXT})
    assert plan.status_code == 200, plan.text
    assert {li["model_used"] for li in plan.json()["line_items"]} == {"demo-deterministic"}
    assert plan.json()["total_llm_cost_usd"] == 0.0
    assert fakes.gemini == [] and fakes.anthropic == []           # no LLM touched
    assert os.environ.get("DEMO_MODE") is None                     # env untouched

    monkeypatch.setenv("GEMINI_API_KEY", "k" * 20)
    config._env_settings.cache_clear()          # pick up the key, keep the overrides
    resp = client.post("/settings/runtime", json={"demo_mode": False, "models": {
        "nl2sql": "gemini:gemini-flash-latest",
        "selector_default": "gemini:gemini-3.1-flash-lite"}})
    assert resp.json()["demo_mode"] is False and resp.json()["gemini_key_configured"]
    plan = client.post("/plan/nl", json={"recipe_text": TEXT})
    assert plan.status_code == 200, plan.text
    assert [b["model"] for b in fakes.gemini] == ["gemini-flash-latest",
                                                   "gemini-3.1-flash-lite"]
    assert {li["model_used"] for li in plan.json()["line_items"]} == \
        {"gemini:gemini-3.1-flash-lite"}


def test_three_phase_classifier_follows_runtime_demo_mode(editable, monkeypatch):
    from pantry_planner import config, db
    from pantry_planner.router.classifier import call_classifier

    fakes = FakeProviders().install(monkeypatch)
    config.set_runtime_overrides(demo_mode=True)
    call_classifier(db.load_recipe("tomato_penne"), db.load_all_products())
    assert fakes.gemini == [] and fakes.anthropic == []


def test_overrides_layer_over_env_and_cache_clear_resets(monkeypatch):
    from pantry_planner import config

    base = config.settings()
    config.set_runtime_overrides(demo_mode=True, models={"classifier": "gemini:g"})
    assert config.settings().demo_mode is True
    assert config.settings().classifier_model == "gemini:g"
    assert config.settings().db_url == base.db_url                 # everything else intact
    assert config.runtime_overrides() == {"demo_mode": True, "classifier_model": "gemini:g"}

    with pytest.raises(ValueError, match="models.nl2sql"):
        config.set_runtime_overrides(demo_mode=False, models={"nl2sql": "nope"})
    assert config.settings().demo_mode is True                      # nothing applied

    config.settings.cache_clear()
    assert config.runtime_overrides() == {}
    assert config.settings().demo_mode is False

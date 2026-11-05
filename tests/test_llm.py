"""The LLM boundary: Gemini over its OpenAI-compatible endpoint, Anthropic
unchanged, and every call site routed by its model spec.

No network and no keys: Gemini is an httpx.MockTransport, Anthropic a fake
SDK class (tests/llm_fakes.py). Gemini is asserted on the wire — URL,
bearer header, forced tool_choice, max_tokens headroom — and on the
failure modes the free tier actually produces (503 "high demand", 429 with
Retry-After, truncated replies, bad keys, unknown models).
"""
from __future__ import annotations

import json
import os

import httpx
import pytest

from tests.llm_fakes import PARSE_REPLY, FakeProviders, gemini_reply

KEY = "test-gemini-key-0123456789"
TOOL = {
    "name": "submit_thing",
    "description": "Submit the thing.",
    "input_schema": {
        "type": "object",
        "required": ["a"],
        "properties": {"a": {"type": "integer"}, "b": {"type": ["string", "null"]}},
    },
}
MSGS = [{"role": "user", "content": "hello"}]


@pytest.fixture(scope="module", autouse=True)
def seeded_db(tmp_path_factory):
    os.environ["DB_URL"] = (os.environ.get("PANTRY_TEST_DB_URL")
                            or f"sqlite:///{tmp_path_factory.mktemp('db') / 'llm.db'}")
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
    """A Gemini key, no Anthropic key, no demo mode, default models."""
    from pantry_planner import config, llm

    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    for var in ("ANTHROPIC_API_KEY", "DEMO_MODE", "GEMINI_BASE_URL",
                "GEMINI_REASONING_EFFORT", "SELECTOR_MODEL_DEFAULT",
                "SELECTOR_MODEL_ESCALATION", "CLASSIFIER_MODEL", "NL2SQL_MODEL",
                "ROUTING_STRATEGY", "ENABLE_THINKING_ON_ESCALATION"):
        monkeypatch.delenv(var, raising=False)
    sleeps: list[float] = []
    monkeypatch.setattr(llm, "_sleep", sleeps.append)
    config.settings.cache_clear()
    yield sleeps
    config.settings.cache_clear()


def _set(monkeypatch, **env):
    from pantry_planner import config

    for k, v in env.items():
        monkeypatch.setenv(k, v)
    config.settings.cache_clear()


def _serve(monkeypatch, *responses):
    """Gemini answers with `responses` in order (httpx.Response or an
    exception to raise); returns the list the requests are recorded into."""
    from pantry_planner import llm

    seen: list[httpx.Request] = []
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(llm, "_transport", httpx.MockTransport(handler))
    return seen


def _ok(args=None, **kw) -> httpx.Response:
    return httpx.Response(200, json=gemini_reply("submit_thing", args or {"a": 1}, **kw))


def _call(model="gemini:gemini-test", max_tokens=1024, **kw):
    from pantry_planner.llm import forced_tool_call

    return forced_tool_call(model=model, system="be terse", messages=MSGS, tool=TOOL,
                            max_tokens=max_tokens, **kw)


# ─── Gemini: request shape ───────────────────────────────────

def test_gemini_request_shape_and_forced_tool_choice(monkeypatch):
    seen = _serve(monkeypatch, _ok({"a": 7, "b": None}, prompt_tokens=120,
                                   total_tokens=300, completion_tokens=40))
    out = _call(temperature=0.0, thinking_budget=4000)

    (req,) = seen
    assert req.method == "POST"
    assert str(req.url) == ("https://generativelanguage.googleapis.com/v1beta/openai"
                            "/chat/completions")
    assert req.headers["authorization"] == f"Bearer {KEY}"
    body = json.loads(req.content)
    assert body["model"] == "gemini-test"                      # prefix stripped
    assert body["messages"] == [{"role": "system", "content": "be terse"},
                                {"role": "user", "content": "hello"}]
    assert body["tools"] == [{"type": "function", "function": {
        "name": "submit_thing", "description": "Submit the thing.",
        "parameters": TOOL["input_schema"]}}]
    assert body["tool_choice"] == {"type": "function", "function": {"name": "submit_thing"}}
    # 1024 is under the 2048 floor; "low" effort adds 2048 of thinking headroom
    assert body["max_tokens"] == 2048 + 2048
    assert body["reasoning_effort"] == "low"
    # Anthropic-only knobs never reach Gemini
    assert "temperature" not in body and "thinking" not in body

    assert out.input == {"a": 7, "b": None}
    assert out.input_tokens == 120
    assert out.output_tokens == 180                  # total - prompt: thinking counts
    assert out.model == "gemini:gemini-test"
    assert out.provider == "gemini"


def test_gemini_base_url_and_reasoning_effort_overrides(monkeypatch):
    _set(monkeypatch, GEMINI_BASE_URL="http://proxy.test/v1beta/openai/",
         GEMINI_REASONING_EFFORT="")
    seen = _serve(monkeypatch, _ok())
    _call(max_tokens=4096)
    assert str(seen[0].url) == "http://proxy.test/v1beta/openai/chat/completions"
    body = json.loads(seen[0].content)
    assert "reasoning_effort" not in body             # "" = omit, model default
    assert body["max_tokens"] == 4096 + 8192          # default-effort headroom


def test_gemini_anthropic_text_blocks_become_one_string(monkeypatch):
    from pantry_planner.llm import forced_tool_call

    seen = _serve(monkeypatch, _ok())
    forced_tool_call(model="gemini:g", system="s", tool=TOOL, max_tokens=10, messages=[
        {"role": "user", "content": [{"type": "text", "text": "a"},
                                     {"type": "text", "text": "b"}]}])
    assert json.loads(seen[0].content)["messages"][1] == {"role": "user", "content": "ab"}


def test_usage_falls_back_to_completion_tokens(monkeypatch):
    reply = gemini_reply("submit_thing", {"a": 1})
    reply["usage"] = {"prompt_tokens": 10, "completion_tokens": 5}
    _serve(monkeypatch, httpx.Response(200, json=reply))
    out = _call()
    assert (out.input_tokens, out.output_tokens) == (10, 5)


# ─── Gemini: reading the reply ───────────────────────────────

def test_arguments_already_an_object_are_accepted(monkeypatch):
    reply = gemini_reply("submit_thing", {"a": 1})
    reply["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = {"a": 2}
    _serve(monkeypatch, httpx.Response(200, json=reply))
    assert _call().input == {"a": 2}


def test_invalid_json_arguments_raise_clearly(monkeypatch):
    from pantry_planner.llm import LLMResponseError

    _serve(monkeypatch, _ok('{"a": 1,,}'))
    with pytest.raises(LLMResponseError, match="invalid JSON arguments for 'submit_thing'"):
        _call()


def test_arguments_that_are_not_an_object_raise(monkeypatch):
    from pantry_planner.llm import LLMResponseError

    _serve(monkeypatch, _ok("[1, 2]"))
    with pytest.raises(LLMResponseError, match="not a JSON object"):
        _call()


def test_length_truncated_reply_without_tool_call_names_the_cause(monkeypatch):
    from pantry_planner.llm import LLMResponseError

    reply = {"choices": [{"finish_reason": "length",
                          "message": {"role": "assistant", "content": None}}],
             "usage": {"prompt_tokens": 10, "total_tokens": 4106}}
    _serve(monkeypatch, httpx.Response(200, json=reply))
    with pytest.raises(LLMResponseError) as e:
        _call()
    msg = str(e.value)
    assert "finish_reason=length" in msg and "max_tokens=4096" in msg
    assert "GEMINI_REASONING_EFFORT" in msg


def test_length_truncated_arguments_say_they_were_cut_off(monkeypatch):
    from pantry_planner.llm import LLMResponseError

    _serve(monkeypatch, _ok('{"a": ', finish="length"))
    with pytest.raises(LLMResponseError, match="cut off at max_tokens"):
        _call()


def test_empty_reply_and_no_tool_call_raise(monkeypatch):
    from pantry_planner.llm import LLMResponseError

    _serve(monkeypatch,
           httpx.Response(200, json={"choices": []}),
           httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {
               "role": "assistant", "content": "I'd rather chat"}}]}))
    with pytest.raises(LLMResponseError, match="returned no choices"):
        _call()
    with pytest.raises(LLMResponseError, match="didn't call the tool 'submit_thing'"):
        _call()


# ─── Gemini: retries ─────────────────────────────────────────

HIGH_DEMAND = [{"error": {"code": 503, "status": "UNAVAILABLE",
                          "message": "This model is currently experiencing high demand."}}]


def test_503_is_retried_with_exponential_backoff(monkeypatch, env):
    seen = _serve(monkeypatch, httpx.Response(503, json=HIGH_DEMAND),
                  httpx.Response(503, json=HIGH_DEMAND), _ok({"a": 3}))
    assert _call().input == {"a": 3}
    assert len(seen) == 3
    assert env == [1.0, 2.0]                          # the recorded sleeps


def test_429_honours_a_numeric_retry_after(monkeypatch, env):
    seen = _serve(monkeypatch, httpx.Response(429, headers={"Retry-After": "7"},
                                              json={"error": {"message": "slow down"}}),
                  _ok())
    _call()
    assert len(seen) == 2 and env == [7.0]


def test_429_gives_up_after_four_attempts(monkeypatch, env):
    from pantry_planner.llm import LLMRateLimitError

    quota = {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED",
                       "message": "Quota exceeded for metric generate_requests_per_day"}}
    seen = _serve(monkeypatch, *[httpx.Response(429, json=quota) for _ in range(4)])
    with pytest.raises(LLMRateLimitError) as e:
        _call()
    assert len(seen) == 4 and env == [1.0, 2.0, 4.0]
    assert "HTTP 429, 4 attempts" in str(e.value)
    assert "RESOURCE_EXHAUSTED: Quota exceeded" in str(e.value)
    assert e.value.http_status == 429


def test_a_retry_after_too_long_to_wait_fails_now(monkeypatch, env):
    from pantry_planner.llm import LLMRateLimitError

    seen = _serve(monkeypatch, httpx.Response(429, headers={"Retry-After": "3600"},
                                              json={"error": {"message": "daily quota"}}))
    with pytest.raises(LLMRateLimitError, match="Retry-After 3600s"):
        _call()
    assert len(seen) == 1 and env == []


def test_connection_failures_are_retried(monkeypatch, env):
    seen = _serve(monkeypatch, httpx.ConnectError("refused"), _ok())
    assert _call().input == {"a": 1}
    assert len(seen) == 2 and env == [1.0]


def test_5xx_after_every_attempt_raises_server_error(monkeypatch):
    from pantry_planner.llm import LLMError

    _serve(monkeypatch, *[httpx.Response(500, text="boom") for _ in range(4)])
    with pytest.raises(LLMError, match="server error .*HTTP 500, 4 attempts.*boom"):
        _call()


# ─── Gemini: fail fast ───────────────────────────────────────

@pytest.mark.parametrize("status,body", [
    (401, {"error": {"code": 401, "message": "Request had invalid authentication"}}),
    (403, [{"error": {"code": 403, "status": "PERMISSION_DENIED", "message": "denied"}}]),
    (400, [{"error": {"code": 400, "status": "INVALID_ARGUMENT",
                      "message": "API key not valid. Please pass a valid API key."}}]),
])
def test_rejected_key_fails_fast_and_names_gemini_api_key(monkeypatch, env, status, body):
    from pantry_planner.llm import LLMAuthError

    seen = _serve(monkeypatch, httpx.Response(status, json=body))
    with pytest.raises(LLMAuthError) as e:
        _call()
    assert len(seen) == 1 and env == []               # no retry
    assert "GEMINI_API_KEY" in str(e.value) and f"HTTP {status}" in str(e.value)
    assert KEY not in str(e.value)


def test_unknown_model_is_a_404_naming_the_model(monkeypatch):
    from pantry_planner.llm import LLMModelNotFoundError

    seen = _serve(monkeypatch, httpx.Response(404, json=[{"error": {
        "code": 404, "status": "NOT_FOUND", "message": "models/gemini-nope is not found"}}]))
    with pytest.raises(LLMModelNotFoundError) as e:
        _call(model="gemini:gemini-nope")
    assert len(seen) == 1
    assert "'gemini-nope' not found (HTTP 404)" in str(e.value)
    assert "NL2SQL_MODEL" in str(e.value)


def test_other_4xx_is_reported_with_the_provider_message(monkeypatch):
    from pantry_planner.llm import LLMAuthError, LLMError

    _serve(monkeypatch, httpx.Response(400, json={"error": {
        "status": "INVALID_ARGUMENT", "message": "Unknown name \"foo\" at 'tools[0]'"}}))
    with pytest.raises(LLMError) as e:
        _call()
    assert not isinstance(e.value, LLMAuthError)
    assert "HTTP 400" in str(e.value) and "Unknown name" in str(e.value)


def test_missing_key_is_an_error_only_when_gemini_is_called(monkeypatch):
    from pantry_planner.llm import LLMConfigError

    _set(monkeypatch, GEMINI_API_KEY="")
    seen = _serve(monkeypatch)
    with pytest.raises(LLMConfigError, match="GEMINI_API_KEY is not set"):
        _call()
    assert seen == []


def test_spec_without_a_model_name_is_rejected():
    from pantry_planner.llm import LLMConfigError

    with pytest.raises(LLMConfigError, match="names no model"):
        _call(model="gemini:")


# ─── Model specs, cost ───────────────────────────────────────

@pytest.mark.parametrize("spec,expected", [
    ("gemini:gemini-flash-latest", ("gemini", "gemini-flash-latest")),
    ("GEMINI: gemini-x ", ("gemini", "gemini-x")),
    ("anthropic:claude-sonnet-4-6", ("anthropic", "claude-sonnet-4-6")),
    ("claude-haiku-4-5-20251001", ("anthropic", "claude-haiku-4-5-20251001")),
    ("us.anthropic.claude-x:0", ("anthropic", "us.anthropic.claude-x:0")),
])
def test_split_model_spec(spec, expected):
    from pantry_planner.config import split_model_spec

    assert split_model_spec(spec) == expected


@pytest.mark.parametrize("spec", ["", "  ", "gemini:", "anthropic: ", "gpt-4o", "haiku"])
def test_validate_model_spec_rejects(spec):
    from pantry_planner.config import validate_model_spec

    with pytest.raises(ValueError):
        validate_model_spec(spec)


def test_validate_model_spec_normalises():
    from pantry_planner.config import validate_model_spec

    assert validate_model_spec(" Gemini:gemini-x ") == "gemini:gemini-x"
    assert validate_model_spec("claude-sonnet-4-6") == "claude-sonnet-4-6"


def test_gemini_costs_nothing_and_anthropic_prefix_prices_the_same():
    from pantry_planner.config import HAIKU, estimate_cost_usd

    assert estimate_cost_usd("gemini:gemini-flash-latest", 10**6, 10**6) == 0.0
    assert estimate_cost_usd(HAIKU, 10**6, 10**6) == 6.0
    assert estimate_cost_usd(f"anthropic:{HAIKU}", 10**6, 10**6) == 6.0
    assert estimate_cost_usd("gemini:a+claude-x", 1, 1) == 0.0      # merged label: unknown


# ─── Call-site routing ───────────────────────────────────────

def _tomato_penne():
    from pantry_planner import db

    return db.load_recipe("tomato_penne"), db.load_all_products()


ANTHROPIC_PLAN_KWARGS = {"model", "max_tokens", "system", "tools", "tool_choice", "messages"}


@pytest.mark.parametrize("spec,provider", [
    ("gemini:gemini-3.1-flash-lite", "gemini"),
    ("claude-haiku-4-5-20251001", "anthropic"),
    ("anthropic:claude-haiku-4-5-20251001", "anthropic"),
])
def test_selector_routes_by_model_spec(monkeypatch, spec, provider):
    from pantry_planner.prompts import SELECTOR_SYSTEM, SELECTOR_TOOL
    from pantry_planner.selector import call_selector

    fakes = FakeProviders().install(monkeypatch)
    recipe, products = _tomato_penne()
    result = call_selector(recipe.ingredients, products, model=spec)

    assert len(result.selections) == len(recipe.ingredients)
    assert result.model_used == spec                   # the spec, prefix and all
    if provider == "gemini":
        assert len(fakes.gemini) == 1 and fakes.anthropic == []
        assert fakes.gemini[0]["model"] == "gemini-3.1-flash-lite"
        assert fakes.gemini[0]["tool_choice"]["function"]["name"] == "submit_plan"
        assert result.cost_usd == 0.0
        assert (result.input_tokens, result.output_tokens) == (100, 80)
    else:
        assert len(fakes.anthropic) == 1 and fakes.gemini == []
        kwargs = fakes.anthropic[0]
        # Today's request, exactly: no thinking, no temperature, forced tool
        assert set(kwargs) == ANTHROPIC_PLAN_KWARGS
        assert kwargs["model"] == "claude-haiku-4-5-20251001"
        assert kwargs["max_tokens"] == 4096
        assert kwargs["system"] == SELECTOR_SYSTEM
        assert kwargs["tools"] == [SELECTOR_TOOL]
        assert kwargs["tool_choice"] == {"type": "tool", "name": "submit_plan"}
        assert result.cost_usd == pytest.approx((1000 * 1.0 + 200 * 5.0) / 1e6)
        assert (result.input_tokens, result.output_tokens) == (1000, 200)


def test_selector_thinking_reaches_anthropic_only(monkeypatch):
    from pantry_planner.selector import call_selector

    fakes = FakeProviders().install(monkeypatch)
    recipe, products = _tomato_penne()
    call_selector(recipe.ingredients, products, model="claude-sonnet-4-6",
                  enable_thinking=True)
    call_selector(recipe.ingredients, products, model="gemini:gemini-flash-latest",
                  enable_thinking=True)
    assert fakes.anthropic[0]["thinking"] == {"type": "enabled", "budget_tokens": 4000}
    assert "thinking" not in fakes.gemini[0]
    assert fakes.gemini[0]["max_tokens"] == 4096 + 2048


@pytest.mark.parametrize("spec,provider", [
    ("gemini:gemini-flash-latest", "gemini"),
    ("claude-sonnet-4-6", "anthropic"),
])
def test_parser_routes_by_model_spec(monkeypatch, spec, provider):
    from pantry_planner.nlsearch.query_parser import PARSER_TOOL, parse_input

    _set(monkeypatch, NL2SQL_MODEL=spec)
    fakes = FakeProviders().install(monkeypatch)
    parsed = parse_input("Garlic pasta\n- 200g spaghetti\n- 3 cloves garlic\n- olive oil")

    assert parsed.error is None
    assert [i.name for i in parsed.recipe.ingredients] == \
        [i["name"] for i in PARSE_REPLY["recipe"]["ingredients"]]
    if provider == "gemini":
        assert len(fakes.gemini) == 1 and fakes.anthropic == []
        body = fakes.gemini[0]
        assert body["model"] == "gemini-flash-latest"
        assert body["tool_choice"]["function"]["name"] == "submit_parse"
        assert body["tools"][0]["function"]["parameters"] == PARSER_TOOL["input_schema"]
        assert parsed.cost_usd == 0.0
    else:
        assert len(fakes.anthropic) == 1 and fakes.gemini == []
        kwargs = fakes.anthropic[0]
        assert set(kwargs) == ANTHROPIC_PLAN_KWARGS | {"temperature"}
        assert kwargs["temperature"] == 0.0 and kwargs["max_tokens"] == 2048
        assert kwargs["tool_choice"] == {"type": "tool", "name": "submit_parse"}
        assert parsed.cost_usd == pytest.approx((1000 * 3.0 + 200 * 15.0) / 1e6)


def test_parser_still_degrades_instead_of_raising(monkeypatch):
    from pantry_planner.nlsearch.query_parser import parse_input

    _set(monkeypatch, NL2SQL_MODEL="gemini:gemini-flash-latest")
    _serve(monkeypatch, httpx.Response(401, json={"error": {"message": "bad key"}}))
    parsed = parse_input("- 200g spaghetti")
    assert parsed.recipe.ingredients == []
    assert "GEMINI_API_KEY" in parsed.error


@pytest.mark.parametrize("spec,provider", [
    ("gemini:gemini-3.1-flash-lite", "gemini"),
    ("claude-haiku-4-5-20251001", "anthropic"),
])
def test_classifier_routes_by_model_spec(monkeypatch, spec, provider):
    from pantry_planner.router.classifier import call_classifier

    _set(monkeypatch, CLASSIFIER_MODEL=spec)
    fakes = FakeProviders().install(monkeypatch)
    metrics = call_classifier(*_tomato_penne())

    assert metrics.match_confidence_1_to_10 == 8
    assert metrics.ambiguous_ingredients == [{"name": "garlic", "why": "fresh or powder"}]
    if provider == "gemini":
        assert len(fakes.gemini) == 1 and fakes.anthropic == []
        assert fakes.gemini[0]["tool_choice"]["function"]["name"] == "submit_triage"
        assert fakes.gemini[0]["max_tokens"] == 2048 + 2048   # 1024 raised to the floor
        assert metrics.cost_usd == 0.0
    else:
        assert len(fakes.anthropic) == 1 and fakes.gemini == []
        assert fakes.anthropic[0]["max_tokens"] == 1024
        assert fakes.anthropic[0]["tool_choice"] == {"type": "tool", "name": "submit_triage"}
        assert metrics.cost_usd > 0


def test_cascade_escalates_from_gemini_to_anthropic(monkeypatch):
    """Default and escalation specs are independent: a low-confidence Gemini
    line is re-run on Anthropic, and the plan says both."""
    from pantry_planner import flow

    _set(monkeypatch, SELECTOR_MODEL_DEFAULT="gemini:gemini-3.1-flash-lite",
         SELECTOR_MODEL_ESCALATION="claude-sonnet-4-6")
    fakes = FakeProviders().install(monkeypatch)
    fakes.low_conf_lines["gemini"] = {1}
    plan = flow.run("tomato_penne")

    assert len(fakes.gemini) == 1 and len(fakes.anthropic) == 1
    rerun = json.loads(fakes.anthropic[0]["messages"][0]["content"])["recipe_ingredients"]
    assert [i["line_no"] for i in rerun] == [1]
    assert plan.escalated is True
    assert plan.preselected_model == "gemini:gemini-3.1-flash-lite"
    assert {li.model_used for li in plan.line_items} == \
        {"gemini:gemini-3.1-flash-lite+claude-sonnet-4-6"}
    # only the Anthropic escalation costs money
    assert plan.total_llm_cost_usd == pytest.approx((1000 * 3.0 + 200 * 15.0) / 1e6)


def test_gemini_plan_records_gemini_metrics_at_zero_cost(monkeypatch):
    from pantry_planner import flow, metrics

    _set(monkeypatch, SELECTOR_MODEL_DEFAULT="gemini:gemini-3.1-flash-lite")
    FakeProviders().install(monkeypatch)
    labels = ("select_products", "gemini:gemini-3.1-flash-lite")
    before = metrics.llm_calls.labels(*labels)._value.get()
    plan = flow.run("tomato_penne")

    assert plan.total_llm_cost_usd == 0.0
    assert metrics.llm_calls.labels(*labels)._value.get() == before + 1
    assert metrics.llm_cost_usd.labels(*labels)._value.get() == 0.0


def test_week_plan_selector_follows_the_default_spec(monkeypatch):
    from pantry_planner import weekplan

    _set(monkeypatch, SELECTOR_MODEL_DEFAULT="gemini:gemini-3.1-flash-lite")
    fakes = FakeProviders().install(monkeypatch)
    week = weekplan.plan_week(days=1)
    assert len(fakes.gemini) == 1 and fakes.anthropic == []
    items = [li for d in week.days for li in d.line_items]
    assert items and {li.model_used for li in items} == {"gemini:gemini-3.1-flash-lite"}
    assert week.total_llm_cost_usd == 0.0


def test_three_phase_low_complexity_picks_the_default_spec(monkeypatch):
    """Phase C used to hard-code Haiku for easy tasks, so a Gemini default
    was silently ignored under ROUTING_STRATEGY=three_phase."""
    from pantry_planner.router.three_phase import ThreePhaseRouter

    _set(monkeypatch, ROUTING_STRATEGY="three_phase",
         SELECTOR_MODEL_DEFAULT="gemini:gemini-3.1-flash-lite",
         CLASSIFIER_MODEL="gemini:gemini-3.1-flash-lite",
         SELECTOR_MODEL_ESCALATION="gemini:gemini-flash-latest")
    fakes = FakeProviders().install(monkeypatch)
    recipe, products = _tomato_penne()
    pre = ThreePhaseRouter().preselect_model(recipe, products)
    assert len(fakes.gemini) == 1 and fakes.anthropic == []        # the classifier
    assert pre.model in {"gemini:gemini-3.1-flash-lite", "gemini:gemini-flash-latest"}
    assert pre.routing_cost_usd == 0.0


def test_decide_defaults_are_unchanged():
    from pantry_planner.config import HAIKU
    from pantry_planner.models import PhaseAMetrics, PhaseBMetrics
    from pantry_planner.router.decision import decide

    a = PhaseAMetrics(ingredient_count=3, mean_max_similarity=0.7, min_max_similarity=0.6,
                      count_below_0_3=0, category_density=2.0)
    b = PhaseBMetrics(match_confidence_1_to_10=9, cost_complexity_1_to_10=2,
                      confidence_in_own_estimate_1_to_10=8)
    assert decide(a, b)[0] == HAIKU
    assert decide(a, b, default_model="gemini:g")[0] == "gemini:g"


# ─── API surface ─────────────────────────────────────────────

def test_parse_failure_is_a_502_naming_the_cause_not_a_422(monkeypatch):
    """A rejected key on the parse used to come back as "couldn't find an
    ingredient list" — blaming the user's text for the provider's error."""
    from fastapi.testclient import TestClient

    from pantry_planner.api import app

    _set(monkeypatch, NL2SQL_MODEL="gemini:gemini-flash-latest")
    _serve(monkeypatch, httpx.Response(401, json={"error": {"message": "bad key"}}))
    resp = TestClient(app).post("/plan/nl", json={"recipe_text": "- 200g spaghetti"})
    assert resp.status_code == 502, resp.text
    body = resp.json()
    assert body["error"] == "llm_call_failed"
    assert "Recipe parse failed (gemini:gemini-flash-latest)" in body["detail"]
    assert "GEMINI_API_KEY" in body["detail"] and KEY not in resp.text


def test_selector_quota_error_is_a_429_from_the_api(monkeypatch):
    from fastapi.testclient import TestClient

    from pantry_planner.api import app

    _set(monkeypatch, SELECTOR_MODEL_DEFAULT="gemini:gemini-3.1-flash-lite")
    _serve(monkeypatch, httpx.Response(429, headers={"Retry-After": "86400"},
                                       json={"error": {"message": "daily quota"}}))
    resp = TestClient(app).post("/plan/tomato_penne")
    assert resp.status_code == 429, resp.text
    assert "quota" in resp.json()["detail"]


def test_missing_gemini_key_is_a_503_from_the_api(monkeypatch):
    from fastapi.testclient import TestClient

    from pantry_planner.api import app

    _set(monkeypatch, GEMINI_API_KEY="", SELECTOR_MODEL_DEFAULT="gemini:gemini-x")
    resp = TestClient(app).post("/plan/tomato_penne")
    assert resp.status_code == 503, resp.text
    assert "GEMINI_API_KEY is not set" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_mcp_plan_tools_surface_the_llm_error(monkeypatch):
    """The MCP SDK masks unexpected exceptions as "Error executing tool";
    an LLM failure is passed through as a ToolError that says what to fix."""
    from mcp.server.mcpserver.exceptions import ToolError

    from pantry_planner.mcp_server import server

    _set(monkeypatch, GEMINI_API_KEY="", SELECTOR_MODEL_DEFAULT="gemini:gemini-x",
         NL2SQL_MODEL="gemini:gemini-x")
    with pytest.raises(ToolError, match="LLM call failed — GEMINI_API_KEY is not set"):
        await server.call_tool("plan_recipe", {"slug": "tomato_penne"})
    with pytest.raises(ToolError, match="Recipe parse failed .*GEMINI_API_KEY is not set"):
        await server.call_tool("plan_from_text", {"recipe_text": "- 200g spaghetti"})
    with pytest.raises(ToolError, match="LLM call failed — GEMINI_API_KEY is not set"):
        await server.call_tool("plan_week", {"days": 1})

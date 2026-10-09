"""
The one LLM boundary: a single forced-tool call, routed by model spec.

Every server-side LLM call here — recipe parse, product selector, the
three-phase classifier — has the same shape: a system prompt, one user
message, ONE tool the model must call, and the tool's arguments are the
result. forced_tool_call() is that shape, once:

  * "gemini:<model>"                  → Google's OpenAI-compatible endpoint
  * "anthropic:<model>" or a bare name → the Anthropic Messages API

The Anthropic request is exactly what the call sites sent before this
module existed: the same parameters, thinking only when a budget is
passed, the first tool_use block's input, usage read off the response, and
the SDK's own exceptions propagate untouched.

Gemini errors are raised as LLMError subclasses whose message leads with
what to fix (GEMINI_API_KEY, the model spec, the quota) — never the key.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any

import httpx
from anthropic import Anthropic

from . import httptrace
from .config import ANTHROPIC, GEMINI, estimate_cost_usd, is_priced, settings, split_model_spec

DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"

# Gemini's thinking tokens count against max_tokens. A call site's budget
# is sized for the tool arguments alone (Anthropic, no thinking), so the
# Gemini request gets a floor plus headroom for the configured reasoning
# effort — without it the reply can stop at finish_reason "length" before
# the tool call is written.
GEMINI_MIN_MAX_TOKENS = 2048
_THINKING_HEADROOM = {"none": 0, "minimal": 1024, "low": 2048,
                      "medium": 8192, "high": 24576}
_DEFAULT_HEADROOM = 8192               # effort omitted: the model's own default

GEMINI_MAX_ATTEMPTS = 4                # 429 and 5xx (503 "high demand" is common)
_BACKOFF_BASE_S = 1.0                  # 1s, 2s, 4s between the four attempts
_MAX_RETRY_AFTER_S = 60.0              # a longer Retry-After fails now, not later
_TIMEOUT = httpx.Timeout(120.0, connect=10.0)

# Test seams: tests swap in httpx.MockTransport and a no-op sleep.
_transport: httpx.BaseTransport | None = None
_sleep = time.sleep


class LLMError(RuntimeError):
    """An LLM call failed. The API maps it to `http_status`."""

    http_status = 502

    def __init__(self, message: str, *, provider: str = "", status: int | None = None):
        super().__init__(message)
        self.provider = provider
        self.status = status


class LLMConfigError(LLMError):
    """The call can't be made as configured (no key, empty model name)."""

    http_status = 503


class LLMAuthError(LLMError):
    """The provider rejected the key."""


class LLMModelNotFoundError(LLMError):
    """The provider doesn't know the model name."""


class LLMRateLimitError(LLMError):
    """429 after every retry, or a Retry-After too long to wait out."""

    http_status = 429


class LLMResponseError(LLMError):
    """The reply came back but holds no usable tool call."""


@dataclass(frozen=True)
class ToolCallResult:
    input: dict[str, Any]       # the forced tool's arguments
    input_tokens: int
    output_tokens: int          # Gemini: includes thinking tokens
    model: str                  # Anthropic: resp.model; Gemini: "gemini:<model>"
    provider: str
    # Gemini: where the call's time went, per attempt (httptrace.py); None for Anthropic.
    http: dict[str, Any] | None = None


def forced_tool_call(*, model: str, system: str, messages: list[dict],
                     tool: dict, max_tokens: int,
                     temperature: float | None = None,
                     thinking_budget: int | None = None) -> ToolCallResult:
    """Make one call that MUST invoke `tool`; return its arguments + usage.

    `tool` is an Anthropic tool definition (name, description,
    input_schema). `temperature` and `thinking_budget` are sent only when
    given, and only to Anthropic: Gemini thinks according to
    GEMINI_REASONING_EFFORT, and Gemini 3 is documented to degrade (loops,
    worse output) at temperatures below its default, so neither is sent.
    """
    provider, name = split_model_spec(model)
    if not name:
        raise LLMConfigError(f"model spec {model!r} names no model", provider=provider)
    if provider == GEMINI:
        result = _gemini_call(name, system=system, messages=messages, tool=tool,
                              max_tokens=max_tokens)
    else:
        result = _anthropic_call(name, system=system, messages=messages, tool=tool,
                                 max_tokens=max_tokens, temperature=temperature,
                                 thinking_budget=thinking_budget)
    # Every call counts toward the daily ceiling (limits.py), whichever endpoint made it.
    from . import limits

    if not is_priced(result.model):
        limits.warn_unpriced(result.model)
    limits.record_spend(estimate_cost_usd(result.model, result.input_tokens,
                                          result.output_tokens))
    return result


# ─── Anthropic ────────────────────────────────────────────────

def _anthropic_call(name: str, *, system: str, messages: list[dict], tool: dict,
                    max_tokens: int, temperature: float | None,
                    thinking_budget: int | None) -> ToolCallResult:
    client = Anthropic(api_key=settings().anthropic_api_key)
    kwargs: dict = {
        "model": name,
        "max_tokens": max_tokens,
        "system": system,
        "tools": [tool],
        "tool_choice": {"type": "tool", "name": tool["name"]},
        "messages": messages,
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    if thinking_budget is not None:
        kwargs["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}

    resp = client.messages.create(**kwargs)
    tool_block = next((b for b in resp.content if b.type == "tool_use"), None)
    if tool_block is None:
        raise LLMResponseError(
            f"{name} didn't call the tool {tool['name']!r}. Response: {resp.content!r}",
            provider=ANTHROPIC)
    return ToolCallResult(
        input=tool_block.input,
        input_tokens=resp.usage.input_tokens,
        output_tokens=resp.usage.output_tokens,
        model=resp.model,
        provider=ANTHROPIC,
    )


# ─── Gemini (OpenAI-compatible endpoint) ──────────────────────

def gemini_max_tokens(max_tokens: int, reasoning_effort: str) -> int:
    """The max_tokens a Gemini request carries for a call site's budget."""
    headroom = _THINKING_HEADROOM.get(reasoning_effort, _DEFAULT_HEADROOM)
    return max(max_tokens, GEMINI_MIN_MAX_TOKENS) + headroom


def gemini_request_body(name: str, *, system: str, messages: list[dict], tool: dict,
                        max_tokens: int, reasoning_effort: str) -> dict:
    """The chat/completions body: the forced tool as a function call."""
    chat: list[dict] = [{"role": "system", "content": system}] if system else []
    for m in messages:
        content = m["content"]
        if isinstance(content, list):          # Anthropic text blocks → one string
            content = "".join(b.get("text", "") for b in content
                              if isinstance(b, dict) and b.get("type") == "text")
        chat.append({"role": m["role"], "content": content})
    body: dict = {
        "model": name,
        "messages": chat,
        "tools": [{
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool["input_schema"],
            },
        }],
        "tool_choice": {"type": "function", "function": {"name": tool["name"]}},
        "max_tokens": gemini_max_tokens(max_tokens, reasoning_effort),
    }
    if reasoning_effort:
        body["reasoning_effort"] = reasoning_effort
    return body


def _gemini_call(name: str, *, system: str, messages: list[dict], tool: dict,
                 max_tokens: int) -> ToolCallResult:
    cfg = settings()
    key = cfg.gemini_api_key
    if not key:
        raise LLMConfigError(
            f"GEMINI_API_KEY is not set, and the model 'gemini:{name}' needs it. "
            "Set GEMINI_API_KEY, or point this model setting at an Anthropic model.",
            provider=GEMINI)
    url = (cfg.gemini_base_url or DEFAULT_GEMINI_BASE_URL).rstrip("/") + "/chat/completions"
    body = gemini_request_body(name, system=system, messages=messages, tool=tool,
                               max_tokens=max_tokens,
                               reasoning_effort=cfg.gemini_reasoning_effort)
    trace: dict[str, Any] = {"step": tool["name"]}
    data = _gemini_post(url, body, key=key, name=name, trace=trace)
    result = _gemini_tool_result(data, name=name, tool_name=tool["name"],
                                 max_tokens=body["max_tokens"])
    return replace(result, http=trace)


def _gemini_post(url: str, body: dict, *, key: str, name: str,
                 trace: dict[str, Any] | None = None) -> Any:
    """POST with retries on 429/5xx and on connections that never completed.

    Every attempt is timed phase by phase into `trace` (httptrace.py: DNS, connect, TLS,
    upload, waiting for Google, download, Google's own server-timing, the wait before a
    retry), and the whole call is logged, as a warning when it is slow, whether it succeeds
    or fails."""
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    trace = {} if trace is None else trace
    trace.update(provider=GEMINI, model=f"{GEMINI}:{name}", host=httpx.URL(url).host,
                 started_at=datetime.now(UTC).astimezone().isoformat(timespec="seconds"),
                 attempts=[])
    attempts: list[dict[str, Any]] = trace["attempts"]
    call_started = time.perf_counter()
    try:
        with httpx.Client(timeout=_TIMEOUT, transport=_transport) as client:
            for attempt in range(1, GEMINI_MAX_ATTEMPTS + 1):
                clock = httptrace.PhaseClock()
                # Under a test transport there is no network: nothing to resolve.
                dns_ms = httptrace.resolve_ms(url) if _transport is None else None
                started = time.perf_counter()
                try:
                    resp = client.post(url, json=body, headers=headers,
                                       extensions={"trace": clock})
                except (httpx.ConnectError, httpx.ConnectTimeout,
                        httpx.RemoteProtocolError) as e:
                    attempts.append(httptrace.attempt_record(
                        clock, n=attempt, started=started, dns_ms=dns_ms,
                        error=type(e).__name__, request=_request_of(e)))
                    if attempt == GEMINI_MAX_ATTEMPTS:
                        raise LLMError(
                            f"Gemini unreachable after {attempt} attempts "
                            f"({type(e).__name__}) — check network / GEMINI_BASE_URL",
                            provider=GEMINI) from e
                    attempts[-1]["backoff_ms"] = round(_backoff(attempt) * 1000)
                    _sleep(_backoff(attempt))
                    continue
                except httpx.TimeoutException as e:
                    attempts.append(httptrace.attempt_record(
                        clock, n=attempt, started=started, dns_ms=dns_ms,
                        error=type(e).__name__, request=_request_of(e)))
                    raise LLMError(
                        f"Gemini model {name!r} timed out ({type(e).__name__})",
                        provider=GEMINI) from e
                attempts.append(httptrace.attempt_record(
                    clock, n=attempt, started=started, dns_ms=dns_ms,
                    status=resp.status_code, request=resp.request, response=resp))
                if resp.status_code != 200:     # what Google said, e.g. "UNAVAILABLE: ..."
                    attempts[-1]["detail"] = _error_detail(resp)

                if resp.status_code == 200:
                    try:
                        return resp.json()
                    except ValueError as e:
                        raise LLMResponseError(
                            f"Gemini model {name!r} returned a non-JSON body: "
                            f"{_clip(resp.text)!r}", provider=GEMINI, status=200) from e

                if (resp.status_code == 429 or resp.status_code >= 500) \
                        and attempt < GEMINI_MAX_ATTEMPTS:
                    delay = _retry_delay(resp, attempt)
                    if delay is not None:
                        attempts[-1]["backoff_ms"] = round(delay * 1000)
                        _sleep(delay)
                        continue
                raise _gemini_http_error(resp, name=name, attempts=attempt, key=key)
        raise AssertionError("unreachable")  # pragma: no cover
    finally:
        trace["total_ms"] = round((time.perf_counter() - call_started) * 1000)
        httptrace.log_trace(trace, server="Google")


def _request_of(e: httpx.RequestError) -> httpx.Request | None:
    try:
        return e.request
    except RuntimeError:            # raised without a request attached
        return None


def _backoff(attempt: int) -> float:
    return _BACKOFF_BASE_S * 2 ** (attempt - 1)


def _retry_delay(resp: httpx.Response, attempt: int) -> float | None:
    """Seconds to wait before the next attempt; None = don't retry.

    A numeric Retry-After is honoured as given, unless it is longer than
    anyone should wait inside a request (then the error is raised now).
    An HTTP-date Retry-After is ignored in favour of the backoff."""
    raw = resp.headers.get("retry-after")
    if raw:
        try:
            seconds = float(raw)
        except ValueError:
            seconds = None
        if seconds is not None:
            return None if seconds > _MAX_RETRY_AFTER_S else max(seconds, 0.0)
    return _backoff(attempt)


def _error_detail(resp: httpx.Response) -> str:
    """The provider's own message. Bodies come as {"error": {...}} or as a
    list [{"error": {...}}]; anything else is clipped raw text."""
    try:
        body = resp.json()
    except ValueError:
        return _clip(resp.text) or resp.reason_phrase
    if isinstance(body, list):
        body = body[0] if body else {}
    err = body.get("error", body) if isinstance(body, dict) else body
    if isinstance(err, dict):
        msg = str(err.get("message") or "").strip()
        status = str(err.get("status") or err.get("code") or "").strip()
        if msg and status:
            return _clip(f"{status}: {msg}")
        return _clip(msg or status or json.dumps(err))
    return _clip(str(err))


def _gemini_http_error(resp: httpx.Response, *, name: str, attempts: int,
                       key: str) -> LLMError:
    status = resp.status_code
    detail = _error_detail(resp)
    if key:                                    # belt and braces: never echo the key
        detail = detail.replace(key, "***")
    tried = f"{attempts} attempt{'s' if attempts > 1 else ''}"
    key_rejected = status == 400 and any(
        s in detail.lower() for s in ("api_key_invalid", "api key not valid",
                                      "api key expired"))
    if status in (401, 403) or key_rejected:
        return LLMAuthError(
            f"Gemini rejected the API key (HTTP {status}) — check GEMINI_API_KEY: {detail}",
            provider=GEMINI, status=status)
    if status == 404:
        return LLMModelNotFoundError(
            f"Gemini model {name!r} not found (HTTP 404) — check the 'gemini:<model>' "
            "setting (SELECTOR_MODEL_DEFAULT / SELECTOR_MODEL_ESCALATION / "
            f"CLASSIFIER_MODEL / NL2SQL_MODEL): {detail}",
            provider=GEMINI, status=status)
    if status == 429:
        retry_after = resp.headers.get("retry-after")
        wait = f", Retry-After {retry_after}s" if retry_after else ""
        return LLMRateLimitError(
            f"Gemini rate limit / quota exceeded for {name!r} (HTTP 429, {tried}{wait}): "
            f"{detail}", provider=GEMINI, status=status)
    if status >= 500:
        return LLMError(
            f"Gemini server error for {name!r} (HTTP {status}, {tried}): {detail}",
            provider=GEMINI, status=status)
    return LLMError(
        f"Gemini rejected the request for {name!r} (HTTP {status}): {detail}",
        provider=GEMINI, status=status)


def _gemini_tool_result(data: Any, *, name: str, tool_name: str,
                        max_tokens: int) -> ToolCallResult:
    """choices[0].message.tool_calls[0].function.arguments → dict, + usage."""
    choices = data.get("choices") if isinstance(data, dict) else None
    if not choices:
        raise LLMResponseError(
            f"Gemini model {name!r} returned no choices: {_clip(json.dumps(data))}",
            provider=GEMINI)
    choice = choices[0] or {}
    finish = choice.get("finish_reason")
    message = choice.get("message") or {}
    calls = message.get("tool_calls") or []
    truncated = finish == "length"
    if not calls:
        if truncated:
            raise LLMResponseError(
                f"Gemini model {name!r} ran out of tokens before calling {tool_name!r} "
                f"(finish_reason=length, max_tokens={max_tokens}); thinking tokens count "
                "against max_tokens — lower GEMINI_REASONING_EFFORT or shorten the input",
                provider=GEMINI)
        raise LLMResponseError(
            f"Gemini model {name!r} didn't call the tool {tool_name!r} "
            f"(finish_reason={finish!r}). Content: {_clip(message.get('content'))!r}",
            provider=GEMINI)
    fn = next((c.get("function") or {} for c in calls
               if (c.get("function") or {}).get("name") == tool_name), None)
    if fn is None:
        called = [(c.get("function") or {}).get("name") for c in calls]
        raise LLMResponseError(
            f"Gemini model {name!r} called {called} instead of {tool_name!r}",
            provider=GEMINI)
    raw = fn.get("arguments")
    if isinstance(raw, dict):
        args: Any = raw
    else:
        try:
            args = json.loads(raw) if raw else None
        except (TypeError, ValueError) as e:
            cut = " (finish_reason=length: the reply was cut off at max_tokens)" \
                if truncated else ""
            raise LLMResponseError(
                f"Gemini model {name!r} sent invalid JSON arguments for "
                f"{tool_name!r}{cut}: {e}", provider=GEMINI) from e
    if not isinstance(args, dict):
        raise LLMResponseError(
            f"Gemini model {name!r} sent {tool_name!r} arguments that are not a JSON "
            f"object: {_clip(raw)!r}", provider=GEMINI)

    usage = data.get("usage") or {}
    prompt = int(usage.get("prompt_tokens") or 0)
    total = usage.get("total_tokens")
    # total - prompt counts the thinking tokens, which completion_tokens may not
    output = int(total) - prompt if total is not None \
        else int(usage.get("completion_tokens") or 0)
    return ToolCallResult(input=args, input_tokens=prompt, output_tokens=max(output, 0),
                          model=f"{GEMINI}:{name}", provider=GEMINI)


def _clip(text: Any, limit: int = 300) -> str:
    s = "" if text is None else str(text)
    return s if len(s) <= limit else s[:limit] + "…"

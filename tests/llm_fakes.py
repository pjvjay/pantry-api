"""Fake LLM providers for tests: no network, no keys, every call recorded.

`FakeProviders.install(monkeypatch)` swaps BOTH boundaries inside
pantry_planner.llm — the Anthropic SDK class and the httpx transport the
Gemini path posts through — so a test can assert which provider a model
spec reached and exactly what was sent. Replies are built from the request
(each recipe line gets a product whose name shares a word with it), so the
real pipeline downstream of the boundary runs unchanged.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import httpx

PARSE_REPLY = {
    "recipe": {
        "title": "Garlic pasta",
        "servings": 2,
        "ingredients": [
            {"name": "spaghetti", "quantity": 200, "unit": "g", "category_hint": None},
            {"name": "garlic", "quantity": 3, "unit": "cloves"},
            {"name": "olive oil", "form": None, "prep": None},
        ],
    },
    "constraints": {"exclude_tags": [], "soft_text": ""},
}

TRIAGE_REPLY = {
    "match_confidence_1_to_10": 8,
    "cost_complexity_1_to_10": 3,
    "ambiguous_ingredients": [{"name": "garlic", "why": "fresh or powder"}],
    "confidence_in_own_estimate_1_to_10": 7,
    "reasoning": "mostly direct matches",
}


def _plan_reply(user_content: str, low_conf_lines: set[int]) -> dict:
    payload = json.loads(user_content)
    products = payload["available_products"]
    selections = []
    for ing in payload["recipe_ingredients"]:
        words = ing["name"].lower().split()
        pick = next((p for p in products
                     if any(w in p["name"].lower() for w in words)), products[0])
        selections.append({
            "line_no": ing["line_no"], "product_id": pick["id"],
            "confidence": 0.5 if ing["line_no"] in low_conf_lines else 0.95,
            "reasoning": "fake",
        })
    return {"selections": selections,
            "total_cost": round(sum(p["price"] for p in products[:1]), 2)}


class FakeProviders:
    def __init__(self) -> None:
        self.gemini: list[dict] = []          # request bodies
        self.gemini_headers: list[dict] = []
        self.anthropic: list[dict] = []       # messages.create kwargs
        self.anthropic_keys: list[str] = []
        self.low_conf_lines: dict[str, set[int]] = {"gemini": set(), "anthropic": set()}

    # ── reply builders ──
    def reply_for(self, provider: str, tool_name: str, user_content: str) -> dict:
        if tool_name == "submit_plan":
            return _plan_reply(user_content, self.low_conf_lines[provider])
        if tool_name == "submit_parse":
            return PARSE_REPLY
        if tool_name == "submit_triage":
            return TRIAGE_REPLY
        raise AssertionError(f"unexpected tool {tool_name}")

    # ── Gemini (httpx.MockTransport handler) ──
    def gemini_handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.gemini.append(body)
        self.gemini_headers.append(dict(request.headers))
        tool_name = body["tool_choice"]["function"]["name"]
        args = self.reply_for("gemini", tool_name, body["messages"][-1]["content"])
        return httpx.Response(200, json=gemini_reply(tool_name, args, model=body["model"]))

    # ── Anthropic (stands in for anthropic.Anthropic) ──
    def anthropic_class(self):
        fakes = self

        class _Client:
            def __init__(self, api_key=None, **_kw):
                fakes.anthropic_keys.append(api_key)
                self.messages = self

            def create(self, **kwargs):
                fakes.anthropic.append(kwargs)
                tool_name = kwargs["tool_choice"]["name"]
                args = fakes.reply_for("anthropic", tool_name,
                                       kwargs["messages"][-1]["content"])
                return SimpleNamespace(
                    content=[SimpleNamespace(type="text", text="ok"),
                             SimpleNamespace(type="tool_use", name=tool_name, input=args)],
                    usage=SimpleNamespace(input_tokens=1000, output_tokens=200),
                    model=kwargs["model"],
                )

        return _Client

    def install(self, monkeypatch) -> FakeProviders:
        from pantry_planner import llm

        monkeypatch.setattr(llm, "_transport", httpx.MockTransport(self.gemini_handler))
        monkeypatch.setattr(llm, "Anthropic", self.anthropic_class())
        monkeypatch.setattr(llm, "_sleep", lambda _s: None)
        return self


def gemini_reply(tool_name: str, args, *, model: str = "gemini-test",
                 finish: str = "tool_calls", prompt_tokens: int = 100,
                 total_tokens: int = 180, completion_tokens: int = 50) -> dict:
    """An OpenAI-compatible chat.completion carrying one tool call. `args`
    is serialised to JSON unless it is already a string."""
    arguments = args if isinstance(args, str) else json.dumps(args)
    return {
        "id": "x", "object": "chat.completion", "model": model,
        "choices": [{
            "index": 0, "finish_reason": finish,
            "message": {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call_0", "type": "function",
                "function": {"name": tool_name, "arguments": arguments},
            }]},
        }],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                  "total_tokens": total_tokens},
    }

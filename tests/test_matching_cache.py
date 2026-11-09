"""The matcher's database cache and daily budget (matching/cache.py): a name described once,
an identity embedded once, every instance counting against one allowance."""
from __future__ import annotations

import json
import threading
import uuid

import httpx
import pytest

from pantry_planner.matching import cache
from tests.mealplan_fixtures import done_db, use_db


@pytest.fixture(scope="module", autouse=True)
def seeded_db(tmp_path_factory):
    use_db(tmp_path_factory, "matching_cache")
    yield
    done_db()


@pytest.fixture(autouse=True)
def own_day_and_names(monkeypatch):
    """On a shared Postgres the tables outlive a test run: each test counts under its own day
    and embeds its own texts, so nothing an earlier run wrote can answer."""
    day = f"test-{uuid.uuid4().hex[:12]}"
    monkeypatch.setattr(cache, "today", lambda: day)
    yield


def _u(text: str) -> str:
    return f"{text} #{uuid.uuid4().hex[:8]}"


def _ident(n, ptype="butter"):
    return {"name": n, "normalized": n.lower(), "product_type": ptype, "form": "block",
            "variant": "", "look_alikes": ("ghee",), "identity": f"{ptype} | block | ",
            "model": "gemini:test", "prompt_version": 1, "fingerprint": "abc", "status": "auto"}


def test_an_answer_survives_the_process_and_an_older_row_is_never_replaced():
    k = cache.identity_key(1, "gemini:test", _u("abc"), "clarified butter")
    assert cache.identities.get_many([k]) == {}
    cache.identities.put_many({k: _ident("clarified butter", "ghee")})
    cache.identities.memo.clear()                           # another process
    got = cache.identities.get_many([k])[k]
    assert (got["product_type"], got["look_alikes"], got["status"]) == ("ghee", ("ghee",), "auto")
    assert cache.identities.state()["db_hits"] == 1
    # a second instance writing the same key (a race, or a reviewed row) changes nothing
    cache.identities.put_many({k: _ident("clarified butter", "butter")})
    cache.identities.memo.clear()
    assert cache.identities.get_many([k])[k]["product_type"] == "ghee"


def test_keys_carry_the_version_the_model_and_the_catalog():
    a = cache.identity_key(1, "gemini:test", "abc", "butter")
    assert a != cache.identity_key(2, "gemini:test", "abc", "butter")
    assert a != cache.identity_key(1, "gemini:other", "abc", "butter")
    assert a != cache.identity_key(1, "gemini:test", "def", "butter")
    assert cache.embedding_key("m", 768, "x") != cache.embedding_key("m", 256, "x")


def test_embeddings_round_trip_as_json_text():
    text = _u("butter | block | salted")
    k = cache.embedding_key("gemini-embedding-001", 768, text)
    v = {"model": "gemini-embedding-001", "dims": 768, "text": text,
         "vector": [0.5, -0.25] + [0.0] * 766}
    cache.embeddings.put_many({k: v})
    cache.embeddings.memo.clear()
    assert cache.embeddings.get_many([k])[k]["vector"] == v["vector"]


def test_the_budget_is_counted_across_instances_and_refuses_before_a_call(monkeypatch):
    from pantry_planner import config

    monkeypatch.setenv("MATCHING_DAILY_EMBED_TEXTS", "25")
    config.settings.cache_clear()
    cache.reset()
    b = cache.budget()
    b.spend("embed_texts", 20)
    assert (b.used("embed_texts"), b.left("embed_texts")) == (20, 5)
    cache.reset()                                           # another instance starts
    assert cache.budget().used("embed_texts") == 20, "the day's count is in the database"
    with pytest.raises(cache.BudgetExceededError, match="allowance of 25 embed texts"):
        cache.budget().spend("embed_texts", 6)
    assert cache.budget().used("embed_texts") == 20, "a refused spend counts nothing"
    cache.budget().spend("embed_texts", 5)
    assert cache.budget().left("embed_texts") == 0
    config.settings.cache_clear()


def test_an_allowance_of_zero_is_no_limit(monkeypatch):
    from pantry_planner import config

    monkeypatch.setenv("MATCHING_DAILY_DESCRIBE_CALLS", "0")
    config.settings.cache_clear()
    cache.reset()
    cache.budget().spend("describe_calls", 10_000)
    assert cache.budget().state()["describe_calls"] == {"used": 10_000, "allowance": 0}
    config.settings.cache_clear()


def test_the_gemini_embedder_asks_once_per_text_whatever_the_process(monkeypatch):
    from pantry_planner.matching import embed

    asked: list[list[str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        texts = json.loads(request.content)["input"]
        asked.append(texts)
        return httpx.Response(200, json={"data": [{"embedding": [1.0] + [0.0] * 767}
                                                  for _ in texts]})

    make = lambda: embed.GeminiEmbedder("gemini-embedding-001", "k", "https://g.test",  # noqa: E731
                                        transport=httpx.MockTransport(handler))
    a, b, c = _u("a | fresh |"), _u("b | fresh |"), _u("c | fresh |")
    e = make()
    e.embed([a, b])
    e.embed([b, c])
    assert asked == [[a, b], [c]]
    cache.embeddings.memo.clear()
    make().embed([a, c])
    assert len(asked) == 2, "a new process read them from the database"
    assert cache.budget().used("embed_texts") == 3


def test_a_spent_allowance_makes_the_embedder_unavailable_with_the_reason(monkeypatch):
    from pantry_planner import config
    from pantry_planner.matching import embed

    monkeypatch.setenv("MATCHING_DAILY_EMBED_TEXTS", "1")
    config.settings.cache_clear()
    cache.reset()
    e = embed.GeminiEmbedder("gemini-embedding-001", "k", "https://g.test",
                             transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    with pytest.raises(embed.EmbedderUnavailableError, match="allowance of 1 embed texts"):
        e.embed([_u("x"), _u("y")])
    config.settings.cache_clear()


def test_two_threads_asking_for_the_same_name_share_one_model_call(monkeypatch):
    from pantry_planner import llm
    from pantry_planner.matching import describe

    calls = []
    gate = threading.Barrier(2)

    def slow(*, model, system, messages, tool, max_tokens, **_kw):
        calls.append(json.loads(messages[0]["content"])["names"])
        return llm.ToolCallResult(input={"items": [{"name": "kale", "item": "kale",
                                                   "catalog_type": "kale", "relation": "same",
                                                   "form": "fresh", "variant": ""}]},
                                  input_tokens=1, output_tokens=1, model=model, provider="gemini")

    monkeypatch.setattr(llm, "forced_tool_call", slow)
    d = describe.Describer(f"gemini:test-{uuid.uuid4().hex[:8]}")
    out = {}

    def run(i):
        gate.wait()
        out[i] = d.describe(["kale"])["kale"].product_type

    ts = [threading.Thread(target=run, args=(i,)) for i in range(2)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert out == {0: "kale", 1: "kale"} and len(calls) == 1


def test_an_older_database_without_the_tables_serves_from_memory(monkeypatch):
    from sqlalchemy.exc import OperationalError

    def missing(*_a, **_k):
        raise OperationalError("select", {}, Exception("no such table: embedding_cache"))

    monkeypatch.setattr(cache.Session, "execute", missing)
    k = cache.embedding_key("m", 768, _u("x"))
    assert cache.embeddings.get_many([k]) == {}
    assert cache.embeddings.table_missing is True
    cache.embeddings.put_many({k: {"model": "m", "dims": 768, "text": "x", "vector": [1.0]}})
    assert cache.embeddings.get_many([k])[k]["vector"] == [1.0]
    assert cache.embeddings.state()["table"] == "missing"

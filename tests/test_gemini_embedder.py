"""Gemini's embedding API as the matcher's embedder (embed.GeminiEmbedder), over a mocked
endpoint: what it sends, what it keeps, and when it refuses."""
from __future__ import annotations

import json

import httpx
import pytest


def _gemini(answer, status=200):
    from pantry_planner.matching import embed

    asked: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        asked.append({"path": request.url.path, "auth": request.headers.get("authorization"),
                      **body})
        if status != 200:
            return httpx.Response(status, json={"error": {"code": status}})
        return httpx.Response(200, json={"data": [{"embedding": answer(t)} for t in body["input"]],
                                         "model": body["model"]})

    return embed.GeminiEmbedder("gemini-embedding-001", "test-key", "https://gemini.test/v1beta/openai",
                                transport=httpx.MockTransport(handler)), asked


def test_it_sends_every_text_in_one_request_and_keeps_the_vectors():
    from pantry_planner.matching import embed

    e, asked = _gemini(lambda t: [3.0, 4.0] + [0.0] * 766)
    a, b = e.embed(["butter | block | salted", "ghee | jarred | "])
    assert asked == [{"path": "/v1beta/openai/embeddings", "auth": "Bearer test-key",
                      "model": "gemini-embedding-001",
                      "input": ["butter | block | salted", "ghee | jarred | "],
                      "dimensions": embed.GEMINI_DIMS}]
    assert len(a) == embed.GEMINI_DIMS and a[:2] == [0.6, 0.8]       # unit length
    assert e.embed(["ghee | jarred | "]) == [b] and len(asked) == 1  # kept in memory


def test_a_refusal_makes_it_unavailable_for_a_while_without_asking_again():
    from pantry_planner.matching import embed

    e, asked = _gemini(lambda t: [1.0] * 768, status=429)
    for text in ("kale", "leeks"):
        with pytest.raises(embed.EmbedderUnavailableError, match="HTTP 429"):
            e.embed([text])
    assert len(asked) == 1


def _clock(monkeypatch):
    """A clock the embedder reads and a sleep that only advances it."""
    from pantry_planner.matching import embed

    now = {"t": 1000.0}
    monkeypatch.setattr(embed, "_monotonic", lambda: now["t"])
    monkeypatch.setattr(embed, "_sleep", lambda s: now.__setitem__("t", now["t"] + s))
    return now


def test_the_minutes_allowance_is_paced_and_a_call_beyond_it_keeps_what_it_got(monkeypatch):
    """Gemini counts texts, 100 a minute on the free tier. In the app (max_wait_s 0) a call
    for more than fits sends what fits, keeps those vectors, and fails for the rest with the
    wait named; the kept ones are then served from the cache without a request."""
    from pantry_planner.matching import embed

    now = _clock(monkeypatch)
    e, asked = _gemini(lambda t: [1.0] * 768)
    e.per_minute, e.max_wait_s = 5, 0.0
    texts = [f"pace {i} | jar | plain" for i in range(7)]
    with pytest.raises(embed.EmbedderUnavailableError,
                       match=r"allowance of 5 texts is spent, room for 2 more in 60s; "
                             r"5 of 7 texts were embedded and kept"):
        e.embed(texts)
    assert [len(a["input"]) for a in asked] == [5]
    assert len(e.embed(texts[:5])) == 5 and len(asked) == 1      # cached, no request
    now["t"] += 60.0                                              # the minute has passed
    assert len(e.embed(texts)) == 7 and [len(a["input"]) for a in asked] == [5, 2]


def test_a_script_waits_for_the_minutes_room_instead(monkeypatch):
    from pantry_planner.matching import embed

    now = _clock(monkeypatch)
    e, asked = _gemini(lambda t: [1.0] * 768)
    e.per_minute, e.max_wait_s = 5, 600.0
    before = now["t"]
    assert len(e.embed([f"wait {i} | jar | plain" for i in range(12)])) == 12
    assert [len(a["input"]) for a in asked] == [5, 5, 2]
    assert now["t"] - before == pytest.approx(120.0)             # slept out two minutes
    assert embed.GEMINI_BATCH == 20 and e.per_minute == 5         # chunks never exceed the room


def test_a_refusal_naming_a_delay_is_honoured_and_explained(monkeypatch):
    from pantry_planner.matching import embed

    now = _clock(monkeypatch)
    asked: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        asked.append(1)
        if len(asked) == 1:
            return httpx.Response(429, json={"error": {"code": 429, "message": (
                "You exceeded your current quota.\n* Quota exceeded for metric: embed_content"
                "_free_tier_requests, limit: 100 ... per minute ...\nPlease retry in 7.5s.")}})
        n = len(json.loads(request.content)["input"])
        return httpx.Response(200, json={"data": [{"embedding": [1.0] * 768}] * n})

    e = embed.GeminiEmbedder("gemini-embedding-001", "k", "https://gemini.test/v1beta/openai",
                             transport=httpx.MockTransport(handler))
    with pytest.raises(embed.EmbedderUnavailableError,
                       match="HTTP 429: Quota exceeded for metric: embed_content_free_tier_"
                             "requests, limit: 100"):
        e.embed(["kale | bunch | "])
    now["t"] += 7.0                                               # still within the delay
    with pytest.raises(embed.EmbedderUnavailableError, match="HTTP 429"):
        e.embed(["kale | bunch | "])
    assert len(asked) == 1
    now["t"] += 1.0                                               # the delay has passed
    assert len(e.embed(["kale | bunch | "])) == 1 and len(asked) == 2


def test_a_script_waits_out_a_named_delay_and_asks_again(monkeypatch):
    """Gemini's count of the minute can run ahead of ours: with max_wait_s a 429 naming a
    delay is slept out and the same chunk is asked again, once; the day's allowance is charged
    for both attempts (what Gemini may have counted)."""
    from pantry_planner.matching import cache, embed

    now = _clock(monkeypatch)
    asked: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        n = len(json.loads(request.content)["input"])
        asked.append(n)
        if len(asked) == 1:
            return httpx.Response(429, json={"error": {"code": 429, "message": (
                "Quota exceeded for metric: embed_content_free_tier_requests, limit: 100, "
                "model: gemini-embedding-1.0, description: ... per minute ... See x. "
                "Please retry in 12.5s.")}})
        return httpx.Response(200, json={"data": [{"embedding": [1.0] * 768}] * n})

    e = embed.GeminiEmbedder("gemini-embedding-001", "k", "https://gemini.test/v1beta/openai",
                             transport=httpx.MockTransport(handler), max_wait_s=600.0)
    before = now["t"]
    assert len(e.embed(["retry 1 | jar | ", "retry 2 | jar | "])) == 2
    assert asked == [2, 2] and now["t"] - before == pytest.approx(12.5)
    assert cache.budget().used("embed_texts") >= 4
    e.max_wait_s = 0.0                                            # the app: no second try
    asked.clear()
    with pytest.raises(embed.EmbedderUnavailableError,
                       match=r"HTTP 429: Quota exceeded for metric: \S+, limit: 100"):
        e.embed(["retry 3 | jar | "])
    assert asked == [1]


def test_a_refusal_naming_the_days_quota_is_never_retried_and_lasts_until_it_resets(monkeypatch):
    """Gemini counts refused attempts too, so after the day's quota a script must stop asking:
    nothing is sent again until the delay Gemini named (hours) has passed."""
    from pantry_planner.matching import embed

    now = _clock(monkeypatch)
    asked: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        asked.append(len(json.loads(request.content)["input"]))
        return httpx.Response(429, json={"error": {"code": 429, "message": (
            "You exceeded your current quota. * Quota exceeded for metric: generativelanguage."
            "googleapis.com/embed_content_free_tier_requests, limit: 1000, model: gemini-"
            "embedding-1.0, description: Embed content requests per day limit exceeded on Free "
            "Tier. See https://x. Please retry in 21h17m36.850534475s.")}})

    e = embed.GeminiEmbedder("gemini-embedding-001", "k", "https://gemini.test/v1beta/openai",
                             transport=httpx.MockTransport(handler), max_wait_s=600.0)
    with pytest.raises(embed.EmbedderUnavailableError, match="limit: 1000.*per day"):
        e.embed(["day 1 | jar | "])
    assert asked == [1] and e.retry_at - now["t"] == pytest.approx(21 * 3600 + 17 * 60 + 36.85)
    now["t"] += 3600.0
    with pytest.raises(embed.EmbedderUnavailableError, match="per day"):
        e.embed(["day 2 | jar | "])
    assert asked == [1], "nothing was sent again"


def test_no_key_means_no_request():
    from pantry_planner.matching import embed

    e = embed.GeminiEmbedder("gemini-embedding-001", "")
    assert e.ping() == (False, "GEMINI_API_KEY is empty")
    with pytest.raises(embed.EmbedderUnavailableError, match="GEMINI_API_KEY is empty"):
        e.embed(["kale"])


def test_it_is_never_compared_with_seeds_another_model_made():
    """The committed default seeds are EmbeddingGemma's: Gemini's vectors are refused there."""
    e, asked = _gemini(lambda t: [1.0] * 768)
    assert e.agrees_with_seeds() == (
        False, "the seeds were embedded with 'embeddinggemma', not 'gemini-embedding-001'")
    assert asked == []


def test_embedding_model_gemini_chooses_it(monkeypatch):
    from pantry_planner import config
    from pantry_planner.matching import embed

    monkeypatch.setenv("EMBEDDING_MODEL", "gemini:gemini-embedding-001")
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    config.settings.cache_clear()
    embed.reset_embedder()
    try:
        e = embed.get_embedder()
        assert isinstance(e, embed.GeminiEmbedder) and e.model == "gemini-embedding-001"
    finally:
        config.settings.cache_clear()
        embed.set_embedder(None)


def test_without_vectors_for_the_embedding_model_nothing_is_matched_by_meaning(monkeypatch):
    """EMBEDDING_MODEL names Gemini but seeds/gemini_vectors.json is missing (or made from other
    product rows): seed names included, every name is matched by words, which keeps peanut butter
    out of butter's Same item, and /health says mode "words" with the reason."""
    from pantry_planner import config
    from pantry_planner.matching import matcher, seeds

    monkeypatch.setenv("EMBEDDING_MODEL", "gemini:gemini-embedding-001")
    monkeypatch.setattr(seeds, "GEMINI_VECTORS", "no-such-file.json")
    config.settings.cache_clear()
    seeds.clear_cache()
    try:
        s = seeds.load()
        assert s.profiles and not s.vectors and not s.centroids
        m = matcher.meaning_of("Butter")
        assert (m.source, m.matching) == ("words", "words")
        assert "no vectors" in m.why_words or "missing" in m.why_words
        assert matcher.candidates(m, {}) == []
        assert matcher.state()["mode"] == "words"
    finally:
        config.settings.cache_clear()
        seeds.clear_cache()

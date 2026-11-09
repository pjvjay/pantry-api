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

"""A live embedder must reproduce the seeds' vectors before the matcher compares its vectors
with them (OllamaEmbedder.agrees_with_seeds): another build of the model, another
quantization or another prompt would move every name a little and the fixed floors with it."""
from __future__ import annotations

import httpx
import pytest


def _probe():
    from pantry_planner.matching import seeds

    return min((i for i in seeds.load().ingredients.values() if i.vector), key=lambda i: i.name)


def _ollama(tmp_path, answer, model="embeddinggemma"):
    """An OllamaEmbedder whose server answers every text with `answer(text)`, recording what
    it was asked."""
    from pantry_planner.matching import embed

    asked: list[list[str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": f"{model}:latest"}]})
        texts = json.loads(request.content)["input"]
        asked.append(texts)
        return httpx.Response(200, json={"embeddings": [list(answer(t)) for t in texts]})

    return embed.OllamaEmbedder("http://ollama.test", model, tmp_path,
                                transport=httpx.MockTransport(handler)), asked


def test_the_seeds_own_model_and_prompt_agree_once_and_names_are_then_embedded(tmp_path):
    from pantry_planner.matching import embed, matcher

    probe = _probe()
    e, asked = _ollama(tmp_path, lambda t: probe.vector)
    embed.set_embedder(e)
    m = matcher.meaning_of("saffron threads")
    assert m.source == "embedding"
    assert asked[0] == [f"task: search result | query: {probe.name}"], "the probe first"
    assert asked[1] == ["task: search result | query: saffron threads"]
    matcher.meaning_of("kale")
    assert [a for a in asked if probe.name in a[0]] == [asked[0]], "probed once per process"
    assert matcher.state()["embedder_matches_seeds"] is True


def test_another_build_of_the_model_is_not_compared_with_the_seeds(tmp_path):
    from pantry_planner.matching import embed, matcher
    from tests.fake_embedder import far_vector

    e, asked = _ollama(tmp_path, lambda t: far_vector())
    embed.set_embedder(e)
    m = matcher.meaning_of("saffron threads")
    assert m.source == "words" and "does not reproduce the seeds' vectors" in m.why_words
    assert len(asked) == 1, "only the probe: no name is embedded with a model that disagrees"
    st = matcher.state()
    assert (st["embedder_matches_seeds"], st["unknown_names"]) == (False, "words")
    assert "does not reproduce" in st["detail"]


def test_a_model_the_seeds_do_not_name_is_refused_without_asking(tmp_path):
    from pantry_planner.matching import embed, matcher

    e, asked = _ollama(tmp_path, lambda t: _probe().vector, model="nomic-embed-text")
    embed.set_embedder(e)
    assert matcher.meaning_of("saffron threads").source == "words"
    assert asked == []
    assert e.agrees_with_seeds() == (
        False, "the seeds were embedded with 'embeddinggemma', not 'nomic-embed-text'")


def test_the_probe_never_reads_the_disk_cache(tmp_path):
    """A vector cached from an older build under the same name must not vouch for a new one."""
    from pantry_planner.matching import embed
    from tests.fake_embedder import far_vector

    probe = _probe()
    old, _ = _ollama(tmp_path, lambda t: probe.vector)
    old.embed([probe.name], "query")                       # cached on disk
    new, asked = _ollama(tmp_path, lambda t: far_vector())
    assert new.cached(probe.name) is not None
    ok, why = new.agrees_with_seeds()
    assert not ok and asked == [[f"task: search result | query: {probe.name}"]]
    assert embed.PROBE_MIN == 0.99


def test_an_unreachable_embedder_is_probed_again_once_it_answers(tmp_path):
    from pantry_planner.matching import embed

    e, _ = _ollama(tmp_path, lambda t: _probe().vector)
    real = e._request
    calls = {"n": 0}

    def flaky(prompts):
        calls["n"] += 1
        if calls["n"] == 1:
            raise embed.EmbedderUnavailableError("ollama.test is unreachable (ConnectError)")
        return real(prompts)

    e._request = flaky
    with pytest.raises(embed.EmbedderUnavailableError):
        e.agrees_with_seeds()
    assert e.agrees_with_seeds() == (True, "")

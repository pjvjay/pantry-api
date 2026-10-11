"""The matcher (pantry_planner/matching/matcher.py) and its embedder seam, with no database:
what a name means (its seed, its embedding, or words), its candidates by group, the hard
parts of the score, and the Ollama embedder's prompts, truncation, cache and failure.

No test reaches Ollama: the embedder is tests/fake_embedder.py, an httpx mock, or none.
Expected values come from the seed files, never from the matcher under test."""
from __future__ import annotations

import json

import httpx
import pytest

from tests.fake_embedder import DownEmbedder, FakeEmbedder
from tests.seed_catalog import SEED_PRODUCTS

PRODUCTS = {p["id"]: p for p in SEED_PRODUCTS}
BUTTER, PEANUT_BUTTER, PBJ, EVOO, OLIVE_OIL_1L = 18, 6, 5, 16, 62


def _catalog():
    from pantry_planner.models import Product

    return {p["id"]: Product(**{k: v for k, v in p.items() if k in Product.model_fields})
            for p in SEED_PRODUCTS}


def _groups(name: str, form: str | None = None) -> dict[str, list[int]]:
    from pantry_planner.matching import matcher

    out: dict[str, list[int]] = {g: [] for g in matcher.GROUPS}
    for c in matcher.candidates(matcher.meaning_of(name, form), _catalog()):
        out[c.group].append(c.product_id)
    return out


# ─── Groups ──────────────────────────────────────────────────

def test_butter_never_gets_peanut_butter_as_the_same_item():
    groups = _groups("Butter")
    assert groups["same_item"] == [BUTTER]
    assert PEANUT_BUTTER not in sum(groups.values(), [])      # unrelated: not even looser
    assert PBJ not in groups["same_item"]


def test_olive_oil_gets_extra_virgin_olive_oil_as_the_same_item():
    groups = _groups("olive oil")
    assert set(groups["same_item"]) == {EVOO, OLIVE_OIL_1L}
    for pid in groups["works_instead"]:
        assert "olive" not in PRODUCTS[pid]["name"].lower()


def test_a_name_with_its_form_is_looked_up_with_it():
    """The parser splits "1 can canned tomatoes" into the name "tomatoes" and the form
    "canned": that is the seed "Canned Tomatoes", all three cans the same item."""
    from pantry_planner.matching import matcher

    m = matcher.meaning_of("tomatoes", "canned")
    assert (m.source, m.form) == ("seed", "canned")
    same = _groups("tomatoes", "canned")["same_item"]
    assert {PRODUCTS[pid]["name"] for pid in same} == {
        "Canned Diced Tomatoes", "Crushed Tomatoes Canned 796ml", "Petite Diced Tomatoes 796ml"}


def test_a_product_the_seeds_do_not_know_is_matched_by_its_words():
    """A row the catalog gained after the seeds were made, or one whose row changed since
    its profile was written, is not matched by meaning; when every word of the name is in
    its own, it is the same item, after the profiled ones."""
    from pantry_planner.matching import matcher
    from pantry_planner.models import Product

    catalog = _catalog()
    catalog[9100] = Product(id=9100, name="Garlic Bulb Pack", description="fresh garlic",
                            price=1.0)
    catalog[13] = catalog[13].model_copy(update={"description": "changed since the seeds"})
    found = matcher.candidates(matcher.meaning_of("Garlic"), catalog)
    same = [c for c in found if c.group == "same_item"]
    assert [c.product_id for c in same] == [13, 9100]           # 13: by its words now
    assert all(c.relevance == 0.0 for c in same)


def test_the_pack_fit_component():
    from pantry_planner.matching import matcher
    from pantry_planner.models import Product

    def pack(qty, uom="g"):
        return Product(id=1, name="x", description="", price=1, unit_qty=qty, unit_uom=uom)

    assert matcher.pack_fit(pack(300), (300, "g")) == 1.0               # exactly
    assert matcher.pack_fit(pack(1800), (300, "g")) == 0.6              # CATERING x
    assert matcher.pack_fit(pack(1801), (300, "g")) == 0.25             # a catering pack
    assert matcher.pack_fit(pack(200), (300, "g")) == 0.3               # short
    assert matcher.pack_fit(pack(200), (300, "g"), packs=2) == pytest.approx(0.973333, 1e-5)
    assert matcher.pack_fit(pack(500, "ml"), (300, "g")) == 0.5         # another unit
    assert matcher.pack_fit(pack(500), None) == 0.5                     # no need
    # 5 g of salt: every box is catering-size, so boxes up to 6x the smallest are normal
    assert matcher.pack_fit(pack(1000), (5, "g"), smallest=750) == 0.6
    assert matcher.pack_fit(pack(5000), (5, "g"), smallest=750) == 0.25


def test_the_score_is_the_documented_weighted_sum():
    from pantry_planner.matching import matcher

    assert sum(matcher.WEIGHTS.values()) == pytest.approx(1.0)
    f = matcher.Facts(relevance=0.5, form_variant=1.0, pack_fit=0.6, value=0.8, origin=0.0,
                      trip=1.0)
    assert f.score() == pytest.approx(sum(w * getattr(f, k)
                                          for k, w in matcher.WEIGHTS.items()))


# ─── Names no seed knows ─────────────────────────────────────

def test_an_imported_name_maps_through_embeddings_to_the_nearest_product_types():
    from pantry_planner.matching import embed, matcher

    fake = FakeEmbedder.mapping(**{"unsalted butter": "butter"})
    embed.set_embedder(fake)
    m = matcher.meaning_of("unsalted butter")
    assert (m.source, m.matching, m.exact) == ("embedding", "meaning", ("butter",))
    assert m.nearest == ("butter", 1.0)
    assert _groups("unsalted butter")["same_item"] == [BUTTER]
    assert fake.asked == [("unsalted butter", "query"), ("unsalted butter", "query")]
    # a name far from every product type is not stocked
    saffron = matcher.meaning_of("saffron threads")
    assert (saffron.source, saffron.exact, saffron.substitutes) == ("embedding", (), ())
    assert not saffron.stocked


def test_with_the_embedder_down_a_name_no_seed_knows_is_matched_by_words():
    from pantry_planner.matching import embed, matcher

    embed.set_embedder(DownEmbedder())
    m = matcher.meaning_of("unsalted butter")
    assert (m.source, m.matching) == ("words", "words")
    assert "down-embedder is unreachable" in m.why_words
    assert matcher.candidates(m, _catalog()) == []      # the caller matches it by words
    assert matcher.meaning_of("Butter").source == "seed"   # a seed needs no embedder
    embed.set_embedder(None)
    assert "OLLAMA_URL is empty" in matcher.meaning_of("unsalted butter").why_words


def test_the_ollama_embedder_asks_with_task_prompts_truncates_and_caches(tmp_path):
    from pantry_planner.matching import embed

    asked: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        asked.append(body)
        return httpx.Response(200, json={"embeddings": [[3.0, 4.0] + [0.0] * 766
                                                        for _ in body["input"]]})

    e = embed.OllamaEmbedder("http://ollama.test", "embeddinggemma", tmp_path,
                             transport=httpx.MockTransport(handler))
    (v,) = e.embed(["butter"], "query")
    assert asked == [{"model": "embeddinggemma",
                      "input": ["task: search result | query: butter"]}]
    assert len(v) == embed.DIMS and v[:2] == [0.6, 0.8]           # re-normalized
    (d,) = e.embed(["butter (block): spread."], "document", titles=["butter"])
    assert asked[-1]["input"] == ["title: butter | text: butter (block): spread."]
    again = embed.OllamaEmbedder("http://ollama.test", "embeddinggemma", tmp_path,
                                 transport=httpx.MockTransport(handler))
    assert again.embed(["butter"], "query") == [v] and len(asked) == 2   # from the disk


def test_an_unreachable_ollama_is_not_asked_again_for_a_minute(tmp_path):
    from pantry_planner.matching import embed

    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        raise httpx.ConnectError("refused")

    e = embed.OllamaEmbedder("http://ollama.test", "embeddinggemma", tmp_path,
                             transport=httpx.MockTransport(handler))
    for name in ("saffron", "kale"):
        with pytest.raises(embed.EmbedderUnavailableError, match="unreachable"):
            e.embed([name], "query")
    assert len(calls) == 1

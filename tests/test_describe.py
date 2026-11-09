"""Names no seed knows, identified by a low-cost model before they are matched (describe.py),
and what the matcher does with that identity (matcher._described). No model is called: the one
LLM boundary, llm.forced_tool_call, answers from a table."""
from __future__ import annotations

import json

import pytest

from tests.fake_embedder import FakeEmbedder
from tests.mealplan_fixtures import done_db, use_db


@pytest.fixture(scope="module", autouse=True)
def seeded_db(tmp_path_factory):
    use_db(tmp_path_factory, "describe")
    yield
    done_db()

BUTTER, PEANUT_BUTTER, GHEE = 18, 6, 95


@pytest.fixture
def answers(monkeypatch):
    """name -> the describer's fields; every call is recorded."""
    from pantry_planner import llm

    table: dict[str, dict] = {}
    asked: list[list[str]] = []

    def fake(*, model, system, messages, tool, max_tokens, **_kw):
        names = json.loads(messages[0]["content"])["names"]
        asked.append(names)
        assert tool["name"] == "submit_identities"
        assert "Catalog types:" in system and '"butter"' in system
        from pantry_planner.matching.describe import normalize

        items = [{"name": n, **table[normalize(n)]} for n in names if normalize(n) in table]
        return llm.ToolCallResult(input={"items": items}, input_tokens=100, output_tokens=40,
                                  model=model, provider="gemini")

    monkeypatch.setattr(llm, "forced_tool_call", fake)
    return table, asked


def _ident(item, catalog_type="", form="", variant="", relation=None):
    """The describer's answer for a name (prompt version 2's fields); the relation defaults to
    "same" when a catalog type is given and "none" otherwise."""
    return {"item": item, "catalog_type": catalog_type, "form": form, "variant": variant,
            "relation": relation or ("same" if catalog_type else "none")}


def _groups(name):
    from pantry_planner.matching import matcher
    from tests.test_matcher import _catalog

    out = {g: [] for g in matcher.GROUPS}
    for c in matcher.candidates(matcher.meaning_of(name), _catalog()):
        out[c.group].append(c.product_id)
    return out


def test_a_described_name_is_the_catalog_type_the_describer_names(answers):
    from pantry_planner.matching import describe, matcher

    table, asked = answers
    table["clarified butter"] = _ident("clarified butter", "ghee", "jarred")
    describe.set_describer(describe.Describer("gemini:test-lite"))
    m = matcher.meaning_of("clarified butter")
    assert (m.source, m.matching, m.exact) == ("described", "meaning", ("ghee",))
    assert m.form == "jarred"
    assert asked == [["clarified butter"]]
    ident = describe.get_describer().describe(["clarified butter"])["clarified butter"]
    assert (ident.item, ident.product_type, ident.text) == (
        "clarified butter", "ghee", "clarified butter | jarred | ")
    groups = _groups("clarified butter")
    assert groups["same_item"] == [GHEE]
    # what works instead is the ghee seed's own list, and peanut butter is never offered
    assert PEANUT_BUTTER not in sum(groups.values(), [])


def test_a_name_the_catalog_does_not_stock_has_no_same_item(answers):
    """The describer names an unstocked item by its own name ("lamb shoulder"), which is no
    catalog type, so nothing is offered as the same item: pork shoulder at most works instead."""
    from pantry_planner.matching import describe, embed, matcher, seeds

    table, _ = answers
    table["lamb shoulder"] = _ident("lamb shoulder", "", "raw")
    describe.set_describer(describe.Describer("gemini:test-lite"))
    embed.set_embedder(FakeEmbedder.mapping(**{"lamb shoulder": "pork shoulder roast"}))
    m = matcher.meaning_of("lamb shoulder")
    assert m.source == "described" and m.exact == ()
    assert "pork shoulder roast" in m.substitutes
    assert "lamb shoulder" not in seeds.load().types()


def test_the_answer_is_validated_against_the_catalog(answers):
    """A catalog_type that is not a type exactly leaves the item's own name standing (so it is
    never read as a type); a form off the catalog's list is dropped, and a variant that merely
    repeats the form is dropped; a cached row gives the same Identity back."""
    from pantry_planner.matching import describe

    table, asked = answers
    table["cooking onion"] = _ident("onion", "Yellow Onions ", "whole", "yellow")
    table["fresh thyme"] = _ident("thyme", "dried thyme leaves", "bunch", "fresh")
    table["thyme sprigs"] = _ident("thyme", "", "fresh", "fresh")
    d = describe.Describer("gemini:test-lite")
    got = d.describe(["cooking onion", "fresh thyme", "thyme sprigs"])
    assert (got["cooking onion"].product_type, got["cooking onion"].form,
            got["cooking onion"].variant) == ("yellow onions", "whole", "yellow")
    assert (got["fresh thyme"].product_type, got["fresh thyme"].form) == ("thyme", "")
    assert (got["thyme sprigs"].form, got["thyme sprigs"].variant) == ("fresh", "")
    assert got["thyme sprigs"].text == "thyme | fresh | "
    from pantry_planner.matching import cache

    cache.identities.memo.clear()                     # the database row, not the memo
    again = describe.Describer("gemini:test-lite").describe(["fresh thyme"])
    assert again["fresh thyme"] == got["fresh thyme"] and len(asked) == 1


def test_a_stand_in_is_offered_as_working_instead_never_as_the_same_item(answers):
    """The relation outlet: a catalog type the describer calls a substitute is not the Same
    item (the item's own name stands), but it is the first thing that works instead, before
    what the vector finds; a cached row keeps it."""
    from pantry_planner.matching import cache, describe, embed, matcher

    table, asked = answers
    table["self-raising flour"] = _ident("self-raising flour", "all-purpose flour", "",
                                         "self-raising", relation="substitute")
    describe.set_describer(describe.Describer("gemini:test-lite"))
    embed.set_embedder(FakeEmbedder.mapping(**{"self-raising flour": "sugar"}))
    m = matcher.meaning_of("self-raising flour")
    assert m.source == "described" and m.exact == ()
    assert m.substitutes[:2] == ("all-purpose flour", "sugar")
    ident = describe.get_describer().describe(["self-raising flour"])["self-raising flour"]
    assert (ident.product_type, ident.substitute) == ("self-raising flour", "all-purpose flour")
    cache.identities.memo.clear()
    assert describe.Describer("gemini:test-lite").describe(
        ["self-raising flour"])["self-raising flour"] == ident and len(asked) == 1


def test_a_field_the_model_ran_into_the_next_one_is_cut_there(answers):
    """Seen once in 400 answers: "white fish fillets,form:" as the catalog type."""
    from pantry_planner.matching import describe

    table, _ = answers
    table["basa fillets"] = _ident("basa", "white fish fillets,form:", "frozen", "",
                                   relation="variety")
    got = describe.Describer("gemini:test-lite").describe(["basa fillets"])["basa fillets"]
    assert (got.product_type, got.form) == ("white fish fillets", "frozen")


def test_the_model_is_called_at_temperature_zero_without_a_reasoning_effort(monkeypatch):
    from pantry_planner import llm
    from pantry_planner.matching import describe

    seen = {}

    def fake(**kw):
        seen.update(kw)
        return llm.ToolCallResult(input={"items": []}, input_tokens=1, output_tokens=1,
                                  model=kw["model"], provider="gemini")

    monkeypatch.setattr(llm, "forced_tool_call", fake)
    describe.Describer("gemini:test-lite").describe(["kale"])
    assert (seen["gemini_temperature"], seen["gemini_reasoning_effort"]) == (0.0, "")


def test_a_name_is_described_once_whatever_its_case_or_spacing(answers):
    from pantry_planner.matching import describe

    table, asked = answers
    table["caster sugar"] = _ident("caster sugar", "sugar", "", "caster")
    d = describe.Describer("gemini:test-lite")
    first = d.describe(["Caster  Sugar"])
    again = d.describe(["caster sugar", "CASTER SUGAR."])
    assert asked == [["Caster  Sugar"]], "the cache answered the second time"
    assert first["Caster  Sugar"].text == "caster sugar |  | caster"
    assert first["Caster  Sugar"].product_type == "sugar"
    assert set(again) == {"caster sugar", "CASTER SUGAR."}
    assert (d.calls, d.tokens_in, d.tokens_out) == (1, 100, 40)


def test_an_unanswered_name_is_asked_again_and_a_failure_is_not_cached(answers, monkeypatch):
    from pantry_planner import llm
    from pantry_planner.matching import describe

    table, asked = answers
    d = describe.Describer("gemini:test-lite")
    assert d.describe(["harissa"]) == {}                 # the model skipped it
    table["harissa"] = _ident("harissa", "", "paste")
    assert d.describe(["harissa"])["harissa"].product_type == "harissa"
    assert asked == [["harissa"], ["harissa"]]

    def down(**_kw):
        raise llm.LLMError("Gemini: 503 high demand")

    monkeypatch.setattr(llm, "forced_tool_call", down)
    with pytest.raises(describe.DescriberUnavailableError, match="unavailable"):
        d.describe(["leeks"])
    assert d.describe(["harissa"])["harissa"].form == "paste", "the cached answer still serves"


def test_when_the_describer_is_down_the_name_is_embedded_as_it_is(monkeypatch):
    from pantry_planner import llm
    from pantry_planner.matching import describe, embed, matcher

    def down(**_kw):
        raise llm.LLMError("quota")

    monkeypatch.setattr(llm, "forced_tool_call", down)
    describe.set_describer(describe.Describer("gemini:test-lite"))
    embed.set_embedder(FakeEmbedder.mapping(**{"unsalted butter": "butter"}))
    m = matcher.meaning_of("unsalted butter")
    assert (m.source, m.exact) == ("embedding", ("butter",))
    embed.set_embedder(None)
    assert matcher.meaning_of("plain flour").source == "words"


def test_a_type_far_from_its_identity_string_is_offered_as_working_instead(answers, monkeypatch):
    """The embedding check: when the describer files a name under a type its identity string is
    nowhere near (below floors()["check"]), the type is demoted to Works instead."""
    from pantry_planner.matching import describe, embed, matcher

    table, _ = answers
    table["smoked salmon"] = _ident("smoked salmon", "salmon fillets", "sliced", "smoked")
    describe.set_describer(describe.Describer("gemini:test-lite"))
    embed.set_embedder(FakeEmbedder())                    # far from every type
    base = matcher.floors()
    monkeypatch.setattr(matcher, "floors", lambda: {**base, "check": 0.5})
    m = matcher.meaning_of("smoked salmon")
    assert m.exact == () and m.substitutes[:1] == ("salmon fillets",)
    monkeypatch.setattr(matcher, "floors", lambda: {**base, "check": 0.0})
    assert matcher.meaning_of("smoked salmon").exact == ("salmon fillets",)


def test_the_prompt_asks_for_identity_never_flavour_or_use():
    from pantry_planner.matching import describe

    text = describe.system_prompt(["butter", "ghee"], ["block", "fresh"])
    assert "never flavour or how it is used" in text
    assert '"butter"' in text and '"block"' in text
    assert describe.normalize("  Fresh   Dill, ") == "fresh dill"


def test_health_reports_the_describer(answers):
    from pantry_planner.matching import describe, matcher

    assert matcher.state()["describer"]["model"] is None
    describe.set_describer(describe.Describer("gemini:test-lite"))
    answers[0]["kale"] = _ident("kale", "kale", "fresh")
    describe.get_describer().describe(["kale"])
    assert matcher.state()["describer"] == {"model": "gemini:test-lite", "calls": 1, "cached": 1}

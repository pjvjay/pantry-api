"""The cascade escalates on the matcher's signals as well as on stated confidence
(router/signals.py): a pick that leaves the Same item group, or the candidates, is re-run
whatever confidence the model gave it."""
from __future__ import annotations

import json
import os

import pytest

from tests.llm_fakes import FakeProviders


def _result(*picks):
    from pantry_planner.models import Selection, SelectorResult

    return SelectorResult(selections=[Selection(line_no=n, product_id=p, confidence=c)
                                      for n, p, c in picks], total_cost=0.0)


ORDER = {1: [(10, "same_item"), (11, "same_item"), (12, "works_instead")],
         2: [(20, "works_instead"), (21, "looser_match")]}


def test_a_pick_outside_the_same_item_group_or_the_candidates_is_flagged():
    from pantry_planner.router.signals import matcher_flags

    r = _result((1, 11, 0.9), (2, 21, 0.99))
    assert matcher_flags(r, ORDER) == {}, "another same item, and a line with no same item"
    assert matcher_flags(_result((1, 12, 0.99)), ORDER) == {1: "left_same_item"}
    assert matcher_flags(_result((1, 99, 0.99)), ORDER) == {1: "off_list"}
    assert matcher_flags(_result((3, 99, 0.99)), ORDER) == {}, "a line matched by words"
    assert matcher_flags(_result((1, 12, 0.99)), None) == {}


def test_flags_join_the_cascades_own_decision_without_repeating_a_line():
    from pantry_planner.flow import _with_matcher_flags
    from pantry_planner.models import EscalationDecision

    none = EscalationDecision(escalate=False, reason="cascade: all met threshold")
    got = _with_matcher_flags(none, _result((1, 12, 0.99), (2, 20, 0.99)), ORDER)
    assert got.escalate and got.ingredients_to_rerun == [1]
    assert got.escalation_model and got.reason.endswith("matcher: line 1 left same item")
    some = EscalationDecision(escalate=True, ingredients_to_rerun=[1], escalation_model="m",
                              reason="cascade: 1/2 below threshold")
    got = _with_matcher_flags(some, _result((1, 12, 0.5), (2, 99, 0.99)), ORDER)
    assert got.ingredients_to_rerun == [1, 2] and got.escalation_model == "m"
    assert _with_matcher_flags(none, _result((1, 10, 0.99)), ORDER) is none


@pytest.fixture(scope="module")
def seeded_db(tmp_path_factory):
    os.environ["DB_URL"] = (os.environ.get("PANTRY_TEST_DB_URL")
                            or f"sqlite:///{tmp_path_factory.mktemp('db') / 'signals.db'}")
    from pantry_planner import config, db
    from pantry_planner.nlsearch import vocab

    config.settings.cache_clear()
    db.seed_from_json()
    vocab.clear_cache()
    yield
    config.settings.cache_clear()
    vocab.clear_cache()


@pytest.fixture
def live(monkeypatch, seeded_db):
    """A Gemini selector, an Anthropic escalation, the cascade, no demo mode."""
    from pantry_planner import config

    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key-0123456789")
    monkeypatch.setenv("SELECTOR_MODEL_DEFAULT", "gemini:gemini-3.1-flash-lite")
    monkeypatch.setenv("SELECTOR_MODEL_ESCALATION", "claude-sonnet-4-6")
    monkeypatch.setenv("ROUTING_STRATEGY", "cascade")
    monkeypatch.delenv("DEMO_MODE", raising=False)
    config.settings.cache_clear()
    yield FakeProviders().install(monkeypatch)
    config.settings.cache_clear()


def test_a_confident_pick_that_leaves_the_matchers_groups_is_escalated(live):
    """Every line answered with full confidence, but with a product the matcher does not
    call the same item: the cascade re-runs those lines on the escalation model."""
    from pantry_planner import flow

    seen: dict[int, list[int]] = {}
    real = live.reply_for

    def reply_for(provider, tool, content):
        reply = real(provider, tool, content)
        if tool == "submit_plan" and provider == "gemini":
            ids = [p["id"] for p in json.loads(content)["available_products"]]
            for s in reply["selections"]:
                # the last product offered: rarely the line's own same item
                s["product_id"], s["confidence"] = ids[-1], 1.0
                seen[s["line_no"]] = ids
        return reply

    live.reply_for = reply_for
    plan = flow.run("tomato_penne")
    assert plan.escalated is True
    assert len(live.anthropic) == 1, "one escalation call, though no line was unsure"
    rerun = json.loads(live.anthropic[0]["messages"][0]["content"])["recipe_ingredients"]
    assert rerun and {i["line_no"] for i in rerun} <= set(seen)


def test_confident_same_item_picks_are_not_escalated(live):
    from pantry_planner import flow

    plan = flow.run("tomato_penne")
    assert plan.escalated is False and live.anthropic == []

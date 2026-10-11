"""Product matching by meaning in the plan, its Options and the meal plan: a name no seed knows
planned through its embedding, the fallback to words when the embedder is down (and the answer
saying so), the hard rules a plan's diet adds, the event log and /health. The matcher alone is
tests/test_matcher.py; the seeds, tests/test_matching_seeds.py.

No test reaches Ollama or Gemini: the seeds are the committed files, and a name no seed knows
is embedded by tests/fake_embedder.py or not at all (conftest installs no embedder). Expected
values come from the seed files and the catalog, never from the matcher under test.
"""
from __future__ import annotations

import pytest

from tests.fake_embedder import DownEmbedder, FakeEmbedder
from tests.mealplan_fixtures import done_db, use_db

BUTTER = 18


@pytest.fixture(scope="module", autouse=True)
def seeded_db(tmp_path_factory):
    use_db(tmp_path_factory, "matching")
    yield
    done_db()


# ─── Names no seed knows ─────────────────────────────────────

def test_an_imported_name_is_planned_by_meaning_and_its_options_say_so():
    from pantry_planner import alternatives, flow
    from pantry_planner.matching import embed

    embed.set_embedder(FakeEmbedder.mapping(**{"unsalted butter": "butter"}))
    plan = flow.run_nl("Toast (serves 2)\n- 30 g unsalted butter\n- 2 slices sourdough\n",
                       allow_partial=True)
    (li,) = [li for li in plan.line_items if li.ingredient_name == "unsalted butter"]
    assert (li.product_id, li.matching, li.group, li.match) == (
        BUTTER, "meaning", "same_item", "exact")
    ranking = alternatives.rank_alternatives(plan.basis, li.line_no)
    assert (ranking.matching, ranking.matching_note) == ("meaning", "Matched by meaning.")
    assert ranking.items[0].product_id == BUTTER and ranking.items[0].current


def test_with_the_embedder_down_an_unknown_name_falls_back_to_words_and_says_so():
    from pantry_planner import alternatives, flow
    from pantry_planner.matching import embed, matcher

    down = DownEmbedder()
    embed.set_embedder(down)
    m = matcher.meaning_of("unsalted butter")
    assert (m.source, m.matching) == ("words", "words")
    assert "down-embedder is unreachable" in m.why_words
    # a seed needs no embedder
    assert matcher.meaning_of("Butter").source == "seed"
    plan = flow.run_nl("Toast (serves 2)\n- 30 g unsalted butter\n- 1 tbsp olive oil\n")
    lines = {li.ingredient_name: li for li in plan.line_items}
    assert lines["unsalted butter"].matching == "words"
    assert lines["unsalted butter"].product_id == BUTTER          # today's word matching
    assert lines["olive oil"].matching == "meaning"
    ranking = alternatives.rank_alternatives(plan.basis, lines["unsalted butter"].line_no)
    assert ranking.matching == "words"
    assert ranking.matching_note.startswith("Matched by words: down-embedder is unreachable")
    assert ranking.order[0] == "tier"                              # the word ranking's keys

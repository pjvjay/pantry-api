"""Product matching by meaning in the plan, its Options and the meal plan: a name no seed knows
planned through its embedding, the fallback to words when the embedder is down (and the answer
saying so), the hard rules a plan's diet adds, the event log and /health. The matcher alone is
tests/test_matcher.py; the seeds, tests/test_matching_seeds.py.

No test reaches Ollama or Gemini: the seeds are the committed files, and a name no seed knows
is embedded by tests/fake_embedder.py or not at all (conftest installs no embedder). Expected
values come from the seed files and the catalog, never from the matcher under test.
"""
from __future__ import annotations

import json

import pytest

from tests.fake_embedder import DownEmbedder, FakeEmbedder
from tests.mealplan_fixtures import PRODUCTS, client, done_db, starter_draft, use_db

BUTTER = 18
GREEK_YOGURT, COCONUT_YOGURT, SOUR_CREAM = 20, 42, 117


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


# ─── Rules ───────────────────────────────────────────────────

def _yogurt_bowl(**prefs) -> dict:
    return starter_draft({"yogurt_fruit_bowl": 2}, prefs=prefs)


@pytest.mark.parametrize("prefs", [{"avoid": ["dairy"]}, {"diet": "vegan"}])
def test_a_dairy_free_plan_never_sees_a_dairy_substitute(prefs):
    """The bowl's Plain Yogurt buys Plain Greek Yogurt (the plan's own pick, shown as
    current). Coconut yogurt works instead; sour cream would too, but it is dairy, which the
    plan leaves out: never a row, filtered with the reason, and refused as a pin."""
    from pantry_planner import alternatives
    from pantry_planner.models import Product

    draft = _yogurt_bowl(**prefs)
    key = "starter:yogurt_fruit_bowl"
    line = next(ln for ln in draft["resolved"][key]["lines"] if ln["name"] == "Plain Yogurt")
    assert line["product_id"] == GREEK_YOGURT
    r = client().post("/mealplan/alternatives",
                      json={"draft": draft, "product_id": GREEK_YOGURT})
    assert r.status_code == 200, r.text
    ranking = r.json()["ranking"]
    for it in ranking["items"]:
        if not it["current"]:
            p = Product(**{k: v for k, v in PRODUCTS[it["product_id"]].items()
                           if k in Product.model_fields})
            assert "dairy" not in alternatives.contains(p), it["product"]
    assert COCONUT_YOGURT in {it["product_id"] for it in ranking["items"]}
    held = {f["product_id"]: f for f in ranking["filtered"]}
    assert (held[SOUR_CREAM]["why"], held[SOUR_CREAM]["tags"]) == ("diet", ["dairy"])
    pin = client().post("/mealplan/basket", json={"draft": {
        **draft, "pins": {key: {str(line["line_no"]): SOUR_CREAM}}}})
    assert pin.status_code == 422
    assert "contains dairy, which this plan leaves out" in pin.json()["detail"]["detail"]


def test_a_plan_with_no_diet_offers_the_dairy_substitute():
    r = client().post("/mealplan/alternatives",
                      json={"draft": _yogurt_bowl(), "product_id": GREEK_YOGURT})
    items = {it["product_id"]: it for it in r.json()["ranking"]["items"]}
    assert items[SOUR_CREAM]["group"] == "works_instead"
    assert items[GREEK_YOGURT]["group"] == "same_item" and items[GREEK_YOGURT]["rank"] == 1


# ─── Events and health ───────────────────────────────────────

def test_what_was_shown_and_picked_is_logged_without_personal_data(tmp_path, monkeypatch):
    from pantry_planner import alternatives, config, flow
    from pantry_planner.matching import events
    from pantry_planner.models import Pin

    monkeypatch.setenv("PANTRY_DATA_DIR", str(tmp_path))
    config.settings.cache_clear()
    events.clear()
    try:
        plan = flow.run_nl("Toast (serves 2)\n- 30 g butter\n", lat=49.28, lon=-123.12)
        ranking = alternatives.rank_alternatives(plan.basis, 1)
        second = ranking.items[1]
        flow.reprice(plan.basis, [Pin(line_no=1, product_id=second.product_id)])
        rows = [json.loads(ln) for ln in
                (tmp_path / "matching" / "events.jsonl").read_text().splitlines()]
    finally:
        monkeypatch.delenv("PANTRY_DATA_DIR")
        config.settings.cache_clear()
    shown = [r for r in rows if r["event"] == "shown"]
    picked = [r for r in rows if r["event"] == "picked"]
    assert shown and shown[0]["ingredient"] == "butter" and shown[0]["matching"] == "meaning"
    assert shown[0]["items"][0] == {"product_id": BUTTER, "rank": 1, "group": "same_item",
                                    "score": ranking.items[0].score, "current": True}
    (pick,) = picked
    assert (pick["product_id"], pick["was"], pick["rank"], pick["group"]) == (
        second.product_id, BUTTER, 2, second.group)
    assert pick["passed_over"] == [{"product_id": BUTTER, "rank": 1, "group": "same_item"}]
    text = json.dumps(rows)
    for private in ("49.28", "123.12", "lat", "lon", "address", "Toast"):
        assert private not in text

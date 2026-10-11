"""The golden set for product matching (evals/substitutes_golden.json, labelled once by a
Gemini judge): matching by meaning must rank at least as well as today's word matching, and
above a floor set from its measured score, on the committed labels. No LLM: the labels are
committed, and both rankings are computed here from the seeded database and the seeds.

Measured on 2026-10-11 (evals/reports/substitutes.md): words nDCG@3 0.924, Hit@1 96%;
meaning nDCG@3 0.966, Hit@1 98%. The floors sit a little under the measured values, so a
change that loses a top-3 product somewhere fails, and a tie moving does not.
"""
from __future__ import annotations

import pytest

from tests.mealplan_fixtures import done_db, use_db

FLOOR_NDCG = 0.95
FLOOR_HIT = 0.95


@pytest.fixture(scope="module", autouse=True)
def seeded_db(tmp_path_factory):
    use_db(tmp_path_factory, "substitutes")
    yield
    done_db()


@pytest.fixture(scope="module")
def scored():
    from evals import substitutes_eval

    return substitutes_eval.score()


def test_the_golden_set_labels_every_seed_ingredient_name():
    from evals import substitutes_eval
    from pantry_planner.matching.seeds import seed_ingredient_names

    golden = substitutes_eval.load_golden()
    assert set(golden) == set(seed_ingredient_names())
    assert all(set(labels.values()) <= {"E", "S", "C", "I"} for labels in golden.values())


def test_meaning_ranks_at_least_as_well_as_words_and_above_the_floor(scored):
    words, meaning = scored["mean"]["words"], scored["mean"]["meaning"]
    assert meaning["ndcg"] >= words["ndcg"]
    assert meaning["ndcg"] >= FLOOR_NDCG, meaning
    assert meaning["hit"] >= FLOOR_HIT, meaning


def test_butter_by_meaning_never_ranks_peanut_butter_in_its_top_three(scored):
    """By words, butter's second product was Canadian Peanut Butter Cream, which the judge
    calls irrelevant; by meaning it is not a candidate at all."""
    row = next(r for r in scored["rows"] if r["name"] == "Butter")
    assert row["meaning"]["top"][0][1] == "E"
    assert all(label != "I" for _pid, label in row["meaning"]["top"])
    assert any(label == "I" for _pid, label in row["words"]["top"])


def test_ndcg_weighs_labels_as_the_esci_paper_does():
    from evals.substitutes_eval import ndcg

    labels = {1: "E", 2: "S", 3: "I", 4: "E"}
    assert ndcg([1, 4, 2], labels) == pytest.approx(1.0)
    assert ndcg([3, 2, 1], labels) < ndcg([2, 1, 3], labels) < ndcg([1, 2, 4], labels)
    assert ndcg([], labels) == 0.0
    assert ndcg([9], {9: "I"}) == 0.0            # nothing to gain: no ideal

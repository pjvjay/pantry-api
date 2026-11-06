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


def test_the_paired_bootstrap_is_reproducible_and_centred_on_the_difference():
    from evals.substitutes_eval import paired_bootstrap

    a = [0.5, 0.9, 1.0, 0.7, 0.8]
    same = paired_bootstrap(a, a, reps=500)
    assert same["diff"] == (0.0, 0.0) and same["p_not_better"] == 1.0
    up = paired_bootstrap(a, [x + 0.1 for x in a], reps=500)
    assert abs(up["diff_mean"] - 0.1) < 1e-9
    assert up["diff"][0] > 0 and up["p_not_better"] == 0.0
    assert paired_bootstrap(a, a[::-1], reps=500) == paired_bootstrap(a, a[::-1], reps=500)


def test_with_the_exact_item_missing_the_first_safe_offer_is_scored():
    from evals.substitutes_eval import missing_exact
    from pantry_planner.models import Product

    def p(pid, name, tags=""):
        return Product(id=pid, name=name, description="", price=1.0, category="pantry",
                       dietary_tags=tags)

    # ids no catalog product has: seeds/allergens.json is read by product id
    a, b, c, d, e = 9001, 9002, 9003, 9004, 9005
    catalog = {a: p(a, "Penne", "gluten"), b: p(b, "Gluten-Free Penne"),
               c: p(c, "Spaghetti", "gluten"), d: p(d, "Rice Noodles"), e: p(e, "Pesto", "nuts")}
    labels = {a: "E", b: "E", c: "S", d: "S", e: "C"}
    # the plan bought Penne (this method's first E): spaghetti adds nothing, so it is offered
    got = missing_exact([a, b, c, d, e], labels, catalog)
    assert got["sub"] and got["first"] == (c, "S") and not got["unsafe_before_rule"]
    # the plan bought the gluten-free penne: spaghetti adds gluten, so rice noodles come first
    got = missing_exact([b, a, c, d], labels, catalog)
    assert got["first"] == (d, "S") and got["unsafe_before_rule"]
    # nothing safe is left
    got = missing_exact([b, c, e], {b: "E", c: "S", e: "C"}, catalog)
    assert not got["offered"] and not got["sub"]
    assert missing_exact([c], {c: "S"}, catalog) == {"applies": False}


def test_with_the_exact_item_missing_meaning_offers_a_substitute_more_often_than_words():
    """Measured 2026-11-05, with the reviewed allergens: words Sub@1 54%, meaning 74%
    (evals/reports/substitutes.md)."""
    from evals import substitutes_eval

    share = substitutes_eval.score_missing()["share"]
    assert share["meaning"]["sub"] >= share["words"]["sub"]
    assert share["meaning"]["sub"] >= 0.70, share

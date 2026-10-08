"""Quick add (POST /mealplan/selection/parse): counted dishes read with no LLM, matched exact,
plural, alias or fuzzy, and near misses left unmatched or ambiguous.

Expected matches come from the seed files' titles and aliases.
"""
from __future__ import annotations

import pytest

from tests.mealplan_fixtures import SENTENCE, client, done_db, use_db


@pytest.fixture(scope="module", autouse=True)
def seeded_db(tmp_path_factory):
    use_db(tmp_path_factory, "selection")
    yield
    done_db()


def _parse(text: str, **body) -> dict:
    r = client().post("/mealplan/selection/parse", json={"text": text, **body})
    assert r.status_code == 200, r.text
    return r.json()


def _one(text: str, **body) -> dict:
    (sel,) = _parse(text, **body)["selections"]
    return sel


def test_the_verbatim_sentence():
    out = _parse(SENTENCE)
    assert out["period_days"] == 14 and out["unmatched"] == [] and out["warnings"] == []
    got = [(s["count"], s["matched_as"]["title"], s["matched_as"]["how"])
           for s in out["selections"]]
    assert got == [(3, "Pepperoni Pizza", "exact"), (2, "Chicken Fried Rice", "exact"),
                   (3, "Chicken Biryani", "alias"), (7, "Mango Milkshake", "plural")]
    # Only exact and plural may be accepted without asking.
    assert [s["needs_confirmation"] for s in out["selections"]] == [False, False, True, False]
    assert [s["matched_as"]["recipe_key"] for s in out["selections"]] == [
        "starter:pepperoni_pizza", "starter:chicken_fried_rice", "starter:chicken_biryani",
        "starter:mango_milkshake"]
    # A count is a meal occasion at household servings.
    assert out["selections"][0]["meaning"] == "3 × Pepperoni Pizza = 3 dinners for 2 people"
    assert out["selections"][3]["meaning"] == "7 × Mango Milkshake = 7 snacks for 2 people"


@pytest.mark.parametrize("text", ["chicken curry", "fried chicken", "mango"])
def test_near_misses_stay_unmatched(text):
    sel = _one(text)
    assert sel["matched_as"] is None and sel["status"] == "unmatched"
    assert sel["needs_confirmation"] is True


def test_near_misses_offer_candidates_without_choosing_one():
    assert [c["title"] for c in _one("chicken curry")["candidates"]] == ["Simple Chicken Curry"]
    rice = _one("rice")
    assert rice["status"] == "ambiguous" and rice["matched_as"] is None
    assert sorted(c["title"] for c in rice["candidates"]) == [
        "Beef & Broccoli Rice Bowl", "Chicken Fried Rice"]


def test_pizza_alone_is_ambiguous_once_there_are_two_pizzas():
    alone = _one("pizza")
    assert alone["matched_as"] is None and alone["status"] == "unmatched"
    two = _one("pizza", recipes=[{"key": "my:7", "title": "Margherita Pizza"}])
    assert two["status"] == "ambiguous" and two["matched_as"] is None
    assert sorted(c["title"] for c in two["candidates"]) == ["Margherita Pizza",
                                                              "Pepperoni Pizza"]


def test_counts_and_periods():
    out = _parse("three pepperoni pizzas, 2x chicken biryani and ×3 mango milkshake "
                 "for a fortnight")
    assert out["period_days"] == 14
    assert [(s["count"], s["count_stated"], s["matched_as"]["title"]) for s in out["selections"]
            ] == [(3, True, "Pepperoni Pizza"), (2, True, "Chicken Biryani"),
                  (3, True, "Mango Milkshake")]
    assert _parse("2 grilled cheese in 10 days")["period_days"] == 10
    assert _parse("pepperoni pizza in three weeks")["period_days"] == 21
    assert _parse("pepperoni pizza in three weeks")["warnings"]
    assert _one("chicken biryani x4")["count"] == 4
    assert _one("2 × chicken biryani")["count"] == 2
    sel = _one("pepperoni pizza")
    assert (sel["count"], sel["count_stated"]) == (1, False)


def test_and_splits_only_before_a_count():
    out = _parse("peanut butter and jelly sandwich and 2 grilled cheese sandwiches")
    assert [(s["count"], s["matched_as"]["title"], s["matched_as"]["how"])
            for s in out["selections"]] == [
        (1, "Peanut Butter & Jelly Sandwich", "exact"), (2, "Grilled Cheese Sandwich", "plural")]


def test_a_slot_word_is_a_hint():
    sel = _one("2 mango milkshakes for breakfast")
    assert (sel["slot_hint"], sel["matched_as"]["title"]) == ("breakfast", "Mango Milkshake")
    assert sel["meaning"] == "2 × Mango Milkshake = 2 breakfasts for 2 people"


def test_fuzzy_matches_within_the_stated_distances_and_always_asks():
    sel = _one("chiken biryani")             # "chicken": 7 letters, one deletion
    assert (sel["matched_as"]["how"], sel["matched_as"]["distance"]) == ("fuzzy", 1)
    assert sel["needs_confirmation"] is True
    # A 5-letter title word allows 1 edit: "mngo" (1) matches, "mnga" (2) does not.
    assert _one("mngo milkshake")["matched_as"]["how"] == "fuzzy"
    assert _one("mnga milkshake")["matched_as"] is None
    # No extra input word may be left over.
    assert _one("spicy chiken biryani")["matched_as"] is None


def test_aliases_from_the_library_file_and_the_starters():
    assert _one("spag bol")["matched_as"]["title"] == "Spaghetti Bolognese"
    sel = _one("chicken biriyani")
    assert (sel["matched_as"]["how"], sel["matched_as"]["kind"]) == ("alias", "starter")


def test_the_shoppers_own_recipes_are_matched_too():
    sel = _one("2 nan's dal", recipes=[{"key": "my:3", "title": "Nan's Dal", "slot": "lunch"}])
    assert (sel["matched_as"]["recipe_key"], sel["matched_as"]["kind"]) == ("my:3", "my")
    assert sel["meaning"] == "2 × Nan's Dal = 2 lunches for 2 people"


def test_unknown_names_are_unmatched_and_listed():
    out = _parse("2 beef wellington + 3 pepperoni pizza")
    assert out["unmatched"] == ["beef wellington"]


def test_damerau_levenshtein_and_singularising():
    from pantry_planner.mealplan.selection import damerau_levenshtein as dl
    from pantry_planner.mealplan.selection import singular

    assert dl("briyani", "biryani") == 1          # one adjacent transposition
    assert dl("ca", "abc") == 2                    # unrestricted, not optimal string alignment
    assert dl("kitten", "sitting") == 3
    assert [singular(w) for w in ("berries", "tomatoes", "dishes", "boxes", "pizzas",
                                  "milkshakes", "glass", "bus", "cookies", "hummus")] == [
        "berry", "tomato", "dish", "box", "pizza", "milkshake", "glass", "bus", "cookie",
        "hummus"]

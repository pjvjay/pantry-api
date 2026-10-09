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


def test_a_slot_word_in_a_title_is_matched_as_written():
    """A bare slot word is tried as part of the name first: "Breakfast Burrito" and "Dinner
    Rolls" match as typed. After for, as or at it is always the hint."""
    mine = [{"key": "my:b", "title": "Breakfast Burrito"},
            {"key": "my:r", "title": "Dinner Rolls", "slot": "dinner"}]
    sel = _one("Breakfast Burrito", recipes=mine)
    assert (sel["status"], sel["matched_as"]["title"], sel["matched_as"]["how"],
            sel["slot_hint"]) == ("matched", "Breakfast Burrito", "exact", None)
    sel = _one("2 dinner rolls", recipes=mine)
    assert (sel["count"], sel["matched_as"]["title"], sel["slot_hint"]) == \
        (2, "Dinner Rolls", None)
    sel = _one("2 dinner rolls for dinner", recipes=mine)
    assert (sel["matched_as"]["title"], sel["slot_hint"]) == ("Dinner Rolls", "dinner")
    sel = _one("breakfast burrito as a snack", recipes=mine)
    assert (sel["matched_as"]["title"], sel["slot_hint"]) == ("Breakfast Burrito", "snack")
    # a bare slot word that is not in a title is still the hint
    sel = _one("2 mango milkshakes breakfast")
    assert (sel["matched_as"]["title"], sel["slot_hint"]) == ("Mango Milkshake", "breakfast")


def test_a_leading_days_of_is_the_period():
    out = _parse("10 days of grilled cheese")
    (sel,) = out["selections"]
    assert out["period_days"] == 10
    assert (sel["name"], sel["count"], sel["count_stated"], sel["status"]) == \
        ("grilled cheese", 1, False, "matched")
    assert _parse("two weeks of pepperoni pizza")["period_days"] == 14
    assert _parse("grilled cheese for 10 days")["period_days"] == 10


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


def test_a_household_of_one_is_one_person():
    sel = _one("3 pepperoni pizza", household_servings=1)
    assert sel["meaning"] == "3 × Pepperoni Pizza = 3 dinners for 1 person"


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


# ─── Bounded work: the route is public and calls no LLM ─────

def _reference_fuzzy(title: list[str], given: list[str]) -> int | None:
    """The permutation search the assignment replaced: every order of the input words."""
    import itertools

    from pantry_planner.mealplan.selection import allowed_distance, damerau_levenshtein

    if len(title) != len(given) or not title or len(title) > 7:
        return None
    best = None
    for perm in itertools.permutations(given):
        dists = [damerau_levenshtein(t, g) for t, g in zip(title, perm, strict=True)]
        if all(d <= allowed_distance(t) for d, t in zip(dists, title, strict=True)):
            best = sum(dists) if best is None else min(best, sum(dists))
    return best


def _reference_match(name: str, cands) -> tuple[list[tuple[str, int]], str]:
    """match_name as it was before the assignment, levels and all, as (key, distance)."""
    from pantry_planner.mealplan.selection import singular, words

    given = words(name)
    if not given:
        return [], ""
    g_set, g_sing = set(given), {singular(w) for w in given}

    def unalias(ws, token_aliases):
        back = {singular(v.casefold()): k for k, vs in token_aliases.items() for v in vs}
        return [back.get(w, w) for w in ws]

    levels = [
        ("exact", lambda c: set(words(c.title)) == g_set),
        ("plural", lambda c: {singular(w) for w in words(c.title)} == g_sing),
        ("alias", lambda c: any({singular(w) for w in words(a)} == g_sing for a in c.aliases)
         or {singular(w) for w in unalias([singular(w) for w in given], c.token_aliases)}
         == {singular(w) for w in words(c.title)}),
    ]
    for how, test in levels:
        found = [(c.key, 0) for c in cands if test(c)]
        if found:
            return found, how
    fuzzy = [(c.key, d) for c in cands
             if (d := _reference_fuzzy([singular(w) for w in words(c.title)],
                                       [singular(w) for w in given])) is not None]
    if fuzzy:
        low = min(d for _, d in fuzzy)
        return [(k, d) for k, d in fuzzy if d == low], "fuzzy"
    return [], ""


SMALL_FUZZY_CASES = [
    # realistic: the seeds' titles with slips, swapped words and plurals
    ("chicken biryani", "chiken biryani"), ("chicken biryani", "biryani chiken"),
    ("chicken fried rice", "fried chiken rice"), ("chicken fried rice", "chikcen fride rice"),
    ("mango milkshake", "mngo milkshake"), ("mango milkshake", "mnga milkshake"),
    ("pepperoni pizza", "pizza peperoni"), ("grilled cheese sandwich", "griled chese sandwich"),
    ("peanut butter jelly sandwich", "peanut buter jely sandwich"),
    ("beef broccoli rice bowl", "bowl rice brocoli beef"),
    # adversarial: repeated words, every pairing close, boundaries of each allowance
    ("abcdefg abcdefg abcdefg", "abcdefh abcdefh abcdefh"),
    ("abcdefg abcdefh abcdefg abcdefh", "abcdefh abcdefg abcdefh abcdefi"),
    ("aaaa bbbb aaaa", "aaab bbba aaaa"), ("abc abd", "abd abc"), ("abc abd", "abd abe"),
    ("abcd abce", "abce abcd"), ("abcd abcd", "abce abcf"), ("abcdef abcdeg", "abcdgf bacdef"),
    ("ca ca", "ca ac"), ("cabbage cabbages", "abcbage acbbage"), ("stew stew stew", "stew"),
    ("tikka masala curry", "masala tika curry"), ("tikka masala curry", "masala tikka"),
    ("a b c d e f g", "g f e d c b a"), ("abcdefgh hgfedcba", "hgfedcab abcdefhg"),
]


@pytest.mark.parametrize(("title", "given"), SMALL_FUZZY_CASES)
def test_the_assignment_agrees_with_the_permutation_search(title, given):
    from pantry_planner.mealplan.selection import _fuzzy, singular, words

    t = [singular(w) for w in words(title)]
    g = [singular(w) for w in words(given)]
    assert _fuzzy(t, g) == _reference_fuzzy(t, g)


def test_the_assignment_agrees_on_random_small_inputs():
    import random

    from pantry_planner.mealplan.selection import _fuzzy

    rng = random.Random(20261008)
    for _ in range(500):
        n = rng.randint(1, 5)
        title = ["".join(rng.choice("abc") for _ in range(rng.randint(3, 7))) for _ in range(n)]
        given = [w if rng.random() < 0.3 else
                 "".join(rng.choice("abc") for _ in range(max(1, len(w) + rng.randint(-1, 1))))
                 for w in rng.sample(title, n)]
        assert _fuzzy(title, given) == _reference_fuzzy(title, given), (title, given)


def test_the_matcher_agrees_with_the_old_one_on_whole_names():
    """Every level, the seeded library, the starters and a shopper's titles."""
    from pantry_planner import db
    from pantry_planner.mealplan.resolve import starters_file
    from pantry_planner.mealplan.selection import Matcher, candidates

    mine = [{"key": "my:1", "title": "Margherita Pizza"}, {"key": "my:2", "title": "Nan's Dal"},
            {"key": "my:3", "title": "Chicken Tikka Masala", "aliases": ["ctm"]},
            {"key": "my:4", "title": "Chicken Tikka Masala"}]
    cands = candidates([(r.slug, r.name) for r in db.load_all_recipes()],
                       starters_file()["starters"], mine)
    matcher = Matcher(cands)
    names = [s for _, s in SMALL_FUZZY_CASES] + [
        "pepperoni pizza", "pepperoni pizzas", "chicken briyani", "spag bol", "pizza",
        "chicken tika masala", "masala chicken tikka", "ctm", "nans dal", "nan's dhal",
        "margarita pizza", "rice", "mango", "chikcen fried rcie", "grilled chese sandwiches"]
    for name in names:
        found, how = matcher.match(name)
        assert ([(m.candidate.key, m.distance) for m in found], how) == \
            _reference_match(name, cands), name


def test_within_distance_is_the_distance_up_to_the_bound():
    import random

    from pantry_planner.mealplan.selection import damerau_levenshtein, within_distance

    rng = random.Random(7)
    for _ in range(20000):
        a = "".join(rng.choice("abc") for _ in range(rng.randint(0, 8)))
        b = "".join(rng.choice("abc") for _ in range(rng.randint(0, 8)))
        k = rng.randint(0, 3)
        full = damerau_levenshtein(a, b)
        assert within_distance(a, b, k) == (full if full <= k else None), (a, b, k)
    # a common prefix and suffix, and a transposition with a letter added between
    assert within_distance("x" + "a" * 30 + "y", "w" + "a" * 30 + "z", 2) == 2
    assert within_distance("pre" + "ca" + "post", "pre" + "abc" + "post", 2) == 2


def _timed_parse(text: str, recipes: list[dict]) -> tuple[float, dict]:
    """The request's time, not the app's: a test run on its own (-k) would otherwise time
    the first import of the API and its first request too, about a second."""
    import time

    _parse("pepperoni pizza", recipes=recipes)
    t0 = time.perf_counter()
    out = _parse(text, recipes=recipes)
    return time.perf_counter() - t0, out


def test_the_reviewers_worst_case_body_is_fast():
    """7-word items of one 7-letter word against 50 recipes of seven other 7-letter words, one
    edit apart: 5040 word orders per recipe per item before, about 6,900 s for 8 KB."""
    item = " ".join(["abcdefh"] * 7)
    text = "+".join([item] * ((8000 + 1) // (len(item) + 1)))
    assert 7900 < len(text) <= 8000
    mine = [{"key": f"my:{i}", "title": " ".join(["abcdefg"] * 7)} for i in range(50)]
    elapsed, out = _timed_parse(text, mine)
    assert elapsed < 1.0, elapsed
    # 20 dishes read, the rest named; every one fits all 50 recipes equally (distance 7)
    assert len(out["selections"]) == 20
    assert out["warnings"] == [
        f"Quick add reads 20 dishes at a time; 122 more were not read, from {item!r}"]
    first = out["selections"][0]
    assert first["status"] == "ambiguous" and len(first["candidates"]) == 50


def test_distinct_close_words_and_long_blank_runs_are_fast():
    """Every word different, so no distance is reused, and every pair two edits apart: the
    request-wide budget stops fuzzy matching and says from where. A long run of spaces no
    longer makes the patterns backtrack."""
    import itertools

    ends = itertools.product("bcdefghijklmnopqrtuvwxyz0123456789", repeat=2)

    def word():
        x, y = next(ends)
        return x + "a" * 25 + y

    mine = [{"key": f"my:{i}", "title": " ".join(word() for _ in range(7))} for i in range(50)]
    text = " + ".join(" ".join(word() for _ in range(7)) for _ in range(20))
    elapsed, out = _timed_parse(text, mine)
    assert elapsed < 1.0, elapsed
    assert any("too many to compare for spelling slips" in w for w in out["warnings"])
    for text in ("2 pepperoni" + " " * 7980 + "pizza", "for" + " " * 7990 + "x",
                 "pizza\t" + "\t" * 7990 + "x"):
        elapsed, out = _timed_parse(text, [])
        assert elapsed < 1.0, elapsed
    assert _one("2 pepperoni" + " " * 7980 + "pizza")["matched_as"]["title"] == \
        "Pepperoni Pizza"


def test_past_the_fuzzy_budget_a_dish_gets_no_fuzzy_match_and_a_warning(monkeypatch):
    from pantry_planner.mealplan import selection

    monkeypatch.setattr(selection, "MAX_FUZZY_ROWS", 0)
    out = _parse("2 pepperoni pizza + chiken biryani + mango milkshakes")
    pizza, biryani, shake = out["selections"]
    assert pizza["matched_as"]["how"] == "exact" and shake["matched_as"]["how"] == "plural"
    assert biryani["matched_as"] is None and biryani["status"] == "unmatched"
    assert out["warnings"] == [
        "from 'chiken biryani' on, names were matched exactly, as plurals or by alias only: "
        "there were too many to compare for spelling slips"]


def test_a_long_name_is_not_matched_and_said_so():
    long = "pepperoni pizza " + " ".join(f"word{i}" for i in range(11))
    out = _parse(long)
    (sel,) = out["selections"]
    assert sel["status"] == "unmatched" and sel["candidates"] == []
    assert out["warnings"] == [f"{long!r}: a dish name of more than 12 words is not matched"]


def test_the_same_recipe_sent_twice_is_one_candidate_and_same_titles_stay_ambiguous():
    same_key = [{"key": "my:7", "title": "Margherita Pizza"}] * 2
    sel = _one("margherita pizza", recipes=same_key)
    assert (sel["status"], sel["matched_as"]["recipe_key"]) == ("matched", "my:7")
    # two of the shopper's recipes with one title are two recipes: still asked, not picked
    two = [{"key": "my:7", "title": "Margherita Pizza"},
           {"key": "my:8", "title": "Margherita Pizza"}]
    sel = _one("margherita pizza", recipes=two)
    assert sel["status"] == "ambiguous" and sel["matched_as"] is None
    assert [c["recipe_key"] for c in sel["candidates"]] == ["my:7", "my:8"]

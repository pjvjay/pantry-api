"""nlsearch.lineparse: the demo parser's line reading, lifted with no behaviour change.

tests/fixtures/lineparse_golden.json was recorded from demomode.parse_recipe
BEFORE the per-line logic moved into lineparse.parse_line: 144 single lines
(every bullet line the suite used, plus amounts, units, forms, prep words,
notes in brackets and malformed lines) and 9 whole recipes (title, servings,
constraints). Both the lifted function and the demo parser that now calls it
must reproduce every recorded value, warts included ("2 x 400g cans
tomatoes" was never read well; changing that is a separate, visible change).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from pantry_planner import demomode
from pantry_planner.nlsearch import lineparse

GOLDEN = json.loads((Path(__file__).parent / "fixtures" / "lineparse_golden.json")
                    .read_text(encoding="utf-8"))


def _fields(x) -> dict:
    return {"name": x.name, "form": x.form, "quantity": x.quantity, "unit": x.unit,
            "prep": x.prep}


def test_the_golden_file_covers_what_it_claims():
    assert len(GOLDEN["lines"]) == 144
    assert len(GOLDEN["recipes"]) == 9
    assert all(g["parsed"] is not None for g in GOLDEN["lines"])


@pytest.mark.parametrize("golden", GOLDEN["lines"], ids=lambda g: g["line"])
def test_parse_line_reproduces_the_demo_parser(golden):
    assert _fields(lineparse.parse_line(golden["line"])) == golden["parsed"]
    # the same line under a bullet, as a pasted list writes it
    assert _fields(lineparse.parse_line("- " + golden["line"].strip())) == golden["parsed"]


@pytest.mark.parametrize("golden", GOLDEN["lines"], ids=lambda g: g["line"])
def test_demo_parser_lines_are_unchanged(golden):
    raw = golden["line"].strip()
    bullet = raw if raw.startswith(("-", "*", "•")) else "- " + raw
    (ing,) = demomode.parse_recipe("Golden\n" + bullet).recipe.ingredients
    assert _fields(ing) == golden["parsed"]


@pytest.mark.parametrize("golden", GOLDEN["recipes"], ids=lambda g: g["text"][:30])
def test_demo_parser_recipes_are_unchanged(golden):
    p = demomode.parse_recipe(golden["text"])
    got = {"title": p.recipe.title, "servings": p.recipe.servings,
           "ingredients": [_fields(i) for i in p.recipe.ingredients],
           "constraints": p.constraints.model_dump()}
    assert got == golden["parsed"]


def test_doc_name_puts_the_purchase_form_back():
    assert lineparse.parse_line("500g ground beef").doc_name == "ground beef"
    assert lineparse.parse_line("2 cups frozen peas").doc_name == "frozen peas"
    assert lineparse.parse_line("1 cup canned tomatoes").doc_name == "canned tomatoes"
    # a can says it already: "1 can crushed tomatoes" is crushed tomatoes, unit can
    assert lineparse.parse_line("1 can crushed tomatoes").doc_name == "crushed tomatoes"
    assert lineparse.parse_line("olive oil").doc_name == "olive oil"


def test_without_bullet_leaves_nothing_of_a_bullet_alone():
    assert lineparse.without_bullet("  - 2 eggs ") == "2 eggs"
    assert lineparse.without_bullet("• salt") == "salt"
    assert [lineparse.without_bullet(b) for b in ("-", "•", " * ", "- -")] == [""] * 4


@pytest.mark.parametrize("text, servings", [
    ("Serves 4", 4), ("serves 12 hungry people", 12), ("4 servings", 4),
    ("Makes 6 portions", 6), ("for 3 people", 3), ("4", 4), (" 2 ", 2),
    ("", None), ("a big pot", None), ("Makes 24 cookies", None), ("0", None),
])
def test_servings_from_yield_never_guesses(text, servings):
    assert lineparse.servings_from_yield(text) == servings


def test_parse_servings_is_the_demo_rule():
    assert lineparse.parse_servings("Soup (serves 3)") == 3
    assert lineparse.parse_servings("4 servings") is None


# ─── POST /recipes/parse-lines ───────────────────────────────

def _post(body: dict):
    from fastapi.testclient import TestClient

    from pantry_planner.api import app

    return TestClient(app).post("/recipes/parse-lines", json=body)


def test_parse_lines_reads_each_line_like_the_demo_parser():
    resp = _post({"title": "Pasta", "yield_text": "Serves 2",
                  "lines": ["500g penne", "", "- 2 cloves garlic, minced", "500g ground beef",
                            "1 can crushed tomatoes", "salt"]})
    assert resp.status_code == 200, resp.text
    out = resp.json()
    assert (out["servings"], out["servings_stated"]) == (2, True)
    assert [(ln["line_no"], ln["text"], ln["name"], ln["quantity"], ln["unit"], ln["note"])
            for ln in out["lines"]] == [
        (1, "500g penne", "penne", 500.0, "g", ""),
        (2, "- 2 cloves garlic, minced", "garlic", 2.0, "cloves", "minced"),
        (3, "500g ground beef", "ground beef", 500.0, "g", ""),
        (4, "1 can crushed tomatoes", "crushed tomatoes", 1.0, "can", ""),
        (5, "salt", "salt", None, "", "")]
    assert {ln["amount_basis"] for ln in out["lines"]} == {"parsed_from_your_paste"}
    assert out["warnings"] == ["1 blank line(s) dropped", "line 5 (salt) states no amount"]


def test_parse_lines_never_guesses_servings_and_labels_a_pages_amounts():
    out = _post({"lines": ["2 eggs"], "origin": "page"}).json()
    assert (out["servings"], out["servings_stated"]) == (None, False)
    assert out["warnings"] == ["servings not stated"]
    assert out["lines"][0]["amount_basis"] == "stated_by_source"
    # the title is read when the yield says nothing
    assert _post({"title": "Soup (serves 3)", "lines": ["1 onion"]}).json()["servings"] == 3


def test_a_bullet_alone_is_a_blank_line():
    out = _post({"yield_text": "Serves 2",
                 "lines": ["-", "500g penne", "•", " * ", "- 2 eggs"]}).json()
    assert [(ln["line_no"], ln["name"]) for ln in out["lines"]] == [(1, "penne"), (2, "eggs")]
    assert out["warnings"] == ["3 blank line(s) dropped"]


def test_a_catering_yield_is_not_stated_and_says_why():
    out = _post({"yield_text": "Makes 150 servings", "title": "Soup (serves 3)",
                 "lines": ["2 eggs"]}).json()
    # 150 is over what a RecipeDoc holds; the title's 3 is not read in its place, because
    # the yield did state a number
    assert (out["servings"], out["servings_stated"]) == (None, False)
    assert out["warnings"] == [
        "servings stated as 150, more than the 100 a plan takes: say how many you are "
        "cooking for"]
    assert _post({"yield_text": "Serves 100", "lines": ["2 eggs"]}).json()["servings"] == 100


def test_a_name_longer_than_a_recipe_line_holds_is_cut_and_named():
    out = _post({"lines": ["x" * 300, "2 eggs"]}).json()
    assert len(out["lines"][0]["name"]) == 200 and out["lines"][0]["text"] == "x" * 300
    assert out["warnings"][1] == "line 1's name was cut to 200 characters"
    assert out["warnings"][2] == f"line 1 ({'x' * 200}) states no amount"


def test_warnings_are_one_per_kind_naming_the_lines():
    out = _post({"yield_text": "Serves 2", "lines": ["salt", "2 eggs", "pepper"]}).json()
    assert out["warnings"] == ["lines 1 (salt), 3 (pepper) state no amount"]
    out = _post({"yield_text": "Serves 2", "lines": ["x" * 300, "y" * 300]}).json()
    assert out["warnings"][0] == "the names of lines 1, 2 were cut to 200 characters"


def test_an_amount_over_what_a_line_holds_is_not_stated_and_says_why():
    out = _post({"yield_text": "Serves 2", "lines": ["2000000 g flour", "500g penne"]}).json()
    assert [(ln["name"], ln["quantity"], ln["unit"]) for ln in out["lines"]] == \
        [("flour", None, ""), ("penne", 500.0, "g")]
    assert out["warnings"] == ["line 1 (flour) states more than the 1,000,000 a plan takes: "
                               "say how much you need"]
    # the bound itself is an amount a line holds
    assert _post({"lines": ["1000000 g flour"]}).json()["lines"][0]["quantity"] == 1_000_000


def test_the_worst_paste_still_fits_a_recipe_doc():
    """60 lines, each a problem of every kind a line can have, plus a blank and a catering
    yield: the warnings stay inside RecipeDoc's bound, so the doc a client builds from the
    whole output, warnings included, validates."""
    from pantry_planner.models import MAX_DOC_WARNINGS, RecipeDoc

    out = _post({"yield_text": "Makes 150 servings",
                 "lines": ["-"] + ["x" * 290 + f" {n}" for n in range(30)]
                 + ["9999999 " + "x" * 290 for _ in range(29)]}).json()
    assert len(out["lines"]) == 59
    assert 0 < len(out["warnings"]) <= MAX_DOC_WARNINGS
    RecipeDoc.model_validate({
        "key": "imp:1", "title": "Pasted", "servings": out["servings"],
        "servings_stated": out["servings_stated"], "lines": out["lines"],
        "warnings": out["warnings"], "source": {"kind": "pasted", "method": "paste"}})


def test_parse_lines_bounds():
    assert _post({"lines": ["2 eggs"] * 60}).status_code == 200
    assert _post({"lines": ["2 eggs"] * 61}).status_code == 422
    assert _post({"lines": ["x" * 300]}).status_code == 200
    assert _post({"lines": ["x" * 301]}).status_code == 422
    assert _post({"lines": ["2 eggs"], "title": "t" * 201}).status_code == 422
    assert _post({"lines": ["2 eggs"], "origin": "url"}).status_code == 422
    assert _post({"lines": []}).json()["lines"] == []

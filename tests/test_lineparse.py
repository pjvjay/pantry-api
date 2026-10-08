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

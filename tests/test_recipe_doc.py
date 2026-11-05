"""RecipeDoc: the one recipe schema, and recipe_doc.to_spec / to_recipe_text.

No database and no LLM: these are pure functions over the doc. Expected values
are the doc's own lines, written out in each test.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from pantry_planner.models import RecipeDoc
from pantry_planner.recipe_doc import UnconfirmedLines, to_recipe_text, to_spec


def _doc(lines, **kw) -> RecipeDoc:
    return RecipeDoc.model_validate({
        "key": "imp:1", "title": "Garlic Pasta",
        "source": {"kind": "pasted", "method": "paste"},
        "lines": [{"line_no": i, "amount_basis": "parsed_from_your_paste", **ln}
                  for i, ln in enumerate(lines, start=1)],
        **kw})


LINES = [
    {"text": "500g penne", "name": "penne", "quantity": 500, "unit": "g"},
    {"text": "2 cloves garlic, minced", "name": "garlic", "quantity": 2, "unit": "cloves",
     "note": "minced"},
    {"text": "1 can crushed tomatoes", "name": "crushed tomatoes", "quantity": 1,
     "unit": "can"},
    {"text": "olive oil", "name": "olive oil"},
]


def test_to_spec_plans_each_line_exactly_as_reviewed():
    spec = to_spec(_doc(LINES, servings=2, servings_stated=True, servings_basis="source"))
    assert spec.title == "Garlic Pasta"
    assert (spec.servings, spec.servings_stated) == (2, True)
    assert [(i.name, i.quantity, i.unit) for i in spec.ingredients] == [
        ("penne", 500.0, "g"), ("garlic", 2.0, "cloves"),
        ("crushed tomatoes", 1.0, "can"), ("olive oil", None, "")]
    # the note is the cook's (prep), shown to the selector, never part of the name
    assert [i.prep for i in spec.ingredients] == [None, "minced", None, None]
    # a can is a canned purchase, as the demo parser reads "1 can ..."
    assert [i.form for i in spec.ingredients] == [None, None, "canned", None]


def test_unstated_servings_plan_one_batch_and_say_so():
    spec = to_spec(_doc(LINES))
    assert (spec.servings, spec.servings_stated) == (1, False)
    # the shopper's own answer is a real number for planning
    spec = to_spec(_doc(LINES, servings=4, servings_basis="your_setting"))
    assert (spec.servings, spec.servings_stated) == (4, True)


def test_unconfirmed_lines_are_refused_by_number():
    lines = [dict(LINES[0]), {**LINES[1], "confirmed": False},
             {**LINES[2], "confirmed": False, "evidence": {"at": "1:05"}}]
    with pytest.raises(UnconfirmedLines) as e:
        to_spec(_doc(lines))
    assert e.value.line_nos == [2, 3]
    assert "2, 3" in str(e.value)


def test_lines_must_be_numbered_in_order():
    data = _doc(LINES).model_dump()
    data["lines"][1]["line_no"] = 5
    with pytest.raises(ValidationError, match="numbered 1..4 in order"):
        RecipeDoc.model_validate(data)


def test_doc_bounds():
    with pytest.raises(ValidationError):
        _doc([LINES[0]] * 61)
    assert len(_doc([LINES[0]] * 60).lines) == 60
    with pytest.raises(ValidationError):
        _doc([{**LINES[0], "evidence": {"at": "about a minute in"}}])
    with pytest.raises(ValidationError):
        _doc([{**LINES[0], "quantity": -1}])
    with pytest.raises(ValidationError):
        _doc([{**LINES[0], "name": ""}])
    with pytest.raises(ValidationError):
        _doc([{**LINES[0], "amount_basis": "a guess"}])


def test_a_sources_image_url_is_a_link_to_an_http_or_https_image():
    """The page's own preview image, kept as a link: an absolute http(s) URL of at most 2,000
    characters, or none."""
    def source(**kw) -> dict:
        return {"kind": "web", "method": "jsonld", "url": "https://blog.example/dal", **kw}

    assert _doc(LINES, source=source()).source.image_url is None
    for url in ("https://blog.example/img/dal.jpg", "http://cdn.example/a.webp?w=640",
                "HTTPS://i.ytimg.com/vi/abcdefghijk/hqdefault.jpg",
                "https://cdn.example/" + "a" * (2000 - 20)):
        assert _doc(LINES, source=source(image_url=url)).source.image_url == url
    for url in ("/img/dal.jpg", "//cdn.example/dal.jpg", "data:image/png;base64,iVBORw0KGgo=",
                "javascript:alert(1)", "file:///etc/passwd", "ftp://cdn.example/dal.jpg",
                "https://", "https://cdn.example/a b.jpg", "https://cdn.example/a\nb.jpg",
                "https://[::1/x.jpg", "", "https://cdn.example/" + "a" * (2001 - 20)):
        with pytest.raises(ValidationError):
            _doc(LINES, source=source(image_url=url))
    doc = _doc(LINES, source=source(image_url="https://blog.example/dal.jpg"))
    assert RecipeDoc.model_validate(doc.model_dump(mode="json")) == doc


def test_to_recipe_text_is_the_pasted_format():
    text = to_recipe_text(_doc(LINES, servings=2))
    assert text == ("Garlic Pasta (serves 2)\n"
                    "- 500 g penne\n"
                    "- 2 cloves garlic, minced\n"
                    "- 1 can crushed tomatoes\n"
                    "- olive oil\n")
    assert to_recipe_text(_doc(LINES[:1])).startswith("Garlic Pasta\n- 500 g penne")


def test_to_recipe_text_reads_back_the_same_through_the_demo_parser():
    from pantry_planner.demomode import parse_recipe

    p = parse_recipe(to_recipe_text(_doc(LINES, servings=2)))
    assert p.recipe.servings == 2
    assert [(i.name, i.quantity, i.unit) for i in p.recipe.ingredients] == [
        ("penne", 500.0, "g"), ("garlic", 2.0, "cloves"),
        ("crushed tomatoes", 1.0, "can"), ("olive oil", None, None)]

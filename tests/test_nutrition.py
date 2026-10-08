"""nutrition.py: per-portion nutrition from recipe amounts and reference foods. No LLM.

The math tests build a small Reference by hand, so each expected value is plain arithmetic on
numbers written here. The seed tests read the real reference data, and their expectations
come from seeds/*.json directly, never from the code under test.
"""
from __future__ import annotations

import json
import os

import pytest
from pydantic import ValidationError

from pantry_planner.db import SEEDS_DIR, NutrientFood, NutrientMapEntry, NutrientMeasure, Reference
from pantry_planner.models import (
    NutrientTotal,
    NutritionTarget,
    RecipeDoc,
    RecipeLine,
    RecipeSource,
)

RECIPES = json.loads((SEEDS_DIR / "recipes.json").read_text(encoding="utf-8"))
NUTRIENTS = json.loads((SEEDS_DIR / "nutrients.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module", autouse=True)
def seeded_db(tmp_path_factory):
    os.environ["DB_URL"] = (os.environ.get("PANTRY_TEST_DB_URL")
                            or f"sqlite:///{tmp_path_factory.mktemp('db') / 'nutrition.db'}")
    from pantry_planner import config, db

    config.settings.cache_clear()
    db.seed_from_json()
    yield
    config.settings.cache_clear()


# ─── A hand-made reference ───────────────────────────────────

def food(ref_id, **per):
    return NutrientFood(ref_id, "test", ref_id.split(":")[1], ref_id.split(":")[1],
                        f"Test food {ref_id}", "raw", per)


def ref(*, foods=(), mapping=None, measures=None, deployed=True) -> Reference:
    return Reference(sources={"test": {"source": "test", "name": "Test source"}},
                     foods={f.ref_id: f for f in foods},
                     measures=measures or {},
                     map={k: NutrientMapEntry(v, "generic" if v else "none", "")
                          for k, v in (mapping or {}).items()},
                     measures_deployed=deployed)


def line(n, name, quantity, unit, basis="parsed_from_your_paste", note="") -> RecipeLine:
    return RecipeLine(line_no=n, text=f"{quantity} {unit} {name}", name=name, quantity=quantity,
                      unit=unit, note=note, amount_basis=basis)


def doc(*lines, servings=2) -> RecipeDoc:
    return RecipeDoc(key="my:test", title="Test", servings=servings, lines=list(lines),
                     source=RecipeSource(kind="pasted", method="paste"))


RICE = food("test:1", energy_kcal=150.0, protein_g=20.0, fat_g=1.0, satfat_g=0.1,
            carbohydrate_g=10.0, fibre_g=1.0, sugars_g=0.5, sodium_mg=5.0)
OIL = food("test:2", energy_kcal=880.0, protein_g=0.0, fat_g=99.0, satfat_g=14.0,
           carbohydrate_g=0.0, fibre_g=0.0, sugars_g=0.0)          # sodium not published
MILK = food("test:3", energy_kcal=60.0, protein_g=3.0, fat_g=3.0, satfat_g=2.0,
            carbohydrate_g=5.0, fibre_g=0.0, sugars_g=5.0, sodium_mg=40.0)
EGG = food("test:4", energy_kcal=140.0, protein_g=12.0, fat_g=10.0, satfat_g=3.0,
           carbohydrate_g=1.0, fibre_g=0.0, sugars_g=0.4, sodium_mg=140.0)
BASIC = ref(foods=[RICE, OIL, MILK, EGG],
            mapping={"rice": "test:1", "olive oil": "test:2", "milk": "test:3", "egg": "test:4",
                     "garam masala": None},
            measures={"test:2": [NutrientMeasure("vol_15ml", 13.5, 15.0, "15ml", "")],
                      "test:3": [NutrientMeasure("vol_cup", 244.0, 236.6, "1 cup", "")],
                      "test:4": [NutrientMeasure("1_small_egg", 40.0, None, "1 small egg", ""),
                                 NutrientMeasure("1_large_egg", 50.0, None, "1 large egg", "")]})


def meal(d, reference=BASIC, **kw):
    from pantry_planner.nutrition import meal_nutrition

    return meal_nutrition(d, reference, floor=0.8, **kw)


# ─── Recipe math ─────────────────────────────────────────────

def test_meal_math_divides_by_servings():
    mn = meal(doc(line(1, "Rice", 200, "g"), servings=2))
    assert mn.basis == "per_serving" and mn.status == "complete"
    assert mn.totals["energy_kcal"].amount == 150.0       # 200 g x 150 / 100 / 2
    assert mn.totals["protein_g"].amount == 20.0
    assert mn.totals["energy_kcal"].status == "complete"


def test_portions_scale_linearly():
    one = meal(doc(line(1, "Rice", 200, "g")), portions=1)
    three = meal(doc(line(1, "Rice", 200, "g")), portions=3)
    assert three.totals["energy_kcal"].amount == pytest.approx(3 * one.totals["energy_kcal"].amount)
    assert three.portions == 3


def test_amount_eaten_not_packs():
    """300 g of rice serving 4 is 75 g a serving, whatever bag the planner buys: nutrition
    reads the recipe, and has no pack or pack size to read."""
    mn = meal(doc(line(1, "Rice", 300, "g"), servings=4))
    assert mn.lines[0].grams == 300
    assert mn.totals["energy_kcal"].amount == pytest.approx(75 * 150 / 100)


def test_unstated_servings_are_per_recipe_not_per_one():
    mn = meal(doc(line(1, "Rice", 300, "g"), servings=None), portions=2)
    assert mn.basis == "per_recipe" and mn.servings is None
    assert mn.totals["energy_kcal"].amount == 450.0      # the whole recipe; portions unused
    assert "does not say how many it serves" in mn.note


# ─── Unknowns ────────────────────────────────────────────────

def test_unmapped_line_is_lower_bound_never_zero():
    mn = meal(doc(line(1, "Rice", 200, "g"), line(2, "Saffron", 1, "g")))
    t = mn.totals["energy_kcal"]
    assert t.amount == 150.0 and t.status == "at_least" and not t.complete
    assert t.gaps == ["Saffron"] and (t.lines_counted, t.lines_total) == (1, 2)
    assert mn.status in {"incomplete", "below_floor"}
    assert [(m.ingredient, m.reason) for m in mn.missing] == [("Saffron", "no_reference")]


def test_nothing_counted_is_unknown():
    mn = meal(doc(line(1, "Saffron", 1, "g"), line(2, "Rice", None, "")))
    for t in mn.totals.values():
        assert t.amount is None and t.status == "unknown"
    assert mn.status == "below_floor"
    assert mn.coverage.count_fraction == 0.0


def test_absent_nutrient_is_per_nutrient():
    mn = meal(doc(line(1, "Olive Oil", 20, "g")))
    assert mn.totals["energy_kcal"].complete
    sodium = mn.totals["sodium_mg"]
    assert sodium.amount is None and sodium.status == "unknown" and not sodium.complete
    assert mn.lines[0].absent == ["sodium_mg"] and "sodium_mg" not in mn.lines[0].values
    assert mn.status == "incomplete"            # counted, but not every nutrient is known


def test_reviewed_none_vs_unreviewed():
    mn = meal(doc(line(1, "Garam Masala", 10, "g"), line(2, "Saffron", 1, "g")))
    reviewed, unreviewed = mn.lines
    assert reviewed.status == unreviewed.status == "no_reference"
    assert reviewed.match_kind == "none" and reviewed.reason.startswith("reviewed")
    assert unreviewed.match_kind is None
    assert "no reference food has been chosen" in unreviewed.reason


# ─── Units, keys, measures ───────────────────────────────────

def test_ml_each_and_count_words_need_a_source_measure():
    """No 1 g/ml is assumed, not even for milk: a volume or a count converts only through the
    source's own measure."""
    bare = ref(foods=[MILK, EGG], mapping={"milk": "test:3", "egg": "test:4"})
    mn = meal(doc(line(1, "Milk", 250, "ml"), line(2, "Eggs", 2, "each")), bare)
    assert [ln.status for ln in mn.lines] == ["no_conversion", "no_conversion"]
    assert "no density is assumed" in mn.lines[0].reason
    assert mn.lines[0].grams is None
    no_table = ref(foods=[MILK], mapping={"milk": "test:3"}, deployed=False)
    assert "pantry-db 0009" in meal(doc(line(1, "Milk", 250, "ml")), no_table).lines[0].reason
    can = meal(doc(line(1, "Rice", 1, "can")))
    assert can.lines[0].status == "no_conversion" and "can's size" in can.lines[0].reason
    for unit in ("", "clove", "pinch"):
        ln = meal(doc(line(1, "Rice", 2, unit))).lines[0]
        assert ln.status in {"no_conversion", "no_quantity"} and ln.grams is None, unit


def test_a_volume_uses_the_measures_own_volume():
    """A 236.6 ml cup measure is not the 250 ml metric cup: the density is grams over that
    row's own volume."""
    mn = meal(doc(line(1, "Milk", 250, "ml")), servings=1)
    ln = mn.lines[0]
    assert ln.status == "counted"
    assert ln.grams == pytest.approx(250 * 244.0 / 236.6, abs=0.001)
    assert "'1 cup' = 244 g" in ln.conversion
    assert ln.values["energy_kcal"] == pytest.approx(250 * 244.0 / 236.6 * 60 / 100, abs=0.01)
    oil = meal(doc(line(1, "Olive Oil", 2, "tbsp")), servings=1).lines[0]
    assert oil.grams == pytest.approx(30 * 13.5 / 15)


def test_counts_use_the_named_size_or_stay_unknown():
    unnamed = meal(doc(line(1, "Eggs", 3, "each"))).lines[0]
    assert unnamed.status == "no_conversion" and "several sizes" in unnamed.reason
    named = meal(doc(line(1, "Large Eggs", 3, "each"))).lines[0]
    assert named.status == "counted" and named.grams == 150.0
    in_note = meal(doc(line(1, "Eggs", 2, "each", note="small"))).lines[0]
    assert in_note.grams == 80.0


def test_water_with_no_reviewed_reference_is_excluded():
    mn = meal(doc(line(1, "Rice", 200, "g"), line(2, "Water", 500, "ml"),
                  line(3, "Ice cubes", 4, "each")))
    assert [ln.status for ln in mn.lines] == ["counted", "excluded", "excluded"]
    assert mn.coverage.lines_total == 1 and mn.status == "complete"


def test_water_counts_through_its_reviewed_reference(seeded_db):
    """The seed's map reviews water (CNF municipal water, with its own measures), so a water
    line is counted like any other: its published sodium is not dropped."""
    from pantry_planner import db

    reference = db.load_reference()
    mn = meal(doc(line(1, "Water", 250, "ml")), reference, portions=1)
    assert mn.lines[0].status == "counted" and mn.lines[0].ref_id == "cnf-api:2933"


def test_key_lookup_order():
    from pantry_planner.nutrition import candidate_keys, lookup

    beef = ref(foods=[RICE, OIL], mapping={"ground beef": "test:1", "beef": "test:2"})
    assert lookup("Ground Beef", beef)[0] == "ground beef"      # the full key wins
    assert lookup("Lean Ground Beef", beef)[0] == "beef"        # head noun, no full row
    assert candidate_keys("Light Soy Sauce") == ["light soy sauce", "soy sauce", "sauce"]
    # no descriptor dropped: generic_tokens is [] and the generic step is skipped
    assert candidate_keys("Basmati Rice") == ["basmati rice", "rice"]
    assert lookup("Saffron", beef) == ("saffron", None)


# ─── Coverage ────────────────────────────────────────────────

def test_coverage_count_and_mass(monkeypatch):
    from pantry_planner import config
    from pantry_planner.nutrition import meal_nutrition

    # 4 of 5 lines counted (0.8), but the uncounted line is most of the weight
    d = doc(line(1, "Rice", 10, "g"), line(2, "Rice", 10, "g"), line(3, "Rice", 10, "g"),
            line(4, "Rice", 10, "g"), line(5, "Saffron", 500, "g"))
    mn = meal_nutrition(d, BASIC, floor=0.8)
    assert mn.coverage.count_fraction == 0.8
    assert mn.coverage.mass_fraction == pytest.approx(40 / 540, abs=1e-4)
    assert not mn.coverage.meets_floor and mn.status == "below_floor"     # the AND rule
    assert "Saffron" in mn.note and "totals are minimums" in mn.coverage.note
    # mass is over the lines whose weight is known; an unweighable line counts by count only
    d2 = doc(*(line(i, "Rice", 10, "g") for i in range(1, 5)), line(5, "Saffron", 2, "each"))
    mn2 = meal_nutrition(d2, BASIC, floor=0.8)
    assert mn2.coverage.mass_fraction == 1.0 and mn2.coverage.lines_mass_unknown == 1
    assert mn2.coverage.meets_floor and mn2.status == "incomplete"
    # the floor comes from NUTRITION_MIN_COVERAGE
    monkeypatch.setenv("NUTRITION_MIN_COVERAGE", "0.9")
    config.settings.cache_clear()
    try:
        assert meal_nutrition(d2, BASIC).coverage.floor == 0.9
        assert meal_nutrition(d2, BASIC).status == "below_floor"
    finally:
        monkeypatch.delenv("NUTRITION_MIN_COVERAGE")
        config.settings.cache_clear()


# ─── Days and periods ────────────────────────────────────────

def _day_meal(d, slot="dinner", reference=BASIC, mid="m1"):
    from pantry_planner.nutrition import day_meal

    return day_meal(mid, "my:test", d.title, slot, meal(d, reference))


def test_day_totals_dinner_only_never_complete():
    from pantry_planner.nutrition import day_totals

    day = day_totals([_day_meal(doc(line(1, "Rice", 200, "g")))], meals_counted=["dinner"],
                     all_meals_planned=False, note="dinner only: other meals are not planned")
    assert day.totals["energy_kcal"].amount == 150.0
    assert not day.complete and day.totals["energy_kcal"].status == "at_least"
    assert day.note == "dinner only: other meals are not planned"


def test_day_totals_add_meals_and_a_per_recipe_meal_adds_nothing():
    from pantry_planner.nutrition import day_totals

    a = _day_meal(doc(line(1, "Rice", 200, "g")), "lunch", mid="a")
    b = _day_meal(doc(line(1, "Rice", 400, "g")), "dinner", mid="b")
    day = day_totals([a, b], meals_counted=["lunch", "dinner"], all_meals_planned=True)
    assert day.complete and day.totals["energy_kcal"].amount == 450.0
    unknown = _day_meal(doc(line(1, "Rice", 400, "g"), servings=None), "snack", mid="c")
    day = day_totals([a, b, unknown], meals_counted=["lunch", "dinner", "snack"],
                     all_meals_planned=True)
    assert not day.complete and day.totals["energy_kcal"].amount == 450.0
    assert "Test: servings unknown" in day.totals["energy_kcal"].gaps


def test_a_period_with_no_complete_day_has_no_average():
    from pantry_planner.nutrition import day_totals, period_nutrition

    d = day_totals([_day_meal(doc(line(1, "Rice", 200, "g")))], meals_counted=["dinner"],
                   all_meals_planned=False)
    p = period_nutrition([d, d], ["dinner"])
    assert p.days_complete == 0 and p.per_day_average_over_complete_days is None
    assert p.lower_bound_total["energy_kcal"].amount == 300.0
    assert not p.lower_bound_total["energy_kcal"].complete


def test_days_complete_counts_only_the_slots_that_are_on():
    from pantry_planner.nutrition import day_totals, period_nutrition

    dinner = _day_meal(doc(line(1, "Rice", 200, "g")))
    # with only dinner on, a day with a complete dinner is complete
    on = day_totals([dinner], meals_counted=["dinner"], all_meals_planned=True)
    off = day_totals([], meals_counted=["dinner"], all_meals_planned=False)
    p = period_nutrition([on, off], ["dinner"])
    assert (p.days_total, p.days_complete) == (2, 1)
    assert p.per_day_average_over_complete_days["energy_kcal"] == 150.0
    assert p.slots_counted == ["dinner"]


# ─── Demo amounts ────────────────────────────────────────────

def test_library_recipes_carry_demo_amounts():
    from pantry_planner import db
    from pantry_planner.nutrition import meal_nutrition
    from pantry_planner.recipe_doc import library_doc

    reference = db.load_reference()
    amounts = db.load_line_amounts([r["slug"] for r in RECIPES])
    for r in db.load_all_recipes():
        mn = meal_nutrition(library_doc(r, amounts[r.slug]), reference)
        assert mn.demo_amounts and mn.amounts_basis == ["demo_house_amounts"], r.slug
        assert "demo house amounts" in mn.note


def test_an_imported_doc_with_source_amounts_carries_no_demo_badge():
    mn = meal(doc(line(1, "Rice", 200, "g", basis="stated_by_source")))
    assert not mn.demo_amounts and mn.amounts_basis == ["stated_by_source"]
    mixed = meal(doc(line(1, "Rice", 200, "g", basis="stated_by_source"),
                     line(2, "Rice", 50, "g", basis="demo_house_amounts")))
    assert mixed.demo_amounts


# ─── The seed recipes ────────────────────────────────────────

def _library():
    from pantry_planner import db
    from pantry_planner.recipe_doc import library_doc

    amounts = db.load_line_amounts([r["slug"] for r in RECIPES])
    return [library_doc(r, amounts[r.slug]) for r in db.load_all_recipes()]


def test_library_per_serving_plausible():
    from pantry_planner import db
    from pantry_planner.nutrition import meal_nutrition

    reference = db.load_reference()
    for d in _library():
        kcal = meal_nutrition(d, reference).totals["energy_kcal"].amount
        assert 150 <= kcal <= 1500, (d.key, kcal)


def test_the_curry_names_its_unknown_line():
    from pantry_planner import db
    from pantry_planner.nutrition import meal_nutrition

    curry = next(d for d in _library() if d.key == "lib:chicken_curry")
    mn = meal_nutrition(curry, db.load_reference())
    assert [m.ingredient for m in mn.missing] == ["Garam Masala"]
    assert mn.totals["energy_kcal"].status == "at_least"


@pytest.mark.parametrize("slug", [r["slug"] for r in RECIPES])
def test_removing_any_mapping_never_raises_a_total_and_clears_complete(slug):
    """A parametrized loop instead of a property test (hypothesis is not a dependency)."""
    from pantry_planner import db
    from pantry_planner.nutrition import meal_nutrition, nutrition_key

    reference = db.load_reference()
    d = next(x for x in _library() if x.key == f"lib:{slug}")
    full = meal_nutrition(d, reference)
    for key in sorted({nutrition_key(ln.name) for ln in d.lines} & set(reference.map)):
        less = reference._replace(map={k: v for k, v in reference.map.items() if k != key})
        cut = meal_nutrition(d, less)
        for n, t in cut.totals.items():
            before = full.totals[n].amount
            assert t.amount is None or (before is not None and t.amount <= before + 1e-9), (key, n)
            assert not t.complete, (key, n)


# ─── Targets ─────────────────────────────────────────────────

def _t(amount, complete):
    return NutrientTotal(amount=amount, unit="kcal", complete=complete, lines_counted=1,
                         lines_total=1,
                         status="complete" if complete else "at_least" if amount else "unknown")


@pytest.mark.parametrize("amount,complete,lo,hi,want_min,want_max", [
    # a max: 'over' from a lower bound; 'within' only when complete
    (2100, False, None, 2000, None, "over"),
    (1900, False, None, 2000, None, "unknown"),
    (1900, True, None, 2000, None, "within"),
    (2100, True, None, 2000, None, "over"),
    # a min: 'met' from a lower bound; 'short' only when complete
    (60, False, 50, None, "met", None),
    (40, False, 50, None, "unknown", None),
    (40, True, 50, None, "short", None),
    (50, True, 50, None, "met", None),
    # nothing known
    (None, False, 50, 2000, "unknown", "unknown"),
])
def test_check_targets_truth_table(amount, complete, lo, hi, want_min, want_max):
    from pantry_planner.nutrition import check_targets

    got = check_targets({"energy_kcal": _t(amount, complete)},
                        {"energy_kcal": NutritionTarget(min=lo, max=hi)})["energy_kcal"]
    assert (got.min, got.max) == (want_min, want_max)


def test_a_dinner_only_day_never_says_short_or_within():
    from pantry_planner.nutrition import day_totals

    day = day_totals([_day_meal(doc(line(1, "Rice", 200, "g")))], meals_counted=["dinner"],
                     all_meals_planned=False,
                     targets={"energy_kcal": NutritionTarget(min=2000, max=2500),
                              "protein_g": NutritionTarget(min=10)})
    assert day.targets["energy_kcal"].min == "unknown"
    assert day.targets["energy_kcal"].max == "unknown"
    assert day.targets["protein_g"].min == "met"          # a lower bound can prove 'met'


def test_a_health_canada_dv_preset_for_energy_or_protein_is_rejected():
    from pydantic import TypeAdapter

    from pantry_planner.models import NutritionTargets

    adapter = TypeAdapter(NutritionTargets)
    for key in ("energy_kcal", "protein_g"):
        with pytest.raises(ValidationError, match="no Daily Value"):
            adapter.validate_python({key: {"max": 2000, "source": "health_canada_dv"}})
    ok = adapter.validate_python({"sodium_mg": {"max": 2300, "source": "health_canada_dv"},
                                  "energy_kcal": {"min": 1800}})
    assert set(ok) == {"sodium_mg", "energy_kcal"}
    with pytest.raises(ValidationError):
        adapter.validate_python({"vitamin_c_mg": {"min": 75}})
    with pytest.raises(ValidationError):
        NutritionTarget()
    with pytest.raises(ValidationError):
        NutritionTarget(min=10, max=5)

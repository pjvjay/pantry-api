"""POST /mealplan/alternatives: Options for a meal-plan trip line, on both engines (SQLite
here, Postgres with PANTRY_TEST_DB_URL).

The ranking is the chat cart's (alternatives.rank_alternatives); what these tests hold the
meal plan to is the rest: a row's figures are what a re-schedule with that product pinned
then charges, to the cent; the pins it leads to are checked like a cart swap (an origin
exclusion holds a product back, a product no store in range sells is refused); a pinned
line stays pinned through later edits; an approved trip whose products change needs review;
and the draft's new fields are bounded.

No LLM: drafts are demo starters resolved in DEMO_MODE, and the pipeline is patched to raise
where a test says so. Expected values come from /mealplan/schedule, the seed files and the
plan's own origin filter, never from the ranking under test.
"""
from __future__ import annotations

import copy
import datetime as dt
import json
import time

import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session

from tests.mealplan_fixtures import (
    PRODUCTS,
    START,
    STARTERS,
    client,
    done_db,
    index,
    lines_of,
    schedule,
    starter_draft,
    strategy,
    use_db,
)

THIGHS = 11          # Chicken Thighs Bone-In, the demo selector's pick for the biryani
BONELESS = 46        # Chicken Thighs Boneless 450g, another product for the same line
SALT = 122           # Table Salt 1kg: the pizza and the biryani buy it as one purchase
BIRYANI = "starter:chicken_biryani"
PIZZA = "starter:pepperoni_pizza"
FAR_STORE = 4        # MegaSave Richmond, about 14 km from the server's reference point


@pytest.fixture(scope="module", autouse=True)
def seeded_db(tmp_path_factory):
    use_db(tmp_path_factory, "options")
    yield
    done_db()


@pytest.fixture(scope="module")
def example():
    return starter_draft()


@pytest.fixture()
def reseed():
    """For tests that change offers or evidence: the seed is restored afterwards."""
    yield
    from pantry_planner import db
    from pantry_planner.db import ProductOriginEvidenceRow

    with Session(db.engine()) as s:
        s.query(ProductOriginEvidenceRow).delete()
        s.commit()
    db.seed_from_json()


def _sql(stmt: str, **params) -> None:
    from pantry_planner import db

    with Session(db.engine()) as s:
        s.execute(text(stmt), params)
        s.commit()


def _sql_rows(stmt: str, **params) -> list[dict]:
    from pantry_planner import db

    with Session(db.engine()) as s:
        return [dict(r) for r in s.execute(text(stmt), params).mappings()]


def options(draft: dict, date: str, product_id: int, **extra) -> dict:
    r = client().post("/mealplan/alternatives",
                      json={"draft": draft, "trip_date": date, "product_id": product_id,
                            **extra})
    assert r.status_code == 200, r.text
    return r.json()


def pinned(draft: dict, opts: dict, product_id: int | None) -> dict:
    """The draft with `product_id` chosen for every line the trip line covers, as the console
    pins it: a pin equal to the planner's pick (or None) is no pin."""
    out = copy.deepcopy(draft)
    pins = out.setdefault("pins", {})
    for c in opts["lines"]:
        lines = pins.setdefault(c["recipe_key"], {})
        if product_id is None or product_id == c["planner_product_id"]:
            lines.pop(str(c["line_no"]), None)
        else:
            lines[str(c["line_no"])] = product_id
        if not lines:
            del pins[c["recipe_key"]]
    return out


def _approve(sched: dict, name: str = "fresh", days: list[int] | None = None) -> list[dict]:
    out = []
    for t in strategy(sched, name)["trips"]:
        if days is not None and index(t["date"]) not in days:
            continue
        out.append({"date": t["date"], "fingerprint": t["fingerprint"], "strategy": name,
                    "snapshot": [{"product_id": ln["product"]["id"], "packs": ln["packs"],
                                  "storage": ln["storage"],
                                  "price_at_approval": ln["price"]} for ln in t["lines"]]})
    return out


def _cents(x: float | None) -> int | None:
    return None if x is None else round(x * 100)


# ─── What a row says is what the re-schedule charges ─────────

def _check_rows(draft: dict, name: str, trip: dict, product_id: int) -> int:
    """Every row of the line's options against the plan re-scheduled with it chosen."""
    from pantry_planner import limits

    limits.reset()      # one re-schedule per row: faster than any shopper clicks
    opts = options(draft, trip["date"], product_id, strategy=name, limit=25)
    meals = {m["meal_id"] for ln in trip["lines"] if ln["product"]["id"] == product_id
             for m in ln["for_meals"]}
    plan_total = strategy(schedule(draft), name)["total_cost"]
    assert _cents(opts["plan_total"]) == _cents(plan_total)
    checked = 0
    for item in opts["ranking"]["items"]:
        after = schedule(pinned(draft, opts, item["product_id"]))
        got = [ln for t in strategy(after, name)["trips"] for ln in t["lines"]
               if ln["product"]["id"] == item["product_id"]
               and meals & {m["meal_id"] for m in ln["for_meals"]}]
        on_day = [ln for t, ln in lines_of(after, name, item["product_id"])
                  if t["date"] == trip["date"] and ln in got]
        got = on_day or got
        assert got, item["product"]
        if item["trip"] is None:
            # no trip total: the re-schedule cannot price the line either
            assert item["cost_for_need"] is None
            assert any(ln["price"] is None for ln in got), item["product"]
            continue
        assert _cents(item["trip"]["total"]) == _cents(strategy(after, name)["total_cost"])
        assert _cents(item["trip"]["delta"]) == _cents(
            strategy(after, name)["total_cost"] - plan_total)
        assert _cents(item["cost_for_need"]) == _cents(sum(ln["price"] for ln in got))
        assert item["packs"] == sum(ln["packs"] for ln in got)
        assert item["trip"]["buys_at"]["store"] == got[0]["store"]
        assert _cents(item["trip"]["buys_at"]["price"] * got[0]["packs"]) == _cents(
            got[0]["price"])
        if item["current"]:
            assert item["trip"]["delta"] == 0
        checked += 1
    return checked


def test_every_rows_figures_are_what_a_reschedule_then_charges_to_the_cent(example):
    sched = schedule(example)
    checked = 0
    for name in ("fresh", "fewest_trips"):
        trip = strategy(sched, name)["trips"][0]
        products = sorted({ln["product"]["id"] for ln in trip["lines"]})
        for pid in products[:: 1 if name == "fresh" else 4]:
            checked += _check_rows(example, name, trip, pid)
    assert checked >= 60


def test_a_purchase_two_recipes_share_is_ranked_and_pinned_for_both(example):
    sched = schedule(example)
    trip, line = next((t, ln) for t, ln in lines_of(sched, "fresh", SALT))
    assert {m["recipe_key"] for m in line["for_meals"]} == {BIRYANI, PIZZA}
    opts = options(example, trip["date"], SALT)
    assert {c["recipe_key"] for c in opts["lines"]} == {BIRYANI, PIZZA}
    assert opts["ranking"]["ingredient"] == "Salt"
    other = next(it for it in opts["ranking"]["items"] if not it["current"])
    after = schedule(pinned(example, opts, other["product_id"]))
    assert not lines_of(after, "fresh", SALT)
    ((_t, now),) = [(t, ln) for t, ln in lines_of(after, "fresh", other["product_id"])
                    if t["date"] == trip["date"]]
    assert {m["recipe_key"] for m in now["for_meals"]} == {BIRYANI, PIZZA}
    _check_rows(example, "fresh", trip, SALT)


# ─── Pins are checked like a cart swap ───────────────────────

def test_a_pin_the_origin_exclusion_holds_back_is_refused(example, reseed):
    from pantry_planner import db, origins

    db.save_origin_evidence([dict(
        product_id=BONELESS, source="label-photo", source_ref=f"{BONELESS}.jpg",
        claim_type="made-in", verbatim="Made in United States", ingredient_origin="",
        manufactured_in="United States", confidence="high", importer_only=False, note="",
        observed_at="")])
    draft = {**example, "settings": {"exclude_origin": ["United States"]}}
    trip, _line = lines_of(schedule(draft), "fresh", THIGHS)[0]
    opts = options(draft, trip["date"], THIGHS)
    held = {h["product_id"] for h in opts["ranking"]["held_back"]}
    ranked = {it["product_id"] for it in opts["ranking"]["items"]}
    assert BONELESS in held and BONELESS not in ranked
    # what is held back is what the plan's own origin filter drops
    catalog = {p.id: p for p in db.load_all_products()}
    _kept, dropped = origins.filter_pool([catalog[i] for i in held | ranked],
                                         exclude=["United States"],
                                         origins=origins.resolve_all())
    assert {p.id for p, _c, _f in dropped} == held

    r = client().post("/mealplan/schedule", json={**draft, "pins": {BIRYANI: {"1": BONELESS}}})
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert (detail["error"], detail["recipe_key"], detail["line_no"]) == (
        "pin_invalid", BIRYANI, 1)
    assert "evidenced as United States" in detail["detail"]
    # without the exclusion the same pin is the shopper's choice
    assert lines_of(schedule({**example, "pins": {BIRYANI: {"1": BONELESS}}}), "fresh",
                    BONELESS)


def test_a_pin_no_store_in_range_sells_is_refused(example, reseed):
    draft = {**example, "settings": {"max_km": 5}}
    _sql("DELETE FROM store_products WHERE product_id = :p AND store_id <> :far",
         p=BONELESS, far=FAR_STORE)
    trip, _line = lines_of(schedule(draft), "fresh", THIGHS)[0]
    opts = options(draft, trip["date"], THIGHS)
    assert BONELESS not in {it["product_id"] for it in opts["ranking"]["items"]}
    assert opts["ranking"]["unavailable"] >= 1
    r = client().post("/mealplan/schedule", json={**draft, "pins": {BIRYANI: {"1": BONELESS}}})
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert (detail["error"], detail["line_no"]) == ("pin_invalid", 1)
    assert "no store within 5 km sells it" in detail["detail"]
    assert "the nearest offer is MegaSave Richmond" in detail["detail"]


def test_a_pinned_substitute_no_store_sells_any_more_says_so(example, reseed):
    """A same-aisle substitute is found among offers, so once no store sells it the pin is
    refused for that reason, not because its words do not match the line."""
    sched = schedule(example)
    trip, _line = lines_of(sched, "fresh", THIGHS)[0]
    opts = options(example, trip["date"], THIGHS, limit=25)
    sub = next(it for it in opts["ranking"]["items"] if it["match"] == "substitute")
    draft = pinned(example, opts, sub["product_id"])
    assert schedule(draft)
    _sql("DELETE FROM store_products WHERE product_id = :p", p=sub["product_id"])
    r = client().post("/mealplan/schedule", json=draft)
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert (detail["error"], detail["line_no"]) == ("pin_invalid", 1)
    assert detail["detail"].endswith("cannot be bought for line 1 (Chicken Thighs): no store "
                                     "sells it"), detail["detail"]


def test_a_pin_that_is_no_option_for_the_line_is_refused(example):
    mango = next(pid for pid, p in PRODUCTS.items() if p["name"].startswith("Frozen Mango"))
    r = client().post("/mealplan/schedule", json={**example, "pins": {BIRYANI: {"1": mango}}})
    assert r.status_code == 422
    assert r.json()["detail"]["error"] == "pin_invalid"
    assert "is not an option for line 1" in r.json()["detail"]["detail"]


# ─── A pin lasts ─────────────────────────────────────────────

def test_a_pinned_line_survives_a_reschedule(example):
    sched = schedule(example)
    trip, _line = lines_of(sched, "fresh", THIGHS)[0]
    opts = options(example, trip["date"], THIGHS)
    assert opts["pinned"] is False
    assert [(c["recipe_key"], c["line_no"], c["planner_product_id"]) for c in opts["lines"]] \
        == [(BIRYANI, 1, THIGHS)]
    draft = pinned(example, opts, BONELESS)
    assert draft["pins"] == {BIRYANI: {"1": BONELESS}}

    after = schedule(draft)
    assert not lines_of(after, "fresh", THIGHS) and not lines_of(after, "fewest_trips", THIGHS)
    biryani = [m for m in after["meals"] if m["recipe_key"] == BIRYANI and m["date"]]
    bought = {m["meal_id"] for _t, ln in lines_of(after, "fresh", BONELESS)
              for m in ln["for_meals"]}
    assert {m["id"] for m in biryani} <= bought

    # Every meal dated and the last biryani moved two days later: still the shopper's product.
    meals = [{"id": m["id"], "recipe_key": m["recipe_key"], "date": m["date"],
              "slot": m["slot"]} for m in after["meals"]]
    last = max((m for m in meals if m["recipe_key"] == BIRYANI), key=lambda m: m["date"])
    free = next(i for i in range(index(last["date"]) + 1, 14)
                if not any(index(m["date"]) == i and m["slot"] == last["slot"] for m in meals
                           if m["date"]))
    last["date"] = (dt.date.fromisoformat(example["start_date"])
                    + dt.timedelta(days=free)).isoformat()
    moved = schedule({**draft, "meals": meals})
    assert not lines_of(moved, "fresh", THIGHS)
    assert last["id"] in {m["meal_id"] for _t, ln in lines_of(moved, "fresh", BONELESS)
                          for m in ln["for_meals"]}

    # Its options name the pin, and the planner's pick is one choice away.
    t2, _l2 = lines_of(moved, "fresh", BONELESS)[0]
    again = options({**draft, "meals": meals}, t2["date"], BONELESS)
    assert again["pinned"] is True
    assert again["lines"][0]["pinned_product_id"] == BONELESS
    current = next(it for it in again["ranking"]["items"] if it["current"])
    assert current["product_id"] == BONELESS
    back = pinned({**draft, "meals": meals}, again, None)
    assert back["pins"] == {}
    assert lines_of(schedule(back), "fresh", THIGHS)


def test_an_approved_trip_whose_product_changes_needs_review(example):
    sched = schedule(example)
    trip, _line = lines_of(sched, "fresh", THIGHS)[0]
    approved = _approve(sched, days=[index(trip["date"])])
    draft = {**example, "trips": approved}
    before = next(t for t in strategy(schedule(draft), "fresh")["trips"]
                  if t["date"] == trip["date"])
    assert before["status"] == "approved"
    opts = options(draft, trip["date"], THIGHS)
    after = schedule(pinned(draft, opts, BONELESS))
    t = next(t for t in strategy(after, "fresh")["trips"] if t["date"] == trip["date"])
    assert t["status"] == "needs_review" and t["fingerprint"] != approved[0]["fingerprint"]
    assert {d["product_id"] for d in t["diff"]["added"]} == {BONELESS}
    assert {d["product_id"] for d in t["diff"]["removed"]} == {THIGHS}
    review = [w for w in after["warnings"] if w["code"] == "needs_review"
              and w["trip_date"] == trip["date"] and w["strategy"] == "fresh"]
    assert review and review[0]["remedies"][0]["op"] == "approve_trip"
    # undoing the choice puts the approved list back
    back = schedule(pinned(draft, opts, None))
    assert next(t for t in strategy(back, "fresh")["trips"]
                if t["date"] == trip["date"])["status"] == "approved"


def test_a_line_no_longer_stocked_still_has_options(example, reseed):
    sched = schedule(example)
    trip, _line = lines_of(sched, "fresh", THIGHS)[0]
    draft = {**example, "trips": _approve(sched, days=[index(trip["date"])])}
    _sql("DELETE FROM store_products WHERE product_id = :p", p=THIGHS)
    after = schedule(draft)
    gone = next(w for w in after["warnings"] if w["code"] == "no_longer_stocked")
    op = gone["remedies"][0]
    assert (op["op"], op["product_id"], op["date"]) == ("open_options", THIGHS, trip["date"])
    opts = options(draft, op["date"], op["product_id"])
    assert opts["stocked"] is False
    ids = [it["product_id"] for it in opts["ranking"]["items"]]
    assert THIGHS not in ids and BONELESS in ids
    assert not any(it["current"] for it in opts["ranking"]["items"])
    fixed = schedule(pinned(draft, opts, BONELESS))
    assert not [w for w in fixed["warnings"] if w["code"] == "no_longer_stocked"]
    t = next(t for t in strategy(fixed, "fresh")["trips"] if t["date"] == trip["date"])
    assert t["status"] == "needs_review" and not t["not_stocked"]


# ─── The basis is the one resolve planned with ───────────────

def test_the_meal_basis_is_the_plan_basis_resolve_used(example):
    from pantry_planner import db, flow
    from pantry_planner.mealplan.models import MealPlanDraft, RecipeRef
    from pantry_planner.mealplan.pins import meal_basis
    from pantry_planner.mealplan.resolve import recipe_doc, resolve_one
    from pantry_planner.recipe_doc import to_recipe_text, to_spec

    slug = db.load_all_recipes()[0].slug
    lib = f"lib:{slug}"
    ref = RecipeRef(key=lib, slug=slug)
    draft = MealPlanDraft.model_validate({
        **example,
        "recipes": {**example["recipes"], lib: {"ref": ref.model_dump(mode="json"),
                                                "wanted": 1}},
        "resolved": {**example["resolved"], lib: resolve_one(ref).model_dump(mode="json")}})
    keys = sorted(draft.recipes)
    assert len(keys) == 5
    for key in keys:
        r = draft.recipes[key].ref
        doc = recipe_doc(r)
        plan = (flow.run(r.slug, allow_partial=True) if r.slug else
                flow.run_spec(to_spec(doc), allow_partial=True,
                              display_text=to_recipe_text(doc)))
        mine = meal_basis(draft, key, doc)
        fields = ("line_no", "name", "form", "prep", "quantity", "unit", "product_id")
        want = [tuple(getattr(ln, f) for f in fields) for ln in plan.basis.lines
                if ln.product_id is not None]
        got = [tuple(getattr(ln, f) for f in fields) for ln in mine.lines
               if ln.product_id is not None]
        assert got == want, key
        assert (mine.path, mine.constraints) == (plan.basis.path, plan.basis.constraints)
        assert (mine.exclude_origin, mine.preference) == ([], [])


# ─── With no trip date (the wizard's Ingredients stage) ─────

def _no_trip(draft: dict, product_id: int, **extra) -> dict:
    r = client().post("/mealplan/alternatives",
                      json={"draft": draft, "product_id": product_id, **extra})
    assert r.status_code == 200, r.text
    return r.json()


def test_without_a_trip_date_every_line_buying_the_product_is_ranked_with_no_trip(example):
    """The purchase is every line of the plan that buys the product, for all its recipe's
    meals (the basket's lines): Salt for the pizza's three meals and the biryani's three. No
    row has a trip; its packs are what those meals need and its cost those packs at its
    cheapest offer in range."""
    import math

    from tests.mealplan_fixtures import EXAMPLE, STARTERS

    opts = _no_trip(example, SALT)
    assert (opts["trip_date"], opts["plan_total"], opts["product"]) == (
        None, None, PRODUCTS[SALT]["name"])
    assert {(c["recipe_key"], c["ingredient"]) for c in opts["lines"]} == {
        (BIRYANI, "Salt"), (PIZZA, "Salt")}
    basket = client().post("/mealplan/basket", json={"draft": example}).json()
    assert {(r["key"], ln["line_no"]) for r in basket["recipes"] for ln in r["lines"]
            if ln["product"]["id"] == SALT} == {(c["recipe_key"], c["line_no"])
                                                 for c in opts["lines"]}
    grams = sum(ln["quantity"] * EXAMPLE[key] * 2 / STARTERS[key]["servings"]
                for key in ("pepperoni_pizza", "chicken_biryani")
                for ln in STARTERS[key]["lines"] if ln["name"] == "Salt")
    assert opts["ranking"]["need"] == f"{grams:g} g"
    items = opts["ranking"]["items"]
    assert items and all(it["trip"] is None for it in items)
    for it in items:
        p = PRODUCTS[it["product_id"]]
        if p["unit_uom"] != "g":
            continue
        packs = math.ceil(grams / p["unit_qty"])
        (cheapest,) = _sql_rows("SELECT MIN(price) AS p FROM store_products "
                                "WHERE product_id = :p", p=it["product_id"])
        assert it["packs"] == packs
        assert _cents(it["cost_for_need"]) == _cents(cheapest["p"] * packs)
        trip_reason = next(r for r in it["reasons"] if r["code"] == "trip")
        assert trip_reason["text"].startswith("No trip is planned yet")
    # choosing one pins every covered line, and the basket then buys it for both
    other = next(it for it in items if not it["current"])
    after = client().post("/mealplan/basket", json={
        "draft": pinned(example, opts, other["product_id"])}).json()
    assert {ln["product"]["id"] for r in after["recipes"] for ln in r["lines"]
            if ln["ingredient"] == "Salt"} == {other["product_id"]}
    # the same draft gives the same answer
    assert _no_trip(example, SALT)["ranking"] == opts["ranking"]


def test_without_a_trip_date_a_product_nothing_buys_is_a_409(example):
    bought = {ln["product_id"] for r in example["resolved"].values() for ln in r["lines"]}
    absent = next(pid for pid in sorted(PRODUCTS) if pid not in bought)
    r = client().post("/mealplan/alternatives", json={"draft": example, "product_id": absent})
    assert r.status_code == 409 and r.json()["detail"]["error"] == "no_trip_line"


def _made_in_the_us(pid: int) -> None:
    from pantry_planner import db

    db.save_origin_evidence([dict(
        product_id=pid, source="label-photo", source_ref=f"{pid}.jpg", claim_type="made-in",
        verbatim="Made in United States", ingredient_origin="", manufactured_in="United States",
        confidence="high", importer_only=False, note="", observed_at="")])


def test_a_pin_the_origin_rule_now_holds_back_is_replaced_from_its_options(example, reseed):
    """Pinned, then evidenced as made in the United States, which the plan excludes: the
    schedule refuses the pin and says why, naming the product and the origin rule. The
    Ingredients stage's Options for it (no trip date) read the draft as the basket does,
    keeping the pin on its line: the rows are what can take its place, none current, and the
    pin is the first product held back, with its evidence, as the basket names it first on
    its line. A row chosen replaces the pin and the plan is scheduled again. A trip's Options
    refuse the pin as the schedule does."""
    rule = {"exclude_origin": ["United States"]}
    draft = {**example, "settings": rule, "pins": {BIRYANI: {"1": BONELESS}}}
    trip, _line = lines_of(schedule(draft), "fresh", BONELESS)[0]
    _made_in_the_us(BONELESS)

    refused = {
        "error": "pin_invalid", "recipe_key": BIRYANI, "line_no": 1, "product_id": BONELESS,
        "unavailable_kind": "origin_excluded",
        "detail": "Chicken Biryani: Chicken Thighs Boneless 450g, your product for Chicken "
                  "Thighs, is held back by the origin rule: it is evidenced as United States "
                  "(made or packed there), which this plan excludes"}
    r = client().post("/mealplan/schedule", json=draft)
    assert (r.status_code, r.json()["detail"]) == (422, refused)

    opts = _no_trip(draft, BONELESS)
    assert (opts["product_id"], opts["pinned"], opts["trip_date"], opts["plan_total"]) == (
        BONELESS, True, None, None)
    assert [(c["recipe_key"], c["line_no"], c["planner_product_id"], c["pinned_product_id"])
            for c in opts["lines"]] == [(BIRYANI, 1, THIGHS, BONELESS)]
    ranking = opts["ranking"]
    first = ranking["held_back"][0]
    assert (first["product_id"], first["country"], first["field"], first["verbatim"]) == (
        BONELESS, "United States", "manufactured_in", "Made in United States")
    rows = [it["product_id"] for it in ranking["items"]]
    assert THIGHS in rows and BONELESS not in rows
    assert not any(it["current"] for it in ranking["items"])
    basket = client().post("/mealplan/basket", json={"draft": draft}).json()
    (line,) = [ln for r in basket["recipes"] if r["key"] == BIRYANI for ln in r["lines"]
               if ln["line_no"] == 1]
    assert (line["unavailable_kind"], line["held_back"][0]["product_id"]) == (
        "origin_excluded", BONELESS)
    # every other line's Options answer too, with the pin kept on its line
    assert any(it["current"] for it in _no_trip(draft, SALT)["ranking"]["items"])

    other = next(pid for pid in rows if pid != THIGHS)
    for chosen in (THIGHS, other):           # the planner's own pick is no pin
        after = pinned(draft, opts, chosen)
        assert after["pins"] == ({} if chosen == THIGHS else {BIRYANI: {"1": chosen}})
        sched = schedule(after)
        assert lines_of(sched, "fresh", chosen) and not lines_of(sched, "fresh", BONELESS)

    r = client().post("/mealplan/alternatives", json={
        "draft": draft, "trip_date": trip["date"], "product_id": BONELESS})
    assert (r.status_code, r.json()["detail"]) == (422, refused)


def test_a_product_planned_before_its_evidence_is_no_row_of_its_own_options(example, reseed):
    """The resolve's own pick, evidenced as made in the United States after it was planned:
    no trip buys it, so a trip's Options for it are refused as for any product the trip does
    not buy, and its Options with no trip (the Ingredients stage, where its line waits for
    the shopper's answer) have no current row and name it first among the products held
    back, also once no store sells it any more."""
    (trip, _line), *_ = lines_of(schedule(example), "fresh", THIGHS)
    _made_in_the_us(THIGHS)
    draft = {**example, "settings": {"exclude_origin": ["United States"]}}
    ruled = schedule(draft)
    assert lines_of(ruled, "fresh", THIGHS) == lines_of(ruled, "fewest_trips", THIGHS) == []
    r = client().post("/mealplan/alternatives",
                      json={"draft": draft, "trip_date": trip["date"], "product_id": THIGHS})
    assert (r.status_code, r.json()["detail"]["error"]) == (409, "no_trip_line")
    assert r.json()["detail"]["detail"].endswith("or the origin rule now holds it back")
    for gone in (False, True):
        if gone:
            _sql("DELETE FROM store_products WHERE product_id = :p", p=THIGHS)
        opts = _no_trip(draft, THIGHS)
        ranking = opts["ranking"]
        assert opts["stocked"] is not gone
        assert ranking["held_back"][0]["product_id"] == THIGHS
        assert THIGHS not in {it["product_id"] for it in ranking["items"]}
        assert ranking["items"] and not any(it["current"] for it in ranking["items"])


# ─── A pin on a line the resolve bought nothing for ──────────

YELLOW_ONIONS = (12, 162, 163, 164)    # Yellow Onion, and the 3 lb, 10 lb and organic bags
# What works instead of a yellow onion, as its seed lists them (red onion, shallots, green
# onions): the line's Options when every yellow onion is held back.
RED_ONION, SHALLOTS, GREEN_ONIONS = 147, 148, 142
PILAF = "starter:tomato_pilaf"
ONION_LINES = {f"starter:{k}": next(ln["line_no"] for ln in s["lines"]
                                    if ln["name"] == "Yellow Onion")
               for k, s in STARTERS.items() if k in ("tomato_pilaf", "chicken_biryani")}


@pytest.fixture()
def onions():
    """Each test's own origin evidence and offers; the seed is restored, and what the resolve
    and the catalog cached forgotten, afterwards."""
    yield
    from pantry_planner import db
    from pantry_planner.db import ProductOriginEvidenceRow
    from pantry_planner.mealplan import catalog, resolve

    with Session(db.engine()) as s:
        s.query(ProductOriginEvidenceRow).delete()
        s.commit()
    db.seed_from_json()
    resolve.clear_cache()
    catalog.clear_cache()


def _grown_in_the_us(*pids: int) -> None:
    from pantry_planner import db

    db.save_origin_evidence([dict(
        product_id=pid, source="label-photo", source_ref=f"{pid}.jpg", claim_type="grown-in",
        verbatim="Grown in USA", ingredient_origin="United States", manufactured_in="",
        confidence="high", importer_only=False, note="", observed_at="") for pid in pids])


def _no_yellow_onion(**settings) -> dict:
    """The pilaf and the biryani, two meals each, resolved under the United States exclusion
    (demo mode, no LLM): with every yellow onion evidenced as grown there, neither Yellow
    Onion line buys anything."""
    from pantry_planner.mealplan import catalog, resolve
    from pantry_planner.mealplan.models import RecipeRef
    from pantry_planner.mealplan.resolve import resolve_one

    resolve.clear_cache()
    catalog.clear_cache()
    recipes, resolved = {}, {}
    for key, slot in (("tomato_pilaf", "lunch"), ("chicken_biryani", "dinner")):
        ref = RecipeRef(key=f"starter:{key}", starter=key)
        recipes[ref.key] = {"ref": ref.model_dump(mode="json"), "wanted": 2, "slot": slot}
        resolved[ref.key] = resolve_one(ref, exclude=["United States"]).model_dump(mode="json")
    return {"start_date": START.isoformat(), "days": 7, "recipes": recipes,
            "resolved": resolved, "settings": {"exclude_origin": ["United States"], **settings}}


def _onion_lines(basket: dict) -> dict[str, dict]:
    return {r["key"]: ln for r in basket["recipes"] for ln in r["lines"]
            if ln["ingredient"] == "Yellow Onion"}


def test_a_pin_may_fill_a_line_the_resolve_bought_nothing_for(onions):
    """A product chosen for a line the resolve left out is checked as any pin is, and the
    line then buys it everywhere: the basket, the schedule's trips and the catalog."""
    _grown_in_the_us(*YELLOW_ONIONS)
    draft = _no_yellow_onion()
    chosen = {**draft, "pins": {k: {str(n): RED_ONION} for k, n in ONION_LINES.items()}}
    after = _onion_lines(client().post("/mealplan/basket", json={"draft": chosen}).json())
    assert {k: (ln["product"]["id"], ln["available"], ln["pinned"], ln["unavailable_kind"])
            for k, ln in after.items()} == {k: (RED_ONION, True, True, None)
                                            for k in ONION_LINES}
    assert {m["recipe_key"] for t in strategy(schedule(chosen), "fresh")["trips"]
            for ln in t["lines"] if ln["product"]["id"] == RED_ONION
            for m in ln["for_meals"]} == set(ONION_LINES)
    cat = client().post("/mealplan/catalog", json={
        "settings": draft["settings"], "pins": chosen["pins"]}).json()
    pilaf = next(r for r in cat["recipes"] if r["key"] == PILAF)
    (onion,) = [ln for ln in pilaf["lines"] if ln["ingredient"] == "Yellow Onion"]
    assert onion["product_id"] == RED_ONION and onion["pack_price"] is not None
    assert "Yellow Onion" not in pilaf["unavailable"]
    # what the US rule holds back is refused, as for a line the resolve did buy for
    r = client().post("/mealplan/schedule", json={**draft, "pins": {PILAF: {
        str(ONION_LINES[PILAF]): 12}}})
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert (detail["error"], detail["recipe_key"], detail["line_no"]) == (
        "pin_invalid", PILAF, ONION_LINES[PILAF])
    assert "evidenced as United States" in detail["detail"]
    # and so is a product no store in range sells
    _sql("DELETE FROM store_products WHERE product_id = :p AND store_id <> :far",
         p=GREEN_ONIONS, far=FAR_STORE)
    far = {**draft, "settings": {**draft["settings"], "max_km": 5},
           "pins": {PILAF: {str(ONION_LINES[PILAF]): GREEN_ONIONS}}}
    r = client().post("/mealplan/basket", json={"draft": far})
    assert r.status_code == 422 and "no store within 5 km sells it" in r.json()["detail"][
        "detail"]


def _line_options(draft: dict, key: str = PILAF, **extra) -> dict:
    r = client().post("/mealplan/alternatives", json={
        "draft": draft, "line": {"recipe_key": key, "line_no": ONION_LINES[key]}, **extra})
    assert r.status_code == 200, r.text
    return r.json()


def test_a_line_that_buys_nothing_has_options_and_a_row_chosen_makes_it_available(onions):
    """The Yellow Onion lines buy nothing (the US rule holds every yellow onion back). Their
    Options rank what could fill both at once, with no current pick; choosing a row pins both
    lines, and the basket then buys it for them."""
    _grown_in_the_us(*YELLOW_ONIONS)
    draft = _no_yellow_onion()
    before = _onion_lines(client().post("/mealplan/basket", json={"draft": draft}).json())
    assert {k: (ln["product"], ln["unavailable_kind"]) for k, ln in before.items()} == {
        k: (None, "origin_excluded") for k in ONION_LINES}

    opts = _line_options(draft)
    assert (opts["product_id"], opts["product"], opts["stocked"], opts["pinned"]) == (
        None, "", False, False)
    assert (opts["trip_date"], opts["plan_total"]) == (None, None)
    assert [(c["recipe_key"], c["line_no"], c["ingredient"], c["planner_product_id"],
             c["pinned_product_id"]) for c in opts["lines"]] == [
        (k, ONION_LINES[k], "Yellow Onion", None, None) for k in sorted(ONION_LINES)]
    ranking = opts["ranking"]
    assert ranking["lines"] == [1, 2] and ranking["line_no"] == 2      # the pilaf's, asked for
    grams = sum(ln["quantity"] * 2 * 2 / STARTERS[k]["servings"]
                for k in ("tomato_pilaf", "chicken_biryani")
                for ln in STARTERS[k]["lines"] if ln["name"] == "Yellow Onion")
    assert ranking["need"] == f"{grams:g} g"
    items = ranking["items"]
    assert [it["product_id"] for it in items] == [RED_ONION, SHALLOTS, GREEN_ONIONS]
    assert not any(it["current"] for it in items) and {it["match"] for it in items} == {
        "substitute"}
    assert {it["group"] for it in items} == {"works_instead"} and ranking["matching"] == "meaning"
    held = {h["product_id"]: h for h in ranking["held_back"]}
    assert held[12]["country"] == "United States" and held[12]["verbatim"] == "Grown in USA"
    assert not set(held) & {it["product_id"] for it in items}
    assert ranking["out_of_reach"] == []

    chosen = pinned(draft, opts, RED_ONION)
    assert chosen["pins"] == {k: {str(n): RED_ONION} for k, n in ONION_LINES.items()}
    after = _onion_lines(client().post("/mealplan/basket", json={"draft": chosen}).json())
    assert {k: (ln["product"]["id"], ln["available"], ln["pinned"])
            for k, ln in after.items()} == {k: (RED_ONION, True, True) for k in ONION_LINES}
    # with a trip date, each row's trip is the plan re-scheduled with it bought for them
    trip = strategy(schedule(draft), "fresh")["trips"][0]
    on_trip = _line_options(draft, trip_date=trip["date"])
    row = next(it for it in on_trip["ranking"]["items"] if it["product_id"] == RED_ONION)
    assert on_trip["plan_total"] == strategy(schedule(draft), "fresh")["total_cost"]
    assert _cents(row["trip"]["total"]) == _cents(strategy(schedule(chosen), "fresh")[
        "total_cost"])


def test_with_nothing_nearby_but_what_the_us_rule_holds_back_no_row_is_choosable(onions):
    """Every onion nearby (and the shallots that work instead of one) is evidenced as grown in
    the United States: no row to choose, and held_back lists them with their evidence."""
    _grown_in_the_us(*YELLOW_ONIONS, RED_ONION, SHALLOTS, GREEN_ONIONS)
    draft = _no_yellow_onion()
    for extra in ({}, {"trip_date": strategy(schedule(draft), "fresh")["trips"][0]["date"]}):
        ranking = _line_options(draft, **extra)["ranking"]
        assert (ranking["items"], ranking["total"], ranking["out_of_reach"]) == ([], 0, [])
        held = {h["product_id"]: h for h in ranking["held_back"]}
        assert {12, RED_ONION, SHALLOTS, GREEN_ONIONS} <= set(held)
        for h in held.values():
            assert (h["country"], h["verbatim"], h["source"]) == (
                "United States", "Grown in USA", "label-photo"), h


def test_rows_sold_only_at_stores_turned_off_or_beyond_the_distance_say_so(onions):
    """Red Onion is sold within 5 km only at a store the shopper turned off, Green Onions and
    Shallots only at the store 14 km away: none is choosable, and out_of_reach says why."""
    _grown_in_the_us(*YELLOW_ONIONS)
    _sql("DELETE FROM store_products WHERE product_id = :p AND store_id <> 3", p=RED_ONION)
    for far_only in (GREEN_ONIONS, SHALLOTS):
        _sql("DELETE FROM store_products WHERE product_id = :p AND store_id <> :far",
             p=far_only, far=FAR_STORE)
    draft = _no_yellow_onion(stores_off=[3], max_km=5)
    ranking = _line_options(draft)["ranking"]
    assert ranking["items"] == [] and ranking["unavailable"] == 3
    (store3,) = _sql_rows("SELECT name FROM stores WHERE id = 3")
    (far,) = _sql_rows("SELECT price FROM store_products WHERE product_id = :p",
                       p=GREEN_ONIONS)
    reach = {r["product_id"]: r for r in ranking["out_of_reach"]}
    assert reach[RED_ONION] == {"product_id": RED_ONION, "product": "Red Onion",
                                "why": "stores_off", "stores": [store3["name"]],
                                "nearest": None}
    green = reach[GREEN_ONIONS]
    assert (green["why"], green["stores"]) == ("out_of_range", [])
    assert (green["nearest"]["store"], green["nearest"]["price"]) == (
        "MegaSave Richmond", far["price"])
    assert green["nearest"]["distance_km"] > 5
    assert reach[SHALLOTS]["why"] == "out_of_range"
    # with the store back on, Red Onion is a row again
    back = _line_options({**draft, "settings": {**draft["settings"], "stores_off": []}})
    assert [it["product_id"] for it in back["ranking"]["items"]] == [RED_ONION]


def test_a_line_is_asked_for_in_place_of_a_product_and_must_buy_nothing(onions):
    _grown_in_the_us(*YELLOW_ONIONS)
    draft = _no_yellow_onion()
    line = {"recipe_key": PILAF, "line_no": ONION_LINES[PILAF]}

    def post(**body) -> dict:
        r = client().post("/mealplan/alternatives", json={"draft": draft, **body})
        return {"status": r.status_code, **r.json()}

    assert post(line=line, product_id=RED_ONION)["status"] == 422     # both
    assert post()["status"] == 422                                     # neither
    assert post(line={"recipe_key": PILAF, "line_no": 0})["status"] == 422
    rice = post(line={"recipe_key": PILAF, "line_no": 1})
    assert rice["status"] == 409 and rice["detail"]["error"] == "no_trip_line"
    assert "ask for its options by product_id" in rice["detail"]["detail"]
    nope = post(line={"recipe_key": "starter:nope", "line_no": 1})
    assert nope["status"] == 422 and nope["detail"]["error"] == "unknown_recipe_key"
    absent = post(line={"recipe_key": PILAF, "line_no": 40})
    assert absent["status"] == 409 and "has no line 40" in absent["detail"]["detail"]


# ─── The route ───────────────────────────────────────────────

def test_options_call_no_llm_and_answer_the_same_bytes(example, monkeypatch):
    from pantry_planner import demomode, flow, selector
    from pantry_planner.nlsearch import planner, query_parser

    def boom(*_a, **_k):
        raise AssertionError("the options called the pipeline")

    for mod, name in ((selector, "call_selector"), (flow, "call_selector"),
                      (query_parser, "parse_input"), (planner, "parse_input"),
                      (demomode, "parse_recipe"), (demomode, "select_products")):
        monkeypatch.setattr(mod, name, boom)
    trip, _line = lines_of(schedule(example), "fresh", THIGHS)[0]
    body = {"draft": example, "trip_date": trip["date"], "product_id": THIGHS}
    a = client().post("/mealplan/alternatives", json=body)
    b = client().post("/mealplan/alternatives", json=body)
    assert a.status_code == 200 and a.content == b.content
    got = a.json()
    assert got["rev"] == example.get("rev", 0) and got["strategy"] == "fresh"
    assert got["ranking"]["data_note"]
    assert "plan's trips" in json.dumps(got["ranking"]["items"])


def test_a_line_the_trip_does_not_buy_is_a_409(example):
    sched = schedule(example)
    trip = strategy(sched, "fresh")["trips"][0]
    absent = next(pid for pid in sorted(PRODUCTS)
                  if pid not in {ln["product"]["id"] for ln in trip["lines"]})
    r = client().post("/mealplan/alternatives",
                      json={"draft": example, "trip_date": trip["date"], "product_id": absent})
    assert r.status_code == 409 and r.json()["detail"]["error"] == "no_trip_line"


def test_options_refuse_what_the_schedule_refuses(example):
    trip, _line = lines_of(schedule(example), "fresh", THIGHS)[0]
    r = client().post("/mealplan/alternatives", json={
        "draft": {**example, "pins": {BIRYANI: {"99": BONELESS}}},
        "trip_date": trip["date"], "product_id": THIGHS})
    assert r.status_code == 422 and r.json()["detail"]["error"] == "pin_invalid"
    for bad in ({"limit": 0}, {"limit": 26}, {"product_id": 0}, {"strategy": "cheapest"}):
        r = client().post("/mealplan/alternatives", json={
            "draft": example, "trip_date": trip["date"], "product_id": THIGHS, **bad})
        assert r.status_code == 422, bad
    big = {"draft": {**example, "padding": "x" * 300_000}, "trip_date": trip["date"],
           "product_id": THIGHS}
    assert client().post("/mealplan/alternatives", json=big).status_code == 413


def test_options_for_a_line_answer_well_under_two_seconds(example):
    trip = strategy(schedule(example), "fresh")["trips"][0]
    worst = 0.0
    for ln in trip["lines"][:8]:
        t0 = time.perf_counter()
        options(example, trip["date"], ln["product"]["id"])
        worst = max(worst, time.perf_counter() - t0)
    assert worst < 2.0


def test_options_have_their_own_sixty_a_minute_bucket(example, monkeypatch):
    from pantry_planner import limits

    assert limits.LIMITS["/mealplan/alternatives"] == limits.Limit(per_minute=60, burst=60)
    monkeypatch.setitem(limits.LIMITS, "/mealplan/alternatives",
                        limits.Limit(per_minute=60, burst=1))
    trip, _line = lines_of(schedule(example), "fresh", THIGHS)[0]
    body = {"draft": example, "trip_date": trip["date"], "product_id": THIGHS}
    assert client().post("/mealplan/alternatives", json=body).status_code == 200
    r = client().post("/mealplan/alternatives", json=body)
    assert r.status_code == 429 and r.headers["Retry-After"]
    assert client().post("/mealplan/schedule", json=example).status_code == 200


# ─── Bounds on the draft's pins and origin rules ─────────────

def test_pins_and_origin_rules_in_the_draft_are_bounded(example):
    def status(**fields) -> int:
        return client().post("/mealplan/schedule", json={**example, **fields}).status_code

    many = {str(n): BONELESS for n in range(1, 42)}
    assert status(pins={BIRYANI: many}) == 422                     # 41 lines of one recipe
    assert status(pins={BIRYANI: {"12345": BONELESS}}) == 422       # a 5-digit line number
    assert status(pins={BIRYANI: {"": BONELESS}}) == 422
    assert status(pins={BIRYANI: {"1": 0}}) == 422
    assert status(pins={BIRYANI: {"1": 2**31}}) == 422              # past the database's int
    assert status(pins={"k" * 101: {"1": BONELESS}}) == 422
    assert status(pins={f"r{i}": {} for i in range(13)}) == 422     # more recipes than a plan
    assert status(settings={"exclude_origin": ["Canada"] * 51}) == 422
    assert status(settings={"preference": ["x" * 101]}) == 422
    assert status(settings={"exclude_origin": ["Atlantis"]}) == 422
    assert status(settings={"exclude_origin": [""]}) == 422
    # what the bounds allow is planned
    assert status(pins={BIRYANI: {"1": BONELESS}},
                  settings={"exclude_origin": ["Canada"] * 50, "preference": ["Mexico"]}) == 200

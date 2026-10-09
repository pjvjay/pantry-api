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

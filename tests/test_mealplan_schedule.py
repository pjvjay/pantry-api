"""POST /mealplan/schedule on both engines (SQLite here, Postgres with PANTRY_TEST_DB_URL):
pure, byte-identical, facts re-read from the database, approval and its review states,
prices and stock that change after approval, pins, needs_servings and the trip list text.

No LLM: the parser and the selector are patched to raise where a test says so. Expected
prices are read from the database directly, never from the code under test.
"""
from __future__ import annotations

import json

import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session

from tests.mealplan_fixtures import (
    PRODUCTS,
    client,
    day,
    doc_draft,
    doc_recipe,
    done_db,
    index,
    lines_of,
    schedule,
    starter_draft,
    strategy,
    use_db,
)

THIGHS = 11          # Chicken Thighs Bone-In, the demo selector's pick for the biryani


@pytest.fixture(scope="module", autouse=True)
def seeded_db(tmp_path_factory):
    use_db(tmp_path_factory, "schedule")
    yield
    done_db()


@pytest.fixture(scope="module")
def example():
    return starter_draft()


@pytest.fixture()
def reseed():
    """For tests that change prices or stock: the seed is restored afterwards."""
    yield
    from pantry_planner import db

    db.seed_from_json()


def _sql(stmt: str, **params) -> list:
    from pantry_planner import db

    with Session(db.engine()) as s:
        rows = s.execute(text(stmt), params)
        out = list(rows.mappings()) if rows.returns_rows else []
        s.commit()
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


def _trip(sched: dict, name: str, i: int) -> dict:
    return next(t for t in strategy(sched, name)["trips"] if index(t["date"]) == i)


# ─── Pure and deterministic ──────────────────────────────────

def test_no_llm_is_called(example, monkeypatch):
    from pantry_planner import demomode, flow, selector
    from pantry_planner.nlsearch import planner, query_parser

    def boom(*_a, **_k):
        raise AssertionError("the schedule called the pipeline")

    for mod, name in ((selector, "call_selector"), (flow, "call_selector"),
                      (query_parser, "parse_input"), (planner, "parse_input"),
                      (demomode, "parse_recipe"), (demomode, "select_products")):
        monkeypatch.setattr(mod, name, boom)
    assert strategy(schedule(example), "fresh")["trips"]


def test_the_same_draft_gives_byte_identical_output(example):
    a = client().post("/mealplan/schedule", json=example)
    b = client().post("/mealplan/schedule", json=example)
    assert a.status_code == 200 and a.content == b.content


def test_names_and_prices_are_read_from_the_database_not_the_draft(example, reseed):
    tampered = json.loads(json.dumps(example))
    for r in tampered["resolved"].values():
        for ln in r["lines"]:
            ln["product_name"] = "Gold Bars"
    sched = schedule(tampered)
    names = {ln["product"]["name"] for t in strategy(sched, "fresh")["trips"]
             for ln in t["lines"]}
    assert "Gold Bars" not in names and PRODUCTS[THIGHS]["name"] in names
    ((trip, line),) = lines_of(sched, "fresh", 166)[:1]
    store_price = _sql("SELECT sp.price AS price FROM store_products sp JOIN stores s ON "
                       "s.id = sp.store_id WHERE s.name = :n AND sp.product_id = 166",
                       n=line["store"])[0]["price"]
    assert line["price"] == round(store_price * line["packs"], 2)
    _sql("UPDATE store_products SET price = price + 1 WHERE product_id = 166")
    ((_t, again),) = lines_of(schedule(example), "fresh", 166)[:1]
    assert again["price"] == round((store_price + 1) * line["packs"], 2)


def test_a_draft_over_256_kb_is_refused(example):
    r = client().post("/mealplan/schedule", json={**example, "padding": "x" * 300_000})
    assert r.status_code == 413


def _post_raw(path: str, body: str):
    return client().post(path, content=body, headers={"content-type": "application/json"})


def test_a_line_quantity_must_be_a_finite_number_up_to_a_million():
    """JSON's Infinity, NaN and 1e309 read as non-finite floats in Python. One in a doc line
    used to fail the needs with a 500; a huge finite one came back as computed packs and a
    price. Both routes that compute a schedule now refuse the draft, naming the line."""
    entry = doc_recipe("my:s", "Salmon Rice", 2, [("Atlantic Salmon", 400, "g", 47),
                                                  ("Basmati Rice", 200, "g", 8)])
    body = json.dumps(doc_draft({"my:s": entry},
                                meals=[{"id": "m", "recipe_key": "my:s",
                                        "date": day(2).isoformat()}]))
    # the doc's line comes first; the resolved copy after it is not what gets planned
    assert body.index('"quantity": 400') < body.index('"resolved"')
    for bad in ("Infinity", "-Infinity", "NaN", "1e309", "1.7e308", "1000001"):
        for path in ("/mealplan/schedule", "/mealplan/suggest-cook-days"):
            r = _post_raw(path, body.replace('"quantity": 400', f'"quantity": {bad}', 1))
            assert r.status_code == 422, (bad, path, r.text)
            (err,) = r.json()["detail"]
            assert err["loc"][-4:] == ["doc", "lines", 0, "quantity"]
    # the bound itself is an amount a line holds
    ok = _post_raw("/mealplan/schedule",
                   body.replace('"quantity": 400', '"quantity": 1000000', 1))
    assert ok.status_code == 200


def test_pack_counts_and_approved_prices_in_the_draft_are_bounded(example):
    """A pack override or an approved trip's snapshot comes back from the browser. An integer
    past 1e308 used to fail the price sum with a 500, and an infinite or huge price at
    approval came back as a delta."""
    sched = schedule(example)
    trip = strategy(sched, "fresh")["trips"][0]
    first = trip["lines"][0]
    key = f'{trip["date"]}:{first["product"]["id"]}'
    for packs in (10**400, 10**11):
        r = _post_raw("/mealplan/schedule",
                      json.dumps({**example, "packs_override": {key: packs}}))
        assert r.status_code == 422, packs
        snap = _approve(sched, days=[index(trip["date"])])
        snap[0]["snapshot"][0]["packs"] = packs
        r = _post_raw("/mealplan/schedule", json.dumps({**example, "trips": snap}))
        assert r.status_code == 422, packs
    snap = _approve(sched, days=[index(trip["date"])])
    snap[0]["snapshot"][0]["price_at_approval"] = 12345.0
    body = json.dumps({**example, "trips": snap})
    assert body.count("12345.0") == 1
    for bad in ("Infinity", "NaN", "1e308", "-1"):
        r = _post_raw("/mealplan/schedule", body.replace("12345.0", bad))
        assert r.status_code == 422, bad
    # an override the shopper can type is still planned as theirs
    r = schedule({**example, "packs_override": {key: 5}})
    ((_t, line),) = [(t, ln) for t, ln in lines_of(r, "fresh", first["product"]["id"])
                     if t["date"] == trip["date"]]
    assert (line["packs"], line["packs_basis"]) == (5, "your_setting")


def test_a_product_the_catalog_no_longer_has_is_stale(example):
    stale = json.loads(json.dumps(example))
    stale["resolved"]["starter:mango_milkshake"]["lines"][0]["product_id"] = 99999
    r = client().post("/mealplan/schedule", json=stale)
    assert (r.status_code, r.json()["detail"]["error"]) == (422, "stale_product")


def test_pins_choose_the_product_and_are_validated(example):
    pinned = schedule({**example, "pins": {"starter:chicken_biryani": {"1": 46}}})
    assert lines_of(pinned, "fresh", 46) and not lines_of(pinned, "fresh", THIGHS)
    for bad in ({"1": 99999}, {"99": 46}, {"x": 46}):
        r = client().post("/mealplan/schedule",
                          json={**example, "pins": {"starter:chicken_biryani": bad}})
        assert (r.status_code, r.json()["detail"]["error"]) == (422, "pin_invalid")


# ─── Approval ────────────────────────────────────────────────

def test_an_approved_trip_stays_approved_until_its_lines_change(example):
    sched = schedule(example)
    approved = _approve(sched)
    again = schedule({**example, "trips": approved})
    assert {t["status"] for t in strategy(again, "fresh")["trips"]} == {"approved"}
    assert [t["fingerprint"] for t in strategy(again, "fresh")["trips"]] == [
        a["fingerprint"] for a in approved]
    assert again["approved_schedule"]["exportable"] is True

    # Move the last biryani from day 11 to day 13: the day-10 trip no longer buys its
    # chicken, so it needs review with the diff; trips 1 and 4 are untouched.
    # The console stores the spread back, so every meal is dated; then one moves.
    meals = [{"id": m["id"], "recipe_key": m["recipe_key"], "date": m["date"],
              "slot": m["slot"]} for m in sched["meals"]]
    moving = next(m for m in meals if m["id"] == "starter:chicken_biryani#3")
    assert index(moving["date"]) == 11
    moving.update(date=day(13).isoformat(), pinned=True)
    moved = schedule({**example, "trips": approved, "meals": meals})
    t10 = _trip(moved, "fresh", 10)
    assert t10["status"] == "needs_review"
    assert any(r["product_id"] == THIGHS for r in t10["diff"]["removed"])
    assert any(s.startswith("-") and PRODUCTS[THIGHS]["name"] in s for s in t10["diff"]["text"])
    for i in (1, 4):
        assert _trip(moved, "fresh", i)["status"] == "approved"
        assert _trip(moved, "fresh", i)["fingerprint"] == _trip(sched, "fresh", i)["fingerprint"]
    review = [w for w in moved["warnings"] if w["code"] == "needs_review"]
    assert review and review[0]["remedies"][0]["op"] == "approve_trip"
    assert moved["approved_schedule"]["exportable"] is False


def test_two_approvals_of_one_date_are_refused(example):
    """The console keeps one approval a day; a draft with two (under either strategy) would
    put two lists for one shop in the approved schedule."""
    sched = schedule(example)
    (fresh,) = _approve(sched, "fresh", days=[1])
    for other in ("fewest_trips", "fresh"):
        twice = {**example, "trips": [fresh, {**fresh, "strategy": other}]}
        for path in ("/mealplan/schedule", "/mealplan/suggest-cook-days"):
            r = client().post(path, json=twice)
            assert r.status_code == 422, (other, path)
            detail = r.json()["detail"]
            assert detail["error"] == "invalid_dates"
            assert detail["detail"] == (f"approved trip {fresh['date']} is approved twice "
                                        f"(fresh and {other}); a day has one approved trip")
    assert schedule({**example, "trips": [fresh]})["approved_schedule"]["exportable"] is True


def test_the_fingerprint_is_sha256_of_date_and_lines_without_price(example):
    import hashlib

    trip = strategy(schedule(example), "fresh")["trips"][0]
    body = ";".join(f"{ln['product']['id']}:{ln['packs']}:{ln['storage']}"
                    for ln in sorted(trip["lines"],
                                     key=lambda ln: (ln["product"]["id"], ln["storage"])))
    assert trip["fingerprint"] == hashlib.sha256(f"{trip['date']}|{body}".encode()).hexdigest()


def test_a_price_change_after_approval_is_a_delta_and_the_status_stays(example, reseed):
    sched = schedule(example)
    approved = _approve(sched, days=[1])
    ((trip, before),) = lines_of(sched, "fresh", THIGHS)[:1]
    assert index(trip["date"]) == 1
    _sql("UPDATE store_products SET price = price + 1.2 WHERE product_id = :p", p=THIGHS)
    after = schedule({**example, "trips": approved})
    t1 = _trip(after, "fresh", 1)
    assert t1["status"] == "approved"
    line = next(ln for ln in t1["lines"] if ln["product"]["id"] == THIGHS)
    assert line["price_at_approval"] == before["price"]
    assert line["price_delta"] == pytest.approx(1.2 * line["packs"])
    assert t1["price_delta"] == pytest.approx(1.2 * line["packs"])
    notes = [w for w in after["warnings"] if w["code"] == "price_changed"]
    assert notes and "(demo prices)" in notes[0]["message"] and notes[0]["level"] == "note"


def test_a_pack_count_changed_after_approval_is_a_diff_not_a_price_change(example):
    """The snapshot keeps each line's price for all its packs. Comparing line totals read
    two more packs at the same prices as "Price changed since you approved: +$X"."""
    sched = schedule(example)
    approved = _approve(sched, days=[1])
    ((trip, before),) = lines_of(sched, "fresh", THIGHS)[:1]
    assert index(trip["date"]) == 1 and before["packs"] >= 1
    key = f'{trip["date"]}:{THIGHS}'
    after = schedule({**example, "trips": approved,
                      "packs_override": {key: before["packs"] + 2}})
    t1 = _trip(after, "fresh", 1)
    line = next(ln for ln in t1["lines"] if ln["product"]["id"] == THIGHS)
    assert line["packs"] == before["packs"] + 2
    assert line["price"] != before["price"]                 # more packs cost more ...
    assert line["price_at_approval"] == before["price"]
    assert line["price_delta"] == 0 and t1["price_delta"] == 0   # ... at the same prices
    assert not [w for w in after["warnings"] if w["code"] == "price_changed"]
    # The fingerprint holds the packs: the trip needs review, with the change in its diff.
    assert t1["status"] == "needs_review"
    assert t1["diff"]["changed"] == [{
        "product_id": THIGHS, "name": PRODUCTS[THIGHS]["name"], "packs_before": before["packs"],
        "packs_after": before["packs"] + 2, "storage": before["storage"]}]
    assert any(w["code"] == "needs_review" and w["trip_date"] == trip["date"]
               for w in after["warnings"])


def test_a_unit_price_change_with_changed_packs_is_counted_on_the_packs_bought_now(
        example, reseed):
    sched = schedule(example)
    approved = _approve(sched, days=[1])
    ((trip, before),) = lines_of(sched, "fresh", THIGHS)[:1]
    packs_now = before["packs"] + 3
    _sql("UPDATE store_products SET price = price + 1.2 WHERE product_id = :p", p=THIGHS)
    after = schedule({**example, "trips": approved,
                      "packs_override": {f'{trip["date"]}:{THIGHS}': packs_now}})
    t1 = _trip(after, "fresh", 1)
    line = next(ln for ln in t1["lines"] if ln["product"]["id"] == THIGHS)
    # (unit now - unit then) x packs now, the unit then being the approved line price over
    # the approved packs; read from the database, not from the engine
    store_price = _sql("SELECT sp.price AS price FROM store_products sp JOIN stores s ON "
                       "s.id = sp.store_id WHERE s.name = :n AND sp.product_id = :p",
                       n=line["store"], p=THIGHS)[0]["price"]
    unit_then = before["price"] / before["packs"]
    assert line["price"] == round(store_price * packs_now, 2)
    assert line["price_delta"] == pytest.approx(round((store_price - unit_then) * packs_now, 2))
    assert line["price_delta"] == pytest.approx(1.2 * packs_now)
    others = [ln["price_delta"] for ln in t1["lines"] if ln["product"]["id"] != THIGHS]
    assert t1["price_delta"] == pytest.approx(line["price_delta"] + sum(others))
    assert all(d == 0 for d in others)
    (note,) = [w for w in after["warnings"] if w["code"] == "price_changed"]
    assert f"+${1.2 * packs_now:.2f} (demo prices)" in note["message"]


def test_an_offer_removed_after_approval_is_no_longer_stocked(example, reseed):
    sched = schedule(example)
    approved = _approve(sched, days=[1])
    _sql("DELETE FROM store_products WHERE product_id = :p", p=THIGHS)
    after = schedule({**example, "trips": approved})
    t1 = _trip(after, "fresh", 1)
    assert t1["status"] == "needs_review"
    line = next(ln for ln in t1["lines"] if ln["product"]["id"] == THIGHS)
    assert (line["stocked"], line["store"], line["price"]) == (False, None, None)
    assert t1["total_is_floor"] is True and PRODUCTS[THIGHS]["name"] in t1["not_stocked"]
    gone = [w for w in after["warnings"] if w["code"] == "no_longer_stocked"]
    assert gone and gone[0]["level"] == "must_fix" and gone[0]["product_id"] == THIGHS
    assert [r["op"] for r in gone[0]["remedies"]] == ["open_options", "set_packs"]
    assert gone[0]["remedies"][1]["packs"] == 0


# ─── Unknown servings ────────────────────────────────────────

def test_needs_servings_leaves_no_number_in_the_recipes_trip_lines():
    entry = doc_recipe("my:s", "Salmon Rice", None, [("Atlantic Salmon", 400, "g", 47),
                                                     ("Basmati Rice", 200, "g", 8)])
    meals = [{"id": "m", "recipe_key": "my:s", "date": day(2).isoformat()}]
    draft = doc_draft({"my:s": entry}, meals=meals)
    sched = schedule(draft)
    for pid in (47, 8):
        ((trip, line),) = lines_of(sched, "fresh", pid)
        assert line["packs_basis"] == "needs_servings"
        assert [line[k] for k in ("packs", "need_qty", "leftover_qty", "price",
                                  "price_delta")] == [None] * 5
        assert trip["total_is_floor"] is True
        assert "does not say how many it serves" in trip["list_text"]
    # no line on the trip has a price, so neither the trip nor the plan has a total: an
    # unknown is never $0.00, in the JSON or in the list the shopper copies
    for name in ("fresh", "fewest_trips"):
        st = strategy(sched, name)
        assert (st["total_cost"], st["total_is_floor"]) == (None, True)
        for t in st["trips"]:
            assert t["total_cost"] is None
            assert t["list_text"].splitlines()[-2] == "Total unknown (no line has a price yet)"
            assert "$0.00" not in t["list_text"]
    must = [w for w in sched["warnings"] if w["code"] == "needs_servings"]
    assert must and must[0]["level"] == "must_fix"
    assert must[0]["remedies"] == [{"op": "set_servings", "recipe_key": "my:s"}]

    draft["recipes"]["my:s"]["servings"] = 2          # the shopper's answer
    answered = schedule(draft)
    ((_t, salmon),) = lines_of(answered, "fresh", 47)
    assert (salmon["packs"], salmon["need_qty"]) == (1, 400.0) and salmon["price"] is not None
    assert "needs_servings" not in [w["code"] for w in answered["warnings"]]
    ((trip, rice),) = lines_of(answered, "fresh", 8)
    assert trip["total_cost"] == round(salmon["price"] + rice["price"], 2)
    assert trip["list_text"].splitlines()[-2] == f"Total ${trip['total_cost']:.2f}"
    assert strategy(answered, "fresh")["total_cost"] == trip["total_cost"]


def test_a_trip_with_a_priced_line_keeps_a_floor_and_an_empty_plan_costs_nothing():
    """One priced line and one unknown: the total is the priced line's, marked "at least".
    A plan with no meals has no trips, and its 0.00 is a fact, not an unknown."""
    known = doc_recipe("my:k", "Rice Bowl", 2, [("Basmati Rice", 200, "g", 8)])
    unknown = doc_recipe("my:u", "Salmon", None, [("Atlantic Salmon", 400, "g", 47)])
    meals = [{"id": "a", "recipe_key": "my:k", "date": day(2).isoformat()},
             {"id": "b", "recipe_key": "my:u", "date": day(2).isoformat(), "slot": "lunch"}]
    sched = schedule(doc_draft({"my:k": known, "my:u": unknown}, meals=meals))
    ((trip, rice),) = lines_of(sched, "fresh", 8)
    ((_t, salmon),) = lines_of(sched, "fresh", 47)
    assert rice["price"] is not None and salmon["price"] is None
    assert (trip["total_cost"], trip["total_is_floor"]) == (rice["price"], True)
    assert trip["list_text"].splitlines()[-2] == f"Total at least ${rice['price']:.2f}"
    empty = schedule(doc_draft({"my:k": known}))
    st = strategy(empty, "fresh")
    assert (st["trips"], st["total_cost"], st["total_is_floor"]) == ([], 0.0, False)


# ─── Facts on every line ─────────────────────────────────────

def test_every_cited_line_carries_its_source_and_unknowns_carry_none(example):
    for name in ("fresh", "fewest_trips"):
        for t in strategy(schedule(example), name)["trips"]:
            for ln in t["lines"]:
                info = ln["shelf_life"]
                if info["status"] == "cited":
                    assert info["url"].startswith("https://") and info["page_date"]
                    assert info["verbatim"] and info["rule_ids"]
                else:
                    assert (info["verbatim"], info["rule_ids"], info["url"]) == ([], [], None)
                    assert (info["days_planned"] is None) == (info["status"] == "unknown")


def test_list_text_is_grouped_by_store_then_aisle_with_the_demo_footer(example):
    for t in strategy(schedule(example), "fresh")["trips"]:
        text_lines = t["list_text"].splitlines()
        assert text_lines[-1] == "Prices and stock are demo data."
        stores = [s for s in text_lines if s and not s.startswith(" ")][1:-2]
        assert stores == t["stores"] + (["Not stocked within range"] if t["not_stocked"] else [])
        # Under each store, aisles in order, and every line under its own store and aisle.
        current_store, aisles = None, []
        for s in text_lines[1:-2]:
            if s and not s.startswith(" "):
                assert aisles == sorted(aisles)
                current_store, aisles = s, []
            elif s.startswith(" ") and not s.startswith("  "):
                aisles.append(s.strip())
            elif s.startswith("  - "):
                match = [ln for ln in t["lines"] if f" {ln['product']['name']}" in s]
                assert match and match[0]["store"] == current_store
                assert (match[0]["category"] or "other").capitalize() == aisles[-1]
        for ln in t["lines"]:
            if ln["product"]["demo_product"]:
                assert f"{ln['product']['name']} (demo product)" in t["list_text"]
    assert "Stores, prices and stock are synthetic" in schedule(example)["synthetic_notice"]


def test_each_trips_recommended_option_is_the_trip_optimizers(example):
    from pantry_planner import tripopt
    from pantry_planner.config import settings

    sched = schedule(example)
    t = strategy(sched, "fresh")["trips"][0]
    basket = sorted({ln["product"]["id"]: ln["packs"] for ln in t["lines"]}.items())
    rows = _sql("SELECT s.id AS store_id, s.name AS store_name, s.lat AS lat, s.lon AS lon, "
                "0.0 AS dist_km2, sp.product_id AS product_id, sp.price AS price "
                "FROM store_products sp JOIN stores s ON s.id = sp.store_id")
    rows = [dict(r) for r in rows if r["product_id"] in dict(basket)]
    cfg = settings()
    options = tripopt.optimize_trips(
        rows, [(pid, PRODUCTS[pid]["name"]) for pid, _ in basket], home_lat=cfg.default_lat,
        home_lon=cfg.default_lon, cost_per_km=cfg.travel_cost_per_km, packs=dict(basket))
    best = next(o for o in options if o.recommended)
    assert t["recommended"]["stores"] == best.stores
    assert t["recommended"]["basket_cost"] == best.basket_cost

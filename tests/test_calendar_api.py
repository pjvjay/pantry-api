"""/calendar/preview and /calendar/ics end to end, on both engines (SQLite here, Postgres with
PANTRY_TEST_DB_URL): a demo-starter plan is scheduled, its trips approved as the console
approves them, and the approved_schedule /mealplan/schedule returns is posted back as it came.

Covers PLAN.md P7's list: RFC 5545 structure with folded list text and multibyte product
names, stable UIDs, store hours unknown, no seeded address anywhere, a reminder without a
source (422), a trip that needs review (409), a trip whose product is no longer stocked (409),
the same dates with and without a TZ across 2026-11-01, and no calendar MCP tool. No network:
the export works with every socket refused.
"""
from __future__ import annotations

import copy
import datetime as dt
import json
import re
import socket
import time
from urllib.parse import parse_qs, unquote, urlsplit

import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session

from tests.mealplan_fixtures import (
    START,
    client,
    day,
    doc_draft,
    doc_recipe,
    done_db,
    index,
    schedule,
    starter_draft,
    strategy,
    use_db,
)
from tests.test_calendar_export import unescape, unfold

THIGHS = 11          # Chicken Thighs Bone-In, the demo selector's pick for the biryani
PLAN_ID = "3f1c9a52-6a7e-4d1b-9a43-0d2c0f4b7e15"


@pytest.fixture(scope="module", autouse=True)
def seeded_db(tmp_path_factory):
    use_db(tmp_path_factory, "calendar")
    yield
    done_db()


@pytest.fixture()
def reseed():
    """For tests that change products or stock: the seed is restored afterwards."""
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


def approve(draft: dict, name: str = "fresh") -> tuple[dict, dict]:
    """The console's flow: schedule, store the spread dates back on the meals, approve every
    trip of `name` with its snapshot, schedule again. Returns (draft, schedule)."""
    first = schedule(draft)
    draft = {**draft, "meals": [{"id": m["id"], "recipe_key": m["recipe_key"],
                                 "date": m["date"], "slot": m["slot"]}
                                for m in first["meals"]],
             "trips": [{"date": t["date"], "fingerprint": t["fingerprint"], "strategy": name,
                        "snapshot": [{"product_id": ln["product"]["id"], "packs": ln["packs"],
                                      "storage": ln["storage"],
                                      "price_at_approval": ln["price"]} for ln in t["lines"]]}
                       for t in strategy(first, name)["trips"]]}
    return draft, schedule(draft)


def example(**extra) -> dict:
    return starter_draft(id=PLAN_ID, rev=4, **extra)


@pytest.fixture(scope="module")
def approved():
    return approve(example())


@pytest.fixture(scope="module")
def frozen():
    """fewest_trips with Saturday-only shopping: chicken frozen on arrival, thawed later."""
    return approve(example(prefs={"strategy": "fewest_trips", "shop_weekdays": [5]}),
                   "fewest_trips")


def post(path: str, sched: dict, **body):
    return client().post(path, json={"schedule": sched, **body})


def ics(sched: dict, **body) -> str:
    r = post("/calendar/ics", sched, **body)
    assert r.status_code == 200, r.text
    return r.content.decode("utf-8")


def events_of(ics_text: str) -> list[dict[str, str]]:
    """Each VEVENT as {property name with params: unescaped value}."""
    out, cur = [], None
    for ln in unfold(ics_text):
        if ln == "BEGIN:VEVENT":
            cur = {}
        elif ln == "END:VEVENT":
            out.append(cur)
            cur = None
        elif cur is not None:
            k, v = ln.split(":", 1)
            cur[k] = unescape(v)
    return out


# ─── The emission ────────────────────────────────────────────

def test_the_schedule_emits_approved_trips_placed_meals_and_sourced_reminders(approved, frozen):
    draft, sched = approved
    a = sched["approved_schedule"]
    assert (a["plan_id"], a["rev"], a["exportable"], a["blocked"]) == (PLAN_ID, 4, True, [])
    fresh = strategy(sched, "fresh")["trips"]
    assert [(t["id"], t["date"], t["status"]) for t in a["trips"]] == [
        (t["id"], t["date"], "approved") for t in fresh]
    assert [t["list_text"] for t in a["trips"]] == [t["list_text"] for t in fresh]
    placed = sorted(m["id"] for m in sched["meals"] if m["date"])
    assert sorted(c["meal_id"] for c in a["cooks"]) == placed and len(placed) == 15
    for t in a["trips"]:
        assert t["reason"]["basis"] == "cited" and t["reason"]["rule_ids"]
    # fewest_trips freezes chicken on arrival: its freeze and thaw reminders cite their rows
    rem = frozen[1]["approved_schedule"]["reminders"]
    assert {r["kind"] for r in rem} == {"freeze", "thaw"}
    assert all(r["source"]["basis"] == "cited" and r["source"]["rule_ids"] for r in rem)

    # Before any approval the placed meals are still exportable as cook days; an empty plan
    # has nothing to export.
    unapproved = schedule({**draft, "trips": []})["approved_schedule"]
    assert unapproved["trips"] == [] and len(unapproved["cooks"]) == 15
    assert unapproved["exportable"] is True
    assert schedule({"start_date": START.isoformat(), "days": 7})["approved_schedule"] is None


# ─── RFC 5545 ────────────────────────────────────────────────

def test_the_ics_file_is_rfc_5545_all_day_with_the_trip_list_in_its_description(approved):
    _draft, sched = approved
    a = sched["approved_schedule"]
    r = post("/calendar/ics", a)
    assert r.status_code == 200
    assert r.headers["content-type"] == "text/calendar; charset=utf-8"
    first = min(t["date"] for t in a["trips"])
    assert r.headers["content-disposition"] == f'attachment; filename="pantry-plan-{first}.ics"'
    assert r.headers["cache-control"] == "no-store"
    raw = r.content
    assert raw.startswith(b"BEGIN:VCALENDAR\r\nVERSION:2.0\r\nPRODID:")
    assert raw.endswith(b"END:VCALENDAR\r\n")
    assert b"\n" not in raw.replace(b"\r\n", b"") and b"\r" not in raw.replace(b"\r\n", b"")
    assert max(len(ln) for ln in raw.split(b"\r\n")) <= 75
    assert b"TZID" not in raw and b"VTIMEZONE" not in raw
    events = events_of(raw.decode("utf-8"))
    assert len(events) == len(a["trips"]) + len(a["cooks"]) + len(a["reminders"])
    for e in events:
        start = dt.datetime.strptime(e["DTSTART;VALUE=DATE"], "%Y%m%d").date()
        end = dt.datetime.strptime(e["DTEND;VALUE=DATE"], "%Y%m%d").date()
        assert end - start == dt.timedelta(days=1)
        assert re.fullmatch(r"\d{8}T\d{6}Z", e["DTSTAMP"]) and e["SEQUENCE"] == "4"
        assert {"UID", "SUMMARY", "DESCRIPTION"} <= e.keys()
    for t in a["trips"]:
        (e,) = [e for e in events if e["DTSTART;VALUE=DATE"] == t["date"].replace("-", "")
                and e["SUMMARY"].startswith("Groceries")]
        assert e["DESCRIPTION"].startswith(t["list_text"])
        tail = e["DESCRIPTION"][len(t["list_text"]):].splitlines()
        assert tail[:3] == ["", "Store hours: unknown, check before you go.",
                            f"Why this date: {t['reason']['text']}"]
        assert tail[-1] == f"Demo data: {a['synthetic_notice']}"
        assert e["LOCATION"] == "; ".join(f"{s} (demo store)" for s in t["stores"])
        assert e["CATEGORIES"] == "Pantry plan,Groceries"


def test_folded_list_text_keeps_multibyte_product_names_whole(reseed):
    names = {THIGHS: "Crème fraîche Jalapeño Chicken Thighs (Ñandú édition) – ½ kg",
             166: "Jalapeño Pepperoni à la crème fraîche, tranché finement 🌶"}
    for pid, name in names.items():
        _sql("UPDATE products SET name = :n WHERE id = :p", n=name, p=pid)
    _draft, sched = approve(example())
    a = sched["approved_schedule"]
    assert any(n in t["list_text"] for t in a["trips"] for n in names.values())
    raw = post("/calendar/ics", a).content
    for line in raw.split(b"\r\n"):
        assert len(line) <= 75
        line.decode("utf-8")            # each physical line is whole UTF-8: nothing split
    descriptions = "\n".join(e["DESCRIPTION"] for e in events_of(raw.decode("utf-8")))
    for t in a["trips"]:
        assert t["list_text"] in descriptions
    for name in names.values():
        assert name in descriptions
    # the folds really do fall among the multibyte names
    assert any(re.search(rb"[\x80-\xff]", ln) for ln in raw.split(b"\r\n") if
               ln.startswith(b" "))


# ─── Identity ────────────────────────────────────────────────

def test_uids_are_stable_across_repeated_calls_and_moves(approved):
    draft, sched = approved
    a = sched["approved_schedule"]
    one, two = ics(a), ics(a)
    stamp = re.compile(r"DTSTAMP:\d{8}T\d{6}Z\r\n")
    assert stamp.sub("", one) == stamp.sub("", two)
    uids = [e["UID"] for e in events_of(one)]
    assert len(uids) == len(set(uids))
    assert [e["uid"] for e in post("/calendar/preview", a).json()["events"]] == uids
    # The plan scheduled again gives the same emission and the same UIDs.
    again = schedule(draft)["approved_schedule"]
    assert again == a
    assert [e["UID"] for e in events_of(ics(again))] == uids

    # A meal moved to another day keeps its UID, on its new date, with the new revision as
    # SEQUENCE (the console bumps rev on every edit).
    meals = copy.deepcopy(draft["meals"])
    milk = next(m for m in meals if m["recipe_key"] == "starter:mango_milkshake")
    taken = {(m["date"], m["slot"]) for m in meals}
    to = next(day(i).isoformat() for i in range(14) if (day(i).isoformat(), "snack")
              not in taken)
    milk.update(date=to, pinned=True)
    # The move may change what a trip buys: approved again, as the console would.
    moved = approve({**draft, "meals": meals, "rev": 5})[1]["approved_schedule"]
    before = {e["uid"]: e for e in post("/calendar/preview", a).json()["events"]}
    after = {e["uid"]: e for e in post("/calendar/preview", moved).json()["events"]
             if e["kind"] == "cook"}
    uid = next(u for u, e in before.items() if e["item_id"] == f"cook-{milk['id']}")
    assert after[uid]["date"] == to != before[uid]["date"]
    assert (before[uid]["sequence"], after[uid]["sequence"]) == (4, 5)
    # Another plan never shares a UID.
    other = schedule({**draft, "id": "another-plan"})["approved_schedule"]
    assert not set(uids) & {e["UID"] for e in events_of(ics(other))}


# ─── What the events say ─────────────────────────────────────

def test_store_hours_are_unknown_and_no_seeded_address_appears_anywhere(approved):
    from pantry_planner import storeseed

    _draft, sched = approved
    a = sched["approved_schedule"]
    preview = post("/calendar/preview", a)
    assert preview.status_code == 200
    out = [ics(a), preview.text, json.dumps(a), json.dumps(sched)]
    out += [unquote(e["google_url"]) for e in preview.json()["events"]]
    flat = "\n".join(out)
    stored = [r["address"] for r in _sql("SELECT address FROM stores")]
    addresses = {s[4] for s in storeseed.STORES} | {x for x in stored if x}
    assert addresses
    for address in addresses:
        street = address.split(",")[0]
        assert address not in flat and street not in flat, address
        assert street not in "\n".join(unfold(out[0]))
    trips = [e for e in preview.json()["events"] if e["kind"] == "trip"]
    assert trips and all("store hours unknown" in e["labels"] for e in trips)
    assert all("Store hours: unknown" in e["description"] for e in trips)
    assert all(e["location"].endswith("(demo store)") for e in trips)


def test_an_address_appears_only_where_the_stores_are_real(approved, monkeypatch):
    from pantry_planner import config

    _draft, sched = approved
    a = sched["approved_schedule"]
    monkeypatch.setenv("STORES_SYNTHETIC", "false")
    config.settings.cache_clear()
    try:
        trip = next(e for e in post("/calendar/preview", a).json()["events"]
                    if e["kind"] == "trip")
        store = a["trips"][0]["stores"][0]
        address = _sql("SELECT address FROM stores WHERE name = :n", n=store)[0]["address"]
        assert trip["location"].startswith(f"{store}, {address}")
        assert "demo store" not in trip["labels"]
    finally:
        monkeypatch.delenv("STORES_SYNTHETIC")
        config.settings.cache_clear()


def test_cook_events_carry_ingredients_servings_and_nutrition_with_the_badge(approved):
    _draft, sched = approved
    a = sched["approved_schedule"]
    events = {e["item_id"]: e for e in post("/calendar/preview", a).json()["events"]}
    for c in a["cooks"]:
        e = events[c["item_id"]]
        assert e["date"] == c["date"] and e["title"].startswith(f"Cook: {c['title']}")
        assert all(f"- {line}" in e["description"] for line in c["ingredients"])
        assert f"for {c['servings']} people" in e["description"]
        if c["nutrition"] is None:
            assert "Nutrition:" in e["description"]
        elif c["nutrition"]["demo_amounts"]:
            assert "Nutrition per serving (demo amounts):" in e["description"]
            assert "demo amounts" in e["labels"]
    # The starters are demo house amounts, so the badge is on every one of them.
    assert all(c["nutrition"] is None or c["nutrition"]["demo_amounts"] for c in a["cooks"])

    # A recipe the shopper pasted, with its own amounts: no badge.
    entry = doc_recipe("my:s", "Salmon Rice", 2, [("Atlantic Salmon", 400, "g", 47),
                                                  ("Basmati Rice", 200, "g", 8)])
    meals = [{"id": "m", "recipe_key": "my:s", "date": day(2).isoformat()}]
    mine = schedule(doc_draft({"my:s": entry}, meals=meals, id=PLAN_ID))["approved_schedule"]
    (e,) = post("/calendar/preview", mine).json()["events"]
    assert "demo amounts" not in e["description"] and "demo amounts" not in e["labels"]
    assert "- 400 g Atlantic Salmon" in e["description"]


def test_freeze_and_thaw_reminders_cite_their_rows(frozen):
    _draft, sched = frozen
    a = sched["approved_schedule"]
    events = [e for e in post("/calendar/preview", a).json()["events"]
              if e["kind"] in ("freeze", "thaw")]
    assert len(events) == len(a["reminders"]) > 0
    for e in events:
        assert e["labels"] == ["cited"]
        assert "Source: " in e["description"] and "https://" in e["description"]


def test_include_and_nothing_to_export(approved):
    _draft, sched = approved
    a = sched["approved_schedule"]
    only = post("/calendar/preview", a, include=["cooks"]).json()
    assert {e["kind"] for e in only["events"]} == {"cook"}
    assert only["counts"]["trip"] == 0
    empty = {**a, "trips": [], "cooks": [], "reminders": []}
    assert post("/calendar/preview", empty).json()["events"] == []
    r = post("/calendar/ics", empty)
    assert (r.status_code, r.json()["detail"]["error"]) == (422, "nothing_to_export")
    assert post("/calendar/ics", a, include=[]).status_code == 422


def test_google_links_are_built_for_every_event(approved):
    _draft, sched = approved
    for e in post("/calendar/preview", sched["approved_schedule"]).json()["events"]:
        parts = urlsplit(e["google_url"])
        assert (parts.scheme, parts.netloc, parts.path) == ("https", "calendar.google.com",
                                                            "/calendar/render")
        q = {k: v[0] for k, v in parse_qs(parts.query).items()}
        assert q["action"] == "TEMPLATE" and q["text"] == e["title"]
        assert q["dates"] == f"{e['date'].replace('-', '')}/{e['end'].replace('-', '')}"
        assert len(q["details"]) <= 1000


# ─── Refusals ────────────────────────────────────────────────

def test_a_reminder_without_a_source_is_422(frozen):
    a = frozen[1]["approved_schedule"]
    for broken in ({k: v for k, v in a["reminders"][0].items() if k != "source"},
                   {**a["reminders"][0], "source": None},
                   {**a["reminders"][0], "source": {"basis": "cited", "rule_ids": []}},
                   {**a["reminders"][0], "source": {"basis": "your_setting"}}):
        bad = {**a, "reminders": [broken, *a["reminders"][1:]]}
        for path in ("/calendar/preview", "/calendar/ics"):
            r = post(path, bad)
            assert r.status_code == 422, (path, r.text)
            assert any("source" in str(err["loc"]) for err in r.json()["detail"])
    made_up = {**a, "reminders": [{**a["reminders"][0],
                                   "source": {"basis": "cited", "rule_ids": ["no-such-rule"]}}]}
    r = post("/calendar/ics", made_up)
    assert (r.status_code, r.json()["detail"]["error"]) == (422, "unknown_rule")


def test_a_trip_that_needs_review_is_409(approved):
    draft, sched = approved
    # Move the last biryani two days later: the trip that bought its chicken changes.
    meals = copy.deepcopy(draft["meals"])
    moving = next(m for m in meals if m["id"] == "starter:chicken_biryani#3")
    assert index(moving["date"]) == 11
    moving.update(date=day(13).isoformat(), pinned=True)
    moved = schedule({**draft, "meals": meals})
    a = moved["approved_schedule"]
    review = [t for t in a["trips"] if t["status"] == "needs_review"]
    assert review and not any(t["not_stocked"] for t in review)
    assert a["exportable"] is False
    assert [b["code"] for b in a["blocked"]] == ["needs_review"] * len(review)
    for path in ("/calendar/preview", "/calendar/ics"):
        r = post(path, a)
        assert r.status_code == 409
        detail = r.json()["detail"]
        assert (detail["error"], detail["item_id"]) == ("needs_review", review[0]["item_id"])
    # Approving it again as it stands makes it exportable.
    again = approve({**draft, "meals": meals})[1]["approved_schedule"]
    assert again["exportable"] is True and post("/calendar/ics", again).status_code == 200


def test_a_trip_whose_product_is_no_longer_stocked_is_409(approved, reseed):
    draft, _sched = approved
    _sql("DELETE FROM store_products WHERE product_id = :p", p=THIGHS)
    after = schedule(draft)["approved_schedule"]
    gone = [t for t in after["trips"] if t["not_stocked"]]
    assert gone and "Chicken Thighs Bone-In" in gone[0]["not_stocked"]
    assert gone[0]["status"] == "needs_review"
    assert "no_longer_stocked" in [b["code"] for b in after["blocked"]]
    for path in ("/calendar/preview", "/calendar/ics"):
        r = post(path, after)
        assert r.status_code == 409
        detail = r.json()["detail"]
        assert (detail["error"], detail["item_id"]) == ("no_longer_stocked", gone[0]["item_id"])
        assert "Chicken Thighs Bone-In" in detail["detail"]


def test_a_suggested_trip_posted_as_part_of_the_schedule_is_409(approved):
    a = copy.deepcopy(approved[1]["approved_schedule"])
    a["trips"][0]["status"] = "suggested"
    r = post("/calendar/ics", a)
    assert (r.status_code, r.json()["detail"]["error"]) == (409, "not_approved")


def test_a_request_over_512_kb_is_413(approved):
    a = approved[1]["approved_schedule"]
    r = client().post("/calendar/ics", json={"schedule": a, "pad": "x" * 600_000})
    assert r.status_code == 413


# ─── Time zones ──────────────────────────────────────────────

ZONES = [None, "America/Vancouver", "Pacific/Kiritimati", "Pacific/Pago_Pago", "UTC"]


def test_dates_are_the_same_with_and_without_a_tz_across_2026_11_01(monkeypatch):
    start = dt.date(2026, 10, 26)
    draft, sched = approve(starter_draft(id=PLAN_ID, start_date=start.isoformat()))
    a = sched["approved_schedule"]
    plan_dates = sorted({c["date"] for c in a["cooks"]} | {t["date"] for t in a["trips"]})
    assert plan_dates[0] < "2026-11-01" < plan_dates[-1]

    def dates() -> tuple[list, list]:
        evs = events_of(ics(a))
        prev = post("/calendar/preview", a).json()["events"]
        links = [parse_qs(urlsplit(e["google_url"]).query)["dates"][0] for e in prev]
        return ([(e["UID"], e["DTSTART;VALUE=DATE"], e["DTEND;VALUE=DATE"]) for e in evs],
                links)

    seen, offsets = [], set()
    try:
        for zone in ZONES:
            if zone is None:
                monkeypatch.delenv("TZ", raising=False)
            else:
                monkeypatch.setenv("TZ", zone)
            time.tzset()
            offsets.add(time.strftime("%z", time.localtime(1793520000)))   # 2026-11-01 08:00 UTC
            seen.append(dates())
            # the meals' own dates, whatever the zone
            got = {e["item_id"]: e["date"] for e in post("/calendar/preview", a).json()["events"]}
            assert all(got[c["item_id"]] == c["date"] for c in a["cooks"])
    finally:
        monkeypatch.undo()
        time.tzset()
    assert {"+1400", "-1100"} <= offsets          # the zones really did change
    assert all(s == seen[0] for s in seen[1:])
    assert {d for _uid, d, _end in seen[0][0]} == {d.replace("-", "") for d in plan_dates}


# ─── No agent tool, no network ───────────────────────────────

@pytest.mark.asyncio
async def test_no_calendar_mcp_tool_is_registered():
    from pantry_planner.mcp_server import server

    names = [t.name for t in await server.list_tools()]
    words = {"calendar", "calendars", "ics", "ical", "event", "events", "export", "gcal"}
    assert names and not [n for n in names if words & set(n.lower().split("_"))
                          or "calendar" in n.lower()]


def test_the_export_makes_no_network_call(approved, monkeypatch):
    def refuse(*_a, **_k):
        raise AssertionError("the calendar export opened a network connection")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    a = approved[1]["approved_schedule"]
    assert post("/calendar/preview", a).status_code == 200
    assert post("/calendar/ics", a).status_code == 200

"""calendar_export.py on its own: RFC 5545 text, folding, UIDs, Google links and the event
builder, from hand-built ApprovedSchedules. No database, no network.

The end-to-end path (/mealplan/schedule, then /calendar/preview and /calendar/ics on both
engines) is test_calendar_api.py.
"""
from __future__ import annotations

import datetime as dt
import hashlib
from urllib.parse import parse_qs, urlsplit

import pytest
from pydantic import ValidationError

from pantry_planner import calendar_export as ce
from pantry_planner.models import NutrientTotal

D = dt.date(2026, 10, 31)          # the Saturday before BC's last clock change


def unfold(text: str) -> list[str]:
    """Content lines from an iCalendar text: CRLF splits, a leading space continues."""
    out: list[str] = []
    for line in text.split("\r\n"):
        if line.startswith(" "):
            out[-1] += line[1:]
        else:
            out.append(line)
    assert out[-1] == ""
    return out[:-1]


def unescape(value: str) -> str:
    out, i = [], 0
    while i < len(value):
        if value[i] == "\\" and i + 1 < len(value):
            nxt = value[i + 1]
            out.append("\n" if nxt in "nN" else nxt)
            i += 2
        else:
            out.append(value[i])
            i += 1
    return "".join(out)


def trip(item: str = "trip-fresh-2026-10-31", date: dt.date = D, **kw) -> ce.ScheduleTrip:
    base = dict(
        item_id=item, id=item.removeprefix("trip-"), strategy="fresh", date=date,
        status="approved", stores=["Pantry Mart Downtown", "ValueFoods East Van"],
        reason=ce.TripReason(text="Chicken Biryani on Sun 1 Nov needs Chicken Thighs, which "
                                  "keeps 1 to 2 days in the fridge (FoodSafety.gov, Cold Food "
                                  "Storage Chart, fs-poultry-pieces-fridge).", basis="cited",
                             rule_ids=["fs-poultry-pieces-fridge"],
                             url="https://www.foodsafety.gov/x", page_date="2023-09-19"),
        list_text="Shopping trip Sat 31 Oct 2026 (fresh)\n\nPantry Mart Downtown\n Dairy\n"
                  "  - 1 x Crème fraîche 250ml: need 100 ml; $3.49\n\nTotal $3.49\n"
                  "Prices and stock are demo data.\n",
        total_cost=3.49, total_is_floor=False)
    return ce.ScheduleTrip(**{**base, **kw})


def cook(item: str = "cook-m1", date: dt.date = D, **kw) -> ce.ScheduleCook:
    base = dict(item_id=item, meal_id=item.removeprefix("cook-"), recipe_key="my:x",
                title="Jalapeño Poppers", date=date, slot="dinner", servings=2,
                recipe_servings=4, ingredients=["6 Jalapeño peppers", "100 g Crème fraîche"])
    return ce.ScheduleCook(**{**base, **kw})


def schedule(**kw) -> ce.ApprovedSchedule:
    base = dict(plan_id="plan-1", rev=5, start_date=D, days=14, trips=[trip()], cooks=[cook()],
                synthetic_notice="Stores, prices and stock are synthetic demo data.")
    return ce.ApprovedSchedule(**{**base, **kw})


NOW = dt.datetime(2026, 10, 8, 12, 0, tzinfo=dt.UTC)


# ─── Folding and escaping ────────────────────────────────────

@pytest.mark.parametrize("ch", ["é", "ñ", "€", "🥭"])       # 2, 2, 3 and 4 UTF-8 octets
def test_folding_never_splits_a_multibyte_character_at_any_offset(ch):
    for prefix in range(0, 80):
        line = "DESCRIPTION:" + "x" * prefix + ch * 60
        folded = ce.fold(line)
        parts = folded.split("\r\n")
        for i, part in enumerate(parts):
            raw = part.encode("utf-8")              # str parts: decoding cannot have failed
            assert len(raw) <= 75, (prefix, i, len(raw))
            assert i == 0 or part.startswith(" ")
            # the bytes as a reader sees them: each physical line is valid UTF-8 on its own
            raw.decode("utf-8", errors="strict")
        assert "".join(p[1:] if i else p for i, p in enumerate(parts)) == line
        # every line but the last is full: no break is made earlier than it has to be
        for part in parts[:-1]:
            assert len(part.encode("utf-8")) > 75 - len(ch.encode("utf-8"))


def test_a_short_line_is_not_folded():
    assert ce.fold("SUMMARY:" + "a" * 67) == "SUMMARY:" + "a" * 67
    assert "\r\n" in ce.fold("SUMMARY:" + "a" * 68)


def test_text_escaping_follows_rfc_5545():
    assert ce.escape_text("a,b;c\\d\ne") == "a\\,b\\;c\\\\d\\ne"
    assert ce.escape_text("one\r\ntwo\rthree") == "one\\ntwo\\nthree"
    assert ce.escape_text("bell\x07 tab\tdel\x7f") == "bell tab\tdel"
    assert unescape(ce.escape_text("Crème; fraîche, \\ok\n")) == "Crème; fraîche, \\ok\n"


# ─── The file ────────────────────────────────────────────────

def test_rfc_5545_structure_of_an_all_day_calendar():
    events = ce.build_events(schedule())
    text = ce.render_ics(events, now=NOW)
    raw = text.encode("utf-8")
    assert raw.endswith(b"END:VCALENDAR\r\n")
    assert b"\n" not in raw.replace(b"\r\n", b"") and b"\r" not in raw.replace(b"\r\n", b"")
    assert all(len(ln) <= 75 for ln in raw.split(b"\r\n"))
    lines = unfold(text)
    assert lines[:5] == ["BEGIN:VCALENDAR", "VERSION:2.0", f"PRODID:{ce.PRODID}",
                         "CALSCALE:GREGORIAN", "X-WR-CALNAME:Pantry plan"]
    assert lines.count("BEGIN:VEVENT") == lines.count("END:VEVENT") == 2
    assert not any(ln.startswith(("TZID", "BEGIN:VTIMEZONE", "METHOD")) or ";TZID=" in ln
                   for ln in lines)
    for block in _events(lines):
        props = {k.split(";")[0]: v for k, v in (ln.split(":", 1) for ln in block)}
        for name in ("UID", "DTSTAMP", "DTSTART", "DTEND", "SUMMARY", "DESCRIPTION",
                     "SEQUENCE"):
            assert name in props, name
        assert props["DTSTAMP"] == "20261008T120000Z"
        assert props["SEQUENCE"] == "5"
        assert "DTSTART;VALUE=DATE:20261031" in block and "DTEND;VALUE=DATE:20261101" in block
        assert props["TRANSP"] == "TRANSPARENT"


def test_a_trip_with_no_priced_line_exports_with_its_total_unknown():
    unpriced = trip(total_cost=None, total_is_floor=True,
                    list_text="Shopping trip Sat 31 Oct 2026 (fresh)\n\nPantry Mart Downtown\n"
                              "  - 1 x Crème fraîche 250ml: need 100 ml; price unknown\n\n"
                              "Total unknown (no line has a price yet)\n")
    text = "".join(unfold(ce.render_ics(ce.build_events(schedule(trips=[unpriced])), now=NOW)))
    assert "Total unknown (no line has a price yet)" in text
    assert "$0.00" not in text


def _events(lines: list[str]) -> list[list[str]]:
    out, cur = [], None
    for ln in lines:
        if ln == "BEGIN:VEVENT":
            cur = []
        elif ln == "END:VEVENT":
            out.append(cur)
            cur = None
        elif cur is not None:
            cur.append(ln)
    return out


def test_the_trip_description_round_trips_through_escaping_and_folding():
    s = schedule()
    text = ce.render_ics(ce.build_events(s), now=NOW)
    (trip_block, _) = _events(unfold(text))
    desc = unescape(next(v for ln in trip_block for k, v in [ln.split(":", 1)]
                         if k == "DESCRIPTION"))
    t = s.trips[0]
    assert desc == "\n".join([
        t.list_text.rstrip("\n"), "", "Store hours: unknown, check before you go.",
        f"Why this date: {t.reason.text}",
        "Source: https://www.foodsafety.gov/x, page dated 2023-09-19.",
        "Demo data: Stores, prices and stock are synthetic demo data."])
    assert "LOCATION:Pantry Mart Downtown (demo store)\\; ValueFoods East Van (demo store)" \
        in unfold(text)


# ─── Identity ────────────────────────────────────────────────

def test_uids_hash_the_plan_and_the_item_and_repeat():
    a = ce.build_events(schedule())
    b = ce.build_events(schedule())
    assert [e.uid for e in a] == [e.uid for e in b]
    want = hashlib.sha256(b"plan-1\ncook-m1").hexdigest()[:32] + "@pantry-planner"
    assert next(e.uid for e in a if e.kind == "cook") == want
    other = ce.build_events(schedule(plan_id="plan-2"))
    assert not {e.uid for e in a} & {e.uid for e in other}
    # a moved meal keeps its UID; the revision is its SEQUENCE
    moved = ce.build_events(schedule(rev=6, cooks=[cook(date=D + dt.timedelta(days=3))]))
    c0, c1 = (next(e for e in x if e.kind == "cook") for x in (a, moved))
    assert (c0.uid, c0.date, c0.sequence) == (c1.uid, D, 5)
    assert (c1.date, c1.sequence) == (D + dt.timedelta(days=3), 6)
    # a plan saved without an id still gets stable UIDs, from its start date
    assert ce.plan_key(schedule(plan_id="")) == "plan-2026-10-31"


def test_item_ids_must_be_unique():
    with pytest.raises(ValidationError, match="unique"):
        schedule(cooks=[cook(), cook()])


# ─── Refusals ────────────────────────────────────────────────

def test_a_trip_that_needs_review_or_lost_a_product_is_refused_with_409():
    with pytest.raises(ce.CalendarExportError) as e:
        ce.build_events(schedule(trips=[trip(status="needs_review")]))
    assert (e.value.status, e.value.code) == (409, "needs_review")
    assert e.value.extra == {"item_id": "trip-fresh-2026-10-31", "date": "2026-10-31"}
    with pytest.raises(ce.CalendarExportError) as e:
        ce.build_events(schedule(trips=[trip(status="needs_review",
                                             not_stocked=["Chicken Thighs Bone-In"])]))
    assert (e.value.status, e.value.code) == (409, "no_longer_stocked")
    assert "Chicken Thighs Bone-In" in e.value.detail
    with pytest.raises(ce.CalendarExportError) as e:
        ce.build_events(schedule(trips=[trip(status="suggested")]))
    assert (e.value.status, e.value.code) == (409, "not_approved")


def reminder(**kw) -> dict:
    return {"item_id": "thaw-m1-11", "kind": "thaw", "date": "2026-10-30", "product_id": 11,
            "product": "Chicken Thighs Bone-In", "text": "Move Chicken Thighs Bone-In to the "
            "fridge to thaw for Jalapeño Poppers on Sat 31 Oct", "trip_id": "fresh-2026-10-31",
            **kw}


def test_a_reminder_needs_a_source():
    base = schedule().model_dump(mode="json")
    for bad in (reminder(), reminder(source=None), reminder(source={"basis": "cited"}),
                reminder(source={"basis": "cited", "rule_ids": []}),
                reminder(source={"basis": "your_setting"}),
                reminder(source={"basis": "a guess", "rule_ids": ["x"]})):
        with pytest.raises(ValidationError):
            ce.ApprovedSchedule.model_validate({**base, "reminders": [bad]})
    ok = ce.ApprovedSchedule.model_validate({**base, "reminders": [
        reminder(source={"basis": "cited", "rule_ids": ["fsis-thaw-fridge-small"]}),
        reminder(item_id="thaw-m1-12", source={"basis": "your_setting",
                                               "setting": "thaw reminders: the evening before"})]})
    events = ce.build_events(ok)
    thaw = [e for e in events if e.kind == "thaw"]
    assert thaw[0].labels == ["cited"] and "fsis-thaw-fridge-small" in thaw[0].description
    assert "page dated" in thaw[0].description and "https://" in thaw[0].description
    assert thaw[1].labels == ["your setting"]
    assert "Source: your setting (thaw reminders: the evening before)." in thaw[1].description


def test_a_reminder_citing_a_rule_that_does_not_exist_is_refused():
    s = ce.ApprovedSchedule.model_validate({**schedule().model_dump(mode="json"), "reminders": [
        reminder(source={"basis": "cited", "rule_ids": ["made-up-rule"]})]})
    with pytest.raises(ce.CalendarExportError) as e:
        ce.build_events(s)
    assert (e.value.status, e.value.code) == (422, "unknown_rule")


# ─── What the events say ─────────────────────────────────────

def test_locations_name_demo_stores_and_never_an_address():
    assert ce.location_for(["A", "B"], None) == "A (demo store); B (demo store)"
    assert ce.location_for([], None) is None
    assert ce.location_for(["A", "B"], {"A": "1 Real St"}) == "A, 1 Real St; B"


def total(amount, status, unit="kcal") -> NutrientTotal:
    return NutrientTotal(amount=amount, unit=unit, status=status,
                            complete=status == "complete", lines_counted=1, lines_total=1)


def test_nutrient_text_never_shows_0_for_a_missing_figure():
    assert ce.nutrient_text("energy_kcal", total(642.5, "complete")) == "643 kcal"
    assert ce.nutrient_text("energy_kcal", total(1840.4, "at_least")) == "≥ 1,840 kcal"
    assert ce.nutrient_text("fibre_g", total(2.46, "complete", "g")) == "2.5 g fibre"
    assert ce.nutrient_text("sodium_mg", total(None, "unknown", "mg")) == "sodium unknown"
    assert ce.nutrient_text("energy_kcal", total(0, "unknown")) == "kcal unknown"
    assert ce.nutrient_text("protein_g", None) == "protein unknown"


def nutrition(demo: bool) -> ce.CookNutrition:
    return ce.CookNutrition(basis="per_serving", servings=4, status="incomplete",
                            totals={"energy_kcal": total(610, "at_least"),
                                    "protein_g": total(35.2, "complete", "g")},
                            demo_amounts=demo, coverage_note="5 of 6 ingredients counted")


def test_a_cook_event_lists_ingredients_servings_and_nutrition_with_its_badge():
    (c,) = [e for e in ce.build_events(schedule(cooks=[cook(nutrition=nutrition(True))]))
            if e.kind == "cook"]
    assert c.title == "Cook: Jalapeño Poppers (dinner)"
    assert c.description.splitlines()[:6] == [
        "Dinner for 2 people (your household setting).",
        "The recipe makes 4 servings; for 2, use 1/2 of each amount.", "",
        "Ingredients, as the recipe gives them:", "- 6 Jalapeño peppers",
        "- 100 g Crème fraîche"]
    assert ("Nutrition per serving (demo amounts): ≥ 610 kcal, 35 g protein, fat unknown"
            in c.description)
    assert "Recipe amounts are demo house amounts, not from a published recipe." in c.description
    assert c.labels == ["demo amounts"]

    (plain,) = [e for e in ce.build_events(schedule(cooks=[cook(nutrition=nutrition(False))]))
                if e.kind == "cook"]
    assert "demo amounts" not in plain.description and plain.labels == []

    (none,) = [e for e in ce.build_events(schedule(cooks=[cook(
        recipe_servings=None, nutrition_note="nutrition not shown: tables not deployed",
        occurrence=2, of=3)])) if e.kind == "cook"]
    assert none.title == "Cook: Jalapeño Poppers (dinner, 2 of 3)"
    assert "does not say how many it serves" in none.description
    assert "Nutrition: nutrition not shown: tables not deployed." in none.description
    assert none.labels == ["nutrition not shown", "servings unknown"]


def test_include_picks_the_kinds_of_event():
    s = schedule()
    assert {e.kind for e in ce.build_events(s, include={"cooks"})} == {"cook"}
    assert {e.kind for e in ce.build_events(s, include={"trips"})} == {"trip"}


def test_events_come_in_date_then_kind_order():
    s = schedule(cooks=[cook("cook-b", D, slot="dinner"), cook("cook-a", D, slot="lunch"),
                        cook("cook-c", D - dt.timedelta(days=1))])
    assert [e.item_id for e in ce.build_events(s)] == [
        "cook-c", "trip-fresh-2026-10-31", "cook-a", "cook-b"]


# ─── Google add-event links ──────────────────────────────────

def test_google_links_are_all_day_template_urls_with_a_capped_description():
    url = ce.google_template_url("Cook: <b>Pie</b> & tea", D, D + dt.timedelta(days=1),
                                 "x" * 3000, "Pantry Mart Downtown (demo store)")
    parts = urlsplit(url)
    assert (parts.scheme, parts.netloc, parts.path) == ("https", "calendar.google.com",
                                                        "/calendar/render")
    q = {k: v[0] for k, v in parse_qs(parts.query).items()}
    assert q["action"] == "TEMPLATE" and q["dates"] == "20261031/20261101"
    assert q["text"] == "Cook: <b>Pie</b> & tea"
    assert len(q["details"]) == ce.GOOGLE_DETAILS_MAX
    assert q["details"].endswith("the full text is in the .ics export.")
    assert q["location"] == "Pantry Mart Downtown (demo store)"
    escaped = ce.google_template_url("t", D, D, "<script>&", None)
    assert parse_qs(urlsplit(escaped).query)["details"] == ["&lt;script&gt;&amp;"]
    assert "location" not in parse_qs(urlsplit(escaped).query)

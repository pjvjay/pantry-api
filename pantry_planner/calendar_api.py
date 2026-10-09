"""POST /calendar/preview and POST /calendar/ics: an approved meal plan as calendar events.

Both take {schedule: ApprovedSchedule, include?: [trips, cooks, reminders]}, where schedule is
the approved_schedule that /mealplan/schedule returned, posted back as it came. They are pure:
no LLM, no write, no outbound call, no credentials. The events come from the one builder in
calendar_export.py, so the preview shows exactly what the file holds.

- preview: 200 {calendar_name, all_day, events[], counts, filename, notes}, each event with
  its Google add-event link (built, never fetched);
- ics: 200 text/calendar, an attachment named pantry-plan-<first date>.ics; 422
  nothing_to_export when no event is left.

Refusals: 409 not_approved, needs_review or no_longer_stocked while a trip is not approved as
it stands; 422 for a body that fails validation (a reminder without a source among them) or a
reminder citing a rule shelf_life.json does not have; 413 above 512 KB. There is no MCP tool
for any of this: every export is a click by the shopper.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from . import db, limits
from .calendar_export import (
    INCLUDE_ALL,
    ApprovedSchedule,
    CalendarEvent,
    CalendarExportError,
    CalendarPreview,
    Include,
    build_events,
    ics_filename,
    preview,
    render_ics,
)
from .config import settings

MAX_BODY_BYTES = 512 * 1024

router = APIRouter(tags=["calendar"])


class CalendarRequest(BaseModel):
    schedule: ApprovedSchedule
    include: list[Include] = Field(default_factory=lambda: list(INCLUDE_ALL), min_length=1,
                                   max_length=3)


async def _body_size(request: Request) -> None:
    declared = request.headers.get("content-length", "")
    size = int(declared) if declared.isdigit() else len(await request.body())
    if size > MAX_BODY_BYTES:
        raise HTTPException(status_code=413, detail={
            "error": "too_large",
            "detail": f"A calendar export request is at most {MAX_BODY_BYTES} bytes."})


def _addresses() -> dict[str, str] | None:
    """Store addresses by name, only for a deployment whose stores are real. While they are
    synthetic (the default) the addresses are not even read."""
    if settings().stores_synthetic:
        return None
    with Session(db.engine()) as s:
        rows = s.execute(select(db.StoreRow)).scalars().all()
        return {r.name: r.address for r in rows if r.address}


def _events(req: CalendarRequest) -> list[CalendarEvent]:
    try:
        return build_events(req.schedule, include=set(req.include), addresses=_addresses())
    except CalendarExportError as e:
        raise HTTPException(status_code=e.status, detail={
            "error": e.code, "detail": e.detail, **e.extra}) from e


@router.post("/calendar/preview", response_model=CalendarPreview,
             dependencies=[Depends(_body_size),
                           Depends(limits.rate_limit("/calendar/preview"))])
def calendar_preview(req: CalendarRequest) -> CalendarPreview:
    """The all-day events an export of this approved schedule holds, each with its Google
    add-event link: trips (the shopping list, store hours unknown, the cited reason, demo
    stores), freeze and thaw reminders (each with its source) and cook days (ingredients,
    servings, nutrition with the demo-amounts badge). 409 while a trip needs review or a
    product on it is no longer stocked."""
    events = _events(req)
    return preview(req.schedule, events)


@router.post("/calendar/ics", response_class=Response,
             responses={200: {"content": {"text/calendar": {}},
                              "description": "An RFC 5545 calendar of all-day events."}},
             dependencies=[Depends(_body_size), Depends(limits.rate_limit("/calendar/ics"))])
def calendar_ics(req: CalendarRequest) -> Response:
    """The same events as /calendar/preview, as an .ics file to import (RFC 5545: CRLF, lines
    folded at 75 octets, all-day VALUE=DATE events, stable UIDs, SEQUENCE the plan's
    revision)."""
    events = _events(req)
    if not events:
        raise HTTPException(status_code=422, detail={
            "error": "nothing_to_export",
            "detail": "Nothing to export: approve a trip or place a meal first."})
    body = render_ics(events, calendar_name=req.schedule.calendar_name)
    name = ics_filename(req.schedule, events)
    return Response(content=body.encode("utf-8"), media_type="text/calendar; charset=utf-8",
                    headers={"Content-Disposition": f'attachment; filename="{name}"',
                             "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"})

"""
Thin FastAPI wrapper — turns the Burr flow into an HTTP endpoint.

Deliberately minimal: this repo is about the pipeline design, not the
web layer. Every route delegates to pantry_planner.flow or pantry_planner.db.
"""
from __future__ import annotations

import contextlib
import math
import os
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from . import db, flow, limits
from .config import (
    RUNTIME_MODEL_FIELDS,
    set_runtime_overrides,
    settings,
    validate_model_spec,
)
from .llm import LLMError
from .mealplan.models import MealPlanDraft, MealSchedule, RecipeRef, ResolvedRecipe
from .models import (
    MAX_DOC_LINES,
    MAX_LINE_NAME,
    MAX_LINE_QUANTITY,
    MAX_SERVINGS,
    AlternativeRanking,
    OriginRanking,
    Pin,
    PlanBasis,
    Product,
    ProductOrigin,
    Recipe,
    RecipeDoc,
    ShoppingPlan,
    WeekPlan,
)

# The MCP Streamable-HTTP endpoint rides this app at /mcp (mounted at
# the bottom of the file). MCP_HTTP_ENABLED=false turns it off — the
# endpoint shares the API's no-auth posture, and plan tools spend
# Anthropic credits.
MCP_HTTP_ENABLED = os.environ.get("MCP_HTTP_ENABLED", "true").lower() != "false"


@contextlib.asynccontextmanager
async def _lifespan(app: FastAPI):
    # Starlette never runs a mounted sub-app's lifespan, so the MCP
    # session manager must be driven from here — without it every /mcp
    # request 500s even though everything imports cleanly.
    # Drive the manager of the app mounted below, not whichever one the
    # SDK built last: every streamable_http_app() call installs a fresh
    # session manager on the server, so a later call elsewhere (a test
    # inspecting routes) would otherwise leave the mounted app's own
    # manager never started and every /mcp request a 500.
    # Fail at startup, not on the first request: a weak or malformed
    # MCP_AUTH_TOKENS must stop the process here, before readiness passes.
    from .config import validate_startup

    validate_startup()
    if MCP_HTTP_ENABLED:
        async with _MCP_SESSION_MANAGER.run():
            yield
    else:
        yield


app = FastAPI(
    title="pantry-planner",
    version="0.1.0",
    description=(
        "Match recipe ingredients to store products with an LLM-driven pipeline. "
        "Toggle routing strategy via ROUTING_STRATEGY env var."
    ),
    lifespan=_lifespan,
)


@app.exception_handler(LLMError)
async def _llm_error(request: Request, exc: LLMError) -> JSONResponse:
    """A failed LLM call is the provider's (or the config's) failure, not a
    500 and not the caller's 4xx: 502 by default, 429 when the quota ran
    out, 503 when the call can't be made as configured (no key). The
    message names what to fix and never carries a key."""
    return JSONResponse(status_code=exc.http_status, content={
        "error": "llm_call_failed", "provider": exc.provider or None,
        "detail": str(exc)})


@app.exception_handler(limits.LimitError)
async def _limit_refused(request: Request, exc: limits.LimitError) -> JSONResponse:
    """A rate limit or the daily LLM ceiling, shaped like an LLM failure: `detail` is the
    sentence to show. The console shows a string `detail` as it is and anything else as
    "[object Object]", and these refusals now reach endpoints it already calls."""
    return JSONResponse(status_code=exc.status, headers=exc.headers,
                        content={"error": exc.error, "detail": exc.detail})


def _json_safe(value: Any) -> Any:
    """`value` with every float JSON cannot carry (inf, -inf, nan) written as its name."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


@app.exception_handler(RequestValidationError)
async def _invalid_request(request: Request, exc: RequestValidationError) -> JSONResponse:
    """FastAPI's own 422, with the same body, made renderable for any input. Python's JSON
    parser reads 1e309 as inf and accepts NaN, neither of which JSON can carry; the 422
    echoes the rejected input back, so rendering it raised and the client got a 500 for
    what was its own mistake."""
    return JSONResponse(status_code=422,
                        content={"detail": _json_safe(jsonable_encoder(exc.errors()))})


def _check_countries(*lists: list[str] | None) -> None:
    """Reject unrecognised country names up front.

    A filter that accepts "Amerca", excludes nothing and returns 200 has
    reported success on a request it did not honour. For a boycott tool that
    is the worst available outcome, so it is a 422 with the closest matches.
    """
    from .origins import validate_countries

    names = [n for lst in lists for n in (lst or [])]
    unknown = validate_countries(names)
    if unknown:
        raise HTTPException(status_code=422, detail={
            "error": "unrecognised country name(s)",
            "unknown": unknown,
            "hint": "Use a country name or common alias, e.g. 'United States', 'USA', 'Canada'.",
        })


@app.get("/health")
def health() -> dict:
    cfg = settings()            # effective: env + any runtime overrides
    return {
        "status": "ok",
        "routing_strategy": cfg.routing_strategy,
        "default_model": cfg.selector_model_default,
        "escalation_model": cfg.selector_model_escalation,
        "classifier_model": cfg.classifier_model,
        "nl2sql_model": cfg.nl2sql_model,
        "confidence_threshold": cfg.confidence_threshold,
        "demo_mode": cfg.demo_mode,
        "gemini_key_configured": bool(cfg.gemini_api_key),
        # Today's estimated LLM spend on this replica and the ceiling (None: no ceiling).
        "llm_budget": limits.budget(),
    }


# ─── Runtime settings (demo UI) ──────────────────────────────
# Demo mode and the four model specs, switchable without a restart. GET is
# always available (no secrets: key PRESENCE only). POST needs
# RUNTIME_SETTINGS_ENABLED=1 — on a public deployment it would let any
# visitor turn real LLM spend on.

class RuntimeModels(BaseModel):
    """Model specs: "gemini:<model>", "anthropic:<model>" or "claude-..."."""

    model_config = ConfigDict(extra="forbid")
    selector_default: str | None = None
    selector_escalation: str | None = None
    classifier: str | None = None
    nl2sql: str | None = None

    @field_validator("*")
    @classmethod
    def _valid_spec(cls, v: str | None) -> str | None:
        return None if v is None else validate_model_spec(v)


class RuntimeSettingsUpdate(BaseModel):
    """Any subset; omitted fields keep their current value."""

    model_config = ConfigDict(extra="forbid")
    demo_mode: bool | None = None
    models: RuntimeModels | None = None


class RuntimeSettings(BaseModel):
    demo_mode: bool
    models: dict[str, str]
    gemini_key_configured: bool
    anthropic_key_configured: bool
    editable: bool


def _runtime_settings() -> RuntimeSettings:
    cfg = settings()
    return RuntimeSettings(
        demo_mode=cfg.demo_mode,
        models={api: getattr(cfg, field) for api, field in RUNTIME_MODEL_FIELDS.items()},
        gemini_key_configured=bool(cfg.gemini_api_key),
        anthropic_key_configured=bool(cfg.anthropic_api_key),
        editable=cfg.runtime_settings_enabled,
    )


def _require_runtime_settings_enabled() -> None:
    # A dependency, so it runs before the body is validated: a disabled
    # endpoint answers 403 whatever was posted.
    if not settings().runtime_settings_enabled:
        raise HTTPException(status_code=403, detail=(
            "Runtime settings are read-only here; set RUNTIME_SETTINGS_ENABLED=1 "
            "to allow POST /settings/runtime."))


@app.get("/settings/runtime", response_model=RuntimeSettings)
def get_runtime_settings() -> RuntimeSettings:
    return _runtime_settings()


@app.post("/settings/runtime", response_model=RuntimeSettings,
          dependencies=[Depends(_require_runtime_settings_enabled)])
def update_runtime_settings(req: RuntimeSettingsUpdate) -> RuntimeSettings:
    """Switch demo mode and/or model specs for every later request."""
    try:
        set_runtime_overrides(
            demo_mode=req.demo_mode,
            models=req.models.model_dump() if req.models else None)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return _runtime_settings()


@app.get("/metrics")
def metrics():
    """Prometheus exposition. Free, no LLM calls, safe to scrape often.

    /health says the process is up. This says whether it is doing its job:
    LLM spend, plan outcomes by gate code, and origin coverage — which is
    the series that matters most, because if coverage collapses the origin
    filter silently stops protecting anyone."""
    from fastapi.responses import Response
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    from . import metrics as m

    m.refresh_db_gauges()
    return Response(generate_latest(m.REGISTRY), media_type=CONTENT_TYPE_LATEST)


class Store(BaseModel):
    id: int
    name: str
    lat: float
    lon: float
    address: str = ""


@app.get("/stores", response_model=list[Store])
def list_stores() -> list[Store]:
    """Every store the catalog prices products at: what a client checks store names against
    (an answer naming a store that is not here made it up)."""
    from sqlalchemy import select
    from sqlalchemy.orm import Session

    with Session(db.engine()) as s:
        rows = s.execute(select(db.StoreRow).order_by(db.StoreRow.id)).scalars().all()
        return [Store(id=r.id, name=r.name, lat=r.lat, lon=r.lon, address=r.address or "")
                for r in rows]


@app.get("/recipes", response_model=list[Recipe])
def list_recipes() -> list[Recipe]:
    return db.load_all_recipes()


@app.get("/recipes/{slug}", response_model=Recipe)
def get_recipe(slug: str) -> Recipe:
    try:
        return db.load_recipe(slug)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/recipes/{slug}/doc", response_model=RecipeDoc)
def get_recipe_doc(slug: str) -> RecipeDoc:
    """A library recipe as a RecipeDoc with its demo house amounts (synthetic, labelled on
    every line), to show line by line. Plan it by slug, POST /plan/{slug} (PLAN.md C4): that
    path gives the selector each line's category, while POST /plan/spec first checks by name
    alone that each line is stocked, as /plan/nl does, and a library name such as "Peanut
    Butter and Jelly Jam" finds nothing that way (a 409). Before pantry-db migration 0007
    the lines have no amounts and `warnings` says so."""
    from .recipe_doc import library_doc

    try:
        recipe = db.load_recipe(slug)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    amounts = db.load_line_amounts([slug])
    return library_doc(recipe, None if amounts is None else amounts.get(slug, {}))


class ParseLinesRequest(BaseModel):
    """Ingredient lines to read, with no LLM: a paste, or a page's ingredient list the hub
    extracted (origin "page", whose amounts the source stated)."""

    title: str | None = Field(default=None, max_length=200)
    yield_text: str | None = Field(default=None, max_length=200)
    lines: list[Annotated[str, Field(max_length=300)]] = Field(max_length=MAX_DOC_LINES)
    origin: Literal["paste", "page"] = "paste"


class ParsedLineOut(BaseModel):
    line_no: int
    text: str                   # the line as given, trimmed
    name: str                   # what to buy, purchase form included ("ground beef")
    quantity: float | None      # None: the line states no amount
    unit: str                   # "" when there is no unit
    note: str                   # prep and the text after the first comma
    amount_basis: Literal["parsed_from_your_paste", "stated_by_source"]


class ParsedLines(BaseModel):
    servings: int | None        # None: nothing says how many it serves (never a guessed 1)
    servings_stated: bool
    lines: list[ParsedLineOut]
    warnings: list[str]


@app.post("/recipes/parse-lines", response_model=ParsedLines,
          dependencies=[Depends(limits.rate_limit("/recipes/parse-lines"))])
def parse_lines(req: ParseLinesRequest) -> ParsedLines:
    """Read ingredient lines into the RecipeLine fields a shopper reviews before planning
    (nlsearch.lineparse, the demo parser's line reading). Pure: no URL, no LLM, no database.
    Blank lines, and bullets with nothing after them, are dropped and renumbered; a line
    with no amount is kept and named in `warnings`, as is an unstated servings count.

    The result stays inside RecipeDoc's bounds, so a client can post it to /plan/spec as it
    stands, warnings included: a name longer than a RecipeLine holds is cut, and a servings
    count over MAX_SERVINGS or an amount over MAX_LINE_QUANTITY comes back as not stated.
    Each says so in `warnings`, which has one entry per kind of problem, naming its lines."""
    from .nlsearch import lineparse

    basis = "stated_by_source" if req.origin == "page" else "parsed_from_your_paste"
    texts = [t.strip() for t in req.lines]
    # A bullet on its own has nothing to buy: it would come back with an empty name,
    # which no RecipeLine accepts.
    kept = [t for t in texts if lineparse.without_bullet(t)]
    out: list[ParsedLineOut] = []
    # The lines with each problem, gathered into one warning per kind below. One warning per
    # line would let a long paste with no amounts outgrow the MAX_DOC_WARNINGS a RecipeDoc
    # holds, and the doc a client builds from this output would be a 422.
    cut: list[str] = []
    too_big: list[str] = []
    unstated: list[str] = []
    for t in kept:
        p = lineparse.parse_line(t)
        n, name = len(out) + 1, p.doc_name
        if len(name) > MAX_LINE_NAME:
            # Only a line whose text before its first comma runs past this gets here, which
            # is a sentence rather than an ingredient; the shopper sees the cut name and the
            # full text side by side.
            name = name[:MAX_LINE_NAME].rstrip()
            cut.append(str(n))
        qty, unit = p.quantity, p.unit or ""
        if qty is not None and qty > MAX_LINE_QUANTITY:
            # As with a catering yield, clamping would plan an amount nobody wrote, so the
            # line has no amount and the shopper says how much they need.
            too_big.append(f"{n} ({name})")
            qty, unit = None, ""
        out.append(ParsedLineOut(line_no=n, text=t, name=name,
                                 quantity=qty, unit=unit, note=p.prep or "",
                                 amount_basis=basis))
        if p.quantity is None:
            unstated.append(f"{n} ({name})")
    warnings: list[str] = []
    stated = (lineparse.servings_from_yield(req.yield_text or "")
              or lineparse.servings_from_yield(req.title or ""))
    # Over the cap is a catering yield. Clamping it would plan a number nobody wrote, so it
    # is not stated, and the shopper says how many they are cooking for.
    servings = stated if stated is not None and stated <= MAX_SERVINGS else None
    if stated is not None and servings is None:
        warnings.append(f"servings stated as {stated}, more than the {MAX_SERVINGS} a plan "
                        "takes: say how many you are cooking for")
    elif servings is None:
        warnings.append("servings not stated")
    if blank := len(texts) - len(kept):
        warnings.append(f"{blank} blank line(s) dropped")
    if cut:
        warnings.append(f"line {cut[0]}'s name was cut to {MAX_LINE_NAME} characters"
                        if len(cut) == 1 else
                        f"the names of lines {', '.join(cut)} were cut to {MAX_LINE_NAME} "
                        "characters")
    if too_big:
        more = f"more than the {MAX_LINE_QUANTITY:,} a plan takes: say how much you need"
        warnings.append(f"line {too_big[0]} states {more}" if len(too_big) == 1
                        else f"lines {', '.join(too_big)} state {more}")
    if unstated:
        warnings.append(f"line {unstated[0]} states no amount" if len(unstated) == 1
                        else f"lines {', '.join(unstated)} state no amount")
    return ParsedLines(servings=servings, servings_stated=servings is not None,
                       lines=out, warnings=warnings)


@app.get("/products", response_model=list[Product])
def list_products() -> list[Product]:
    return db.load_all_products()


class NLPlanRequest(BaseModel):
    """The FULL pasted recipe text (+ optional inline shopping notes).
    lat/lon: optional shopping location for the distance constraint;
    defaults to the configured reference point."""

    recipe_text: str = Field(max_length=8000)   # public endpoint: bound the paste
    lat: float | None = None
    lon: float | None = None
    # Provenance: exclude removes candidates positively evidenced as coming
    # from these countries; preference is soft guidance to the selector.
    exclude_origin: list[str] = Field(default_factory=list, max_length=50)
    preference: list[str] = Field(default_factory=list, max_length=50)
    # Same knobs as the MCP plan_from_text tool: max_km overrides any
    # distance stated in the text; allow_partial plans what is stocked and
    # in range and lists the rest (not_stocked / out_of_range) instead of a
    # 409. Defaults keep the original behaviour.
    max_km: float | None = Field(default=None, ge=0.5, le=100)
    allow_partial: bool = False


@app.post("/plan/nl", response_model=ShoppingPlan,
          dependencies=[Depends(limits.require_llm_budget),
                        Depends(limits.rate_limit("/plan/nl"))])
def plan_nl(req: NLPlanRequest) -> ShoppingPlan:
    """NL2SQL path: parse a pasted recipe, execute the staged query plan
    (existence → options → brand stats → lookups), route, select. Returns
    the plan + interpretation + the full per-step SQL trace. A gate abort
    returns 409 with the alert and the trace up to the failed step."""
    from . import metrics as m
    from .nlsearch import PlanAborted, UnparseableRecipe

    _check_countries(req.exclude_origin, req.preference)
    try:
        plan = flow.run_nl(req.recipe_text, lat=req.lat, lon=req.lon,
                           exclude=req.exclude_origin,
                           preference=req.preference,
                           max_km=req.max_km, allow_partial=req.allow_partial)
        m.record_plan("nl", "ok")
        m.record_coverage(plan.origin_coverage)
        return plan
    except UnparseableRecipe:
        m.record_plan("nl", "unparseable")
        raise HTTPException(status_code=422, detail=(
            "Couldn't find an ingredient list in that text. Paste a recipe "
            "with its ingredients (quantities optional), e.g.:\n"
            "Spaghetti Bolognese (serves 4)\n"
            "- 400g spaghetti\n- 500g ground beef\n- 1 can crushed tomatoes\n"
            "Notes: under $30, no dairy"))
    except PlanAborted as e:
        code = e.execution.aborted.code.value if e.execution.aborted else "unknown"
        m.record_plan("nl", "gated", gate=code)
        raise HTTPException(status_code=409, detail=e.execution.model_dump(mode="json"))


class SpecPlanRequest(BaseModel):
    """A recipe the shopper reviewed (RecipeDoc), planned exactly as given: the same
    location, origin and partial-plan knobs as /plan/nl, and no shopping notes (there is
    no text to read them from)."""

    doc: RecipeDoc
    lat: float | None = Field(default=None, ge=-90, le=90)
    lon: float | None = Field(default=None, ge=-180, le=180)
    max_km: float | None = Field(default=None, ge=0.5, le=100)
    exclude_origin: list[str] = Field(default_factory=list, max_length=50)
    preference: list[str] = Field(default_factory=list, max_length=50)
    allow_partial: bool = False


@app.post("/plan/spec", response_model=ShoppingPlan,
          dependencies=[Depends(limits.require_llm_budget),
                        Depends(limits.rate_limit("/plan/spec"))])
def plan_spec(req: SpecPlanRequest) -> ShoppingPlan:
    """Plan reviewed lines with no parse: what the shopper reviewed is what gets planned
    (names, quantities and units byte for byte on `basis.lines`). The selector still picks
    the products. 422 `unconfirmed_lines` names any line not confirmed yet; an unknown
    country is a 422 and a gate abort a 409, as on /plan/nl."""
    from . import metrics as m
    from .nlsearch import PlanAborted, UnparseableRecipe
    from .recipe_doc import UnconfirmedLines, to_recipe_text, to_spec

    _check_countries(req.exclude_origin, req.preference)
    try:
        spec = to_spec(req.doc)
    except UnconfirmedLines as e:
        m.record_plan("spec", "unconfirmed")
        raise HTTPException(status_code=422, detail={
            "error": "unconfirmed_lines", "line_nos": e.line_nos, "detail": str(e)}) from e
    try:
        plan = flow.run_spec(spec, lat=req.lat, lon=req.lon, exclude=req.exclude_origin,
                             preference=req.preference, max_km=req.max_km,
                             allow_partial=req.allow_partial,
                             display_text=to_recipe_text(req.doc))
    except UnparseableRecipe as e:
        m.record_plan("spec", "unparseable")
        raise HTTPException(status_code=422, detail={
            "error": "no_lines", "detail": "The recipe has no ingredient lines to plan."}) from e
    except PlanAborted as e:
        code = e.execution.aborted.code.value if e.execution.aborted else "unknown"
        m.record_plan("spec", "gated", gate=code)
        raise HTTPException(status_code=409,
                            detail=e.execution.model_dump(mode="json")) from e
    m.record_plan("spec", "ok")
    m.record_coverage(plan.origin_coverage)
    return plan


# ─── Plan follow-ups: alternatives and re-pricing (no LLM) ───
# REST twins of the MCP tools for clients without the hub: PlannerView, the
# meal plan's per-line Options and the Hugging Face demo. The client holds the
# plan's basis and sends it back; every pin is checked on the server
# (alternatives.validate_pins), so the client never supplies trusted state.

class AlternativesRequest(BaseModel):
    """`basis` is a plan's `basis` as returned; `line_no` one of its planned lines."""

    basis: PlanBasis
    line_no: int = Field(ge=1, le=MAX_DOC_LINES)
    limit: int = Field(default=12, ge=1, le=25)


class RepriceRequest(BaseModel):
    """`pins`: the shopper's product per line, at most 40; a pin equal to the plan's own
    pick is dropped (that undoes a swap). The basis's own pins are merged first."""

    basis: PlanBasis
    pins: list[Pin] = Field(default_factory=list, max_length=40)


def _alternatives_error(e: Exception) -> HTTPException:
    """One mapping for both routes: a stale basis is a 409 (plan again), anything else the
    client sent wrong a 422, each with a code and the reason in words."""
    from . import alternatives as alts

    if isinstance(e, alts.StaleBasisError):
        return HTTPException(status_code=409, detail={"error": "stale_basis", "detail": str(e)})
    code = ("line_not_planned" if isinstance(e, alts.LineNotPlannedError)
            else "invalid_pin" if isinstance(e, alts.PinError) else "invalid_basis")
    return HTTPException(status_code=422, detail={"error": code, "detail": str(e)})


@app.post("/plan/alternatives", response_model=AlternativeRanking,
          dependencies=[Depends(limits.rate_limit("/plan/alternatives"))])
def plan_alternatives(req: AlternativesRequest) -> AlternativeRanking:
    """The other products that could fill one planned line, in the planner's order, with
    the plan's own pick flagged `current` (alternatives.rank_alternatives). No LLM call.
    422 `line_not_planned`, `invalid_pin` or `invalid_basis`; 409 `stale_basis` when the
    catalog changed under the plan."""
    from . import alternatives as alts

    _check_countries(req.basis.exclude_origin, req.basis.preference)
    try:
        return alts.rank_alternatives(req.basis, req.line_no, req.limit)
    except (alts.BasisError, alts.PinError) as e:
        raise _alternatives_error(e) from e


@app.post("/plan/reprice", response_model=ShoppingPlan,
          dependencies=[Depends(limits.rate_limit("/plan/reprice"))])
def plan_reprice(req: RepriceRequest) -> ShoppingPlan:
    """The plan priced again with the shopper's pins (flow.reprice): the same ShoppingPlan
    shape as /plan/nl, so a client redraws it in place. With no pins it is the plan's own;
    `basis.pins` on the result are the merged pins. No LLM call. Errors as
    /plan/alternatives."""
    from . import alternatives as alts

    _check_countries(req.basis.exclude_origin, req.basis.preference)
    try:
        return flow.reprice(req.basis, req.pins)
    except (alts.BasisError, alts.PinError) as e:
        raise _alternatives_error(e) from e


class WeekPlanRequest(BaseModel):
    """Plan `days` dinners from the recipe library under an optional budget.
    Deterministic menu selection (marginal-cost greedy over one batched
    retrieval); the LLM only does the per-day product mapping."""

    days: int = Field(5, ge=1, le=14)
    max_total_budget: float | None = None
    exclude_tags: list[str] = []
    lat: float | None = None
    lon: float | None = None
    max_distance_km: float | None = None
    exclude_origin: list[str] = Field(default_factory=list, max_length=50)
    preference: list[str] = Field(default_factory=list, max_length=50)


@app.post("/plan/week", response_model=WeekPlan,
          dependencies=[Depends(limits.require_llm_budget)])
def plan_week(req: WeekPlanRequest) -> WeekPlan:
    """5A: weekly menu optimizer. Rewards ingredient overlap exactly (a
    shared product costs $0 marginal), gates on the cheapest-basket floor
    before any LLM spend, and prices the merged basket's store split."""
    from . import weekplan
    from .nlsearch import PlanAborted

    _check_countries(req.exclude_origin, req.preference)
    try:
        return weekplan.plan_week(
            days=req.days, max_total_budget=req.max_total_budget,
            exclude_tags=req.exclude_tags, lat=req.lat, lon=req.lon,
            max_distance_km=req.max_distance_km,
            exclude_origin=req.exclude_origin, preference=req.preference)
    except PlanAborted as e:
        raise HTTPException(status_code=409, detail=e.execution.model_dump(mode="json"))


@app.post("/plan/{slug}", response_model=ShoppingPlan,
          dependencies=[Depends(limits.require_llm_budget)])
def plan_recipe(slug: str,
                exclude_origin: Annotated[list[str] | None, Query()] = None,
                preference: Annotated[list[str] | None, Query()] = None,
                lat: Annotated[float | None, Query(ge=-90, le=90)] = None,
                lon: Annotated[float | None, Query(ge=-180, le=180)] = None,
                max_km: Annotated[float | None, Query(ge=0.5, le=100)] = None,
                allow_partial: bool = False) -> ShoppingPlan:
    """Run the pipeline for one recipe. Returns the shopping plan.

    `exclude_origin` removes candidates positively evidenced as coming from
    those countries — never candidates that merely lack evidence. The
    returned plan carries per-line provenance and a spend-weighted coverage
    figure saying how much of the basket was actually checked. An ingredient
    the exclusion leaves with no candidate is a 409 naming it and what to buy
    instead; with `allow_partial` the rest is planned and it goes to
    `out_of_range` with those options.

    Any of `lat`/`lon`/`max_km` makes the plan store-aware: each line is
    priced at its cheapest store within `max_km` of the point (the server's
    default point when lat/lon are omitted; any distance when max_km is),
    `trip_options` splits the basket across stores, and a chosen product no
    store in range sells goes to `out_of_range` naming the nearest offer."""
    from . import metrics as m
    from .nlsearch import PlanAborted

    _check_countries(exclude_origin, preference)
    try:
        plan = flow.run(slug, exclude=exclude_origin, preference=preference,
                        lat=lat, lon=lon, max_km=max_km, allow_partial=allow_partial)
    except ValueError as e:
        m.record_plan("recipe", "not_found")
        raise HTTPException(status_code=404, detail=str(e)) from e
    except PlanAborted as e:
        # The origin gate is a 409 like every other gate — it was escaping
        # this route as a 500 with no alert payload.
        code = e.execution.aborted.code.value if e.execution.aborted else "unknown"
        m.record_plan("recipe", "gated", gate=code)
        raise HTTPException(status_code=409,
                            detail=e.execution.model_dump(mode="json")) from e
    m.record_plan("recipe", "ok")
    m.record_coverage(plan.origin_coverage)
    return plan


# ─── Meal plan ───────────────────────────────────────────────
# The plan lives in the browser (a MealPlanDraft); these endpoints are stateless.

class MealPlanResolveRequest(BaseModel):
    """Recipes to resolve to products, each once: a library slug, a demo starter or a
    RecipeDoc. The location and origin knobs are those of /plan/spec."""

    recipes: list[RecipeRef] = Field(min_length=1, max_length=12)
    lat: float | None = Field(default=None, ge=-90, le=90)
    lon: float | None = Field(default=None, ge=-180, le=180)
    max_km: float | None = Field(default=None, ge=0.5, le=100)
    exclude_origin: list[str] = Field(default_factory=list, max_length=50)
    preference: list[str] = Field(default_factory=list, max_length=50)


class MealPlanResolveResponse(BaseModel):
    resolved: list[ResolvedRecipe]
    llm_cost_usd: float          # this call's LLM spend; a cached recipe adds nothing
    latency_ms: int


@app.post("/mealplan/resolve", response_model=MealPlanResolveResponse,
          dependencies=[Depends(limits.require_llm_budget),
                        Depends(limits.rate_limit("/mealplan/resolve"))])
def mealplan_resolve(req: MealPlanResolveRequest) -> MealPlanResolveResponse:
    """Which products each recipe buys: library recipes through the classic pipeline, every
    RecipeDoc through /plan/spec's path (reviewed lines planned as given). A recipe that
    fails comes back with a status of its own (needs_servings, unconfirmed_lines, not_found,
    unparseable, aborted, llm_error); the request does not fail. Slow when live, instant in
    demo mode; results are cached per recipe and knobs."""
    import time

    from .mealplan.resolve import resolve_one

    _check_countries(req.exclude_origin, req.preference)
    t0 = time.perf_counter()
    out = [resolve_one(ref, lat=req.lat, lon=req.lon, max_km=req.max_km,
                       exclude=req.exclude_origin, preference=req.preference)
           for ref in req.recipes]
    return MealPlanResolveResponse(
        resolved=out, llm_cost_usd=round(sum(r.llm_cost_usd for r in out if not r.cached), 6),
        latency_ms=int((time.perf_counter() - t0) * 1000))


async def _draft_size(request: Request) -> None:
    """A draft is at most 256 KB: 413 above it."""
    from .mealplan.models import MAX_DRAFT_BYTES

    declared = request.headers.get("content-length", "")
    size = int(declared) if declared.isdigit() else len(await request.body())
    if size > MAX_DRAFT_BYTES:
        raise HTTPException(status_code=413, detail={
            "error": "too_large", "detail": f"A meal plan is at most {MAX_DRAFT_BYTES} bytes."})


def _meal_plan_error(e) -> HTTPException:
    return HTTPException(status_code=422, detail={"error": e.code, "detail": e.detail,
                                                  **e.extra})


@app.post("/mealplan/schedule", response_model=MealSchedule,
          dependencies=[Depends(_draft_size),
                        Depends(limits.rate_limit("/mealplan/schedule"))])
def mealplan_schedule(draft: MealPlanDraft) -> MealSchedule:
    """Where the meals sit, what each needs, the trips for both strategies (fresh and
    fewest_trips), warnings with remedies, and each trip's shopping list. Pure: no LLM, no
    writes, facts re-read by id; the same draft gives byte-identical JSON. 422 with
    {error: slot_capacity | unknown_recipe_key | stale_product | invalid_dates | pin_invalid
    | version}."""
    from .mealplan.models import MealPlanError
    from .mealplan.schedule import compute

    try:
        return compute(draft)
    except MealPlanError as e:
        raise _meal_plan_error(e) from e


class MyRecipeName(BaseModel):
    """One of the shopper's own recipes, for Quick add to match against."""

    key: str = Field(min_length=1, max_length=100)
    title: str = Field(min_length=1, max_length=200)
    slot: Literal["breakfast", "lunch", "dinner", "snack"] | None = None
    aliases: list[Annotated[str, Field(max_length=200)]] = Field(default_factory=list,
                                                                 max_length=10)


class SelectionParseRequest(BaseModel):
    text: str = Field(min_length=1, max_length=8000)
    recipes: list[MyRecipeName] = Field(default_factory=list, max_length=50)
    household_servings: int = Field(default=2, ge=1, le=20)


@app.post("/mealplan/selection/parse",
          dependencies=[Depends(limits.rate_limit("/mealplan/selection/parse"))])
def mealplan_selection_parse(req: SelectionParseRequest) -> dict:
    """Quick add: "3 Pepperoni Pizza + 2 Chicken Fried Rice ... in 2 weeks" read into counted
    selections matched against the library, the demo starters and the shopper's recipes
    (exact, plural, alias, fuzzy), with no LLM. Only exact and plural matches may be
    accepted without asking: alias and fuzzy come back with needs_confirmation true."""
    from .mealplan import selection
    from .mealplan.resolve import starters_file

    cands = selection.candidates(
        [(r.slug, r.name) for r in db.load_all_recipes()], starters_file()["starters"],
        [r.model_dump() for r in req.recipes])
    return selection.parse(req.text, cands, req.household_servings)


@app.get("/mealplan/starters")
def mealplan_starters() -> list[dict]:
    """The demo starter recipes (labelled "demo recipe", demo house amounts), each with its
    RecipeDoc, default slot and the aliases Quick add matches."""
    from .mealplan.resolve import starters

    return starters()


@app.get("/shelf-life")
def shelf_life(product_id: Annotated[list[int] | None, Query()] = None) -> dict:
    """Cited storage and thaw times per product (all products, or those named). A product
    with no cited row has mapped false, status unknown, its reason, and no number.
    synthetic_product marks the demo products (166-169)."""
    from .mealplan import shelf

    names = {p.id: p.name for p in db.load_all_products()}
    wanted = sorted(set(product_id)) if product_id else sorted(names)
    out = []
    for pid in wanted:
        if pid not in names:
            continue
        ps = shelf.for_product(pid)
        rules = {}
        if ps.mapped:
            for kind, ids in (("fridge", ps.fridge), ("freezer", ps.freezer),
                              ("after_thaw_fridge", ps.after_thaw), ("thaw", ps.thaw)):
                if ids:
                    rules[kind] = [shelf.rule(r) for r in ids]
        out.append({"product_id": pid, "name": names[pid], "mapped": ps.mapped,
                    "status": "cited" if ps.mapped else "unknown",
                    "storage_class": ps.storage_class, "bought_state": ps.bought_state,
                    "rules": rules, "note": ps.note, "reason": ps.reason,
                    "synthetic_product": ps.synthetic_product})
    data = shelf.load()
    return {"sources": shelf.sources(), "rules_of_use": data["rules_of_use"],
            "products": out}


# ─── Provenance ──────────────────────────────────────────────

class RankRequest(BaseModel):
    """Rank the catalog against a country preference the CALLER supplies.

    preference: ordered, most-preferred first (e.g. ["Canada", "Mexico"]).
    exclude:    countries to filter out. A product is only ever excluded on
                positive evidence — never for lacking any.
    """

    preference: list[str] = Field(default_factory=list, max_length=50)
    exclude: list[str] = Field(default_factory=list, max_length=50)
    search: str | None = Field(default=None, max_length=200)
    product_ids: list[int] | None = None


@app.get("/origins", response_model=list[ProductOrigin])
def list_origins(search: str | None = None) -> list[ProductOrigin]:
    """Resolved provenance per product, from ingested evidence only."""
    from . import origins

    products = db.load_all_products()
    if search:
        needle = search.lower()
        products = [p for p in products
                    if needle in f"{p.name} {p.brand} {p.category}".lower()]
    resolved = origins.resolve_all([p.id for p in products])
    return [resolved[p.id] for p in products if p.id in resolved]


@app.post("/origins/rank", response_model=OriginRanking)
def rank_by_origin(req: RankRequest) -> OriginRanking:
    """Rank products by provenance against the caller's preference order.

    Returns four things, kept apart on purpose: ranked, excluded,
    unranked (no evidence / conflicting / lookup failed) and the counts.
    Unverified products are never folded into the ranking — with coverage
    as thin as it is, that would present silence as a verdict.
    """
    from . import origins

    _check_countries(req.preference, req.exclude)
    products = db.load_all_products()
    if req.product_ids is not None:
        wanted = set(req.product_ids)
        products = [p for p in products if p.id in wanted]
    elif req.search:
        needle = req.search.lower()
        products = [p for p in products
                    if needle in f"{p.name} {p.brand} {p.category}".lower()]
    return origins.rank_products(
        products, preference=req.preference, exclude=req.exclude)


@app.get("/origins/triage")
def origin_triage() -> list[dict]:
    """Products worth photographing next. Hints, never origins."""
    from . import origins

    return origins.triage_candidates(db.load_all_products())


# MCP Streamable HTTP — mounted last so the REST routes above keep
# priority; the sub-app serves exactly /mcp (public:
# https://<host>/pantry/api/mcp). Mounting also instantiates the
# session manager that _lifespan drives.
if MCP_HTTP_ENABLED:
    from .mcp_server import http_app
    from .mcp_server import server as _mcp_server

    _MCP_APP = http_app()
    _MCP_SESSION_MANAGER = _mcp_server.session_manager   # the one _MCP_APP serves
    app.mount("/", _MCP_APP)

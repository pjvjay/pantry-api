"""Public endpoint limits: a token bucket per client and a daily LLM cost ceiling.

The REST API is public on every deployment, and on AKS it runs a live LLM
(ANTHROPIC_API_KEY mounted, no DEMO_MODE), so one client in a loop could
spend the month's budget in an afternoon. Two limits, both in-process and
per replica, which is enough for one or two replicas and needs no store:

- A token bucket per client IP and endpoint. Over the limit is a 429 with
  Retry-After. The client IP is the TCP peer, or, only when
  TRUSTED_PROXY_HOPS says how many proxies in front of us append to
  X-Forwarded-For, the entry that many hops from the right: anything further
  left was written by the client and proves nothing. Behind a proxy with
  TRUSTED_PROXY_HOPS unset, the peer is the proxy and every client shares its
  buckets (uvicorn rewrites the peer only for a proxy on loopback), so the
  first request that carries X-Forwarded-For logs a warning saying so.
- A daily ceiling on estimated LLM spend (LLM_DAILY_COST_CAP_USD; unset means
  no ceiling). Every LLM call adds its estimate (llm.forced_tool_call calls
  record_spend), and the day resets at UTC midnight. The estimate is $0 for
  Gemini (the free tier) and for any model config.COST_PER_MTOK does not
  list; the first call to an unlisted model logs a warning. Above the ceiling the
  endpoints that call an LLM answer 503, and so do the MCP plan tools; the
  endpoints that call none (parse-lines, and the meal-plan schedule when it
  lands) keep working, and so does demo mode, which makes no LLM call.

Both refusals are LimitError, which the API renders as {error, detail} with
`detail` a sentence, the body an LLM failure already has (api._llm_error).
Clients written before the limits show a string `detail` as it is, and an
object one as "[object Object]".

Endpoints join the bucket table as they land (PLAN.md 3.12 lists the ones to
come). The MCP endpoint's own bearer tokens are unchanged, and buckets apply
to REST only.
"""
from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime

from fastapi import Request

from .config import settings

PAUSED = "Live planning is paused for today; the demo planner still works."

log = logging.getLogger(__name__)


class LimitError(Exception):
    """A request a limit refused: `status` 429 or 503, `error` a code for programs, `detail`
    the sentence a person reads, and the response headers (Retry-After)."""

    def __init__(self, status: int, error: str, detail: str,
                 headers: dict[str, str] | None = None) -> None:
        super().__init__(detail)
        self.status, self.error, self.detail = status, error, detail
        self.headers = headers or {}


@dataclass(frozen=True)
class Limit:
    per_minute: float
    burst: int


# Endpoint -> its bucket. Shared names share nothing: each endpoint has its own
# bucket per client.
LIMITS: dict[str, Limit] = {
    "/plan/nl": Limit(per_minute=10, burst=10),
    "/plan/spec": Limit(per_minute=10, burst=10),
    "/recipes/parse-lines": Limit(per_minute=60, burst=60),
    # No LLM behind either, but each runs a handful of queries and up to 40 trip
    # optimisations: generous for a shopper clicking through options, not unbounded.
    "/plan/alternatives": Limit(per_minute=60, burst=60),
    "/plan/reprice": Limit(per_minute=60, burst=60),
    # The meal plan. resolve runs the planner once per recipe (up to 12, each an LLM call
    # when live), so it is the tightest.
    "/mealplan/resolve": Limit(per_minute=6, burst=3),
}

# Buckets idle this long are full again and can be forgotten.
_IDLE_S = 600
_MAX_BUCKETS = 10_000

_lock = threading.Lock()
_buckets: dict[tuple[str, str], tuple[float, float]] = {}   # (endpoint, ip) -> (tokens, at)
_spend = {"day": "", "usd": 0.0}
# Whether this process has logged the untrusted-proxy warning: once is enough to be seen.
_proxy_warned = False
# Models whose calls counted $0 toward a ceiling, each logged once per process.
_unpriced_warned: set[str] = set()

# The clocks, as module attributes so tests can move time.
monotonic = time.monotonic


def utc_day() -> str:
    return datetime.now(UTC).date().isoformat()


def reset() -> None:
    """Forget every bucket, today's spend and the warnings already logged (tests)."""
    global _proxy_warned
    with _lock:
        _buckets.clear()
        _spend.update(day="", usd=0.0)
        _proxy_warned = False
        _unpriced_warned.clear()


# ─── Client identity ─────────────────────────────────────────

def client_ip(request: Request) -> str:
    """The TCP peer, or with TRUSTED_PROXY_HOPS=n the n-th X-Forwarded-For entry from
    the right (each trusted proxy appends the address it saw). A header shorter than n
    hops has been through fewer proxies than configured, so its leftmost entry is the
    one a trusted proxy wrote."""
    peer = request.client.host if request.client else "unknown"
    hops = settings().trusted_proxy_hops
    if hops <= 0:
        if "x-forwarded-for" in request.headers:
            _warn_untrusted_proxy(peer)
        return peer
    chain = [h.strip() for h in request.headers.get("x-forwarded-for", "").split(",")
             if h.strip()]
    if not chain:
        return peer
    return chain[-hops] if len(chain) >= hops else chain[0]


def _warn_untrusted_proxy(peer: str) -> None:
    """Say once per process that X-Forwarded-For arrives while no proxy is trusted. If the
    peer is a proxy (ingress-nginx, a hosting platform's front end), every client behind it
    shares one bucket per endpoint, which looks like a limit far lower than configured. The
    header is still not read: only TRUSTED_PROXY_HOPS can say which entry a proxy wrote."""
    global _proxy_warned
    with _lock:
        if _proxy_warned:
            return
        _proxy_warned = True
    log.warning("requests from %s carry X-Forwarded-For but TRUSTED_PROXY_HOPS is 0, so the "
                "rate limits key on %s: if that is a proxy, every client behind it shares one "
                "bucket per endpoint. Set TRUSTED_PROXY_HOPS to the number of proxies in front.",
                peer, peer)


# ─── Token bucket ────────────────────────────────────────────

def take(endpoint: str, ip: str) -> float | None:
    """Spend one token from (endpoint, ip). None when allowed; otherwise the seconds
    until the next token."""
    limit = LIMITS[endpoint]
    rate = limit.per_minute / 60.0
    now = monotonic()
    with _lock:
        if len(_buckets) > _MAX_BUCKETS:
            for key in [k for k, (_t, at) in _buckets.items() if now - at > _IDLE_S]:
                del _buckets[key]
        tokens, at = _buckets.get((endpoint, ip), (float(limit.burst), now))
        tokens = min(float(limit.burst), tokens + (now - at) * rate)
        if tokens >= 1.0:
            _buckets[(endpoint, ip)] = (tokens - 1.0, now)
            return None
        _buckets[(endpoint, ip)] = (tokens, now)
        return (1.0 - tokens) / rate


def rate_limit(endpoint: str):
    """A FastAPI dependency: 429 with Retry-After when this client is over the limit."""
    if endpoint not in LIMITS:
        raise KeyError(f"no limit configured for {endpoint}")

    def check(request: Request) -> None:
        wait = take(endpoint, client_ip(request))
        if wait is not None:
            seconds = max(1, math.ceil(wait))
            raise LimitError(429, "rate_limited",
                                f"Too many requests to {endpoint}; retry in {seconds} s.",
                                headers={"Retry-After": str(seconds)})

    return check


# ─── Daily LLM cost ceiling ──────────────────────────────────

def record_spend(usd: float) -> None:
    """Add one LLM call's estimated cost to today's total."""
    if usd <= 0:
        return
    day = utc_day()
    with _lock:
        if _spend["day"] != day:
            _spend.update(day=day, usd=0.0)
        _spend["usd"] += usd


def warn_unpriced(model: str) -> None:
    """Say once per model that its calls count $0 toward the ceiling. Only the models in
    config.COST_PER_MTOK have a price, and validate_model_spec accepts any "claude-" name,
    so a deployment that sets another model spends with the ceiling seeing nothing, and no
    response says so. Without a ceiling there is nothing to undercount."""
    if settings().llm_daily_cost_cap_usd is None:
        return
    with _lock:
        if model in _unpriced_warned:
            return
        _unpriced_warned.add(model)
    log.warning("LLM_DAILY_COST_CAP_USD is set, but %s has no price in config.COST_PER_MTOK, "
                "so its calls count $0 toward the ceiling. Add its rates there.", model)


def budget() -> dict:
    """{spent, cap} for today (UTC); cap None means no ceiling."""
    day = utc_day()
    with _lock:
        spent = _spend["usd"] if _spend["day"] == day else 0.0
    return {"spent": round(spent, 6), "cap": settings().llm_daily_cost_cap_usd}


def llm_paused() -> bool:
    """True when today's spend has reached the ceiling and planning would call an LLM.
    Demo mode makes no LLM call, so it is never paused."""
    cfg = settings()
    if cfg.demo_mode or cfg.llm_daily_cost_cap_usd is None:
        return False
    return budget()["spent"] >= cfg.llm_daily_cost_cap_usd


def require_llm_budget() -> None:
    """A FastAPI dependency for endpoints that call an LLM: 503 above the daily ceiling."""
    if llm_paused():
        raise LimitError(503, "llm_budget_exhausted", PAUSED)

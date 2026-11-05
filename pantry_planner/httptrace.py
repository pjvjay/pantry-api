"""Where an outbound HTTP request spends its time, phase by phase.

httpx hands a request's connection events to its ``trace`` extension: httpcore's
``connection.connect_tcp``, ``connection.start_tls``, ``http11.send_request_headers``,
``http11.receive_response_headers`` ... each ``.started`` and ``.complete``. PhaseClock stamps
them and ``attempt_record`` turns one request into its phases:

  dns       resolving the host (timed by a lookup of our own just before the request; the
            connect that follows then hits the resolver's cache)
  connect   the TCP handshake (absent when a kept-alive connection is reused)
  tls       the TLS handshake
  upload    sending the request headers and body
  wait      from the last byte sent to the response headers: the provider's processing and
            any queue in front of it, plus one network round trip
  download  reading the response body
  other     whatever the phases do not cover (connection pool, encoding, the client itself)

Google's front end reports its own share of ``wait`` in the ``server-timing`` header
(``gfet4t7; dur=1234``, milliseconds): a ``wait`` far above it was spent on the network or
before the request reached Google, one close to it inside Google.
"""
from __future__ import annotations

import logging
import re
import socket
import time
from typing import Any

import httpx

log = logging.getLogger("pantry_planner.llm.trace")

SLOW_CALL_S = 10.0          # a call slower than this is logged as a warning
PHASES = ("dns", "connect", "tls", "upload", "wait", "download", "other")
PHASE_NAMES = {"dns": "DNS lookup", "connect": "TCP connect", "tls": "TLS handshake",
               "upload": "upload", "wait": "waiting for the server", "download": "download",
               "other": "other (client)", "backoff": "retry wait"}
_DUR = re.compile(r"\bdur=([\d.]+)")


class PhaseClock:
    """httpx's ``trace`` extension: the first ``.started`` and last ``.complete`` of each
    connection event, on the perf_counter clock."""

    def __init__(self) -> None:
        self.marks: dict[str, float] = {}

    def __call__(self, event: str, info: dict[str, Any]) -> None:
        # "http11.send_request_body.started" -> "send_request_body.started"
        name = event.partition(".")[2]
        now = time.perf_counter()
        if name.endswith(".started"):
            self.marks.setdefault(name, now)
        else:                                   # .complete or .failed
            self.marks[name.rsplit(".", 1)[0] + ".complete"] = now

    def between(self, start: str, end: str) -> float | None:
        a, b = self.marks.get(start), self.marks.get(end)
        return (b - a) * 1000 if a is not None and b is not None else None

    def span(self, event: str) -> float | None:
        return self.between(f"{event}.started", f"{event}.complete")


def resolve_ms(url: str) -> float | None:
    """How long resolving ``url``'s host takes now; None when it fails (the request that
    follows will say why)."""
    u = httpx.URL(url)
    started = time.perf_counter()
    try:
        socket.getaddrinfo(u.host, u.port or (443 if u.scheme == "https" else 80),
                           type=socket.SOCK_STREAM)
    except OSError:
        return None
    return (time.perf_counter() - started) * 1000


def server_ms(headers: httpx.Headers | None) -> float | None:
    """The provider's own processing time from ``server-timing`` (the largest ``dur``)."""
    raw = headers.get("server-timing") if headers is not None else None
    durs = [float(d) for d in _DUR.findall(raw or "")]
    return max(durs) if durs else None


def attempt_record(clock: PhaseClock, *, n: int, started: float, dns_ms: float | None,
                   status: int | None = None, error: str | None = None,
                   request: httpx.Request | None = None,
                   response: httpx.Response | None = None) -> dict[str, Any]:
    """One request, as its phases in milliseconds (None when a phase did not happen or was
    not observed, e.g. under a mock transport)."""
    total = (time.perf_counter() - started) * 1000 + (dns_ms or 0.0)
    phases = {
        "dns": dns_ms,
        "connect": clock.span("connect_tcp"),
        "tls": clock.span("start_tls"),
        "upload": clock.between("send_request_headers.started", "send_request_body.complete"),
        "wait": clock.span("receive_response_headers"),
        "download": clock.span("receive_response_body"),
    }
    phases["other"] = max(total - sum(v for v in phases.values() if v), 0.0)
    rec: dict[str, Any] = {
        "n": n, "status": status, "error": error, "total_ms": _ms(total),
        **{f"{k}_ms": _ms(v) for k, v in phases.items()},
        "new_connection": clock.span("connect_tcp") is not None,
        "server_ms": _ms(server_ms(response.headers if response is not None else None)),
        "request_bytes": len(request.content) if request is not None else None,
        "response_bytes": len(response.content) if response is not None else None,
        "retry_after": response.headers.get("retry-after") if response is not None else None,
        "backoff_ms": 0,
    }
    return rec


def phase_list(trace: dict[str, Any], server: str = "the server") -> list[dict[str, Any]]:
    """The trace as consecutive phases for a timeline: [{name, ms, attempt}], the waits
    between attempts included, phases that took no time left out."""
    out: list[dict[str, Any]] = []
    for a in trace.get("attempts") or []:
        for key in PHASES:
            ms = a.get(f"{key}_ms")
            if ms:
                name = f"waiting for {server}" if key == "wait" else PHASE_NAMES[key]
                out.append({"name": name, "ms": ms, "attempt": a["n"]})
        if a.get("backoff_ms"):
            out.append({"name": PHASE_NAMES["backoff"], "ms": a["backoff_ms"], "attempt": a["n"]})
    return out


def describe(trace: dict[str, Any], server: str = "the server") -> str:
    """One line per attempt, for the log and the trace views."""
    attempts = trace.get("attempts") or []
    head = (f"{trace.get('model', '?')} {trace.get('step', '')}".strip()
            + (f" started {trace['started_at']}" if trace.get("started_at") else "")
            + f": {_s(trace.get('total_ms'))} s, {len(attempts)} attempt"
            + ("s" if len(attempts) != 1 else ""))
    lines = [head]
    for a in attempts:
        outcome = f"HTTP {a['status']}" if a.get("status") else (a.get("error") or "no reply")
        if a.get("detail"):
            outcome += f" ({a['detail']})"
        parts = [f"{PHASE_NAMES[k] if k != 'wait' else 'waiting for ' + server} "
                 f"{_s(a.get(f'{k}_ms'))} s" for k in PHASES if a.get(f"{k}_ms")]
        extra = []
        if a.get("server_ms") is not None:
            extra.append(f"{server} reports {_s(a['server_ms'])} s of its own (server-timing)")
        if a.get("request_bytes") is not None:
            extra.append(f"{_kb(a['request_bytes'])} up / {_kb(a.get('response_bytes'))} down")
        extra.append("new connection" if a.get("new_connection") else "reused connection")
        if a.get("retry_after"):
            extra.append(f"Retry-After {a['retry_after']}")
        if a.get("backoff_ms"):
            extra.append(f"then waited {_s(a['backoff_ms'])} s before retrying")
        lines.append(f"  #{a['n']} {outcome} in {_s(a.get('total_ms'))} s = "
                     + " + ".join(parts) + (" · " + " · ".join(extra) if extra else ""))
    return "\n".join(lines)


def log_trace(trace: dict[str, Any], server: str = "the server") -> None:
    slow = (trace.get("total_ms") or 0) >= SLOW_CALL_S * 1000
    log.log(logging.WARNING if slow else logging.INFO, "LLM call trace%s %s",
            " (slow)" if slow else "", describe(trace, server))


def _ms(v: float | None) -> int | None:
    return None if v is None else round(v)


def _s(ms: float | None) -> str:
    return "?" if ms is None else f"{ms / 1000:.2f}"


def _kb(n: int | None) -> str:
    return "?" if n is None else f"{n / 1024:.1f} KB"

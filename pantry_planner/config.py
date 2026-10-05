"""
Configuration — all env-var driven, all validated at startup.

Every knob interviewers care about (model choice, routing strategy,
thresholds) is here so it's easy to point at when explaining the design.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from urllib.parse import quote


# ─── Model names ──────────────────────────────────────────────
# Kept as strings (not enums) so a new model version can be swapped
# in via env var without a code change.
HAIKU = "claude-haiku-4-5-20251001"
SONNET = "claude-sonnet-4-6"
SONNET_THINKING = "claude-sonnet-4-6"  # same model, thinking enabled at call site


def _db_url_from_env() -> str:
    """Resolve the SQLAlchemy DB URL.

    Precedence:
      1. DB_URL — full URL, used verbatim (local dev, docker-compose).
      2. DB_HOST + friends — composed into a Postgres URL. This is the
         Kubernetes path: the CNPG-generated credential secret exposes
         username/password as separate keys, so the Deployment injects
         parts rather than assembling a URL in YAML (K8s `$(VAR)`
         interpolation can't URL-encode a password; Python can).
      3. Neither — local SQLite file.
    """
    url = os.environ.get("DB_URL", "")
    if url:
        return url
    host = os.environ.get("DB_HOST", "")
    if host:
        user = os.environ.get("DB_USER", "pantry")
        password = quote(os.environ.get("DB_PASSWORD", ""), safe="")
        port = os.environ.get("DB_PORT", "5432")
        name = os.environ.get("DB_NAME", "pantry")
        return f"postgresql+psycopg://{user}:{password}@{host}:{port}/{name}"
    return "sqlite:///./pantry.db"


def redact_db_url(url: str) -> str:
    """Mask the password portion of a DB URL for safe logging."""
    if "@" in url and "://" in url:
        scheme, rest = url.split("://", 1)
        creds, host = rest.rsplit("@", 1)
        if ":" in creds:
            user = creds.split(":", 1)[0]
            return f"{scheme}://{user}:***@{host}"
    return url


# MCP bearer tokens: "label:secret[,label:secret...]". The label is what a
# submission records as its submitter/reviewer, so it should name a person
# or a client, never the secret. Parsed here so a weak token fails the
# process at startup rather than quietly protecting nothing.
MCP_TOKEN_MIN_LENGTH = 16


def parse_mcp_auth_tokens(raw: str) -> tuple[tuple[str, str], ...]:
    """Parse MCP_AUTH_TOKENS into (label, secret) pairs.

    Entries are comma-separated and whitespace-stripped; empty entries are
    ignored; an entry without a colon (or with an empty label) is labelled
    `token-<n>` by its 1-based position. A secret shorter than
    MCP_TOKEN_MIN_LENGTH raises — a guessable token is worse than none,
    because it reads as protection. Labels must be unique: they are the
    submitter/reviewer identity on origin submissions. Errors name only the
    entry position, never any part of the entry — an operator who wrote
    `secret:label` would otherwise see the secret echoed as the "label".
    """
    out: list[tuple[str, str]] = []
    for n, entry in enumerate(
            (e.strip() for e in raw.split(",") if e.strip()), start=1):
        label, sep, secret = entry.partition(":")
        if not sep:
            label, secret = "", entry
        label, secret = label.strip() or f"token-{n}", secret.strip()
        if len(secret) < MCP_TOKEN_MIN_LENGTH:
            raise ValueError(
                f"MCP_AUTH_TOKENS entry {n}: secret must be at least "
                f"{MCP_TOKEN_MIN_LENGTH} characters, got {len(secret)} "
                "(entries are label:secret — check the order)")
        for m, (existing, _) in enumerate(out, start=1):
            if existing == label:
                raise ValueError(
                    f"MCP_AUTH_TOKENS entry {n}: same label as entry {m}; labels "
                    "identify the submitter and must be unique")
        out.append((label, secret))
    return tuple(out)


def validate_startup() -> Settings:
    """Parse the configuration NOW.

    settings() is lazy and cached, so without this a bad MCP_AUTH_TOKENS
    would let the process start, pass its liveness check and then 500 on
    every request — including /health — instead of dying once with the
    ValueError that names the broken entry. Called from the API lifespan and
    the stdio entry point before anything is served.
    """
    return settings()


@dataclass(frozen=True)
class Settings:
    # Auth
    anthropic_api_key: str

    # DB
    db_url: str

    # Routing
    routing_strategy: str          # "cascade" | "three_phase"
    confidence_threshold: float    # cascade: escalate when any selection < this

    # Models
    selector_model_default: str    # main selector; cascade uses this first
    selector_model_escalation: str # what cascade escalates to; what 3-phase may pick
    classifier_model: str          # Phase B (three_phase only)
    nl2sql_model: str              # recipe/constraint extractor (NL2SQL stage)

    # Feature flags
    enable_thinking_on_escalation: bool  # if True, escalation model runs with thinking on

    # Query-plan retrieval: default shopping location when the request
    # doesn't send one (matches the seeded stores' reference point)
    default_lat: float
    default_lon: float

    # Provenance: a basket whose spend-weighted origin coverage falls below
    # this is returned LABELLED, not presented as clean. Without a floor,
    # missing data biases selection — the cheapest candidate is usually the
    # one nobody measured.
    origin_min_coverage: float

    # Split-trip optimizer: how a km of driving trades against basket
    # savings ("save $4 by adding a 6 km detour?"). CAD per km.
    travel_cost_per_km: float

    # DEMO_MODE=1: deterministic stand-ins replace both LLM boundaries
    # (recipe parse + product selection) so the public demo runs with no
    # API key, no cost, and no abuse surface. The query-plan machinery —
    # SQL templates, gates, stats, trip/week optimizers — runs unchanged.
    demo_mode: bool

    # MCP_AUTH_TOKENS: (label, secret) pairs accepted as bearer tokens on
    # /mcp. Empty → the endpoint is anonymous and write tools refuse over
    # HTTP (stdio is the operator's own process and stays trusted).
    mcp_auth_tokens: tuple[tuple[str, str], ...]

    @staticmethod
    def from_env() -> "Settings":
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            # Don't raise — tests mock the SDK. Runtime code that actually
            # calls the SDK will error clearly.
            api_key = ""

        strategy = os.environ.get("ROUTING_STRATEGY", "cascade").lower()
        if strategy not in {"cascade", "three_phase"}:
            raise ValueError(
                f"ROUTING_STRATEGY must be 'cascade' or 'three_phase', got: {strategy!r}"
            )

        return Settings(
            anthropic_api_key=api_key,
            db_url=_db_url_from_env(),
            routing_strategy=strategy,
            confidence_threshold=float(os.environ.get("CONFIDENCE_THRESHOLD", "0.80")),
            selector_model_default=os.environ.get("SELECTOR_MODEL_DEFAULT", HAIKU),
            selector_model_escalation=os.environ.get("SELECTOR_MODEL_ESCALATION", SONNET),
            classifier_model=os.environ.get("CLASSIFIER_MODEL", HAIKU),
            nl2sql_model=os.environ.get("NL2SQL_MODEL", SONNET),
            enable_thinking_on_escalation=(
                os.environ.get("ENABLE_THINKING_ON_ESCALATION", "false").lower() == "true"
            ),
            origin_min_coverage=float(
                os.environ.get("ORIGIN_MIN_COVERAGE", "0.6")),
            default_lat=float(os.environ.get("DEFAULT_LAT", "49.28")),
            default_lon=float(os.environ.get("DEFAULT_LON", "-123.12")),
            travel_cost_per_km=float(os.environ.get("TRAVEL_COST_PER_KM", "0.50")),
            demo_mode=os.environ.get("DEMO_MODE", "").lower() in {"1", "true", "yes"},
            mcp_auth_tokens=parse_mcp_auth_tokens(os.environ.get("MCP_AUTH_TOKENS", "")),
        )


@lru_cache(maxsize=1)
def settings() -> Settings:
    return Settings.from_env()


# ─── Router factory ───────────────────────────────────────────
# Kept here so the flow can import a single symbol and the strategy
# swap is one env var away. Uses lazy imports to avoid cycles.
def get_router():
    from .router.cascade import CascadeRouter
    from .router.three_phase import ThreePhaseRouter

    return {
        "cascade": CascadeRouter,
        "three_phase": ThreePhaseRouter,
    }[settings().routing_strategy]()


# ─── Cost rate cards (per 1M tokens) ──────────────────────────
# Used by tracing.py to attach cost estimates to each LLM call. Keep in
# sync with Anthropic's published pricing.
COST_PER_MTOK: dict[str, tuple[float, float]] = {
    # (input, output) in USD per million tokens
    HAIKU: (1.00, 5.00),
    SONNET: (3.00, 15.00),
}


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate USD cost of a Claude call. Returns 0.0 for unknown models."""
    rates = COST_PER_MTOK.get(model)
    if not rates:
        return 0.0
    in_rate, out_rate = rates
    return (input_tokens * in_rate + output_tokens * out_rate) / 1_000_000

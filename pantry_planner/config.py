"""
Configuration — all env-var driven, all validated at startup.

Every knob interviewers care about (model choice, routing strategy,
thresholds) is here so it's easy to point at when explaining the design.
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass, replace
from functools import lru_cache
from urllib.parse import quote

# ─── Model names ──────────────────────────────────────────────
# Kept as strings (not enums) so a new model version can be swapped
# in via env var without a code change.
HAIKU = "claude-haiku-4-5-20251001"
SONNET = "claude-sonnet-4-6"
SONNET_THINKING = "claude-sonnet-4-6"  # same model, thinking enabled at call site

# ─── Model specs: which provider serves a model setting ──────
# A model setting is a SPEC: "gemini:<model>" routes to Google Gemini,
# "anthropic:<model>" or a bare name ("claude-haiku-4-5-20251001") to
# Anthropic. The bare form is what every existing deployment sets, so it
# keeps meaning Anthropic. llm.forced_tool_call is the one place that acts
# on the provider; everything else passes the spec through untouched (it is
# what model_used and the metrics labels show).
GEMINI = "gemini"
ANTHROPIC = "anthropic"
_PROVIDERS = (GEMINI, ANTHROPIC)


def split_model_spec(spec: str) -> tuple[str, str]:
    """(provider, provider-side model name). Never raises: anything without
    a known `provider:` prefix is a bare Anthropic name."""
    spec = (spec or "").strip()
    prefix, sep, rest = spec.partition(":")
    if sep and prefix.strip().lower() in _PROVIDERS:
        return prefix.strip().lower(), rest.strip()
    return ANTHROPIC, spec


def validate_model_spec(spec: str) -> str:
    """Return the normalised spec, or raise ValueError saying what is wrong.

    Accepted: "gemini:<model>", "anthropic:<model>" (non-empty model), or a
    bare Anthropic name starting "claude-". Used where a spec arrives from a
    caller at runtime; env values are not re-validated so an existing
    deployment's bare model names keep working."""
    raw = (spec or "").strip()
    if not raw:
        raise ValueError("model spec must not be empty")
    prefix, sep, _ = raw.partition(":")
    if sep and prefix.strip().lower() in _PROVIDERS:
        provider, name = split_model_spec(raw)
        if not name:
            raise ValueError(f"model spec {raw!r} names no model after '{provider}:'")
        return f"{provider}:{name}"
    if not raw.startswith("claude-"):
        raise ValueError(
            f"model spec {raw!r} must be 'gemini:<model>', 'anthropic:<model>' "
            "or a bare 'claude-...' name")
    return raw


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

    # Gemini (any model setting of the form "gemini:<model>"). The key may
    # be empty: only a Gemini call made without one is an error.
    gemini_api_key: str = ""
    gemini_base_url: str = ""          # "" → Google's OpenAI-compatible endpoint
    gemini_reasoning_effort: str = "low"   # "" → omit (model default)

    # RUNTIME_SETTINGS_ENABLED=1: POST /settings/runtime may switch demo
    # mode and the model specs without a restart (demo UI). Off by default:
    # on a public deployment it would let anyone turn real LLM spend on.
    runtime_settings_enabled: bool = False

    # Public endpoint limits (limits.py). TRUSTED_PROXY_HOPS: how many proxies
    # in front of the API append to X-Forwarded-For (gitops sets the ingress
    # depth); 0 means the header is client input and is ignored.
    trusted_proxy_hops: int = 0
    # LLM_DAILY_COST_CAP_USD: estimated LLM spend per replica per UTC day above
    # which LLM-calling endpoints answer 503. None (unset) means no ceiling.
    llm_daily_cost_cap_usd: float | None = None

    @staticmethod
    def from_env() -> Settings:
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
            gemini_api_key=os.environ.get("GEMINI_API_KEY", "").strip(),
            gemini_base_url=os.environ.get("GEMINI_BASE_URL", "").strip(),
            gemini_reasoning_effort=os.environ.get(
                "GEMINI_REASONING_EFFORT", "low").strip().lower(),
            runtime_settings_enabled=(
                os.environ.get("RUNTIME_SETTINGS_ENABLED", "").lower()
                in {"1", "true", "yes"}),
            trusted_proxy_hops=_proxy_hops(os.environ.get("TRUSTED_PROXY_HOPS", "")),
            llm_daily_cost_cap_usd=_cost_cap(os.environ.get("LLM_DAILY_COST_CAP_USD", "")),
        )


def _proxy_hops(raw: str) -> int:
    """TRUSTED_PROXY_HOPS as a count of proxies, 0 when unset. A bad value stops the process
    at startup: guessing would either trust a forged header or rate-limit the ingress."""
    raw = raw.strip()
    if not raw:
        return 0
    if not raw.isdigit():
        raise ValueError(f"TRUSTED_PROXY_HOPS must be a whole number of proxies, got {raw!r}")
    return int(raw)


def _cost_cap(raw: str) -> float | None:
    """LLM_DAILY_COST_CAP_USD in dollars, None when unset (no ceiling)."""
    raw = raw.strip()
    if not raw:
        return None
    try:
        cap = float(raw)
    except ValueError:
        raise ValueError(f"LLM_DAILY_COST_CAP_USD must be a dollar amount, got {raw!r}") from None
    if not cap >= 0:
        raise ValueError(f"LLM_DAILY_COST_CAP_USD must be 0 or more, got {raw!r}")
    return cap


# ─── Effective settings = env (cached) + runtime overrides ───
# The demo UI switches demo mode and the model specs without a restart.
# Overrides live in one module-level mapping applied over the cached env
# settings with dataclasses.replace, so every settings() caller sees a
# change on its next call. The mapping is replaced, never mutated, so a
# reader always gets a consistent snapshot without taking the lock.
RUNTIME_MODEL_FIELDS: dict[str, str] = {
    # API name -> Settings field
    "selector_default": "selector_model_default",
    "selector_escalation": "selector_model_escalation",
    "classifier": "classifier_model",
    "nl2sql": "nl2sql_model",
}
_runtime_overrides: dict[str, object] = {}
_runtime_lock = threading.Lock()


@lru_cache(maxsize=1)
def _env_settings() -> Settings:
    return Settings.from_env()


def settings() -> Settings:
    overrides = _runtime_overrides          # one read: a consistent snapshot
    base = _env_settings()
    return replace(base, **overrides) if overrides else base


def runtime_overrides() -> dict[str, object]:
    """The overrides currently applied (a copy), keyed by Settings field."""
    return dict(_runtime_overrides)


def set_runtime_overrides(*, demo_mode: bool | None = None,
                          models: dict[str, str | None] | None = None) -> Settings:
    """Apply runtime overrides; returns the new effective settings.

    `models` is keyed by the API names in RUNTIME_MODEL_FIELDS; a None value
    leaves that model alone. Everything is validated before anything is
    applied, so a bad spec changes nothing (ValueError names the key)."""
    updates: dict[str, object] = {}
    if demo_mode is not None:
        updates["demo_mode"] = bool(demo_mode)
    for key, spec in (models or {}).items():
        if key not in RUNTIME_MODEL_FIELDS:
            raise ValueError(f"unknown model setting {key!r}; expected one of "
                             f"{sorted(RUNTIME_MODEL_FIELDS)}")
        if spec is None:
            continue
        try:
            updates[RUNTIME_MODEL_FIELDS[key]] = validate_model_spec(spec)
        except ValueError as e:
            raise ValueError(f"models.{key}: {e}") from None
    global _runtime_overrides
    with _runtime_lock:
        _runtime_overrides = {**_runtime_overrides, **updates}
    return settings()


def clear_runtime_overrides() -> None:
    global _runtime_overrides
    with _runtime_lock:
        _runtime_overrides = {}


def _reset_settings() -> None:
    """Re-read the environment AND drop runtime overrides — a full reset.

    Exposed as settings.cache_clear(), the name every test fixture already
    calls after changing the environment, so a test that flips a runtime
    override can never leak it into the next module."""
    _env_settings.cache_clear()
    clear_runtime_overrides()


settings.cache_clear = _reset_settings  # type: ignore[attr-defined]


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
# sync with Anthropic's published pricing. Keyed by the provider-side
# model name, so "anthropic:claude-..." and the bare name price the same.
COST_PER_MTOK: dict[str, tuple[float, float]] = {
    # (input, output) in USD per million tokens
    HAIKU: (1.00, 5.00),
    SONNET: (3.00, 15.00),
}


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate USD cost of an LLM call. Returns 0.0 for unknown models.

    Gemini is priced at 0.0: this deployment uses the free tier. The token
    counts still flow to the metrics and traces unchanged."""
    provider, name = split_model_spec(model)
    if provider == GEMINI:
        return 0.0
    rates = COST_PER_MTOK.get(name)
    if not rates:
        return 0.0
    in_rate, out_rate = rates
    return (input_tokens * in_rate + output_tokens * out_rate) / 1_000_000

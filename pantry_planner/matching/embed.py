"""EmbeddingGemma through the local Ollama: the vectors product matching compares.

EmbeddingGemma (`embeddinggemma` in Ollama: 768 dimensions, trained with Matryoshka
representation learning) is asked the way its model card says, with a task prompt in front of
the text. Matching is retrieval, so a product profile is a document and an ingredient name a
query:

  document   "title: <product_type> | text: <profile_text>"
  query      "task: search result | query: <ingredient name>"

(The card's symmetric "task: sentence similarity" prompt is for comparing two texts of the same
kind, which matching never does.) Only the first DIMS dimensions are kept and the vector is
re-normalized to unit length (Matryoshka truncation; 256 of 768 lose little at a catalog of
~170 products), so the cosine of two vectors is their dot product.

At request time only ingredient names no seed knows are embedded (imported, pasted or written
by the assistant), each once: vectors are cached on disk by model, size, prompt and text, under
the data dir. When Ollama cannot be reached, embed raises EmbedderUnavailableError and the matcher
matches those names by words instead (matcher.py); for RETRY_AFTER_S afterwards Ollama is not
asked again, so a recipe of twenty unknown names waits for one timeout, not twenty.

Tests never reach Ollama: they install a fake embedder (set_embedder) or none at all.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import threading
import time
from collections import deque
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, Protocol

import httpx

DIMS = 256
QUERY_PROMPT = "task: search result | query: "
DOCUMENT_PROMPT = "title: {title} | text: "
TIMEOUT = httpx.Timeout(20.0, connect=1.5)
RETRY_AFTER_S = 60.0
# The seeds' vectors were made by one build of the model, with these prompts. A live embedder
# must reproduce them before its vectors are compared with them: a seed ingredient name is
# embedded (never from the disk cache) and must come back this close to its committed vector.
# The same model and prompt give ~1.0 (the seeds are rounded to 5 decimals); another model, a
# different quantization or a changed prompt moves it well below.
PROBE_MIN = 0.99
PING_EVERY_S = 30.0

Kind = Literal["query", "document"]


class EmbedderUnavailableError(RuntimeError):
    """The embedding model cannot be reached (or is turned off); the message says why."""


def prompt(text: str, kind: Kind, title: str | None = None) -> str:
    """The text as EmbeddingGemma's model card asks for it: a retrieval query, or a document
    with its title ("none" when it has none)."""
    if kind == "query":
        return QUERY_PROMPT + text
    return DOCUMENT_PROMPT.format(title=title or "none") + text


def truncate(vector: Sequence[float], dims: int = DIMS) -> list[float]:
    """The first `dims` dimensions, re-normalized to unit length (Matryoshka truncation)."""
    head = [float(x) for x in vector[:dims]]
    norm = math.sqrt(sum(x * x for x in head))
    if not norm:
        raise ValueError("an embedding of zero length cannot be normalized")
    return [x / norm for x in head]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """The cosine of two unit vectors: their dot product."""
    return sum(x * y for x, y in zip(a, b, strict=True))


class Embedder(Protocol):
    """Anything that turns texts into unit vectors of DIMS dimensions."""

    name: str

    def embed(self, texts: list[str], kind: Kind,
              titles: list[str] | None = None) -> list[list[float]]: ...


class OllamaEmbedder:
    """POST /api/embed on a local Ollama, every text in one request, with a disk cache."""

    def __init__(self, url: str, model: str, cache_dir: Path | None,
                 transport: httpx.BaseTransport | None = None):
        self.url = url.rstrip("/")
        self.model = model
        self.name = f"{model} via {self.url}"
        self.cache_dir = cache_dir
        self._transport = transport
        self._memo: dict[str, list[float]] | None = None
        self._lock = threading.Lock()
        self.down_since: float | None = None
        self.last_error = ""
        self._ping: tuple[float, tuple[bool, str]] | None = None
        self._agrees: tuple[bool, str] | None = None

    # ── disk cache: one JSON line per text, read once ──
    def _cache_file(self) -> Path | None:
        return None if self.cache_dir is None else self.cache_dir / "embeddings.jsonl"

    def _key(self, prompted: str) -> str:
        return hashlib.sha256(f"{self.model}|{DIMS}|{prompted}".encode()).hexdigest()

    def _load(self) -> dict[str, list[float]]:
        if self._memo is None:
            self._memo = {}
            path = self._cache_file()
            if path is not None and path.exists():
                for line in path.read_text(encoding="utf-8").splitlines():
                    try:
                        row = json.loads(line)
                        self._memo[row["k"]] = row["v"]
                    except (ValueError, KeyError, TypeError):
                        continue        # a torn line from a crash costs one re-embed
        return self._memo

    def _store(self, rows: dict[str, list[float]]) -> None:
        path = self._cache_file()
        if path is None or not rows:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                for k, v in rows.items():
                    f.write(json.dumps({"k": k, "v": [round(x, 6) for x in v]}) + "\n")
        except OSError:
            pass                        # a read-only data dir only costs the cache

    def cached(self, text: str) -> list[float] | None:
        """A query's vector when it is cached, without asking Ollama."""
        with self._lock:
            return self._load().get(self._key(prompt(text, "query")))

    def embed(self, texts: list[str], kind: Kind,
              titles: list[str] | None = None) -> list[list[float]]:
        """One unit vector per text, from the cache or one request for the rest. `titles`:
        the documents' titles (documents only)."""
        prompts = [prompt(t, kind, titles[i] if titles else None) for i, t in enumerate(texts)]
        with self._lock:
            memo = self._load()
            keys = [self._key(p) for p in prompts]
            todo = list(dict.fromkeys(p for p, k in zip(prompts, keys, strict=True)
                                      if k not in memo))
        if todo:
            fresh = self._request(todo)
            with self._lock:
                rows = {self._key(p): v for p, v in zip(todo, fresh, strict=True)}
                memo.update(rows)
                self._store(rows)
        return [memo[k] for k in keys]

    def _request(self, prompts: list[str]) -> list[list[float]]:
        if not self.url:
            raise EmbedderUnavailableError("OLLAMA_URL is empty: embedding is turned off")
        if self.down_since is not None and time.monotonic() - self.down_since < RETRY_AFTER_S:
            raise EmbedderUnavailableError(f"{self.name} is unreachable ({self.last_error})")
        body = {"model": self.model, "input": prompts}
        try:
            with httpx.Client(timeout=TIMEOUT, transport=self._transport) as client:
                resp = client.post(f"{self.url}/api/embed", json=body)
            resp.raise_for_status()
            vectors = resp.json()["embeddings"]
            if len(vectors) != len(prompts):
                raise ValueError(f"{len(vectors)} embeddings for {len(prompts)} texts")
            out = [truncate(v) for v in vectors]
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as e:
            self.down_since = time.monotonic()
            self.last_error = f"{type(e).__name__}: {str(e)[:200]}"
            raise EmbedderUnavailableError(f"{self.name} is unreachable ({self.last_error})") from e
        self.down_since, self.last_error = None, ""
        return out

    def agrees_with_seeds(self) -> tuple[bool, str]:
        """(whether this embedder reproduces the seeds' vectors, why not): the model the seeds
        name, and one seed ingredient's vector within PROBE_MIN of its committed one. Asked
        once per process when it answers; EmbedderUnavailableError when it cannot be reached
        (and is not remembered, so a later call asks again)."""
        if self._agrees is not None:
            return self._agrees
        from . import seeds as _seeds

        s = _seeds.load()
        if s.model and s.model != self.model:
            self._agrees = (False, f"the seeds were embedded with {s.model!r}, not {self.model!r}")
            return self._agrees
        probe = min((i for i in s.ingredients.values() if i.vector), key=lambda i: i.name,
                    default=None)
        if probe is None:
            self._agrees = (True, "")
            return self._agrees
        got = self._request([prompt(probe.name, "query")])[0]
        cos = cosine(got, probe.vector)
        self._agrees = ((True, "") if cos >= PROBE_MIN else
                        (False, f"{self.name} does not reproduce the seeds' vectors (cosine "
                                f"{cos:.3f} for {probe.name!r}, at least {PROBE_MIN} needed): "
                                "another build of the model or another prompt"))
        return self._agrees

    def ping(self) -> tuple[bool, str]:
        """(whether Ollama answers and has the model, why not), from GET /api/tags, at most
        once every PING_EVERY_S: /health asks on every call."""
        now = time.monotonic()
        if self._ping is not None and now - self._ping[0] < PING_EVERY_S:
            return self._ping[1]
        try:
            with httpx.Client(timeout=httpx.Timeout(1.0), transport=self._transport) as client:
                resp = client.get(f"{self.url}/api/tags")
            resp.raise_for_status()
            names = {m.get("name", "") for m in resp.json().get("models", [])}
            ok = any(n == self.model or n.startswith(f"{self.model}:") for n in names)
            result = (ok, "" if ok else f"Ollama has no model {self.model!r}")
        except (httpx.HTTPError, ValueError, AttributeError) as e:
            result = (False, f"{self.name} is unreachable ({type(e).__name__})")
        self._ping = (now, result)
        return result


GEMINI_DIMS = 768
GEMINI_TIMEOUT = httpx.Timeout(30.0, connect=5.0)
# Texts per request. Gemini counts every text of a request against its quotas, and the free tier
# allows 100 texts a minute and 1,000 a day (measured 2026-11-09: a 20-text request got HTTP 429
# "requests per minute ... limit: 100" seconds after a 1-text one passed), so a call for more
# texts is sent in parts, each kept as it arrives, paced to MATCHING_EMBED_TEXTS_PER_MINUTE.
GEMINI_BATCH = 20
GEMINI_TEXTS_PER_MINUTE = 100
GEMINI_RETRIES = 2            # more attempts after a refusal naming a short delay (scripts)
_monotonic = time.monotonic   # the Gemini embedder's clock and sleep: tests replace them
_sleep = time.sleep


def _gemini_error(resp) -> tuple[str, float | None]:
    """Gemini's error message, shortened so a refusal says why (which quota, per minute or per
    day), and the delay it asks for ("Please retry in 56.8s", "retry in 21h17m36.85s"), when
    it gives one."""
    try:
        msg = " ".join(str(resp.json()["error"]["message"]).split())
    except Exception:  # noqa: BLE001 - any body shape: the status alone must do
        return "", None
    m = re.search(r"retry in (?:(\d+)h)?(?:(\d+)m)?([0-9.]+)s", msg)
    delay = (int(m.group(1) or 0) * 3600 + int(m.group(2) or 0) * 60 + float(m.group(3))
             if m else None)
    quota = re.search(r"Quota exceeded for metric: \S+, limit: \d+.*?(?=\. See |\. Please|$)", msg)
    text = (quota.group(0) if quota else msg)[:200]
    return (f": {text}" if text else ""), delay


class GeminiEmbedder:
    """Gemini's embedding model through its OpenAI-compatible endpoint (POST /embeddings),
    GEMINI_DIMS dimensions re-normalized to unit length, every text of a call in one request.
    Its vectors are compared only with seeds/gemini_vectors.json's (seeds.py), whose texts are
    identity strings, so `kind` and `titles` are not sent: the endpoint has no task types.

    Each text counts as one request against Gemini's quotas (the free tier allows 100 a minute
    and 1,000 a day), so every vector is kept in the database cache (matching/cache.py), the
    day's allowance is counted before a request is made, and requests are paced to `per_minute`
    texts: a call that would have to wait longer than `max_wait_s` for room fails instead
    (the app's default, 0, so an import falls back to words for the rest rather than stall;
    scripts wait). After a refusal nothing is asked until the delay Gemini named has passed
    (RETRY_AFTER_S when it named none)."""

    def __init__(self, model: str, api_key: str, base_url: str = "",
                 transport: httpx.BaseTransport | None = None,
                 per_minute: int = GEMINI_TEXTS_PER_MINUTE, max_wait_s: float = 0.0):
        self.model = model
        self.api_key = api_key
        self.url = (base_url or "https://generativelanguage.googleapis.com/v1beta/openai").rstrip("/")
        self.name = f"{model} via Gemini"
        self._transport = transport
        self.per_minute = max(0, per_minute)      # 0 = unpaced (a paid key)
        self.max_wait_s = max_wait_s
        self.retry_at: float | None = None        # _monotonic() before which nothing is asked
        self.last_error = ""
        self._sent: deque[tuple[float, int]] = deque()   # (when, texts) within the last minute
        self._agrees: tuple[bool, str] | None = None

    def _key(self, text: str) -> str:
        from . import cache

        return cache.embedding_key(self.model, GEMINI_DIMS, text)

    def embed(self, texts: list[str], kind: Kind = "query",
              titles: list[str] | None = None) -> list[list[float]]:
        """One unit vector per text: from the cache (matching/cache.py, every instance's
        answers), else one request for the rest, within the day's allowance of texts."""
        from . import cache

        keys = {t: self._key(t) for t in dict.fromkeys(texts)}
        found = {k: v["vector"] for k, v in cache.embeddings.get_many(keys.values()).items()}
        todo = [t for t, k in keys.items() if k not in found]
        if todo:
            with cache.embeddings.lock(keys[todo[0]]):
                found.update({k: v["vector"] for k, v in
                              cache.embeddings.get_many(keys[t] for t in todo).items()})
                todo = [t for t in todo if keys[t] not in found]
                step = min(GEMINI_BATCH, self.per_minute or GEMINI_BATCH)
                for i in range(0, len(todo), step):
                    chunk = todo[i:i + step]
                    try:
                        fresh = self._request_patiently(chunk)
                    except (cache.BudgetExceededError, EmbedderUnavailableError) as e:
                        kept = f"; {i} of {len(todo)} texts were embedded and kept" if i else ""
                        why = str(e) if isinstance(e, EmbedderUnavailableError) else \
                            f"{self.name}: {e}"
                        raise EmbedderUnavailableError(why + kept) from e
                    rows = {keys[t]: {"model": self.model, "dims": GEMINI_DIMS, "text": t,
                                      "vector": v} for t, v in zip(chunk, fresh, strict=True)}
                    cache.embeddings.put_many(rows)
                    found.update({k: v["vector"] for k, v in rows.items()})
        return [found[keys[t]] for t in texts]

    def _request_patiently(self, chunk: list[str]) -> list[list[float]]:
        """One chunk, within the minute's room and the day's allowance. A refusal that names a
        short delay (Gemini's own count of the minute runs ahead of ours now and then) is
        waited out and the chunk asked again, at most GEMINI_RETRIES times, while the wait fits
        max_wait_s: so a script finishes, and the app gives up at once. Every attempt, refused
        or not, counts against Gemini's day (measured 2026-11-09: a retry loop spent the whole
        allowance), so the day's allowance is charged per attempt and a refusal that names the
        day's quota is never retried."""
        from . import cache

        for attempt in range(GEMINI_RETRIES + 1):
            self._wait_for_room(len(chunk))
            cache.budget().spend("embed_texts", len(chunk))
            try:
                return self._request(chunk)
            except EmbedderUnavailableError:
                wait = (self.retry_at - _monotonic()) if self.retry_at is not None else None
                if (attempt == GEMINI_RETRIES or wait is None or wait <= 0
                        or wait > self.max_wait_s or "per day" in self.last_error):
                    raise
                _sleep(wait)
                self.retry_at = None
        raise AssertionError("unreachable")

    def _wait_for_room(self, n: int) -> None:
        """Waits until `n` more texts fit in the minute's allowance, or raises when that would
        take longer than max_wait_s (the room opens as the texts sent a minute ago age out)."""
        if not self.per_minute:
            return
        now = _monotonic()
        while self._sent and now - self._sent[0][0] >= 60.0:
            self._sent.popleft()
        over = sum(c for _, c in self._sent) + n - self.per_minute
        if over <= 0:
            return
        freed, until = 0, now
        for when, count in self._sent:
            freed += count
            if freed >= over:
                until = when + 60.0
                break
        wait = max(0.0, until - now)
        if wait > self.max_wait_s:
            raise EmbedderUnavailableError(
                f"{self.name}: the minute's allowance of {self.per_minute} texts is spent, "
                f"room for {n} more in {wait:.0f}s")
        _sleep(wait)

    def _request(self, texts: list[str]) -> list[list[float]]:
        """One request for `texts` (at most GEMINI_BATCH, within the minute's room)."""
        if not self.api_key:
            raise EmbedderUnavailableError("GEMINI_API_KEY is empty: Gemini embedding is off")
        if self.retry_at is not None and _monotonic() < self.retry_at:
            raise EmbedderUnavailableError(f"{self.name} is unavailable ({self.last_error})")
        self._wait_for_room(len(texts))
        self._sent.append((_monotonic(), len(texts)))
        try:
            with httpx.Client(timeout=GEMINI_TIMEOUT, transport=self._transport) as client:
                resp = client.post(f"{self.url}/embeddings",
                                   json={"model": self.model, "input": texts,
                                         "dimensions": GEMINI_DIMS},
                                   headers={"Authorization": f"Bearer {self.api_key}"})
            resp.raise_for_status()
            rows = resp.json()["data"]
            if len(rows) != len(texts):
                raise ValueError(f"{len(rows)} embeddings for {len(texts)} texts")
            out = [truncate(r["embedding"], GEMINI_DIMS) for r in rows]
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as e:
            resp = getattr(e, "response", None)
            status = getattr(resp, "status_code", None)
            reason, delay = _gemini_error(resp) if status else ("", None)
            self.last_error = f"HTTP {status}{reason}" if status else type(e).__name__
            self.retry_at = _monotonic() + (delay if delay is not None else RETRY_AFTER_S)
            raise EmbedderUnavailableError(f"{self.name} is unavailable ({self.last_error})") from e
        self.retry_at, self.last_error = None, ""
        return out

    def agrees_with_seeds(self) -> tuple[bool, str]:
        """As OllamaEmbedder's: the seeds must name this model, and one seed name's identity
        string must come back within PROBE_MIN of its committed vector (asked once)."""
        if self._agrees is not None:
            return self._agrees
        from . import seeds as _seeds

        s = _seeds.load()
        if s.model != self.model:
            self._agrees = (False, f"the seeds were embedded with {s.model!r}, not {self.model!r}")
            return self._agrees
        probe = min((i for i in s.ingredients.values() if i.vector and i.identity),
                    key=lambda i: i.name, default=None)
        if probe is None:
            self._agrees = (False, "the Gemini seeds hold no ingredient vector to check against")
            return self._agrees
        got = self._request([probe.identity])[0]
        cos = cosine(got, probe.vector)
        self._agrees = ((True, "") if cos >= PROBE_MIN else
                        (False, f"{self.name} does not reproduce the seeds' vectors (cosine "
                                f"{cos:.3f} for {probe.identity!r}, at least {PROBE_MIN} needed)"))
        return self._agrees

    def ping(self) -> tuple[bool, str]:
        return (True, "") if self.api_key else (False, "GEMINI_API_KEY is empty")


# ─── The process's embedder ──────────────────────────────────

_embedder: Embedder | None = None
_installed = False
_lock = threading.Lock()


def get_embedder() -> Embedder | None:
    """The embedder request time uses: one installed with set_embedder, else Gemini when
    EMBEDDING_MODEL is "gemini:<model>", else Ollama as the settings name it (None when
    OLLAMA_URL is empty)."""
    global _embedder, _installed
    with _lock:
        if not _installed:
            from ..config import settings

            cfg = settings()
            if cfg.embedding_model.startswith("gemini:"):
                _embedder = GeminiEmbedder(cfg.embedding_model.split(":", 1)[1],
                                           cfg.gemini_api_key, cfg.gemini_base_url,
                                           per_minute=cfg.matching_embed_texts_per_minute)
            else:
                cache = Path(cfg.data_dir).expanduser() / "matching"
                _embedder = (OllamaEmbedder(cfg.ollama_url, cfg.embedding_model, cache)
                             if cfg.ollama_url else None)
            _installed = True
        return _embedder


def set_embedder(embedder: Embedder | None) -> None:
    """Install `embedder` for this process (None: no embedder, so unknown names are matched by
    words). The test seam; reset_embedder goes back to the settings' Ollama."""
    global _embedder, _installed
    with _lock:
        _embedder, _installed = embedder, True


def reset_embedder() -> None:
    global _embedder, _installed
    with _lock:
        _embedder, _installed = None, False

"""Shared test configuration.

Every module seeds its own throwaway SQLite DB. Set PANTRY_TEST_DB_URL to a
Postgres URL and the same suite runs against production's engine instead —
that is how CI's `postgres` job works. It was added after an `IN ()` clause,
valid on SQLite and a syntax error on Postgres, shipped unseen; the first
version of that job never connected to Postgres at all because every module
hard-set SQLite. The URL is read in each module's fixture so this file only
documents the contract.
"""

import os
import tempfile

import pytest

# Burr traces from the suite go to a throwaway folder, not the developer's .burr: every plan call
# is its own Burr run, so one suite run would otherwise add a hundred runs to the Burr UI. Read by
# tracing.py at import, which follows this file. An explicit BURR_TRACKING_DIR still wins.
os.environ.setdefault("BURR_TRACKING_DIR", tempfile.mkdtemp(prefix="pantry-burr-tests-"))
# The matcher's embedding cache and event log go to a throwaway folder too, never the
# developer's ~/.pantry-data.
os.environ["PANTRY_DATA_DIR"] = tempfile.mkdtemp(prefix="pantry-data-tests-")


@pytest.fixture(autouse=True)
def _no_embedder():
    """Tests never reach Ollama (or Gemini): a name no seed knows is matched by words, as it is
    when the embedder is down, unless a test installs a fake embedder (tests/fake_embedder.py)
    or a fake describer (tests/test_describe.py).
    The committed seeds are the matcher's only knowledge."""
    from pantry_planner.matching import cache, describe, embed

    embed.set_embedder(None)
    describe.set_describer(None)
    cache.reset()
    yield
    embed.set_embedder(None)
    describe.set_describer(None)
    cache.reset()


@pytest.fixture(autouse=True)
def _fresh_limits():
    """Every test starts with full rate-limit buckets and no LLM spend today: the limiter is
    per process, and the suite calls the public endpoints far faster than any client may."""
    from pantry_planner import limits

    limits.reset()
    yield


@pytest.fixture
def by_words(monkeypatch):
    """Every name matched by words, as one no seed knows is when the embedder is down: the
    word fallback's own tests (its levels, the size cap, its tiers) run on seed names too."""
    from pantry_planner.matching import matcher

    def words(items):
        return [matcher.Meaning(name=name, source="words", form=(form or "").lower(),
                                why_words="matched by words for this test")
                for name, form in items]

    monkeypatch.setattr(matcher, "meanings", words)

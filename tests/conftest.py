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

# Burr traces from the suite go to a throwaway folder, not the developer's .burr: every plan call
# is its own Burr run, so one suite run would otherwise add a hundred runs to the Burr UI. Read by
# tracing.py at import, which follows this file. An explicit BURR_TRACKING_DIR still wins.
os.environ.setdefault("BURR_TRACKING_DIR", tempfile.mkdtemp(prefix="pantry-burr-tests-"))

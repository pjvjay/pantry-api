"""Shared test configuration.

Every module seeds its own throwaway SQLite DB. Set PANTRY_TEST_DB_URL to a
Postgres URL and the same suite runs against production's engine instead —
that is how CI's `postgres` job works. It was added after an `IN ()` clause,
valid on SQLite and a syntax error on Postgres, shipped unseen; the first
version of that job never connected to Postgres at all because every module
hard-set SQLite. The URL is read in each module's fixture so this file only
documents the contract.
"""

"""Which release of pantry-api is running: /health, MCP serverInfo and the X-Pantry-Version header.

A release image has its version baked in. build.yml passes APP_VERSION, GIT_SHA and BUILD_TIME
as build args, and the Dockerfile turns them into the PANTRY_API_* variables read here. A source
checkout falls back to ``git describe`` against vX.Y.Z tags, so a local run says how far it is
from the last release (``0.2.0-3-gabc1234``: three commits past v0.2.0). Anything else reports
"unknown".

pyproject's version is a 0.0.0 placeholder and is never read. The release number lives in the
git tag (RELEASING.md in pantry-platform), and a literal nobody bumps would be a confident wrong
answer.
"""
from __future__ import annotations

import functools
import os
import re
import subprocess
from pathlib import Path

UNKNOWN = "unknown"
RELEASES_URL = "https://github.com/pjvjay/pantry-api/releases/tag/"
_RELEASE_RE = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")
# The checkout this package was imported from (the repo root, one level above the package).
_REPO = Path(__file__).resolve().parent.parent


def _env(name: str) -> str | None:
    return os.environ.get(name, "").strip() or None


@functools.cache
def _git(repo: Path, *args: str) -> str | None:
    """git's output in `repo`, or None. Only when `repo` is itself a checkout: an installed copy
    must not report the version of whatever repository happens to enclose site-packages.
    Cached, because /health is polled and the answer cannot change under a running process
    in any way that matters."""
    if not (repo / ".git").exists():
        return None
    try:
        proc = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True,
                              timeout=3, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout.strip() or None


def app_version() -> str:
    """X.Y.Z baked into the image; else the checkout's describe without the leading v; else
    "unknown"."""
    baked = _env("PANTRY_API_VERSION")
    if baked:
        return baked
    described = _git(_REPO, "describe", "--tags", "--dirty", "--match", "v[0-9]*.[0-9]*.[0-9]*")
    return described.removeprefix("v") if described else UNKNOWN


def app_revision() -> str | None:
    """The full commit the image was built from, or the checkout's HEAD; None when unknown."""
    return _env("PANTRY_API_REVISION") or _git(_REPO, "rev-parse", "HEAD")


def build_info() -> dict[str, str | None]:
    """release: the GitHub Release page when the version is exactly a release (X.Y.Z), else
    None. built_at: the image's build time (UTC, ISO 8601), None outside a release image."""
    version = app_version()
    return {
        "release": RELEASES_URL + "v" + version if _RELEASE_RE.match(version) else None,
        "built_at": _env("PANTRY_API_BUILD_TIME"),
    }

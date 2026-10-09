"""Which release is running: version.py's source order, /health and the
X-Pantry-Version header.

The order is the baked image environment, then `git describe` of a source
checkout, then "unknown". pyproject's 0.0.0 is a placeholder that is never
reported: a number nobody bumps would be a confident wrong answer. The MCP
serverInfo version is tested in test_mcp_http.py, the one module that may
start the /mcp session manager.
"""
from __future__ import annotations

import subprocess
import tomllib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from pantry_planner import version

ROOT = Path(__file__).resolve().parent.parent
BAKED = {"PANTRY_API_VERSION": "9.9.9",
         "PANTRY_API_REVISION": "0123456789abcdef0123456789abcdef01234567",
         "PANTRY_API_BUILD_TIME": "2026-10-08T12:00:00Z"}


@pytest.fixture(autouse=True)
def app_built_first():
    """Build the app before any test here bakes a version or fakes a checkout.
    The MCP server records its version when it is built, so whichever test
    imports the app first fixes serverInfo for the whole session; it must
    come from the real environment, or test_mcp_http's check depends on the
    test order."""
    import pantry_planner.api  # noqa: F401


@pytest.fixture()
def no_release(monkeypatch, tmp_path):
    """No baked environment and no checkout: nothing to read the release from."""
    for name in BAKED:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(version, "_REPO", tmp_path)
    version._git.cache_clear()
    yield
    version._git.cache_clear()


@pytest.fixture()
def baked(monkeypatch):
    for name, value in BAKED.items():
        monkeypatch.setenv(name, value)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True,
                          check=True).stdout.strip()


@pytest.fixture()
def checkout(monkeypatch, tmp_path):
    """A throwaway repository tagged v1.2.3, standing in for a source checkout."""
    for name in BAKED:
        monkeypatch.delenv(name, raising=False)
    repo = tmp_path / "pantry-api"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@example.com",
         "commit", "-q", "--allow-empty", "-m", "one")
    _git(repo, "tag", "v1.2.3")
    monkeypatch.setattr(version, "_REPO", repo)
    version._git.cache_clear()
    yield repo
    version._git.cache_clear()


def test_baked_environment_wins(checkout, baked):
    assert version.app_version() == "9.9.9"
    assert version.app_revision() == BAKED["PANTRY_API_REVISION"]
    assert version.build_info() == {
        "release": "https://github.com/pjvjay/pantry-api/releases/tag/v9.9.9",
        "built_at": "2026-10-08T12:00:00Z"}


def test_checkout_on_a_tag_reports_the_tag(checkout):
    assert version.app_version() == "1.2.3"
    assert version.app_revision() == _git(checkout, "rev-parse", "HEAD")
    assert version.build_info()["release"].endswith("/releases/tag/v1.2.3")
    assert version.build_info()["built_at"] is None


def test_checkout_past_a_tag_says_how_far(checkout):
    _git(checkout, "-c", "user.name=t", "-c", "user.email=t@example.com",
         "commit", "-q", "--allow-empty", "-m", "two")
    version._git.cache_clear()
    described = version.app_version()
    assert described.startswith("1.2.3-1-g"), described
    # Not a release, so there is no Release page to point at.
    assert version.build_info()["release"] is None


def test_nothing_to_read_is_unknown_never_a_number(no_release):
    assert version.app_version() == "unknown"
    assert version.app_revision() is None
    assert version.build_info() == {"release": None, "built_at": None}


def test_an_empty_baked_value_counts_as_unset(monkeypatch, no_release):
    # A local `docker build` without --build-arg bakes empty strings.
    monkeypatch.setenv("PANTRY_API_VERSION", "")
    monkeypatch.setenv("PANTRY_API_REVISION", " ")
    assert version.app_version() == "unknown"
    assert version.app_revision() is None


def test_health_reports_the_release(baked):
    from pantry_planner.api import app

    body = TestClient(app).get("/health").json()
    assert body["status"] == "ok"
    assert body["version"] == "9.9.9"
    assert body["revision"] == BAKED["PANTRY_API_REVISION"]
    assert body["build"]["built_at"] == "2026-10-08T12:00:00Z"
    assert body["build"]["release"].endswith("/v9.9.9")


def test_health_reports_nulls_when_unknown(no_release):
    from pantry_planner.api import app

    body = TestClient(app).get("/health").json()
    assert body["version"] == "unknown"
    assert body["revision"] is None
    assert body["build"] == {"release": None, "built_at": None}
    # The fields the console already reads are still there.
    assert {"routing_strategy", "default_model", "demo_mode"} <= set(body)


def test_every_response_carries_the_version_header(baked):
    from pantry_planner.api import app

    client = TestClient(app)
    assert client.get("/health").headers["x-pantry-version"] == "9.9.9"
    # Errors too: a request refused before any route code runs.
    refused = client.post("/plan/nl", json={})
    assert refused.status_code == 422
    assert refused.headers["x-pantry-version"] == "9.9.9"


def test_pyproject_version_is_the_placeholder():
    # The release number is the git tag, baked into the image. If someone
    # bumps this literal by hand, it still must never be what we report.
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert project["version"] == "0.0.0"

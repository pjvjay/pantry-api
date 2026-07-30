"""Single-container demo server: SPA + API on one port.

The GitOps deployment splits these (nginx serves the SPA, ingress
rewrites /pantry/api to the API pod). The public demo (Hugging Face
Space) compresses the same artifacts into one process:

    /pantry/api/...  -> the FastAPI app (pantry_planner.api)
    /pantry/...      -> the built SPA (SPA_DIST, vite base=/pantry/)
    /                -> redirect to /pantry/

Run:  DEMO_MODE=1 SPA_DIST=/app/spa uvicorn pantry_planner.demo_server:root
"""
from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from .api import app as api

root = FastAPI(title="pantry-planner demo", docs_url=None, redoc_url=None)
root.mount("/pantry/api", api)

_dist = os.environ.get("SPA_DIST", "")
if _dist and os.path.isdir(_dist):
    root.mount("/pantry", StaticFiles(directory=_dist, html=True), name="spa")


@root.get("/")
def index() -> RedirectResponse:
    return RedirectResponse(url="/pantry/")

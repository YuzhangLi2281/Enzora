"""FastAPI application: thin adapter over the fixed-enzyme-agent package.

The main process never imports torch or model weights — scientific models run
in subprocesses (see jobs.py). The web frontend talks only to this API; all
external network access (UniProt/NCBI/AlphaFold/Kazusa/Gemini) happens here.
"""
from __future__ import annotations

from pathlib import Path
import hmac
import os

from .routers import health, input, llm, lookup, runs
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .routers import settings
from .config import load_settings

app = FastAPI(title="Enzora", version="0.2.0",
              description="Enzora enzyme engineering agent web backend")


@app.middleware("http")
async def require_access_code(request: Request, call_next):
    """Optional shared workspace login; model credentials remain server-side."""
    expected = os.environ.get("ENZORA_ACCESS_TOKEN", "")
    if request.url.path == "/api/auth/status":
        return JSONResponse({"required": bool(expected)})
    if expected and request.method != "OPTIONS" and request.url.path.startswith("/api/"):
        supplied = request.headers.get("Authorization", "")
        if not hmac.compare_digest(supplied.encode(), ("Bearer " + expected).encode()):
            return JSONResponse({"detail": "Authentication required"}, status_code=401)
    return await call_next(request)

# In production restrict origins to the deployed frontend origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=load_settings()["cors_origins"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(runs.router)
app.include_router(input.router)
app.include_router(lookup.router)
app.include_router(health.router)
app.include_router(settings.router)
app.include_router(llm.router)

# Optional same-origin production deployment: serve the built frontend if present.
_STATIC_DIR = Path(__file__).resolve().parents[3] / "web" / "dist"
if _STATIC_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(_STATIC_DIR), html=True), name="static")

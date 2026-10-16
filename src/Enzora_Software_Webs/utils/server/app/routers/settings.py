"""Settings + session key endpoints. API keys live in memory only (security.py)."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .. import config as server_config
from ..security import key_status, store_key

router = APIRouter(prefix="/api", tags=["settings"])


class KeyBody(BaseModel):
    api_key: str


@router.post("/session/key")
def session_key(payload: KeyBody) -> dict[str, Any]:
    if not payload.api_key.strip():
        raise HTTPException(status_code=400, detail="api_key is required")
    return {"session_id": store_key(payload.api_key.strip())}


@router.get("/session/status")
def session_status(session_id: str | None = None) -> dict[str, str | bool]:
    return key_status(session_id)


@router.get("/hosts")
def hosts() -> list[dict[str, Any]]:
    """The 11 offline built-in codon-optimization hosts (zh name / species / TaxID)."""
    from Enzora_Agent.outputs.codon_optimization import supported_hosts

    return supported_hosts()


@router.get("/settings")
def get_settings() -> dict[str, Any]:
    return server_config.load_settings()


@router.put("/settings")
def put_settings(partial: dict[str, Any]) -> dict[str, Any]:
    """Merge partial settings (models/llm/runtime/report/plugin_roots) and persist."""
    try:
        return server_config.save_settings(partial)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"failed to save settings: {type(exc).__name__}: {exc}") from exc

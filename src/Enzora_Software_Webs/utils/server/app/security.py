"""Secrets and filesystem safety.

Gemini API keys live only in server memory: the web UI POSTs the key once,
receives an opaque session id, and passes that session id with run/parse/plan
requests. Keys are never written to settings.json, job configs, or logs.
"""
from __future__ import annotations

import os
import threading
import uuid
from pathlib import Path

from dotenv import load_dotenv

# Also support direct uvicorn launches, which do not pass through run.py.
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)

_keys: dict[str, str] = {}
_lock = threading.Lock()


def store_key(api_key: str) -> str:
    session_id = uuid.uuid4().hex
    with _lock:
        _keys[session_id] = api_key
    return session_id


def get_key(session_id: str | None) -> str | None:
    if session_id:
        with _lock:
            key = _keys.get(session_id)
        if key:
            return key
    from .config import load_settings
    key_name = {"minimax": "MINIMAX_API_KEY", "gemini": "GEMINI_API_KEY"}[load_settings()["llm"]["provider"]]
    return os.environ.get(key_name, "").strip() or None


def key_status(session_id: str | None = None) -> dict[str, str | bool]:
    """Expose configuration state, never the credential or a fragment of it."""
    if session_id:
        with _lock:
            if _keys.get(session_id):
                return {"configured": True, "source": "session"}
    if get_key(None):
        return {"configured": True, "source": "environment"}
    return {"configured": False, "source": "none"}


def safe_join(root: Path, rel: str) -> Path:
    """Resolve rel under root, refusing anything that escapes the root."""
    root_resolved = root.resolve()
    target = (root_resolved / rel).resolve()
    if target != root_resolved and root_resolved not in target.parents:
        raise PermissionError(f"path escapes root: {rel!r}")
    return target

"""Tool catalog and preflight health checks (mirrors CLI check-paths/check-gpu/validate-cpu)."""
from __future__ import annotations

import threading
import shutil
from typing import Any

from fastapi import APIRouter, HTTPException

from ..config import load_settings

router = APIRouter(prefix="/api", tags=["health"])

_last_checks: dict[str, dict[str, Any]] = {}
_lock = threading.Lock()


def _config_for_package() -> dict[str, Any]:
    """Assemble a minimal fixed-enzyme-agent config dict for tool registration/preflight."""
    settings = load_settings()
    return settings


@router.get("/ping")
def ping() -> dict[str, Any]:
    return {"ok": True, "service": "enzora"}


@router.get("/tools")
def tools() -> list[dict[str, Any]]:
    from Enzora_Agent.tools import ToolRegistry, register_configured_tools

    settings = load_settings()
    registry = ToolRegistry()
    try:
        register_configured_tools(registry, _config_for_package())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"built-in tool registration failed: {exc}") from exc
    for root in settings.get("plugin_roots", []):
        try:
            registry.discover([root])
        except Exception:
            continue  # a broken plug-in must not take down the catalog
    return registry.catalog()


@router.post("/health/check-paths")
def check_paths() -> dict[str, Any]:
    from Enzora_Agent.diagnostics import check_paths

    errors = check_paths(_config_for_package())
    return {"ok": not errors, "errors": errors}


@router.post("/health/check-gpu")
def check_gpu() -> dict[str, Any]:
    settings = _config_for_package()
    if settings["execution"]["backend"] == "slurm":
        missing = [command for command in ("sbatch", "squeue", "sacct", "scancel")
                   if shutil.which(command) is None]
        return {
            "scheduler": {"ok": not missing, "missing_commands": missing},
            "gpu": {
                "ok": None,
                "status": "allocated_per_job",
                "details": ["The API runs on a CPU service allocation. GPU availability is checked inside each scheduled inference job."],
            },
        }
    from Enzora_Agent.diagnostics import gpu_preflight

    with _lock:
        if "gpu" in _last_checks:
            return _last_checks["gpu"]
    try:
        result = gpu_preflight(settings)
    except Exception as exc:
        result = {"paths": {"ok": False, "details": [f"{type(exc).__name__}: {exc}"]}}
    with _lock:
        _last_checks["gpu"] = result
    return result


@router.post("/health/validate-cpu")
def validate_cpu() -> dict[str, Any]:
    """Real CPU inference of all three models (very slow; first mutation only)."""
    from Enzora_Agent.diagnostics import validate_three_models_cpu

    settings = load_settings()
    cfg = _config_for_package()
    cfg.update({
        "activity": settings["activity"],
        "sequence": settings.get("validation_sequence") or "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAN",
        "output_dir": str(load_settings()["jobs_dir"] + "/_validation"),
        "runtime": {**settings["runtime"], "keep_temp": False},
        "visualization": {"enabled": False},
    })
    try:
        result = validate_three_models_cpu(cfg)
    except Exception as exc:
        result = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
    with _lock:
        _last_checks["validate_cpu"] = result
    return result

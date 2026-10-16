"""Run endpoints: submit, list, detail, SSE events, cancel, data, files."""
from __future__ import annotations

import asyncio
import copy
import csv
import json
import yaml
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from ..jobs import get_manager
from ..security import get_key, safe_join
from .. import rerank as rerank_module

router = APIRouter(prefix="/api/runs", tags=["runs"])

POOL_FILES = {
    "scan": "01_mutation_library.csv",
    "scores": "scorer_predictions.csv",
    "ranked": "02_candidate_pool_ranked.csv",
    "selected": "03_selected_topk.csv",
    "features": "04_mutation_features.csv",
}


def _active_try_dir(run_dir: Path) -> Path:
    """Artifacts live under output_dir/tryN; return the newest tryN (or run_dir)."""
    try_dirs = sorted(
        (p for p in run_dir.glob("try*") if p.is_dir()),
        key=lambda p: int(p.name[3:]) if p.name[3:].isdigit() else 0,
    )
    return try_dirs[-1] if try_dirs else run_dir


def _report_path(report_dir: Path, language: str) -> Path:
    """Resolve the legacy report.md alias within one attempt only."""
    legacy = safe_join(report_dir, "report.md")
    if legacy.is_file():
        return legacy
    result_path = safe_join(report_dir, "result.json")
    if result_path.is_file():
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
            report = result.get("markdown_report") if isinstance(result, dict) else None
            # Core reports store an absolute server path. Only use its known
            # filename, so metadata never allows downloads outside this run.
            filename = (
                report.replace("\\", "/").rsplit("/", 1)[-1]
                if isinstance(report, str) else None
            )
            if filename in {"report.md", "report_zh.md", "report_en.md"}:
                return safe_join(report_dir, filename)
        except (OSError, ValueError):
            pass
    names = (
        ("report_en.md", "report_zh.md") if language == "en"
        else ("report_zh.md", "report_en.md")
    )
    for name in names:
        path = safe_join(report_dir, name)
        if path.is_file():
            return path
    return legacy


class RunSubmit(BaseModel):
    request: dict[str, Any]
    session_id: str | None = None
    api_key: str | None = None


class RerankRequest(BaseModel):
    weights: dict[str, float]
    session_id: str | None = None


@router.post("")
def submit_run(payload: RunSubmit) -> dict[str, Any]:
    request = copy.deepcopy(payload.request)
    sequence = request.get("sequence")
    if not isinstance(sequence, str) or not sequence or len(sequence) > 1000:
        raise HTTPException(status_code=400, detail="Protein sequence must contain 1–1000 amino acids.")
    if not set(sequence) <= set("ACDEFGHIKLMNPQRSTVWY"):
        raise HTTPException(status_code=400, detail="Confirm a protein sequence using the 20 standard amino acids before running.")
    api_key = payload.api_key or get_key(payload.session_id)
    settings = get_manager().settings
    candidate = {**settings['candidate'], **request.get('candidate', {})}
    selection = {**settings['selection'], **request.get('selection', {})}
    request["candidate"], request["selection"] = candidate, selection
    llm = request.setdefault("llm", {"enabled": False})
    if llm.get("enabled") and not api_key:
        raise HTTPException(status_code=400, detail="Configure GEMINI_API_KEY on the server or set a key in Settings before enabling AI.")
    if llm.get("enabled") and llm.get("parameter_parsing", True):
        from .llm import ParseBody, parse

        preview = parse(ParseBody(request_text=str(request.get("user_request") or ""),
                                  candidate=candidate, selection=selection,
                                  session_id=payload.session_id, api_key=api_key))
        request["candidate"] = preview["resolved_candidate"]
        request["selection"] = preview["resolved_selection"]
    # Confirmed parameters go directly to the model engine. It must not parse
    # the goal again and silently change what was reviewed in the browser.
    llm["parameter_parsing"] = False
    llm.pop("api_key", None)
    try:
        from ..parameter_parse import validate_parameters

        validate_parameters(request)
        job_id = get_manager().submit(request, api_key)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"{type(exc).__name__}: {exc}") from exc
    return {"job_id": job_id}


@router.get("")
def list_runs() -> list[dict[str, Any]]:
    return get_manager().list_jobs()


@router.get("/{job_id}")
def get_run(job_id: str) -> dict[str, Any]:
    manager = get_manager()
    try:
        meta = manager.get_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="unknown job") from exc
    progress_path = _active_try_dir(Path(meta['run_dir'])) / 'progress.json'
    if progress_path.is_file():
        meta['progress'] = json.loads(progress_path.read_text())
    result_paths = sorted(
        Path(meta["run_dir"]).glob("try*/result.json"),
        key=lambda p: p.stat().st_mtime,
    )
    result_path = result_paths[-1] if result_paths else None
    if result_path is not None and result_path.is_file():
        try:
            meta["result"] = json.loads(result_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            meta["result"] = None
    return meta


@router.post("/{job_id}/cancel")
def cancel_run(job_id: str) -> dict[str, Any]:
    try:
        get_manager().cancel(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="unknown job") from exc
    return {"job_id": job_id, "status": "cancelling"}


@router.get("/{job_id}/events")
async def run_events(job_id: str, transport: Literal["sse", "poll"] = "sse") -> Any:
    manager = get_manager()
    try:
        manager.get_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="unknown job") from exc

    if transport == "poll":
        events = manager.drain_events(job_id)
        return {"events": [{"event": event, "data": data} for event, data in events],
                "finished": manager.is_finished(job_id)}

    async def stream():
        idle = 0
        while True:
            events = manager.drain_events(job_id)
            if events:
                idle = 0
                for event, data in events:
                    payload = json.dumps({"event": event, "data": data}, ensure_ascii=False)
                    yield f"data: {payload}\n\n"
            elif manager.is_finished(job_id):
                yield "data: " + json.dumps({"event": "stream_end", "data": {}}, ensure_ascii=False) + "\n\n"
                return
            else:
                idle += 1
                if idle % 8 == 0:  # heartbeat every ~4s (poll 0.5s)
                    yield f": heartbeat\n\n"
            await asyncio.sleep(0.5)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def _read_pool_csv(run_dir: Path, pool: str) -> Path:
    filename = POOL_FILES.get(pool)
    if not filename:
        raise HTTPException(status_code=400, detail=f"unknown pool; use one of {sorted(POOL_FILES)}")
    path = _active_try_dir(run_dir) / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"pool file not available yet: {filename}")
    return path


@router.get("/{job_id}/candidates")
def get_candidates(
    job_id: str,
    pool: str = Query("selected"),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
    sort_by: str | None = None,
    order: str = Query("desc"),
    filter: str | None = Query(None, description="passed | pareto | failed"),
    q: str | None = Query(None),
) -> dict[str, Any]:
    try:
        meta = get_manager().get_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="unknown job") from exc
    path = _read_pool_csv(Path(meta["run_dir"]), pool)
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"failed to read {path.name}: {exc}") from exc
    df = df.where(pd.notnull(df), None)
    if filter == "passed":
        df = df[df["passed"].astype(str).str.lower().isin({"true", "1"})]
    elif filter == "failed":
        df = df[~df["passed"].astype(str).str.lower().isin({"true", "1"})]
    elif filter == "pareto" and "pareto_optimal" in df.columns:
        df = df[df["pareto_optimal"].astype(str).str.lower().isin({"true", "1"})]
    if q:
        df = df[df["mutation"].astype(str).str.contains(q, case=False, na=False)]
    if sort_by and sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=(order != "desc"), na_position="last")
    total = int(len(df))
    start = (page - 1) * page_size
    rows = json.loads(df.iloc[start : start + page_size].to_json(orient="records"))
    return {
        "pool": pool,
        "total": total,
        "page": page,
        "page_size": page_size,
        "columns": [str(c) for c in df.columns],
        "rows": rows,
    }


@router.post("/{job_id}/rerank")
def rerank(job_id: str, payload: RerankRequest) -> dict[str, Any]:
    """Authoritative re-ranking with new weights (no model re-run)."""
    try:
        meta = get_manager().get_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="unknown job") from exc
    run_dir = _active_try_dir(Path(meta["run_dir"]))
    ranked_path = run_dir / "02_candidate_pool_ranked.csv"
    if not ranked_path.is_file():
        raise HTTPException(status_code=404, detail="ranked pool not available yet")
    with ranked_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    config_path = Path(meta['run_dir']).parent / 'request.yaml'
    selection = yaml.safe_load(config_path.read_text())['selection']
    selection['weights'] = payload.weights
    reference_path = run_dir / 'scoring_reference.json'
    if reference_path.is_file():
        selection.update(json.loads(reference_path.read_text()))
    try:
        from Enzora_Agent.config import validate_candidate_and_selection
        validate_candidate_and_selection({'candidate': {'top_k': len(rows), 'allowed_positions': None, 'excluded_positions': []},
                                          'selection': selection}, strict_ph_target=True)
        reranked = rerank_module.rerank_candidates(rows, selection)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {'weights': payload.weights, 'rows': reranked}



@router.get("/{job_id}/figures/{name}")
def get_figure(job_id: str, name: str) -> FileResponse:
    try:
        meta = get_manager().get_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="unknown job") from exc
    try:
        path = safe_join(_active_try_dir(Path(meta["run_dir"])) / "figures", name)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    if not path.is_file():
        raise HTTPException(status_code=404, detail="figure not found")
    return FileResponse(path, media_type="image/svg+xml" if path.suffix == ".svg" else "image/png")


@router.get("/{job_id}/files/{file_path:path}")
def get_file(job_id: str, file_path: str) -> FileResponse:
    try:
        meta = get_manager().get_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="unknown job") from exc
    base = Path(meta["run_dir"])
    try:
        # Accept both "report.md" and "try1/report.md" — resolve bare names
        # against the newest tryN dir when they don't exist at the top level.
        path = safe_join(base, file_path)
        if path.name == "report.md":
            # A bare alias follows the active attempt, never an older report.
            # Explicit tryN/report.md stays inside that requested attempt.
            report_dir = _active_try_dir(base) if file_path == "report.md" else path.parent
            language = str(meta.get("config_summary", {}).get("report_language", "zh"))
            path = _report_path(report_dir, language)
        elif not path.is_file():
            path = safe_join(_active_try_dir(base), file_path)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    if not path.is_file():
        detail = (
            "Report is not available for this attempt. "
            "Wait for report generation to finish, or check the run error."
            if path.name in {"report.md", "report_zh.md", "report_en.md"}
            else "file not found"
        )
        raise HTTPException(status_code=404, detail=detail)
    suffix = path.suffix.casefold()
    media = {
        ".csv": "text/csv", ".json": "application/json", ".txt": "text/plain",
        ".svg": "image/svg+xml", ".yaml": "text/yaml", ".fasta": "text/plain", ".md": "text/markdown", ".png": "image/png",
    }.get(suffix, "application/octet-stream")
    return FileResponse(path, media_type=media, filename=path.name)


class BalanceRequest(BaseModel):
    weights: dict[str, float]
    confidence: dict[str, float]


class AdvanceRequest(BaseModel):
    phase: int


@router.post('/{job_id}/balance')
def balance_run(job_id: str, payload: BalanceRequest) -> dict:
    from Enzora_Agent.config import validate_candidate_and_selection
    from Enzora_Agent.core.screening import balance_weights
    try:
        handle = get_manager().get_handle(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail='unknown active job') from exc
    run_dir = _active_try_dir(handle.run_dir)
    progress = run_dir / 'progress.json'
    if not progress.is_file():
        raise HTTPException(status_code=409, detail='Wait for the workflow to start')
    state = json.loads(progress.read_text())
    if state.get('phase', 0) > 2 or (run_dir / 'advance_3').exists() or handle.status != 'running':
        raise HTTPException(status_code=409, detail='Balancing is available before phase 3 starts')
    config = copy.deepcopy(handle.config)
    config['selection']['weights'] = payload.weights
    config['selection']['confidence'] = payload.confidence
    try:
        validate_candidate_and_selection(config, strict_ph_target=True)
        effective = balance_weights(payload.weights, payload.confidence)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    path = run_dir / 'selection.yaml'
    temporary = path.with_suffix('.yaml.tmp')
    temporary.write_text(yaml.safe_dump(config['selection']))
    temporary.replace(path)
    return {'weights': payload.weights, 'confidence': payload.confidence, 'effective_weights': effective}


@router.post('/{job_id}/advance')
def advance_run(job_id: str, payload: AdvanceRequest) -> dict:
    try:
        handle = get_manager().get_handle(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail='unknown active job') from exc
    run_dir = _active_try_dir(handle.run_dir)
    progress = run_dir / 'progress.json'
    if not progress.is_file() or handle.status != 'running':
        raise HTTPException(status_code=409, detail='No stage is awaiting Next Process')
    state = json.loads(progress.read_text())
    if payload.phase not in (2, 3) or state.get('status') != 'awaiting_next' or state.get('next_phase') != payload.phase:
        raise HTTPException(status_code=409, detail='Only the next waiting phase may be started')
    (run_dir / f'advance_{payload.phase}').touch(exist_ok=False)
    return {'phase': payload.phase, 'status': 'starting'}

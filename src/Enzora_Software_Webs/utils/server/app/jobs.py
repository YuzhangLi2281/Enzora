"""Run the pipeline locally or through Slurm and publish shared-file progress.

Execution settings come only from the administrator YAML. Model files, staged
artifacts and phase-control files remain in the same shared run directory.
"""
from __future__ import annotations

import copy
import json
import math
import re
import shlex
import os
import subprocess
import shutil
import sys
import threading
import time
import uuid
import yaml
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import config as server_config

# Stage order mirrors Enzora_Agent.workflow.STAGES (frozen engine; kept
# duplicated here deliberately to avoid importing heavy pipeline modules).
from Enzora_Agent.workflow import STAGES
from Enzora_Agent.core.runtime import cleanup_run_cache

STAGE_FILES = {
    0: '00_resolved_request.json',
    1: '01_mutation_library.csv',
    2: 'phenom/output/inference_results.csv',
    3: 'ephod/prediction.csv',
    4: 'unistab/output/agent_results.npz',
    5: '02_candidate_pool_ranked.csv',
    6: '04_mutation_features.csv',
    7: '05_codon_optimized_sequences.csv',
    8: 'umap.json',
    9: 'analysis.txt',
    10: 'report.md',
}
REPORT_FILES = ("report.md", "report_zh.md", "report_en.md")

LOG_FILES = [('vesm', 'vesm/vesm.log'), ('phenom', 'phenom/phenom.log'),
             ('protein_cache', 'phenom/protein_cache.log'), ('ligand_cache', 'phenom/ligand_cache.log'),
             ('ephod', 'ephod/ephod.log'), ('unistab', 'unistab/unistab.log')]

_POLL_INTERVAL = 1.2
_KNOWN_STAGE_COUNTS = {1: "mutations", 2: "scan_mutations"}
_SLURM_TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED"}


@dataclass
class JobHandle:
    id: str
    job_dir: Path
    run_dir: Path
    config: dict[str, Any]
    force_cpu: bool
    api_key: str | None = None
    proc: subprocess.Popen | None = None
    backend: str = "local"
    slurm_id: str | None = None
    scheduler_state: str | None = None
    returncode: int | None = None
    status: str = "queued"
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    stages: list[dict[str, Any]] = field(default_factory=list)
    events: deque[tuple[str, dict[str, Any]]] = field(default_factory=lambda: deque(maxlen=4000))
    stop_flag: threading.Event = field(default_factory=threading.Event)
    meta_lock: threading.Lock = field(default_factory=threading.Lock)


class JobManager:
    def __init__(self, settings: dict[str, Any]):
        self.settings = settings
        self.execution = copy.deepcopy(settings.get("execution", {"backend": "local"}))
        if self.execution.get("backend") not in {"local", "slurm"}:
            raise ValueError("execution.backend must be local or slurm")
        if self.execution["backend"] == "slurm":
            self._validate_slurm_settings()
        self.jobs_dir = Path(settings["jobs_dir"])
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, JobHandle] = {}
        self._queue: deque[JobHandle] = deque()
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._gpu_slots = threading.BoundedSemaphore(int(settings["runtime"].get("max_gpu_jobs", 1)))
        self._cpu_slots = threading.BoundedSemaphore(int(settings["runtime"].get("max_cpu_jobs", 2)))
        self._active: set[str] = set()
        self._scheduler = threading.Thread(target=self._schedule_loop, daemon=True)
        self._resume_slurm_jobs()
        self._scheduler.start()

    # ---------------------------------------------------------------- submit
    def submit(self, request: dict[str, Any], api_key: str | None) -> str:
        job_id = uuid.uuid4().hex[:12]
        job_dir = self.jobs_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=False)
        try:
            config = self._build_config(request, job_dir)
            config_path = job_dir / 'request.yaml'
            config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding='utf-8')
            from Enzora_Agent.config import load_config
            load_config(config_path)
        except Exception:
            shutil.rmtree(job_dir)
            raise

        handle = JobHandle(
            id=job_id,
            job_dir=job_dir,
            run_dir=job_dir / "outputs",
            config=config,
            force_cpu=bool(request.get("runtime", {}).get("force_cpu", False)),
            api_key=api_key,
            backend=self.execution["backend"],
        )
        self._write_meta(handle)
        with self._lock:
            self._jobs[job_id] = handle
            self._queue.append(handle)
        self._emit(handle, 'queued', {'job_id': job_id})
        self._wake.set()
        return job_id

    def _build_config(self, request: dict[str, Any], job_dir: Path) -> dict[str, Any]:
        settings = self.settings
        runtime = dict(settings.get("runtime", {}))
        updates = request.get('runtime', {})
        if set(updates) - {'gpu_id', 'force_cpu', 'timeout_seconds', 'keep_temp', 'phase_control'}:
            raise ValueError('Unsupported runtime controls')
        runtime.update(updates)
        # The server retention policy is an upper bound on user debug retention.
        runtime['keep_temp'] = settings['runtime']['keep_temp'] and runtime['keep_temp']
        runtime.pop("force_cpu", None)
        llm = dict(settings.get("llm", {}))
        llm_updates = request.get('llm', {})
        if set(llm_updates) - {'enabled', 'parameter_parsing', 'result_analysis', 'analysis_fail_open'}:
            raise ValueError('Unsupported LLM controls')
        llm.update(llm_updates)
        selection = {**settings["selection"], **request.get("selection", {})}
        selection["weights"] = dict(selection.get("weights", {"ph": 0.3, "activity": 0.4, "stability": 0.3}))
        mutation = dict(settings['mutation'])
        requested_mutation = request.get('mutation', {})
        forbidden = set(requested_mutation) - {'counts', 'random_seed', 'chain_id'}
        if forbidden:
            raise ValueError(f'Unsupported mutation controls: {sorted(forbidden)}')
        mutation.update(requested_mutation)
        # JSON transports integer map keys as strings; parse only exact integer keys.
        counts = mutation['counts']
        if not isinstance(counts, dict) or any(str(k) != str(int(k)) for k in counts):
            raise ValueError('mutation.counts keys must be integer mutation orders')
        mutation['counts'] = {int(k): v for k, v in counts.items()}
        candidate = {**settings['candidate'], **request.get('candidate', {})}
        if 'counts' in requested_mutation and candidate['top_k'] != sum(mutation['counts'].values()):
            raise ValueError('candidate.top_k must match mutation.counts total')
        activity = {**settings['activity'], **request.get('activity', {})}
        config = {
            "sequence_name": request.get("sequence_name") or "enzyme",
            "sequence": request["sequence"],
            "user_request": request.get("user_request") or None,
            "output_dir": str(job_dir / "outputs"),
            "candidate": candidate,
            "mutation": mutation,
            "activity": activity,
            "models": settings.get("models", {}),
            "selection": selection,
            "llm": llm,
            "visualization": {**settings["visualization"], **request.get("visualization", {})},
            "runtime": runtime,
            "report": request.get("report", settings.get("report", {})),
        }
        return config

    # -------------------------------------------------------------- scheduler
    def _schedule_loop(self) -> None:
        while True:
            self._start_ready()
            self._wake.wait(timeout=2.0)
            self._wake.clear()

    def _start_ready(self) -> None:
        with self._lock:
            pending = list(self._queue)
        for handle in pending:
            if handle.status != "queued":
                continue
            slot = self._gpu_slots if not handle.force_cpu else self._cpu_slots
            if not slot.acquire(blocking=False):
                continue
            with self._lock:
                if handle.status != "queued":
                    slot.release()
                    continue
                self._queue.remove(handle)
                if handle.backend == "local":
                    handle.status = "running"
                    handle.started_at = time.time()
                self._active.add(handle.id)
            threading.Thread(
                target=self._run_job, args=(handle, slot), daemon=True, name=f"job-{handle.id}"
            ).start()

    def _release(self, handle: JobHandle, slot: threading.BoundedSemaphore) -> None:
        slot.release()
        with self._lock:
            self._active.discard(handle.id)
            # Finished jobs are served from their persisted metadata and artifacts.
            self._jobs.pop(handle.id, None)
            handle.api_key = None
        self._wake.set()

    # ------------------------------------------------------------------- run
    def _run_job(self, handle: JobHandle, slot: threading.BoundedSemaphore, resume: bool = False) -> None:
        try:
            if not resume:
                self._spawn(handle)
            self._watch(handle)
        except Exception as exc:  # keep the job record even on unexpected failure
            if handle.slurm_id and handle.scheduler_state not in _SLURM_TERMINAL:
                try:
                    self._slurm_cancel(handle)
                except (OSError, RuntimeError, subprocess.SubprocessError):
                    pass
            handle.status = "failed"
            handle.error = f"{type(exc).__name__}: {exc}"
            handle.finished_at = time.time()
            self._emit(handle, "failed", {"error": handle.error})
            self._write_meta(handle)
        finally:
            try:
                self._cleanup_stopped_job(handle)
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
                self._emit(handle, 'cache_cleanup', {'errors': [f'{type(exc).__name__}: {exc}']})
            finally:
                self._release(handle, slot)

    def _cleanup_stopped_job(self, handle: JobHandle) -> None:
        if handle.config['runtime']['keep_temp']:
            return
        if handle.backend == 'slurm' and handle.slurm_id:
            # scancel acknowledgement alone does not mean model writers have exited.
            for attempt in range(30):
                result = self._slurm_command(['sacct', '--noheader', '--parsable2', '--jobs',
                    str(handle.slurm_id), '--format=JobIDRaw,State'])
                states = [row.split('|')[1].split()[0].rstrip('+') for row in result.stdout.splitlines()
                          if row.split('|')[0] == str(handle.slurm_id)]
                if states and states[0] in _SLURM_TERMINAL:
                    handle.scheduler_state = states[0]
                    self._write_meta(handle)
                    break
                time.sleep(2)
            else:
                raise RuntimeError('Cache retained: scheduler has not confirmed process termination')
        elif handle.proc is not None and handle.proc.poll() is None:
            raise RuntimeError('Cache retained: local model process is still running')
        for run_dir in handle.run_dir.glob('try[0-9]*'):
            marker = run_dir / 'cache_cleanup.json'
            if marker.is_file() and not json.loads(marker.read_text())['errors']:
                continue
            self._emit(handle, 'cache_cleanup', cleanup_run_cache(run_dir))

    def _spawn(self, handle: JobHandle) -> None:
        cmd = [
            sys.executable,
            "-m",
            "Enzora_Agent.cli",
            "run",
            str(handle.job_dir / "request.yaml"),
        ]
        if handle.force_cpu:
            cmd.append("--cpu")
        env = os.environ.copy()
        source_dir = str(Path(__file__).resolve().parents[4])
        env['PYTHONPATH'] = source_dir + os.pathsep + env.get('PYTHONPATH', '')
        env['PYTHONUNBUFFERED'] = '1'
        if handle.api_key:
            env["GEMINI_API_KEY"] = handle.api_key
        kwargs: dict[str, Any] = {}
        if os.name == "posix":
            kwargs["start_new_session"] = True  # allows killpg on cancel
        if handle.stop_flag.is_set():
            return
        if handle.backend == "slurm":
            self._submit_slurm(handle, cmd, env)
            return
        log_handle = (handle.job_dir / 'process.log').open('w', encoding='utf-8')
        try:
            handle.proc = subprocess.Popen(
            cmd,
            cwd=str(handle.job_dir),
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            **kwargs,
            )
        finally:
            log_handle.close()
        self._emit(handle, "started", {"job_id": handle.id})

    # ------------------------------------------------------------------ watch
    @staticmethod
    def _active_try_dir(run_dir: Path) -> Path:
        """The engine writes into output_dir/tryN; pick the newest tryN."""
        try_dirs = sorted(
            (p for p in run_dir.glob("try*") if p.is_dir()),
            key=lambda p: int(p.name[3:]) if p.name[3:].isdigit() else 0,
        )
        return try_dirs[-1] if try_dirs else run_dir

    def _watch(self, handle: JobHandle) -> None:
        seen_files: set[str] = {f"{stage['index']}:{STAGE_FILES[stage['index']]}" for stage in handle.stages if stage["index"] in STAGE_FILES}
        emitted_stages: set[int] = {stage["index"] for stage in handle.stages}
        log_offsets: dict[str, int] = {}
        last_progress = {}
        stage_started: float | None = handle.started_at
        current_stage_index = max(emitted_stages, default=-1) + 1

        def scan_once() -> None:
            nonlocal stage_started, current_stage_index, last_progress
            # detect stage files in order (artifacts live under output_dir/tryN)
            run_dir = self._active_try_dir(handle.run_dir)
            for progress_path in (run_dir / 'progress.json', run_dir / 'vesm/progress.json'):
                if progress_path.is_file():
                    raw = progress_path.read_text()
                    if raw != last_progress.get(str(progress_path)):
                        try:
                            progress = json.loads(raw)
                        except json.JSONDecodeError:
                            continue  # The shared-file writer may still be publishing this update.
                        self._emit(handle, 'progress', progress)
                        last_progress[str(progress_path)] = raw
            for index in range(len(STAGES)):
                marker = STAGE_FILES.get(index)
                if marker is None:
                    continue
                path = run_dir / marker
                if index == 10:
                    # The engine emits one language-specific Markdown report;
                    # report.md remains supported for older engine versions.
                    path = next(
                        (run_dir / name for name in REPORT_FILES if (run_dir / name).is_file()),
                        path,
                    )
                if not path.exists():
                    continue
                key = f"{index}:{marker}"
                if key in seen_files:
                    continue
                seen_files.add(key)
                # close out all intermediate stages first
                now = time.time()
                for gap in range(current_stage_index, index + 1):
                    if gap in emitted_stages:
                        continue
                    stats = self._stage_stats(handle, gap, run_dir)
                    finished_at = now if gap == index else None
                    self._record_stage(handle, gap, stage_started, finished_at, stats)
                    emitted_stages.add(gap)
                stage_started = now
                current_stage_index = index + 1
            self._tail_logs(handle, log_offsets, run_dir)

        timeout_error = None
        scheduler_errors = 0
        while not handle.stop_flag.is_set():
            if handle.backend == "slurm":
                try:
                    if self._poll_slurm(handle):
                        break
                    scheduler_errors = 0
                except (RuntimeError, OSError, subprocess.TimeoutExpired) as exc:
                    scheduler_errors += 1
                    self._emit(handle, "scheduler_warning", {"message": str(exc), "attempt": scheduler_errors})
                    if scheduler_errors >= self.execution["slurm"]["monitor_failure_limit"]:
                        raise RuntimeError("Slurm monitoring repeatedly failed; stopping the allocation") from exc
                limit = (self.execution["slurm"]["queue_timeout_seconds"] if handle.started_at is None
                         else handle.config["runtime"]["timeout_seconds"])
                since = handle.created_at if handle.started_at is None else handle.started_at
                delay = self.execution["slurm"]["poll_seconds"]
            else:
                proc = handle.proc
                if proc is not None and proc.poll() is not None:
                    handle.returncode = proc.returncode
                    break
                limit = handle.config["runtime"]["timeout_seconds"]
                since = handle.started_at
                delay = _POLL_INTERVAL
            if since is not None and time.time() - since > limit:
                timeout_error = "queue timeout" if handle.started_at is None else "execution timeout"
                self._terminate(handle)
                break
            scan_once()
            self._write_meta(handle)
            handle.stop_flag.wait(delay)
        # one final pass so fast-failing runs still emit the stages they reached
        scan_once()

        # process finished (or cancelled)
        proc = handle.proc
        if timeout_error:
            handle.status = "failed"
            handle.error = timeout_error
            handle.finished_at = time.time()
            self._emit(handle, "failed", {"error": handle.error})
        elif handle.stop_flag.is_set():
            handle.status = "cancelled"
            handle.error = "cancelled by user"
            handle.finished_at = time.time()
            self._emit(handle, "cancelled", {})
        else:
            result_paths = sorted(
                handle.run_dir.glob("try*/result.json"),
                key=lambda p: p.stat().st_mtime,
            )
            result_path = result_paths[-1] if result_paths else None
            handle.finished_at = time.time()
            if result_path is not None and result_path.is_file():
                try:
                    result = json.loads(result_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    result = {}
                if result.get("status") == "success" and handle.returncode == 0:
                    handle.status = "success"
                    self._emit(handle, "completed", {"result": result})
                else:
                    handle.status = "failed"
                    handle.error = result.get("error") or f"exit code {handle.returncode}; scheduler state {handle.scheduler_state}"
                    self._emit(handle, "failed", {"error": handle.error})
            else:
                handle.status = "failed"
                tail = self._process_tail(handle)
                handle.error = f"process exited without result.json (exit code {handle.returncode}; scheduler state {handle.scheduler_state}); tail:\n{tail}"
                self._emit(handle, "failed", {"error": handle.error})
        self._write_meta(handle)

    def _record_stage(
        self,
        handle: JobHandle,
        index: int,
        started_at: float | None,
        finished_at: float | None,
        stats: dict[str, Any] | None,
    ) -> None:
        entry: dict[str, Any] = {
            "index": index,
            "stage": STAGES[index],
            "started_at": started_at,
            "finished_at": finished_at,
            "seconds": round((finished_at - started_at), 2) if (started_at and finished_at) else None,
        }
        if stats:
            entry["stats"] = stats
        handle.stages.append(entry)
        self._emit(handle, "stage_completed", {"stage": STAGES[index], "index": index, "total": len(STAGES), "stats": stats})

    def _stage_stats(self, handle: JobHandle, index: int, run_dir: Path) -> dict[str, Any] | None:
        try:
            if index == 0 and (run_dir / "00_resolved_request.json").is_file():
                data = json.loads((run_dir / "00_resolved_request.json").read_text(encoding="utf-8"))
                if data.get("llm_parsed"):
                    self._emit(handle, "llm_parsed", {"interpretation": data["llm_parsed"].get("interpretation")})
                return {
                    "llm_enabled": data.get("llm_enabled"),
                    "codon_optimization": data.get("codon_optimization", {}).get("enabled"),
                }
            if index == 1 and (run_dir / "01_mutation_library.csv").is_file():
                lines = sum(1 for _ in (run_dir / "01_mutation_library.csv").open(encoding="utf-8")) - 1
                return {"saved_sequences": max(0, lines)}
            if index == 3 and (run_dir / "ephod" / "prediction.csv").is_file():
                lines = sum(1 for _ in (run_dir / "ephod" / "prediction.csv").open(encoding="utf-8")) - 1
                return {"count": max(0, lines - 1)}
            if index == 5 and (run_dir / "02_candidate_pool_ranked.csv").is_file():
                lines = sum(1 for _ in (run_dir / "02_candidate_pool_ranked.csv").open(encoding="utf-8")) - 1
                passed = 0
                import csv

                with (run_dir / "02_candidate_pool_ranked.csv").open(encoding="utf-8", newline="") as fh:
                    for row in csv.DictReader(fh):
                        if str(row.get("passed", "")).lower() in {"true", "1"}:
                            passed += 1
                return {"ranked": lines, "passed": passed}
            if index == 6 and (run_dir / "04_mutation_features.csv").is_file():
                lines = sum(1 for _ in (run_dir / "04_mutation_features.csv").open(encoding="utf-8")) - 1
                pareto = 0
                import csv

                with (run_dir / "04_mutation_features.csv").open(encoding="utf-8", newline="") as fh:
                    for row in csv.DictReader(fh):
                        if str(row.get("pareto_optimal", "")).lower() in {"true", "1"}:
                            pareto += 1
                return {"feature_count": lines, "pareto_optimal": pareto}
            if index == 7 and (run_dir / "05_codon_optimized_sequences.csv").is_file():
                lines = sum(1 for _ in (run_dir / "05_codon_optimized_sequences.csv").open(encoding="utf-8")) - 1
                return {"count": max(0, lines)}
            if index == 8:
                return {"figures": len(list(run_dir.glob("figures/*.svg")))}
            return None
        except OSError:
            return None

    def _tail_logs(self, handle: JobHandle, offsets: dict[str, int], run_dir: Path) -> None:
        for source, rel in LOG_FILES:
            path = run_dir / rel
            if not path.is_file():
                continue
            offset = offsets.get(rel, 0)
            try:
                size = path.stat().st_size
                if size <= offset:
                    continue
                with path.open("r", encoding="utf-8", errors="replace") as fh:
                    fh.seek(offset)
                    chunk = fh.read(size - offset)
                offsets[rel] = size
                for line in chunk.splitlines():
                    line = line.strip()
                    if line:
                        self._emit(handle, "log_line", {"source": source, "line": line[:400]})
            except OSError:
                continue

    def _process_tail(self, handle: JobHandle) -> str:
        path = handle.job_dir / 'process.log'
        return '\n'.join(path.read_text(encoding='utf-8', errors='replace').splitlines()[-40:]) if path.is_file() else ''

    # ----------------------------------------------------------------- events
    def _emit(self, handle: JobHandle, event: str, data: dict[str, Any]) -> None:
        with self._lock:
            handle.events.append((event, data))

    def drain_events(self, job_id: str) -> list[tuple[str, dict[str, Any]]]:
        with self._lock:
            handle = self._jobs.get(job_id)
            if handle is None:
                return []
            events = list(handle.events)
            handle.events.clear()
            return events

    def is_finished(self, job_id: str) -> bool:
        with self._lock:
            handle = self._jobs.get(job_id)
            return handle is None or handle.status in {"success", "failed", "cancelled"}

    # ------------------------------------------------------------------ meta
    def _write_meta(self, handle: JobHandle) -> None:
        meta = {
            "id": handle.id,
            "status": handle.status,
            "created_at": handle.created_at,
            "started_at": handle.started_at,
            "finished_at": handle.finished_at,
            "error": handle.error,
            "force_cpu": handle.force_cpu,
            "backend": handle.backend,
            "slurm_id": handle.slurm_id,
            "scheduler_state": handle.scheduler_state,
            "returncode": handle.returncode,
            "config_summary": {
                "sequence_name": handle.config.get("sequence_name"),
                "sequence_length": len(str(handle.config.get("sequence", ""))),
                "user_request": handle.config.get("user_request"),
                "report_language": handle.config.get("report", {}).get("language", "zh"),
            },
            "stages": handle.stages,
            "run_dir": str(handle.run_dir),
        }
        with handle.meta_lock:
            temporary = handle.job_dir / "meta.json.tmp"
            temporary.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(handle.job_dir / "meta.json")

    # ------------------------------------------------------------------ query
    def list_jobs(self) -> list[dict[str, Any]]:
        metas: list[dict[str, Any]] = []
        with self._lock:
            jobs = list(self._jobs.values())
        # include jobs from previous server restarts
        known_ids = {j.id for j in jobs}
        for path in sorted(self.jobs_dir.iterdir(), reverse=True):
            meta_path = path / "meta.json"
            if meta_path.is_file() and path.name not in known_ids:
                try:
                    metas.append(json.loads(meta_path.read_text(encoding="utf-8")))
                except json.JSONDecodeError:
                    continue
        for job in jobs:
            metas.append(json.loads((job.job_dir / "meta.json").read_text(encoding="utf-8")))
        metas.sort(key=lambda m: m.get("created_at", 0), reverse=True)
        return metas

    def get_job(self, job_id: str) -> dict[str, Any]:
        meta_path = self.jobs_dir / job_id / "meta.json"
        if not meta_path.is_file():
            raise KeyError(job_id)
        return json.loads(meta_path.read_text(encoding="utf-8"))

    def get_handle(self, job_id: str) -> JobHandle:
        with self._lock:
            handle = self._jobs.get(job_id)
        if handle is None:
            raise KeyError(job_id)
        return handle

    def cancel(self, job_id: str) -> None:
        handle = self.get_handle(job_id)
        if handle.status not in {"queued", "running"}:
            return
        handle.stop_flag.set()
        if handle.status == 'queued' and handle.id not in self._active:
            with self._lock:
                handle.status = 'cancelled'
                handle.finished_at = time.time()
                if handle in self._queue:
                    self._queue.remove(handle)
            self._emit(handle, 'cancelled', {})
            self._write_meta(handle)
            with self._lock:
                self._jobs.pop(handle.id, None)
                handle.api_key = None
            return
        self._terminate(handle)
        self._wake.set()

    def _terminate(self, handle: JobHandle) -> None:
        if handle.backend == "slurm":
            if handle.slurm_id:
                self._slurm_cancel(handle)
            return
        proc = handle.proc
        if proc is not None and proc.poll() is None:
            if os.name == "nt":
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            else:
                try:
                    os.killpg(os.getpgid(proc.pid), 9)
                except (ProcessLookupError, PermissionError, OSError):
                    proc.kill()

    def _validate_slurm_settings(self) -> None:
        cfg = self.execution.get("slurm")
        if not isinstance(cfg, dict):
            raise ValueError("execution.slurm must be an administrator mapping")
        for key in ("cpus_per_task", "command_timeout_seconds", "queue_timeout_seconds", "monitor_failure_limit"):
            if type(cfg.get(key)) is not int or cfg[key] <= 0:
                raise ValueError(f"execution.slurm.{key} must be a positive integer")
        if not isinstance(cfg.get("poll_seconds"), (int, float)) or not math.isfinite(cfg["poll_seconds"]) or cfg["poll_seconds"] < 1:
            raise ValueError("execution.slurm.poll_seconds must be at least 1")
        for key in ("account", "partition", "memory", "gpu_resource"):
            if not isinstance(cfg.get(key), str) or not cfg[key].strip() or "\n" in cfg[key]:
                raise ValueError(f"execution.slurm.{key} must be a nonempty single-line string")
        setup = cfg.get("setup_commands", [])
        if not isinstance(setup, list) or not all(isinstance(line, str) and "\x00" not in line for line in setup):
            raise ValueError("execution.slurm.setup_commands must contain administrator shell commands")

    def _slurm_command(self, argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        result = subprocess.run(argv, capture_output=True, text=True,
                                timeout=self.execution["slurm"]["command_timeout_seconds"], **kwargs)
        if result.returncode:
            # Do not include environment variables or the submitted script in errors.
            raise RuntimeError(f"{argv[0]} failed ({result.returncode}): {result.stderr.strip()[:500]}")
        return result

    def _submit_slurm(self, handle: JobHandle, cmd: list[str], env: dict[str, str]) -> None:
        cfg = self.execution["slurm"]
        seconds = handle.config["runtime"]["timeout_seconds"]
        walltime = f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"
        argv = ["sbatch", "--parsable", "--job-name", f"enzora-{handle.id}",
                "--account", cfg["account"], "--partition", cfg["partition"],
                "--cpus-per-task", str(cfg["cpus_per_task"]), "--mem", cfg["memory"],
                "--time", walltime, "--chdir", str(handle.job_dir),
                "--output", str(handle.job_dir / "process.log"), "--error", str(handle.job_dir / "process.log"),
                "--export=ALL"]
        if not handle.force_cpu:
            argv += ["--gres", cfg["gpu_resource"]]
        if cfg.get("qos"):
            argv += ["--qos", cfg["qos"]]
        if cfg.get("constraint"):
            argv += ["--constraint", cfg["constraint"]]
        script = "#!/bin/bash\nset -euo pipefail\n" + "\n".join(cfg.get("setup_commands", [])) + "\nexec " + shlex.join(cmd) + "\n"
        response = self._slurm_command(argv, input=script, env=env)
        identifier = response.stdout.strip().split(";", 1)[0]
        if not re.fullmatch(r"[1-9][0-9]*", identifier):
            raise RuntimeError("sbatch did not return a valid allocation ID")
        handle.slurm_id = identifier
        handle.scheduler_state = "PENDING"
        self._write_meta(handle)
        self._emit(handle, "scheduler", {"backend": "slurm", "job_id": identifier, "state": "PENDING"})
        if handle.stop_flag.is_set():
            self._slurm_cancel(handle)

    def _poll_slurm(self, handle: JobHandle) -> bool:
        queue_error = None
        try:
            queued = self._slurm_command(["squeue", "--noheader", "--jobs", str(handle.slurm_id), "--format=%T"])
            state = queued.stdout.strip().splitlines()
        except RuntimeError as exc:
            # A completed allocation can disappear between queries; sacct is authoritative.
            queue_error = exc
            state = []
        exit_code = None
        state_name = state[0].strip().upper() if state else None
        if not state or state_name in _SLURM_TERMINAL:
            accounting = self._slurm_command(["sacct", "--noheader", "--parsable2", "--jobs", str(handle.slurm_id),
                                              "--format=JobIDRaw,State,ExitCode"])
            records = [line.split("|") for line in accounting.stdout.splitlines()]
            row = next((row for row in records if len(row) >= 3 and row[0].strip() == handle.slurm_id), None)
            if row is None:
                if queue_error is not None:
                    raise queue_error
                return False  # Accounting propagation can lag behind squeue removal.
            state_name = row[1].split()[0].rstrip("+").upper()
            exit_parts = row[2].split(":")
            if len(exit_parts) != 2 or not all(part.isdigit() for part in exit_parts):
                raise RuntimeError("sacct returned an invalid ExitCode")
            exit_code = int(exit_parts[0]) or (128 + int(exit_parts[1]) if int(exit_parts[1]) else 0)
        if state_name != handle.scheduler_state:
            handle.scheduler_state = state_name
            self._emit(handle, "scheduler", {"backend": "slurm", "job_id": handle.slurm_id, "state": state_name})
        if state_name in {"RUNNING", "COMPLETING", "SUSPENDED"} and handle.started_at is None:
            handle.started_at = time.time()
            handle.status = "running"
            self._emit(handle, "started", {"job_id": handle.id})
        if state_name in _SLURM_TERMINAL:
            handle.returncode = exit_code if state_name == "COMPLETED" else (exit_code or 1)
            # squeue may briefly expose a terminal state without accounting ExitCode.
            return exit_code is not None
        return False

    def _slurm_cancel(self, handle: JobHandle) -> None:
        self._slurm_command(["scancel", str(handle.slurm_id)])

    def _resume_slurm_jobs(self) -> None:
        if self.execution["backend"] != "slurm":
            return
        for meta_path in sorted(self.jobs_dir.glob("*/meta.json")):
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("backend") != "slurm" or meta.get("status") not in {"queued", "running"} or not meta.get("slurm_id"):
                continue
            identifier = str(meta["slurm_id"])
            if not re.fullmatch(r"[1-9][0-9]*", identifier):
                raise ValueError("Persisted Slurm allocation ID is invalid")
            job_dir = meta_path.parent
            config = yaml.safe_load((job_dir / "request.yaml").read_text(encoding="utf-8"))
            handle = JobHandle(id=meta["id"], job_dir=job_dir, run_dir=job_dir / "outputs", config=config,
                               force_cpu=meta["force_cpu"], backend="slurm", slurm_id=identifier,
                               scheduler_state=meta.get("scheduler_state"), status=meta["status"],
                               created_at=meta["created_at"], started_at=meta.get("started_at"), stages=meta.get("stages", []))
            slot = self._cpu_slots if handle.force_cpu else self._gpu_slots
            if not slot.acquire(blocking=False):
                raise RuntimeError("Existing Slurm allocations exceed configured execution slots")
            self._jobs[handle.id] = handle
            self._active.add(handle.id)
            threading.Thread(target=self._run_job, args=(handle, slot, True), daemon=True,
                             name=f"job-{handle.id}").start()


_manager: JobManager | None = None
_manager_lock = threading.Lock()


def get_manager() -> JobManager:
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = JobManager(server_config.load_settings())
        return _manager

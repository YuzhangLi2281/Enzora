"""运行环境、模型子进程、任务目录及产物读写。"""
from __future__ import annotations

import csv
import json
import os
import shlex
import subprocess
import shutil
from pathlib import Path

from .types import Candidate


class ExternalCommandError(RuntimeError):
    pass


def cleanup_run_cache(run_dir: Path) -> dict:
    """Remove only reproducible, run-owned tensors and temporary files."""
    root = run_dir.resolve()
    removed_bytes = 0
    errors = []
    targets = [root / 'phenom/protein_graphs', root / 'phenom/ligand_graphs',
               root / 'vesm/model_logits_llrs_and_calibrated_scores.npz', root / 'tmp']
    for target in targets:
        if not target.exists() and not target.is_symlink():
            continue
        if target.is_symlink() or not target.resolve().is_relative_to(root):
            errors.append(f'Refused cache path outside run: {target.name}')
            continue
        try:
            files = [target] if target.is_file() else [p for p in target.rglob('*') if p.is_file() and not p.is_symlink()]
            size = sum(p.stat().st_size for p in files)
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
            removed_bytes += size
        except OSError as exc:
            errors.append(f'{target.relative_to(root)}: {type(exc).__name__}: {exc}')
    result = {'removed_bytes': removed_bytes, 'errors': errors}
    write_json(root / 'cache_cleanup.json', result)
    return result


def build_env(*, force_cpu: bool, gpu_id: int, cache_dir: str | Path) -> dict[str, str]:
    env = os.environ.copy()
    env.pop("GEMINI_API_KEY", None)
    if force_cpu:
        env["CUDA_VISIBLE_DEVICES"] = ""
    elif env.get('SLURM_JOB_ID'):
        if not env.get('CUDA_VISIBLE_DEVICES'):
            raise ValueError('Slurm GPU inference requires an allocated CUDA_VISIBLE_DEVICES mask')
        # Slurm owns physical GPU selection; model adapters use logical cuda:0.
    else:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    cache = str(Path(cache_dir).expanduser().resolve())
    Path(cache).mkdir(parents=True, exist_ok=True)
    env.setdefault("TORCH_HOME", str(Path(cache) / "torch"))
    env.setdefault("HF_HOME", str(Path(cache) / "huggingface"))
    env.setdefault("TRANSFORMERS_CACHE", str(Path(cache) / "huggingface" / "transformers"))
    env["PYTHONUNBUFFERED"] = "1"
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    return env


def _apply_ca_bundle(env: dict[str, str], cmd: list[str]) -> dict[str, str]:
    """Provide external model subprocesses with a usable HTTPS CA bundle.

    Existing explicit CA variables are preserved. Otherwise system CA is used
    first, then certifi from the subprocess Python environment is attempted.
    SSL verification is never disabled.
    """
    out = dict(env)
    existing_ssl = out.get("SSL_CERT_FILE") or os.environ.get("SSL_CERT_FILE")
    existing_requests = out.get("REQUESTS_CA_BUNDLE") or os.environ.get("REQUESTS_CA_BUNDLE")
    if existing_ssl:
        out["SSL_CERT_FILE"] = existing_ssl
        out.setdefault("REQUESTS_CA_BUNDLE", existing_ssl)
        return out
    if existing_requests:
        out["REQUESTS_CA_BUNDLE"] = existing_requests
        out.setdefault("SSL_CERT_FILE", existing_requests)
        return out

    for candidate in [
        "/etc/ssl/certs/ca-certificates.crt",
        "/etc/pki/tls/certs/ca-bundle.crt",
        "/etc/ssl/ca-bundle.pem",
    ]:
        if Path(candidate).is_file():
            out["SSL_CERT_FILE"] = candidate
            out["REQUESTS_CA_BUNDLE"] = candidate
            return out

    if cmd:
        python_exe = str(cmd[0])
        if Path(python_exe).is_file():
            try:
                probe = subprocess.run(
                    [python_exe, "-c", "import certifi; print(certifi.where())"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    timeout=5,
                    check=True,
                )
                ca = probe.stdout.strip()
                if ca and Path(ca).is_file():
                    out["SSL_CERT_FILE"] = ca
                    out["REQUESTS_CA_BUNDLE"] = ca
            except Exception:
                pass
    return out


def run_checked(
    cmd: list[str],
    *,
    cwd: str | Path,
    env: dict[str, str],
    timeout: int,
    log_file: str | Path,
) -> subprocess.CompletedProcess[str]:
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    child_env = _apply_ca_bundle(env, cmd)
    header = '$ ' + ' '.join(shlex.quote(x) for x in cmd) + '\n\n'
    with log_path.open('w', encoding='utf-8') as handle:
        handle.write(header)
        handle.flush()
        proc = subprocess.run(cmd, cwd=str(cwd), env=child_env, stdout=handle,
                              stderr=subprocess.STDOUT, text=True, timeout=timeout)
    if proc.returncode != 0:
        tail = '\n'.join(log_path.read_text(encoding='utf-8', errors='replace').splitlines()[-40:])
        raise ExternalCommandError(f'External model failed ({proc.returncode}); log: {log_path}\n{tail}')
    return proc


CANDIDATE_CSV_FIELDS = [
    "mutation",
    "substitutions",
    "wt_sequence",
    "position",
    "wt_aa",
    "mutant_aa",
    "activity_log10",
    "activity_sigma_log10",
    "activity_kcat",
    "library_score",
    "ph_opt",
    "ph_uncertainty",
    "delta_ph_from_wt",
    "ddg",
    "stability_uncertainty",
    "score_normalization",
    "activity_component",
    "ph_component",
    "stability_component",
    "final_score",
    "passed",
    "reasons",
    "blosum62",
    "conservative_substitution",
    "pareto_optimal",
    "wt_residue_class",
    "mutant_residue_class",
    "residue_class_change",
    "wt_hydrophobicity",
    "mutant_hydrophobicity",
    "delta_hydrophobicity",
    "wt_volume",
    "mutant_volume",
    "delta_volume",
    "wt_charge",
    "mutant_charge",
    "delta_charge",
    "mutant_sequence",
]


def write_candidates_csv(path: Path, candidates: list[Candidate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANDIDATE_CSV_FIELDS)
        writer.writeheader()
        for candidate in candidates:
            data = candidate.to_dict()
            data["substitutions"] = json.dumps(candidate.substitutions, ensure_ascii=False)
            data["reasons"] = "; ".join(candidate.reasons)
            writer.writerow({key: data.get(key) for key in CANDIDATE_CSV_FIELDS})
    temporary.replace(path)


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def create_try_dir(out_dir: Path) -> Path:
    """Atomically create try1, try2, ... under an output directory."""
    out_dir.mkdir(parents=True, exist_ok=True)
    numbers: list[int] = []
    for path in out_dir.iterdir():
        if path.is_dir() and path.name.startswith("try"):
            suffix = path.name[3:]
            if suffix.isdigit():
                numbers.append(int(suffix))
    next_number = max(numbers, default=0) + 1
    while True:
        run_dir = out_dir / f"try{next_number}"
        try:
            run_dir.mkdir(exist_ok=False)
            return run_dir
        except FileExistsError:
            next_number += 1

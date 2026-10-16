from __future__ import annotations

import csv
from pathlib import Path
import numpy as np

from ..core.runtime import run_checked


class UniStabAdapter:
    def __init__(self, repo: str, python: str, checkpoint: str, weights_dir: str):
        self.repo = Path(repo)
        self.python = python
        self.checkpoint = Path(checkpoint)
        self.weights_dir = Path(weights_dir)

    @property
    def entry(self) -> Path:
        return self.repo / "src" / "inference.py"

    def preflight(self) -> list[str]:
        errors = []
        if not self.repo.is_dir():
            errors.append(f"UniStab repo 不存在: {self.repo}")
        if not self.entry.is_file():
            errors.append(f"UniStab 推理入口不存在: {self.entry}")
        for relative in ('vendor/fair_esm/esm/pretrained.py',
                         'vendor/openfold/openfold/__init__.py',
                         'vendor/dllogger/dllogger/__init__.py'):
            if not (self.repo / relative).is_file():
                errors.append(f'UniStab bundled dependency missing: {self.repo / relative}')
        if not self.checkpoint.is_file():
            errors.append(f"UniStab checkpoint 不存在: {self.checkpoint}")
        for name in ('esmfold_3B_v1.pt', 'esm2_t36_3B_UR50D.pt', 'esm2_t36_3B_UR50D-contact-regression.pt'):
            path = self.weights_dir / 'hub/checkpoints' / name
            if not path.is_file():
                errors.append(f'UniStab local backbone weight missing: {path}')
        return errors

    def predict_ddg_with_uncertainty(
        self,
        wt_sequence: str,
        mutants: dict[str, str],
        *,
        work_dir: Path,
        env: dict[str, str],
        timeout: int,
        force_cpu: bool,
    ) -> dict[str, dict[str, float]]:
        work_dir.mkdir(parents=True, exist_ok=True)
        in_csv = work_dir / "unistab_input.csv"
        names = list(mutants)
        with in_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["WT_name", "ddG_ML", "mut_seq", "wt_seq"])
            writer.writeheader()
            for name in names:
                # Upstream UniversalMutationDataset requires a finite ddG_ML target even for inference.
                # We use a dummy 0.0 target and only consume predictions.
                writer.writerow({
                    "WT_name": name,
                    "ddG_ML": 0.0,
                    "mut_seq": mutants[name],
                    "wt_seq": wt_sequence,
                })
        out_dir = work_dir / "output"
        device = "cpu" if force_cpu else "cuda:0"
        run_name = "agent"
        cmd = [
            self.python,
            str(self.entry),
            "--checkpoint", str(self.checkpoint),
            "--data", str(in_csv),
            "--output", str(out_dir),
            "--name", run_name,
            "--batch_size", "1",
            "--mc_samples", "3",
            "--device", device,
        ]
        # UniStab inference imports modules from both repo root and src/.
        child_env = dict(env, TORCH_HOME=str(self.weights_dir))
        old_pythonpath = child_env.get("PYTHONPATH", "")
        required = ":".join(str(self.repo / relative) for relative in (
            'vendor/fair_esm', 'vendor/openfold', 'vendor/dllogger', '.', 'src'))
        child_env["PYTHONPATH"] = required + ((":" + old_pythonpath) if old_pythonpath else "")
        run_checked(cmd, cwd=self.repo, env=child_env, timeout=timeout, log_file=work_dir / "unistab.log")
        npz = out_dir / f"{run_name}_results.npz"
        if not npz.is_file():
            raise RuntimeError(f"UniStab 未生成结果: {npz}")
        with np.load(npz, allow_pickle=False) as z:
            preds = np.asarray(z["predictions"], dtype=float).reshape(-1)
            uncertainties = np.asarray(z["uncertainties"], dtype=float).reshape(-1)
        if len(preds) != len(names):
            raise RuntimeError(
                f"UniStab 返回数量 {len(preds)} 与输入 {len(names)} 不一致。"
                "请查看 unistab.log；上游脚本可能跳过了无效 batch。"
            )
        if not np.all(np.isfinite(preds)):
            raise RuntimeError("UniStab 返回 NaN/Inf")
        if len(uncertainties) != len(names) or not np.all(np.isfinite(uncertainties)) or np.any(uncertainties < 0):
            raise ValueError('Invalid UniStab MC uncertainty array')
        return {name: {'ddg':float(value), 'uncertainty':float(uncertainty)}
                for name,value,uncertainty in zip(names,preds,uncertainties)}

    def predict_ddg(self, wt_sequence, mutants, *, work_dir, env, timeout, force_cpu):
        values = self.predict_ddg_with_uncertainty(wt_sequence, mutants, work_dir=work_dir,
                    env=env, timeout=timeout, force_cpu=force_cpu)
        return {name: row['ddg'] for name,row in values.items()}

from __future__ import annotations

import csv
import math
from pathlib import Path

from ..core.runtime import run_checked


class EpHodAdapter:
    def __init__(self, repo: str, python: str, weights_dir: str):
        self.repo = Path(repo)
        self.python = python
        self.weights_dir = Path(weights_dir)

    @property
    def entry(self) -> Path:
        return self.repo / "ephod" / "run.py"

    def preflight(self) -> list[str]:
        errors = []
        if not self.repo.is_dir():
            errors.append(f"EpHod repo 不存在: {self.repo}")
        if not self.entry.is_file():
            errors.append(f"EpHod 入口不存在: {self.entry}")
        for name in ('ESM1v-SVR.pkl', 'ESM1v-RLATtr.pt', 'esm1v_t33_650M_UR90S_1.pt'):
            path = self.weights_dir / name
            if not path.is_file():
                errors.append(f'EpHod local weights missing: {path}')
        return errors

    def predict_with_uncertainty(
        self,
        sequences: dict[str, str],
        *,
        work_dir: Path,
        env: dict[str, str],
        timeout: int,
    ) -> dict[str, dict[str, float]]:
        work_dir.mkdir(parents=True, exist_ok=True)
        fasta = work_dir / "ephod_input.fasta"
        with fasta.open("w", encoding="utf-8") as f:
            for name, seq in sequences.items():
                f.write(f">{name}\n{seq}\n")
        out_name = "prediction.csv"
        cmd = [
            self.python,
            str(self.entry),
            "--fasta_path", str(fasta),
            "--save_dir", str(work_dir),
            "--csv_name", out_name,
            "--verbose", "1",
            "--save_attention_weights", "0",
            "--save_embeddings", "0",
        ]
        child_env = dict(env, ENZORA_EPHOD_WEIGHTS=str(self.weights_dir))
        run_checked(cmd, cwd=self.repo, env=child_env, timeout=timeout, log_file=work_dir / "ephod.log")
        out_csv = work_dir / out_name
        if not out_csv.is_file():
            raise RuntimeError(f"EpHod 未生成结果文件: {out_csv}")
        result: dict[str, dict[str, float]] = {}
        with out_csv.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if not {"Ensemble", "RLATtr", "SVR"} <= set(reader.fieldnames or []):
                raise RuntimeError(f"EpHod 输出缺少 Ensemble 列: {reader.fieldnames}")
            first_col = (reader.fieldnames or [""])[0]
            for row in reader:
                name = row.get(first_col, "")
                if name not in sequences or name in result:
                    raise ValueError(f'Unexpected/duplicate EpHod ID: {name}')
                value = float(row['Ensemble'])
                if not math.isfinite(value) or not 0 <= value <= 14:
                    raise ValueError(f'Invalid EpHod pH: {name}')
                rlattr, svr = float(row['RLATtr']), float(row['SVR'])
                if not all(math.isfinite(x) for x in (rlattr,svr)):
                    raise ValueError('Invalid EpHod submodel outputs')
                result[name] = {'score':value, 'uncertainty':abs(rlattr-svr)/2,
                                'rlattr':rlattr, 'svr':svr}
        missing = [k for k in sequences if k not in result]
        if missing:
            raise RuntimeError(f"EpHod 输出缺少序列: {missing[:10]}")
        return result

    def predict(self, sequences, *, work_dir, env, timeout):
        detailed = self.predict_with_uncertainty(sequences, work_dir=work_dir, env=env, timeout=timeout)
        return {name: row['score'] for name, row in detailed.items()}

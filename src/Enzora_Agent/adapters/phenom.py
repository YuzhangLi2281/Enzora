"""Phenom subprocess integration; all language-model weights stay local."""
from __future__ import annotations

import csv
import hashlib
import math
from pathlib import Path

from ..core.runtime import run_checked


class PhenomAdapter:
    def __init__(self, config: dict):
        self.config = config
        self.repo = Path(config['repo'])
        self.python = config['python']

    def preflight(self) -> list[str]:
        files = [self.repo / 'inference/inference.py', self.repo / 'inference/build_esmc_cache.py',
                 self.repo / 'inference/build_ligand_fingerprints.py']
        files += [Path(self.config['model_dir']) / name for name in ('model.pt', 'degree.pt', 'config.json')]
        errors = [f'PHENOM file missing: {path}' for path in files if not path.is_file()]
        for key in ('esmc_model_dir', 'molt5_model_dir'):
            directory = Path(self.config[key])
            if not directory.is_dir() or not any(directory.iterdir()):
                errors.append(f'PHENOM local {key} missing: {directory}')
        return errors

    def predict(self, sequences: dict[str, str], *, ligand_smiles: str, work_dir: Path,
                env: dict[str, str], timeout: int, force_cpu: bool) -> dict[str, dict[str, float]]:
        if not sequences or not ligand_smiles.strip():
            raise ValueError('PHENOM requires sequences and explicit ligand SMILES')
        work_dir.mkdir(parents=True, exist_ok=True)
        # Names are sequence hashes, so cached embeddings cannot be confused across runs.
        keys = {name: hashlib.sha256(seq.encode()).hexdigest() for name, seq in sequences.items()}
        seq_map = work_dir / 'protein_sequences.csv'
        input_csv = work_dir / 'pairs.csv'
        with seq_map.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle); writer.writerow(['Protein', 'Sequence'])
            for name, seq in sequences.items():
                writer.writerow([keys[name], seq])
        with input_csv.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle); writer.writerow(['ID', 'Protein', 'Ligand'])
            for name in sequences:
                writer.writerow([name, keys[name], ligand_smiles])
        protein_cache = work_dir / 'protein_graphs'
        ligand_cache = work_dir / 'ligand_graphs'
        device = 'cpu' if force_cpu else 'cuda:0'
        child_env = dict(env)
        child_env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        commands = [
            ('protein_cache', [str(self.repo / 'inference/build_esmc_cache.py'),
                '--protein-sequence-map-csv', str(seq_map), '--protein-cache-dir', str(protein_cache),
                '--esmc-model-dir', self.config['esmc_model_dir'], '--device', device]),
            ('ligand_cache', [str(self.repo / 'inference/build_ligand_fingerprints.py'),
                '--csv', str(input_csv), '--ligand-cache-dir', str(ligand_cache),
                '--molt5-model', self.config['molt5_model_dir'], '--device', device]),
            ('phenom', [str(self.repo / 'inference/inference.py'), '--input-file', str(input_csv),
                '--output-folder', str(work_dir / 'output'), '--model-dir', self.config['model_dir'],
                '--protein-cache-dir', str(protein_cache), '--ligand-cache-dir', str(ligand_cache),
                '--task', 'kcat', '--num-workers', '0', '--device', device]),
        ]
        for name, args in commands:
            run_checked([self.python, *args], cwd=self.repo, env=child_env, timeout=timeout,
                        log_file=work_dir / f'{name}.log')
        result = {}
        with (work_dir / 'output/inference_results.csv').open(newline='', encoding='utf-8') as handle:
            for row in csv.DictReader(handle):
                name = row['ID']
                if name not in sequences or name in result:
                    raise ValueError(f'PHENOM returned an unexpected/duplicate ID: {name}')
                values = {key: float(row[key]) for key in ('mu_log10', 'sigma_log10', 'prediction')}
                if not all(math.isfinite(v) for v in values.values()) or values['sigma_log10'] <= 0 or values['prediction'] <= 0:
                    raise ValueError(f'PHENOM returned invalid kcat distribution: {name}')
                result[name] = values
        if set(result) != set(sequences):
            raise ValueError('PHENOM result IDs do not match requested sequences')
        return result

from __future__ import annotations

import csv
import math
from collections import Counter
from pathlib import Path

import yaml

from ..core.sequence import parse_mutation, apply_mutation
from ..core.runtime import run_checked


class VESMAdapter:
    """Run the existing three-model constructor with per-run YAML inputs."""
    def __init__(self, config: dict):
        self.config = config
        self.entry = Path(__file__).resolve().parents[2] / 'Enzora_Mutation/VESM_Mutation_Database_Construction/vesm_mutation_database_construct.py'

    def preflight(self) -> list[str]:
        errors = []
        for path in (self.entry, Path(self.config['config']), Path(self.config['structure_path'])):
            if not path.is_file():
                errors.append(f'VESM file missing: {path}')
        if not Path(self.config['config']).is_file():
            return errors
        base = Path(self.config['config']).parent
        cfg = yaml.safe_load(Path(self.config['config']).read_text())
        for model in cfg['models'].values():
            for key in ('weights_path', 'base_weights_path', 'repository_path', 'tokenizer_path', 'residue_annotations_path'):
                if key in model:
                    path = (base / model[key]).resolve()
                    if not path.exists():
                        errors.append(f'VESM {key} missing: {path}')
        return errors

    def build(self, sequence: str, candidate: dict, *, work_dir, env: dict, timeout: int, force_cpu: bool) -> list[dict]:
        config_path = Path(self.config['config'])
        cfg = yaml.safe_load(config_path.read_text())
        # Model paths in the shared template are relative to Enzora/config.
        for model in cfg['models'].values():
            for key in ('weights_path', 'base_weights_path', 'repository_path', 'tokenizer_path', 'residue_annotations_path'):
                if key in model:
                    model[key] = str((config_path.parent / model[key]).resolve())
        work_dir.mkdir(parents=True, exist_ok=True)
        sequence_file = work_dir / 'input_sequence.txt'
        sequence_file.write_text(sequence + '\n')
        cfg['input']['sequence_path'] = str(sequence_file)
        cfg['input']['structure'].update(path=self.config['structure_path'], chain_id=self.config['chain_id'])
        cfg['library']['counts'] = self.config['counts']
        cfg['library']['random_seed'] = self.config['random_seed']
        cfg['mutation_space']['positions'] = {'include': candidate['allowed_positions'], 'exclude': candidate['excluded_positions']}
        cfg['runtime']['device'] = 'cpu' if force_cpu else 'cuda:0'
        cfg['runtime']['allow_cpu_fallback'] = False
        cfg['output']['directory'] = str(work_dir)
        request_path = work_dir / 'mutation.yaml'
        request_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        run_checked([self.config['python'], str(self.entry), '--config', str(request_path)],
                    cwd=self.entry.parent, env=env, timeout=timeout, log_file=work_dir / 'vesm.log')
        with (work_dir / cfg['output']['library_manifest']).open(newline='') as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != candidate['top_k']:
            raise ValueError(f'VESM returned {len(rows)} rows; requested {candidate["top_k"]}')
        seen = set()
        observed_counts = Counter()
        for row in rows:
            mutations = row['mutations'].split(':')
            if int(row['n_mut']) != len(mutations) or not math.isfinite(float(row['ensemble_llr_sum'])):
                raise ValueError('VESM returned an invalid mutation count or generation score')
            observed_counts[len(mutations)] += 1
            expected = sequence
            positions = set()
            for mutation in mutations:
                _, pos, _ = parse_mutation(mutation)
                if pos in positions:
                    raise ValueError('Duplicate mutated position in VESM output')
                positions.add(pos)
                expected = apply_mutation(expected, mutation)
            if row['sequence'] != expected or row['sequence'] in seen:
                raise ValueError('VESM returned inconsistent or duplicate sequences')
            seen.add(row['sequence'])
        if dict(observed_counts) != self.config['counts']:
            raise ValueError('VESM output does not match the requested per-order quotas')
        return rows

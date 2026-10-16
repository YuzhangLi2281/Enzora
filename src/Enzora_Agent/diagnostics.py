"""模型资源检查、GPU 检查与三评分器 CPU 验证。"""
from __future__ import annotations

import math
import os
import subprocess
from pathlib import Path

from .adapters import EpHodAdapter, PhenomAdapter, UniStabAdapter
from .adapters.vesm import VESMAdapter
from .core.runtime import build_env, create_try_dir
from .core.sequence import apply_mutation, generate_single_substitutions


def adapters_from_config(cfg: dict):
    m = cfg['models']
    return (VESMAdapter(cfg['mutation']), EpHodAdapter(m['ephod']['repo'], m['ephod']['python'], m['ephod']['weights_dir']),
            PhenomAdapter(m['phenom']),
            UniStabAdapter(m['unistab']['repo'], m['unistab']['python'], m['unistab']['checkpoint'], m['unistab']['weights_dir']))


def check_paths(cfg: dict) -> list[str]:
    return [error for adapter in adapters_from_config(cfg) for error in adapter.preflight()]


def gpu_preflight(cfg: dict) -> dict:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(cfg['runtime']['gpu_id']))
    output = {}
    providers = {**{name: model['python'] for name, model in cfg['models'].items()}, 'vesm': cfg['mutation']['python']}
    for name, python in providers.items():
        try:
            proc = subprocess.run([python, '-c',
                "import torch; print(torch.__version__); print(torch.cuda.is_available()); "
                "print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO_CUDA')"],
                env=env, capture_output=True, text=True, timeout=30)
            output[name] = {'ok': proc.returncode == 0 and 'True' in proc.stdout and 'NO_CUDA' not in proc.stdout,
                            'details': proc.stdout + proc.stderr}
        except (OSError, subprocess.TimeoutExpired) as exc:
            output[name] = {'ok': False, 'details': str(exc)}
    return output


def validate_three_models_cpu(cfg: dict) -> dict:
    seq = cfg['sequence']
    mutation = generate_single_substitutions(seq)[0]
    mutant = apply_mutation(seq, mutation)
    m = cfg['models']
    ephod = EpHodAdapter(m['ephod']['repo'], m['ephod']['python'], m['ephod']['weights_dir'])
    phenom = PhenomAdapter(m['phenom'])
    unistab = UniStabAdapter(m['unistab']['repo'], m['unistab']['python'], m['unistab']['checkpoint'], m['unistab']['weights_dir'])
    errors = ephod.preflight() + phenom.preflight() + unistab.preflight()
    if errors:
        raise RuntimeError('\n'.join(errors))
    work = create_try_dir(Path(cfg['output_dir']) / 'validation')
    env = build_env(force_cpu=True, gpu_id=0, cache_dir=cfg['runtime']['cache_dir'])
    timeout = cfg['runtime']['timeout_seconds']
    activity = phenom.predict({mutation: mutant}, ligand_smiles=cfg['activity']['ligand_smiles'],
        work_dir=work / 'phenom', env=env, timeout=timeout, force_cpu=True)[mutation]
    ph = ephod.predict({'WT': seq, mutation: mutant}, work_dir=work / 'ephod', env=env, timeout=timeout)
    ddg = unistab.predict_ddg(seq, {mutation: mutant}, work_dir=work / 'unistab', env=env, timeout=timeout, force_cpu=True)[mutation]
    values = {'phenom_log10_kcat': activity['mu_log10'], 'phenom_sigma_log10': activity['sigma_log10'],
              'ephod_wt_ph': ph['WT'], 'ephod_mutant_ph': ph[mutation], 'unistab_ddg': ddg}
    if not all(math.isfinite(v) for v in values.values()):
        raise RuntimeError('Nonfinite scorer output')
    return {'mutation': mutation, **values, 'passed': True, 'run_dir': str(work)}

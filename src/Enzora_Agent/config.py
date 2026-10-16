from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import yaml

from .core.sequence import normalize_sequence


class ConfigError(ValueError):
    pass


def validate_candidate_and_selection(data: dict[str, Any], *, strict_ph_target: bool) -> None:
    candidate, selection = data['candidate'], data['selection']
    for section, value in [('candidate', candidate), ('selection', selection)]:
        if type(value['top_k']) is not int or value['top_k'] < 1:
            raise ConfigError(f'{section}.top_k must be a positive integer')
    if selection['top_k'] > candidate['top_k']:
        raise ConfigError('selection.top_k cannot exceed the library size candidate.top_k')
    for field in ('allowed_positions', 'excluded_positions'):
        values = candidate[field]
        if values is not None and (not isinstance(values, list) or any(type(v) is not int or v < 1 for v in values)):
            raise ConfigError(f'candidate.{field} must be a list of 1-based positions or null')
        if values is not None and len(values) != len(set(values)):
            raise ConfigError(f'candidate.{field} contains duplicate positions')
        if 'sequence' in data and values and max(values) > len(data['sequence']):
            raise ConfigError(f'candidate.{field} exceeds sequence length')
    if selection['stability_direction'] not in {'lower_is_better', 'higher_is_better'}:
        raise ConfigError('Invalid stability_direction')
    if selection['stability_operator'] not in {'none', 'le', 'ge'}:
        raise ConfigError('Invalid stability_operator')
    for field in ('target_ph', 'ph_tolerance', 'min_activity_log10', 'stability_threshold'):
        value = selection[field]
        if value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
            raise ConfigError(f'selection.{field} must be finite or null')
    if selection['stability_operator'] != 'none' and selection['stability_threshold'] is None:
        raise ConfigError('stability_threshold is required for a hard stability filter')
    weights = selection['weights']
    validate_weights(weights)
    if strict_ph_target and weights['ph'] > 0 and selection['target_ph'] is None:
        raise ConfigError('A positive pH weight requires target_ph')
    if selection['target_ph'] is not None and not 0 <= selection['target_ph'] <= 14:
        raise ConfigError('target_ph must be between 0 and 14')
    if selection['ph_tolerance'] is not None and not 0 < selection['ph_tolerance'] <= 14:
        raise ConfigError('ph_tolerance must be in (0, 14]')
    if selection['ph_tolerance'] is not None and selection['target_ph'] is None:
        raise ConfigError('ph_tolerance requires target_ph')
    if 'confidence' in selection:
        validate_weights(selection['confidence'], confidence=True)
        if sum(weights[k] * selection['confidence'][k] for k in weights) <= 0:
            raise ConfigError('Confidence-weighted preference sum must be positive')


def validate_weights(values: dict, *, confidence: bool = False) -> None:
    if not isinstance(values, dict) or set(values) != {'ph', 'activity', 'stability'}:
        raise ConfigError('Exactly ph/activity/stability are required')
    for key, value in values.items():
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (confidence and value > 1):
            raise ConfigError(f'Invalid {key}: expected a finite nonnegative number' + (' <= 1' if confidence else ''))
    if sum(values.values()) <= 0:
        raise ConfigError('At least one weight/confidence must be positive')


def load_config(path: str | Path) -> dict[str, Any]:
    cfg_path = Path(path).expanduser().resolve()
    data = yaml.safe_load(cfg_path.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ConfigError('Configuration must be a YAML mapping')
    data = copy.deepcopy(data)
    base = cfg_path.parent
    for key in ('sequence', 'models', 'candidate', 'selection', 'runtime', 'mutation', 'activity', 'llm', 'output_dir', 'report', 'visualization'):
        if key not in data:
            raise ConfigError(f'Missing configuration section: {key}')
    if not isinstance(data['sequence'], str):
        raise ConfigError('sequence must be a protein string')
    data['sequence'] = normalize_sequence(data['sequence'])
    def resolve(value):
        p = Path(value).expanduser()
        return str((base / p).resolve() if not p.is_absolute() else p)
    data['output_dir'] = resolve(data['output_dir'])
    data['runtime']['cache_dir'] = resolve(data['runtime']['cache_dir'])
    if 'api_key' in data['llm']:
        raise ConfigError('Credentials must use the configured environment variable, never a run config')
    if set(data['models']) != {'ephod', 'phenom', 'unistab'}:
        raise ConfigError('Scorers must be exactly ephod, phenom, unistab; generation models belong in mutation config')
    path_fields = {'repo', 'checkpoint', 'model_dir', 'esmc_model_dir', 'molt5_model_dir', 'weights_dir'}
    for name, model in data['models'].items():
        for field in path_fields & model.keys():
            model[field] = resolve(model[field])
        if '/' in model['python'] or '\\' in model['python']:
            model['python'] = resolve(model['python'])
    for field in ('config', 'structure_path'):
        value = data['mutation'][field]
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(f'mutation.{field} is required; no structure is inferred')
        data['mutation'][field] = resolve(value)
    if '/' in data['mutation']['python']:
        data['mutation']['python'] = resolve(data['mutation']['python'])
    counts = data['mutation']['counts']
    if not isinstance(counts, dict) or not counts or any(type(k) is not int or type(v) is not int or k < 1 or v < 1 for k, v in counts.items()):
        raise ConfigError('mutation.counts must map positive integer mutation orders to positive quotas')
    if sum(counts.values()) != data['candidate']['top_k']:
        raise ConfigError('mutation.counts quotas must sum to candidate.top_k')
    if max(counts) > len(data['sequence']):
        raise ConfigError('Mutation order exceeds sequence length')
    if type(data['mutation']['random_seed']) is not int:
        raise ConfigError('mutation.random_seed must be an integer')
    if not isinstance(data['mutation']['chain_id'], str) or not data['mutation']['chain_id']:
        raise ConfigError('mutation.chain_id is required')
    allowed = data['candidate']['allowed_positions']
    excluded = set(data['candidate']['excluded_positions'] or [])
    eligible = (set(range(1, len(data['sequence']) + 1)) if allowed is None else set(allowed)) - excluded
    for order, quota in counts.items():
        if order > len(eligible) or quota > math.comb(len(eligible), order) * 19 ** order:
            raise ConfigError(f'Requested {quota} distinct {order}-mutants exceeds the eligible mutation space')
    if data['activity']['task'] != 'kcat':
        raise ConfigError('Activity ranking currently requires PHENOM kcat (log10 s^-1)')
    if not isinstance(data['activity']['ligand_smiles'], str) or not data['activity']['ligand_smiles'].strip():
        raise ConfigError('PHENOM kcat requires an explicit ligand_smiles')
    if data['report']['language'] not in {'en', 'zh'}:
        raise ConfigError('report.language must be en or zh')
    for field in ('parameter_model', 'analysis_model'):
        if not isinstance(data['llm'][field], str) or not data['llm'][field].startswith('gemini-'):
            raise ConfigError(f'llm.{field} must name a Gemini model')
    for field in ('timeout_seconds',):
        if type(data['runtime'][field]) is not int or data['runtime'][field] <= 0:
            raise ConfigError(f'runtime.{field} must be a positive integer')
    strict = not (data['llm']['enabled'] and data['llm']['parameter_parsing'] and data.get('user_request'))
    validate_candidate_and_selection(data, strict_ph_target=bool(strict))
    data['_config_path'] = str(cfg_path)
    return data

"""Private server settings live in Enzora/config/server.yaml, relative paths to config/."""
from __future__ import annotations

import copy
import os
import threading
from pathlib import Path
from typing import Any

import yaml

from Enzora_Agent.config import ConfigError

PROJECT_DIR = Path(__file__).resolve().parents[5]
SETTINGS_PATH = Path(os.environ.get('ENZORA_SETTINGS', str(PROJECT_DIR / 'config/server.yaml'))).resolve()
_lock = threading.Lock()


def load_settings() -> dict[str, Any]:
    with _lock:
        data = yaml.safe_load(SETTINGS_PATH.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ConfigError('server.yaml must contain a mapping')
    required = ('models', 'runtime', 'llm', 'jobs_dir', 'mutation', 'activity', 'candidate', 'selection', 'cors_origins')
    for key in required:
        if key not in data:
            raise ConfigError(f'server.yaml missing {key}')
    def resolve(value):
        p = Path(value).expanduser()
        return str((SETTINGS_PATH.parent / p).resolve())
    out = copy.deepcopy(data)
    for model in out['models'].values():
        for key in ('repo', 'checkpoint', 'model_dir', 'esmc_model_dir', 'molt5_model_dir', 'weights_dir'):
            if key in model:
                model[key] = resolve(model[key])
        if '/' in model['python']:
            model['python'] = resolve(model['python'])
    for key in ('config', 'structure_path'):
        out['mutation'][key] = resolve(out['mutation'][key])
    if '/' in out['mutation']['python']:
        out['mutation']['python'] = resolve(out['mutation']['python'])
    out['runtime']['cache_dir'] = resolve(out['runtime']['cache_dir'])
    out['jobs_dir'] = resolve(out['jobs_dir'])
    out['plugin_roots'] = [resolve(value) for value in out['plugin_roots']]
    return out


def save_settings(partial: dict[str, Any]) -> dict[str, Any]:
    """Persist only existing nonsecret fields; malformed config is never discarded."""
    with _lock:
        current = yaml.safe_load(SETTINGS_PATH.read_text(encoding='utf-8'))
        for section, updates in partial.items():
            if section == "execution":
                raise ConfigError("execution is administrator-only; edit the server YAML directly")
            if section not in current:
                raise ConfigError(f'Unknown settings section: {section}')
            if section in ('models', 'llm', 'runtime', 'report', 'mutation', 'activity', 'candidate', 'selection', 'visualization'):
                if not isinstance(updates, dict):
                    raise ConfigError(f'{section} must be an object')
                for key, value in updates.items():
                    if key not in current[section] or key in {'api_key', 'secret'}:
                        raise ConfigError(f'Unsupported or secret settings field: {section}.{key}')
                    if section == 'models':
                        if not isinstance(value, dict) or set(value) - set(current[section][key]):
                            raise ConfigError(f'Unsupported model fields: {key}')
                        current[section][key].update(value)
                    else:
                        current[section][key] = value
            else:
                current[section] = updates
        temporary = SETTINGS_PATH.with_suffix('.yaml.tmp')
        temporary.write_text(yaml.safe_dump(current, allow_unicode=True, sort_keys=False), encoding='utf-8')
        temporary.replace(SETTINGS_PATH)
    return load_settings()

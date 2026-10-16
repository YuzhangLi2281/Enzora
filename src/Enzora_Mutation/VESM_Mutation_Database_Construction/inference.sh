#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python "$script_dir/vesm_mutation_database_construct.py" --config "$script_dir/../../../config/mutation.yaml" "$@"

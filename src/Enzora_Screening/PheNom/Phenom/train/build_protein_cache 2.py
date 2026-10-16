from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
CODEBASE_DIR = SCRIPT_DIR.parent
if str(CODEBASE_DIR) not in sys.path:
    sys.path.insert(0, str(CODEBASE_DIR))

from utils.protein_init import save_protein_cache_with_keys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build PSICHIC+ protein graph cache files from a protein sequence map.")
    parser.add_argument("--protein-sequence-map-csv", type=Path, required=True)
    parser.add_argument("--protein-map-key-column", default="Protein")
    parser.add_argument("--protein-map-sequence-column", default="Sequence")
    parser.add_argument("--protein-cache-dir", type=Path, required=True)
    parser.add_argument(
        "--filter-csv",
        type=Path,
        action="append",
        default=[],
        help="Optional train/valid/test CSV. If passed, only protein keys used in these files are built.",
    )
    parser.add_argument("--filter-key-column", default="Protein")
    parser.add_argument("--limit", type=int, help="Optional maximum number of missing proteins to build.")
    parser.add_argument("--overwrite", action="store_true", help="Rebuild proteins even if cache files already exist.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.protein_cache_dir.mkdir(parents=True, exist_ok=True)

    protein_map = pd.read_csv(
        args.protein_sequence_map_csv,
        usecols=[args.protein_map_key_column, args.protein_map_sequence_column],
    )
    protein_map = protein_map.dropna(subset=[args.protein_map_key_column, args.protein_map_sequence_column])
    protein_map[args.protein_map_key_column] = protein_map[args.protein_map_key_column].astype(str)
    protein_map[args.protein_map_sequence_column] = protein_map[args.protein_map_sequence_column].astype(str)
    protein_map = protein_map.drop_duplicates(subset=[args.protein_map_key_column], keep="first")

    if args.filter_csv:
        requested_keys = set()
        for csv_path in args.filter_csv:
            frame = pd.read_csv(csv_path, usecols=[args.filter_key_column])
            requested_keys.update(frame[args.filter_key_column].dropna().astype(str).unique())
        protein_map = protein_map[protein_map[args.protein_map_key_column].isin(requested_keys)]
        missing_from_map = sorted(requested_keys.difference(set(protein_map[args.protein_map_key_column])))
        if missing_from_map:
            preview = ", ".join(missing_from_map[:10])
            raise ValueError(f"{len(missing_from_map)} requested protein keys are absent from the sequence map: {preview}")

    missing_rows = []
    for key, sequence in protein_map[
        [args.protein_map_key_column, args.protein_map_sequence_column]
    ].itertuples(index=False, name=None):
        cache_path = args.protein_cache_dir / f"{key}.pt"
        if args.overwrite or not cache_path.exists():
            missing_rows.append((key, sequence))

    if args.limit is not None:
        missing_rows = missing_rows[: args.limit]

    print(f"protein_map_rows={len(protein_map)}")
    print(f"missing_cache_files={len(missing_rows)}")
    print(f"protein_cache_dir={args.protein_cache_dir}")
    if not missing_rows:
        return

    keys = [key for key, _ in missing_rows]
    sequences = [sequence for _, sequence in missing_rows]
    save_protein_cache_with_keys(keys, sequences, args.protein_cache_dir)


if __name__ == "__main__":
    main()

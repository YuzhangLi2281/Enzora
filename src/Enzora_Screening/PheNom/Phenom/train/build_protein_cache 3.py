from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import esm
import pandas as pd
import torch
from transformers import EsmTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
CODEBASE_DIR = SCRIPT_DIR.parent
WORKSPACE_DIR = CODEBASE_DIR.parents[1]
ENZGFM_CODE_DIR = WORKSPACE_DIR / "Protein_seq_Mutation_Database_Construction/VESM_Enzyme/EnzGFM"
if str(CODEBASE_DIR) not in sys.path:
    sys.path.insert(0, str(CODEBASE_DIR))

from utils.protein_init import contact_map, esm_extract, pro_res_table, replace_non_standard_residues, seq_feature


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build PHENOME protein graphs with EnzGFM residue embeddings.")
    parser.add_argument("--protein-sequence-map-csv", type=Path, required=True)
    parser.add_argument("--protein-map-key-column", default="Protein")
    parser.add_argument("--protein-map-sequence-column", default="Sequence")
    parser.add_argument("--protein-cache-dir", type=Path, required=True)
    parser.add_argument("--enzgfm-model-dir", type=Path, required=True)
    parser.add_argument("--enzgfm-tokenizer-dir", type=Path, required=True)
    parser.add_argument("--filter-csv", type=Path, action="append", default=[])
    parser.add_argument("--filter-key-column", default="Protein")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.protein_cache_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(
        args.protein_sequence_map_csv,
        usecols=[args.protein_map_key_column, args.protein_map_sequence_column],
    ).dropna().drop_duplicates(args.protein_map_key_column)
    frame[args.protein_map_key_column] = frame[args.protein_map_key_column].astype(str)
    frame[args.protein_map_sequence_column] = frame[args.protein_map_sequence_column].astype(str)

    if args.filter_csv:
        requested = set()
        for path in args.filter_csv:
            requested.update(pd.read_csv(path, usecols=[args.filter_key_column])[args.filter_key_column].astype(str))
        available = set(frame[args.protein_map_key_column])
        missing = sorted(requested - available)
        if missing:
            raise ValueError(f"Protein sequence map is missing {len(missing)} IDs: {missing[:10]}")
        frame = frame[frame[args.protein_map_key_column].isin(requested)]

    rows = [
        (key, sequence)
        for key, sequence in frame[[args.protein_map_key_column, args.protein_map_sequence_column]].itertuples(index=False)
        if args.overwrite or not (args.protein_cache_dir / f"{key}.pt").exists()
    ]
    if args.limit is not None:
        rows = rows[: args.limit]
    print(f"protein_map_rows={len(frame)}")
    print(f"cache_files_to_build={len(rows)}")
    if not rows:
        return

    if not ENZGFM_CODE_DIR.exists():
        raise FileNotFoundError(f"EnzGFM source directory not found: {ENZGFM_CODE_DIR}")
    sys.path.insert(0, str(ENZGFM_CODE_DIR))
    from models.modeling_EnzGFM import EnzGFM_Model

    tokenizer = EsmTokenizer.from_pretrained(args.enzgfm_tokenizer_dir)
    enzgfm = EnzGFM_Model.from_pretrained(args.enzgfm_model_dir).to(args.device).eval()
    esm_model, esm_alphabet = esm.pretrained.load_model_and_alphabet("esm2_t33_650M_UR50D")
    esm_model = esm_model.to(args.device).eval()
    batch_converter = esm_alphabet.get_batch_converter()

    with torch.inference_mode():
        for index, (key, raw_sequence) in enumerate(rows, start=1):
            sequence = replace_non_standard_residues(re.sub(r"[UZOB]", "X", raw_sequence.upper()), pro_res_table)
            if len(sequence) > 1000:
                sequence = sequence[:500] + sequence[-500:]

            tokens = tokenizer(sequence, add_special_tokens=True, return_tensors="pt")
            enzgfm_output = enzgfm(
                input_ids=tokens["input_ids"].to(args.device),
                attention_mask=tokens["attention_mask"].to(args.device),
            ).last_hidden_state[0]
            token_representation = enzgfm_output[1 : len(sequence) + 1].cpu()
            if token_representation.shape != (len(sequence), 1024):
                raise ValueError(
                    f"EnzGFM output for {key} has shape {tuple(token_representation.shape)}, expected {(len(sequence), 1024)}"
                )

            _, contact_probability, _ = esm_extract(
                esm_model, batch_converter, sequence, layer=33, approach="last", dim=1280
            )
            edge_index, edge_weight = contact_map(torch.as_tensor(contact_probability))
            graph = {
                "seq": sequence,
                "seq_feat": torch.from_numpy(seq_feature(sequence)),
                "token_representation": token_representation.half(),
                "num_nodes": len(sequence),
                "num_pos": torch.arange(len(sequence)).reshape(-1, 1),
                "edge_index": edge_index,
                "edge_weight": edge_weight,
                "encoder": "EnzGFM",
                "encoder_dimension": 1024,
                "contact_encoder": "ESM2",
            }
            torch.save(graph, args.protein_cache_dir / f"{key}.pt")
            print(f"[{index}/{len(rows)}] cached {key} ({len(sequence)} residues)")


if __name__ == "__main__":
    main()

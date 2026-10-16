#!/usr/bin/env python3
"""Validate PHENOM protein caches against their source sequence map."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
import torch


AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWYX")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-map", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=0)
    return parser.parse_args()


def processed_sequence(raw: str) -> str:
    sequence = re.sub(r"[UZOB]", "X", raw.upper())
    sequence = "".join(residue if residue in AMINO_ACIDS else "X" for residue in sequence)
    return sequence if len(sequence) <= 1000 else sequence[:500] + sequence[-500:]


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.sequence_map, usecols=["Protein", "Sequence"]).drop_duplicates("Protein")
    missing = [key for key in frame.Protein.astype(str) if not (args.cache_dir / f"{key}.pt").is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} protein caches; first IDs: {missing[:10]}")

    if args.sample_size and len(frame) > args.sample_size:
        indices = torch.linspace(0, len(frame) - 1, args.sample_size).round().long().unique().tolist()
        audited = frame.iloc[indices]
    else:
        audited = frame

    residue_total = edge_total = 0
    embedding_dim = encoder = contact_encoder = None
    for key, raw_sequence in audited[["Protein", "Sequence"]].itertuples(index=False):
        graph = torch.load(args.cache_dir / f"{key}.pt", map_location="cpu")
        required = {
            "seq", "seq_feat", "token_representation", "num_nodes", "num_pos",
            "edge_index", "edge_weight", "encoder", "encoder_dimension", "contact_encoder",
        }
        absent = required.difference(graph)
        if absent:
            raise ValueError(f"Cache {key} is missing fields: {sorted(absent)}")
        sequence = processed_sequence(str(raw_sequence))
        length = len(sequence)
        if graph["seq"] != sequence or int(graph["num_nodes"]) != length:
            raise ValueError(f"Cache {key} sequence/node count does not match its source")
        representation = graph["token_representation"]
        if embedding_dim is None:
            embedding_dim = int(graph["encoder_dimension"])
            encoder = graph["encoder"]
            contact_encoder = graph["contact_encoder"].split(" last-", 1)[0]
        if representation.shape != (length, embedding_dim) or representation.dtype != torch.float16:
            raise ValueError(f"Cache {key} has invalid embedding tensor {representation.shape}/{representation.dtype}")
        if not torch.isfinite(representation).all() or float(representation.float().std()) == 0.0:
            raise ValueError(f"Cache {key} has non-finite or constant EnzGFM values")
        if graph["seq_feat"].shape != (length, 33):
            raise ValueError(f"Cache {key} has invalid residue feature shape")
        edge_index, edge_weight = graph["edge_index"], graph["edge_weight"]
        if edge_index.ndim != 2 or edge_index.shape[0] != 2 or edge_weight.shape != (edge_index.shape[1],):
            raise ValueError(f"Cache {key} has inconsistent ESM2 contact graph tensors")
        if (
            not torch.isfinite(edge_weight).all()
            or float(edge_weight.min()) < (0.0 if encoder.startswith("ESMC") else 0.5)
            or float(edge_weight.max()) > 1.0
        ):
            raise ValueError(f"Cache {key} has invalid contact edge weights")
        if int(edge_index.min()) < 0 or int(edge_index.max()) >= length:
            raise ValueError(f"Cache {key} has out-of-range contact edges")
        edges = set(map(tuple, edge_index.t().tolist()))
        if any((target, source) not in edges for source, target in edges):
            raise ValueError(f"Cache {key} contact graph is not undirected")
        if any((node, node) not in edges for node in range(length)):
            raise ValueError(f"Cache {key} contact graph lacks PHENOM self loops")
        if graph["encoder"] != encoder or int(graph["encoder_dimension"]) != embedding_dim:
            raise ValueError(f"Cache {key} does not declare the requested embedding contract")
        if not graph["contact_encoder"].startswith(contact_encoder):
            raise ValueError(f"Cache {key} does not declare the requested contact encoder")
        residue_total += length
        edge_total += edge_index.shape[1]

    report = {
        "status": "PASS",
        "sequence_map": str(args.sequence_map.resolve()),
        "cache_dir": str(args.cache_dir.resolve()),
        "expected_cache_files": len(frame),
        "audited_cache_files": len(audited),
        "audited_residues": residue_total,
        "audited_contact_edges": edge_total,
        "embedding_contract": f"{encoder} residue embedding ({embedding_dim} channels)",
        "contact_contract": contact_encoder,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

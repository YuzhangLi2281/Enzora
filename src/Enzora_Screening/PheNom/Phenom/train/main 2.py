from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
CODEBASE_DIR = SCRIPT_DIR.parent
if str(CODEBASE_DIR) not in sys.path:
    sys.path.insert(0, str(CODEBASE_DIR))

from models.net import net
from trainers import SourceAwareUncertaintyTrainer, TrainerConfig, write_json
from utils.dataloader import DataLoader
from utils.dataset import ProteinMoleculeDataset
from utils.protein_init import save_protein_cache_with_keys


DEFAULT_CONFIG: dict[str, Any] = {
    "params": {
        "mol_in_channels": 61,
        "mol_edge_channels": 30,
        "clique_pe_walk_length": 20,
        "prot_in_channels": 33,
        "prot_evo_channels": 1280,
        "hidden_channels": 200,
        "aggregators": ["mean", "min", "max", "std"],
        "scalers": ["identity", "amplification", "linear"],
        "pre_layers": 2,
        "post_layers": 1,
        "total_layer": 3,
        "K": [5, 10, 20],
        "dropout": 0,
        "dropout_attn_score": 0.2,
        "heads": 5,
    }
}


class CustomWeightedRandomSampler(torch.utils.data.WeightedRandomSampler):
    """Weighted sampler that supports very large sample counts."""

    def __iter__(self):
        if not self.replacement:
            return super().__iter__()
        weights = self.weights.detach().cpu().numpy().astype(np.float64, copy=False)
        probabilities = weights / weights.sum()

        def sample_stream():
            remaining = int(self.num_samples)
            chunk_size = min(1_000_000, max(1, remaining))
            while remaining > 0:
                draw = min(chunk_size, remaining)
                choices = np.random.choice(len(probabilities), size=draw, p=probabilities, replace=True)
                for choice in choices:
                    yield int(choice)
                remaining -= draw

        return sample_stream()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train PSICHIC+ on one large sequence/ligand affinity table with source-aware uncertainty control."
    )
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--valid-csv", type=Path)
    parser.add_argument("--test-csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--init-model-dir", type=Path, help="Optional model dir with config.json, degree.pt, and model.pt.")
    parser.add_argument("--config-json", type=Path)
    parser.add_argument("--degree-path", type=Path)
    parser.add_argument("--protein-cache-dir", type=Path, required=True)
    parser.add_argument("--ligand-cache-dir", type=Path, required=True)
    parser.add_argument("--build-missing-protein-cache", action="store_true")
    parser.add_argument("--id-column", default="ID")
    parser.add_argument("--protein-sequence-column", default="Protein")
    parser.add_argument("--protein-key-column")
    parser.add_argument("--protein-sequence-map-csv", type=Path)
    parser.add_argument("--protein-map-key-column", default="Protein")
    parser.add_argument("--protein-map-sequence-column", default="Sequence")
    parser.add_argument("--ligand-column", default="Ligand")
    parser.add_argument("--label-column", default="regression_label")
    parser.add_argument("--classification-label-column", default="classification_label")
    parser.add_argument("--source-column", default="source")
    parser.add_argument("--quality-column", default="quality")
    parser.add_argument("--sampling-weight-column")
    parser.add_argument("--source-priors-json", type=Path)
    parser.add_argument("--default-max-phi", type=float, default=24.0)
    parser.add_argument("--quality-min-phi", type=float, default=8.0)
    parser.add_argument("--quality-max-phi", type=float, default=80.0)
    parser.add_argument("--source-constraint-weight", type=float, default=1.0)
    parser.add_argument("--mincut-loss-weight", type=float, default=1.0)
    parser.add_argument("--orthogonality-loss-weight", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--total-steps", type=int, default=1_000_000)
    parser.add_argument("--eval-every", type=int, default=25_000)
    parser.add_argument("--save-every", type=int, default=25_000)
    parser.add_argument("--max-eval-batches", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--adam-eps", type=float, default=1e-5)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--pki-min", type=float, default=0.0)
    parser.add_argument("--pki-max", type=float, default=12.0)
    parser.add_argument("--classification-threshold", type=float, default=4.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=888)
    parser.add_argument("--no-standardize-ligands", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.ligand_cache_dir.mkdir(parents=True, exist_ok=True)
    args.protein_cache_dir.mkdir(parents=True, exist_ok=True)

    protein_sequence_map = load_protein_sequence_map(args)
    train_frame, train_sequences = load_training_frame(args.train_csv, args, protein_sequence_map)
    valid_frame, valid_sequences = load_optional_frame(args.valid_csv, args, protein_sequence_map)
    test_frame, test_sequences = load_optional_frame(args.test_csv, args, protein_sequence_map)
    all_sequences = {**train_sequences, **valid_sequences, **test_sequences}
    ensure_protein_cache(all_sequences, args.protein_cache_dir, args.build_missing_protein_cache)

    config = load_config(args)
    degree_dict = load_degree_dict(args)
    model = build_model(config, degree_dict, args.device)
    load_initial_state(model, args.init_model_dir, args.device)

    source_priors = build_source_priors(train_frame, args)
    save_run_contract(args, config, degree_dict, source_priors, train_frame, valid_frame, test_frame)
    train_loader = build_loader(train_frame, args, split="train", shuffle=True)
    valid_loader = build_loader(valid_frame, args, split="valid", shuffle=False) if valid_frame is not None else None
    test_loader = build_loader(test_frame, args, split="test", shuffle=False) if test_frame is not None else None
    del train_frame, valid_frame, test_frame

    trainer = SourceAwareUncertaintyTrainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        config=TrainerConfig(
            output_dir=args.output_dir,
            total_steps=args.total_steps,
            eval_every=args.eval_every,
            save_every=args.save_every,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            adam_eps=args.adam_eps,
            adam_betas=(args.adam_beta1, args.adam_beta2),
            grad_clip=args.grad_clip,
            pki_min=args.pki_min,
            pki_max=args.pki_max,
            classification_threshold=args.classification_threshold,
            source_priors=source_priors,
            default_source_prior={"max_phi": args.default_max_phi},
            source_constraint_weight=args.source_constraint_weight,
            mincut_loss_weight=args.mincut_loss_weight,
            orthogonality_loss_weight=args.orthogonality_loss_weight,
            max_eval_batches=args.max_eval_batches,
        ),
        device=args.device,
    )
    summary = trainer.train()
    write_json(args.output_dir / "run_complete.json", summary)


def load_protein_sequence_map(args: argparse.Namespace) -> dict[str, str] | None:
    if args.protein_sequence_map_csv is None:
        return None
    if not args.protein_sequence_map_csv.exists():
        raise FileNotFoundError(args.protein_sequence_map_csv)
    frame = pd.read_csv(args.protein_sequence_map_csv)
    require_columns(frame, [args.protein_map_key_column, args.protein_map_sequence_column])
    keys = frame[args.protein_map_key_column].astype(str)
    sequences = frame[args.protein_map_sequence_column].astype(str)
    return dict(zip(keys, sequences, strict=False))


def load_training_frame(
    path: Path,
    args: argparse.Namespace,
    protein_sequence_map: dict[str, str] | None,
) -> tuple[pd.DataFrame, dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    return prepare_frame(read_input_frame(path, args), args, protein_sequence_map, require_sampling_weight=True)


def read_input_frame(path: Path, args: argparse.Namespace) -> pd.DataFrame:
    header = set(pd.read_csv(path, nrows=0).columns)
    requested = {args.ligand_column, args.label_column}
    if args.id_column in header:
        requested.add(args.id_column)
    elif args.id_column == "ID" and "Activity_ID" in header:
        requested.add("Activity_ID")
    if args.protein_key_column:
        requested.add(args.protein_key_column)
        if args.protein_sequence_column in header:
            requested.add(args.protein_sequence_column)
    else:
        requested.add(args.protein_sequence_column)
    for optional in (
        args.classification_label_column,
        args.source_column,
        args.quality_column,
        args.sampling_weight_column,
    ):
        if optional and optional in header:
            requested.add(optional)
    return pd.read_csv(path, usecols=lambda column: column in requested)


def load_optional_frame(
    path: Path | None,
    args: argparse.Namespace,
    protein_sequence_map: dict[str, str] | None,
) -> tuple[pd.DataFrame | None, dict[str, str]]:
    if path is None:
        return None, {}
    frame = read_input_frame(path, args)
    frame, sequences = prepare_frame(frame, args, protein_sequence_map, require_sampling_weight=False)
    return frame, sequences


def prepare_frame(
    frame: pd.DataFrame,
    args: argparse.Namespace,
    protein_sequence_map: dict[str, str] | None,
    require_sampling_weight: bool,
) -> tuple[pd.DataFrame, dict[str, str]]:
    require_columns(frame, [args.ligand_column, args.label_column])
    working = pd.DataFrame()
    if args.id_column in frame.columns:
        working["ID"] = frame[args.id_column].astype(str)
    elif args.id_column == "ID" and "Activity_ID" in frame.columns:
        working["ID"] = frame["Activity_ID"].astype(str)
    else:
        working["ID"] = pd.Series(pd.NA, index=frame.index, dtype="string")
    if args.protein_key_column:
        require_columns(frame, [args.protein_key_column])
        protein_keys = frame[args.protein_key_column].astype(str)
        if args.protein_sequence_column in frame.columns:
            protein_sequences = frame[args.protein_sequence_column].astype(str)
            sequence_map = dict(zip(protein_keys, protein_sequences, strict=False))
        elif protein_sequence_map is not None:
            unique_keys = set(protein_keys)
            missing = sorted(key for key in unique_keys if key not in protein_sequence_map)
            if missing:
                preview = ", ".join(missing[:5])
                raise ValueError(
                    f"Protein sequence map is missing {len(missing)} protein keys. Examples: {preview}"
                )
            sequence_map = {key: protein_sequence_map[key] for key in unique_keys}
        else:
            raise ValueError(
                f"{args.protein_sequence_column!r} is not in the input frame. "
                "Provide --protein-sequence-map-csv when using --protein-key-column with key-only training data."
            )
    else:
        require_columns(frame, [args.protein_sequence_column])
        protein_sequences = frame[args.protein_sequence_column].astype(str)
        protein_keys = protein_sequences.map(sequence_key)
        sequence_map = dict(zip(protein_keys, protein_sequences, strict=False))
    working["Protein"] = protein_keys
    working["Ligand"] = frame[args.ligand_column].astype(str)
    working["regression_label"] = pd.to_numeric(frame[args.label_column], errors="coerce")
    if args.classification_label_column in frame.columns:
        working["classification_label"] = pd.to_numeric(frame[args.classification_label_column], errors="coerce")
    else:
        working["classification_label"] = np.nan
    if args.source_column in frame.columns:
        working["_source"] = frame[args.source_column].map(normal_source_label)
    else:
        working["_source"] = "unknown"
    if args.quality_column in frame.columns:
        working["_quality"] = frame[args.quality_column].map(quality_to_unit_interval)
    else:
        working["_quality"] = np.nan
    if args.sampling_weight_column and args.sampling_weight_column in frame.columns:
        require_columns(frame, [args.sampling_weight_column])
        working["_sample_weight"] = pd.to_numeric(frame[args.sampling_weight_column], errors="coerce").fillna(0.0)
    elif args.sampling_weight_column and require_sampling_weight:
        require_columns(frame, [args.sampling_weight_column])
    missing_labels = working["regression_label"].isna() & working["classification_label"].isna()
    if missing_labels.any():
        working = working.loc[~missing_labels].reset_index(drop=True)
    return working, sequence_map


def build_loader(frame: pd.DataFrame, args: argparse.Namespace, split: str, shuffle: bool) -> DataLoader:
    sampler = None
    effective_shuffle = shuffle
    if split == "train" and args.sampling_weight_column:
        if "_sample_weight" not in frame.columns:
            raise ValueError(f"Training frame is missing sampling weights from {args.sampling_weight_column}.")
        weights = torch.as_tensor(frame["_sample_weight"].to_numpy(dtype=float), dtype=torch.double)
        weights = torch.clamp(weights, min=0)
        if float(weights.sum()) <= 0:
            raise ValueError(f"Sampling weights in {args.sampling_weight_column} sum to zero.")
        sampler_samples = max(args.batch_size, args.batch_size * max(args.total_steps + 1, 1))
        sampler = CustomWeightedRandomSampler(weights, sampler_samples, replacement=True)
        effective_shuffle = False
    dataset = ProteinMoleculeDataset(
        frame,
        molecule_dict={},
        molecule_folder=str(args.ligand_cache_dir),
        protein_dict={},
        protein_folder=str(args.protein_cache_dir),
        result_path=str(args.output_dir),
        dataset_tag=split,
        source_data_column="_source",
        device=args.device,
        molecule_error_log=str(args.output_dir / f"{split}_molecule_errors.log"),
        standardize=not args.no_standardize_ligands,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=effective_shuffle,
        sampler=sampler,
        follow_batch=["mol_x", "clique_x", "prot_node_aa"],
        num_workers=args.num_workers,
    )


def ensure_protein_cache(sequence_map: dict[str, str], cache_dir: Path, build_missing: bool) -> None:
    missing = [(key, seq) for key, seq in sequence_map.items() if not (cache_dir / f"{key}.pt").exists()]
    if not missing:
        return
    if not build_missing:
        preview = ", ".join(key for key, _ in missing[:5])
        raise FileNotFoundError(
            f"Protein cache is missing {len(missing)} graphs under {cache_dir}. "
            f"Examples: {preview}. Re-run with --build-missing-protein-cache to create them."
        )
    keys = [key for key, _ in missing]
    seqs = [seq for _, seq in missing]
    save_protein_cache_with_keys(keys, seqs, cache_dir)


def build_source_priors(frame: pd.DataFrame, args: argparse.Namespace) -> dict[str, dict[str, float]] | None:
    if args.source_priors_json:
        with args.source_priors_json.open() as handle:
            raw = json.load(handle)
        return {str(key): {"max_phi": float(value["max_phi"])} for key, value in raw.items()}
    if args.source_column not in frame.columns and "_source" not in frame.columns:
        return None
    priors: dict[str, dict[str, float]] = {}
    for source, group in frame.groupby("_source", dropna=False):
        quality = pd.to_numeric(group["_quality"], errors="coerce").dropna()
        if quality.empty:
            max_phi = args.default_max_phi
        else:
            q = float(quality.clip(0.0, 1.0).mean())
            max_phi = args.quality_min_phi + q * (args.quality_max_phi - args.quality_min_phi)
        priors[str(source)] = {"max_phi": float(max_phi)}
    priors.setdefault("unknown", {"max_phi": args.default_max_phi})
    return priors


def load_config(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config_json
    if config_path is None and args.init_model_dir is not None:
        candidate = args.init_model_dir / "config.json"
        config_path = candidate if candidate.exists() else None
    if config_path is None:
        return DEFAULT_CONFIG
    with config_path.open() as handle:
        return json.load(handle)


def load_degree_dict(args: argparse.Namespace) -> dict[str, torch.Tensor]:
    degree_path = args.degree_path
    if degree_path is None and args.init_model_dir is not None:
        candidate = args.init_model_dir / "degree.pt"
        degree_path = candidate if candidate.exists() else None
    if degree_path is None:
        raise ValueError("A degree.pt source is required via --degree-path or --init-model-dir.")
    return torch.load(degree_path, map_location=args.device)


def build_model(config: dict[str, Any], degree_dict: dict[str, torch.Tensor], device: str) -> torch.nn.Module:
    params = config["params"]
    model = net(
        degree_dict["ligand_deg"],
        degree_dict["protein_deg"],
        mol_in_channels=params["mol_in_channels"],
        mol_edge_channels=params["mol_edge_channels"],
        clique_pe_walk_length=params["clique_pe_walk_length"],
        prot_in_channels=params["prot_in_channels"],
        prot_evo_channels=params["prot_evo_channels"],
        hidden_channels=params["hidden_channels"],
        pre_layers=params["pre_layers"],
        post_layers=params["post_layers"],
        aggregators=params["aggregators"],
        scalers=params["scalers"],
        total_layer=params["total_layer"],
        K=params["K"],
        heads=params["heads"],
        dropout=params["dropout"],
        dropout_attn_score=params["dropout_attn_score"],
        device=device,
    ).to(device)
    model.reset_parameters()
    return model


def load_initial_state(model: torch.nn.Module, init_model_dir: Path | None, device: str) -> None:
    if init_model_dir is None:
        return
    model_path = init_model_dir / "model.pt"
    if not model_path.exists():
        return
    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=False)


def save_run_contract(
    args: argparse.Namespace,
    config: dict[str, Any],
    degree_dict: dict[str, torch.Tensor],
    source_priors: dict[str, dict[str, float]] | None,
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame | None,
    test_frame: pd.DataFrame | None,
) -> None:
    with (args.output_dir / "config.json").open("w") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
    torch.save(degree_dict, args.output_dir / "degree.pt")
    if args.init_model_dir is not None:
        for filename in ("model.pt",):
            src = args.init_model_dir / filename
            if src.exists():
                shutil.copy2(src, args.output_dir / f"initial_{filename}")
    payload = {
        "train_csv": str(args.train_csv),
        "valid_csv": str(args.valid_csv) if args.valid_csv else None,
        "test_csv": str(args.test_csv) if args.test_csv else None,
        "output_dir": str(args.output_dir),
        "init_model_dir": str(args.init_model_dir) if args.init_model_dir else None,
        "protein_cache_dir": str(args.protein_cache_dir),
        "ligand_cache_dir": str(args.ligand_cache_dir),
        "train_rows": int(len(train_frame)),
        "valid_rows": int(len(valid_frame)) if valid_frame is not None else None,
        "test_rows": int(len(test_frame)) if test_frame is not None else None,
        "source_priors": source_priors,
        "loss_weights": {
            "source_constraint": args.source_constraint_weight,
            "mincut": args.mincut_loss_weight,
            "orthogonality": args.orthogonality_loss_weight,
        },
        "columns": {
            "id": args.id_column,
            "protein_sequence": args.protein_sequence_column,
            "protein_key": args.protein_key_column,
            "protein_sequence_map_csv": str(args.protein_sequence_map_csv) if args.protein_sequence_map_csv else None,
            "protein_map_key": args.protein_map_key_column,
            "protein_map_sequence": args.protein_map_sequence_column,
            "ligand": args.ligand_column,
            "label": args.label_column,
            "source": args.source_column,
            "quality": args.quality_column,
            "sampling_weight": args.sampling_weight_column,
        },
    }
    write_json(args.output_dir / "run_contract.json", payload)


def require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def sequence_key(sequence: str) -> str:
    digest = hashlib.sha256(sequence.encode()).hexdigest()[:24]
    return f"seq_{digest}"


def normal_source_label(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "unknown"
    label = str(value).strip()
    if label.lower() in {"", "nan", "none", "<na>"}:
        return "unknown"
    return label


def quality_to_unit_interval(value: Any) -> float:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return float("nan")
    if isinstance(value, str):
        text = value.strip().lower().replace("-", "_").replace(" ", "_")
        named = {
            "very_low": 0.05,
            "low": 0.2,
            "inactive": 0.15,
            "hts_inactive": 0.2,
            "random_inactive": 0.2,
            "generated_inactive": 0.25,
            "crosstarget_inactive": 0.35,
            "weak_binder": 0.35,
            "medium": 0.5,
            "moderate": 0.5,
            "moderate_binder": 0.6,
            "binder": 0.65,
            "high": 0.8,
            "strong_binder": 0.85,
            "very_strong_binder": 0.95,
            "very_high": 1.0,
            "primary": 1.0,
            "curated": 1.0,
            "secondary": 0.5,
            "inferred": 0.2,
        }
        if text in named:
            return named[text]
        try:
            value = float(text)
        except ValueError:
            return float("nan")
    numeric = float(value)
    if numeric > 1.0:
        numeric = numeric / 100.0
    return min(max(numeric, 0.0), 1.0)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()

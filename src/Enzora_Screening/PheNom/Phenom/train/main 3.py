from __future__ import annotations

import argparse
import json
import random
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

from models.enzyme_prediction import TASK_NAMES
from models.net import net
from train.trainers import PHENOMETrainer, TrainerConfig, write_json
from utils.dataloader import DataLoader
from utils.dataset import ProteinMoleculeDataset


DEFAULT_CONFIG: dict[str, Any] = {
    "model": "PHENOME",
    "task_names": list(TASK_NAMES),
    "base_outputs": ["log10_kcat", "log10_km", "ph", "esp", "fitness"],
    "params": {
        "mol_in_channels": 61,
        "mol_edge_channels": 30,
        "clique_pe_walk_length": 20,
        "prot_in_channels": 33,
        "prot_evo_channels": 1024,
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
        "sigma_epsilon": 1e-6,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PHENOME on six enzyme prediction tasks.")
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--valid-csv", type=Path)
    parser.add_argument("--test-csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--init-model-dir", type=Path)
    parser.add_argument("--config-json", type=Path)
    parser.add_argument("--degree-path", type=Path)
    parser.add_argument("--protein-cache-dir", type=Path, required=True)
    parser.add_argument("--ligand-cache-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--total-steps", type=int, default=1_000_000)
    parser.add_argument("--eval-every", type=int, default=25_000)
    parser.add_argument("--save-every", type=int, default=25_000)
    parser.add_argument("--max-eval-batches", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--mincut-loss-weight", type=float, default=1.0)
    parser.add_argument("--orthogonality-loss-weight", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=888)
    parser.add_argument("--no-standardize-ligands", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.ligand_cache_dir.mkdir(parents=True, exist_ok=True)

    frames = [_read_frame(args.train_csv)]
    frames.append(_read_frame(args.valid_csv) if args.valid_csv else None)
    frames.append(_read_frame(args.test_csv) if args.test_csv else None)
    _require_protein_caches(frames, args.protein_cache_dir)

    config = _load_config(args.config_json)
    degree = _load_degree(args.degree_path, args.init_model_dir, args.device)
    model = _build_model(config, degree, args.device)
    loaded = _load_initial_state(model, args.init_model_dir, args.device)

    loaders = [
        _loader(frame, args, shuffle=index == 0) if frame is not None else None
        for index, frame in enumerate(frames)
    ]
    with (args.output_dir / "config.json").open("w") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
    torch.save(degree, args.output_dir / "degree.pt")
    write_json(args.output_dir / "run_contract.json", {
        "model": "PHENOME",
        "train_csv": str(args.train_csv),
        "valid_csv": str(args.valid_csv) if args.valid_csv else None,
        "test_csv": str(args.test_csv) if args.test_csv else None,
        "initialized_backbone_tensors": loaded,
        "task_names": list(TASK_NAMES),
    })

    trainer = PHENOMETrainer(
        model,
        loaders[0],
        loaders[1],
        loaders[2],
        TrainerConfig(
            output_dir=args.output_dir,
            total_steps=args.total_steps,
            eval_every=args.eval_every,
            save_every=args.save_every,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            grad_clip=args.grad_clip,
            mincut_loss_weight=args.mincut_loss_weight,
            orthogonality_loss_weight=args.orthogonality_loss_weight,
            max_eval_batches=args.max_eval_batches,
        ),
        args.device,
    )
    trainer.train()


def _read_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = ["Protein", "task_index", "label", "is_binary"]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    return frame


def _require_protein_caches(frames: list[pd.DataFrame | None], cache_dir: Path) -> None:
    keys = set()
    for frame in frames:
        if frame is None:
            continue
        keys.update(frame["Protein"].astype(str))
        if "WT_Protein" in frame:
            keys.update(frame["WT_Protein"].dropna().astype(str))
    missing = sorted(key for key in keys if not (cache_dir / f"{key}.pt").exists())
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} protein caches; first IDs: {missing[:10]}")


def _loader(frame: pd.DataFrame, args: argparse.Namespace, shuffle: bool) -> DataLoader:
    dataset = ProteinMoleculeDataset(
        frame,
        molecule_folder=args.ligand_cache_dir,
        protein_folder=args.protein_cache_dir,
        standardize=not args.no_standardize_ligands,
    )
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers)


def _load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return DEFAULT_CONFIG
    with path.open() as handle:
        config = json.load(handle)
    config["model"] = "PHENOME"
    config["task_names"] = list(TASK_NAMES)
    config["base_outputs"] = DEFAULT_CONFIG["base_outputs"]
    config["params"]["prot_evo_channels"] = 1024
    config["params"]["sigma_epsilon"] = 1e-6
    return config


def _load_degree(path: Path | None, init_dir: Path | None, device: str) -> dict[str, torch.Tensor]:
    degree_path = path or (init_dir / "degree.pt" if init_dir else None)
    if degree_path is None:
        raise ValueError("Provide --degree-path or --init-model-dir")
    return torch.load(degree_path, map_location=device)


def _build_model(config: dict[str, Any], degree: dict[str, torch.Tensor], device: str) -> net:
    params = config["params"]
    return net(
        degree["ligand_deg"], degree["protein_deg"],
        **{key: params[key] for key in (
            "mol_in_channels", "mol_edge_channels", "clique_pe_walk_length", "prot_in_channels",
            "prot_evo_channels", "hidden_channels", "pre_layers", "post_layers", "aggregators", "scalers",
            "total_layer", "K", "heads", "dropout", "dropout_attn_score",
        )},
        device=device,
    ).to(device)


def _load_initial_state(model: net, init_dir: Path | None, device: str) -> int:
    if init_dir is None:
        return 0
    state = torch.load(init_dir / "model.pt", map_location=device)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    current = model.state_dict()
    compatible = {key: value for key, value in state.items() if key in current and current[key].shape == value.shape}
    skipped = set(state).difference(compatible)
    if not all(key.startswith(("mu_out.", "sigma_out.")) for key in skipped):
        raise RuntimeError(f"Unexpected incompatible checkpoint keys: {sorted(skipped)}")
    if len(state) == 492 and len(compatible) != 484:
        raise RuntimeError(f"Expected 484 reusable backbone tensors, found {len(compatible)}")
    model.load_state_dict(compatible, strict=False)
    return len(compatible)


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()

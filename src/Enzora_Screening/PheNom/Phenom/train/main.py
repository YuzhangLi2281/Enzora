#!/usr/bin/env python3
"""Canonical PHENOM six-task training entry point."""

from __future__ import annotations

import argparse
import gc
import csv
import json
import math
import os
import random
import signal
import subprocess
import sys
from collections import Counter, defaultdict
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

CODEBASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODEBASE_DIR))

from models.enzyme_prediction import (  # noqa: E402
    MUTATION,
    TASK_NAMES,
)
from models.net import net  # noqa: E402
from train.trainers import (  # noqa: E402
    PHENOMETrainer,
    PROBABILITY_EPSILON,
    SIGMA_FLOOR,
    TrainerConfig,
    gaussian_nll_loss,
    gaussian_probability_above_zero,
    mutation_distribution,
    select_task_distribution,
)
from utils.dataloader import DataLoader as EvaluationDataLoader  # noqa: E402
from utils.dataloader import collate_enzyme_samples  # noqa: E402
from utils.dataset import ProteinMoleculeDataset  # noqa: E402


BASE_OUTPUTS = (
    ("01_log10_kcat", "mu_log10_kcat", 0),
    ("02_log10_km", "mu_log10_km", 1),
    ("03_ph", "mu_ph", 2),
    ("04_esp", "mu_esp", 3),
    ("05_fitness", "mu_fitness", 4),
)


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
        "sigma_epsilon": SIGMA_FLOOR,
        "prediction_head_layout": "unified",
    },
}


Group = tuple[int, bool, bool]


class TemperatureTaskBatchSampler(torch.utils.data.Sampler[list[int]]):
    """Sample tasks by dataset size, then temperature-balance input modes within each task."""

    def __init__(
        self,
        groups: Sequence[Group],
        cluster_ids: Sequence[str],
        batch_size: int,
        temperature: float = 0.5,
        task_temperature: float = 1.0,
        seed: int = 888,
        batches_per_epoch: int | None = None,
    ) -> None:
        if len(groups) != len(cluster_ids):
            raise ValueError("groups and cluster_ids must have equal length")
        if not 0 <= temperature <= 1:
            raise ValueError("temperature must be in [0, 1]")
        self.groups = list(groups)
        self.cluster_ids = [str(value) for value in cluster_ids]
        self.batch_size = int(batch_size)
        self.temperature = float(temperature)
        self.task_temperature = float(task_temperature)
        self.seed = int(seed)
        self.epoch = 0
        self.group_indices: dict[Group, list[int]] = defaultdict(list)
        for index, group in enumerate(self.groups):
            self.group_indices[group].append(index)
        requested_batches = batches_per_epoch or sum(
            math.ceil(len(indices) / self.batch_size) for indices in self.group_indices.values()
        )
        self.batches_per_epoch = requested_batches

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.batches_per_epoch

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        groups = sorted(self.group_indices)
        task_groups: dict[int, list[Group]] = defaultdict(list)
        for group in groups:
            task_groups[group[0]].append(group)
        tasks = sorted(task_groups)
        task_sizes = {task: sum(len(self.group_indices[group]) for group in task_groups[task]) for task in tasks}
        task_schedule = tasks + rng.choices(
            tasks,
            weights=[task_sizes[task] ** self.task_temperature for task in tasks],
            k=self.batches_per_epoch - len(tasks),
        )
        rng.shuffle(task_schedule)
        schedule = [
            rng.choices(
                task_groups[task],
                weights=[len(self.group_indices[group]) ** self.temperature for group in task_groups[task]],
            )[0]
            for task in task_schedule
        ]

        flat_pools: dict[Group, list[int]] = {}
        flat_offsets: dict[Group, int] = {}
        esp_clusters: dict[Group, dict[str, list[int]]] = {}
        esp_cluster_order: dict[Group, list[str]] = {}
        esp_cluster_offset: dict[Group, int] = {}
        esp_row_offsets: dict[tuple[Group, str], int] = {}

        for group, indices in self.group_indices.items():
            if group[0] == 4:
                clusters: dict[str, list[int]] = defaultdict(list)
                for index in indices:
                    clusters[self.cluster_ids[index]].append(index)
                for rows in clusters.values():
                    rng.shuffle(rows)
                order = list(clusters)
                rng.shuffle(order)
                esp_clusters[group] = clusters
                esp_cluster_order[group] = order
                esp_cluster_offset[group] = 0
            else:
                pool = indices.copy()
                rng.shuffle(pool)
                flat_pools[group] = pool
                flat_offsets[group] = 0

        for group in schedule:
            if group[0] == 4:
                order = esp_cluster_order[group]
                if not order:
                    raise RuntimeError("ESP group has no clusters")
                batch = []
                used: set[str] = set()
                attempts = 0
                while len(batch) < self.batch_size and attempts < max(self.batch_size * 4, len(order) * 2):
                    offset = esp_cluster_offset[group]
                    cluster = order[offset % len(order)]
                    esp_cluster_offset[group] = offset + 1
                    attempts += 1
                    if cluster in used and len(used) < len(order):
                        continue
                    rows = esp_clusters[group][cluster]
                    row_key = (group, cluster)
                    row_offset = esp_row_offsets.get(row_key, 0)
                    if row_offset and row_offset % len(rows) == 0:
                        rng.shuffle(rows)
                    batch.append(rows[row_offset % len(rows)])
                    esp_row_offsets[row_key] = row_offset + 1
                    used.add(cluster)
                yield batch
                continue

            pool = flat_pools[group]
            offset = flat_offsets[group]
            batch = []
            for _ in range(self.batch_size):
                if offset and offset % len(pool) == 0:
                    rng.shuffle(pool)
                batch.append(pool[offset % len(pool)])
                offset += 1
            flat_offsets[group] = offset
            yield batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--init-model-dir", type=Path)
    parser.add_argument("--degree-path", type=Path)
    parser.add_argument("--from-scratch", action="store_true")
    parser.add_argument("--protein-cache-dir", type=Path, required=True)
    parser.add_argument("--ligand-cache-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--batch-tune-csv", type=Path)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--total-steps", type=int, default=1_000_000)
    parser.add_argument("--eval-every", type=int, default=25_000)
    parser.add_argument("--save-every", type=int, default=10_000)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--task-temperature", type=float, default=1.0)
    parser.add_argument("--mincut-loss-weight", type=float, default=1.0)
    parser.add_argument("--orthogonality-loss-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=888)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--evaluate-checkpoints", type=Path, nargs="+")
    parser.add_argument("--exclude-task", choices=TASK_NAMES, action="append", default=[])
    parser.add_argument("--prediction-head-layout", choices=("unified", "task_independent"), default="unified")
    parser.add_argument("--wandb-project", default="PHENOM")
    parser.add_argument("--wandb-name")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default=None)
    parser.add_argument("--debug-steps", type=int)
    return parser.parse_args()


def model_dataset(frame: pd.DataFrame, args: argparse.Namespace) -> ProteinMoleculeDataset:
    return ProteinMoleculeDataset(
        frame,
        molecule_folder=args.ligand_cache_dir,
        protein_folder=args.protein_cache_dir,
        protein_embedding_dim=args.protein_embedding_dim,
        standardize=True,
    )


def training_loader(dataset: ProteinMoleculeDataset, args: argparse.Namespace):
    clusters = dataset.frame.get("cluster_id", dataset.frame["Protein"]).fillna(dataset.frame["Protein"]).astype(str)
    sampler = TemperatureTaskBatchSampler(
        dataset.batch_groups,
        clusters.tolist(),
        batch_size=args.batch_size,
        temperature=args.temperature,
        task_temperature=args.task_temperature,
        seed=args.seed,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_enzyme_samples,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    return loader, sampler


def as_list(value, size: int, default="") -> list[str]:
    if value is None:
        return [default] * size
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)] * size


def evaluate_and_save(
    model, loader, device: torch.device, output_dir: Path, step: int, exclude_tasks: Sequence[str]
) -> dict:
    model.eval()
    numeric = {key: [] for key in (
        "target", "prediction", "selected_mu", "selected_sigma", "is_binary", "task_index", "base_mu", "base_sigma"
    )}
    metadata = {key: [] for key in ("ID", "Protein", "WT_Protein", "Ligand", "source")}
    losses = []
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        for batch in loader:
            primary = batch["primary"].to(device)
            mu, sigma, ortho, mincut, _ = model(
                primary, protein_only=batch["protein_only"], all_task_outputs=True
            )
            task = int(primary.task_index.view(-1)[0])
            task_index = torch.full((mu.size(0),), task, dtype=torch.long, device=device)
            target = primary.label.view(-1)
            is_binary = primary.is_binary.view(-1)
            wt_keys = [""] * target.numel()
            if task == MUTATION:
                reference = batch["reference"].to(device)
                wt_mu, wt_sigma, wt_ortho, wt_mincut, _ = model(reference, protein_only=batch["protein_only"])
                selected_mu, selected_sigma = mutation_distribution(mu, sigma, wt_mu, wt_sigma)
                ortho = 0.5 * (ortho + wt_ortho)
                mincut = 0.5 * (mincut + wt_mincut)
                wt_keys = as_list(reference.prot_key, target.numel())
            else:
                selected_mu, selected_sigma = select_task_distribution(mu, sigma, task_index)
            if bool(is_binary[0]):
                with torch.autocast(device_type=device.type, enabled=False):
                    prediction = gaussian_probability_above_zero(selected_mu, selected_sigma)
                    supervised = F.binary_cross_entropy(prediction, target.double())
            else:
                prediction = selected_mu
                with torch.autocast(device_type=device.type, enabled=False):
                    supervised = gaussian_nll_loss(
                        target,
                        selected_mu,
                        selected_sigma,
                        primary.sample_weight.view(-1),
                    )
            evaluation_loss = supervised + ortho.float().mean() + mincut.float().mean()
            if not torch.isfinite(evaluation_loss):
                raise FloatingPointError(f"Non-finite evaluation loss: task={int(task_index[0])}")
            losses.append(float(evaluation_loss.cpu()))
            for key, tensor in {
                "target": target,
                "prediction": prediction,
                "selected_mu": selected_mu,
                "selected_sigma": selected_sigma,
                "is_binary": is_binary,
                "task_index": task_index,
                "base_mu": mu,
                "base_sigma": sigma,
            }.items():
                numeric[key].append(tensor.detach().float().cpu().numpy())
            size = target.numel()
            metadata["ID"].extend(as_list(primary.id, size))
            metadata["Protein"].extend(as_list(primary.prot_key, size))
            metadata["WT_Protein"].extend(wt_keys)
            metadata["Ligand"].extend(as_list(getattr(primary, "mol_key", None), size))
            metadata["source"].extend(as_list(primary.source, size))
    arrays = {key: np.concatenate(values) for key, values in numeric.items()}
    evaluation = output_dir / "evaluation"
    evaluation.mkdir(parents=True, exist_ok=True)
    common = pd.DataFrame({
        **metadata,
        "task_index": arrays["task_index"],
        "task": [TASK_NAMES[int(index)] for index in arrays["task_index"]],
        "target": arrays["target"],
        "is_binary": arrays["is_binary"],
        "selected_prediction": arrays["prediction"],
        "selected_mu": arrays["selected_mu"],
        "selected_sigma": arrays["selected_sigma"],
    })
    prediction_paths = {}
    for folder, column, dimension in BASE_OUTPUTS:
        destination = evaluation / folder
        destination.mkdir(exist_ok=True)
        frame = common.copy()
        frame[column] = arrays["base_mu"][:, dimension]
        frame[column.replace("mu_", "sigma_")] = arrays["base_sigma"][:, dimension]
        prediction_path = destination / f"raw_predictions_step_{step:09d}.csv"
        frame.to_csv(prediction_path, index=False)
        prediction_paths[folder] = prediction_path

    model_name = f"{output_dir.name}_step_{step:09d}"
    evaluation_code = CODEBASE_DIR / "Phenome_Evaluation"
    benchmark_root = CODEBASE_DIR.parent / "data" / "Test"
    protocol_tasks = (
        ("kinetic", "01_log10_kcat", benchmark_root / "kinetic_prediction" / "test.csv", {"kcat", "km", "kcat_km"}),
        ("ph", "03_ph", benchmark_root / "ph_prediction" / "test.csv", {"ph"}),
        ("mutation_effect", "05_fitness", benchmark_root / "mutation_effect" / "test.csv", {"mutation"}),
        ("esp", "04_esp", benchmark_root / "esp_prediction" / "test.csv", {"esp"}),
    )
    protocol_metrics = {}
    for task, folder, test_set, task_names in protocol_tasks:
        command = [
                sys.executable,
                str(evaluation_code / "phenome_evaluation.py"),
                "--task", task,
                "--model", model_name,
                "--input-format", "phenome",
                "--predictions", str(prediction_paths[folder]),
                "--test-set", str(test_set),
            ]
        for excluded_task in set(exclude_tasks).intersection(task_names):
            command.extend(("--exclude-task", excluded_task))
        subprocess.run(
            command,
            cwd=evaluation_code,
            check=True,
        )
        protocol_metrics[task] = yaml.safe_load(
            (evaluation_code / "Results" / task / model_name / "metrics.yaml").read_text()
        )
    model.train()
    return {"loss": float(np.mean(losses)), "model": model_name, "tasks": protocol_metrics}


def atomic_save(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def read_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    required = ["Protein", "task_index", "label", "is_binary"]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    return frame


def load_config() -> dict[str, Any]:
    return DEFAULT_CONFIG


def load_degree(path: Path, device: str) -> dict[str, torch.Tensor]:
    return torch.load(path, map_location=device)


def build_model(config: dict[str, Any], degree: dict[str, torch.Tensor], device: str) -> net:
    params = config["params"]
    model = net(
        degree["ligand_deg"], degree["protein_deg"],
        **{key: params[key] for key in (
            "mol_in_channels", "mol_edge_channels", "clique_pe_walk_length", "prot_in_channels",
            "prot_evo_channels", "hidden_channels", "pre_layers", "post_layers", "aggregators", "scalers",
            "total_layer", "K", "heads", "dropout", "dropout_attn_score",
            "prediction_head_layout",
        )},
        device=device,
    ).to(device)
    model.prediction_head.eps = params["sigma_epsilon"]
    return model


def load_initial_state(model: net, init_dir: Path, device: str) -> int:
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


def seed(value: int) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def tune_batch_size(
    frame: pd.DataFrame,
    args: argparse.Namespace,
    config: dict[str, Any],
    degree: dict[str, torch.Tensor],
) -> int:
    model = build_model(config, degree, args.device)
    if not args.from_scratch:
        load_initial_state(model, args.init_model_dir, args.device)
    model.train()
    trainer = PHENOMETrainer(model, TrainerConfig(output_dir=args.output_dir, total_steps=1), args.device)
    candidates = [64, 48, 32, 24, 16, 12, 8, 4, 2, 1]
    grouped = frame.assign(has_ligand=frame["Ligand"].notna())

    for candidate_index, batch_size in enumerate(candidates):
        print(f"testing_batch_size={batch_size}", flush=True)
        try:
            for _, group in grouped.groupby(["task_index", "is_binary", "has_ligand"], sort=True):
                row = group.iloc[0].drop(labels="has_ligand").to_dict()
                dataset = model_dataset(pd.DataFrame([row] * batch_size), args)
                loader = EvaluationDataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
                batch = next(iter(loader))
                trainer.optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.device.startswith("cuda")):
                    loss, _, _ = trainer.batch_loss(batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip, error_if_nonfinite=True
                )
        except torch.cuda.OutOfMemoryError:
            trainer.optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            gc.collect()
            print(f"batch_size_oom={batch_size}", flush=True)
            continue

        recommended = candidates[min(candidate_index + 1, len(candidates) - 1)]
        (args.output_dir / "recommended_batch_size.txt").write_text(f"{recommended}\n")
        del trainer, model
        torch.cuda.empty_cache()
        gc.collect()
        print(f"maximum_tested_batch_size={batch_size}", flush=True)
        print(f"recommended_batch_size={recommended}", flush=True)
        return recommended

    raise RuntimeError("Batch size 1 exceeded available GPU memory")


def main() -> None:
    args = parse_args()
    seed(args.seed)
    sample_key = str(
        pd.read_csv(
            args.test_csv if args.evaluate_checkpoints else args.train_csv,
            usecols=["Protein"],
            nrows=1,
        )["Protein"].iat[0]
    )
    sample_cache = torch.load(args.protein_cache_dir / f"{sample_key}.pt", map_location="cpu")
    args.protein_embedding_dim = int(sample_cache["token_representation"].shape[1])
    config = load_config()
    config["task_names"] = [task for task in TASK_NAMES if task not in args.exclude_task]
    config["params"]["prot_evo_channels"] = args.protein_embedding_dim
    config["params"]["prediction_head_layout"] = args.prediction_head_layout
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.ligand_cache_dir.mkdir(parents=True, exist_ok=True)
    degree_path = args.degree_path or args.init_model_dir / "degree.pt"
    for folder, _, _ in BASE_OUTPUTS:
        (args.output_dir / "evaluation" / folder).mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    if args.evaluate_checkpoints:
        excluded_indices = [TASK_NAMES.index(task) for task in args.exclude_task]
        test_frame = read_frame(args.test_csv)
        test_frame = test_frame[~test_frame["task_index"].isin(excluded_indices)]
        device = torch.device(args.device)
        degree = load_degree(degree_path, args.device)
        model = build_model(config, degree, args.device)
        test_loader = EvaluationDataLoader(
            model_dataset(test_frame, args),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
        )
        sweep = {}
        for checkpoint_path in args.evaluate_checkpoints:
            state = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(state["model_state_dict"])
            step = int(state["step"])
            sweep[str(step)] = evaluate_and_save(
                model, test_loader, device, args.output_dir, step, args.exclude_task
            )
        (args.output_dir / "checkpoint_sweep.yaml").write_text(
            yaml.safe_dump(sweep, sort_keys=True, allow_unicode=True)
        )
        return

    excluded_indices = [TASK_NAMES.index(task) for task in args.exclude_task]
    train_frame, test_frame = read_frame(args.train_csv), read_frame(args.test_csv)
    train_frame = train_frame[~train_frame["task_index"].isin(excluded_indices)]
    test_frame = test_frame[~test_frame["task_index"].isin(excluded_indices)]
    active_tasks = tuple(TASK_NAMES[index] for index in sorted(train_frame["task_index"].unique()))
    device = torch.device(args.device)
    degree = load_degree(degree_path, args.device)
    if args.batch_size == 0:
        tune_frame = read_frame(args.batch_tune_csv)
        tune_frame = tune_frame[~tune_frame["task_index"].isin(excluded_indices)]
        args.batch_size = tune_batch_size(tune_frame, args, config, degree)
        seed(args.seed)
    model = build_model(config, degree, args.device)
    initialized = 0 if args.from_scratch else load_initial_state(model, args.init_model_dir, args.device)
    train_dataset, test_dataset = model_dataset(train_frame, args), model_dataset(test_frame, args)
    train_loader, sampler = training_loader(train_dataset, args)
    test_loader = EvaluationDataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    core = PHENOMETrainer(
        model,
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
        ),
        args.device,
    )
    optimizer = core.optimizer
    start_step, epoch = 0, 0
    task_step_counts = Counter()
    resume = args.resume or (args.output_dir / "latest_checkpoint.pt")
    if resume.exists():
        state = torch.load(resume, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        start_step, epoch = int(state["step"]), int(state.get("sampler_epoch", 0))
        task_step_counts.update(state["task_step_counts"])
        random.setstate(state["python_rng_state"])
        np.random.set_state(state["numpy_rng_state"])
        torch.set_rng_state(state["torch_rng_state"].cpu())
        if torch.cuda.is_available() and state.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng_state"]])

    run_contract = {
        "train_csv": str(args.train_csv), "test_csv": str(args.test_csv),
        "protein_cache_dir": str(args.protein_cache_dir), "ligand_cache_dir": str(args.ligand_cache_dir),
        "protein_encoder": sample_cache["encoder"],
        "protein_embedding_dim": args.protein_embedding_dim,
        "protein_contact_encoder": sample_cache["contact_encoder"],
        "tasks": list(active_tasks), "five_base_output_folders": [item[0] for item in BASE_OUTPUTS],
        "excluded_tasks": args.exclude_task,
        "prediction_head_layout": args.prediction_head_layout,
        "eval_every": args.eval_every, "task_sampling": "power_law_by_training_rows",
        "task_sampling_temperature": args.task_temperature,
        "within_task_temperature": args.temperature,
        "initialized_backbone_tensors": initialized, "resume_step": start_step,
        "from_scratch": args.from_scratch, "degree_path": str(degree_path),
        "batch_size": args.batch_size, "sigma_floor": SIGMA_FLOOR,
        "learning_rate": args.learning_rate, "warmup_steps": args.warmup_steps,
        "probability_epsilon": PROBABILITY_EPSILON, "loss_precision": "float64",
        "evaluation_protocol": {
            "implementation": "Phenome_Evaluation/phenome_evaluation.py",
            "results_dir": "Phenome_Evaluation/Results/<task>/<run>_step_<step>",
            "benchmarks": {
                "kinetic": "../data/Test/kinetic_prediction/test.csv",
                "ph": "../data/Test/ph_prediction/test.csv",
                "mutation_effect": "../data/Test/mutation_effect/test.csv",
                "esp": "../data/Test/esp_prediction/test.csv",
            },
        },
        "graph_regularization": {
            "protein": "learned residue clusters",
            "ligand": "learned atom clusters",
            "combination": "mean for ligand-conditioned batches; protein only otherwise",
            "mincut_weight": args.mincut_loss_weight,
            "orthogonality_weight": args.orthogonality_loss_weight,
        },
    }
    (args.output_dir / "run_contract.json").write_text(json.dumps(run_contract, indent=2) + "\n")
    torch.save(degree, args.output_dir / "degree.pt")

    import wandb
    mode = args.wandb_mode or ("online" if os.environ.get("WANDB_API_KEY") else "offline")
    run_id_path = args.output_dir / "wandb_run_id.txt"
    if run_id_path.exists():
        wandb_run_id = run_id_path.read_text().strip()
    else:
        wandb_run_id = wandb.util.generate_id()
        temporary_run_id_path = run_id_path.with_suffix(".txt.tmp")
        temporary_run_id_path.write_text(wandb_run_id + "\n")
        os.replace(temporary_run_id_path, run_id_path)
    wandb_config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        id=wandb_run_id,
        dir=str(args.output_dir),
        mode=mode,
        config=wandb_config,
        resume="allow",
    )
    stop_requested = False

    def request_stop(signum, _frame):
        nonlocal stop_requested
        print(f"Received signal {signum}; checkpointing after current batch", flush=True)
        stop_requested = True

    signal.signal(signal.SIGUSR1, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    metrics_path = args.output_dir / "training_metrics.csv"
    if start_step == 0 or not metrics_path.exists():
        with metrics_path.open("w", newline="") as handle:
            csv.writer(handle).writerow([
                "step", "task", "loss", "supervised_loss", "orthogonality_loss", "mincut_loss", "grad_norm"
            ])

    total_steps = args.debug_steps or args.total_steps
    sampler.set_epoch(epoch)
    iterator = iter(train_loader)
    model.train()

    def checkpoint(step: int, metrics: dict) -> None:
        payload = {
            "step": step,
            "sampler_epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "task_step_counts": dict(task_step_counts),
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        atomic_save(payload, args.output_dir / "latest_checkpoint.pt")
        if step % args.save_every == 0:
            atomic_save(payload, args.output_dir / f"checkpoint_{step:09d}.pt")

    last_metrics = {}
    for step in range(start_step + 1, total_steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            epoch += 1
            sampler.set_epoch(epoch)
            iterator = iter(train_loader)
            batch = next(iterator)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            loss, last_metrics, _ = core.batch_loss(batch)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.grad_clip, error_if_nonfinite=True
        )
        learning_rate = args.learning_rate
        if args.warmup_steps > 0 and step <= args.warmup_steps:
            learning_rate *= step / args.warmup_steps
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = learning_rate
        optimizer.step()
        task_name = TASK_NAMES[int(last_metrics["task_index"])]
        task_step_counts[task_name] += 1
        run.log({
            "train/loss": last_metrics["loss"],
            "train/supervised_loss": last_metrics["supervised_loss"],
            "train/orthogonality_loss": last_metrics["orthogonality_loss"],
            "train/mincut_loss": last_metrics["mincut_loss"],
            f"train/task_loss/{task_name}": last_metrics["loss"],
            "train/task_index": int(last_metrics["task_index"]),
            "train/epoch": epoch,
            "train/grad_norm": float(grad_norm.detach().float().cpu()),
            "train/learning_rate": learning_rate,
        }, step=step)
        with metrics_path.open("a", newline="") as handle:
            csv.writer(handle).writerow([
                step, task_name, last_metrics["loss"], last_metrics["supervised_loss"],
                last_metrics["orthogonality_loss"], last_metrics["mincut_loss"],
                float(grad_norm.detach().float().cpu()),
            ])
        if step % args.eval_every == 0 or step == total_steps:
            metrics = evaluate_and_save(
                model, test_loader, device, args.output_dir, step, args.exclude_task
            )
            run.log({
                "test/loss": metrics["loss"],
                "test/kinetic/macro_rmse": metrics["tasks"]["kinetic"]["macro_average"]["rmse"],
                "test/ph/rmse": metrics["tasks"]["ph"]["regression"]["rmse"],
                "test/mutation/auroc": metrics["tasks"]["mutation_effect"]["auroc"],
                "test/esp/auroc": metrics["tasks"]["esp"]["auroc"],
            }, step=step)
        if step % args.save_every == 0 or stop_requested:
            checkpoint(step, last_metrics)
        if stop_requested:
            run.finish(exit_code=99)
            raise SystemExit(99)

    checkpoint(total_steps, last_metrics)
    if set(task_step_counts) != set(active_tasks):
        raise RuntimeError(f"Training missed tasks: {sorted(set(active_tasks) - set(task_step_counts))}")
    torch.save(model.state_dict(), args.output_dir / "model.pt")
    (args.output_dir / "training_summary.json").write_text(json.dumps({
        "status": "completed",
        "step": total_steps,
        "wandb_run_id": run.id,
        "task_step_counts": dict(task_step_counts),
    }, indent=2) + "\n")
    run.finish()


if __name__ == "__main__":
    main()

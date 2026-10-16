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
from scipy.special import ndtr
from sklearn.metrics import average_precision_score, roc_auc_score

CODEBASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODEBASE_DIR))

from models.enzyme_prediction import (  # noqa: E402
    ESP,
    MUTATION,
    TASK_NAMES,
    task_distribution,
)
from models.net import Phenom  # noqa: E402
from train.trainers import (  # noqa: E402
    PHENOMETrainer,
    PROBABILITY_EPSILON,
    SIGMA_FLOOR,
    TrainerConfig,
    gaussian_probability_above_zero,
    mutation_distribution,
)
from utils.dataloader import DataLoader as EvaluationDataLoader  # noqa: E402
from utils.dataloader import collate_enzyme_samples  # noqa: E402
from utils.dataset import ProteinMoleculeDataset  # noqa: E402


BASE_OUTPUTS = (
    ("01_log10_kcat", "mu_log10_kcat", 0),
    ("02_log10_km", "mu_log10_km", 1),
    ("03_log10_kcat_km", "mu_log10_kcat_km", 2),
    ("04_ph", "mu_ph", 3),
    ("05_esp", "mu_esp", 4),
    ("06_fitness", "mu_fitness", 5),
)
BASE_OUTPUT_TASKS = {
    "01_log10_kcat": "kcat",
    "02_log10_km": "km",
    "03_log10_kcat_km": "kcat_km",
    "04_ph": "ph",
    "05_esp": "esp",
    "06_fitness": "mutation",
}


DEFAULT_CONFIG: dict[str, Any] = {
    "model": "Phenom",
    "model_class": "Phenom",
    "architecture": "Phenom",
    "task_names": list(TASK_NAMES),
    "base_outputs": ["log10_kcat", "log10_km", "log10_kcat_km", "ph", "esp", "fitness"],
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
        "prediction_head_layout": "task_independent",
        "ligand_language_embedding": True,
        "enzyme_aware_interaction": True,
    },
}


Group = tuple[int, bool, bool]


class RankingTaskBatchSampler(torch.utils.data.Sampler[list[int]]):
    """Fixed-quota task sampling with source-balanced, ranking-friendly kinetic batches."""

    def __init__(
        self,
        frame: pd.DataFrame,
        batch_size: int,
        task_weights: dict[int, float],
        seed: int = 888,
        batches_per_epoch: int | None = None,
        balance_esp_sources: bool = False,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.batch_size = int(batch_size)
        self.task_weights = task_weights
        self.seed = int(seed)
        self.epoch = 0
        self.balance_esp_sources = balance_esp_sources
        self.sources = tuple(sorted(self.frame["source_class"].unique()))
        self.task_sources = {
            int(task): tuple(sorted(group["source_class"].unique()))
            for task, group in self.frame.groupby("task_index", sort=True)
        }
        self.group_indices = {
            (int(group[0]), bool(group[1]), bool(group[2])): indices.tolist()
            for group, indices in self.frame.groupby(
                ["task_index", "has_ligand", "is_binary"], sort=True
            ).indices.items()
        }
        self.esp_source_indices = {
            source: group.index.tolist()
            for source, group in self.frame[self.frame["task_index"].eq(ESP)].groupby(
                "source_class", sort=True
            )
        }
        requested_batches = batches_per_epoch or sum(
            math.ceil(len(indices) / self.batch_size) for indices in self.group_indices.values()
        )
        self.batches_per_epoch = requested_batches

        self.kinetic_indices = {
            task: indices.tolist()
            for task in (0, 1, 2)
            if len(indices := self.frame.index[self.frame["task_index"].eq(task)])
        }
        self.quartiles = {}
        self.quartile_sources = {}
        self.quartile_source_clusters = {}
        self.pair_groups = {}
        for task, indices in self.kinetic_indices.items():
            ranks = self.frame.loc[indices, "label"].rank(method="first")
            bins = pd.qcut(ranks, 4, labels=False)
            self.quartiles[task] = {
                quartile: bins.index[bins.eq(quartile)].tolist() for quartile in range(4)
            }
            for quartile, rows in self.quartiles[task].items():
                for source in self.task_sources[task]:
                    self.quartile_sources[task, quartile, source] = [
                        index for index in rows if self.frame.at[index, "source_class"] == source
                    ]
                    source_rows = self.frame.loc[self.quartile_sources[task, quartile, source]]
                    self.quartile_source_clusters[task, quartile, source] = {
                        cluster: group.index.tolist()
                        for cluster, group in source_rows.groupby("cluster_id", sort=False)
                    }
            task_frame = self.frame.loc[indices]
            for group_column, distinct_column in (("Ligand", "Protein"), ("Protein", "Ligand")):
                for source in self.task_sources[task]:
                    source_frame = task_frame[task_frame["source_class"].eq(source)]
                    self.pair_groups[task, group_column, source] = [
                        rows.drop_duplicates(distinct_column).index.tolist()
                        for _, rows in source_frame.groupby(group_column, sort=False)
                        if rows[distinct_column].nunique() >= 2
                    ]

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
        total_weight = sum(self.task_weights[task] for task in tasks)
        task_counts = {task: 0 for task in tasks}
        task_schedule = []
        for position in range(self.batches_per_epoch):
            task = max(
                tasks,
                key=lambda value: (position + 1) * self.task_weights[value] / total_weight - task_counts[value],
            )
            task_schedule.append(task)
            task_counts[task] += 1
        for task in task_schedule:
            if task in (0, 1, 2):
                yield self._kinetic_batch(task, rng)
            elif task == ESP and self.balance_esp_sources:
                sources = sorted(self.esp_source_indices)
                counts = Counter(sources[position % len(sources)] for position in range(self.batch_size))
                indices = []
                for source, count in counts.items():
                    indices.extend(rng.sample(self.esp_source_indices[source], count))
                rng.shuffle(indices)
                yield [(index, 0) for index in indices]
            else:
                group = rng.choices(
                    task_groups[task], weights=[len(self.group_indices[value]) for value in task_groups[task]]
                )[0]
                yield [(index, 0) for index in rng.sample(self.group_indices[group], k=self.batch_size)]

    def _kinetic_batch(self, task: int, rng: random.Random) -> list[tuple[int, int]]:
        stratified_count = self.batch_size // 2
        ligand_count = self.batch_size // 4
        protein_count = self.batch_size - stratified_count - ligand_count
        batch = []
        used_clusters = set()
        for position in range(stratified_count):
            quartile = position % 4
            sources = self.task_sources[task]
            source = sources[
                ((position // 4) + quartile + self.epoch) % len(sources)
            ]
            clusters = self.quartile_source_clusters[task, quartile, source]
            cluster = rng.choice([value for value in clusters if value not in used_clusters])
            index = rng.choice(clusters[cluster])
            batch.append((index, 0))
            used_clusters.add(cluster)
        batch.extend((index, 1) for index in self._paired_rows(
            task, "Ligand", "Protein", ligand_count, rng, source_offset=0
        ))
        batch.extend((index, 2) for index in self._paired_rows(
            task, "Protein", "Ligand", protein_count, rng, source_offset=1
        ))
        rng.shuffle(batch)
        return batch

    def _paired_rows(
        self, task: int, group_column: str, distinct_column: str, count: int,
        rng: random.Random, source_offset: int,
    ) -> list[int]:
        result = []
        for pair_number in range(count // 2):
            sources = self.task_sources[task]
            source = sources[(pair_number + source_offset) % len(sources)]
            groups = self.pair_groups[task, group_column, source]
            if groups:
                result.extend(rng.sample(rng.choice(groups), 2))
            else:
                quartile = pair_number % 4
                result.extend(rng.sample(self.quartile_sources[task, quartile, source], 2))
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--additional-train-csv", type=Path, action="append", default=[])
    parser.add_argument("--validation-csv", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path, required=True)
    parser.add_argument("--scaler-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--init-model-dir", type=Path)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--degree-path", type=Path)
    parser.add_argument("--from-scratch", action="store_true")
    parser.add_argument("--protein-cache-dir", type=Path, required=True)
    parser.add_argument("--ligand-cache-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--batch-tune-csv", type=Path)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--total-steps", type=int, default=200_000)
    parser.add_argument("--eval-every", type=int, default=10_000)
    parser.add_argument("--eval-start-step", type=int, default=50_000)
    parser.add_argument("--save-every", type=int, default=10_000)
    parser.add_argument("--save-start-step", type=int, default=50_000)
    parser.add_argument("--milestone-save-step", type=int, default=0)
    parser.add_argument(
        "--training-phase",
        choices=(
            "mean", "task_mean", "head_mean", "sigma",
        ),
        default="mean",
    )
    parser.add_argument("--head-only-steps", type=int, default=5_000)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--global-learning-rate", type=float, default=5e-5)
    parser.add_argument("--head-learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=5_000)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--task-weights", default="kcat=20,km=20,kcat_km=10,ph=20,esp=30,mutation=10")
    parser.add_argument("--mincut-loss-weight", type=float, default=0.02)
    parser.add_argument("--orthogonality-loss-weight", type=float, default=0.02)
    parser.add_argument("--global-rank-loss-weight", type=float, default=0.15)
    parser.add_argument("--same-ligand-rank-loss-weight", type=float, default=0.25)
    parser.add_argument("--same-protein-rank-loss-weight", type=float, default=0.25)
    parser.add_argument("--rank-tau", type=float, default=0.5)
    parser.add_argument("--rank-min-delta", type=float, default=0.2)
    parser.add_argument("--rank-loss-mode", choices=("margin", "gaussian"), default="margin")
    parser.add_argument("--mean-loss-mode", choices=("huber", "mse"), default="huber")
    parser.add_argument(
        "--binary-loss-mode", choices=("gaussian_cdf", "logits"), default="gaussian_cdf"
    )
    parser.add_argument("--selection-task", choices=TASK_NAMES, default="kcat")
    parser.add_argument(
        "--selection-subgroup",
        choices=("all", "unseen_protein", "unseen_ligand", "unseen_cluster"),
        default="all",
    )
    parser.add_argument("--sigma-loss-weight", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=888)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--evaluate-checkpoints", type=Path, nargs="+")
    parser.add_argument("--exclude-task", choices=TASK_NAMES, action="append", default=[])
    parser.add_argument(
        "--prediction-head-layout",
        choices=("task_independent",),
        default="task_independent",
    )
    parser.add_argument("--wandb-project", default="PHENOM")
    parser.add_argument("--wandb-name")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default=None)
    parser.add_argument("--debug-steps", type=int)
    parser.add_argument("--skip-final-test", action="store_true")
    return parser.parse_args()


def model_dataset(frame: pd.DataFrame, args: argparse.Namespace) -> ProteinMoleculeDataset:
    return ProteinMoleculeDataset(
        frame,
        molecule_folder=args.ligand_cache_dir,
        protein_folder=args.protein_cache_dir,
        protein_embedding_dim=args.protein_embedding_dim,
        ligand_language_embedding=True,
        standardize=True,
    )


def set_training_phase(model: torch.nn.Module, phase: str, head_only: bool = False) -> None:
    for name, parameter in model.named_parameters():
        if phase == "sigma":
            parameter.requires_grad_(name.startswith("prediction_head.sigma_heads"))
        elif phase == "task_mean":
            parameter.requires_grad_(
                name.startswith((
                    "prediction_head.adapters", "prediction_head.mu_heads",
                    "ligand_language_global", "catalytic_",
                ))
            )
        elif phase == "head_mean":
            parameter.requires_grad_(
                name.startswith(("prediction_head.adapters", "prediction_head.mu_heads"))
            )
        elif head_only:
            parameter.requires_grad_(name.startswith((
                "catalytic_", "ligand_global", "ligand_language_global",
                "prediction_head",
            )) and "sigma_heads" not in name)
        else:
            parameter.requires_grad_("sigma_heads" not in name)
    model.train(phase != "sigma")


def training_loader(dataset: ProteinMoleculeDataset, args: argparse.Namespace):
    task_weights = {
        TASK_NAMES.index(item.split("=", 1)[0]): float(item.split("=", 1)[1])
        for item in args.task_weights.replace("+", ",").split(",")
    }
    sampler = RankingTaskBatchSampler(
        dataset.frame,
        batch_size=args.batch_size,
        task_weights=task_weights,
        seed=args.seed,
        balance_esp_sources=False,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_enzyme_samples,
        num_workers=args.num_workers,
        pin_memory=args.num_workers > 0,
        persistent_workers=args.num_workers > 0,
    )
    return loader, sampler


def as_list(value, size: int, default="") -> list[str]:
    if value is None:
        return [default] * size
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)] * size


def validation_metrics(
    core: PHENOMETrainer,
    loader,
    device: torch.device,
    seen_proteins: set[str],
    seen_ligands: set[str],
    seen_clusters: set[str],
) -> dict:
    core.model.eval()
    predictions: dict[int, list[np.ndarray]] = defaultdict(list)
    targets: dict[int, list[np.ndarray]] = defaultdict(list)
    sigmas: dict[int, list[np.ndarray]] = defaultdict(list)
    protein_keys: dict[int, list[str]] = defaultdict(list)
    ligand_keys: dict[int, list[str]] = defaultdict(list)
    cluster_keys: dict[int, list[str]] = defaultdict(list)
    binary_tasks: dict[int, bool] = {}
    losses = []
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        for batch in loader:
            _, metrics, arrays = core.batch_loss(
                batch, training_phase="mean", rank_weight_scale=0.0
            )
            task = int(metrics["task_index"])
            predictions[task].append(arrays["prediction"])
            targets[task].append(arrays["target"])
            sigmas[task].append(arrays["sigma"])
            if task in (0, 1, 2):
                protein_keys[task].extend(as_list(batch["primary"].prot_key, len(arrays["target"])))
                ligand_keys[task].extend(as_list(batch["primary"].mol_key, len(arrays["target"])))
                cluster_keys[task].extend(as_list(batch["primary"].cluster_id, len(arrays["target"])))
            binary_tasks[task] = bool(arrays["is_binary"][0])
            losses.append(metrics["loss"])
    task_metrics = {}
    for task in sorted(predictions):
        prediction = np.concatenate(predictions[task])
        target = np.concatenate(targets[task])
        sigma = np.concatenate(sigmas[task])
        error = prediction - target
        task_metrics[TASK_NAMES[task]] = {
            "spearman": float(pd.Series(prediction).corr(pd.Series(target), method="spearman")),
            "rmse": float(np.sqrt(np.mean(error ** 2))),
            "sigma_error_mae": float(np.mean(np.abs(sigma - np.abs(error)))),
            "gaussian_nll": float(np.mean(np.log(sigma) + 0.5 * (error / sigma) ** 2)),
        }
        if task in (0, 1, 2):
            subgroups = {}
            for name, mask in {
                "unseen_protein": ~pd.Series(protein_keys[task]).isin(seen_proteins).to_numpy(),
                "unseen_ligand": ~pd.Series(ligand_keys[task]).isin(seen_ligands).to_numpy(),
                "unseen_cluster": ~pd.Series(cluster_keys[task]).isin(seen_clusters).to_numpy(),
            }.items():
                subgroups[name] = {
                    "n": int(mask.sum()),
                    "spearman": float(
                        pd.Series(prediction[mask]).corr(pd.Series(target[mask]), method="spearman")
                    ),
                }
            task_metrics[TASK_NAMES[task]]["subgroups"] = subgroups
        if not binary_tasks[task]:
            borda = np.empty(len(prediction), dtype=float)
            for start in range(0, len(prediction), 512):
                stop = min(start + 512, len(prediction))
                pair_scale = np.sqrt(
                    sigma[start:stop, None] ** 2 + sigma[None, :] ** 2
                )
                borda[start:stop] = ndtr(
                    (prediction[start:stop, None] - prediction[None, :]) / pair_scale
                ).mean(axis=1)
            task_metrics[TASK_NAMES[task]]["uncertainty_ranking"] = {
                "confidence_only_negative_sigma": float(
                    pd.Series(-sigma).corr(pd.Series(target), method="spearman")
                ),
                "lower_90_bound": float(
                    pd.Series(prediction - 1.64485363 * sigma).corr(
                        pd.Series(target), method="spearman"
                    )
                ),
                "upper_90_bound": float(
                    pd.Series(prediction + 1.64485363 * sigma).corr(
                        pd.Series(target), method="spearman"
                    )
                ),
                "gaussian_pairwise_borda": float(
                    pd.Series(borda).corr(pd.Series(target), method="spearman")
                ),
            }
        if binary_tasks[task]:
            task_metrics[TASK_NAMES[task]]["auroc"] = float(roc_auc_score(target, prediction))
            task_metrics[TASK_NAMES[task]]["auprc"] = float(
                average_precision_score(target, prediction)
            )
    if "kcat" in task_metrics and "km" in task_metrics:
        task_metrics["macro_spearman"] = float(np.mean([
            task_metrics["kcat"]["spearman"], task_metrics["km"]["spearman"]
        ]))
    return {"loss": float(np.mean(losses)), "tasks": task_metrics}


def evaluate_and_save(
    model, loader, device: torch.device, output_dir: Path, step: int, exclude_tasks: Sequence[str],
    centers: torch.Tensor, scales: torch.Tensor, binary_loss_mode: str,
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
                primary, protein_only=batch["protein_only"]
            )
            raw_mu = mu.double() * scales + centers
            raw_sigma = sigma.double() * scales
            task = int(primary.task_index.view(-1)[0])
            task_index = torch.full((mu.size(0),), task, dtype=torch.long, device=device)
            target = primary.label.view(-1)
            is_binary = primary.is_binary.view(-1)
            wt_keys = [""] * target.numel()
            if task == MUTATION:
                reference = batch["reference"].to(device)
                wt_mu, wt_sigma, wt_ortho, wt_mincut, _ = model(reference, protein_only=batch["protein_only"])
                selected_mu, selected_sigma = mutation_distribution(raw_mu, raw_sigma, wt_mu, wt_sigma)
                ortho = 0.5 * (ortho + wt_ortho)
                mincut = 0.5 * (mincut + wt_mincut)
                wt_keys = as_list(reference.prot_key, target.numel())
            else:
                selected_mu, selected_sigma = task_distribution(raw_mu, raw_sigma, task_index)
            if bool(is_binary[0]):
                with torch.autocast(device_type=device.type, enabled=False):
                    if binary_loss_mode == "logits":
                        prediction = torch.sigmoid(selected_mu.double())
                        supervised = F.binary_cross_entropy_with_logits(
                            selected_mu.double(), target.double()
                        )
                    else:
                        prediction = gaussian_probability_above_zero(selected_mu, selected_sigma)
                        supervised = F.binary_cross_entropy(prediction, target.double())
            else:
                prediction = selected_mu
                supervised = F.smooth_l1_loss(prediction.double(), target.double(), beta=0.5)
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
                "base_mu": raw_mu,
                "base_sigma": raw_sigma,
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
        if BASE_OUTPUT_TASKS[folder] in exclude_tasks:
            continue
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
        ("ph", "04_ph", benchmark_root / "ph_prediction" / "test.csv", {"ph"}),
        ("mutation_effect", "06_fitness", benchmark_root / "mutation_effect" / "test.csv", {"mutation"}),
        ("esp", "05_esp", benchmark_root / "esp_prediction" / "test.csv", {"esp"}),
    )
    protocol_metrics = {}
    for task, folder, test_set, task_names in protocol_tasks:
        if task_names.issubset(exclude_tasks):
            continue
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


def build_model(config: dict[str, Any], degree: dict[str, torch.Tensor], device: str) -> Phenom:
    params = config["params"]
    model = Phenom(
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


def load_initial_state(model: Phenom, path: Path, device: str) -> int:
    state = torch.load(path, map_location=device)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    current = model.state_dict()
    compatible = {key: value for key, value in state.items() if key in current and current[key].shape == value.shape}
    skipped = set(state).difference(compatible)
    if not all(key.startswith((
        "mu_out.", "sigma_out.", "prediction_head.", "inter_convs.", "catalytic_interaction."
    )) for key in skipped):
        raise RuntimeError(f"Unexpected incompatible checkpoint keys: {sorted(skipped)}")
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
        initial_path = args.init_checkpoint or args.init_model_dir / "model.pt"
        load_initial_state(model, initial_path, args.device)
    model.train()
    scalers = json.loads(args.scaler_json.read_text())
    trainer = PHENOMETrainer(
        model, TrainerConfig(output_dir=args.output_dir, total_steps=1), args.device, scalers
    )
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
                    loss, _, _ = trainer.batch_loss(batch, rank_weight_scale=0.0)
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
    scalers = json.loads(args.scaler_json.read_text())
    config["target_scalers"] = scalers
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.ligand_cache_dir.mkdir(parents=True, exist_ok=True)
    degree_path = args.degree_path or args.init_model_dir / "degree.pt"
    for folder, _, _ in BASE_OUTPUTS:
        if BASE_OUTPUT_TASKS[folder] in args.exclude_task:
            continue
        (args.output_dir / "evaluation" / folder).mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    if args.evaluate_checkpoints:
        excluded_indices = [TASK_NAMES.index(task) for task in args.exclude_task]
        test_frame = read_frame(args.test_csv)
        test_frame = test_frame[~test_frame["task_index"].isin(excluded_indices)]
        device = torch.device(args.device)
        degree = load_degree(degree_path, args.device)
        model = build_model(config, degree, args.device)
        core = PHENOMETrainer(model, TrainerConfig(output_dir=args.output_dir), args.device, scalers)
        test_loader = EvaluationDataLoader(
            model_dataset(test_frame, args),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.num_workers > 0,
            persistent_workers=args.num_workers > 0,
        )
        sweep = {}
        for checkpoint_path in args.evaluate_checkpoints:
            state = torch.load(checkpoint_path, map_location=device)
            model_state = state.get("model_state_dict", state)
            model.load_state_dict(model_state)
            step = int(state.get("step", 0)) if isinstance(state, dict) else 0
            sweep[str(step)] = evaluate_and_save(
                model, test_loader, device, args.output_dir, step, args.exclude_task,
                core.centers, core.scales, args.binary_loss_mode,
            )
        (args.output_dir / "checkpoint_sweep.yaml").write_text(
            yaml.safe_dump(sweep, sort_keys=True, allow_unicode=True)
        )
        return

    excluded_indices = [TASK_NAMES.index(task) for task in args.exclude_task]
    train_frame = read_frame(args.train_csv)
    if args.additional_train_csv:
        train_frame = pd.concat(
            [train_frame, *(read_frame(path) for path in args.additional_train_csv)],
            ignore_index=True,
        )
    validation_frame = read_frame(args.validation_csv)
    test_frame = read_frame(args.test_csv)
    train_frame = train_frame[~train_frame["task_index"].isin(excluded_indices)]
    validation_frame = validation_frame[~validation_frame["task_index"].isin(excluded_indices)]
    test_frame = test_frame[~test_frame["task_index"].isin(excluded_indices)]
    requested_training_tasks = {
        TASK_NAMES.index(item.split("=", 1)[0])
        for item in args.task_weights.replace("+", ",").split(",")
    }
    train_frame = train_frame[train_frame["task_index"].isin(requested_training_tasks)]
    active_tasks = tuple(TASK_NAMES[index] for index in sorted(train_frame["task_index"].unique()))
    device = torch.device(args.device)
    degree = load_degree(degree_path, args.device)
    if args.batch_size == 0:
        tune_frame = read_frame(args.batch_tune_csv)
        tune_frame = tune_frame[~tune_frame["task_index"].isin(excluded_indices)]
        args.batch_size = tune_batch_size(tune_frame, args, config, degree)
        seed(args.seed)
    model = build_model(config, degree, args.device)
    if args.from_scratch:
        initialized = 0
    elif args.init_checkpoint:
        initialized = load_initial_state(model, args.init_checkpoint, args.device)
    else:
        initialized = load_initial_state(model, args.init_model_dir / "model.pt", args.device)
    train_dataset = model_dataset(train_frame, args)
    validation_dataset = model_dataset(validation_frame, args)
    test_dataset = model_dataset(test_frame, args)
    train_loader, sampler = training_loader(train_dataset, args)
    validation_loader = EvaluationDataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.num_workers > 0,
        persistent_workers=args.num_workers > 0,
    )
    test_loader = EvaluationDataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.num_workers > 0,
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
            global_learning_rate=args.global_learning_rate,
            head_learning_rate=args.head_learning_rate,
            weight_decay=args.weight_decay,
            grad_clip=args.grad_clip,
            mincut_loss_weight=args.mincut_loss_weight,
            orthogonality_loss_weight=args.orthogonality_loss_weight,
            global_rank_loss_weight=args.global_rank_loss_weight,
            same_ligand_rank_loss_weight=args.same_ligand_rank_loss_weight,
            same_protein_rank_loss_weight=args.same_protein_rank_loss_weight,
            rank_tau=args.rank_tau,
            rank_min_delta=args.rank_min_delta,
            rank_loss_mode=args.rank_loss_mode,
            mean_loss_mode=args.mean_loss_mode,
            binary_loss_mode=args.binary_loss_mode,
            sigma_loss_weight=args.sigma_loss_weight,
        ),
        args.device,
        scalers,
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
        "train_csv": str(args.train_csv),
        "additional_train_csv": [str(path) for path in args.additional_train_csv],
        "test_csv": str(args.test_csv),
        "protein_cache_dir": str(args.protein_cache_dir), "ligand_cache_dir": str(args.ligand_cache_dir),
        "protein_encoder": sample_cache["encoder"],
        "protein_embedding_dim": args.protein_embedding_dim,
        "protein_contact_encoder": sample_cache["contact_encoder"],
        "tasks": list(active_tasks),
        "active_base_output_folders": [
            item[0] for item in BASE_OUTPUTS
            if BASE_OUTPUT_TASKS[item[0]] not in args.exclude_task
        ],
        "excluded_tasks": args.exclude_task,
        "prediction_head_layout": args.prediction_head_layout,
        "ligand_language_embedding": True,
        "enzyme_aware_interaction": True,
        "architecture": config["architecture"],
        "eval_every": args.eval_every, "eval_start_step": args.eval_start_step,
        "save_every": args.save_every, "save_start_step": args.save_start_step,
        "task_sampling": "fixed_quota_with_kinetic_ranking_batches",
        "rank_loss_mode": args.rank_loss_mode,
        "mean_loss_mode": args.mean_loss_mode,
        "binary_loss_mode": args.binary_loss_mode,
        "rank_loss_weights": {
            "global": args.global_rank_loss_weight,
            "same_ligand": args.same_ligand_rank_loss_weight,
            "same_protein": args.same_protein_rank_loss_weight,
        },
        "task_weights": args.task_weights,
        "initialized_backbone_tensors": initialized, "resume_step": start_step,
        "from_scratch": args.from_scratch, "degree_path": str(degree_path),
        "batch_size": args.batch_size, "sigma_floor": SIGMA_FLOOR,
        "learning_rate": args.learning_rate, "warmup_steps": args.warmup_steps,
        "training_schedule": {
            "phase": args.training_phase,
            "head_only_steps": args.head_only_steps,
            "milestone_save_step": args.milestone_save_step,
        },
        "validation_csv": str(args.validation_csv),
        "validation_selection": (
            f"{args.selection_task}_auroc"
            if args.selection_task in ("esp", "mutation")
            else f"{args.selection_task}_sigma_error_mae"
            if args.training_phase == "sigma"
            else f"{args.selection_task}_{args.selection_subgroup}_spearman"
        ),
        "target_scalers": scalers,
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
            "protein": "ESMC contact-aware residue graph",
            "ligand": "atom graph",
            "combination": "ligand-conditioned catalytic top-64 cross-attention",
            "mincut_weight": args.mincut_loss_weight,
            "orthogonality_weight": args.orthogonality_loss_weight,
            "ramp_steps": 1000,
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
                "step", "task", "loss", "supervised_loss", "global_rank_loss",
                "same_ligand_rank_loss", "same_protein_rank_loss",
                "orthogonality_loss", "mincut_loss", "grad_norm"
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
        if (
            step >= args.save_start_step and step % args.save_every == 0
        ) or step == args.milestone_save_step:
            atomic_save(payload, args.output_dir / f"checkpoint_{step:09d}.pt")

    last_metrics = {}
    best_validation = float("-inf")
    training_phase = args.training_phase
    for validation_path in args.output_dir.glob("validation_step_*.json"):
        previous = json.loads(validation_path.read_text())["tasks"][args.selection_task]
        if args.selection_task in ("esp", "mutation"):
            previous_score = previous["auroc"]
        elif training_phase == "sigma":
            previous_score = -previous["sigma_error_mae"]
        else:
            previous_score = (
                previous["spearman"]
                if args.selection_subgroup == "all"
                else previous["subgroups"][args.selection_subgroup]["spearman"]
            )
        best_validation = max(best_validation, previous_score)
    head_only = training_phase == "mean" and start_step < args.head_only_steps
    set_training_phase(model, training_phase, head_only=head_only)
    for step in range(start_step + 1, total_steps + 1):
        next_head_only = training_phase == "mean" and step <= args.head_only_steps
        if next_head_only != head_only:
            head_only = next_head_only
            set_training_phase(model, training_phase, head_only=head_only)
            print(f"head_only={head_only} step={step}", flush=True)
        try:
            batch = next(iterator)
        except StopIteration:
            epoch += 1
            sampler.set_epoch(epoch)
            iterator = iter(train_loader)
            batch = next(iterator)
        optimizer.zero_grad(set_to_none=True)
        graph_weight_scale = (
            min(1.0, max(0, step - args.head_only_steps) / 1000.0)
            if training_phase == "mean" and not head_only else 0.0
        )
        rank_weight_scale = min(1.0, step / 1000.0) if training_phase != "sigma" else 0.0
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            loss, last_metrics, _ = core.batch_loss(
                batch,
                training_phase=training_phase,
                graph_weight_scale=graph_weight_scale,
                rank_weight_scale=rank_weight_scale,
            )
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.grad_clip, error_if_nonfinite=True
        )
        if args.warmup_steps > 0 and step <= args.warmup_steps:
            learning_rate_scale = step / args.warmup_steps
        else:
            progress = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
            learning_rate_scale = 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))
        core.set_learning_rate_scale(learning_rate_scale)
        optimizer.step()
        task_name = TASK_NAMES[int(last_metrics["task_index"])]
        task_step_counts[task_name] += 1
        run.log({
            "train/loss": last_metrics["loss"],
            "train/supervised_loss": last_metrics["supervised_loss"],
            "train/global_rank_loss": last_metrics["global_rank_loss"],
            "train/same_ligand_rank_loss": last_metrics["same_ligand_rank_loss"],
            "train/same_protein_rank_loss": last_metrics["same_protein_rank_loss"],
            "train/orthogonality_loss": last_metrics["orthogonality_loss"],
            "train/mincut_loss": last_metrics["mincut_loss"],
            f"train/task_loss/{task_name}": last_metrics["loss"],
            "train/task_index": int(last_metrics["task_index"]),
            "train/epoch": epoch,
            "train/grad_norm": float(grad_norm.detach().float().cpu()),
            "train/learning_rate_scale": learning_rate_scale,
            "train/phase": {
                "mean": 0,
                "sigma": 1,
                "task_mean": 2,
                "head_mean": 3,
            }[training_phase],
        }, step=step)
        with metrics_path.open("a", newline="") as handle:
            csv.writer(handle).writerow([
                step, task_name, last_metrics["loss"], last_metrics["supervised_loss"],
                last_metrics["global_rank_loss"], last_metrics["same_ligand_rank_loss"],
                last_metrics["same_protein_rank_loss"],
                last_metrics["orthogonality_loss"], last_metrics["mincut_loss"],
                float(grad_norm.detach().float().cpu()),
            ])
        if (
            step >= args.eval_start_step
            and step % args.eval_every == 0
        ) or step == total_steps:
            metrics = validation_metrics(
                core,
                validation_loader,
                device,
                set(train_dataset.frame["Protein"].astype(str)),
                set(train_dataset.frame.loc[train_dataset.frame["has_ligand"], "Ligand"].astype(str)),
                set(train_dataset.frame["cluster_id"].astype(str)),
            )
            set_training_phase(model, training_phase, head_only=head_only)
            validation_log = {"validation/loss": metrics["loss"]}
            for task, values in metrics["tasks"].items():
                if isinstance(values, dict):
                    for metric, value in values.items():
                        validation_log[f"validation/{task}/{metric}"] = value
            if "macro_spearman" in metrics["tasks"]:
                validation_log["validation/kinetic/macro_spearman"] = metrics["tasks"]["macro_spearman"]
            run.log(validation_log, step=step)
            (args.output_dir / f"validation_step_{step:09d}.json").write_text(
                json.dumps(metrics, indent=2) + "\n"
            )
            selected_metrics = metrics["tasks"][args.selection_task]
            if args.selection_task in ("esp", "mutation"):
                selection_score = selected_metrics["auroc"]
            elif training_phase == "sigma":
                selection_score = -selected_metrics["sigma_error_mae"]
            else:
                selection_score = (
                    selected_metrics["spearman"]
                    if args.selection_subgroup == "all"
                    else selected_metrics["subgroups"][args.selection_subgroup]["spearman"]
                )
            if selection_score > best_validation:
                best_validation = selection_score
                checkpoint(step, {**last_metrics, "validation": metrics})
                best_state = torch.load(args.output_dir / "latest_checkpoint.pt", map_location="cpu")
                atomic_save(best_state, args.output_dir / "best_checkpoint.pt")
        if (
            step >= args.save_start_step and step % args.save_every == 0
        ) or step == args.milestone_save_step or stop_requested:
            checkpoint(step, last_metrics)
        if stop_requested:
            run.finish(exit_code=99)
            raise SystemExit(99)

    checkpoint(total_steps, last_metrics)
    if set(task_step_counts) != set(active_tasks):
        raise RuntimeError(f"Training missed tasks: {sorted(set(active_tasks) - set(task_step_counts))}")
    torch.save(model.state_dict(), args.output_dir / "model.pt")
    test_metrics = None
    if not args.skip_final_test:
        test_metrics = evaluate_and_save(
            model, test_loader, device, args.output_dir, total_steps, args.exclude_task,
            core.centers, core.scales, args.binary_loss_mode,
        )
    (args.output_dir / "training_summary.json").write_text(json.dumps({
        "status": "completed",
        "step": total_steps,
        "wandb_run_id": run.id,
        "task_step_counts": dict(task_step_counts),
        "best_validation_selection_score": best_validation,
        "test_metrics": test_metrics,
    }, indent=2) + "\n")
    run.finish()


if __name__ == "__main__":
    main()

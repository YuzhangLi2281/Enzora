from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score

from models.enzyme_prediction import (
    ESP,
    MUTATION,
    TASK_NAMES,
    gaussian_nll,
    mutation_distribution,
    probability_above_zero,
    task_distribution,
)


@dataclass
class TrainerConfig:
    output_dir: Path
    total_steps: int = 1_000_000
    eval_every: int = 25_000
    save_every: int = 10_000
    learning_rate: float = 1e-5
    weight_decay: float = 1e-5
    adam_eps: float = 1e-5
    adam_betas: tuple[float, float] = (0.9, 0.999)
    grad_clip: float = 1.0
    mincut_loss_weight: float = 1.0
    orthogonality_loss_weight: float = 1.0
    max_eval_batches: int | None = 256


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


class PHENOMETrainer:
    def __init__(self, model, train_loader, valid_loader, test_loader, config: TrainerConfig, device: str) -> None:
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.config = config
        self.device = torch.device(device)
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.output_dir / "training_metrics.csv"
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=config.adam_betas,
            eps=config.adam_eps,
        )

    def train(self) -> dict[str, Any]:
        self.model.to(self.device)
        iterator = iter(self.train_loader)
        best_score = math.inf
        best_step = None
        self._write_metrics_header()
        last_metrics = None

        for step in range(1, self.config.total_steps + 1):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(self.train_loader)
                batch = next(iterator)
            last_metrics = self._train_batch(batch)
            valid_metrics = None
            if self.valid_loader is not None and step % self.config.eval_every == 0:
                valid_metrics = self.evaluate(self.valid_loader)
                if valid_metrics["selector"] < best_score:
                    best_score = valid_metrics["selector"]
                    best_step = step
                    self._save("best_checkpoint.pt", step, valid_metrics)
                    torch.save(self.model.state_dict(), self.output_dir / "best_model.pt")
            if step % self.config.save_every == 0:
                self._save(f"checkpoint_{step}.pt", step, valid_metrics or last_metrics)
            self._append_metrics(step, last_metrics, valid_metrics)

        valid_metrics = self.evaluate(self.valid_loader) if self.valid_loader is not None else None
        test_metrics = self.evaluate(self.test_loader) if self.test_loader is not None else None
        torch.save(self.model.state_dict(), self.output_dir / "model.pt")
        self._save("final_checkpoint.pt", self.config.total_steps, valid_metrics or last_metrics or {})
        summary = {
            "status": "completed",
            "best_step": best_step,
            "best_selector": best_score if math.isfinite(best_score) else None,
            "final_valid_metrics": valid_metrics,
            "final_test_metrics": test_metrics,
        }
        write_json(self.output_dir / "training_summary.json", summary)
        return summary

    def _train_batch(self, batch: dict) -> dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        loss, metrics, _ = self._batch_loss(batch)
        loss.backward()
        if self.config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        self.optimizer.step()
        return metrics

    def _batch_loss(self, batch: dict) -> tuple[torch.Tensor, dict[str, float], dict[str, np.ndarray]]:
        primary = batch["primary"].to(self.device)
        protein_only = batch["protein_only"]
        mu, sigma, ortho, mincut, _ = self.model(primary, protein_only=protein_only)
        task_index = primary.task_index.view(-1)
        target = primary.label.view(-1)
        is_binary = primary.is_binary.view(-1)

        if int(task_index[0]) == MUTATION:
            reference = batch["reference"].to(self.device)
            wt_mu, wt_sigma, wt_ortho, wt_mincut, _ = self.model(reference, protein_only=protein_only)
            selected_mu, selected_sigma = mutation_distribution(mu, sigma, wt_mu, wt_sigma)
            ortho = 0.5 * (ortho + wt_ortho)
            mincut = 0.5 * (mincut + wt_mincut)
        else:
            selected_mu, selected_sigma = task_distribution(mu, sigma, task_index)

        if bool(is_binary[0]):
            prediction = probability_above_zero(selected_mu, selected_sigma).clamp(1e-7, 1 - 1e-7)
            supervised_loss = F.binary_cross_entropy(prediction, target)
        else:
            prediction = selected_mu
            supervised_loss = gaussian_nll(target, selected_mu, selected_sigma, primary.sample_weight.view(-1))

        ortho = ortho.mean()
        mincut = mincut.mean()
        loss = supervised_loss + self.config.orthogonality_loss_weight * ortho + self.config.mincut_loss_weight * mincut
        task = int(task_index[0])
        metrics = {
            "loss": float(loss.detach().cpu()),
            "supervised_loss": float(supervised_loss.detach().cpu()),
            "orthogonality_loss": float(ortho.detach().cpu()),
            "mincut_loss": float(mincut.detach().cpu()),
            "mean_sigma": float(selected_sigma.detach().mean().cpu()),
            "task_index": task,
        }
        arrays = {
            "target": target.detach().cpu().numpy(),
            "prediction": prediction.detach().cpu().numpy(),
            "mu": selected_mu.detach().cpu().numpy(),
            "sigma": selected_sigma.detach().cpu().numpy(),
            "is_binary": is_binary.detach().cpu().numpy(),
            "task_index": np.full(target.numel(), task),
        }
        return loss, metrics, arrays

    def evaluate(self, loader: Iterable) -> dict[str, Any]:
        self.model.eval()
        collected = {key: [] for key in ("target", "prediction", "mu", "sigma", "is_binary", "task_index")}
        losses = []
        with torch.no_grad():
            for batch_index, batch in enumerate(loader, start=1):
                loss, _, arrays = self._batch_loss(batch)
                losses.append(float(loss.detach().cpu()))
                for key, value in arrays.items():
                    collected[key].append(value)
                if self.config.max_eval_batches is not None and batch_index >= self.config.max_eval_batches:
                    break
        arrays = {key: np.concatenate(value) for key, value in collected.items()}
        per_task = {}
        selectors = []
        for task_index, task_name in enumerate(TASK_NAMES):
            task_mask = arrays["task_index"] == task_index
            for binary in (False, True):
                mask = task_mask & (arrays["is_binary"] == binary)
                if not mask.any():
                    continue
                target = arrays["target"][mask]
                prediction = arrays["prediction"][mask]
                sigma = arrays["sigma"][mask]
                if binary:
                    labels = (prediction >= 0.5).astype(int)
                    auroc = roc_auc_score(target, prediction) if np.unique(target).size == 2 else float("nan")
                    metrics = {
                        "auroc": auroc,
                        "auprc": average_precision_score(target, prediction),
                        "f1": f1_score(target, labels, zero_division=0),
                        "accuracy": accuracy_score(target, labels),
                        "mean_sigma": float(sigma.mean()),
                    }
                    if math.isfinite(auroc):
                        selectors.append(1.0 - auroc)
                else:
                    error = prediction - target
                    rmse = float(np.sqrt(np.mean(error**2)))
                    metrics = {
                        "mae": float(np.mean(np.abs(error))),
                        "rmse": rmse,
                        "pearson": _correlation(pearsonr, target, prediction),
                        "spearman": _correlation(spearmanr, target, prediction),
                        "coverage_95": float(np.mean(np.abs(target - arrays["mu"][mask]) <= 1.96 * sigma)),
                        "mean_sigma": float(sigma.mean()),
                    }
                    scale = float(np.std(target)) or 1.0
                    selectors.append(rmse / scale)
                key = f"{task_name}_{'binary' if binary else 'continuous'}" if task_index == MUTATION else task_name
                per_task[key] = metrics
        self.model.train()
        return {
            "loss": float(np.mean(losses)),
            "selector": float(np.mean(selectors)) if selectors else float(np.mean(losses)),
            "tasks": per_task,
            "rows": int(arrays["target"].size),
        }

    def _save(self, filename: str, step: int, metrics: Mapping[str, Any]) -> None:
        torch.save(
            {
                "step": step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "metrics": dict(metrics),
                "trainer_config": {**asdict(self.config), "output_dir": str(self.config.output_dir)},
            },
            self.output_dir / filename,
        )

    def _write_metrics_header(self) -> None:
        with self.metrics_path.open("w", newline="") as handle:
            csv.writer(handle).writerow(["step", "train_loss", "train_task", "valid_loss", "valid_selector"])

    def _append_metrics(self, step: int, train: Mapping[str, Any], valid: Mapping[str, Any] | None) -> None:
        with self.metrics_path.open("a", newline="") as handle:
            csv.writer(handle).writerow([
                step,
                train["loss"],
                TASK_NAMES[int(train["task_index"])],
                valid["loss"] if valid else None,
                valid["selector"] if valid else None,
            ])


def _correlation(function, target: np.ndarray, prediction: np.ndarray) -> float:
    if target.size < 2 or np.std(target) == 0 or np.std(prediction) == 0:
        return float("nan")
    return float(function(target, prediction).statistic)

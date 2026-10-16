from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

from utils.beta_distribution import Beta


@dataclass
class TrainerConfig:
    output_dir: Path
    total_steps: int = 1_000_000
    eval_every: int = 25_000
    save_every: int = 25_000
    learning_rate: float = 1e-5
    weight_decay: float = 1e-5
    adam_eps: float = 1e-5
    adam_betas: tuple[float, float] = (0.9, 0.999)
    grad_clip: float = 1.0
    pki_min: float = 0.0
    pki_max: float = 12.0
    classification_threshold: float = 4.0
    source_priors: dict[str, dict[str, float]] | None = None
    default_source_prior: dict[str, float] = field(default_factory=lambda: {"max_phi": 24.0})
    source_constraint_weight: float = 1.0
    mincut_loss_weight: float = 1.0
    orthogonality_loss_weight: float = 1.0
    max_eval_batches: int | None = 256


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def prediction_from_beta(alpha: torch.Tensor, beta: torch.Tensor, pki_min: float, pki_max: float) -> torch.Tensor:
    scaled = alpha / (alpha + beta)
    return scaled * (pki_max - pki_min) + pki_min


def scale_pki(values: torch.Tensor, pki_min: float = 0.0, pki_max: float = 12.0) -> torch.Tensor:
    clipped = torch.clamp(values, pki_min, pki_max)
    return (clipped - pki_min) / (pki_max - pki_min)


def beta_loss_with_mixed_targets(
    alpha: torch.Tensor,
    beta: torch.Tensor,
    reg_targets: torch.Tensor,
    class_targets: torch.Tensor,
    min_pKi: float = 0.0,
    max_pKi: float = 12.0,
    threshold: float = 4.0,
    source_labels: list[str] | None = None,
    source_priors: dict[str, dict[str, float]] | None = None,
    regul_lambda: float = 1.0,
    return_mean: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    alpha = torch.clamp(alpha, min=eps)
    beta = torch.clamp(beta, min=eps)
    reg_targets = torch.as_tensor(reg_targets, dtype=alpha.dtype, device=alpha.device)
    class_targets = torch.as_tensor(class_targets, dtype=alpha.dtype, device=alpha.device)

    scaled_threshold = (threshold - min_pKi) / (max_pKi - min_pKi)
    scaled_reg_targets = scale_pki(reg_targets, min_pKi, max_pKi)
    loss = torch.zeros_like(alpha)

    reg_mask = torch.isfinite(reg_targets)
    if reg_mask.any():
        reg_distribution = Beta(alpha[reg_mask], beta[reg_mask])
        loss[reg_mask] = -reg_distribution.log_prob(scaled_reg_targets[reg_mask].clamp(eps, 1 - eps))

    class_mask = ~reg_mask
    if class_mask.any():
        class_distribution = Beta(alpha[class_mask], beta[class_mask])
        class_probs = class_distribution.cdf(torch.tensor(scaled_threshold, device=alpha.device))

        high_mask = (class_targets == 1) & class_mask
        if high_mask.any():
            loss[high_mask] = -torch.log(1 - class_probs[high_mask[class_mask]] + eps)

        low_mask = (class_targets == 0) & class_mask
        if low_mask.any():
            loss[low_mask] = -torch.log(class_probs[low_mask[class_mask]] + eps)

    if source_labels is not None and source_priors is not None:
        phi = alpha + beta
        if all((isinstance(source, float) and math.isnan(source)) or str(source).lower() == "nan" for source in source_labels):
            return loss.mean() if return_mean else loss

        max_phi = torch.tensor([source_priors[source]["max_phi"] for source in source_labels], device=alpha.device)
        normalized_excess = (phi - max_phi) / max_phi
        prior_loss = 0.5 * torch.square(torch.relu(normalized_excess))
        loss = loss + regul_lambda * prior_loss

    return loss.mean() if return_mean else loss


class SourceAwareUncertaintyTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: Iterable,
        valid_loader: Iterable | None,
        test_loader: Iterable | None,
        config: TrainerConfig,
        device: str,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.config = config
        self.device = torch.device(device)
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.output_dir / "training_metrics.csv"
        self.source_priors = dict(config.source_priors or {})
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=config.adam_betas,
            eps=config.adam_eps,
        )

    def train(self) -> dict[str, Any]:
        self.model.to(self.device)
        self.model.train()
        self._init_metrics_file()
        best_selector = math.inf
        best_step = None
        last_train_metrics: dict[str, Any] | None = None
        train_iterator = iter(self.train_loader)

        for step in range(1, self.config.total_steps + 1):
            batch, train_iterator = self._next_train_batch(train_iterator)
            train_metrics = self._train_batch(batch)
            last_train_metrics = train_metrics

            valid_metrics = None
            if self.valid_loader is not None and step % self.config.eval_every == 0:
                valid_metrics = self.evaluate(self.valid_loader, split="valid")
                selector = valid_metrics["loss"]
                if selector is not None and selector < best_selector:
                    best_selector = selector
                    best_step = step
                    self.save_training_checkpoint("best_checkpoint.pt", step, valid_metrics)
                    self.save_model_weights("best_model.pt")

            if step % self.config.save_every == 0:
                self.save_training_checkpoint(f"checkpoint_{step}.pt", step, valid_metrics or train_metrics)
                self.save_model_weights(f"model_{step}.pt")

            self._append_metrics(step, train_metrics, valid_metrics)

        final_valid_metrics = self.evaluate(self.valid_loader, split="valid") if self.valid_loader is not None else None
        final_test_metrics = self.evaluate(self.test_loader, split="test") if self.test_loader is not None else None
        final_payload = {
            "status": "completed",
            "total_steps": self.config.total_steps,
            "best_step": best_step,
            "best_selector": best_selector if math.isfinite(best_selector) else None,
            "last_train_metrics": last_train_metrics,
            "final_valid_metrics": final_valid_metrics,
            "final_test_metrics": final_test_metrics,
        }
        self.save_training_checkpoint("final_checkpoint.pt", self.config.total_steps, final_valid_metrics or last_train_metrics or {})
        self.save_model_weights("model.pt")
        write_json(self.output_dir / "training_summary.json", final_payload)
        return final_payload

    def evaluate(self, loader: Iterable | None, split: str) -> dict[str, Any] | None:
        if loader is None:
            return None
        self.model.eval()
        losses: list[float] = []
        beta_losses: list[float] = []
        mincut_losses: list[float] = []
        orthogonality_losses: list[float] = []
        maes: list[float] = []
        rmses: list[float] = []
        rows = 0
        with torch.no_grad():
            for batch_index, batch in enumerate(loader, start=1):
                if batch is None:
                    continue
                batch = batch.to(self.device)
                loss, metrics = self._compute_loss_and_metrics(batch)
                losses.append(float(loss.detach().cpu().item()))
                beta_losses.append(float(metrics["beta_loss"]))
                mincut_losses.append(float(metrics["mincut_loss"]))
                orthogonality_losses.append(float(metrics["orthogonality_loss"]))
                rows += int(metrics["rows"])
                if metrics["mae"] is not None:
                    maes.append(float(metrics["mae"]))
                if metrics["rmse"] is not None:
                    rmses.append(float(metrics["rmse"]))
                if self.config.max_eval_batches is not None and batch_index >= self.config.max_eval_batches:
                    break
        self.model.train()
        return {
            "split": split,
            "loss": self._mean(losses),
            "beta_loss": self._mean(beta_losses),
            "mincut_loss": self._mean(mincut_losses),
            "orthogonality_loss": self._mean(orthogonality_losses),
            "mae": self._mean(maes),
            "rmse": self._mean(rmses),
            "rows": rows,
            "batches": len(losses),
        }

    def save_model_weights(self, filename: str) -> None:
        torch.save(self.model.state_dict(), self.output_dir / filename)

    def save_training_checkpoint(self, filename: str, step: int, metrics: Mapping[str, Any] | None) -> None:
        payload = {
            "step": step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": dict(metrics or {}),
            "trainer_config": self._serializable_config(),
        }
        torch.save(payload, self.output_dir / filename)

    def _train_batch(self, batch: Any) -> dict[str, Any]:
        self.model.train()
        batch = batch.to(self.device)
        self.optimizer.zero_grad(set_to_none=True)
        loss, metrics = self._compute_loss_and_metrics(batch)
        loss.backward()
        if self.config.grad_clip and self.config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        self.optimizer.step()
        return metrics

    def _compute_loss_and_metrics(self, batch: Any) -> tuple[torch.Tensor, dict[str, Any]]:
        outputs = self.model(batch)
        reg_alpha, reg_beta = outputs[0], outputs[1]
        source_labels = self._source_labels(batch, len(reg_alpha))
        beta_loss = beta_loss_with_mixed_targets(
            reg_alpha,
            reg_beta,
            batch.reg_y.view(-1),
            batch.cls_y.view(-1),
            min_pKi=self.config.pki_min,
            max_pKi=self.config.pki_max,
            threshold=self.config.classification_threshold,
            source_labels=source_labels,
            source_priors=self.source_priors if source_labels is not None else None,
            regul_lambda=self.config.source_constraint_weight,
        )
        orthogonality_loss = self._auxiliary_loss(outputs, 2, reg_alpha)
        mincut_loss = self._auxiliary_loss(outputs, 3, reg_alpha)
        loss = (
            beta_loss
            + self.config.mincut_loss_weight * mincut_loss
            + self.config.orthogonality_loss_weight * orthogonality_loss
        )
        metrics = self._metrics_from_batch(
            batch,
            reg_alpha,
            reg_beta,
            loss,
            beta_loss,
            mincut_loss,
            orthogonality_loss,
        )
        return loss, metrics

    def _auxiliary_loss(self, outputs: Any, index: int, reference: torch.Tensor) -> torch.Tensor:
        if len(outputs) <= index or outputs[index] is None:
            return reference.new_zeros(())
        value = outputs[index]
        if not torch.is_tensor(value):
            value = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
        else:
            value = value.to(device=reference.device, dtype=reference.dtype)
        return value.mean()

    def _source_labels(self, batch: Any, expected: int) -> list[str] | None:
        raw_labels = getattr(batch, "source_y", None)
        if raw_labels is None:
            return None
        if isinstance(raw_labels, torch.Tensor):
            labels = [str(value.item()) for value in raw_labels.detach().cpu().flatten()]
        elif isinstance(raw_labels, str):
            labels = [raw_labels]
        else:
            labels = [str(value) for value in raw_labels]
        labels = [self._normal_source_label(label) for label in labels]
        if len(labels) != expected:
            return None
        for label in labels:
            self.source_priors.setdefault(label, dict(self.config.default_source_prior))
        return labels

    def _metrics_from_batch(
        self,
        batch: Any,
        reg_alpha: torch.Tensor,
        reg_beta: torch.Tensor,
        loss: torch.Tensor,
        beta_loss: torch.Tensor,
        mincut_loss: torch.Tensor,
        orthogonality_loss: torch.Tensor,
    ) -> dict[str, Any]:
        labels = batch.reg_y.view(-1).to(reg_alpha.device)
        finite_mask = torch.isfinite(labels)
        point = prediction_from_beta(reg_alpha, reg_beta, self.config.pki_min, self.config.pki_max)
        mae = None
        rmse = None
        if finite_mask.any():
            errors = point[finite_mask] - labels[finite_mask]
            mae = float(errors.abs().mean().detach().cpu().item())
            rmse = float(torch.sqrt(torch.mean(errors.square())).detach().cpu().item())
        phi = reg_alpha + reg_beta
        return {
            "loss": float(loss.detach().cpu().item()),
            "beta_loss": float(beta_loss.detach().cpu().item()),
            "mincut_loss": float(mincut_loss.detach().cpu().item()),
            "orthogonality_loss": float(orthogonality_loss.detach().cpu().item()),
            "mae": mae,
            "rmse": rmse,
            "rows": int(reg_alpha.numel()),
            "mean_phi": float(phi.detach().mean().cpu().item()),
            "max_phi": float(phi.detach().max().cpu().item()),
        }

    def _next_train_batch(self, train_iterator: Any) -> tuple[Any, Any]:
        for _ in range(100):
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(self.train_loader)
                batch = next(train_iterator)
            if batch is not None:
                return batch, train_iterator
        raise RuntimeError("Could not retrieve a non-empty training batch after 100 attempts.")

    def _init_metrics_file(self) -> None:
        if self.metrics_path.exists():
            return
        with self.metrics_path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "step",
                    "train_loss",
                    "train_mae",
                    "train_rmse",
                    "train_rows",
                    "train_beta_loss",
                    "train_mincut_loss",
                    "train_orthogonality_loss",
                    "train_mean_phi",
                    "train_max_phi",
                    "valid_loss",
                    "valid_mae",
                    "valid_rmse",
                    "valid_rows",
                    "valid_beta_loss",
                    "valid_mincut_loss",
                    "valid_orthogonality_loss",
                ],
            )
            writer.writeheader()

    def _append_metrics(
        self,
        step: int,
        train_metrics: Mapping[str, Any],
        valid_metrics: Mapping[str, Any] | None,
    ) -> None:
        row = {
            "step": step,
            "train_loss": train_metrics.get("loss"),
            "train_mae": train_metrics.get("mae"),
            "train_rmse": train_metrics.get("rmse"),
            "train_rows": train_metrics.get("rows"),
            "train_beta_loss": train_metrics.get("beta_loss"),
            "train_mincut_loss": train_metrics.get("mincut_loss"),
            "train_orthogonality_loss": train_metrics.get("orthogonality_loss"),
            "train_mean_phi": train_metrics.get("mean_phi"),
            "train_max_phi": train_metrics.get("max_phi"),
            "valid_loss": valid_metrics.get("loss") if valid_metrics else None,
            "valid_mae": valid_metrics.get("mae") if valid_metrics else None,
            "valid_rmse": valid_metrics.get("rmse") if valid_metrics else None,
            "valid_rows": valid_metrics.get("rows") if valid_metrics else None,
            "valid_beta_loss": valid_metrics.get("beta_loss") if valid_metrics else None,
            "valid_mincut_loss": valid_metrics.get("mincut_loss") if valid_metrics else None,
            "valid_orthogonality_loss": valid_metrics.get("orthogonality_loss") if valid_metrics else None,
        }
        with self.metrics_path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            writer.writerow(row)

    def _serializable_config(self) -> dict[str, Any]:
        payload = asdict(self.config)
        payload["output_dir"] = str(payload["output_dir"])
        payload["adam_betas"] = list(payload["adam_betas"])
        return payload

    @staticmethod
    def _normal_source_label(label: str) -> str:
        if label.lower() in {"", "nan", "none", "<na>"}:
            return "unknown"
        return label

    @staticmethod
    def _mean(values: list[float]) -> float | None:
        if not values:
            return None
        return float(sum(values) / len(values))

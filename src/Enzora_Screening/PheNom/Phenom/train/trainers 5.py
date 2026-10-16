from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F

from models.enzyme_prediction import (
    KCAT_DIM,
    KCAT_KM,
    KM_DIM,
    MUTATION,
    FITNESS_DIM,
)


SIGMA_FLOOR = 1e-3
PROBABILITY_EPSILON = 1e-7


def gaussian_nll_loss(
    target: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    variance = sigma.double().square()
    residual = target.double() - mu.double()
    loss = 0.5 * (variance.log() + residual.square() / variance + math.log(2.0 * math.pi))
    return (loss * weight.double()).mean()


def gaussian_probability_above_zero(mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    probability = 0.5 * (1.0 + torch.erf(mu.double() / (sigma.double() * math.sqrt(2.0))))
    return probability.clamp(PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON)


def select_task_distribution(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    task_index: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    task_index = task_index.long().view(-1)
    dimensions = torch.tensor((0, 1, 0, 2, 3, 4), device=mu.device)
    rows = torch.arange(mu.size(0), device=mu.device)
    selected_mu = mu[rows, dimensions[task_index]]
    selected_sigma = sigma[rows, dimensions[task_index]]
    ratio_mask = task_index == KCAT_KM
    selected_mu = torch.where(ratio_mask, mu[:, KCAT_DIM] - mu[:, KM_DIM], selected_mu)
    selected_sigma = torch.where(
        ratio_mask,
        torch.hypot(sigma[:, KCAT_DIM], sigma[:, KM_DIM]),
        selected_sigma,
    )
    return selected_mu, selected_sigma


def mutation_distribution(
    mutant_mu: torch.Tensor,
    mutant_sigma: torch.Tensor,
    wt_mu: torch.Tensor,
    wt_sigma: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        mutant_mu[:, FITNESS_DIM] - wt_mu[:, FITNESS_DIM],
        torch.hypot(mutant_sigma[:, FITNESS_DIM], wt_sigma[:, FITNESS_DIM]),
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


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


class PHENOMETrainer:
    def __init__(self, model, config: TrainerConfig, device: str) -> None:
        self.model = model
        self.config = config
        self.device = torch.device(device)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=config.adam_betas,
            eps=config.adam_eps,
        )

    def batch_loss(self, batch: dict) -> tuple[torch.Tensor, dict[str, float], dict[str, np.ndarray]]:
        primary = batch["primary"].to(self.device)
        protein_only = batch["protein_only"]
        mu, sigma, ortho, mincut, _ = self.model(primary, protein_only=protein_only)
        task = int(primary.task_index.view(-1)[0])
        task_index = torch.full((mu.size(0),), task, dtype=torch.long, device=self.device)
        target = primary.label.view(-1)
        is_binary = primary.is_binary.view(-1)

        if task == MUTATION:
            reference = batch["reference"].to(self.device)
            wt_mu, wt_sigma, wt_ortho, wt_mincut, _ = self.model(reference, protein_only=protein_only)
            selected_mu, selected_sigma = mutation_distribution(mu, sigma, wt_mu, wt_sigma)
            ortho = 0.5 * (ortho + wt_ortho)
            mincut = 0.5 * (mincut + wt_mincut)
        else:
            selected_mu, selected_sigma = select_task_distribution(mu, sigma, task_index)

        if bool(is_binary[0]):
            with torch.autocast(device_type=self.device.type, enabled=False):
                prediction = gaussian_probability_above_zero(selected_mu, selected_sigma)
                supervised_loss = F.binary_cross_entropy(prediction, target.double())
        else:
            prediction = selected_mu
            with torch.autocast(device_type=self.device.type, enabled=False):
                supervised_loss = gaussian_nll_loss(
                    target,
                    selected_mu,
                    selected_sigma,
                    primary.sample_weight.view(-1),
                )

        ortho = ortho.mean()
        mincut = mincut.mean()
        loss = (
            supervised_loss
            + self.config.orthogonality_loss_weight * ortho
            + self.config.mincut_loss_weight * mincut
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite task loss: task={int(task_index[0])}, "
                f"supervised={float(supervised_loss.detach().float().cpu())}, "
                f"orthogonality={float(ortho.detach().float().cpu())}, "
                f"mincut={float(mincut.detach().float().cpu())}, "
                f"finite_mu={int(torch.isfinite(selected_mu).sum())}/{selected_mu.numel()}, "
                f"mu_range=({float(selected_mu.detach().float().min().cpu())},"
                f"{float(selected_mu.detach().float().max().cpu())}), "
                f"finite_sigma={int(torch.isfinite(selected_sigma).sum())}/{selected_sigma.numel()}, "
                f"sigma_range=({float(selected_sigma.detach().float().min().cpu())},"
                f"{float(selected_sigma.detach().float().max().cpu())}), "
                f"target_range=({float(target.detach().float().min().cpu())},"
                f"{float(target.detach().float().max().cpu())})"
            )
        metrics = {
            "loss": float(loss.detach().float().cpu()),
            "supervised_loss": float(supervised_loss.detach().float().cpu()),
            "orthogonality_loss": float(ortho.detach().float().cpu()),
            "mincut_loss": float(mincut.detach().float().cpu()),
            "mean_sigma": float(selected_sigma.detach().float().mean().cpu()),
            "task_index": task,
        }
        arrays = {
            "target": target.detach().float().cpu().numpy(),
            "prediction": prediction.detach().float().cpu().numpy(),
            "mu": selected_mu.detach().float().cpu().numpy(),
            "sigma": selected_sigma.detach().float().cpu().numpy(),
            "is_binary": is_binary.detach().cpu().numpy(),
            "task_index": np.full(target.numel(), task),
        }
        return loss, metrics, arrays

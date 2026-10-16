from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F

from models.enzyme_prediction import ESP, FITNESS_DIM, MUTATION, task_distribution


SIGMA_FLOOR = 0.01
PROBABILITY_EPSILON = 1e-7
KINETIC_TASKS = (0, 1, 2)


def gaussian_probability_above_zero(mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    probability = 0.5 * (1.0 + torch.erf(mu.double() / (sigma.double() * math.sqrt(2.0))))
    return probability.clamp(PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON)


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


def pairwise_rank_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    tau: float,
    minimum_delta: float,
    sigma: torch.Tensor | None = None,
    pair_mask: torch.Tensor | None = None,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    rows, columns = torch.triu_indices(target.numel(), target.numel(), offset=1, device=target.device)
    target_delta = target[rows] - target[columns]
    keep = target_delta.abs() >= minimum_delta
    if pair_mask is not None:
        keep = keep & pair_mask[rows, columns]
    if not keep.any():
        return prediction.sum() * 0.0
    signed_prediction_delta = target_delta[keep].sign() * (prediction[rows[keep]] - prediction[columns[keep]])
    weights = target_delta[keep].abs().clamp(0.2, 2.0)
    if sample_weight is not None:
        weights = weights * torch.sqrt(sample_weight[rows[keep]] * sample_weight[columns[keep]])
    if sigma is None:
        pair_loss = F.softplus(-signed_prediction_delta / tau)
    else:
        pair_scale = torch.sqrt(
            sigma[rows[keep]].square() + sigma[columns[keep]].square() + tau**2
        )
        pair_loss = -torch.special.log_ndtr(signed_prediction_delta / pair_scale)
    return (pair_loss * weights).sum() / weights.sum()


@dataclass
class TrainerConfig:
    output_dir: Path
    total_steps: int = 200_000
    eval_every: int = 2_000
    save_every: int = 10_000
    learning_rate: float = 1e-5
    global_learning_rate: float = 5e-5
    head_learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    adam_eps: float = 1e-5
    adam_betas: tuple[float, float] = (0.9, 0.999)
    grad_clip: float = 1.0
    mincut_loss_weight: float = 0.02
    orthogonality_loss_weight: float = 0.02
    global_rank_loss_weight: float = 0.15
    same_ligand_rank_loss_weight: float = 0.25
    same_protein_rank_loss_weight: float = 0.25
    rank_tau: float = 0.5
    rank_min_delta: float = 0.2
    rank_loss_mode: str = "margin"
    mean_loss_mode: str = "huber"
    binary_loss_mode: str = "gaussian_cdf"
    sigma_loss_weight: float = 0.1


class PHENOMETrainer:
    def __init__(self, model, config: TrainerConfig, device: str, scalers: Mapping[str, Any]) -> None:
        self.model = model
        self.config = config
        self.device = torch.device(device)
        self.centers = torch.zeros(6, dtype=torch.float64, device=self.device)
        self.scales = torch.ones(6, dtype=torch.float64, device=self.device)
        for scaler in scalers.values():
            task = int(scaler["task_index"])
            self.centers[task] = float(scaler["median"])
            self.scales[task] = float(scaler["iqr"])

        groups = {"backbone": [], "global": [], "head": []}
        for name, parameter in model.named_parameters():
            if name.startswith("prediction_head"):
                groups["head"].append(parameter)
            elif name.startswith((
                "protein_global", "ligand_global", "ligand_language_global", "fusion",
                "catalytic_",
            )):
                groups["global"].append(parameter)
            else:
                groups["backbone"].append(parameter)
        base_lrs = {
            "backbone": config.learning_rate,
            "global": config.global_learning_rate,
            "head": config.head_learning_rate,
        }
        self.optimizer = torch.optim.AdamW(
            [
                {"params": parameters, "lr": base_lrs[name], "base_lr": base_lrs[name]}
                for name, parameters in groups.items()
            ],
            weight_decay=config.weight_decay,
            betas=config.adam_betas,
            eps=config.adam_eps,
        )

    def set_learning_rate_scale(self, scale: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = group["base_lr"] * scale

    def batch_loss(
        self,
        batch: dict,
        training_phase: str = "mean",
        graph_weight_scale: float = 0.0,
        rank_weight_scale: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, float], dict[str, np.ndarray]]:
        primary = batch["primary"].to(self.device)
        protein_only = batch["protein_only"]
        mu, sigma, ortho, mincut, attention = self.model(primary, protein_only=protein_only)
        task = int(primary.task_index.view(-1)[0])
        task_index = torch.full((mu.size(0),), task, dtype=torch.long, device=self.device)
        target_raw = primary.label.view(-1).double()
        target = (target_raw - self.centers[task]) / self.scales[task]
        is_binary = primary.is_binary.view(-1)

        if task == MUTATION:
            reference = batch["reference"].to(self.device)
            wt_mu, wt_sigma, wt_ortho, wt_mincut, _ = self.model(reference, protein_only=protein_only)
            selected_mu, selected_sigma = mutation_distribution(mu, sigma, wt_mu, wt_sigma)
            ortho = 0.5 * (ortho + wt_ortho)
            mincut = 0.5 * (mincut + wt_mincut)
        else:
            selected_mu, selected_sigma = task_distribution(mu, sigma, task_index)

        zero = selected_mu.sum() * 0.0
        global_rank_loss = zero
        same_ligand_rank_loss = zero
        same_protein_rank_loss = zero
        if training_phase == "sigma":
            if bool(is_binary[0]):
                prediction = gaussian_probability_above_zero(selected_mu, selected_sigma)
                supervised_loss = self.config.sigma_loss_weight * F.binary_cross_entropy(
                    prediction, target_raw
                )
            else:
                true_error = (target - selected_mu.double()).abs().detach()
                supervised_loss = self.config.sigma_loss_weight * (
                    F.smooth_l1_loss(selected_sigma.double(), true_error, beta=0.5, reduction="none")
                    * primary.sample_weight.view(-1).double()
                ).mean()
                prediction = selected_mu
        elif bool(is_binary[0]):
            with torch.autocast(device_type=self.device.type, enabled=False):
                if self.config.binary_loss_mode == "logits":
                    supervised_loss = F.binary_cross_entropy_with_logits(
                        selected_mu.double(), target_raw
                    )
                    prediction = torch.sigmoid(selected_mu.double())
                else:
                    prediction = gaussian_probability_above_zero(selected_mu, selected_sigma)
                    supervised_loss = F.binary_cross_entropy(prediction, target_raw)
        else:
            prediction = selected_mu
            if self.config.mean_loss_mode == "mse":
                point_loss = F.mse_loss(selected_mu.double(), target, reduction="none")
            else:
                point_loss = F.smooth_l1_loss(
                    selected_mu.double(), target, beta=0.5, reduction="none"
                )
            mean_loss = (point_loss * primary.sample_weight.view(-1).double()).mean()
            if task in KINETIC_TASKS and rank_weight_scale:
                protein_group = primary.protein_group_index.view(-1)
                ligand_group = primary.ligand_group_index.view(-1)
                ranking_role = primary.ranking_role.view(-1)
                global_pairs = ranking_role[:, None].eq(0) & ranking_role[None, :].eq(0)
                same_ligand = (
                    ranking_role[:, None].eq(1) & ranking_role[None, :].eq(1)
                    & ligand_group[:, None].eq(ligand_group[None, :])
                )
                same_protein = (
                    ranking_role[:, None].eq(2) & ranking_role[None, :].eq(2)
                    & protein_group[:, None].eq(protein_group[None, :])
                )
                sample_weight = primary.sample_weight.view(-1).double()
                rank_sigma = (
                    selected_sigma.double().detach()
                    if self.config.rank_loss_mode == "gaussian"
                    else None
                )
                global_rank_loss = pairwise_rank_loss(
                    selected_mu.double(),
                    target_raw,
                    self.config.rank_tau,
                    self.config.rank_min_delta,
                    rank_sigma,
                    global_pairs,
                    sample_weight,
                )
                same_ligand_rank_loss = pairwise_rank_loss(
                    selected_mu.double(), target_raw, self.config.rank_tau,
                    self.config.rank_min_delta, rank_sigma, same_ligand, sample_weight
                )
                same_protein_rank_loss = pairwise_rank_loss(
                    selected_mu.double(), target_raw, self.config.rank_tau,
                    self.config.rank_min_delta, rank_sigma, same_protein, sample_weight
                )
            supervised_loss = mean_loss + rank_weight_scale * (
                self.config.global_rank_loss_weight * global_rank_loss
                + self.config.same_ligand_rank_loss_weight * same_ligand_rank_loss
                + self.config.same_protein_rank_loss_weight * same_protein_rank_loss
            )

        ortho = ortho.mean()
        mincut = mincut.mean()
        loss = supervised_loss
        if training_phase == "mean" and graph_weight_scale:
            loss = loss + graph_weight_scale * (
                self.config.orthogonality_loss_weight * ortho
                + self.config.mincut_loss_weight * mincut
            )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite task loss: task={task}")

        raw_mu = selected_mu.double() * self.scales[task] + self.centers[task]
        raw_sigma = selected_sigma.double() * self.scales[task]
        metrics = {
            "loss": float(loss.detach().float().cpu()),
            "supervised_loss": float(supervised_loss.detach().float().cpu()),
            "global_rank_loss": float(global_rank_loss.detach().float().cpu()),
            "same_ligand_rank_loss": float(same_ligand_rank_loss.detach().float().cpu()),
            "same_protein_rank_loss": float(same_protein_rank_loss.detach().float().cpu()),
            "orthogonality_loss": float(ortho.detach().float().cpu()),
            "mincut_loss": float(mincut.detach().float().cpu()),
            "mean_sigma": float(raw_sigma.detach().float().mean().cpu()),
            "task_index": task,
        }
        arrays = {
            "target": target_raw.detach().float().cpu().numpy(),
            "prediction": prediction.detach().float().cpu().numpy() if bool(is_binary[0]) else raw_mu.detach().float().cpu().numpy(),
            "mu": raw_mu.detach().float().cpu().numpy(),
            "sigma": raw_sigma.detach().float().cpu().numpy(),
            "is_binary": is_binary.detach().cpu().numpy(),
            "task_index": np.full(target_raw.numel(), task),
        }
        return loss, metrics, arrays

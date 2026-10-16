from __future__ import annotations

import math

import torch
from torch import nn


KCAT = 0
KM = 1
KCAT_KM = 2
PH = 3
ESP = 4
MUTATION = 5

TASK_NAMES = ("kcat", "km", "kcat_km", "ph", "esp", "mutation")
TASK_TO_INDEX = {name: index for index, name in enumerate(TASK_NAMES)}

# Internal Gaussian dimensions. kcat/Km and mutation effects are derived.
KCAT_DIM = 0
KM_DIM = 1
PH_DIM = 2
ESP_DIM = 3
FITNESS_DIM = 4
NUM_BASE_OUTPUTS = 5


class EnzymePredictionHead(nn.Module):
    def __init__(
        self, input_channels: int, hidden_channels: int, eps: float = 1e-6, layout: str = "unified"
    ) -> None:
        super().__init__()
        self.layout = layout
        if layout == "unified":
            self.mu = nn.Sequential(
                nn.Linear(input_channels, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, NUM_BASE_OUTPUTS)
            )
            self.raw_sigma = nn.Sequential(
                nn.Linear(input_channels, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, NUM_BASE_OUTPUTS)
            )
        else:
            for task in TASK_NAMES[:2] + TASK_NAMES[3:]:
                setattr(
                    self,
                    f"{task}_mu",
                    nn.Sequential(
                        nn.Linear(input_channels, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, 1)
                    ),
                )
                setattr(
                    self,
                    f"{task}_raw_sigma",
                    nn.Sequential(
                        nn.Linear(input_channels, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, 1)
                    ),
                )
        self.eps = eps

    def reset_parameters(self) -> None:
        heads = (self.mu, self.raw_sigma) if self.layout == "unified" else (
            self.kcat_mu,
            self.kcat_raw_sigma,
            self.km_mu,
            self.km_raw_sigma,
            self.ph_mu,
            self.ph_raw_sigma,
            self.esp_mu,
            self.esp_raw_sigma,
            self.mutation_mu,
            self.mutation_raw_sigma,
        )
        for head in heads:
            for module in head:
                if hasattr(module, "reset_parameters"):
                    module.reset_parameters()

    def forward(
        self,
        fingerprint: torch.Tensor,
        task_index: torch.Tensor | None = None,
        all_task_outputs: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.layout == "unified":
            mu = self.mu(fingerprint)
            raw_sigma = self.raw_sigma(fingerprint)
        elif all_task_outputs:
            mu = torch.cat(
                tuple(getattr(self, f"{task}_mu")(fingerprint) for task in TASK_NAMES[:2] + TASK_NAMES[3:]),
                dim=1,
            )
            raw_sigma = torch.cat(
                tuple(
                    getattr(self, f"{task}_raw_sigma")(fingerprint)
                    for task in TASK_NAMES[:2] + TASK_NAMES[3:]
                ),
                dim=1,
            )
        else:
            task = int(task_index.view(-1)[0])
            task_name = TASK_NAMES[task]
            output_dimension = {KCAT: 0, KM: 1, PH: 2, ESP: 3, MUTATION: 4}[task]
            mu = fingerprint.new_zeros((fingerprint.size(0), NUM_BASE_OUTPUTS))
            raw_sigma = fingerprint.new_zeros((fingerprint.size(0), NUM_BASE_OUTPUTS))
            mu[:, output_dimension] = getattr(self, f"{task_name}_mu")(fingerprint).view(-1)
            raw_sigma[:, output_dimension] = getattr(self, f"{task_name}_raw_sigma")(fingerprint).view(-1)
        return mu, torch.nn.functional.softplus(raw_sigma) + self.eps


def task_distribution(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    task_index: int | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(task_index, int):
        task_index = torch.full((mu.size(0),), task_index, dtype=torch.long, device=mu.device)
    else:
        task_index = task_index.to(device=mu.device, dtype=torch.long).view(-1)

    selected_mu = mu.new_empty(task_index.shape)
    selected_sigma = sigma.new_empty(task_index.shape)
    base_dimensions = {KCAT: KCAT_DIM, KM: KM_DIM, PH: PH_DIM, ESP: ESP_DIM, MUTATION: FITNESS_DIM}
    for task, dimension in base_dimensions.items():
        mask = task_index == task
        selected_mu[mask] = mu[mask, dimension]
        selected_sigma[mask] = sigma[mask, dimension]

    ratio_mask = task_index == KCAT_KM
    selected_mu[ratio_mask] = mu[ratio_mask, KCAT_DIM] - mu[ratio_mask, KM_DIM]
    selected_sigma[ratio_mask] = torch.sqrt(
        sigma[ratio_mask, KCAT_DIM].square() + sigma[ratio_mask, KM_DIM].square()
    )
    return selected_mu, selected_sigma


def mutation_distribution(
    mutant_mu: torch.Tensor,
    mutant_sigma: torch.Tensor,
    wt_mu: torch.Tensor,
    wt_sigma: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    delta_mu = mutant_mu[:, FITNESS_DIM] - wt_mu[:, FITNESS_DIM]
    delta_sigma = torch.sqrt(mutant_sigma[:, FITNESS_DIM].square() + wt_sigma[:, FITNESS_DIM].square())
    return delta_mu, delta_sigma


def probability_above_zero(mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(mu / (sigma * math.sqrt(2.0))))


def gaussian_nll(
    target: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    loss = torch.log(sigma) + 0.5 * ((target - mu) / sigma).square() + 0.5 * math.log(2.0 * math.pi)
    if weight is not None:
        loss = loss * weight
    return loss.mean()

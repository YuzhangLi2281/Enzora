from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from models.enzyme_prediction import (
    ESP,
    ESP_DIM,
    KCAT,
    KCAT_DIM,
    KCAT_KM,
    KM,
    KM_DIM,
    MUTATION,
    PH,
    TASK_TO_INDEX,
    mutation_distribution,
    probability_above_zero,
    task_distribution,
)


def prepare_inference_frame(frame: pd.DataFrame, task: str) -> pd.DataFrame:
    required = ["Protein"]
    if task in {"kcat", "km", "kcat_km", "esp", "all"}:
        required.append("Ligand")
    if task == "mutation":
        required.append("WT_Protein")
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"Missing columns for {task}: {missing}")

    result = frame.copy()
    if "ID" not in result:
        result["ID"] = [f"Row_{index + 1}" for index in range(len(result))]
    result["task_index"] = KCAT if task == "all" else TASK_TO_INDEX[task]
    result["label"] = 0.0
    result["is_binary"] = task in {"esp", "mutation"}
    result["sample_weight"] = 1.0
    result["source"] = "inference"
    return result


def format_prediction(
    task: str,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    wt_mu: torch.Tensor | None = None,
    wt_sigma: torch.Tensor | None = None,
) -> dict[str, np.ndarray]:
    if task == "all":
        ratio_mu = mu[:, KCAT_DIM] - mu[:, KM_DIM]
        ratio_sigma = torch.sqrt(sigma[:, KCAT_DIM].square() + sigma[:, KM_DIM].square())
        return {
            **_kinetic_columns("kcat", mu[:, KCAT_DIM], sigma[:, KCAT_DIM]),
            **_kinetic_columns("km", mu[:, KM_DIM], sigma[:, KM_DIM]),
            **_kinetic_columns("kcat_km", ratio_mu, ratio_sigma),
            **_esp_columns(mu[:, ESP_DIM], sigma[:, ESP_DIM]),
        }
    if task in {"kcat", "km", "kcat_km"}:
        selected_mu, selected_sigma = task_distribution(mu, sigma, TASK_TO_INDEX[task])
        return _kinetic_columns("", selected_mu, selected_sigma)
    if task == "ph":
        selected_mu, selected_sigma = task_distribution(mu, sigma, PH)
        return {
            "predicted_ph": _numpy(selected_mu),
            "sigma_ph": _numpy(selected_sigma),
            "lower_95": _numpy(selected_mu - 1.96 * selected_sigma),
            "upper_95": _numpy(selected_mu + 1.96 * selected_sigma),
        }
    if task == "esp":
        selected_mu, selected_sigma = task_distribution(mu, sigma, ESP)
        columns = _esp_columns(selected_mu, selected_sigma)
        columns["predicted_class"] = columns.pop("esp_predicted_class")
        return columns
    if wt_mu is None or wt_sigma is None:
        raise ValueError("Mutation prediction requires WT outputs")
    delta_mu, delta_sigma = mutation_distribution(mu, sigma, wt_mu, wt_sigma)
    probability = probability_above_zero(delta_mu, delta_sigma)
    return {
        "delta_mu": _numpy(delta_mu),
        "delta_sigma": _numpy(delta_sigma),
        "better_probability": _numpy(probability),
        "predicted_class": _numpy((probability >= 0.5).long()),
    }


def _kinetic_columns(prefix: str, mu: torch.Tensor, sigma: torch.Tensor) -> dict[str, np.ndarray]:
    stem = f"{prefix}_" if prefix else ""
    return {
        f"{stem}mu_log10": _numpy(mu),
        f"{stem}sigma_log10": _numpy(sigma),
        f"{stem}prediction": _numpy(torch.pow(10.0, mu)),
        f"{stem}lower_95": _numpy(torch.pow(10.0, mu - 1.96 * sigma)),
        f"{stem}upper_95": _numpy(torch.pow(10.0, mu + 1.96 * sigma)),
    }


def _esp_columns(mu: torch.Tensor, sigma: torch.Tensor) -> dict[str, np.ndarray]:
    probability = probability_above_zero(mu, sigma)
    return {
        "esp_latent_mu": _numpy(mu),
        "esp_latent_sigma": _numpy(sigma),
        "activity_probability": _numpy(probability),
        "esp_predicted_class": _numpy((probability >= 0.5).long()),
    }


def _numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()

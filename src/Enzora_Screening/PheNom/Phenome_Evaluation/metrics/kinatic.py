import math

import numpy as np
from scipy.special import ndtr
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(frame):
    target = frame["target"].to_numpy(float)
    prediction = frame["prediction"].to_numpy(float)
    error = prediction - target
    pearson = float(pearsonr(target, prediction).statistic)
    spearman = float(spearmanr(target, prediction).statistic)
    metrics = {
        "n": int(len(frame)),
        "mae": float(mean_absolute_error(target, prediction)),
        "median_ae": float(np.median(np.abs(error))),
        "rmse": float(mean_squared_error(target, prediction) ** 0.5),
        "r2": float(r2_score(target, prediction)),
        "pearson": pearson if math.isfinite(pearson) else None,
        "spearman": spearman if math.isfinite(spearman) else None,
    }
    if "sigma" in frame:
        sigma = frame["sigma"].to_numpy(float)
        metrics.update({
            "gaussian_nll": float(np.mean(0.5 * (np.log(2 * math.pi * sigma**2) + error**2 / sigma**2))),
            "mean_sigma": float(sigma.mean()),
            "coverage_50": float(np.mean(np.abs(error) <= 0.67448975 * sigma)),
            "coverage_90": float(np.mean(np.abs(error) <= 1.64485363 * sigma)),
            "coverage_95": float(np.mean(np.abs(error) <= 1.95996398 * sigma)),
        })
    return metrics


def mutant_ranking_metrics(frame):
    ranking = {}
    for task, task_frame in frame.groupby("task", sort=True):
        task_ranking = {}
        for minimum_size in (20, 30):
            correlations = []
            concordant = 0
            comparable = 0
            groups = 0
            for _, group in task_frame.groupby("group_id"):
                if len(group) < minimum_size:
                    continue
                groups += 1
                target = group["target"].to_numpy(float)
                prediction = group["prediction"].to_numpy(float)
                correlation = float(spearmanr(target, prediction).statistic)
                if math.isfinite(correlation):
                    correlations.append(correlation)
                target_delta = target[:, None] - target[None, :]
                prediction_delta = prediction[:, None] - prediction[None, :]
                upper = np.triu(np.ones(target_delta.shape, dtype=bool), 1)
                mask = upper & (target_delta != 0) & (prediction_delta != 0)
                concordant += int(np.sum(target_delta[mask] * prediction_delta[mask] > 0))
                comparable += int(mask.sum())
            task_ranking[f"n_ge_{minimum_size}"] = {
                "groups": groups,
                "mean_group_spearman": float(np.mean(correlations)) if correlations else None,
                "pairwise_accuracy": concordant / comparable if comparable else None,
                "comparable_pairs": comparable,
            }
        ranking[task] = task_ranking
    return ranking


def uncertainty_ranking_metrics(frame):
    target = frame["target"].to_numpy(float)
    mu = frame["prediction"].to_numpy(float)
    sigma = frame["sigma"].to_numpy(float)
    borda = np.empty(len(frame), dtype=float)
    for start in range(0, len(frame), 512):
        stop = min(start + 512, len(frame))
        pair_scale = np.sqrt(sigma[start:stop, None] ** 2 + sigma[None, :] ** 2)
        borda[start:stop] = ndtr((mu[start:stop, None] - mu[None, :]) / pair_scale).mean(axis=1)
    result = {
        "mu": float(spearmanr(target, mu).statistic),
        "confidence_only_negative_sigma": float(spearmanr(target, -sigma).statistic),
        "lower_90_bound": float(spearmanr(target, mu - 1.64485363 * sigma).statistic),
        "upper_90_bound": float(spearmanr(target, mu + 1.64485363 * sigma).statistic),
        "gaussian_pairwise_borda": float(spearmanr(target, borda).statistic),
        "lowest_sigma_coverage": {},
    }
    order = np.argsort(sigma)
    for coverage in (0.25, 0.5, 0.75):
        keep = order[:round(len(frame) * coverage)]
        result["lowest_sigma_coverage"][str(coverage)] = {
            "n": int(len(keep)),
            "spearman": float(spearmanr(target[keep], mu[keep]).statistic),
            "mae": float(mean_absolute_error(target[keep], mu[keep])),
        }
    return result


def evaluate(frame):
    parameters = {
        task: regression_metrics(group)
        for task, group in frame.groupby("task", sort=True)
    }
    common_metrics = set.intersection(*(set(metrics) for metrics in parameters.values())) - {"n"}
    result = {
        "n": int(len(frame)),
        "macro_average": {
            metric: float(np.mean([metrics[metric] for metrics in parameters.values()]))
            for metric in sorted(common_metrics)
            if all(metrics[metric] is not None for metrics in parameters.values())
        },
        "parameters": parameters,
    }
    if "group_id" in frame:
        result["mutant_ranking"] = mutant_ranking_metrics(frame)
    if "sigma" in frame:
        result["uncertainty_ranking"] = {
            task: uncertainty_ranking_metrics(group)
            for task, group in frame.groupby("task", sort=True)
        }
    return result

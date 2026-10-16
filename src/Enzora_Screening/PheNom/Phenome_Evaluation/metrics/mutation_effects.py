import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def binary_metrics(target, score, threshold):
    prediction = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(target, prediction, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "decision_rule": "prediction >= threshold",
        "accuracy": float(accuracy_score(target, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(target, prediction)),
        "f1": float(f1_score(target, prediction, zero_division=0)),
        "mcc": float(matthews_corrcoef(target, prediction)),
        "precision": float(precision_score(target, prediction, zero_division=0)),
        "recall": float(recall_score(target, prediction, zero_division=0)),
        "specificity": float(tn / (tn + fp)),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def evaluate(frame, threshold=0.5, positive_prevalence=None):
    target = frame["target"].to_numpy(int)
    score = frame["prediction"].to_numpy(float)
    clipped = np.clip(score, 1e-7, 1 - 1e-7)
    calibration_bin = np.minimum((score * 10).astype(int), 9)
    calibration = [
        {
            "bin": index,
            "count": int(mask.sum()),
            "mean_score": float(score[mask].mean()),
            "positive_rate": float(target[mask].mean()),
        }
        for index in range(10)
        if (mask := calibration_bin == index).any()
    ]
    result = {
        "n": int(len(frame)),
        "positives": int(target.sum()),
        "negatives": int((target == 0).sum()),
        "auroc": float(roc_auc_score(target, score)),
        "auprc": float(average_precision_score(target, score)),
        "brier": float(brier_score_loss(target, score)),
        "log_loss": float(log_loss(target, clipped, labels=[0, 1])),
        "ece_10": float(sum(row["count"] * abs(row["mean_score"] - row["positive_rate"]) for row in calibration) / len(frame)),
        "calibration_bins": calibration,
        "classification_at_threshold": binary_metrics(target, score, threshold),
    }
    if positive_prevalence is not None:
        calibrated_threshold = float(np.quantile(score, 1 - positive_prevalence, method="midpoint"))
        result["classification_at_predeclared_prevalence"] = {
            "declared_positive_prevalence": positive_prevalence,
            **binary_metrics(target, score, calibrated_threshold),
        }
    if "sigma" in frame:
        result["mean_sigma"] = float(frame["sigma"].mean())
    if "mutation_count" in frame:
        result["by_mutation_count"] = {
            str(count): evaluate(group.drop(columns="mutation_count"), threshold, positive_prevalence)
            for count, group in frame.groupby("mutation_count", sort=True)
            if group["target"].nunique() == 2
        }
    return result


def evaluate_dms(frame):
    datasets = {}
    for name, group in frame.groupby("source", sort=True):
        target = group["target"].to_numpy(float)
        prediction = group["prediction"].to_numpy(float)
        datasets[name] = {
            "n": int(len(group)),
            "pearson": float(pearsonr(target, prediction).statistic),
            "spearman": float(spearmanr(target, prediction).statistic),
        }
    return {
        "rows": int(len(frame)),
        "datasets": datasets,
        "macro_pearson": float(np.mean([result["pearson"] for result in datasets.values()])),
        "macro_spearman": float(np.mean([result["spearman"] for result in datasets.values()])),
    }

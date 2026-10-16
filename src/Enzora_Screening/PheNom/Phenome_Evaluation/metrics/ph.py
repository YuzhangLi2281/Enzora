import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    roc_curve,
)

from metrics.kinatic import regression_metrics


def evaluate(frame):
    target = frame["target"].to_numpy(float)
    prediction = frame["prediction"].to_numpy(float)
    binary_mask = target != 7
    binary_target = (target[binary_mask] > 7).astype(int)
    binary_score = prediction[binary_mask]
    binary_prediction = (binary_score > 7).astype(int)
    tn, fp, fn, tp = confusion_matrix(binary_target, binary_prediction, labels=[0, 1]).ravel()
    fpr, tpr, thresholds = roc_curve(binary_target, binary_score)

    target_class = np.select([target < 5, target <= 9], [0, 1], default=2)
    prediction_class = np.select([prediction < 5, prediction <= 9], [0, 1], default=2)
    return {
        "regression": regression_metrics(frame),
        "acidic_vs_alkaline": {
            "n": int(binary_mask.sum()),
            "excluded_ph_7": int((~binary_mask).sum()),
            "auroc": float(roc_auc_score(binary_target, binary_score)),
            "accuracy": float(accuracy_score(binary_target, binary_prediction)),
            "balanced_accuracy": float(balanced_accuracy_score(binary_target, binary_prediction)),
            "f1": float(f1_score(binary_target, binary_prediction)),
            "mcc": float(matthews_corrcoef(binary_target, binary_prediction)),
            "specificity": float(tn / (tn + fp)),
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        },
        "ternary": {
            "classes": ["pH<5", "5<=pH<=9", "pH>9"],
            "accuracy": float(accuracy_score(target_class, prediction_class)),
            "balanced_accuracy": float(balanced_accuracy_score(target_class, prediction_class)),
            "macro_f1": float(f1_score(target_class, prediction_class, average="macro")),
            "mcc": float(matthews_corrcoef(target_class, prediction_class)),
            "confusion_matrix": confusion_matrix(target_class, prediction_class, labels=[0, 1, 2]).tolist(),
        },
        "roc_curve": [
            {"fpr": float(x), "tpr": float(y), "threshold": float(z) if np.isfinite(z) else None}
            for x, y, z in zip(fpr, tpr, thresholds)
        ],
    }

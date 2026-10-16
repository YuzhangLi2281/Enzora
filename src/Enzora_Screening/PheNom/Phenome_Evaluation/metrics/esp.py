import pandas as pd

from metrics.mutation_effects import evaluate as evaluate_binary


def evaluate(frame, threshold=0.5, positive_prevalence=None):
    result = evaluate_binary(frame, threshold, positive_prevalence)
    if "sequence_identity" in frame:
        identity_bins = pd.cut(
            frame["sequence_identity"],
            bins=[-float("inf"), 20, 40, 60, 80, float("inf")],
            labels=["<20", "20-40", "40-60", "60-80", ">=80"],
            right=False,
        )
        result["by_sequence_identity"] = {
            str(interval): evaluate_binary(group, threshold, positive_prevalence)
            for interval, group in frame.groupby(identity_bins, observed=True)
            if group["target"].nunique() == 2
        }
    return result

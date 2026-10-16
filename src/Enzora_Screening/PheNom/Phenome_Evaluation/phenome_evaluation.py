import argparse
import re
from pathlib import Path

import pandas as pd
import yaml

from metrics import esp, kinatic, mutation_effects, ph


TASKS = {
    "kinetic": {"kcat", "km", "kcat_km"},
    "mutant_ranking": {"kcat", "km", "kcat_km"},
    "dms": {"dms"},
    "ph": {"ph"},
    "mutation_effect": {"mutation"},
    "esp": {"esp"},
}
TARGET_TASKS = {
    "kcat": "kcat",
    "Km": "km",
    "kcat/Km": "kcat_km",
    "pHopt": "ph",
    "binary_effect": "mutation",
    "binary_pair": "esp",
    "dms_score": "dms",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate saved predictions under the PHENOME benchmark protocol.")
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--test-set", type=Path, required=True)
    parser.add_argument("--input-format", choices=("canonical", "phenome"), default="canonical")
    parser.add_argument("--results-dir", type=Path, default=Path(__file__).parent / "Results")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--positive-prevalence", type=float)
    parser.add_argument("--exclude-task", choices=sorted(set(TARGET_TASKS.values())), action="append", default=[])
    return parser.parse_args()


def main():
    args = parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.model):
        raise ValueError("--model may contain only letters, numbers, '.', '_' and '-'")

    predictions = pd.read_csv(args.predictions, low_memory=False)
    excluded_tasks = sorted(set(args.exclude_task).intersection(TASKS[args.task]))
    if args.input_format == "phenome":
        predictions = predictions.rename(columns={
            "ID": "sample_id",
            "selected_prediction": "prediction",
            "selected_sigma": "sigma",
        })
        if args.task == "dms":
            predictions["task"] = "dms"
    required_predictions = {"sample_id", "task", "prediction"}
    if missing := required_predictions - set(predictions.columns):
        raise ValueError(f"prediction file is missing columns: {sorted(missing)}")
    predictions = predictions[predictions["task"].isin(TASKS[args.task])]
    predictions = predictions[~predictions["task"].isin(excluded_tasks)]
    if predictions["sample_id"].duplicated().any():
        raise ValueError("prediction file contains duplicate sample_id values")

    benchmark = pd.read_csv(args.test_set, low_memory=False).rename(columns={
        "record_id": "sample_id",
        "label_numeric": "target",
        "source_dataset": "source",
        "original__similarity": "sequence_identity",
    })
    benchmark["task"] = benchmark["target_type"].map(TARGET_TASKS)
    benchmark = benchmark[benchmark["task"].isin(TASKS[args.task])]
    benchmark = benchmark[~benchmark["task"].isin(excluded_tasks)]
    if benchmark["sample_id"].duplicated().any():
        raise ValueError("test set contains duplicate sample_id values")
    missing_predictions = set(benchmark["sample_id"]) - set(predictions["sample_id"])
    extra_predictions = set(predictions["sample_id"]) - set(benchmark["sample_id"])
    if missing_predictions or extra_predictions:
        raise ValueError(
            f"prediction/test-set ID mismatch: missing={len(missing_predictions)}, extra={len(extra_predictions)}"
        )

    prediction_columns = ["sample_id", "task", "prediction"]
    if "sigma" in predictions:
        prediction_columns.append("sigma")
    metadata_columns = ["sample_id", "task", "target", "source"]
    for column in ("sequence_identity", "mutation"):
        if column in benchmark:
            metadata_columns.append(column)
    frame = benchmark[metadata_columns].merge(
        predictions[prediction_columns], on=["sample_id", "task"], validate="one_to_one"
    )
    if args.task == "mutation_effect" and "mutation" in frame:
        frame["mutation_count"] = frame["mutation"].fillna("").str.count(r"[,;:+]") + frame["mutation"].notna()
    if args.task in {"kinetic", "mutant_ranking"} and {"original__UniProtID", "substrate_smiles"}.issubset(benchmark.columns):
        group_ids = benchmark.set_index("sample_id")["original__UniProtID"].astype(str) + "|" + benchmark.set_index("sample_id")["substrate_smiles"].astype(str)
        frame["group_id"] = frame["sample_id"].map(group_ids)

    if args.task == "kinetic":
        metrics = kinatic.evaluate(frame)
    elif args.task == "mutant_ranking":
        metrics = kinatic.mutant_ranking_metrics(frame)
    elif args.task == "dms":
        metrics = mutation_effects.evaluate_dms(frame)
    elif args.task == "ph":
        metrics = ph.evaluate(frame)
    elif args.task == "mutation_effect":
        metrics = mutation_effects.evaluate(frame, args.threshold, args.positive_prevalence)
    else:
        metrics = esp.evaluate(frame, args.threshold, args.positive_prevalence)

    destination = args.results_dir / args.task / args.model
    destination.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination / "predictions.csv", index=False)
    (destination / "metrics.yaml").write_text(
        yaml.safe_dump(metrics, sort_keys=True, allow_unicode=True)
    )
    protocol = {
        "model": args.model,
        "task": args.task,
        "prediction_source": str(args.predictions.resolve()),
        "test_set": str(args.test_set.resolve()),
        "rows": len(frame),
        "input_format": args.input_format,
        "decision_threshold": args.threshold if args.task in {"mutation_effect", "esp"} else None,
        "positive_class": {
            "mutation_effect": "mutant_outperforms_wild_type",
            "esp": "active_enzyme_substrate_pair",
        }.get(args.task),
        "positive_prevalence": args.positive_prevalence,
        "excluded_tasks": excluded_tasks,
    }
    (destination / "protocol.yaml").write_text(
        yaml.safe_dump(protocol, sort_keys=True, allow_unicode=True)
    )
    print(yaml.safe_dump({"destination": str(destination), "metrics": metrics}, sort_keys=False))


if __name__ == "__main__":
    main()

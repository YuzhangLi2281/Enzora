#!/usr/bin/env python3
"""Restore EnzGFM folds and build leakage-free PHENOM train/validation tables."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


RAW_DATASETS = {
    0: "kcat-data_0.4simi-10fold.csv",
    1: "Km-data_0.4simi-10fold.csv",
    2: "kcat-over-Km-data_0.4simi-10fold.csv",
}
KINETIC_TASKS = {0, 1, 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path, required=True)
    parser.add_argument("--catapro-datasets", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def reaction_key(frame: pd.DataFrame) -> pd.Series:
    return frame["Protein"].astype(str) + "\x1f" + frame["Ligand"].astype(str)


def aggregate_reactions(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["Protein", "Ligand", "task_index"]
    rows = []
    for _, group in frame.groupby(keys, sort=False, dropna=False):
        row = group.iloc[0].copy()
        values = group["label"].astype(float).to_numpy()
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        row["label"] = median
        row["measurement_mad"] = mad
        row["sample_weight"] = float(np.clip(1.0 / (1.0 + mad), 0.5, 1.0))
        row["source_class"] = group["source_class"].iat[0]
        row["repeat_count"] = len(group)
        rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    train = pd.read_csv(args.train_csv, low_memory=False)
    test = pd.read_csv(args.test_csv, low_memory=False)

    fold_maps = []
    canonical_rows = []
    protein_sequences = []
    for task, filename in RAW_DATASETS.items():
        raw = pd.read_csv(args.catapro_datasets / filename, low_memory=False)
        raw["Protein"] = "P_" + raw["Sequence"].map(lambda sequence: hashlib.sha1(sequence.encode()).hexdigest())
        protein_sequences.append(raw[["Protein", "Sequence"]])
        mapping = raw[["Protein", "Smiles", "fold"]].rename(columns={"Smiles": "Ligand"})
        mapping["task_index"] = task
        fold_maps.append(mapping.drop_duplicates(["Protein", "Ligand", "task_index"]))
        if task == 0:
            labels = np.log10(raw["kcat(s^-1)"].astype(float))
        elif task == 1:
            labels = np.log10(raw["Km(M)"].astype(float) * 1000.0)
        else:
            labels = np.log10(raw["kcat(s^-1)"].astype(float) / (raw["Km(M)"].astype(float) * 1000.0))
        records = pd.DataFrame({
            "ID": [f"EnzGFM_{task}_{index}" for index in range(len(raw))],
            "Protein": raw["Protein"],
            "WT_Protein": pd.NA,
            "Ligand": raw["Smiles"].astype(str),
            "task_index": task,
            "label": labels,
            "is_binary": False,
            "sample_weight": 1.0,
            "source": "EnzGFM",
            "task_type": "kinetic",
            "target_type": ("kcat", "Km", "kcat/Km")[task],
            "source_split": "fold_" + raw["fold"].astype(int).astype(str),
            "mutation": pd.NA,
            "cluster_id": raw["Protein"],
            "fold": raw["fold"].astype(int),
            "source_class": "EnzGFM",
        })
        canonical_rows.append(records)
    fold_map = pd.concat(fold_maps, ignore_index=True)

    kinetic = train[train["task_index"].isin(KINETIC_TASKS)].merge(
        fold_map, on=["Protein", "Ligand", "task_index"], how="left", validate="many_to_one"
    )
    original_enz = kinetic["source"].eq("EnzGFM")
    if kinetic.loc[original_enz, "fold"].isna().any():
        raise RuntimeError("Failed to restore every EnzGFM fold")
    enz = kinetic["fold"].notna()
    kinetic["source_class"] = np.where(enz, "EnzGFM", "Extra")

    canonical = pd.concat(canonical_rows, ignore_index=True)
    validation = canonical[canonical["fold"].eq(8)].copy()
    enz_train = canonical[canonical["fold"].between(0, 7)].copy()
    extra = kinetic[~enz].copy()
    extra["source_class"] = "Extra"
    held_out_pairs = set(reaction_key(validation)) | set(
        reaction_key(test[test["task_index"].isin(KINETIC_TASKS)])
    )
    extra = extra[~reaction_key(extra).isin(held_out_pairs)]

    kinetic_train = aggregate_reactions(pd.concat([enz_train, extra], ignore_index=True))
    validation = aggregate_reactions(validation)
    nonkinetic = train[~train["task_index"].isin(KINETIC_TASKS)].copy()
    nonkinetic["measurement_mad"] = 0.0
    nonkinetic["source_class"] = nonkinetic["source"]
    nonkinetic["repeat_count"] = 1
    full_train = pd.concat([kinetic_train, nonkinetic], ignore_index=True)

    scalers = {}
    for task, name in ((0, "kcat"), (1, "km"), (2, "kcat_km")):
        values = kinetic_train.loc[kinetic_train["task_index"].eq(task), "label"].astype(float)
        median = float(values.median())
        iqr = float(values.quantile(0.75) - values.quantile(0.25))
        scalers[name] = {"task_index": task, "median": median, "iqr": iqr}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    full_train.to_csv(args.output_dir / "train.csv", index=False)
    validation.to_csv(args.output_dir / "validation.csv", index=False)
    test.to_csv(args.output_dir / "test.csv", index=False)
    kinetic_train.to_csv(args.output_dir / "kinetic_train.csv", index=False)
    kinetic_train[kinetic_train["task_index"].isin([0, 1])].to_csv(
        args.output_dir / "kinetic_specialization.csv", index=False
    )
    for task, name in ((0, "kcat"), (1, "km")):
        kinetic_train[kinetic_train["task_index"].eq(task)].to_csv(
            args.output_dir / f"{name}_specialization.csv", index=False
        )
    (args.output_dir / "kinetic_scalers.json").write_text(json.dumps(scalers, indent=2) + "\n")
    pd.concat(protein_sequences, ignore_index=True).drop_duplicates("Protein").to_csv(
        args.output_dir / "protein_sequences.csv", index=False
    )

    report = {
        "train_rows": int(len(full_train)),
        "validation_rows": int(len(validation)),
        "test_rows": int(len(test)),
        "kinetic_train_by_task": kinetic_train["task_index"].value_counts().sort_index().to_dict(),
        "validation_by_task": validation["task_index"].value_counts().sort_index().to_dict(),
        "removed_extra_overlap_rows": int(len(kinetic[~enz]) - len(extra)),
        "duplicate_rows_collapsed": int(len(enz_train) + len(extra) - len(kinetic_train)),
        "scalers": scalers,
    }
    (args.output_dir / "split_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

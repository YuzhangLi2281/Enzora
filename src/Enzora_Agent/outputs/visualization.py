from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save_empty(path: Path, title: str, message: str) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_final_scores(selected: pd.DataFrame, path: Path) -> Path:
    if selected.empty or "final_score" not in selected.columns:
        return _save_empty(path, "Selected mutation final scores", "No selected candidates")
    data = selected.sort_values("final_score", ascending=True)
    fig, ax = plt.subplots(figsize=(9, max(4.5, 0.45 * len(data) + 2)))
    ax.barh(data["mutation"].astype(str), data["final_score"].astype(float))
    ax.set_xlabel("Final score")
    ax.set_ylabel("Mutation")
    ax.set_title("Selected mutation candidates")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_activity_stability(ranked: pd.DataFrame, selected: pd.DataFrame, path: Path) -> Path:
    needed = {"activity_log10", "ddg", "mutation"}
    if ranked.empty or not needed.issubset(ranked.columns):
        return _save_empty(path, "PHENOM log10 kcat vs UniStab ddG", "No plottable candidates")
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.scatter(
        ranked["activity_log10"].astype(float),
        ranked["ddg"].astype(float),
        alpha=0.55,
    )
    if not selected.empty and needed.issubset(selected.columns):
        ax.scatter(
            selected["activity_log10"].astype(float),
            selected["ddg"].astype(float),
            marker="x",
            s=70,
        )
        for _, row in selected.iterrows():
            ax.annotate(
                str(row["mutation"]),
                (float(row["activity_log10"]), float(row["ddg"])),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )
    ax.set_xlabel("PHENOM activity_log10")
    ax.set_ylabel("UniStab ddG")
    ax.set_title("PHENOM log10 kcat vs stability prediction")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_heatmap(selected: pd.DataFrame, path: Path) -> Path:
    columns = ["activity_log10", "ph_opt", "ddg", "final_score"]
    if selected.empty or not set(columns + ["mutation"]).issubset(selected.columns):
        return _save_empty(path, "Selected candidate property profile", "No selected candidates")

    data = selected[columns].astype(float).copy()
    for col in columns:
        arr = data[col].to_numpy(dtype=float)
        std = float(np.nanstd(arr))
        if np.isfinite(std) and std > 0:
            data[col] = (arr - float(np.nanmean(arr))) / std
        else:
            data[col] = 0.0

    fig, ax = plt.subplots(figsize=(8.5, max(4.5, len(selected) * 0.5 + 2)))
    im = ax.imshow(data.to_numpy(), aspect="auto")
    ax.set_xticks(range(len(columns)), labels=columns, rotation=20, ha="right")
    ax.set_yticks(range(len(selected)), labels=selected["mutation"].astype(str).tolist())
    ax.set_title("Selected candidate property profile (column-wise z-score)")
    fig.colorbar(im, ax=ax, label="Standardized value")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_mutation_sequence_map(selected: pd.DataFrame, sequence_length: int, path: Path) -> Path:
    needed = {"mutation", "final_score"}
    if sequence_length <= 0 or selected.empty or not needed.issubset(selected.columns):
        return _save_empty(path, "Mutation sequence map", "No selected candidates")

    from ..core.sequence import parse_mutation
    data = selected.copy()
    data['substitution'] = data['mutation'].str.split(':')
    data = data.explode('substitution')
    data['position'] = [parse_mutation(value)[1] for value in data['substitution']]
    data['final_score'] = pd.to_numeric(data['final_score'], errors='raise')
    fig, ax = plt.subplots(figsize=(11, max(4.2, 0.25 * len(data) + 3)))
    ax.hlines(0.0, 1, sequence_length, linewidth=3)

    ymax = max(1.0, float(data["final_score"].max()) * 1.25)
    for idx, (_, row) in enumerate(data.iterrows()):
        x = float(row["position"])
        y = float(row["final_score"])
        ax.vlines(x, 0.0, y, linewidth=1.4)
        ax.scatter([x], [y], s=55)
        offset = 5 if idx % 2 == 0 else -13
        ax.annotate(
            str(row["mutation"]),
            (x, y),
            xytext=(4, offset),
            textcoords="offset points",
            fontsize=8,
            rotation=30,
            ha="left",
            va="bottom" if offset > 0 else "top",
        )

    ax.set_xlim(0, sequence_length + 1)
    ax.set_ylim(-0.08 * ymax, ymax)
    ax.set_xlabel(f"Protein sequence position (1-{sequence_length})")
    ax.set_ylabel("Final score")
    ax.set_title("Selected mutation sequence map")
    ax.grid(axis="x", alpha=0.15)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_pareto_front(ranked: pd.DataFrame, selected: pd.DataFrame, path: Path) -> Path:
    needed = {
        "mutation",
        "activity_component",
        "ph_component",
        "stability_component",
        "pareto_optimal",
    }
    if ranked.empty or not needed.issubset(ranked.columns):
        return _save_empty(path, "Three-objective Pareto front", "No Pareto data")

    data = ranked.copy()
    for col in ["activity_component", "ph_component", "stability_component"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=["activity_component", "ph_component", "stability_component"])
    if data.empty:
        return _save_empty(path, "Three-objective Pareto front", "No valid Pareto data")

    sizes = 45 + 150 * data["ph_component"].clip(0, 1).to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.scatter(
        data["activity_component"].astype(float),
        data["stability_component"].astype(float),
        s=sizes,
        alpha=0.45,
        label="Candidate pool",
    )

    pareto_mask = data["pareto_optimal"].astype(str).str.lower().isin({"true", "1"})
    pareto = data[pareto_mask]
    if not pareto.empty:
        ax.scatter(
            pareto["activity_component"].astype(float),
            pareto["stability_component"].astype(float),
            s=95 + 150 * pareto["ph_component"].clip(0, 1).to_numpy(dtype=float),
            facecolors="none",
            linewidths=1.6,
            label="3-objective Pareto optimal",
        )

    if not selected.empty and {"mutation", "activity_component", "stability_component"}.issubset(selected.columns):
        ax.scatter(
            selected["activity_component"].astype(float),
            selected["stability_component"].astype(float),
            marker="x",
            s=80,
            label="Selected Top-K",
        )
        for _, row in selected.iterrows():
            ax.annotate(
                str(row["mutation"]),
                (float(row["activity_component"]), float(row["stability_component"])),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )

    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("Normalized activity component (higher is better)")
    ax.set_ylabel("Normalized stability component (higher is better)")
    ax.set_title("Three-objective Pareto view (marker size = pH component)")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_visualizations(
    *,
    ranked_csv: Path,
    selected_csv: Path,
    output_dir: Path,
    sequence_length: int,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ranked = pd.read_csv(ranked_csv)
    selected = pd.read_csv(selected_csv)
    paths = [
        _plot_final_scores(selected, output_dir / "01_final_score.svg"),
        _plot_activity_stability(ranked, selected, output_dir / "02_activity_vs_stability.svg"),
        _plot_heatmap(selected, output_dir / "03_property_heatmap.svg"),
        _plot_mutation_sequence_map(selected, sequence_length, output_dir / "04_mutation_sequence_map.svg"),
        _plot_pareto_front(ranked, selected, output_dir / "05_pareto_front.svg"),
    ]
    return paths


def write_sequence_umap(candidates, selected, path: Path, random_seed: int) -> Path:
    """UMAP of positional amino-acid identity, retaining actual coordinate geometry.

    The view may clip to a circle; the coordinates are never bent into a circle.
    Features use the entire library, with a transparent Hamming-distance basis.
    """
    import json
    from umap import UMAP
    from ..core.sequence import AMINO_ACIDS
    if len(candidates) < 4:
        raise ValueError('UMAP requires at least four library sequences')
    sequences = [c.mutant_sequence for c in candidates]
    if any(not s or len(s) != len(sequences[0]) for s in sequences):
        raise ValueError('UMAP requires nonempty equal-length protein sequences')
    features = np.asarray([[AMINO_ACIDS.index(aa) for aa in seq] for seq in sequences], dtype=np.uint8)
    coordinates = UMAP(n_components=2, metric='hamming', n_neighbors=min(15, len(candidates)-1),
                       init='random', random_state=random_seed).fit_transform(features)
    if not np.isfinite(coordinates).all():
        raise ValueError('UMAP produced nonfinite coordinates')
    ranks = {c.mutation: rank for rank, c in enumerate(selected, start=1)}
    data = {'method': 'UMAP', 'features': 'positional amino-acid identity', 'metric': 'hamming',
            'random_seed': random_seed, 'points': [
                {'mutation': c.mutation, 'x': float(x), 'y': float(y), 'rank': ranks.get(c.mutation),
                 'score': c.final_score, 'activity_log10': c.activity_log10}
                for c, (x, y) in zip(candidates, coordinates, strict=True)]}
    path.write_text(json.dumps(data, allow_nan=False), encoding='utf-8')
    return path

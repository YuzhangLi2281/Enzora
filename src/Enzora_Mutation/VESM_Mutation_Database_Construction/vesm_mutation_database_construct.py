#!/usr/bin/env python3
"""vesm-enzyme: Score substitutions with three PLMs and build an enzyme library.

The three models have different objectives and raw log-probability scales.  Each
model is therefore calibrated independently by mapping the empirical ranks of
its eligible non-WT LLRs to standard-normal quantiles.  The VESM ensemble score
for a substitution is the minimum of the three *aligned* scores at exactly the
same residue and mutant amino acid.  Higher ensemble scores are preferred when
constructing the library.

The checked-in configuration reproduces the released full-length 1M-variant
library. Current requests use explicit per-order quotas (default: 1000 singles);
position restrictions and impossible quotas are validated before loading weights.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import importlib
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterable

import numpy as np
import yaml
from Bio.PDB.MMCIF2Dict import MMCIF2Dict



import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt



AA20 = tuple("ACDEFGHIKLMNPQRSTVWY")
AA_TO_INDEX = {aa: i for i, aa in enumerate(AA20)}
def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    return data


def resolve_path(value: str, config_dir: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (config_dir / path).resolve()


def read_sequence(path: Path) -> str:
    lines = [line.strip() for line in path.read_text().splitlines()]
    sequence = "".join(line for line in lines if line and not line.startswith(">"))
    sequence = sequence.upper()
    invalid = sorted(set(sequence) - set(AA20))
    if not sequence or invalid:
        raise ValueError(f"Invalid protein sequence in {path}; unsupported residues: {invalid}")
    return sequence


def select_structure(structure_cfg: dict[str, Any], config_dir: Path) -> Path:
    explicit = structure_cfg.get("path")
    if explicit:
        path = resolve_path(explicit, config_dir)
        if not path.is_file():
            raise FileNotFoundError(f"Structure file not found: {path}")
        return path
    directory = resolve_path(structure_cfg["directory"], config_dir)
    pattern = structure_cfg.get("pattern", "*_model_*.cif")
    candidates = sorted(directory.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No structure matched {directory / pattern}")
    # AlphaFold Server ranks model_0 first. Requiring an explicit path remains
    # available for servers whose naming convention differs.
    return candidates[0]


def residue_plddt_from_cif(path: Path, chain_id: str, expected_sequence: str) -> np.ndarray:
    cif = MMCIF2Dict(str(path))
    chains = cif.get("_atom_site.auth_asym_id", cif.get("_atom_site.label_asym_id"))
    seq_ids = cif.get("_atom_site.auth_seq_id", cif.get("_atom_site.label_seq_id"))
    residue_names = cif.get("_atom_site.label_comp_id")
    b_factors = cif.get("_atom_site.B_iso_or_equiv")
    if not all((chains, seq_ids, residue_names, b_factors)):
        raise ValueError(f"mmCIF lacks chain/residue/B-factor fields: {path}")

    three_to_one = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
        "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
        "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    }
    by_residue: dict[int, list[float]] = {}
    residue_aa: dict[int, str] = {}
    for chain, seq_id, name, score in zip(chains, seq_ids, residue_names, b_factors):
        if chain != chain_id or seq_id in {".", "?"}:
            continue
        pos = int(float(seq_id))
        by_residue.setdefault(pos, []).append(float(score))
        residue_aa[pos] = three_to_one.get(name, "X")
    ordered = sorted(by_residue)
    observed = "".join(residue_aa[pos] for pos in ordered)
    if observed != expected_sequence:
        raise ValueError(
            f"Structure chain {chain_id} sequence does not exactly match input "
            f"({len(observed)} versus {len(expected_sequence)} residues)"
        )
    return np.asarray([float(np.mean(by_residue[pos])) for pos in ordered], dtype=np.float32)


def eligible_positions(
    plddt: np.ndarray, threshold: float, positions_cfg: dict[str, Any]
) -> np.ndarray:
    allowed = set(np.flatnonzero(plddt > threshold).tolist())
    include = positions_cfg["include"]
    exclude = positions_cfg.get("exclude") or []
    if include is not None:
        allowed &= {int(pos) - 1 for pos in include}
    allowed -= {int(pos) - 1 for pos in exclude}
    if not allowed:
        raise ValueError("No mutable positions remain after pLDDT/include/exclude filters")
    return np.asarray(sorted(allowed), dtype=np.int32)


@dataclass
class ScoreResult:
    logits: np.ndarray  # [sequence_length, 20], canonical AA order
    llr: np.ndarray  # mutant logP - WT logP, same shape


def canonicalize_logits(logits: torch.Tensor, token_ids: dict[str, int]) -> np.ndarray:
    columns = [token_ids[aa] for aa in AA20]
    return logits[..., columns].detach().float().cpu().numpy()


def llr_from_canonical_logits(logits: np.ndarray, sequence: str) -> np.ndarray:
    log_probs = logits - np.logaddexp.reduce(logits, axis=-1, keepdims=True)
    wt_indices = np.asarray([AA_TO_INDEX[aa] for aa in sequence])
    return log_probs - log_probs[np.arange(len(sequence)), wt_indices, None]


def score_enzgfm(sequence: str, cfg: dict[str, Any], device: torch.device) -> ScoreResult:
    import torch
    repo = Path(cfg["repository_path"]).expanduser().resolve()
    sys.path.insert(0, str(repo))
    try:
        enzgfm_class = importlib.import_module("models").EnzGFMForMaskedLM
        tokenizer_class = importlib.import_module("transformers").EsmTokenizer
    finally:
        sys.path.pop(0)
    tokenizer = tokenizer_class.from_pretrained(cfg["tokenizer_path"])
    model_config = enzgfm_class.config_class.from_pretrained(cfg["weights_path"])
    model_config.use_mamba_kernels = False
    model = enzgfm_class.from_pretrained(cfg["weights_path"], config=model_config).eval().to(device)
    tokens = tokenizer(sequence, return_tensors="pt").to(device)
    with torch.inference_mode():
        output = model(**tokens)
    # One BOS token precedes residues; EOS is discarded.
    residue_logits = output.logits[0, 1 : len(sequence) + 1]
    token_ids = {aa: tokenizer.convert_tokens_to_ids(aa) for aa in AA20}
    canonical = canonicalize_logits(residue_logits, token_ids)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return ScoreResult(canonical, llr_from_canonical_logits(canonical, sequence))


def score_vesm3(sequence: str, cfg: dict[str, Any], device: torch.device) -> ScoreResult:
    import torch
    from esm.models.esm3 import ESM3
    from esm.tokenization import get_esm3_model_tokenizers
    from esm.utils.constants.esm3 import data_root
    from esm.utils.constants.models import ESM3_OPEN_SMALL

    with torch.device(device):
        model = ESM3(
            d_model=1536,
            n_heads=24,
            v_heads=256,
            n_layers=48,
            structure_encoder_fn=lambda _: None,
            structure_decoder_fn=lambda _: None,
            function_decoder_fn=lambda _: None,
            tokenizers=get_esm3_model_tokenizers(ESM3_OPEN_SMALL),
        )
    base_state = torch.load(
        cfg["base_weights_path"], map_location=device
    )
    model.load_state_dict(base_state, strict=True)
    checkpoint = torch.load(cfg["weights_path"], map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise ValueError(f"Unexpected VESM3 checkpoint keys: {unexpected[:10]}")
    if missing and cfg.get("require_exact_checkpoint", False):
        raise ValueError(f"Missing VESM3 checkpoint keys: {missing[:10]}")
    model = model.to(torch.float).eval().to(device)
    tokenizer = model.tokenizers.sequence
    encoded = tokenizer.encode(sequence, add_special_tokens=True)
    input_ids = torch.as_tensor(encoded, dtype=torch.long, device=device).unsqueeze(0)
    with torch.inference_mode():
        output = model.forward(sequence_tokens=input_ids)
    residue_logits = output.sequence_logits[0, 1 : len(sequence) + 1]
    vocabulary = tokenizer.get_vocab()
    canonical = canonicalize_logits(residue_logits, vocabulary)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return ScoreResult(canonical, llr_from_canonical_logits(canonical, sequence))


def score_protein_dpo(
    sequence: str, structure_path: Path, chain_id: str, cfg: dict[str, Any], device: torch.device
) -> ScoreResult:
    import torch
    import esm
    from esm.inverse_folding.util import CoordBatchConverter, load_coords
    base_path = cfg.get("base_weights_path")
    if not base_path:
        raise ValueError("protein_dpo.base_weights_path is required")
    model, alphabet = esm.pretrained.load_model_and_alphabet_local(base_path)
    state = torch.load(cfg["weights_path"], map_location="cpu")
    state = state.get("model_state_dict", state) if isinstance(state, dict) else state
    model.load_state_dict(state, strict=True)
    model = model.eval().to(device)

    coords, structure_sequence = load_coords(str(structure_path), chain_id)
    if structure_sequence != sequence:
        raise ValueError("ProteinDPO structure sequence does not match the input sequence")
    converter = CoordBatchConverter(alphabet)
    batch = [(coords, None, sequence)]
    coords_t, confidence, _, tokens, padding_mask = converter(batch, device=device)
    prev_output_tokens = tokens[:, :-1]
    with torch.inference_mode():
        output = model.forward(coords_t, padding_mask, confidence, prev_output_tokens)
    # ESM-IF1 returns [batch, vocabulary, residues]. Teacher forcing makes
    # column i the distribution for residue i conditioned on backbone + WT prefix.
    raw = output[0] if isinstance(output, tuple) else output
    residue_logits = raw[0].transpose(0, 1)[: len(sequence)]
    token_ids = {aa: alphabet.get_idx(aa) for aa in AA20}
    canonical = canonicalize_logits(residue_logits, token_ids)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return ScoreResult(canonical, llr_from_canonical_logits(canonical, sequence))


def gaussian_rank_calibrate(llr: np.ndarray, candidate_mask: np.ndarray) -> np.ndarray:
    """Empirical quantile-to-normal calibration; monotonic and unbounded."""
    values = llr[candidate_mask]
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Calibration requires finite scores for every eligible substitution")
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1)
    # Average tied ranks so identical raw scores stay identical after calibration.
    sorted_values = values[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_values)) + 1]
    ends = np.r_[starts[1:], len(values)]
    for start, end in zip(starts, ends):
        ranks[order[start:end]] = 0.5 * ((start + 1) + end)
    probabilities = (ranks - 0.5) / len(values)
    normal = NormalDist()
    transformed = np.asarray([normal.inv_cdf(float(p)) for p in probabilities], dtype=np.float32)
    result = np.full_like(llr, np.nan, dtype=np.float32)
    result[candidate_mask] = transformed
    return result


def build_candidate_mask(sequence: str, eligible: np.ndarray) -> np.ndarray:
    mask = np.zeros((len(sequence), len(AA20)), dtype=bool)
    mask[eligible, :] = True
    for pos, wt in enumerate(sequence):
        mask[pos, AA_TO_INDEX[wt]] = False
    return mask


def mutation_table(
    sequence: str,
    plddt: np.ndarray,
    eligible: np.ndarray,
    results: dict[str, ScoreResult],
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], np.ndarray]:
    candidate_mask = build_candidate_mask(sequence, eligible)
    calibrated = {
        name: gaussian_rank_calibrate(result.llr, candidate_mask)
        for name, result in results.items()
    }
    ensemble = np.full(candidate_mask.shape, np.nan, dtype=np.float32)
    stacked = np.stack([scores[candidate_mask] for scores in calibrated.values()])
    ensemble[candidate_mask] = np.min(stacked, axis=0)
    rows: list[dict[str, Any]] = []
    for pos in eligible:
        wt = sequence[pos]
        for aa_index, mutant in enumerate(AA20):
            if mutant == wt:
                continue
            row: dict[str, Any] = {
                "mutation": f"{wt}{pos + 1}{mutant}",
                "position": int(pos + 1),
                "wt": wt,
                "mutant": mutant,
                "plddt": float(plddt[pos]),
                "ensemble_min_gaussian_llr": float(ensemble[pos, aa_index]),
            }
            for name in results:
                row[f"{name}_raw_llr"] = float(results[name].llr[pos, aa_index])
                row[f"{name}_gaussian_llr"] = float(calibrated[name][pos, aa_index])
            rows.append(row)
    rows.sort(key=lambda row: row["ensemble_min_gaussian_llr"], reverse=True)
    return rows, calibrated, ensemble


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def weighted_sampler(rows: list[dict[str, Any]], temperature: float, top_fraction: float):
    if not rows or not math.isfinite(temperature) or temperature <= 0 or not 0 < top_fraction <= 1:
        raise ValueError('Sampling requires nonempty scores, positive finite temperature and top_fraction in (0,1]')
    pool_size = min(len(rows), max(3, int(math.ceil(len(rows) * top_fraction))))
    pool = rows[:pool_size]
    scores = np.asarray([row["ensemble_min_gaussian_llr"] for row in pool])
    weights = np.exp((scores - scores.max()) / temperature)
    cumulative = np.cumsum(weights).tolist()

    def draw(rng: random.Random) -> dict[str, Any]:
        target = rng.random() * cumulative[-1]
        return pool[bisect.bisect_left(cumulative, target)]

    return draw, pool_size


def construct_library(
    sequence: str, rows: list[dict[str, Any]], library_cfg: dict[str, Any]
) -> list[dict[str, Any]]:
    counts = library_cfg['counts']
    if not isinstance(counts, dict) or any(type(k) is not int or type(v) is not int for k, v in counts.items()):
        raise ValueError('Mutation orders and quotas must be integers')
    if not counts or any(k < 1 or v < 1 for k, v in counts.items()):
        raise ValueError("library.counts requires positive mutation orders and quotas")
    if counts.get(1, 0) > len(rows):
        raise ValueError(f"Requested {counts[1]} distinct singles but only {len(rows)} are available")
    rng = random.Random(int(library_cfg["random_seed"]))
    draw, pool_size = weighted_sampler(
        rows, float(library_cfg["sampling_temperature"]), float(library_cfg["top_fraction"])
    )
    variants: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()

    def add(selected: Iterable[dict[str, Any]]) -> bool:
        chosen = sorted(selected, key=lambda row: row["position"])
        key = tuple(row["mutation"] for row in chosen)
        if key in seen or len({row["position"] for row in chosen}) != len(chosen):
            return False
        seq_chars = list(sequence)
        for row in chosen:
            seq_chars[row["position"] - 1] = row["mutant"]
        score = float(sum(row["ensemble_min_gaussian_llr"] for row in chosen))
        seen.add(key)
        variants.append({
            "mutations": ":".join(key),
            "n_mut": len(key),
            "ensemble_llr_sum": score,
            "sequence": "".join(seq_chars),
        })
        return True

    for row in rows[:counts.get(1, 0)]:
        add([row])
    max_attempt_factor = library_cfg['max_attempt_factor']
    if type(max_attempt_factor) is not int or max_attempt_factor <= 0:
        raise ValueError('max_attempt_factor must be a positive integer')
    for n_mut in sorted(counts):
        if n_mut == 1:
            continue
        target = counts[n_mut]
        accepted = 0
        attempts = 0
        while accepted < target and attempts < target * max_attempt_factor:
            attempts += 1
            selected = [draw(rng) for _ in range(n_mut)]
            accepted += int(add(selected))
        if accepted != target:
            raise RuntimeError(
                f"Could create only {accepted}/{target} unique {n_mut}-mutants from "
                f"the top-{pool_size} weighted candidate pool"
            )
    rng.shuffle(variants)
    return variants


def write_fasta(path: Path, variants: list[dict[str, Any]], line_width: int = 80) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for index, variant in enumerate(variants, 1):
            handle.write(
                f">VESM_{index:06d}|n_mut={variant['n_mut']}|mutations={variant['mutations']}"
                f"|ensemble_llr_sum={variant['ensemble_llr_sum']:.6f}\n"
            )
            seq = variant["sequence"]
            for start in range(0, len(seq), line_width):
                handle.write(seq[start : start + line_width] + "\n")


def plot_library(variants: list[dict[str, Any]], output_dir: Path) -> None:
    if plt is None:
        raise RuntimeError("Matplotlib is required to generate library plots")

    colors = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]
    levels = sorted({int(v["n_mut"]) for v in variants})
    counts = [sum(v["n_mut"] == n for v in variants) for n in levels]
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    bars = ax.bar(
        [str(n) for n in levels], counts, color=colors[:len(levels)],
        edgecolor="black", linewidth=0.5,
    )
    ax.bar_label(bars, fmt="{:.0f}", padding=3, fontsize=8)
    ax.set_xlabel("Number of substitutions per sequence")
    ax.set_ylabel("Number of sequences")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_dir / "mutation_count_distribution.svg", dpi=300, facecolor="white")
    plt.close(fig)

    values = [v["ensemble_llr_sum"] for v in variants]
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.hist(values, bins=60, color="#0072B2", edgecolor="white", linewidth=0.25)
    ax.axvline(float(np.median(values)), color="#D55E00", linestyle="--", linewidth=1.2,
               label=f"Median = {np.median(values):.2f}")
    ax.set_xlabel("Summed ensemble Gaussian LLR")
    ax.set_ylabel("Number of sequences")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_dir / "library_llr_distribution.svg", dpi=300, facecolor="white")
    plt.close(fig)


def validate_config(cfg: dict[str, Any], config_path: Path, require_weights: bool) -> dict[str, Any]:
    """Resolve inputs and enforce quota consistency before expensive model loads."""
    config_dir = config_path.parent
    for model_cfg in cfg["models"].values():
        for key in ("repository_path", "tokenizer_path", "base_weights_path", "weights_path"):
            value = model_cfg.get(key)
            if value:
                model_cfg[key] = str(resolve_path(str(value), config_dir))
    sequence_path = resolve_path(cfg["input"]["sequence_path"], config_dir)
    sequence = read_sequence(sequence_path)
    structure_path = select_structure(cfg["input"]["structure"], config_dir)
    chain_id = str(cfg["input"]["structure"].get("chain_id", "A"))
    plddt = residue_plddt_from_cif(structure_path, chain_id, sequence)
    mutation_cfg = cfg["mutation_space"]
    scope = mutation_cfg.get("scope", "plddt")
    # The released million-sequence library used all residues. Do not infer a
    # pLDDT restriction merely because confidence values are present in CIF.
    if scope == "full_sequence":
        positions = mutation_cfg['positions']
        include, exclude = positions['include'], positions['exclude']
        for values in (include, exclude):
            if values is not None and any(type(p) is not int or p < 1 or p > len(sequence) for p in values):
                raise ValueError("Mutation positions must be within the input sequence")
        allowed = set(range(1, len(sequence) + 1)) if include is None else set(include)
        eligible = np.asarray(sorted(allowed - set(exclude)), dtype=np.int32) - 1
    elif scope == "plddt":
        eligible = eligible_positions(
            plddt, float(mutation_cfg["plddt_threshold"]), mutation_cfg["positions"]
        )
    else:
        raise ValueError("mutation_space.scope must be 'full_sequence' or 'plddt'")
    counts = cfg['library']['counts']
    if not isinstance(counts, dict) or any(type(k) is not int or type(v) is not int for k, v in counts.items()):
        raise ValueError('Mutation orders and quotas must be integers')
    if not counts or any(k < 1 or v < 1 for k, v in counts.items()):
        raise ValueError("library.counts requires positive mutation orders and quotas")
    if not len(eligible):
        raise ValueError("No eligible mutation positions")
    if max(counts) > len(eligible):
        raise ValueError("Mutation order exceeds eligible positions")
    if counts.get(1, 0) > len(eligible) * 19:
        raise ValueError(f"Requested {counts[1]} singles but only {len(eligible) * 19} are possible")
    if require_weights:
        for name, model_cfg in cfg["models"].items():
            for key in (key for key in ("weights_path", "base_weights_path") if key in model_cfg):
                path = Path(model_cfg[key]).expanduser()
                if not path.exists():
                    raise FileNotFoundError(f"{name}.{key} does not exist: {path}")
    return {
        "sequence": sequence,
        "sequence_path": sequence_path,
        "structure_path": structure_path,
        "chain_id": chain_id,
        "plddt": plddt,
        "eligible": eligible,
        "mutation_scope": scope,
    }


def main() -> None:
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[3] / "config/mutation.yaml")
    parser.add_argument("--validate-only", action="store_true", help="Validate local inputs without loading model weights")
    args = parser.parse_args()
    config_path = args.config.expanduser().resolve()
    cfg = load_yaml(config_path)
    context = validate_config(cfg, config_path, require_weights=not args.validate_only)
    print(
        f"Sequence: {len(context['sequence'])} aa; full-length mutable positions: "
        f"{len(context['eligible'])}; structure: {context['structure_path']}"
    )
    if args.validate_only:
        return

    device_name = cfg["runtime"].get("device", "cuda")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        if cfg["runtime"].get("allow_cpu_fallback", False):
            device_name = "cpu"
        else:
            raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(device_name)
    sequence = context["sequence"]
    output_dir = resolve_path(cfg['output']['directory'], config_path.parent)
    output_dir.mkdir(parents=True, exist_ok=True)
    def progress(payload):
        path = output_dir / 'progress.json'
        temporary = path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps({'phase': 1, **payload}))
        temporary.replace(path)
    results = {}
    for name, score in (
        ('protein_dpo', lambda: score_protein_dpo(sequence, context['structure_path'], context['chain_id'], cfg['models']['protein_dpo'], device)),
        ('enzgfm', lambda: score_enzgfm(sequence, cfg['models']['enzgfm'], device)),
        ('vesm3', lambda: score_vesm3(sequence, cfg['models']['vesm3'], device)),
    ):
        progress({'status': 'scoring', 'model': name, 'completed_models': list(results), 'saved_sequences': 0})
        results[name] = score()
    for name, result in results.items():
        expected = (len(sequence), len(AA20))
        if result.logits.shape != expected or result.llr.shape != expected:
            raise ValueError(f"{name} returned {result.logits.shape}; expected {expected}")

    rows, calibrated, ensemble = mutation_table(
        sequence, context["plddt"], context["eligible"], results
    )
    output_dir = resolve_path(cfg["output"]["directory"], config_path.parent)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / cfg["output"]["single_mutation_scores"], rows)
    np.savez_compressed(
        output_dir / cfg["output"]["intermediate_arrays"],
        amino_acid_order=np.asarray(AA20),
        sequence=np.asarray(sequence),
        plddt=context["plddt"],
        eligible_positions_1based=context["eligible"] + 1,
        ensemble_min_gaussian_llr=ensemble,
        **{f"{name}_logits": result.logits for name, result in results.items()},
        **{f"{name}_raw_llr": result.llr for name, result in results.items()},
        **{f"{name}_gaussian_llr": value for name, value in calibrated.items()},
    )
    variants = construct_library(sequence, rows, cfg["library"])
    write_fasta(output_dir / cfg["output"]["fasta"], variants)
    manifest = output_dir / cfg['output']['library_manifest']
    # Attribution describes the limiting Gaussian-LLR model, not causal generation ownership.
    limiting_counts = {name: 0.0 for name in calibrated}
    with manifest.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(variants[0]))
        writer.writeheader()
        for count, variant in enumerate(variants, start=1):
            writer.writerow(variant)
            for mutation in variant['mutations'].split(':'):
                position, aa = int(mutation[1:-1]) - 1, AA20.index(mutation[-1])
                values = {name: float(scores[position, aa]) for name, scores in calibrated.items()}
                minimum = min(values.values())
                tied = [name for name, value in values.items() if value == minimum]
                for name in tied:
                    limiting_counts[name] += 1 / len(tied)
            if count % 100 == 0 or count == len(variants):
                handle.flush()
                progress({'status': 'saving', 'saved_sequences': count, 'target_sequences': len(variants), 'limiting_model_counts': dict(limiting_counts)})
    plot_library(variants, output_dir)
    metadata = {
        "config": str(config_path),
        "structure": str(context["structure_path"]),
        "sequence_length": len(sequence),
        "mutation_scope": context["mutation_scope"],
        "plddt_filter_applied": context["mutation_scope"] == "plddt",
        "plddt_threshold_strictly_greater_than": (
            float(cfg["mutation_space"]["plddt_threshold"])
            if context["mutation_scope"] == "plddt" else None
        ),
        "eligible_position_count": len(context["eligible"]),
        "eligible_positions_1based": (context["eligible"] + 1).tolist(),
        "amino_acid_order": "".join(AA20),
        "calibration": "per-model empirical quantile to standard Gaussian",
        "ensemble": "minimum calibrated score at identical position and mutant amino acid",
        "selection_direction": "higher ensemble score is preferred",
        "library_counts": {str(k): int(v) for k, v in cfg["library"]["counts"].items()},
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {len(variants):,} unique variants to {output_dir}")


if __name__ == "__main__":
    main()

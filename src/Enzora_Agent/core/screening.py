"""评分配平、候选筛选、突变描述及 Pareto 分析。"""
from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path

from ..config import validate_weights
from .types import Candidate


def _minmax(values: list[float]) -> list[float]:
    lo, hi = min(values), max(values)
    if math.isclose(lo, hi):
        return [0.5] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def rank_candidates(candidates: list[Candidate], selection: dict) -> list[Candidate]:
    if not candidates:
        return []
    for c in candidates:
        if c.activity_log10 is None or c.ph_opt is None or c.ddg is None:
            raise ValueError(f"候选 {c.mutation} 的性质预测不完整")
        if not all(math.isfinite(float(v)) for v in [c.activity_log10, c.ph_opt, c.ddg]):
            raise ValueError(f"候选 {c.mutation} 含 NaN/Inf")

    activity_norm = _minmax([float(c.activity_log10) for c in candidates])
    stability_raw = [float(c.ddg) for c in candidates]
    stability_norm = _minmax(stability_raw)
    if selection["stability_direction"] == "lower_is_better":
        stability_norm = [1.0 - x for x in stability_norm]

    reference_components = None
    if selection.get('score_policy') is not None:
        policy = selection['score_policy']
        if selection['target_ph'] != policy['target_ph'] or selection['stability_direction'] != 'lower_is_better':
            raise ValueError('WT-reference policy requires its frozen pH target and raw lower-is-better ddG')
        reference_components = reference_score_components(candidates, selection['reference'], policy)
    target_ph = selection.get("target_ph")
    ph_tol = selection.get("ph_tolerance")
    weights = balance_weights(selection["weights"], selection["confidence"])
    denom = float(weights["ph"]) + float(weights["activity"]) + float(weights["stability"])

    for i, c in enumerate(candidates):
        c.activity_component = activity_norm[i]
        c.stability_component = stability_norm[i]

        if target_ph is None:
            c.ph_component = 0.0
        else:
            distance = abs(float(c.ph_opt) - float(target_ph))
            scale = float(ph_tol) if ph_tol is not None else max(1.0, float(target_ph))
            c.ph_component = max(0.0, 1.0 - distance / scale)

        if reference_components is not None:
            component = reference_components[i]
            c.activity_component, c.ph_component, c.stability_component = component['kcat'], component['ph'], component['stability']
            c.score_normalization = selection['score_policy']['method']
        c.passed = True
        c.reasons = []
        if target_ph is not None and ph_tol is not None:
            if abs(float(c.ph_opt) - float(target_ph)) > float(ph_tol):
                c.passed = False
                c.reasons.append("pH 超出容差")
        min_activity = selection.get("min_activity_log10")
        if min_activity is not None and float(c.activity_log10) < float(min_activity):
            c.passed = False
            c.reasons.append("PHENOM log10(kcat) 低于阈值")
        op = selection.get("stability_operator", "none")
        threshold = selection.get("stability_threshold")
        if op == "le" and float(c.ddg) > float(threshold):
            c.passed = False
            c.reasons.append("UniStab ΔΔG 高于阈值")
        elif op == "ge" and float(c.ddg) < float(threshold):
            c.passed = False
            c.reasons.append("UniStab ΔΔG 低于阈值")

        c.final_score = (
            float(weights["ph"]) * float(c.ph_component)
            + float(weights["activity"]) * float(c.activity_component)
            + float(weights["stability"]) * float(c.stability_component)
        ) / denom

    return sorted(candidates, key=lambda c: (not c.passed, -(c.final_score or 0.0)))


def balance_weights(preferences: dict, confidence: dict) -> dict[str, float]:
    """Explicit user/model reliability factors, never inferred from model names.

    Confidence here is a configured reliability control, not a calibrated
    probability. No EpHod/UniStab uncertainty is invented.
    """
    validate_weights(preferences)
    validate_weights(confidence, confidence=True)
    products = {key: preferences[key] * confidence[key] for key in preferences}
    total = sum(products.values())
    if total <= 0:
        raise ValueError('Confidence-weighted preference sum must be positive')
    return {key: value / total for key, value in products.items()}


# Kyte-Doolittle hydropathy index.
_HYDROPHOBICITY = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

# Approximate residue volumes (Å^3). These are used only as a deterministic
# comparative descriptor between WT and mutant residues.
_VOLUME = {
    "A": 88.6, "R": 173.4, "N": 114.1, "D": 111.1, "C": 108.5,
    "Q": 143.8, "E": 138.4, "G": 60.1, "H": 153.2, "I": 166.7,
    "L": 166.7, "K": 168.6, "M": 162.9, "F": 189.9, "P": 112.7,
    "S": 89.0, "T": 116.1, "W": 227.8, "Y": 193.6, "V": 140.0,
}

# Simplified nominal side-chain charge class near neutral pH.
_CHARGE = {aa: 0.0 for aa in _HYDROPHOBICITY}
_CHARGE.update({"D": -1.0, "E": -1.0, "K": 1.0, "R": 1.0})

_RESIDUE_CLASS = {
    **{aa: "nonpolar" for aa in "AVLIMGPFW"},
    **{aa: "polar_uncharged" for aa in "STCNQY"},
    **{aa: "positive" for aa in "KRH"},
    **{aa: "negative" for aa in "DE"},
}

_BLOSUM_ORDER = "ARNDCQEGHILKMFPSTWYV"
_BLOSUM62_ROWS = [
    [4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0],
    [-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3],
    [-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3],
    [-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3],
    [0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1],
    [-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2],
    [-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2],
    [0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3],
    [-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3],
    [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3],
    [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1],
    [-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2],
    [-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1],
    [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1],
    [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2],
    [1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2],
    [0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0],
    [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3],
    [-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1],
    [0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4],
]
_BLOSUM62 = {
    (a, b): _BLOSUM62_ROWS[i][j]
    for i, a in enumerate(_BLOSUM_ORDER)
    for j, b in enumerate(_BLOSUM_ORDER)
}

_MUTATION_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")


def parse_point_mutation(mutation: str) -> tuple[str, int, str]:
    match = _MUTATION_RE.fullmatch(str(mutation).strip().upper())
    if not match:
        raise ValueError(f"不支持的点突变格式: {mutation!r}; 期望例如 A23V")
    wt, pos, mutant = match.groups()
    if wt not in _HYDROPHOBICITY or mutant not in _HYDROPHOBICITY:
        raise ValueError(f"点突变包含非标准氨基酸: {mutation!r}")
    return wt, int(pos), mutant


def annotate_mutation_features(candidate: Candidate) -> Candidate:
    wt, position, mutant = parse_point_mutation(candidate.mutation)
    if position < 1 or position > len(candidate.wt_sequence):
        raise ValueError(f"突变位置超出序列长度: {candidate.mutation}")
    actual_wt = candidate.wt_sequence[position - 1]
    if actual_wt != wt:
        raise ValueError(
            f"突变 {candidate.mutation} 的 WT 残基与输入序列不一致: sequence={actual_wt}"
        )

    candidate.position = position
    candidate.wt_aa = wt
    candidate.mutant_aa = mutant
    candidate.wt_residue_class = _RESIDUE_CLASS[wt]
    candidate.mutant_residue_class = _RESIDUE_CLASS[mutant]
    candidate.residue_class_change = (
        f"{candidate.wt_residue_class}->{candidate.mutant_residue_class}"
    )

    candidate.wt_hydrophobicity = _HYDROPHOBICITY[wt]
    candidate.mutant_hydrophobicity = _HYDROPHOBICITY[mutant]
    candidate.delta_hydrophobicity = (
        candidate.mutant_hydrophobicity - candidate.wt_hydrophobicity
    )

    candidate.wt_volume = _VOLUME[wt]
    candidate.mutant_volume = _VOLUME[mutant]
    candidate.delta_volume = candidate.mutant_volume - candidate.wt_volume

    candidate.wt_charge = _CHARGE[wt]
    candidate.mutant_charge = _CHARGE[mutant]
    candidate.delta_charge = candidate.mutant_charge - candidate.wt_charge

    candidate.blosum62 = int(_BLOSUM62[(wt, mutant)])
    candidate.conservative_substitution = candidate.blosum62 >= 0
    return candidate


def annotate_candidates(candidates: list[Candidate]) -> list[Candidate]:
    for candidate in candidates:
        candidate.substitutions = []
        if ':' in candidate.mutation:
            for mutation in candidate.mutation.split(':'):
                feature = annotate_mutation_features(Candidate(mutation=mutation, wt_sequence=candidate.wt_sequence))
                candidate.substitutions.append({key: getattr(feature, key) for key in (
                    'mutation', 'position', 'wt_aa', 'mutant_aa', 'delta_hydrophobicity',
                    'delta_volume', 'delta_charge', 'blosum62', 'conservative_substitution')})
        else:
            annotate_mutation_features(candidate)
    return candidates


def mark_pareto_optimal(candidates: list[Candidate]) -> list[Candidate]:
    """Mark the 3-objective Pareto front using normalized higher-is-better components.

    Hard-filter failures are excluded from the Pareto front when at least one
    candidate passes. If none pass, all candidates are considered so the plot
    remains informative for debugging.
    """
    for candidate in candidates:
        candidate.pareto_optimal = False

    eligible = [c for c in candidates if c.passed]
    if not eligible:
        eligible = list(candidates)

    def vector(c: Candidate) -> tuple[float, float, float]:
        values = (c.activity_component, c.ph_component, c.stability_component)
        if any(v is None for v in values):
            raise ValueError(f"候选 {c.mutation} 缺少 Pareto 所需评分分量")
        return tuple(float(v) for v in values)  # type: ignore[return-value]

    vectors = {id(c): vector(c) for c in eligible}
    for candidate in eligible:
        current = vectors[id(candidate)]
        dominated = False
        for other in eligible:
            if other is candidate:
                continue
            ov = vectors[id(other)]
            if all(b >= a for a, b in zip(current, ov)) and any(
                b > a for a, b in zip(current, ov)
            ):
                dominated = True
                break
        candidate.pareto_optimal = not dominated
    return candidates


def write_mutation_features_csv(path: Path, candidates: list[Candidate]) -> Path:
    fields = [
        "mutation",
        "substitutions",
        "position",
        "wt_aa",
        "mutant_aa",
        "wt_residue_class",
        "mutant_residue_class",
        "residue_class_change",
        "wt_hydrophobicity",
        "mutant_hydrophobicity",
        "delta_hydrophobicity",
        "wt_volume",
        "mutant_volume",
        "delta_volume",
        "wt_charge",
        "mutant_charge",
        "delta_charge",
        "blosum62",
        "conservative_substitution",
        "pareto_optimal",
        "final_score",
        "passed",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for candidate in candidates:
            data = candidate.to_dict()
            data["substitutions"] = json.dumps(candidate.substitutions, ensure_ascii=False)
            writer.writerow({name: data.get(name) for name in fields})
    return path


def summarize_scorer_evidence(rows: list[dict]) -> dict:
    """Numerical LLM context with source-labelled, uncalibrated uncertainty."""
    import statistics
    output = {}
    fields = [('activity','activity_log10','activity_sigma_log10','Phenom Gaussian head sigma_log10'),
              ('ph','ph_opt','ph_uncertainty','EpHod abs(RLATtr - SVR) / 2 disagreement proxy'),
              ('stability','ddg','stability_uncertainty','UniStab regression-head MC-dropout std, not calibration')]
    for name, field_name, uncertainty_field, source in fields:
        values, uncertainties = [], []
        for row in rows:
            for field, destination in ((field_name,values),(uncertainty_field,uncertainties)):
                value = row.get(field)
                if value is None or value == '':
                    continue
                number = float(value)
                if not math.isfinite(number) or (field == uncertainty_field and number < 0):
                    raise ValueError(f'Invalid scorer evidence: {field}')
                destination.append(number)
        output[name] = {'available_predictions':len(values), 'total_candidates':len(rows),
                        'uncertainty':{'available':False, 'reason':'No uncertainty output available for these predictions.'}}
        if values:
            output[name]['prediction_distribution'] = {'min':min(values),'median':statistics.median(values),'max':max(values)}
        if uncertainties:
            output[name]['uncertainty'] = {'available':True,'calibrated':False,'source':source,
                'count':len(uncertainties),'min':min(uncertainties),'median':statistics.median(uncertainties),'max':max(uncertainties)}
    output['interpretation'] = ('Prediction spread is not uncertainty. Confidence controls are policy preferences, '
        'not probabilities. Missing uncertainty is unknown, not evidence of low reliability.')
    return output


def reference_score_components(candidates, reference: dict, policy: dict) -> list[dict]:
    """Server-matched WT-reference components, with measured uncertainty proxies.

    The frozen policy uses server kcat/ph/stability keys. No pool percentile or
    generation score participates. Missing uncertainty is rejected, never filled.
    """
    keys = {'kcat', 'ph', 'stability'}
    if policy['method'] != 'input_reference_three_model_llm_v3':
        raise ValueError('Unsupported WT-reference scoring policy')
    for field in ('weights', 'scales', 'uncertainty_scales'):
        values = policy[field]
        if set(values) != keys or any(type(v) not in (int,float) or not math.isfinite(v) or v <= 0 for v in values.values()):
            raise ValueError(f'Invalid reference policy {field}')
    weights = policy['weights']
    if not math.isclose(sum(weights.values()), 1., abs_tol=1e-8) or weights['kcat'] < .4 or weights['kcat'] < max(weights['ph'],weights['stability']):
        raise ValueError('Reference policy must preserve normalized kcat primary weights')
    target = float(policy['target_ph'])
    wt_mu, wt_ph = float(reference['activity_log10']), float(reference['ph_opt'])
    if not all(math.isfinite(v) for v in (target,wt_mu,wt_ph)) or not 0 <= target <= 14 or not 0 <= wt_ph <= 14:
        raise ValueError('Invalid fixed WT reference/target')
    for name,bounds in policy.get('allowed_scale_bounds',{}).items():
        if name not in keys or len(bounds)!=2 or not bounds[0] <= policy['scales'][name] <= bounds[1]:
            raise ValueError('Reference scale is outside the stored policy bounds')
    output = []
    for c in candidates:
        gains = {'kcat': float(c.activity_log10)-wt_mu,
                 'ph': abs(wt_ph-target)-abs(float(c.ph_opt)-target),
                 'stability': -float(c.ddg)}
        uncertainties = {'kcat': c.activity_sigma_log10, 'ph': c.ph_uncertainty,
                         'stability': c.stability_uncertainty}
        components = {}
        for name in keys:
            u = uncertainties[name]
            if type(u) not in (int,float) or not math.isfinite(u) or u < 0 or not math.isfinite(gains[name]):
                raise ValueError(f'Missing/invalid {name} prediction uncertainty for {c.mutation}')
            z = gains[name] / policy['scales'][name]
            raw = 1/(1+math.exp(-z)) if z >= 0 else math.exp(z)/(1+math.exp(z))
            reliability = 1/(1+u/policy['uncertainty_scales'][name])
            components[name] = .5 + reliability*(raw-.5)
        output.append(components)
    return output

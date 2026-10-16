/** Client-side mirror of Enzora_Agent.scoring.rank_candidates
 *  (same formula as server/app/rerank.py) for instant weight re-ranking.
 *  Re-ranking only recomputes the weighted final_score from the already
 *  normalized components — it never re-runs models. */

import type { Candidate } from '../../shared/api/contracts';

function minmax(values: number[]): number[] {
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  if (lo === hi || Math.abs(hi - lo) <= 1e-9 * Math.max(Math.abs(lo), Math.abs(hi))) return values.map(() => 0.5);
  return values.map((v) => (v - lo) / (hi - lo));
}

export function rerankCandidates(
  candidates: Candidate[],
  weights: { ph: number; activity: number; stability: number },
  selection: {
    target_ph?: number | null;
    ph_tolerance?: number | null;
    stability_direction?: 'lower_is_better' | 'higher_is_better';
    confidence?: { ph: number; activity: number; stability: number };
  } = {},
): Candidate[] {
  if (!candidates.length) return [];
  for (const candidate of candidates) {
    if ([candidate.activity_log10, candidate.ph_opt, candidate.ddg].some((v) => typeof v !== 'number' || !Number.isFinite(v))) throw new Error('Complete finite scorer predictions are required.');
  }
  if (Object.values(weights).some((v) => !Number.isFinite(v) || v < 0)) throw new Error('Invalid score weights.');
  if (selection.confidence && Object.values(selection.confidence).some((v) => !Number.isFinite(v) || v < 0 || v > 1)) throw new Error('Invalid reliability controls.');
  const effective = selection.confidence ? { ph: weights.ph * selection.confidence.ph, activity: weights.activity * selection.confidence.activity, stability: weights.stability * selection.confidence.stability } : weights;
  const act = candidates.map((c) => Number(c.activity_log10));
  const ddg = candidates.map((c) => Number(c.ddg));
  const actN = minmax(act);
  let stabN = minmax(ddg);
  if (selection.stability_direction !== 'higher_is_better') stabN = stabN.map((x) => 1 - x);

  const wp = Number(effective.ph);
  const wa = Number(effective.activity);
  const ws = Number(effective.stability);
  const denom = wp + wa + ws;
  if (denom <= 0) throw new Error('At least one effective weight must be positive.');

  const target = selection.target_ph != null ? Number(selection.target_ph) : null;
  const tol = selection.ph_tolerance != null ? Number(selection.ph_tolerance) : null;
  if (tol !== null && tol <= 0) throw new Error('pH tolerance must be positive.');
  if (wp > 0 && target === null) throw new Error('A pH weight requires a target.');
  const scale = tol != null && tol > 0 ? tol : Math.max(1, target ?? 1);

  return candidates
    .map((c, i) => {
      const referencePolicy = c.score_normalization === 'input_reference_three_model_llm_v3';
      if (referencePolicy) {
        const components = [c.ph_component, c.activity_component, c.stability_component];
        if (components.some(v => typeof v !== 'number' || !Number.isFinite(v))) throw new Error('Frozen reference components are required.');
        return { ...c, final_score: (wp * Number(c.ph_component) + wa * Number(c.activity_component) + ws * Number(c.stability_component)) / denom };
      }
      let phC = 0;
      if (target !== null) {
        const dist = Math.abs(Number(c.ph_opt) - target);
        phC = Math.max(0, 1 - dist / scale);
      }
      const final = (wp * phC + wa * actN[i] + ws * stabN[i]) / denom;
      return {
        ...c,
        activity_component: actN[i],
        ph_component: phC,
        stability_component: stabN[i],
        final_score: final,
      };
    })
    .sort((a, b) => {
      const pa = a.passed === false ? 1 : 0;
      const pb = b.passed === false ? 1 : 0;
      if (pa !== pb) return pa - pb;
      return (Number(b.final_score ?? 0)) - (Number(a.final_score ?? 0));
    });
}

export function normalizedWeights(weights: { ph: number; activity: number; stability: number }) {
  const total = Number(weights.ph ?? 0) + Number(weights.activity ?? 0) + Number(weights.stability ?? 0);
  if (total <= 0 || Object.values(weights).some((v) => !Number.isFinite(v) || v < 0)) throw new Error('Invalid weights.');
  return {
    ph: Number(weights.ph) / total,
    activity: Number(weights.activity) / total,
    stability: Number(weights.stability) / total,
  };
}

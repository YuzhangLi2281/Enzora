/** Client-side mirror of Enzora_Agent.scoring.rank_candidates
 *  (same formula as server/app/rerank.py) for instant weight re-ranking.
 *  Re-ranking only recomputes the weighted final_score from the already
 *  normalized components — it never re-runs models. */

import type { Candidate, RunRequest } from '../../shared/api/contracts';

/** Exact decomposition of the frozen-reference score, with uncertainty kept
 * separate from objective utility. No model prediction is changed here. */
export function candidateScoreEvidence(candidate: Candidate, selection: RunRequest['selection']) {
  const policy = selection.score_policy;
  if (policy?.method !== 'input_reference_three_model_llm_v3') return null;
  const reference = selection.reference;
  if (!reference || !selection.confidence) throw new Error('Missing scoring reference or applied policy.');
  if ([candidate.activity_log10,candidate.ph_opt,candidate.ddg,reference.activity_log10,reference.ph_opt].some(v=>typeof v!=='number'||!Number.isFinite(v)))
    throw new Error('Complete finite model predictions and WT reference are required.');
  if (Object.values(selection.weights).some(v=>!Number.isFinite(v)||v<0||v>1) || Math.abs(Object.values(selection.weights).reduce((a,b)=>a+b,0)-1)>1e-8 ||
      Object.values(selection.confidence).some(v=>!Number.isFinite(v)||v<0||v>1)) throw new Error('Invalid applied scoring policy.');
  const total = Object.entries(selection.weights).reduce((n,[key,value]) => n + value*selection.confidence![key as keyof typeof selection.weights], 0);
  if (!Number.isFinite(total) || total <= 0) throw new Error('Invalid applied scoring policy.');
  const target = policy.target_ph;
  const models = ([['activity','kcat','activity_sigma_log10'], ['ph','ph','ph_uncertainty'], ['stability','stability','stability_uncertainty']] as const).map(([key,scaleKey,uncertaintyKey]) => {
    const uncertainty = candidate[uncertaintyKey];
    const scale = policy.scales[scaleKey], uncertaintyScale = policy.uncertainty_scales[scaleKey];
    const gain = key === 'activity' ? Number(candidate.activity_log10)-reference.activity_log10
      : key === 'ph' ? (target === null ? 0 : Math.abs(reference.ph_opt-target)-Math.abs(Number(candidate.ph_opt)-target))
      : -Number(candidate.ddg);
    if (typeof uncertainty !== 'number' || !Number.isFinite(uncertainty) || uncertainty < 0 ||
        !Number.isFinite(scale) || scale <= 0 || !Number.isFinite(uncertaintyScale) || uncertaintyScale <= 0 || !Number.isFinite(gain))
      throw new Error(`Missing or invalid ${key} uncertainty/reference.`);
    const reliability = 1/(1+uncertainty/uncertaintyScale);
    const z = gain/scale;
    const utility = z >= 0 ? 1/(1+Math.exp(-z)) : Math.exp(z)/(1+Math.exp(z));
    const influence = selection.weights[key]*selection.confidence![key]/total*reliability;
    return {key, uncertainty, uncertaintyScale, reliability, utility, influence, contribution: influence*utility};
  });
  const neutral = .5*models.reduce((n,m)=>n+selection.weights[m.key]*selection.confidence![m.key]/total*(1-m.reliability),0);
  const score = neutral+models.reduce((n,m)=>n+m.contribution,0);
  if (!Number.isFinite(score) || score < 0 || score > 1) throw new Error('Invalid reconstructed candidate score.');
  return {models,neutral,score};
}

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
  if (Object.values(weights).some((v) => !Number.isFinite(v) || v < 0 || v > 1) || Math.abs(Object.values(weights).reduce((a,b)=>a+b,0)-1)>1e-8) throw new Error('Invalid score weights.');
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
        if (components.some(v => typeof v !== 'number' || !Number.isFinite(v) || v < 0 || v > 1)) throw new Error('Frozen reference components are required.');
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
  if (Math.abs(total-1)>1e-8 || Object.values(weights).some((v) => !Number.isFinite(v) || v < 0 || v > 1)) throw new Error('Invalid weights.');
  return {
    ph: Number(weights.ph) / total,
    activity: Number(weights.activity) / total,
    stability: Number(weights.stability) / total,
  };
}

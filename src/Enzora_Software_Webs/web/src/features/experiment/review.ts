import type { ClassifyResult, ParseResult, RunRequest } from '../../shared/api/contracts';

export const defaultCandidate = (): RunRequest['candidate'] => ({ top_k: 1000, allowed_positions: null, excluded_positions: [] });
export const defaultSelection = (): RunRequest['selection'] => ({
  target_ph: null, ph_tolerance: null, min_activity_log10: null,
  stability_direction: 'lower_is_better', stability_operator: 'none', stability_threshold: null,
  confidence: { ph: 1, activity: 1, stability: 1 },
  weights: { ph: 0, activity: 0.6, stability: 0.4 }, top_k: 5,
});

/** Keep every reviewed constraint; do not reconstruct a subset on submission. */
export function resolvedReview(parsed: ParseResult, candidate: RunRequest['candidate'], selection: RunRequest['selection']) {
  return {
    candidate: { ...candidate, ...parsed.resolved_candidate } as RunRequest['candidate'],
    selection: { ...selection, ...parsed.resolved_selection, weights: { ...selection.weights, ...(parsed.resolved_selection.weights as Partial<RunRequest['selection']['weights']> | undefined) } } as RunRequest['selection'],
  };
}

export function selectionError(selection: RunRequest['selection']): string | null {
  if (!Number.isInteger(selection.top_k) || selection.top_k < 1 || selection.top_k > 50) return 'top_k';
  if (selection.target_ph !== null && (!Number.isFinite(selection.target_ph) || selection.target_ph < 0 || selection.target_ph > 14)) return 'target_ph';
  if (selection.ph_tolerance !== null && (!Number.isFinite(selection.ph_tolerance) || selection.ph_tolerance <= 0)) return 'ph_tolerance';
  if (selection.min_activity_log10 !== null && !Number.isFinite(selection.min_activity_log10)) return 'min_activity_log10';
  if (selection.stability_threshold !== null && !Number.isFinite(selection.stability_threshold)) return 'stability_threshold';
  if (selection.stability_operator !== 'none' && selection.stability_threshold === null) return 'stability_required';
  if (selection.confidence) {
    if (Object.values(selection.confidence).some((v) => !Number.isFinite(v) || v < 0 || v > 1)) return 'confidence';
    if (selection.weights.ph * selection.confidence.ph + selection.weights.activity * selection.confidence.activity + selection.weights.stability * selection.confidence.stability <= 0) return 'confidence';
  }
  const weights = Object.values(selection.weights);
  if (weights.some((w) => !Number.isFinite(w) || w < 0) || weights.reduce((a, b) => a + b, 0) <= 0) return 'weights';
  if (selection.weights.ph > 0 && selection.target_ph === null) return 'ph_required';
  return null;
}

export function candidateError(candidate: RunRequest['candidate'], selection: RunRequest['selection'], sequenceLength?: number): string | null {
  if (!Number.isInteger(candidate.top_k) || candidate.top_k < 1) return 'candidate_pool';
  if (selection.top_k > candidate.top_k) return 'pool_too_small';
  const allowed = candidate.allowed_positions || [];
  const excluded = candidate.excluded_positions || [];
  if ([...allowed, ...excluded].some((position) => !Number.isInteger(position) || position < 1 || (sequenceLength !== undefined && position > sequenceLength))) return 'positions';
  if (allowed.some((position) => excluded.includes(position))) return 'position_conflict';
  return null;
}

/** This parses an explicit numeric parameter field, never a conversational input. */
export function positionList(value: string): number[] {
  return [...new Set(value.split(/[\s,;，；]+/).filter(Boolean).map(Number))];
}

export type ReviewContext = { sequence_name: string; sequence_type: 'protein'; sequence_length: number };
type ReviewParser = (goal: string, candidate: RunRequest['candidate'], selection: RunRequest['selection'], sessionId: string | null, context: ReviewContext) => Promise<ParseResult>;

/** Resolve one prepared sequence with its goal, before displaying an editable draft. */
export async function automaticReview({ name, sequence, goal, candidate, selection, sessionId, aiReady, lang, parse }: {
  name: string; sequence: string; goal: string; candidate: RunRequest['candidate']; selection: RunRequest['selection'];
  sessionId: string | null; aiReady: boolean; lang: RunRequest['report']['language']; parse: ReviewParser;
}) {
  if (!aiReady || sequenceError(sequence)) return null;
  const recommendation = lang === 'zh'
    ? '请根据当前蛋白序列信息，推荐合理的初始酶设计参数和 pH、活性、稳定性权重。对未知的目标给出明确标注的建议，供用户修改；不要把建议描述成用户指定条件。'
    : 'Recommend initial enzyme design parameters and pH, activity and stability weights for the current protein. Clearly label recommendations for unspecified goals so the user can edit them; do not present suggestions as user requirements.';
  const parsed = await parse(goal.trim() || recommendation, candidate, selection, sessionId, {
    sequence_name: name, sequence_type: 'protein', sequence_length: sequence.length,
  });
  return { ...resolvedReview(parsed, candidate, selection), parsed };
}

export function sequenceError(sequence: string): 'empty' | 'length' | 'alphabet' | null {
  if (!sequence) return 'empty';
  if (sequence.length > 1000) return 'length';
  const aa = new Set('ACDEFGHIKLMNPQRSTVWY');
  if ([...sequence].some((symbol) => !aa.has(symbol))) return 'alphabet';
  return null;
}

export function reviewedRequest({ name, sequence, goal, candidate, selection, aiReady, lang, forceCpu = false, ligandSmiles = '', mutationOrder = 1 }: {
  name: string; sequence: string; goal: string; candidate: RunRequest['candidate']; selection: RunRequest['selection'];
  aiReady: boolean; lang: RunRequest['report']['language']; forceCpu?: boolean; ligandSmiles?: string; mutationOrder?: number;
}): RunRequest {
  const total = Object.values(selection.weights).reduce((a, b) => a + b, 0);
  return {
    mutation: { counts: { [mutationOrder]: candidate.top_k } },
    activity: { task: 'kcat', ligand_smiles: ligandSmiles },
    sequence_name: name || 'enzyme', sequence, user_request: goal.trim() || null, candidate,
    selection: { ...selection, weights: { ph: selection.weights.ph / total, activity: selection.weights.activity / total, stability: selection.weights.stability / total } },
    llm: { enabled: aiReady, parameter_parsing: false, result_analysis: aiReady, analysis_fail_open: true },
    visualization: { enabled: true, umap: true }, report: { language: lang }, runtime: { force_cpu: forceCpu, phase_control: true },
  };
}

/** Keep a source sequence when AI recognizes a conversational goal update. */
export function conversationUpdate(result: ClassifyResult, hasSequence: boolean, currentGoal: string): { preserveDraft: boolean; goal: string | null } {
  if (result.action === 'update_goal' && hasSequence) {
    const addition = result.request_text || '';
    return { preserveDraft: true, goal: addition ? [currentGoal, addition].filter(Boolean).join('\n') : currentGoal };
  }
  if (result.action === 'clarify' && !result.records?.length) return { preserveDraft: true, goal: null };
  return { preserveDraft: false, goal: null };
}

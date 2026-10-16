import { useEffect, useRef, useState } from 'react';
import type { Lang, ParseResult, RunRequest } from '../../shared/api/contracts';
import { candidateError, positionList, selectionError } from './review';

export function ReviewParameters({ lang, selection, candidate, onChange, onCandidateChange, parseResult, sequenceLength, ligandSmiles, onLigandChange, mutationOrder = 1, onMutationOrderChange, disabled = false }: {
  lang: Lang; selection: RunRequest['selection']; candidate: RunRequest['candidate'];
  onChange: (selection: RunRequest['selection']) => void;
  onCandidateChange?: (candidate: RunRequest['candidate']) => void;
  ligandSmiles?: string; onLigandChange?: (value: string) => void;
  mutationOrder?: number; onMutationOrderChange?: (value: number) => void;
  parseResult?: ParseResult | null; sequenceLength?: number; disabled?: boolean;
}) {
  const zh = lang === 'zh';
  const [editedFields, setEditedFields] = useState<Set<string>>(new Set());
  const [allowedText, setAllowedText] = useState(candidate.allowed_positions?.join(', ') || '');
  const [excludedText, setExcludedText] = useState(candidate.excluded_positions.join(', '));
  const emittedCandidate = useRef(candidate);
  useEffect(() => { setEditedFields(new Set()); }, [parseResult]);
  useEffect(() => {
    if (emittedCandidate.current !== candidate) {
      setAllowedText(candidate.allowed_positions?.join(', ') || '');
      setExcludedText(candidate.excluded_positions.join(', '));
    }
    emittedCandidate.current = candidate;
  }, [candidate]);
  const mark = (field: string) => setEditedFields((fields) => new Set(fields).add(field));
  const set = (key: keyof RunRequest['selection'], value: unknown) => { mark(`selection.${key}`); onChange({ ...selection, [key]: value }); };
  const setCandidate = (key: keyof RunRequest['candidate'], value: unknown) => {
    if (!onCandidateChange) return;
    mark(`candidate.${key}`);
    const next = { ...candidate, [key]: value }; emittedCandidate.current = next; onCandidateChange(next);
  };
  const numeric = (value: string) => value === '' ? null : Number(value);
  const source = (field: string) => {
    if (editedFields.has(field) || (field.startsWith('selection.weights.') && editedFields.has('selection.weights'))) return <span className="parameter-source manual">{zh ? '已手动修改' : 'Edited by you'}</span>;
    if (parseResult?.explicit_fields?.includes(field) || (field.startsWith('selection.weights.') && parseResult?.explicit_fields?.includes('selection.weights'))) return <span className="parameter-source explicit">{zh ? '用户要求' : 'Your requirement'}</span>;
    if (parseResult?.suggested_fields?.includes(field) || (field.startsWith('selection.weights.') && parseResult?.suggested_fields?.includes('selection.weights'))) return <span className="parameter-source suggested">{zh ? 'AI 建议' : 'AI suggestion'}</span>;
    return null;
  };
  const rationale = (field: string) => !editedFields.has(field) && parseResult?.rationale?.[field] ? <small className="dim parameter-rationale">{parseResult.rationale[field]}</small> : null;
  const error = selectionError(selection) || candidateError(candidate, selection, sequenceLength);
  const errors: Record<string, string> = zh ? {
    top_k: '保留候选数须为 1–50 的整数。', target_ph: '目标 pH 须在 0–14 之间。', ph_tolerance: 'pH 容差须大于 0。', weights: '每项权重须在 0–1 之间，三项之和须为 1。', ph_required: 'pH 权重大于 0 时，请填写目标 pH。',
    min_activity_log10: '最低活性评分须为有效数字。', stability_threshold: '稳定性阈值须为有效数字。', stability_required: '使用稳定性筛选时，请填写阈值。',
    candidate_pool: '候选池须为 正整数 的整数。', pool_too_small: '候选池数量须不少于最终保留候选数。', positions: `位点须为 1–${sequenceLength || '序列长度'} 的整数，以逗号分隔。`, position_conflict: '同一位点不能同时被允许和排除。',
  } : {
    top_k: 'Select 1–50 candidates.', target_ph: 'Target pH must be 0–14.', ph_tolerance: 'pH tolerance must be positive.', weights: 'Each weight must be between 0 and 1; the three must sum to 1.', ph_required: 'Enter a target pH when the pH weight is positive.',
    min_activity_log10: 'Minimum activity score must be a valid number.', stability_threshold: 'Stability threshold must be a valid number.', stability_required: 'Enter a threshold when using a stability filter.',
    candidate_pool: 'Candidate pool must be a positive integer.', pool_too_small: 'The candidate pool must cover the final candidate count.', positions: `Positions must be integers from 1 to ${sequenceLength || 'the sequence length'}, separated by commas.`, position_conflict: 'A position cannot be both allowed and excluded.',
  };
  return <fieldset className="ai-parameter-review" disabled={disabled} style={{ border: 0, padding: 0, margin: 0, minWidth: 0 }}>
    <div className="form-grid" style={{ marginTop: 8 }}>
      {onLigandChange && <label className="field">{zh ? '底物 SMILES（PHENOM 必填）' : 'Substrate SMILES (required by PHENOM)'}<input value={ligandSmiles} onChange={(e) => onLigandChange(e.target.value)} placeholder="SMILES" /></label>}
      {onMutationOrderChange && <label className="field">{zh ? '每条序列的突变位点数' : 'Mutations per sequence'}<input type="number" min={1} max={sequenceLength} value={mutationOrder} onChange={(e) => onMutationOrderChange(Number(e.target.value))} /></label>}
    </div>
    <div className="form-grid" style={{ marginTop: 8 }}>
      <label className="field">{zh ? '目标 pH' : 'Target pH'} {source('selection.target_ph')}<input type="number" min={0} max={14} step="any" value={selection.target_ph ?? ''} onChange={(e) => set('target_ph', numeric(e.target.value))} placeholder={zh ? '不限定' : 'Unspecified'} />{rationale('selection.target_ph')}</label>
      <label className="field">{zh ? '保留候选数' : 'Candidates to select'} {source('selection.top_k')}<input type="number" min={1} max={50} value={selection.top_k} onChange={(e) => set('top_k', Number(e.target.value))} />{rationale('selection.top_k')}</label>
      <label className="field">{zh ? 'pH 容差' : 'pH tolerance'} {source('selection.ph_tolerance')}<input type="number" min={0.01} step="any" value={selection.ph_tolerance ?? ''} onChange={(e) => set('ph_tolerance', numeric(e.target.value))} placeholder={zh ? '不限定' : 'Unspecified'} />{rationale('selection.ph_tolerance')}</label>
    </div>
    <div style={{ marginTop: 8 }} className="small dim">{zh ? '相对权重（可修改，提交时归一化）' : 'Relative weights (editable, normalized on submission)'}</div>
    {selection.confidence && <div className="form-grid">{(['ph', 'activity', 'stability'] as const).map((key) => <label key={key} className="field">{key} {zh ? '可信度控制（非校准概率）' : 'Reliability control (not a probability)'}<input type="range" min={0} max={1} step={0.01} value={selection.confidence![key]} onChange={(e) => onChange({ ...selection, confidence: { ...selection.confidence!, [key]: Number(e.target.value) } })} /><span>{Math.round(selection.confidence![key] * 100)}%</span></label>)}</div>}
    {rationale('selection.weights')}
    <div className="form-grid" style={{ marginTop: 4 }}>
      {(['ph', 'activity', 'stability'] as const).map((key) => <label key={key} className="field">{zh ? ({ ph: 'pH', activity: '活性', stability: '稳定性' }[key]) : key} {source(`selection.weights.${key}`)}<input aria-label={`Weight ${key}`} type="number" min={0} max={1} step="0.1" value={selection.weights[key]} onChange={(e) => { mark(`selection.weights.${key}`); onChange({ ...selection, weights: { ...selection.weights, [key]: Number(e.target.value) } }); }} />{rationale(`selection.weights.${key}`)}</label>)}
    </div>
    <div className="small dim" style={{ marginTop: 10 }}>{zh ? '候选生成与筛选条件（均可修改）' : 'Candidate generation and selection (editable)'}</div>
    <div className="form-grid" style={{ marginTop: 4 }}>
      <label className="field">{zh ? '生成候选池' : 'Candidate pool'} {source('candidate.top_k')}<input type="number" min={1} value={candidate.top_k} disabled={!onCandidateChange} onChange={(e) => setCandidate('top_k', Number(e.target.value))} />{rationale('candidate.top_k')}</label>
      <label className="field">{zh ? '允许位点' : 'Allowed positions'} {source('candidate.allowed_positions')}<input value={allowedText} disabled={!onCandidateChange} placeholder={zh ? '留空：全部位点' : 'Empty: all positions'} onChange={(e) => { const text = e.target.value; setAllowedText(text); setCandidate('allowed_positions', text.trim() ? positionList(text) : null); }} />{rationale('candidate.allowed_positions')}</label>
      <label className="field">{zh ? '排除位点' : 'Excluded positions'} {source('candidate.excluded_positions')}<input value={excludedText} disabled={!onCandidateChange} placeholder={zh ? '例如：3, 8, 25' : 'For example: 3, 8, 25'} onChange={(e) => { const text = e.target.value; setExcludedText(text); setCandidate('excluded_positions', positionList(text)); }} />{rationale('candidate.excluded_positions')}</label>
      <label className="field">{zh ? '最低活性评分' : 'Minimum activity score'} {source('selection.min_activity_log10')}<input type="number" step="any" value={selection.min_activity_log10 ?? ''} onChange={(e) => set('min_activity_log10', numeric(e.target.value))} placeholder={zh ? '不限定' : 'Unspecified'} />{rationale('selection.min_activity_log10')}</label>
      <label className="field">{zh ? '稳定性方向' : 'Stability direction'} {source('selection.stability_direction')}<select value={selection.stability_direction} onChange={(e) => set('stability_direction', e.target.value)}><option value="lower_is_better">{zh ? '较低更好' : 'Lower is better'}</option><option value="higher_is_better">{zh ? '较高更好' : 'Higher is better'}</option></select>{rationale('selection.stability_direction')}</label>
      <label className="field">{zh ? '稳定性筛选' : 'Stability filter'} {source('selection.stability_operator')}<select value={selection.stability_operator} onChange={(e) => set('stability_operator', e.target.value)}><option value="none">{zh ? '无阈值' : 'No threshold'}</option><option value="le">≤</option><option value="ge">≥</option></select>{rationale('selection.stability_operator')}</label>
      <label className="field">{zh ? '稳定性阈值' : 'Stability threshold'} {source('selection.stability_threshold')}<input type="number" step="any" value={selection.stability_threshold ?? ''} onChange={(e) => set('stability_threshold', numeric(e.target.value))} placeholder={zh ? '不限定' : 'Unspecified'} />{rationale('selection.stability_threshold')}</label>
    </div>
    {error && <div role="alert" className="error-text small" style={{ marginTop: 6 }}>{errors[error] || error}</div>}
  </fieldset>;
}

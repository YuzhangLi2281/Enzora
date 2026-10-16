import { useRef, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useNavigate } from 'react-router-dom';
import { api } from '../../shared/api/client';
import { automaticReview, candidateError, defaultCandidate, defaultSelection, selectionError, sequenceError, reviewedRequest } from './review';
import { useApp } from '../../app/store';
import type { ClassifyResult, InputRecord, ParseResult } from '../../shared/api/contracts';
import { Panel, Badge } from '../../shared/ui/primitives';
import { InputFiles } from './InputFiles';
import { ReviewParameters } from './ReviewParameters';
import { rememberRequest } from '../runs/requestStorage';

export function Wizard() {
  const lang = useApp((s) => s.lang);
  const sessionId = useApp((s) => s.sessionId);
  const navigate = useNavigate();
  const zh = lang === 'zh';
  const [input, setInput] = useState('');
  const [typeOverride, setTypeOverride] = useState('');
  const [name, setName] = useState('my_enzyme');
  const [sequence, setSequence] = useState('');
  const [records, setRecords] = useState<InputRecord[]>([]);
  const [selectedId, setSelectedId] = useState('');
  const [request, setRequest] = useState('');
  const [interpretation, setInterpretation] = useState('');
  const [warnings, setWarnings] = useState<string[]>([]);
  const [candidate, setCandidate] = useState(defaultCandidate);
  const [selection, setSelection] = useState(defaultSelection);
  const [ligandSmiles, setLigandSmiles] = useState('');
  const [mutationOrder, setMutationOrder] = useState(1);
  const [parsed, setParsed] = useState<ParseResult | null>(null);
  const [goalConfirmed, setGoalConfirmed] = useState(false);
  const [forceCpu, setForceCpu] = useState(false);
  const [busy, setBusy] = useState(false);
  const [preparingParameters, setPreparingParameters] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const guard = useRef(false);
  const status = useQuery({ queryKey: ['session-status', sessionId], queryFn: () => api.sessionStatus(sessionId), refetchInterval: 30_000 });
  const aiReady = status.data?.configured === true;
  const fail = (err: unknown) => setError(err instanceof Error ? err.message : String(err));
  const run = async (action: () => Promise<void>) => {
    if (guard.current) return;
    guard.current = true; setBusy(true); setError(null);
    try { await action(); } catch (err) { fail(err); } finally { guard.current = false; setBusy(false); }
  };
  const prepareParameters = async (protein: string, sequenceName: string, goal: string, baseCandidate = candidate, baseSelection = selection) => {
    setParsed(null); setGoalConfirmed(false);
    if (!aiReady || sequenceError(protein)) return;
    setPreparingParameters(true);
    try {
      const review = await automaticReview({ name: sequenceName, sequence: protein, goal, candidate: baseCandidate, selection: baseSelection, sessionId, aiReady, lang, parse: api.parseRequest });
      if (review) { setCandidate(review.candidate); setSelection(review.selection); setParsed(review.parsed); setGoalConfirmed(true); }
    } finally { setPreparingParameters(false); }
  };
  const choose = async (record: InputRecord, goal = request) => {
    const nextCandidate = defaultCandidate(); const nextSelection = defaultSelection();
    setParsed(null); setGoalConfirmed(false); setCandidate(nextCandidate); setSelection(nextSelection);
    setSelectedId(record.id); setName(record.name); setSequence('');
    let protein = record.sequence;
    if (record.type !== 'protein') {
      const translation = await api.translate(record.sequence, record.type);
      protein = translation.protein_sequence; setWarnings((items) => [...items, ...translation.warnings]);
      setInterpretation((text) => `${text}\n${record.type.toUpperCase()} → ${translation.protein_sequence.length} aa. ${zh ? '请核对翻译结果。' : 'Review the translation.'}`);
    }
    setSequence(protein);
    await prepareParameters(protein, record.name, goal, nextCandidate, nextSelection);
  };
  const consume = async (result: ClassifyResult, found = result.records || []) => {
    setSequence(''); setSelectedId(''); setRecords(found); setInterpretation(result.interpretation || ''); setWarnings(result.warnings || []);
    const nextGoal = result.request_text || request;
    setRequest(nextGoal);
    setGoalConfirmed(false); setParsed(null);
    if (found.length === 1) await choose(found[0], nextGoal);
    if (!found.length) throw new Error(zh ? '表单需要实际序列。名称检索请使用工作台对话。' : 'This form requires a sequence. Use the workbench chat for protein name lookup.');
  };
  const parseGoal = () => run(async () => {
    await prepareParameters(sequence, name, request);
  });
  const submit = () => run(async () => {
    if (sequenceError(sequence) || selectionError(selection) || candidateError(candidate, selection, sequence.length) || (request.trim() && !goalConfirmed)) return;
    if (!ligandSmiles.trim() || !Number.isInteger(mutationOrder) || mutationOrder < 1 || mutationOrder > sequence.length) throw new Error(zh ? '请填写底物并检查突变位点数。' : 'Enter a substrate and a valid mutation order.');
    const runRequest = reviewedRequest({ name, sequence, goal: request, candidate, selection, aiReady, lang, forceCpu, ligandSmiles, mutationOrder });
    const { job_id } = await api.submitRun(runRequest, sessionId);
    rememberRequest(job_id, runRequest); navigate(`/workbench?job=${job_id}`);
  });

  return <main id="main-content" className="page-shell" style={{ maxWidth: 960 }}>
    <div className="page-heading"><div className="eyebrow">ENZORA · DESIGN AN EXPERIMENT</div><h1>{zh ? '新建实验' : 'New experiment'}</h1><p>{zh ? '导入序列，核对 AI 理解和筛选参数，再开始计算。' : 'Import a sequence, review the interpretation and parameters, then run.'}</p></div>
    <Panel title={zh ? '1 · 输入序列或文件' : '1 · Sequence or files'}>
      <label className="field">{zh ? '输入类型' : 'Input type'}<select value={typeOverride} disabled={busy} onChange={(e) => setTypeOverride(e.target.value)}><option value="">{zh ? 'AI 结合上下文理解（推荐）' : 'AI context interpretation (recommended)'}</option><option value="protein">{zh ? '明确指定蛋白序列' : 'Explicit protein sequence'}</option><option value="dna">DNA</option><option value="rna">RNA</option></select></label>
      <textarea rows={5} aria-label={zh ? '原始序列输入' : 'Original sequence input'} value={input} disabled={busy} onChange={(e) => { setInput(e.target.value); setSequence(''); setRecords([]); setSelectedId(''); }} placeholder={zh ? '粘贴 FASTA、蛋白或核酸序列，也可同时描述实验目标。' : 'Paste FASTA, protein or DNA/RNA, optionally with a research goal.'} />
      <div style={{ display: 'flex', gap: 8, marginTop: 8, flexWrap: 'wrap' }}><button className="primary small" disabled={busy || !input.trim()} onClick={() => void run(async () => consume(await api.classify(input, sessionId, typeOverride)))}>{zh ? '读取并准备序列' : 'Prepare sequence'}</button><InputFiles lang={lang} sessionId={sessionId} typeOverride={typeOverride} disabled={busy} onBusy={(value) => { guard.current = value; setBusy(value); }} onResults={async (items) => {
        const all = items.flatMap((item, index) => (item.result.records || []).map((record, i) => ({ ...record, id: `${index}:${i}:${record.id}`, name: `${item.filename} / ${record.name}` })));
        await consume({ ...items[0].result, interpretation: items.map((item) => `${item.filename}: ${item.result.interpretation || ''}`).join('\n'), warnings: items.flatMap((item) => item.result.warnings || []), request_text: items.map((item) => item.result.request_text || '').filter(Boolean).join('\n') || request }, all);
      }} /></div>
      {interpretation && <p className="small" style={{ whiteSpace: 'pre-wrap' }}>{interpretation}</p>}{warnings.map((warning, i) => <div key={i} className="small warn-text">{warning}</div>)}
      {records.length > 1 && <label className="field">{zh ? '选择一条序列 / 结构链' : 'Select a sequence / structure chain'}<select disabled={busy} value={selectedId} onChange={(e) => { const record = records.find((r) => r.id === e.target.value); if (record) void run(async () => choose(record)); }}><option value="">{zh ? '请选择' : 'Select a record'}</option>{records.map((record) => <option key={record.id} value={record.id}>{record.name} · {record.type.toUpperCase()} · {record.sequence.length}</option>)}</select></label>}
    </Panel>
    {selectedId && <div style={{ marginTop: 16 }}><Panel title={zh ? '2 · 核对序列与参数' : '2 · Review sequence and parameters'}>
      <label className="field">{zh ? '实验名称' : 'Experiment name'}<input value={name} disabled={busy} onChange={(e) => setName(e.target.value)} /></label>
      <label className="field">{zh ? '实际送入模型的蛋白序列' : 'Protein sequence sent to the models'} <Badge kind="ok">{sequence.length} aa</Badge><textarea rows={4} className="mono" value={sequence} disabled={busy} onChange={(e) => { setSequence(e.target.value); setParsed(null); setGoalConfirmed(false); }} /></label>
      {sequenceError(sequence) && <div className="small error-text">{zh ? '模型需要 1–1000 个标准氨基酸的大写蛋白序列；请核对未知残基和空格。' : 'Models require 1–1000 uppercase standard amino acids. Check unknown residues and spaces.'}</div>}
      <label className="field">{zh ? '实验目标' : 'Experiment goal'}<textarea rows={3} value={request} disabled={busy} onChange={(e) => { setRequest(e.target.value); setParsed(null); setGoalConfirmed(false); }} placeholder={zh ? '例如：目标 pH 7.5，保留 10 个候选。' : 'For example: target pH 7.5, select 10 candidates.'} /></label>
      <div className="ai-suggestion-summary"><strong>{zh ? 'AI 参数建议' : 'AI parameter suggestions'}</strong><p className="small dim">{zh ? '准备序列后自动填写权重与参数；所有建议均可继续修改。' : 'Weights and parameters are filled automatically after preparing the sequence; all suggestions remain editable.'}</p>{preparingParameters && <p role="status" className="small">{zh ? '正在生成参数和权重建议…' : 'Generating parameter and weight suggestions…'}</p>}</div>
      <div style={{ display: 'flex', gap: 8, marginTop: 8, flexWrap: 'wrap' }}><button className="primary small" disabled={busy || !aiReady || !!sequenceError(sequence)} onClick={() => void parseGoal()}>{zh ? '重新生成 AI 参数建议' : 'Regenerate AI parameter suggestions'}</button><button className="ghost small" disabled={busy} onClick={() => { setGoalConfirmed(true); setParsed(null); }}>{zh ? '采用当前参数' : 'Accept current parameters'}</button></div>
      {parsed?.interpretation && <p className="small">AI: {parsed.interpretation}</p>}{parsed?.notes?.map((note) => <p className="small warn-text" key={note}>{note}</p>)}
      <ReviewParameters ligandSmiles={ligandSmiles} onLigandChange={setLigandSmiles} mutationOrder={mutationOrder} onMutationOrderChange={setMutationOrder} lang={lang} candidate={candidate} selection={selection} onChange={setSelection} onCandidateChange={setCandidate} parseResult={parsed} sequenceLength={sequence.length} disabled={busy} />
      <label className="small" style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 10 }}><input type="checkbox" checked={forceCpu} onChange={(e) => setForceCpu(e.target.checked)} />{zh ? '强制 CPU（仅用于调试，计算较慢）' : 'Force CPU (slow, for debugging)'}</label>
      {!aiReady && <p className="small warn-text">{zh ? '服务端未配置 Gemini。可以明确指定序列类型并设置参数运行；AI 分析暂不可用。' : 'Gemini is not configured. Explicit sequence types and manual parameters work; AI analysis is unavailable.'}</p>}
      {request.trim() && !goalConfirmed && <p className="small warn-text">{zh ? '请解析目标，或检查参数后点击“采用当前参数”。' : 'Parse the goal, or review and accept the current parameters.'}</p>}
      <button className="primary" style={{ marginTop: 12 }} disabled={busy || !!sequenceError(sequence) || !!selectionError(selection) || !!candidateError(candidate, selection, sequence.length) || (!!request.trim() && !goalConfirmed)} onClick={() => void submit()}>{busy ? (zh ? '正在处理…' : 'Processing…') : (zh ? '确认并提交运行' : 'Confirm and submit run')}</button>
    </Panel></div>}
    {error && <div role="alert" className="error-text small" style={{ marginTop: 12 }}>{error}</div>}
  </main>;
}

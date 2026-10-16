import { useRef, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { api, isDemo } from '../../shared/api/client';
import { t } from '../../shared/lib/i18n';
import { useApp } from '../../app/store';
import type { PlanPreview, ToolInfo } from '../../shared/api/contracts';
import { Badge, ErrorNote, Loading, Panel, fmt } from '../../shared/ui/primitives';

const CAPABILITIES = [
  { key: 'activity_prediction', zh: '突变活性预测', en: 'Mutation activity' },
  { key: 'ph_prediction', zh: '最佳 pH 预测', en: 'Optimum pH' },
  { key: 'stability_prediction', zh: '稳定性预测', en: 'Stability' },
  { key: 'codon_optimization', zh: '密码子优化', en: 'Codon optimization' },
  { key: 'sequence_translation', zh: '核酸翻译', en: 'DNA/RNA translation' },
];
const TOOL_LABELS: Record<string, [string, string]> = {
  preview_candidate_preparation: ['准备突变与候选序列', 'Prepare mutations and candidate sequences'],
  phenom: ['Phenom · 活性预测', 'Phenom · Activity prediction'],
  ephod: ['EpHod · pH 预测', 'EpHod · pH prediction'],
  unistab: ['UniStab · 稳定性预测', 'UniStab · Stability prediction'],
  codon_optimizer: ['密码子优化', 'Codon optimization'],
  sequence_translate: ['核酸翻译', 'DNA/RNA translation'],
};

export function Tools() {
  const lang = useApp((s) => s.lang);
  const sessionId = useApp((s) => s.sessionId);
  const runRequest = useApp((s) => s.runRequest);
  const zh = lang === 'zh';
  const toolsQuery = useQuery({ queryKey: ['tools'], queryFn: () => api.tools() });
  const keyStatus = useQuery({ queryKey: ['session-status', sessionId], queryFn: () => api.sessionStatus(sessionId), refetchInterval: 30_000 });
  const [health, setHealth] = useState<Record<string, unknown> | null>(null);
  const [healthBusy, setHealthBusy] = useState<string | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [planCapabilities, setPlanCapabilities] = useState<string[]>(['activity_prediction', 'ph_prediction', 'stability_prediction']);
  const [planRequest, setPlanRequest] = useState('');
  const [initialType, setInitialType] = useState('protein');
  const [planResult, setPlanResult] = useState<PlanPreview | null>(null);
  const [planError, setPlanError] = useState<string | null>(null);
  const [planning, setPlanning] = useState<'ai' | 'deterministic' | null>(null);
  const planGuard = useRef(false);
  const aiReady = keyStatus.data?.configured === true;

  const runHealth = async (kind: 'check-paths' | 'check-gpu' | 'validate-cpu') => {
    setHealthBusy(kind); setHealth(null); setHealthError(null);
    try { setHealth(await api.health(kind)); }
    catch (error) { setHealthError(error instanceof Error ? error.message : String(error)); }
    finally { setHealthBusy(null); }
  };
  const toggleCap = (cap: string) => {
    setPlanResult(null);
    setPlanCapabilities((prev) => prev.includes(cap) ? prev.filter((c) => c !== cap) : [...prev, cap]);
  };
  const runPlan = async (useLlm: boolean) => {
    if (planGuard.current) return;
    setPlanError(null);
    if (useLlm && !aiReady) { setPlanError(t('chat.needKey', lang)); return; }
    planGuard.current = true; setPlanning(useLlm ? 'ai' : 'deterministic'); setPlanResult(null);
    try {
      const artifacts = initialType === 'protein' ? ['protein_sequence'] : ['nucleotide_sequence', 'sequence_type'];
      const inputSummary = {
        sequence_type: initialType,
        current_goal: planRequest,
        ...(initialType === 'protein' && runRequest ? {
          sequence_name: runRequest.sequence_name,
          sequence_length: runRequest.sequence.length,
        } : {}),
      };
      setPlanResult(await api.plan(planCapabilities, useLlm, sessionId, planRequest, artifacts, inputSummary));
    } catch (err) { setPlanError(err instanceof Error ? err.message : String(err)); }
    finally { planGuard.current = false; setPlanning(null); }
  };

  return <main id="main-content" className="page-shell" style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
    <div className="page-heading"><div className="eyebrow">MODEL CAPABILITIES</div><h1>{t('nav.tools', lang)}</h1><p>{zh ? '检查模型服务，按实验目标生成可核对的工作流方案。' : 'Check model services and build a workflow around your experiment goal.'}</p></div>
    <Panel title={t('tools.health', lang)}>
      {healthError && <ErrorNote text={healthError} />}
      <div className="chip-row" style={{ marginBottom: 10 }}>
        <button className="chip" disabled={Boolean(healthBusy)} onClick={() => void runHealth('check-paths')}>{t('tools.checkPaths', lang)}</button>
        <button className="chip" disabled={Boolean(healthBusy)} onClick={() => void runHealth('check-gpu')}>{t('tools.checkGpu', lang)}</button>
        <button className="chip" disabled={Boolean(healthBusy)} onClick={() => void runHealth('validate-cpu')}>{t('tools.validateCpu', lang)}</button>
        {healthBusy && <span className="small dim pulse">{healthBusy}…</span>}
      </div>
      {health && <pre className="muted-box mono small" style={{ overflow: 'auto', maxHeight: 240 }}>{JSON.stringify(health, null, 2)}</pre>}
    </Panel>
    <Panel title={zh ? '实验工作流方案' : 'Experiment workflow plan'}>
      <div className="planner-heading"><Badge kind={aiReady ? 'ok' : 'warn'}>{isDemo() ? (zh ? '演示方案' : 'Demo plan') : aiReady ? (zh ? 'Gemini 已配置' : 'Gemini configured') : (zh ? 'Gemini 未配置' : 'Gemini not configured')}</Badge><span className="small dim">{zh ? '方案预览；实际计算在工作台确认后提交。' : 'Plan preview. Submit actual computation from the workbench after review.'}</span></div>
      <label className="field">{zh ? '实验目标' : 'Experiment goal'}<textarea rows={3} disabled={!!planning} value={planRequest} onChange={(e) => { setPlanRequest(e.target.value); setPlanResult(null); }} placeholder={zh ? '例如：基于蛋白序列筛选在 pH 7.5 附近活性更好的突变，并比较稳定性。' : 'For example: find mutants with better activity near pH 7.5 and compare stability.'} /></label>
      <label className="field">{zh ? '已有输入' : 'Available input'}<select disabled={!!planning} value={initialType} onChange={(e) => { setInitialType(e.target.value); setPlanResult(null); }}><option value="protein">{zh ? '蛋白序列' : 'Protein sequence'}</option><option value="nucleotide">{zh ? 'DNA / RNA 序列' : 'DNA / RNA sequence'}</option></select></label>
      <div className="chip-row">{CAPABILITIES.map((cap) => <button key={cap.key} disabled={!!planning} className={`chip ${planCapabilities.includes(cap.key) ? 'active' : ''}`} onClick={() => toggleCap(cap.key)}>{zh ? cap.zh : cap.en}</button>)}</div>
      <div className="planner-actions"><button className="ghost" disabled={!planCapabilities.length || !!planning} onClick={() => void runPlan(false)}>{planning === 'deterministic' ? (zh ? '正在生成…' : 'Planning…') : (zh ? '基础方案' : 'Deterministic plan')}</button><button className="primary" disabled={!planCapabilities.length || !!planning || keyStatus.isPending} onClick={() => void runPlan(true)}>{planning === 'ai' ? (zh ? 'Gemini 正在规划…' : 'Gemini is planning…') : 'Gemini Plan'}</button></div>
      {planError && <ErrorNote text={planError} />}
      {planResult && <div className="plan-preview" aria-live="polite"><p>{planResult.rationale}</p>{planResult.notes?.map((note) => <p className="small dim" key={note}>{note}</p>)}<ol className="plan-steps">{planResult.steps.map((step, index) => <li key={step.id}><span className="plan-step-number">{index + 1}</span><div><strong>{TOOL_LABELS[step.tool]?.[zh ? 0 : 1] || step.tool}</strong><div className="small dim">{step.depends_on.length ? `${zh ? '依赖步骤' : 'Depends on steps'} ${step.depends_on.map((id) => planResult.steps.findIndex((s) => s.id === id) + 1).join(', ')}` : (zh ? '使用已有输入' : 'Uses available input')}</div></div></li>)}</ol><details><summary className="small dim">{zh ? '查看完整方案数据' : 'Show complete plan data'}</summary><pre className="muted-box mono small" style={{ overflow: 'auto', maxHeight: 260 }}>{JSON.stringify(planResult, null, 2)}</pre></details></div>}
    </Panel>
    <Panel title={t('tools.catalog', lang)}>
      {toolsQuery.isError && <ErrorNote text={toolsQuery.error.message} />}{toolsQuery.isLoading && <Loading />}
      <div className="table-scroll"><table className="data"><thead><tr><th>{zh ? '工具' : 'Tool'}</th><th>{zh ? '能力' : 'Capabilities'}</th><th>{zh ? '输入 → 输出' : 'Inputs → outputs'}</th><th>{zh ? '状态' : 'Health'}</th></tr></thead><tbody>{(toolsQuery.data ?? []).map((tool: ToolInfo) => <tr key={tool.name}><td><span className="badge mono accent">{tool.name}</span><div className="small dim">{tool.description}</div></td><td className="small">{tool.capabilities.join(', ')}</td><td className="small mono">{tool.inputs.join(', ')} → {tool.outputs.join(', ')}</td><td>{tool.health.available ? <Badge kind="ok">{t('tools.available', lang)}</Badge> : <Badge kind="err" title={tool.health.details.join('; ')}>{t('tools.unavailable', lang)}</Badge>}</td></tr>)}</tbody></table></div>
    </Panel>
  </main>;
}
export { fmt };

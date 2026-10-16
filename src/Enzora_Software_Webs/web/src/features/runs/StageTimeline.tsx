/** 11-stage pipeline timeline with glowing status dots and per-stage stats. */

import type { StageRecord } from '../../shared/api/contracts';
import { t } from '../../shared/lib/i18n';
import { useApp } from '../../app/store';

export const STAGE_LABELS: Record<string, { en: string; zh: string }> = {
  resolve_user_request: { en: 'Resolve request', zh: '解析请求' },
  phenom_activity_prediction: { en: 'PHENOM kcat prediction', zh: 'PHENOM kcat 预测' },
  vesm_library_construction: { en: 'VESM library construction', zh: 'VESM 突变库构建' },
  ephod_ph_prediction: { en: 'EpHod pH', zh: 'EpHod pH 预测' },
  unistab_stability_prediction: { en: 'UniStab ddG', zh: 'UniStab 稳定性' },
  personalized_filter_and_rank: { en: 'Filter & rank', zh: '过滤与排序' },
  point_mutation_feature_analysis: { en: 'Features & Pareto', zh: '描述符与 Pareto' },
  codon_optimization: { en: 'Codon optimization', zh: '密码子优化' },
  visualize_results: { en: 'Visualize', zh: '可视化' },
  llm_result_analysis: { en: 'AI analysis', zh: 'AI 分析' },
  write_markdown_report: { en: 'Report', zh: '报告' },
};

export const ALL_STAGES = Object.keys(STAGE_LABELS);

export function StageTimeline(props: { stages: StageRecord[]; status?: string }) {
  const lang = useApp((s) => s.lang);
  const doneStages = new Map(props.stages.map((s) => [s.stage, s]));
  const doneNames = new Set(doneStages.keys());
  const activeIndex = ['running', 'failed', 'cancelled'].includes(props.status ?? '') ? ALL_STAGES.findIndex((stage) => !doneNames.has(stage)) : -1;
  const failed = props.status === 'failed' || props.status === 'cancelled';

  return (
    <div className="timeline">
      <div className="timeline-summary"><span>{t('stage.timeline', lang)}</span><span className="mono dim">{props.stages.length} / {ALL_STAGES.length}</span></div>
      <div className="progress-track" role="progressbar" aria-label={t('stage.timeline', lang)} aria-valuemin={0} aria-valuemax={ALL_STAGES.length} aria-valuenow={props.stages.length}><span style={{ width: `${100 * props.stages.length / ALL_STAGES.length}%` }} /></div>
      {ALL_STAGES.map((stage, index) => {
        const rec = doneStages.get(stage);
        const state = rec ? 'done' : index === activeIndex ? (failed ? 'failed' : 'active') : 'pending';
        const label = STAGE_LABELS[stage];
        return (
          <div className="tl-row" key={stage}>
            <span className={`tl-dot ${state}`} />
            <span className="tl-name" style={{ color: state === 'pending' ? 'var(--text-dim)' : undefined, opacity: state === 'pending' ? 0.65 : 1 }}>
              {label?.[lang] ?? stage}
            </span>
            <span className="tl-time">{rec?.seconds != null ? `${rec.seconds}s` : ''}</span>
            {rec?.stats && <span className="tl-time">{formatStats(rec.stats)}</span>}
          </div>
        );
      })}
      {props.stages.length === 0 && <div className="dim small">{t('stage.timeline', lang)}…</div>}
    </div>
  );
}

function formatStats(stats: Record<string, unknown>): string {
  const parts = Object.entries(stats)
    .filter(([, v]) => v !== null && v !== undefined)
    .slice(0, 3)
    .map(([k, v]) => `${k}=${typeof v === 'number' ? Math.round(v * 100) / 100 : v}`);
  return parts.length ? `· ${parts.join(' ')}` : '';
}

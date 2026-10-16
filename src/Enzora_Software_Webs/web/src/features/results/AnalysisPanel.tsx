/** Panel 2: stage timeline, live model log tail, chart gallery, AI analysis,
 *  codon summary and the markdown report. */

import { useMemo, useState } from 'react';
import { api } from '../../shared/api/client';
import { t } from '../../shared/lib/i18n';
import { useApp } from '../../app/store';
import type { Candidate, RunMeta, RunResult } from '../../shared/api/contracts';
import { ChartGallery } from './Charts';
import { StageTimeline } from '../runs/StageTimeline';
import { Badge, Empty, fmt } from '../../shared/ui/primitives';
import { artifactFilename, downloadText } from '../../shared/lib/download';

const CHART_TABS = [
  { key: 'hero', labelKey: 'charts.hero' },
  { key: 'scores', labelKey: 'charts.scores' },
  { key: 'scatter', labelKey: 'charts.scatter' },
  { key: 'heatmap', labelKey: 'charts.heatmap' },
  { key: 'seqmap', labelKey: 'charts.seqmap' },
  { key: 'pareto', labelKey: 'charts.pareto' },
];

export function AnalysisPanel(props: {
  meta: RunMeta | null;
  result: RunResult | null;
  logs: { source: string; line: string }[];
  wtSequence: string;
  ranked: Candidate[];
  selected: Candidate[];
  analysisText: string | null;
  analysisError: string | null;
}) {
  const lang = useApp((s) => s.lang);
  const [chartTab, setChartTab] = useState('hero');
  const [downloadError, setDownloadError] = useState<string | null>(null);
  const [downloading, setDownloading] = useState<string | null>(null);
  const reportFilename = artifactFilename(props.result?.markdown_report, 'report.md');
  const analysisFilename = artifactFilename(props.result?.analysis_file, 'analysis.txt');
  const download = async (filename: string) => {
    if (!props.meta || downloading) return;
    setDownloadError(null);
    setDownloading(filename);
    try { downloadText(await api.readArtifact(props.meta.id, filename), filename, filename.endsWith('.md') ? 'text/markdown;charset=utf-8' : 'text/plain;charset=utf-8'); }
    catch (error) { setDownloadError(error instanceof Error ? error.message : String(error)); }
    finally { setDownloading(null); }
  };
  const hasResult = Boolean(props.result);
  const codon = props.result?.codon_optimization;

  const seqLength = useMemo(() => props.wtSequence.length, [props.wtSequence]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', minHeight: 0, gap: 10 }}>
      <details open={!hasResult} className="pipeline-details">
        <summary>{t('stage.timeline', lang)} <span className="mono dim">{props.meta?.stages.length ?? 0}/11</span></summary>
        <StageTimeline stages={props.meta?.stages ?? []} status={props.meta?.status} />
      </details>

      {props.logs.length > 0 && props.meta?.status === 'running' && (
        <div className="muted-box" style={{ maxHeight: 110, overflow: 'auto' }}>
          {props.logs.slice(-8).map((log, i) => (
            <div key={i} className="log-line">
              <span style={{ color: 'var(--accent)' }}>[{log.source}]</span> {log.line}
            </div>
          ))}
        </div>
      )}

      {hasResult && (
        <>
          <div className="chip-row">
            {CHART_TABS.map((tab) => (
              <button key={tab.key} className={`chip ${chartTab === tab.key ? 'active' : ''}`} onClick={() => setChartTab(tab.key)}>
                {t(tab.labelKey, lang)}
              </button>
            ))}
          </div>
          <div style={{ minHeight: 260, flexShrink: 0 }}>
            <ChartGallery candidates={props.ranked} selected={props.selected} sequenceLength={seqLength} active={chartTab} />
          </div>

          <div>
            <div className="small dim" style={{ marginBottom: 4 }}>
              {t('analysis.llm', lang)}
              {props.meta?.id && !props.analysisText && !props.analysisError && (
                <span className="small dim"> ({props.result?.analysis_file})</span>
              )}
            </div>
            {props.analysisText && (
              <div className="muted-box small" style={{ maxHeight: 180, overflow: 'auto', whiteSpace: 'pre-wrap' }}>
                {props.analysisText}
              </div>
            )}
            {props.analysisError && <div className="warn-text small">⚠ {t('analysis.llmFailed', lang)}</div>}
          </div>

          {codon?.enabled ? (
            <div>
              <div className="small dim" style={{ marginBottom: 4 }}>{t('codon.title', lang)}</div>
              <div className="chip-row">
                {(codon.resolved_hosts ?? []).map((host) => (
                  <Badge key={host.organism_key} kind="accent" mono>
                    {host.scientific_name ?? host.organism_key} · TaxID {host.taxid}
                  </Badge>
                ))}
                {codon.status === 'partial' && <Badge kind="warn">partial</Badge>}
              </div>
              {(codon.failed_hosts ?? []).map((f, i) => (
                <div key={i} className="error-text small">✗ {f.query}: {f.error}</div>
              ))}
            </div>
          ) : null}

          {props.result?.markdown_report && (
            <div className="small dim">
              {reportFilename} ·{' '}
              <button className="chip" disabled={Boolean(downloading)} onClick={() => void download(reportFilename)}>{downloading === reportFilename ? (lang === 'zh' ? '正在下载…' : 'Downloading…') : `↓ ${reportFilename}`}</button>
              {' · '}
              <button className="chip" disabled={Boolean(downloading)} onClick={() => void download(analysisFilename)}>{downloading === analysisFilename ? (lang === 'zh' ? '正在下载…' : 'Downloading…') : `↓ ${analysisFilename}`}</button>
              {downloadError && <div role="alert" className="error-text">{downloadError}</div>}
            </div>
          )}
        </>
      )}

      {!hasResult && (props.meta?.status === 'failed' || props.meta?.status === 'cancelled') && (
        <Empty text={props.meta.error ?? t('run.status.failed', lang)} />
      )}
      {!props.meta && <Empty text={lang === 'zh' ? '从对话面板提交一个运行开始' : 'Submit a run from the conversation panel to begin'} />}
    </div>
  );
}

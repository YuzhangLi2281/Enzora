import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Link } from 'react-router-dom';
import { api } from '../../shared/api/client';
import { t } from '../../shared/lib/i18n';
import { useApp } from '../../app/store';
import type { RunMeta } from '../../shared/api/contracts';
import { Badge, Empty, ErrorNote, Loading, Panel, fmt } from '../../shared/ui/primitives';

export function History() {
  const lang = useApp((s) => s.lang);
  const zh = lang === 'zh';
  const query = useQuery({ queryKey: ['runs'], queryFn: () => api.listRuns(), refetchInterval: 5000 });
  const [search, setSearch] = useState('');
  const [status, setStatus] = useState('all');
  const runs = (query.data ?? []).filter((run) => (status === 'all' || run.status === status)
    && `${run.id} ${run.config_summary?.sequence_name ?? ''} ${run.config_summary?.user_request ?? ''}`.toLowerCase().includes(search.toLowerCase()));
  const statusKind = (value: RunMeta['status']) => value === 'success' ? 'ok' : value === 'failed' ? 'err' : value === 'cancelled' ? 'warn' : 'accent';
  return (
    <main id="main-content" className="page-shell">
      <div className="page-heading"><div className="eyebrow">EXPERIMENT ARCHIVE</div><h1>{t('history.title', lang)}</h1><p>{zh ? '回顾每一次设计，重新打开实验以继续追踪进度和比较结果。' : 'Revisit your designs, follow active experiments and compare their outcomes.'}</p></div>
      <div className="page-toolbar">
        <input aria-label={zh ? '搜索实验' : 'Search experiments'} placeholder={zh ? '搜索名称、任务编号或研究目标…' : 'Search by name, run ID or research goal…'} value={search} onChange={(e) => setSearch(e.target.value)} />
        <select aria-label={zh ? '任务状态' : 'Run status'} value={status} onChange={(e) => setStatus(e.target.value)}><option value="all">{zh ? '所有状态' : 'All statuses'}</option>{['queued', 'running', 'success', 'failed', 'cancelled'].map((value) => <option key={value} value={value}>{t(`run.status.${value}`, lang)}</option>)}</select>
        <button onClick={() => void query.refetch()} disabled={query.isFetching}>{zh ? '刷新' : 'Refresh'}</button>
        <Link className="button-link" to="/new">{zh ? '新建实验' : 'New experiment'} ↗</Link>
      </div>
      <Panel title={zh ? `实验记录 · ${runs.length}` : `Experiments · ${runs.length}`}>
        {query.isLoading && <Loading />}
        {query.isError && <ErrorNote text={query.error.message} />}
        {!query.isLoading && !query.isError && !runs.length && <Empty text={search || status !== 'all' ? (zh ? '没有匹配的实验，请调整筛选条件。' : 'No experiments match these filters.') : t('history.empty', lang)} />}
        {!!runs.length && <div className="table-scroll"><table className="data">
          <thead><tr>{(zh ? ['实验', '状态', '序列', '进度', '创建时间', '首选候选', '操作'] : ['Experiment', 'Status', 'Sequence', 'Progress', 'Created', 'Top candidate', 'Open']).map((label) => <th key={label}>{label}</th>)}</tr></thead>
          <tbody>{runs.map((run) => {
            const top = run.result?.selected?.[0];
            return <tr key={run.id}>
              <td><strong>{run.config_summary?.sequence_name ?? run.id}</strong><div className="mono small dim">{run.id}</div><div className="small dim">{run.config_summary?.user_request}</div></td>
              <td><Badge kind={statusKind(run.status)}>{t(`run.status.${run.status}`, lang)}</Badge></td>
              <td className="num">{run.config_summary?.sequence_length ?? '—'} aa</td>
              <td className="num">{run.stages.length}/11</td>
              <td className="small dim">{new Date(run.created_at * 1000).toLocaleString(zh ? 'zh-CN' : 'en-US')}</td>
              <td>{top ? <Badge kind="ok" mono>{top.mutation} · {fmt(top.final_score)}</Badge> : '—'}</td>
              <td><Link className="button-link" to={`/workbench?job=${encodeURIComponent(run.id)}`}>{t('run.view', lang)} ↗</Link></td>
            </tr>;
          })}</tbody>
        </table></div>}
      </Panel>
    </main>
  );
}

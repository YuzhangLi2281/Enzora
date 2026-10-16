/** Candidate table (panel 4, lower part): sortable, filterable, row drawer
 *  with the full mutation descriptor set + mutant sequence copy. */

import { useMemo, useState } from 'react';
import type { Candidate } from '../../shared/api/contracts';
import { t } from '../../shared/lib/i18n';
import { useApp } from '../../app/store';
import { Badge, Empty, fmt } from '../../shared/ui/primitives';
import { downloadText, serializeCsv } from '../../shared/lib/download';

const COLUMNS: { key: keyof Candidate; label: { en: string; zh: string }; mono?: boolean }[] = [
  { key: 'mutation', label: { en: 'Mutation', zh: '突变' }, mono: true },
  { key: 'position', label: { en: 'Pos', zh: '位置' } },
  { key: 'activity_log10', label: { en: 'Activity', zh: '活性' }, mono: true },
  { key: 'ph_opt', label: { en: 'pH opt', zh: '最适 pH' }, mono: true },
  { key: 'ddg', label: { en: 'ddG', zh: 'ddG' }, mono: true },
  { key: 'final_score', label: { en: 'Score', zh: '总分' }, mono: true },
  { key: 'blosum62', label: { en: 'BL62', zh: 'BL62' }, mono: true },
];

export function CandidateTable(props: { rows: Candidate[] }) {
  const lang = useApp((s) => s.lang);
  const selectedMutation = useApp((s) => s.selectedMutation);
  const setSelectedMutation = useApp((s) => s.setSelectedMutation);
  const [sortBy, setSortBy] = useState<keyof Candidate>('final_score');
  const [sortAsc, setSortAsc] = useState(false);
  const [onlyPassed, setOnlyPassed] = useState(false);
  const [onlyPareto, setOnlyPareto] = useState(false);
  const [drawerMutation, setDrawerMutation] = useState<string | null>(null);
  const drawer = props.rows.find((row) => row.mutation === drawerMutation) ?? null;
  const [search, setSearch] = useState('');

  const rows = useMemo(() => {
    let out = [...props.rows];
    if (search.trim()) out = out.filter((row) => row.mutation.toLowerCase().includes(search.trim().toLowerCase()));
    if (onlyPassed) out = out.filter((c) => c.passed !== false);
    if (onlyPareto) out = out.filter((c) => String(c.pareto_optimal) === 'true');
    out.sort((a, b) => {
      const av = a[sortBy]; const bv = b[sortBy];
      const cmp = typeof av === 'number' && typeof bv === 'number' ? av - bv : String(av ?? '').localeCompare(String(bv ?? ''));
      return sortAsc ? cmp : -cmp;
    });
    return out;
  }, [props.rows, sortBy, sortAsc, onlyPassed, onlyPareto, search]);

  if (!props.rows.length) return <Empty text={t('table.noData', lang)} />;

  const toggleSort = (key: keyof Candidate) => {
    if (sortBy === key) setSortAsc(!sortAsc);
    else { setSortBy(key); setSortAsc(false); }
  };

  return (
    <div>
      <input style={{ marginBottom: 8 }} aria-label={lang === 'zh' ? '搜索突变' : 'Search mutations'} placeholder={lang === 'zh' ? '搜索突变，如 A23V' : 'Find a mutation, e.g. A23V'} value={search} onChange={(e) => setSearch(e.target.value)} />
      <div className="chip-row" style={{ marginBottom: 8 }}>
        <button className={`chip ${onlyPassed ? 'active' : ''}`} onClick={() => setOnlyPassed(!onlyPassed)}>{t('table.passed', lang)}</button>
        <button className={`chip ${onlyPareto ? 'active' : ''}`} onClick={() => setOnlyPareto(!onlyPareto)}>{t('table.pareto', lang)}</button>
        <span className="spacer" />
        <button className="chip" disabled={!rows.length} onClick={() => downloadText(serializeCsv(rows, [...COLUMNS.map((column) => column.key as string), 'passed', 'pareto_optimal', 'mutant_sequence']), 'enzyme-candidates.csv', 'text/csv;charset=utf-8')}>{lang === 'zh' ? '导出 CSV' : 'Export CSV'}</button>
        <span className="dim small">{rows.length}</span>
      </div>
      <div style={{ maxHeight: 320, overflow: 'auto' }}>
        <table className="data">
          <thead>
            <tr>
              {COLUMNS.map((col) => (
                <th key={col.key} aria-sort={sortBy === col.key ? (sortAsc ? 'ascending' : 'descending') : 'none'}>
                  <button className="table-sort" onClick={() => toggleSort(col.key)}>{col.label[lang]} {sortBy === col.key ? (sortAsc ? '↑' : '↓') : ''}</button>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((c, rank) => (
              <tr
                key={c.mutation}
                className={`${selectedMutation === c.mutation ? 'selected' : ''} ${String(c.pareto_optimal) === 'true' ? 'pareto' : ''}`}
                tabIndex={0}
                aria-label={`${c.mutation}: ${fmt(c.final_score)}`}
                onKeyDown={(event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); setSelectedMutation(c.mutation); setDrawerMutation(c.mutation); } }}
                onClick={() => { setSelectedMutation(c.mutation); setDrawerMutation(c.mutation); }}
                style={{ cursor: 'pointer' }}
              >
                <td>
                  <span className="badge mono accent">{rank + 1} {c.mutation}</span>
                  {String(c.pareto_optimal) === 'true' && <Badge kind="ok">P</Badge>}
                </td>
                <td className="num">{c.position ?? '—'}</td>
                <td className="num">{fmt(c.activity_log10)}</td>
                <td className="num">{fmt(c.ph_opt, 2)}</td>
                <td className="num">{fmt(c.ddg)}</td>
                <td className="num" style={{ fontWeight: 700 }}>{fmt(c.final_score)}</td>
                <td className="num">{c.blosum62 ?? '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {!rows.length && <Empty text={lang === 'zh' ? '没有匹配的候选。' : 'No matching candidates.'} />}
      {drawer && <CandidateDrawer candidate={drawer} onClose={() => setDrawerMutation(null)} />}
    </div>
  );
}

function CandidateDrawer(props: { candidate: Candidate; onClose: () => void }) {
  const lang = useApp((s) => s.lang);
  const c = props.candidate;
  const rows: [string, string][] = [
    [t('metrics.title', lang), c.mutation],
    ['activity_log10', fmt(c.activity_log10)],
    ['ph_opt', fmt(c.ph_opt, 2)],
    ['ΔpH vs WT', fmt(c.delta_ph_from_wt, 2)],
    ['ddG (UniStab)', fmt(c.ddg)],
    ['final_score', fmt(c.final_score)],
    ['BLOSUM62', String(c.blosum62 ?? '—')],
    ['residue class', c.residue_class_change ?? '—'],
    ['Δ hydrophobicity', fmt(c.delta_hydrophobicity, 2)],
    ['Δ volume (Å³)', fmt(c.delta_volume, 1)],
    ['Δ charge', fmt(c.delta_charge, 1)],
  ];
  return (
    <div className="muted-box fade-in" style={{ marginTop: 10 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
        <Badge kind="accent" mono>{c.mutation}</Badge>
        {c.passed === false ? <Badge kind="err">{c.reasons || t('table.failed', lang)}</Badge> : <Badge kind="ok">{t('table.passed', lang)}</Badge>}
        {String(c.pareto_optimal) === 'true' && <Badge kind="ok">{t('table.pareto', lang)}</Badge>}
        <span className="spacer" />
        <button className="ghost small" onClick={props.onClose}>×</button>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(150px, 1fr))', gap: 6 }}>
        {rows.map(([k, v]) => (
          <div key={k} style={{ borderBottom: '1px solid var(--border)', padding: '4px 0' }}>
            <div className="dim" style={{ fontSize: 10 }}>{k}</div>
            <div className="mono" style={{ fontSize: 13 }}>{v}</div>
          </div>
        ))}
      </div>
      {c.mutant_sequence && (
        <div style={{ marginTop: 8 }}>
          <div className="dim small" style={{ marginBottom: 4 }}>mutant_sequence</div>
          <div className="seq-line small">{c.mutant_sequence}</div>
          <button
            className="ghost small"
            style={{ marginTop: 6 }}
            onClick={() => { void navigator.clipboard?.writeText(c.mutant_sequence ?? ''); }}
          >
            ⧉ copy
          </button>
        </div>
      )}
    </div>
  );
}

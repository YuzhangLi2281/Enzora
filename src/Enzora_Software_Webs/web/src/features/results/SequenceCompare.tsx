/** Panel 3, Tab 2: WT vs designed mutants residue-by-residue.
 *  Three sub-views: single candidate, Top-K mutation matrix, codon compare.
 *  Pure frontend rendering — data comes from the candidate rows and codon rows
 *  the backend already produced (no extra backend work). */

import { useMemo, useState } from 'react';
import type { Candidate, HostProfile } from '../../shared/api/contracts';
import { t } from '../../shared/lib/i18n';
import { useApp } from '../../app/store';
import { Badge, Chips, Empty, fmt } from '../../shared/ui/primitives';

const RULER_STEP = 10;

function scoreColor(score: number): string {
  // blue -> cyan -> green as the score rises (final_score is 0..1)
  const v = Math.max(0, Math.min(1, score));
  const hue = 215 - v * 90; // 215 (blue) -> 125 (green)
  return `hsl(${hue} 80% 60%)`;
}

export function SequenceCompare(props: {
  wtSequence: string;
  candidates: Candidate[];
  codonRows: HostProfile[];
  selectedMutation: string | null;
  onPick: (mutation: string | null) => void;
}) {
  const lang = useApp((s) => s.lang);
  const [view, setView] = useState('single');
  const selected = props.candidates.find((c) => c.mutation === props.selectedMutation) ?? props.candidates[0] ?? null;

  const viewOptions = [
    { key: 'single', label: t('molecule.single', lang) },
    { key: 'matrix', label: t('molecule.matrix', lang) },
    { key: 'codon', label: t('molecule.codon', lang) },
  ];

  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10 }}>
        <Chips options={viewOptions} value={view} onChange={setView} />
        {view === 'single' && (
          <select
            style={{ width: 'auto' }}
            value={selected?.mutation ?? ''}
            onChange={(e) => props.onPick(e.target.value || null)}
          >
            {props.candidates.map((c) => (
              <option key={c.mutation} value={c.mutation}>{c.mutation} — {fmt(c.final_score)}</option>
            ))}
          </select>
        )}
      </div>
      {view === 'single' && <SingleView wt={props.wtSequence} candidate={selected} onPickMutation={props.onPick} />}
      {view === 'matrix' && <MatrixView wt={props.wtSequence} candidates={props.candidates} selected={props.selectedMutation} onPick={props.onPick} />}
      {view === 'codon' && <CodonView wt={props.wtSequence} codonRows={props.codonRows} />}
    </div>
  );
}

function Ruler({ length }: { length: number }) {
  const marks: number[] = [];
  for (let p = 10; p <= length; p += RULER_STEP) marks.push(p);
  return (
    <div className="seq-ruler" style={{ paddingLeft: 118 }}>
      <span style={{ width: 118 }} />
      {Array.from({ length }, (_, i) => (
        <span key={i} style={{ width: '1ch', textAlign: 'center', flexShrink: 0 }}>
          {marks.includes(i + 1) ? i + 1 : ''}
        </span>
      ))}
    </div>
  );
}

function SingleView(props: { wt: string; candidate: Candidate | null; onPickMutation: (m: string | null) => void }) {
  const lang = useApp((s) => s.lang);
  const { wt, candidate } = props;
  if (!candidate) return <Empty text={t('table.noData', lang)} />;
  const match = candidate.mutation.match(/^([A-Z])(\d+)([A-Z])$/);
  const pos = match ? Number(match[2]) : null;
  const mutant = candidate.mutant_sequence ?? '';

  return (
    <div className="fade-in">
      <div className="small dim" style={{ marginBottom: 4 }}>WT</div>
      <div className="seq-line">
        {Array.from(wt).map((aa, i) => {
          const isMut = i + 1 === pos;
          return (
            <span
              key={i}
              className={`seq-pos ${isMut ? 'selected' : ''}`}
              onClick={() => isMut && props.onPickMutation(candidate.mutation)}
            >
              {aa}
            </span>
          );
        })}
      </div>
      <div className="small dim" style={{ margin: '8px 0 4px' }}>
        {candidate.mutation} — mutant
      </div>
      <div className="seq-line">
        {Array.from(wt).map((aa, i) => {
          const isMut = i + 1 === pos;
          return isMut ? (
            <span key={i} className="seq-mut" onClick={() => props.onPickMutation(candidate.mutation)}>
              {mutant[i] ?? '?'}
            </span>
          ) : (
            <span key={i} className="seq-dot">·</span>
          );
        })}
      </div>
      {candidate && (
        <div className="muted-box fade-in" style={{ marginTop: 10 }}>
          <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginBottom: 6 }}>
            <Badge kind="accent" mono>{candidate.mutation}</Badge>
            {candidate.passed === false ? <Badge kind="err">{candidate.reasons || 'filtered'}</Badge> : <Badge kind="ok">{t('table.passed', lang)}</Badge>}
            {String(candidate.pareto_optimal) === 'true' && <Badge kind="ok">Pareto</Badge>}
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(130px, 1fr))', gap: 4 }}>
            {[
              ['activity_log10', fmt(candidate.activity_log10)],
              ['ph_opt', fmt(candidate.ph_opt, 2)],
              ['ΔpH', fmt(candidate.delta_ph_from_wt, 2)],
              ['ddG', fmt(candidate.ddg)],
              ['score', fmt(candidate.final_score)],
              ['BLOSUM62', String(candidate.blosum62 ?? '—')],
              ['class', candidate.residue_class_change ?? '—'],
              ['Δhydro', fmt(candidate.delta_hydrophobicity, 2)],
              ['Δvol', fmt(candidate.delta_volume, 1)],
              ['Δcharge', fmt(candidate.delta_charge, 1)],
            ].map(([k, v]) => (
              <div key={k} style={{ borderBottom: '1px solid var(--border)', padding: '3px 0' }}>
                <div className="dim" style={{ fontSize: 10 }}>{k}</div>
                <div className="mono" style={{ fontSize: 12 }}>{v}</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function MatrixView(props: { wt: string; candidates: Candidate[]; selected: string | null; onPick: (m: string | null) => void }) {
  const lang = useApp((s) => s.lang);
  const { positions, rows } = useMemo(() => {
    const posSet = new Set<number>();
    const parsed = props.candidates.map((c) => {
      const m = c.mutation.match(/^([A-Z])(\d+)([A-Z])$/);
      const position = m ? Number(m[2]) : null;
      if (position) posSet.add(position);
      return { c, position, wt: m?.[1], mt: m?.[3] };
    });
    const ordered = [...posSet].sort((a, b) => a - b);
    return { positions: ordered, rows: parsed.slice(0, 18) };
  }, [props.candidates]);

  if (!rows.length) return <Empty text={t('table.noData', lang)} />;

  return (
    <div style={{ overflow: 'auto' }}>
      <table className="data" style={{ fontSize: 12 }}>
        <thead>
          <tr>
            <th>mutant</th>
            {positions.map((p) => <th key={p} style={{ textAlign: 'center' }}>{p}</th>)}
          </tr>
        </thead>
        <tbody>
          <tr>
            <td className="mono dim">WT</td>
            {positions.map((p) => <td key={p} style={{ textAlign: 'center' }} className="mono dim">{props.wt[p - 1]}</td>)}
          </tr>
          {rows.map(({ c, position, mt }) => (
            <tr
              key={c.mutation}
              className={`${props.selected === c.mutation ? 'selected' : ''} ${String(c.pareto_optimal) === 'true' ? 'pareto' : ''}`}
              onClick={() => props.onPick(c.mutation)}
              style={{ cursor: 'pointer' }}
            >
              <td>
                <span className="badge mono accent">{c.mutation}</span>
              </td>
              {positions.map((p) => (
                <td key={p} style={{ textAlign: 'center' }}>
                  {position === p ? (
                    <span
                      className="seq-mut"
                      style={{ background: scoreColor(Number(c.final_score ?? 0)) }}
                      title={`${c.mutation} score=${fmt(c.final_score)}`}
                    >
                      {mt}
                    </span>
                  ) : <span className="seq-dot">·</span>}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      <div className="small dim" style={{ marginTop: 6 }}>{t('table.pareto', lang)} ▸ {lang === 'zh' ? '左缘绿光行' : 'green edge'}</div>
    </div>
  );
}

function CodonView(props: { wt: string; codonRows: HostProfile[] }) {
  const lang = useApp((s) => s.lang);
  const [hostIndex, setHostIndex] = useState(0);
  const rows = props.codonRows.filter((r) => r.optimized_cds);
  if (!rows.length) return <Empty text={t('codon.notRequested', lang)} />;
  const host = rows[Math.min(hostIndex, rows.length - 1)];
  const cds = (host.optimized_cds ?? '').match(/.{3}/g) ?? [];

  return (
    <div className="fade-in">
      <Chips
        options={rows.map((r, i) => ({ key: String(i), label: `${r.scientific_name ?? r.organism_key}` }))}
        value={String(Math.min(hostIndex, rows.length - 1))}
        onChange={(k) => setHostIndex(Number(k))}
      />
      <div style={{ margin: '10px 0 6px', display: 'flex', gap: 12, flexWrap: 'wrap' }}>
        <Badge kind="accent">{host.codon_source ?? 'kazusa'}</Badge>
        <Badge kind="plain" mono>TaxID {host.taxid}</Badge>
        <Badge kind="plain" mono>{t('codon.gc', lang)} {(Number(host.gc_fraction ?? 0) * 100).toFixed(1)}%</Badge>
        <Badge kind="plain" mono>stop {host.stop_codon ?? '—'}</Badge>
      </div>
      <div className="seq-line small" style={{ letterSpacing: 0.5, lineHeight: 2 }}>
        {Array.from(props.wt.slice(0, cds.length)).map((aa, i) => (
          <span key={i} style={{ marginRight: 4 }}>
            <span className="dim" style={{ fontSize: 10 }}>{aa}</span>
            <span className="mono" style={{ color: 'var(--accent2)' }}>{cds[i] ?? '---'}</span>
          </span>
        ))}
      </div>
    </div>
  );
}

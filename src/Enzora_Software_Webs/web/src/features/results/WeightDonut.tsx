/** Tri-color weight donut (pH / activity / stability) with animated segments.
 *  In adjust mode the three sliders drive it; the donut always mirrors the
 *  normalized weights used by the scoring formula. */

import { normalizedWeights } from './rerank';
import { t } from '../../shared/lib/i18n';
import { useApp } from '../../app/store';
import { Badge } from '../../shared/ui/primitives';

const SEGMENTS = [
  { key: 'ph' as const, color: 'var(--accent)', labelKey: 'weights.ph' },
  { key: 'activity' as const, color: 'var(--accent2)', labelKey: 'weights.activity' },
  { key: 'stability' as const, color: 'var(--accent3)', labelKey: 'weights.stability' },
];

export function WeightDonut(props: {
  weights: { ph: number; activity: number; stability: number };
  size?: number;
  adjust?: boolean;
  applied?: boolean;
  onChange?: (w: { ph: number; activity: number; stability: number }) => void;
}) {
  const lang = useApp((s) => s.lang);
  const size = props.size ?? 100;
  if (Object.values(props.weights).some((v) => !Number.isFinite(v) || v < 0) || Object.values(props.weights).reduce((a,b) => a+b, 0) <= 0) return <div role="alert">{lang === 'zh' ? '请设置有效且不全为零的权重。' : 'Enter valid weights with a positive total.'}</div>;
  const norm = normalizedWeights(props.weights);
  const values = [norm.ph, norm.activity, norm.stability];
  const stroke = Math.max(8, size * 0.09);
  const r = (size - stroke) / 2;
  const circumference = 2 * Math.PI * r;

  let offset = 0;
  const arcs = values.map((v) => {
    const dash = Math.max(0, v * circumference - 2);
    const arc = { dash, offset, v };
    offset += v * circumference;
    return arc;
  });

  const setRaw = (key: 'ph' | 'activity' | 'stability', value: number) => {
    if (!props.onChange) return;
    const remaining = SEGMENTS.filter((segment) => segment.key !== key);
    const otherTotal = remaining.reduce((sum, segment) => sum + norm[segment.key], 0);
    const next = { ...norm, [key]: value };
    remaining.forEach((segment) => { next[segment.key] = (1 - value) * (otherTotal ? norm[segment.key] / otherTotal : 0.5); });
    props.onChange(next);
  };

  return (
    <div className="weight-layout">
      <div style={{ position: 'relative', width: size, height: size }}>
        <svg width={size} height={size} style={{ transform: 'rotate(-90deg)' }}>
          <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="var(--bg-hover)" strokeWidth={stroke} />
          {arcs.map((arc, i) => (
            <circle
              key={i}
              cx={size / 2} cy={size / 2} r={r} fill="none"
              stroke={SEGMENTS[i].color}
              strokeWidth={stroke}
              strokeDasharray={`${arc.dash} ${circumference - arc.dash}`}
              strokeDashoffset={-arc.offset}
              strokeLinecap="butt"
              style={{ transition: 'stroke-dasharray .4s ease, stroke-dashoffset .4s ease', filter: 'drop-shadow(0 0 6px currentColor)' }}
            />
          ))}
        </svg>
        <div style={{
          position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column',
          alignItems: 'center', justifyContent: 'center', gap: 2,
        }}>
          <span className="mono" style={{ fontSize: size * 0.15, fontWeight: 700 }}>
            {Math.round(norm.activity * 100)}%
          </span>
          <span className="dim" style={{ fontSize: 9 }}>{lang === 'zh' ? '活性' : 'Activity'}</span>
        </div>
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 6, width: '100%' }}>
        {SEGMENTS.map((seg) => (
          <div key={seg.key} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span style={{ width: 10, height: 10, borderRadius: 3, background: seg.color, flexShrink: 0 }} />
            <span className="small dim" style={{ flex: 1 }}>{t(seg.labelKey, lang)}</span>
            <span className="mono small">{Math.round(norm[seg.key] * 100)}%</span>
            {props.adjust && (
              <input
                aria-label={t(seg.labelKey, lang)}
                type="range" min={0} max={100}
                value={Math.round(norm[seg.key] * 100)}
                onChange={(e) => setRaw(seg.key, Number(e.target.value) / 100)}
                style={{ width: 70, accentColor: seg.color }}
              />
            )}
          </div>
        ))}
      </div>
      <div style={{ display: 'flex', gap: 6, gridColumn: '1 / -1' }}>
        {props.applied && <Badge kind="accent">{t('weights.applied', lang)}</Badge>}
        {props.adjust && <Badge kind="warn">{t('weights.adjust', lang)}</Badge>}
      </div>
    </div>
  );
}

/** Lazy Plotly view. Resolve CSS tokens before handing values to Plotly. */
import { useEffect, useRef, useState } from 'react';
import { useApp } from '../../app/store';

function resolveTokens(value: unknown, style: CSSStyleDeclaration): unknown {
  if (typeof value === 'string' && value.startsWith('var(--')) return style.getPropertyValue(value.slice(4, -1)).trim();
  if (Array.isArray(value)) return value.map((item) => resolveTokens(item, style));
  if (value && typeof value === 'object') return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, resolveTokens(item, style)]));
  return value;
}

export function PlotWrap(props: {
  data: unknown[];
  layout: Record<string, unknown>;
  className?: string;
  style?: React.CSSProperties;
  onPointClick?: (point: unknown) => void;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const theme = useApp((s) => s.theme);
  const lang = useApp((s) => s.lang);
  const [error, setError] = useState(false);
  const clickRef = useRef(props.onPointClick);
  clickRef.current = props.onPointClick;
  useEffect(() => {
    const element = ref.current;
    if (!element) return;
    let disposed = false;
    let loaded: typeof import('plotly.js-dist-min').default | null = null;
    setError(false);
    const style = getComputedStyle(document.documentElement);
    const layout = {
      paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
      font: { color: style.getPropertyValue('--text-dim').trim(), size: 11 },
      ...props.layout,
      margin: { l: 48, r: 16, t: 34, b: 40, ...(props.layout.margin as Record<string, number> | undefined) },
    };
    import('plotly.js-dist-min').then(async ({ default: Plotly }) => {
      if (disposed) return;
      loaded = Plotly;
      await Plotly.newPlot(element, resolveTokens(props.data, style) as unknown[], resolveTokens(layout, style) as Record<string, unknown>, { responsive: true, displaylogo: false });
      if (disposed) return;
      const plotElement = element as HTMLDivElement & { on?: (event: string, callback: (event: unknown) => void) => void };
      plotElement.on?.('plotly_click', (event: unknown) => clickRef.current?.(event));
    }).catch(() => { if (!disposed) setError(true); });
    return () => { disposed = true; loaded?.purge(element); };
  }, [props.data, props.layout, theme]);
  return <div><div ref={ref} className={props.className} style={{ width: '100%', height: 260, ...props.style }} />{error && <div className="error-text small" role="alert">{lang === 'zh' ? '图表加载失败，请刷新页面重试。' : 'The chart could not be loaded. Refresh to try again.'}</div>}</div>;
}

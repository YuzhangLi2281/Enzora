/** All five scientific charts + the deep-space "mutation universe" hero map.
 *  Data comes from the ranked/selected candidate arrays — the same numbers the
 *  backend writes into the CSV artifacts. Charts are client-rendered, so the
 *  backend PNGs remain available as downloads via /figures. */

import { useEffect, useId, useMemo, useRef, useState } from "react";
import type { Candidate } from "../../shared/api/contracts";
import { t } from "../../shared/lib/i18n";
import { useApp } from "../../app/store";
import { PlotWrap } from "../../shared/ui/PlotWrap";

const MAP_COLORSCALE: Array<[number, string]> = [
  [0, "#4361EE"],
  [0.25, "#4CC9F0"],
  [0.5, "#2DE0C7"],
  [0.75, "#87C9EB"],
  [1, "#D4EEFA"],
];

export function useThemePlotly() {
  return {
    paper: "rgba(0,0,0,0)",
    plot: "rgba(0,0,0,0)",
    grid: "var(--border)",
    text: "var(--text-dim)",
    textMain: "var(--text)",
  };
}

export function StarMapHero(props: {
  candidates: Candidate[];
  selected?: Candidate[];
  onPick?: (mutation: string) => void;
}) {
  const lang = useApp((s) => s.lang);
  const { grid, text } = useThemePlotly();
  const data = useMemo(() => {
    const rows = props.candidates.filter(
      (c) => c.activity_log10 != null && c.ddg != null,
    );
    const maxScore = Math.max(
      ...rows.map((c) => Number(c.final_score ?? 0)),
      1e-6,
    );
    const sizes = rows.map(
      (c) => 10 + 30 * (Number(c.final_score ?? 0) / maxScore),
    );
    const selectedSet = new Set((props.selected ?? []).map((c) => c.mutation));
    return [
      {
        type: "scatter",
        mode: "markers",
        x: rows.map((c) => Number(c.activity_log10)),
        y: rows.map((c) => Number(c.ddg)),
        customdata: rows.map((c) => c.mutation),
        marker: {
          size: sizes,
          color: rows.map((c) => Number(c.ph_component ?? 0)),
          colorscale: MAP_COLORSCALE,
          cmin: 0,
          cmax: 1,
          opacity: 0.85,
          line: {
            width: rows.map((c) => (selectedSet.has(c.mutation) ? 2 : 0)),
            color: rows.map((c) =>
              c.passed === false ? "#F87171" : "#FFFFFF",
            ),
          },
        },
        hovertemplate: rows.map(
          (c) =>
            `<b>${c.mutation}</b><br>activity ${Number(c.activity_log10).toFixed(3)}<br>ddG ${Number(c.ddg).toFixed(3)}<br>pH comp ${Number(c.ph_component ?? 0).toFixed(2)}<extra></extra>`,
        ),
        text: rows.map((c) => c.mutation),
      } as unknown,
    ];
  }, [props.candidates, props.selected]);

  const layout = useMemo(
    () => ({
      title: { text: t("charts.hero", lang), font: { size: 13 } },
      xaxis: {
        title: "PHENOM activity_log10",
        gridcolor: grid,
        zerolinecolor: grid,
      },
      yaxis: { title: "UniStab ddG", gridcolor: grid, zerolinecolor: grid },
      hoverlabel: {
        bgcolor: "var(--bg-panel-solid)",
        bordercolor: "var(--border)",
        font: { color: "var(--text)" },
      },
      legend: { orientation: "h" },
    }),
    [lang, grid],
  );

  const onClick = (evt: unknown) => {
    const point = (evt as { points?: { customdata?: string }[] }).points?.[0];
    if (point?.customdata && props.onPick) props.onPick(point.customdata);
  };

  return (
    <PlotWrap
      data={data}
      layout={layout}
      style={{ height: 320 }}
      onPointClick={onClick}
    />
  );
}

export function FinalScoresChart(props: { candidates: Candidate[] }) {
  const lang = useApp((s) => s.lang);
  const { grid } = useThemePlotly();
  const rows = [...props.candidates].sort(
    (a, b) => Number(a.final_score ?? 0) - Number(b.final_score ?? 0),
  );
  const data = [
    {
      type: "bar",
      orientation: "h",
      x: rows.map((c) => Number(c.final_score ?? 0)),
      y: rows.map((c) => c.mutation),
      marker: { color: "var(--accent)" },
    } as unknown,
  ];
  const layout = {
    title: { text: t("charts.scores", lang), font: { size: 13 } },
    xaxis: { gridcolor: grid },
    yaxis: { automargin: true },
  };
  return <PlotWrap data={data} layout={layout} />;
}

export function ActivityStabilityChart(props: {
  candidates: Candidate[];
  selected?: Candidate[];
}) {
  const lang = useApp((s) => s.lang);
  const { grid } = useThemePlotly();
  const data = [
    {
      type: "scatter",
      mode: "markers",
      x: props.candidates.map((c) => Number(c.activity_log10)),
      y: props.candidates.map((c) => Number(c.ddg)),
      text: props.candidates.map((c) => c.mutation),
      marker: { color: "var(--accent)", size: 9, opacity: 0.55 },
      name: "pool",
    } as unknown,
    {
      type: "scatter",
      mode: "markers+text",
      x: (props.selected ?? []).map((c) => Number(c.activity_log10)),
      y: (props.selected ?? []).map((c) => Number(c.ddg)),
      text: (props.selected ?? []).map((c) => c.mutation),
      textposition: "top right",
      marker: { color: "var(--accent2)", size: 11, symbol: "x" },
      name: "selected",
    } as unknown,
  ];
  const layout = {
    title: { text: t("charts.scatter", lang), font: { size: 13 } },
    xaxis: { title: "activity_log10", gridcolor: grid, zerolinecolor: grid },
    yaxis: { title: "ddG", gridcolor: grid, zerolinecolor: grid },
  };
  return <PlotWrap data={data} layout={layout} />;
}

export function PropertyHeatmap(props: { candidates: Candidate[] }) {
  const lang = useApp((s) => s.lang);
  const rows = props.candidates.slice(0, 15);
  const cols = ["activity_log10", "ph_opt", "ddg", "final_score"] as const;
  const normalizedColumns = cols.map((col) => {
    const values = rows.map((c) => Number(c[col] ?? 0));
    const mean = values.reduce((a, b) => a + b, 0) / (values.length || 1);
    const std =
      Math.sqrt(
        values.reduce((a, b) => a + (b - mean) ** 2, 0) / (values.length || 1),
      ) || 1;
    return values.map((v) => (v - mean) / std);
  });
  const matrix = rows.map((_, index) =>
    normalizedColumns.map((column) => column[index]),
  );
  const data = [
    {
      type: "heatmap",
      z: matrix,
      x: [...cols],
      y: rows.map((c) => c.mutation),
      colorscale: "RdBu",
      reversescale: true,
      colorbar: { title: "z" },
    } as unknown,
  ];
  const layout = {
    title: { text: t("charts.heatmap", lang), font: { size: 13 } },
    yaxis: { automargin: true },
  };
  return <PlotWrap data={data} layout={layout} />;
}

export function SequenceMapChart(props: {
  candidates: Candidate[];
  sequenceLength: number;
}) {
  const lang = useApp((s) => s.lang);
  const { grid } = useThemePlotly();
  const rows = props.candidates.filter((c) => c.position != null);
  const data = [
    {
      type: "scatter",
      mode: "markers+text",
      x: rows.map((c) => Number(c.position)),
      y: rows.map((c) => Number(c.final_score ?? 0)),
      text: rows.map((c) => c.mutation),
      textposition: "top center",
      marker: { size: 10, color: "var(--accent2)" },
    } as unknown,
  ];
  const layout = {
    title: { text: t("charts.seqmap", lang), font: { size: 13 } },
    xaxis: {
      title: "position",
      range: [0, props.sequenceLength + 1],
      gridcolor: grid,
      zerolinecolor: grid,
    },
    yaxis: { title: "final_score", gridcolor: grid },
    shapes: [
      {
        type: "line",
        x0: 1,
        x1: props.sequenceLength,
        y0: 0,
        y1: 0,
        line: { color: "var(--text-dim)", width: 2 },
      },
    ],
  };
  return <PlotWrap data={data} layout={layout} />;
}

export function ParetoChart(props: {
  candidates: Candidate[];
  selected?: Candidate[];
}) {
  const lang = useApp((s) => s.lang);
  const { grid } = useThemePlotly();
  const rows = props.candidates.filter((c) => c.activity_component != null);
  const pareto = rows.filter((c) => String(c.pareto_optimal) === "true");
  const data = [
    {
      type: "scatter",
      mode: "markers",
      x: rows.map((c) => Number(c.activity_component)),
      y: rows.map((c) => Number(c.stability_component)),
      text: rows.map((c) => c.mutation),
      marker: {
        size: rows.map((c) => 8 + 12 * Number(c.ph_component ?? 0)),
        color: "var(--accent)",
        opacity: 0.45,
      },
      name: "pool (size=pH)",
    } as unknown,
    {
      type: "scatter",
      mode: "markers",
      x: pareto.map((c) => Number(c.activity_component)),
      y: pareto.map((c) => Number(c.stability_component)),
      text: pareto.map((c) => c.mutation),
      marker: {
        size: pareto.map((c) => 13 + 12 * Number(c.ph_component ?? 0)),
        color: "rgba(0,0,0,0)",
        line: { color: "var(--ok)", width: 2 },
      },
      name: "Pareto optimal",
    } as unknown,
  ];
  const layout = {
    title: { text: t("charts.pareto", lang), font: { size: 13 } },
    xaxis: {
      title: "activity component",
      range: [-0.03, 1.03],
      gridcolor: grid,
      zerolinecolor: grid,
    },
    yaxis: {
      title: "stability component",
      range: [-0.03, 1.03],
      gridcolor: grid,
      zerolinecolor: grid,
    },
  };
  return <PlotWrap data={data} layout={layout} />;
}

export function ChartGallery(props: {
  candidates: Candidate[];
  selected?: Candidate[];
  sequenceLength: number;
  active: string;
}) {
  const { candidates, selected, sequenceLength, active } = props;
  const setSelectedMutation = useApp((state) => state.setSelectedMutation);
  const selectedNames = new Set(selected?.map((row) => row.mutation));
  const currentSelected = selected
    ? candidates.filter((row) => selectedNames.has(row.mutation))
    : candidates;
  switch (active) {
    case "hero":
      return (
        <StarMapHero
          candidates={candidates}
          selected={currentSelected}
          onPick={setSelectedMutation}
        />
      );
    case "scores":
      return <FinalScoresChart candidates={currentSelected} />;
    case "scatter":
      return (
        <ActivityStabilityChart
          candidates={candidates}
          selected={currentSelected}
        />
      );
    case "heatmap":
      return <PropertyHeatmap candidates={currentSelected} />;
    case "seqmap":
      return (
        <SequenceMapChart
          candidates={currentSelected}
          sequenceLength={sequenceLength}
        />
      );
    case "pareto":
      return <ParetoChart candidates={candidates} selected={selected} />;
    default:
      return null;
  }
}

export type AtlasData = {
  method: string;
  points: {
    mutation: string;
    x: number;
    y: number;
    rank: number | null;
    score: number | null;
  }[];
};
type AtlasView = { x: number; y: number; k: number };

/** Keep the data cloud within reach, including when zooming around the pointer. */
function constrainAtlas(view: AtlasView): AtlasView {
  const k = Math.max(0.8, Math.min(8, view.k));
  const limit = Math.max(60, 245 * k - 150);
  const distance = Math.hypot(view.x, view.y);
  const ratio = distance > limit ? limit / distance : 1;
  return { x: view.x * ratio, y: view.y * ratio, k };
}
function zoomAtlas(
  view: AtlasView,
  factor: number,
  anchor = { x: 0, y: 0 },
): AtlasView {
  const k = Math.max(0.8, Math.min(8, view.k * factor));
  return constrainAtlas({
    k,
    x: anchor.x - ((anchor.x - view.x) * k) / view.k,
    y: anchor.y - ((anchor.y - view.y) * k) / view.k,
  });
}

/** The camera moves; the measured coordinates and sequence count never change. */
export function MutationAtlas({
  data,
  onPick,
  picked,
  zh,
}: {
  data: AtlasData | null;
  onPick: (m: string) => void;
  picked: string | null;
  zh: boolean;
}) {
  const macaron = useApp((state) => state.theme === "macaron");
  const clip = useId().replace(/:/g, "");
  const [view, setView] = useState<AtlasView>({ x: 0, y: 0, k: 1 });
  const [hover, setHover] = useState<AtlasData["points"][number] | null>(null);
  const [dragging, setDragging] = useState(false);
  const surface = useRef<SVGSVGElement>(null);
  const animation = useRef(0);
  const drag = useRef<{
    x: number;
    y: number;
    vx: number;
    vy: number;
    previousX: number;
    previousY: number;
    time: number;
    speedX: number;
    speedY: number;
    moved: boolean;
    mutation: string | null;
  } | null>(null);
  useEffect(() => {
    const svg = surface.current!;
    const wheel = (event: WheelEvent) => {
      if (!data?.points.length) return;
      event.preventDefault();
      cancelAnimationFrame(animation.current);
      const box = svg.getBoundingClientRect(),
        scale = 600 / Math.min(box.width, box.height);
      const anchor = {
        x: (event.clientX - box.left - box.width / 2) * scale,
        y: (event.clientY - box.top - box.height / 2) * scale,
      };
      setView((v) =>
        zoomAtlas(
          v,
          Math.exp(-Math.max(-120, Math.min(120, event.deltaY)) * 0.003),
          anchor,
        ),
      );
    };
    svg.addEventListener("wheel", wheel, { passive: false });
    return () => {
      svg.removeEventListener("wheel", wheel);
      cancelAnimationFrame(animation.current);
    };
  }, [data?.points.length]);
  useEffect(() => {
    setView({ x: 0, y: 0, k: 1 });
    setHover(null);
  }, [data]);
  const points = useMemo(() => {
    if (!data?.points.length) return [];
    const xs = data.points.map((p) => p.x),
      ys = data.points.map((p) => p.y);
    const cx = (Math.max(...xs) + Math.min(...xs)) / 2,
      cy = (Math.max(...ys) + Math.min(...ys)) / 2;
    const radius = Math.max(
      ...data.points.map((p) => Math.hypot(p.x - cx, p.y - cy)),
    );
    return data.points.map((p) => ({
      ...p,
      px: 300 + ((p.x - cx) / (radius || 1)) * 245,
      py: 300 + ((p.y - cy) / (radius || 1)) * 245,
    }));
  }, [data]);
  // A soft density wash derived only from actual visible sequence positions.
  const density = useMemo(() => {
    const cells = new Map<string, { x: number; y: number; count: number }>();
    for (const p of points) {
      const x = Math.floor(p.px / 16) * 16,
        y = Math.floor(p.py / 16) * 16,
        key = `${x}:${y}`;
      const cell = cells.get(key);
      if (cell) cell.count++;
      else cells.set(key, { x, y, count: 1 });
    }
    const peak = Math.max(
      1,
      ...Array.from(cells.values(), (cell) => cell.count),
    );
    return Array.from(cells.values())
      .filter((cell) => cell.count > 1)
      .map((cell) => ({ ...cell, intensity: cell.count / peak }));
  }, [points]);
  // Keep the 1,000 glyphs stable while only the camera or tooltip changes.
  const dots = useMemo(
    () =>
      points.map((p) => (
        <circle
          key={p.mutation}
          data-point={p.mutation}
          cx={p.px}
          cy={p.py}
          r={p.mutation === picked ? 4.5 : p.rank ? 3.1 : 1.55}
          fill={
            p.rank
              ? "var(--atlas-selected)"
              : p.score === null
                ? "var(--atlas-unknown)"
                : macaron
                  ? `color-mix(in srgb, ${p.score < 0.5 ? "#8AC3C1" : "#B497C7"} ${(1 - (p.score < 0.5 ? p.score * 2 : (p.score - 0.5) * 2)) * 100}%, ${p.score < 0.5 ? "#B497C7" : "#F1C8DB"})`
                  : `hsl(${225 - p.score * 35} ${78 - p.score * 24}% ${43 + p.score * 24}%)`
          }
          opacity={p.rank || p.mutation === picked ? 1 : 0.9}
          onPointerEnter={() => {
            if (!drag.current) setHover(p);
          }}
          onPointerLeave={() => setHover(null)}
        >
          <title>
            {p.mutation}
            {p.score !== null ? ` · ${p.score.toFixed(3)}` : ""}
          </title>
        </circle>
      )),
    [points, picked, macaron],
  );
  const release = (cancelled = false) => {
    const d = drag.current;
    drag.current = null;
    setDragging(false);
    if (!d || cancelled) return;
    if (!d.moved) {
      if (d.mutation) onPick(d.mutation);
      return;
    }
    if (
      matchMedia("(prefers-reduced-motion: reduce)").matches ||
      performance.now() - d.time > 80
    )
      return;
    let vx = Math.max(-18, Math.min(18, d.speedX * 16)),
      vy = Math.max(-18, Math.min(18, d.speedY * 16));
    const coast = () => {
      vx *= 0.86;
      vy *= 0.86;
      setView((v) => constrainAtlas({ ...v, x: v.x + vx, y: v.y + vy }));
      if (Math.hypot(vx, vy) > 0.15)
        animation.current = requestAnimationFrame(coast);
    };
    animation.current = requestAnimationFrame(coast);
  };
  return (
    <div
      className={`mutation-atlas ${dragging ? "is-dragging" : ""} ${points.length ? "" : "is-empty"}`}
    >
      <svg
        ref={surface}
        viewBox="0 0 600 600"
        tabIndex={points.length ? 0 : -1}
        role="img"
        aria-label={
          zh
            ? "序列嵌入图：拖拽平移，滚轮缩放，方向键移动，0复位"
            : "Sequence atlas: drag to pan, scroll to zoom, arrow keys to move, 0 to reset"
        }
        onKeyDown={(event) => {
          const shifts: Record<string, [number, number]> = {
            ArrowLeft: [25, 0],
            ArrowRight: [-25, 0],
            ArrowUp: [0, 25],
            ArrowDown: [0, -25],
          };
          const shift = shifts[event.key];
          if (shift) {
            event.preventDefault();
            setView((v) =>
              constrainAtlas({ ...v, x: v.x + shift[0], y: v.y + shift[1] }),
            );
          } else if (event.key === "0") {
            event.preventDefault();
            setView({ x: 0, y: 0, k: 1 });
          } else if (event.key === "+" || event.key === "-") {
            event.preventDefault();
            setView((v) => zoomAtlas(v, event.key === "+" ? 1.2 : 1 / 1.2));
          }
        }}
        onPointerDown={(event) => {
          if (event.button !== 0 || !points.length) return;
          cancelAnimationFrame(animation.current);
          setHover(null);
          setDragging(true);
          drag.current = {
            x: event.clientX,
            y: event.clientY,
            vx: view.x,
            vy: view.y,
            previousX: event.clientX,
            previousY: event.clientY,
            time: performance.now(),
            speedX: 0,
            speedY: 0,
            moved: false,
            mutation:
              (event.target as Element)
                .closest("[data-point]")
                ?.getAttribute("data-point") ?? null,
          };
          event.currentTarget.setPointerCapture(event.pointerId);
        }}
        onPointerMove={(event) => {
          const d = drag.current;
          if (!d) return;
          const box = event.currentTarget.getBoundingClientRect(),
            scale = 600 / Math.min(box.width, box.height),
            now = performance.now(),
            dt = Math.max(1, now - d.time);
          d.speedX = ((event.clientX - d.previousX) * scale) / dt;
          d.speedY = ((event.clientY - d.previousY) * scale) / dt;
          d.time = now;
          d.previousX = event.clientX;
          d.previousY = event.clientY;
          if (Math.hypot(event.clientX - d.x, event.clientY - d.y) > 4)
            d.moved = true;
          if (d.moved)
            setView((v) =>
              constrainAtlas({
                ...v,
                x: d.vx + (event.clientX - d.x) * scale,
                y: d.vy + (event.clientY - d.y) * scale,
              }),
            );
        }}
        onPointerUp={() => release()}
        onPointerCancel={() => release(true)}
      >
        <defs>
          <clipPath id={clip}>
            <circle cx="300" cy="300" r="284" />
          </clipPath>
          <filter
            id={`${clip}-wash`}
            x="-60%"
            y="-60%"
            width="220%"
            height="220%"
          >
            <feGaussianBlur stdDeviation="11" />
          </filter>
          <linearGradient id={`${clip}-ground`} x1="0" y1="0" x2="0" y2="1">
            <stop stopColor="var(--atlas-center)" />
            <stop offset=".55" stopColor="var(--atlas-middle)" />
            <stop offset="1" stopColor="var(--atlas-edge)" />
          </linearGradient>
        </defs>
        <circle
          cx="300"
          cy="300"
          r="284"
          fill={`url(#${clip}-ground)`}
          className="atlas-ground"
        />
        <g clipPath={`url(#${clip})`}>
          <g
            data-atlas-camera
            transform={`translate(${300 + view.x} ${300 + view.y}) scale(${view.k}) translate(-300 -300)`}
          >
            <g filter={`url(#${clip}-wash)`} opacity=".38" pointerEvents="none">
              {density.map((cell) => (
                <rect
                  key={`${cell.x}:${cell.y}`}
                  x={cell.x - 6}
                  y={cell.y - 6}
                  width="28"
                  height="28"
                  rx="14"
                  fill={
                    cell.intensity > 0.5
                      ? "var(--density-high)"
                      : "var(--density-low)"
                  }
                  opacity={0.15 + cell.intensity * 0.65}
                />
              ))}
            </g>
            {dots}
          </g>
          {points
            .filter((p) => p.rank && p.rank <= 5)
            .map((p) => {
              const x = 300 + view.x + (p.px - 300) * view.k,
                y = 300 + view.y + (p.py - 300) * view.k,
                labelY = 145 + (p.rank! - 1) * 70;
              return (
                <g key={p.rank} className="atlas-label" data-point={p.mutation}>
                  <path
                    d={`M488 ${labelY} L460 ${labelY} L${x} ${y}`}
                    fill="none"
                    stroke="var(--lime)"
                    strokeWidth=".8"
                    opacity=".8"
                  />
                  <circle
                    cx={x}
                    cy={y}
                    r="6"
                    fill="none"
                    stroke="var(--lime)"
                  />
                  <path
                    d={`M${x - 3} ${y - 8} L${x} ${y} L${x + 7} ${y - 4}`}
                    fill="none"
                    stroke="var(--lime)"
                  />
                  <text x="491" y={labelY - 6}>
                    Top {p.rank}
                  </text>
                  <text x="491" y={labelY + 12} className="atlas-mutation">
                    {p.mutation}
                  </text>
                </g>
              );
            })}
        </g>
        {!points.length && (
          <>
            <text x="300" y="292" textAnchor="middle" className="atlas-empty">
              {zh ? "等待全库序列嵌入" : "Awaiting the library embedding"}
            </text>
            <text x="300" y="320" textAnchor="middle" className="atlas-caption">
              {zh
                ? "完成评分，再选择 Top-K 并生成报告"
                : "Complete scoring, then select Top-K and generate the report."}
            </text>
          </>
        )}
      </svg>
      {hover && !dragging && (
        <div className="atlas-tooltip" role="status">
          <strong>{hover.mutation}</strong>
          <span>
            {hover.score !== null
              ? `${zh ? "综合分" : "Score"} ${hover.score.toFixed(3)}`
              : "—"}
          </span>
        </div>
      )}
      <div className="atlas-controls">
        <button
          aria-label={zh ? "缩小" : "Zoom out"}
          disabled={!points.length}
          onClick={() => setView((v) => zoomAtlas(v, 0.8))}
        >
          −
        </button>
        <button
          aria-label={zh ? "重置视图" : "Reset view"}
          disabled={!points.length}
          onClick={() => {
            cancelAnimationFrame(animation.current);
            setView({ x: 0, y: 0, k: 1 });
          }}
        >
          {Math.round(view.k * 100)}%
        </button>
        <button
          aria-label={zh ? "放大" : "Zoom in"}
          disabled={!points.length}
          onClick={() => setView((v) => zoomAtlas(v, 1.25))}
        >
          +
        </button>
      </div>
      <div className="atlas-color-key">
        <span>{zh ? "综合分" : "Score"}</span>
        <i />
        <span>0 — 1</span>
      </div>
      <p className="atlas-method">
        {data
          ? `${data.method} · ${points.length.toLocaleString()} ${zh ? "条序列" : "sequences"}`
          : zh
            ? "UMAP · 位点氨基酸身份 · Hamming 距离"
            : "UMAP · Positional identity · Hamming distance"}
      </p>
    </div>
  );
}

export function ScoreDistribution({
  rows,
  field,
  color,
  zh,
}: {
  rows: Candidate[];
  field: keyof Candidate;
  color: string;
  zh: boolean;
}) {
  const id = useId().replace(/:/g, "");
  const values = rows
    .map((r) => r[field])
    .filter((v): v is number => typeof v === "number" && Number.isFinite(v));
  const lo = values.length ? Math.min(...values) : 0;
  const hi = values.length ? Math.max(...values) : 1;
  const min = lo === hi ? lo - 0.5 : lo;
  const max = lo === hi ? hi + 0.5 : hi;
  const bins = Array<number>(30).fill(0);
  values.forEach(
    (v) => bins[Math.min(29, Math.floor(((v - min) / (max - min)) * 30))]++,
  );
  const peak = Math.max(...bins, 1);
  const line = bins
    .map((n, i) => `${i ? "L" : "M"}${25 + i * 11},${77 - (n / peak) * 61}`)
    .join(" ");
  return (
    <svg
      className="score-distribution"
      viewBox="0 0 360 108"
      role="img"
      aria-label={`${field}: ${values.length} ${zh ? "条预测的直方图" : "predictions, histogram"}`}
    >
      <defs>
        <linearGradient id={`${id}-bars`} x1="0" y1="0" x2="0" y2="1">
          <stop stopColor={color} />
          <stop offset="1" stopColor={color} stopOpacity=".25" />
        </linearGradient>
      </defs>
      {[16, 46, 77].map((y) => (
        <path key={y} d={`M20 ${y}H354`} className="distribution-grid" />
      ))}
      {values.length ? (
        <>
          {bins.map((n, i) => (
            <rect
              key={i}
              x={21 + i * 11}
              y={77 - (n / peak) * 61}
              width="8"
              height={(n / peak) * 61}
              rx="2"
              fill={`url(#${id}-bars)`}
            >
              <title>{`${(min + (i / 30) * (max - min)).toFixed(2)} – ${(min + ((i + 1) / 30) * (max - min)).toFixed(2)}: ${n}`}</title>
            </rect>
          ))}
          <path
            d={line}
            fill="none"
            stroke={color}
            strokeWidth="2"
            strokeLinejoin="round"
          />
          <text x="14" y="19" textAnchor="end" className="distribution-tick">
            {peak}
          </text>
          {[min, (min + max) / 2, max].map((value, i) => (
            <text
              key={i}
              x={20 + i * 165}
              y="98"
              textAnchor={i === 0 ? "start" : i === 2 ? "end" : "middle"}
              className="distribution-tick"
            >
              {value.toFixed(2)}
            </text>
          ))}
        </>
      ) : (
        <text x="20" y="49" className="distribution-empty">
          {zh ? "启动筛选后显示预测分布" : "Predictions appear after screening"}
        </text>
      )}
    </svg>
  );
}

/** Replay real normalized components; time is presentation, never inference speed. */
export function ScoringWaterfall({ rows, weights, confidence, zh, active }: {
  rows: Candidate[]; weights: {activity:number;ph:number;stability:number};
  confidence: {activity:number;ph:number;stability:number}; zh:boolean; active:boolean;
}) {
  const chart=useRef<HTMLDivElement>(null);
  const [expanded, setExpanded] = useState(false);
  useEffect(() => {
    const update = () => setExpanded(document.fullscreenElement === chart.current);
    document.addEventListener("fullscreenchange", update);
    return () => document.removeEventListener("fullscreenchange", update);
  }, []);
  const id=useId().replace(/:/g,'');
  const models=[{key:'activity',name:'Phenom',color:'var(--score-activity)'},{key:'ph',name:'EpHod',color:'var(--score-ph)'},{key:'stability',name:'UniStab',color:'var(--score-stability)'}] as const;
  const denominator=models.reduce((n,m)=>n+weights[m.key]*confidence[m.key],0);
  const shares=models.map(m=>denominator>0?weights[m.key]*confidence[m.key]/denominator:0);
  const values=rows.filter(c=>[c.activity_component,c.ph_component,c.stability_component].every(v=>typeof v==='number'&&Number.isFinite(v)));
  const samples=values.filter((_,i)=>i%Math.max(1,Math.ceil(values.length/36))===0).map(c=>({mutation:c.mutation,
    parts:models.map((m,i)=>shares[i]*Number(c[`${m.key}_component`]))}));
  const fingerprint=JSON.stringify([shares,samples]);
  const [progress,setProgress]=useState(1);
  useEffect(()=>{
    if(!active || !values.length) return;
    if(matchMedia('(prefers-reduced-motion: reduce)').matches){setProgress(1);return;}
    setProgress(0);
    let request=0; const start=performance.now();
    const tick=(now:number)=>{const p=Math.min(1,(now-start)/5000);setProgress(p);if(p<1)request=requestAnimationFrame(tick);};
    request=requestAnimationFrame(tick);
    return ()=>cancelAnimationFrame(request);
  },[fingerprint,active]);
  const p=Math.max(0,Math.min(1,(progress*5000-600)/3600));
  const y=(i:number)=>75+i/Math.max(1,samples.length-1)*220;
  const leading=samples[Math.min(samples.length-1,Math.floor(p*samples.length))];
  const mean=values.length?values.reduce((sum,c)=>sum+models.reduce((n,m,i)=>n+shares[i]*Number(c[`${m.key}_component`]),0),0)/values.length:null;
  return <div ref={chart} className="scoring-waterfall" data-replay={progress<1}>
    <div className="waterfall-heading"><strong>{zh?'各模型如何组成最终得分':'How the models contribute to the final score'}</strong><span>{values.length?`${values.length} ${zh?'条真实预测':'scored candidates'}`:zh?'等待评分分量':'Awaiting score components'}</span></div>
    <button type="button" className="chart-expand" disabled={!document.fullscreenEnabled} onClick={() => document.fullscreenElement ? document.exitFullscreen() : chart.current?.requestFullscreen()}>{expanded ? (zh ? '关闭整屏视图' : 'Close full screen') : (zh ? '整屏查看 ↗' : 'Expand chart ↗')}</button>
    <svg viewBox="0 0 520 350" role="img" aria-label={zh?'偏好乘可信度归一化后的加权贡献':'Weighted contributions after normalizing preference times confidence'}>
      <defs><clipPath id={`${id}-reveal`}><rect x="0" y="70" width="520" height={p*235}/></clipPath></defs>
      {models.map((m,i)=><g key={m.key}>
        <text x={65+i*190} y="18" textAnchor="middle">{m.name}</text>
        <text x={65+i*190} y="38" textAnchor="middle" fill={m.color}>{(shares[i]*100).toFixed(1)}%</text>
        <path d={`M${65+i*190},47 C${65+i*190},66 ${42+shares.slice(0,i).reduce((a,b)=>a+b,0)*430},56 ${42+shares.slice(0,i).reduce((a,b)=>a+b,0)*430},67`}
          stroke={m.color} strokeWidth={shares[i]>0?3+shares[i]*12:1} strokeDasharray={shares[i]===0?'3 4':undefined} opacity={shares[i]===0?.3:.8} fill="none"/>
      </g>)}
      {!samples.length ? <text x="260" y="185" textAnchor="middle">{zh?'评分完成后，这里显示各模型对候选得分的贡献':'After screening, compare each model’s share of the score'}</text> : <g clipPath={`url(#${id}-reveal)`}>
        {models.map((m,i)=>{
          const upper=samples.map((s,j)=>`${42+s.parts.slice(0,i+1).reduce((a,b)=>a+b,0)*430},${y(j)}`);
          const lower=[...samples].reverse().map((s,j)=>`${42+s.parts.slice(0,i).reduce((a,b)=>a+b,0)*430},${y(samples.length-1-j)}`);
          return <path key={m.key} d={`M${upper.join(' L')} L${lower.join(' L')}Z`} fill={m.color} fillOpacity=".34" stroke={m.color} strokeWidth="1.5" strokeLinejoin="round"/>;
        })}
      </g>}
      {leading && p>0 && <g><circle cx={42+leading.parts.reduce((a,b)=>a+b,0)*430} cy={75+p*220} r="4" fill="var(--text)"/><text x="478" y={Math.min(302,83+p*220)} textAnchor="end">Σ {leading.parts.reduce((a,b)=>a+b,0).toFixed(3)}</text></g>}
      <text x="42" y="334">0</text><text x="257" y="334" textAnchor="middle">0.5</text><text x="472" y="334" textAnchor="end">1.0</text>
    </svg>
    <div className="waterfall-summary"><span>Σ Wᵢ = {denominator>0?'1.000':'—'}</span><span>{zh?'平均综合分':'Mean composite'} <strong>{mean?.toFixed(3)??'—'}</strong></span></div>
    <p>{zh?'每一行是一条候选序列，色带越宽，表示该模型贡献的分数越多。动画回放已保存的预测数据。':'Each row is a candidate; wider bands mean a larger contribution to its score. The animation replays saved predictions.'}</p>
  </div>;
}

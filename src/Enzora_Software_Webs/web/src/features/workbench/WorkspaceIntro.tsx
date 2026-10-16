import { useEffect, useId, useRef, useState } from "react";
import dpoIcon from "../../../../../../resource/models/proteindpo.svg";
import enzymeIcon from "../../../../../../resource/models/enzgfm.svg";
import vesmIcon from "../../../../../../resource/models/vesm3.svg";

const generators = [
  {
    id: "protein_dpo",
    name: "ProteinDPO",
    icon: dpoIcon,
    description: {
      zh: "实验偏好对齐的 ESM-IF1 逆折叠模型",
      en: "Experiment-aligned ESM-IF1 inverse folding",
    },
    explanation: {
      zh: "以 ESM-IF1 为基础，将骨架作为条件、序列作为生成结果，通过实验稳定性偏好做直接偏好优化（DPO）。",
      en: "Uses a backbone-conditioned ESM-IF1 model and direct preference optimization (DPO) on experimental stability preferences.",
    },
    supplement: {
      zh: "对齐信号主要来自蛋白稳定性实验；稳定性改善不保证催化功能保留。当前建库同时使用蛋白序列与骨架。",
      en: "The alignment signal is experimental stability; improved stability does not guarantee preserved catalysis. This workflow supplies sequence and backbone.",
    },
    input: { zh: "当前：序列 + 骨架", en: "Input: sequence + backbone" },
    paper: "https://www.nature.com/articles/s41592-026-03137-3",
    code: "https://github.com/evo-design/protein-dpo",
    title:
      "Aligning protein-generative models to experimental fitness with ProteinDPO",
    journal: "Nature Methods · 2026",
  },
  {
    id: "enzgfm",
    name: "EnzGFM",
    icon: enzymeIcon,
    description: {
      zh: "经酶数据领域适配的专用基础模型",
      en: "Enzyme-adapted foundation model",
    },
    explanation: {
      zh: "采用 Mamba–Transformer 混合架构，先在通用蛋白序列上预训练，再用酶序列继续训练，学习酶领域的序列模式。",
      en: "A Mamba–Transformer hybrid with hierarchical pretraining: general protein sequences first, then enzyme-specific sequences.",
    },
    supplement: {
      zh: "在 Enzora 中仅用于突变库候选评分与构建；后续酶活性预测统一由 PheNom 执行。",
      en: "In Enzora it contributes to mutation-library construction. Subsequent enzyme activity prediction is performed by PheNom.",
    },
    input: { zh: "当前：蛋白序列", en: "Input: protein sequence" },
    paper: "https://www.nature.com/articles/s41467-026-75283-3",
    code: "https://github.com/DeepBxM/EnzGFM",
    title:
      "An enzyme-specific protein language model for catalytic property prediction",
    journal: "Nature Communications · 2026",
  },
  {
    id: "vesm3",
    name: "VESM3",
    icon: vesmIcon,
    description: {
      zh: "知识蒸馏微调的多模态 ESM3",
      en: "Knowledge-distilled multimodal ESM3",
    },
    explanation: {
      zh: "将 VESM-3B 的变异效应知识蒸馏进 ESM3 的序列模块，保留其预训练的结构等多模态通道；推理时可选结构输入。",
      en: "Distills VESM-3B variant-effect knowledge into ESM3’s sequence modules, retaining pretrained multimodal channels and optional structure input at inference.",
    },
    supplement: {
      zh: "蒸馏阶段只使用序列，不新增结构监督。VESM++ 才是 VESM3 与 VESM-3B 的组合。当前 Enzora 适配器仅传入序列，尚未启用结构 token。",
      en: "Distillation uses sequences only. VESM++ is the combination of VESM3 and VESM-3B. The current Enzora adapter supplies sequence only; structure tokens are not enabled.",
    },
    input: {
      zh: "当前：序列；支持结构输入",
      en: "Input: sequence; structure-capable",
    },
    paper: "https://www.nature.com/articles/s41592-026-03050-9",
    code: "https://github.com/ntranoslab/vesm",
    title:
      "Compressing the collective knowledge of ESM into a single protein language model",
    journal: "Nature Methods · 2026",
  },
] as const;

/** Saved-record progress; motion stops when the backend is waiting or idle. */
export function WorkspaceIntro({
  saved,
  target,
  running,
  rate,
  zh,
  model,
  contributions,
  landscape,
  runKey,
}: {
  saved: number;
  target: number;
  running: boolean;
  rate: number;
  zh: boolean;
  model?: string;
  contributions?: Record<string, number>;
  runKey?: string;
  landscape?: { bins: number[]; frames: {saved:number; counts:Record<string,number>; density:Record<string,number[]>}[] };
}) {
  const [compact, setCompact] = useState(
    () => matchMedia("(max-width:1000px)").matches,
  );
  useEffect(() => {
    const query = matchMedia("(max-width:1000px)");
    const update = () => setCompact(query.matches);
    query.addEventListener("change", update);
    return () => query.removeEventListener("change", update);
  }, []);
  const chart = useRef<HTMLDivElement>(null);
  const [expanded, setExpanded] = useState(false);
  useEffect(() => {
    const update = () => setExpanded(document.fullscreenElement === chart.current);
    document.addEventListener("fullscreenchange", update);
    return () => document.removeEventListener("fullscreenchange", update);
  }, []);
  const id = useId().replace(/:/g, "");
  const progress = target > 0 ? Math.min(1, saved / target) : 0;
  const element = useRef<HTMLDivElement>(null);
  const [visible, setVisible] = useState(false);
  const [replay, setReplay] = useState(1);
  const replayed = useRef('');
  const frameCount = landscape?.frames.length ?? 0;
  useEffect(() => {
    const observer = new IntersectionObserver(([entry]) => setVisible(entry.isIntersecting), {threshold:.2});
    if (element.current) observer.observe(element.current);
    return () => observer.disconnect();
  }, []);
  useEffect(() => {
    if (!visible) {setReplay(1);return;}
    if (!frameCount || saved < target || replayed.current === runKey) return;
    replayed.current = runKey ?? '';
    if (matchMedia('(prefers-reduced-motion: reduce)').matches) { setReplay(1); return; }
    setReplay(0);
    const start = performance.now();
    let handle = 0;
    const tick = (now:number) => {
      const value = Math.min(1,(now-start)/5000);
      setReplay(value);
      if (value<1) handle=requestAnimationFrame(tick);
    };
    handle=requestAnimationFrame(tick);
    return () => cancelAnimationFrame(handle);
  }, [visible,frameCount,saved,target,runKey]);
  const frameIndex = Math.min(frameCount-1, Math.floor(Math.max(0,(replay*5000-800)/3400)*frameCount));
  const frame = landscape?.frames[Math.max(0,frameIndex)];
  const counts = frame?.counts ?? contributions;
  const attributionTotal = counts ? Object.values(counts).reduce((a,b)=>a+b,0) : 0;
  const densityMax = landscape ? Math.max(.001,...landscape.frames.flatMap(f=>Object.values(f.density).flat())) : 1;
  const colors = ['var(--score-activity)','var(--score-ph)','var(--score-stability)'];
  return (
    <div ref={element} className={`library-engine ${running ? "is-running" : ""}`}>
      <div className="engine-flow">
        <div className="model-nodes">
          {generators.map((entry, index) => (
            <details
              key={entry.id}
              name={`${id}-generator`}
              className="model-node"
              onKeyDown={(event) => {
                if (event.key === "Escape") {
                  event.currentTarget.open = false;
                  event.currentTarget.querySelector("summary")?.focus();
                }
              }}
              style={{
                top: compact
                  ? undefined
                  : `${([105, 235, 365][index] / 470) * 100}%`,
              }}
            >
              <summary>
                <img src={entry.icon} alt="" width="44" height="44" />
                <div>
                  <strong>{entry.name}</strong>
                  <span>
                    {zh ? entry.description.zh : entry.description.en}
                  </span>
                </div>
                <svg
                  className="model-chevron"
                  viewBox="0 0 16 16"
                  aria-hidden="true"
                >
                  <path
                    d="m4 6 4 4 4-4"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="1.5"
                    strokeLinecap="round"
                  />
                </svg>
              </summary>
              <div className="model-more">
                <span className="model-input-status">
                  {model === entry.id
                    ? zh
                      ? "正在计算"
                      : "Computing"
                    : zh
                      ? entry.input.zh
                      : entry.input.en}
                </span>
                <h3>{entry.name}</h3>
                <p>{zh ? entry.explanation.zh : entry.explanation.en}</p>
                <p>{zh ? entry.supplement.zh : entry.supplement.en}</p>
                <a
                  className="model-paper"
                  href={entry.paper}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <span>{entry.journal}</span>
                  <strong>{entry.title}</strong>
                  <span>{zh ? "阅读原文" : "Read paper"} ↗</span>
                </a>
                <a
                  className="model-code"
                  href={entry.code}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  {zh ? "官方代码与补充说明" : "Official code & documentation"}{" "}
                  ↗
                </a>
              </div>
            </details>
          ))}
        </div>
        <svg
          viewBox={compact ? "360 52 372 366" : "0 0 780 470"}
          role="img"
          aria-label={`${zh ? "已保存序列" : "Saved sequences"} ${saved}/${target}`}
        >
          <defs>
            <linearGradient id={id}>
              <stop stopColor="var(--ring-start)" />
              <stop offset=".55" stopColor="var(--ring-mid)" />
              <stop offset="1" stopColor="var(--ring-end)" />
            </linearGradient>
            <marker id={`${id}-flow-arrow`} viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M1 1 9 5 1 9" fill="none" stroke="var(--text-dim)" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"/></marker>
          </defs>
          {[105, 235, 365].map((y, i) => (
            <g key={y}>
              <path
                className="engine-wire" markerEnd={`url(#${id}-flow-arrow)`}
                d={`M270 ${y} C350 ${y}, 355 235, 418 235`}
              />
              {running && rate > 0 && (
                <path
                  className="engine-pulse"
                  style={{
                    animationDuration: `${Math.max(0.65, 4 / Math.log2(rate + 2))}s`,
                    animationDelay: `${i * -0.5}s`,
                  }}
                  d={`M270 ${y} C350 ${y}, 355 235, 418 235`}
                />
              )}
            </g>
          ))}
          <circle cx="570" cy="235" r="166" className="engine-outer" />
          <circle cx="570" cy="235" r="140" className="engine-track" />
          <circle
            cx="570"
            cy="235"
            r="140"
            fill="none"
            stroke={`url(#${id})`}
            strokeWidth="8"
            strokeLinecap="round"
            strokeDasharray={Math.PI * 280}
            strokeDashoffset={Math.PI * 280 * (1 - progress)}
            transform="rotate(-90 570 235)"
            className="engine-progress"
          />
          <text x="570" y="226" textAnchor="middle" className="engine-count">
            {saved.toLocaleString()}
          </text>
          <text x="570" y="260" textAnchor="middle" className="engine-label">
            {zh ? "条序列已保存" : "sequences saved"}
          </text>
          <text x="570" y="296" textAnchor="middle" className="engine-caption">
            {Math.round(progress * 100)}% · {target.toLocaleString()}{" "}
            {zh ? "目标" : "target"}
          </text>
        </svg>
      </div>
      <div ref={chart} className="model-contributions" data-replay={replay < 1}>
        <div><strong>{zh ? '酶突变库构建的 PLM LLR 分布' : 'PLM LLR score distribution for enzyme mutation database construction'}</strong>
          <span role="status">{!frame ? ''
            : replay<.16 ? 'Syncing with VESM…' : replay<.84 ? (zh?'扫描已保存替换':'Scanning saved substitutions') : `N=${saved} · ${zh?'归因完成':'Attribution complete'}`}</span></div>
        <button type="button" className="chart-expand" disabled={!document.fullscreenEnabled} onClick={() => document.fullscreenElement ? document.exitFullscreen() : chart.current?.requestFullscreen()}>{expanded ? (zh ? '关闭整屏视图' : 'Close full screen') : (zh ? '整屏查看 ↗' : 'Expand chart ↗')}</button>
        <div className="contribution-track">
          {counts && attributionTotal>0 && generators.map((g,i)=><span key={g.id} style={{width:`${counts[g.id]/attributionTotal*100}%`,background:colors[i]}} />)}
        </div>
        <div className="contribution-legend">{generators.map((g,i)=><span key={g.id}><i style={{background:colors[i]}}/>{g.name} <strong>{attributionTotal>0 ? `${(counts![g.id]/attributionTotal*100).toFixed(1)}%` : '—'}</strong></span>)}</div>
        {frame && landscape ? <svg className="llr-landscape" viewBox="0 0 560 220" role="img" aria-label={zh?'真实已保存替换的校准 LLR 分布':'Calibrated LLR distributions of saved substitutions'}>
          <defs>{generators.map((g,i)=><linearGradient key={g.id} id={`${id}-density-${i}`} x1="0" y1="0" x2="0" y2="1"><stop stopColor={colors[i]} stopOpacity=".55"/><stop offset="1" stopColor={colors[i]} stopOpacity=".04"/></linearGradient>)}</defs>
          {generators.map((g,i)=>{
            const values=frame.density[g.id];
            if(!values) return null;
            const points=values.map((v,j)=>`${32+j/(values.length-1)*500},${183-v/densityMax*145}`);
            return <g key={g.id} className="llr-density" style={{opacity:replay<.16?replay/.16:1}}>
              <path d={`M32,183 L${points.join(' L')} L532,183Z`} fill={`url(#${id}-density-${i})`}/>
              <path d={`M${points.join(' L')}`} fill="none" stroke={colors[i]} strokeWidth="2.2" strokeLinejoin="round"/>
            </g>;
          })}
          <text x="32" y="203">{landscape.bins[0].toFixed(1)}</text><text x="532" y="203" textAnchor="end">{landscape.bins[landscape.bins.length-1].toFixed(1)}</text>
          <text x="280" y="217" textAnchor="middle">{zh?'校准 LLR · 已保存替换的前缀回放':'Calibrated LLR · saved-substitution replay'}</text>
        </svg> : <div className="llr-waiting">{zh ? "建库后，这里显示三个模型的 LLR 分布与限制性归因占比。" : "Build the library to compare the three models’ LLR distributions and limiting-model shares."}</div>}
        <details className="llr-method"><summary>{zh?'如何读图':'How to read this plot'}</summary><p>{zh?'各替换取最低校准 LLR，平局均分。5 秒回放展示已保存数据，不代表推理耗时。':'Minimum calibrated LLR per substitution; ties split equally. The 5-second replay shows saved data, not inference time.'}</p></details>
      </div>
    </div>
  );
}

import { useEffect, useId, useState } from "react";
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
}: {
  saved: number;
  target: number;
  running: boolean;
  rate: number;
  zh: boolean;
  model?: string;
  contributions?: Record<string, number>;
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
  const id = useId().replace(/:/g, "");
  const progress = target > 0 ? Math.min(1, saved / target) : 0;
  const [history, setHistory] = useState<
    { saved: number; counts: Record<string, number> }[]
  >([]);
  useEffect(() => {
    setHistory((previous) => {
      if (!contributions) return [];
      const last = previous[previous.length - 1];
      if (last?.saved === saved) return previous;
      const sample = { saved, counts: { ...contributions } };
      return last && last.saved < saved
        ? [...previous, sample].slice(-32)
        : [sample];
    });
  }, [contributions, saved]);
  const attributionTotal = contributions
    ? Object.values(contributions).reduce((a, b) => a + b, 0)
    : 0;
  return (
    <div className={`library-engine ${running ? "is-running" : ""}`}>
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
          </defs>
          {[105, 235, 365].map((y, i) => (
            <g key={y}>
              <path
                className="engine-wire"
                d={`M270 ${y} C345 ${y}, 300 235, 430 235`}
              />
              {running && rate > 0 && (
                <path
                  className="engine-pulse"
                  style={{
                    animationDuration: `${Math.max(0.65, 4 / Math.log2(rate + 2))}s`,
                    animationDelay: `${i * -0.5}s`,
                  }}
                  d={`M270 ${y} C345 ${y}, 300 235, 430 235`}
                />
              )}
            </g>
          ))}
          <circle cx="546" cy="235" r="166" className="engine-outer" />
          <circle cx="546" cy="235" r="140" className="engine-track" />
          <circle
            cx="546"
            cy="235"
            r="140"
            fill="none"
            stroke={`url(#${id})`}
            strokeWidth="8"
            strokeLinecap="round"
            strokeDasharray={Math.PI * 280}
            strokeDashoffset={Math.PI * 280 * (1 - progress)}
            transform="rotate(-90 546 235)"
            className="engine-progress"
          />
          <text x="546" y="226" textAnchor="middle" className="engine-count">
            {saved.toLocaleString()}
          </text>
          <text x="546" y="260" textAnchor="middle" className="engine-label">
            {zh ? "条序列已保存" : "sequences saved"}
          </text>
          <text x="546" y="296" textAnchor="middle" className="engine-caption">
            {Math.round(progress * 100)}% · {target.toLocaleString()}{" "}
            {zh ? "目标" : "target"}
          </text>
        </svg>
      </div>
      <div className="model-contributions">
        <div>
          <span>{zh ? "模型限制占比" : "Limiting-model attribution"}</span>
          <span>
            {contributions
              ? zh
                ? "已保存替换位点"
                : "Saved substitutions"
              : zh
                ? "等待归因数据"
                : "Awaiting attribution"}
          </span>
        </div>
        <div className="contribution-track">
          {contributions &&
            attributionTotal > 0 &&
            Object.entries(contributions).map(([name, count], i) => (
              <span
                key={name}
                style={{
                  width: `${(count / attributionTotal) * 100}%`,
                  background: [
                    "var(--score-activity)",
                    "var(--score-ph)",
                    "var(--score-stability)",
                  ][i],
                }}
                title={`${name}: ${count.toFixed(1)}`}
              />
            ))}
        </div>
        {attributionTotal > 0 && (
          <div className="contribution-legend">
            {generators.map((entry, i) => (
              <span key={entry.id}>
                <i
                  style={{
                    background: [
                      "var(--score-activity)",
                      "var(--score-ph)",
                      "var(--score-stability)",
                    ][i],
                  }}
                />
                {entry.name}{" "}
                <strong>
                  {(
                    (contributions![entry.id] / attributionTotal) *
                    100
                  ).toFixed(1)}
                  %
                </strong>
              </span>
            ))}
          </div>
        )}
        {history.length > 1 && (
          <svg
            className="contribution-history"
            viewBox="0 0 400 38"
            preserveAspectRatio="none"
            role="img"
            aria-label={
              zh
                ? "已保存序列的累计模型限制占比变化"
                : "Cumulative limiting-model shares across saved records"
            }
          >
            {generators.map((entry, i) => {
              const edge = (
                sample: (typeof history)[number],
                upper: boolean,
              ) => {
                const total = Object.values(sample.counts).reduce(
                  (a, b) => a + b,
                  0,
                );
                return (
                  38 -
                  (generators
                    .slice(0, i + (upper ? 1 : 0))
                    .reduce((sum, model) => sum + sample.counts[model.id], 0) /
                    total) *
                    38
                );
              };
              const upper = history.map(
                (sample, j) =>
                  `${(j / (history.length - 1)) * 400},${edge(sample, true)}`,
              );
              const lower = [...history]
                .reverse()
                .map(
                  (sample, j) =>
                    `${((history.length - 1 - j) / (history.length - 1)) * 400},${edge(sample, false)}`,
                );
              return (
                <path
                  key={entry.id}
                  d={`M${upper.join(" L")} L${lower.join(" L")} Z`}
                  fill={
                    [
                      "var(--score-activity)",
                      "var(--score-ph)",
                      "var(--score-stability)",
                    ][i]
                  }
                  opacity=".75"
                />
              );
            })}
          </svg>
        )}
        <p>
          {zh
            ? "统计各替换位点上给出最低校准 LLR 的模型；并列均分，不代表因果贡献。"
            : "Share of minimum calibrated LLR per substitution; ties split equally. Not causal contribution."}
        </p>
      </div>
      <div className="engine-caption-row">
        <span>
          {zh
            ? "三个生成模型 · 一个突变库"
            : "Three generators. One mutation library."}
        </span>
        <span>
          {model
            ? `${generators.find((entry) => entry.id === model)?.name ?? model} · ${zh ? "正在计算" : "Computing"}`
            : saved >= target && target > 0
              ? zh
                ? "目标数量已保存"
                : "Target saved"
              : rate > 0
                ? `${rate.toFixed(0)} seq/s`
                : zh
                  ? "等待保存记录"
                  : "Awaiting saved records"}
        </span>
      </div>
    </div>
  );
}

import { useEffect, useRef, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { api, isDemo } from "../../shared/api/client";
import { rememberRequest } from "../runs/requestStorage";
import { useRun } from "../runs/useRun";
import { useApp } from "../../app/store";
import { ChatPanel } from "../experiment/ChatPanel";
import { WorkspaceIntro } from "./WorkspaceIntro";
import { MutationAtlas, ScoreDistribution, ScoringWaterfall } from "../results/Charts";
import proteinIllustration from "../../../../../../resource/docs/anewomni-protein.svg";
import moleculeDecoration from "../../../../../../resource/docs/enzora-molecule-decoration.svg";
import dnaDecoration from "../../../../../../resource/docs/enzora-dna-decoration.svg";
import {
  defaultSelection,
  reviewedRequest,
  sequenceError,
  selectionError,
  candidateError,
} from "../experiment/review";
import {
  downloadText,
  serializeCsv,
  artifactFilename,
} from "../../shared/lib/download";

const scorers = [
  {
    key: "activity",
    name: "Phenom",
    field: "activity_log10",
    label: "Activity · log₁₀ kcat",
    labelZh: "活性 · log₁₀ kcat",
    color: "var(--score-activity)",
    stage: 2,
  },
  {
    key: "ph",
    name: "EpHod",
    field: "ph_opt",
    label: "Optimal pH",
    labelZh: "最适 pH",
    color: "var(--score-ph)",
    stage: 3,
  },
  {
    key: "stability",
    name: "UniStab",
    field: "ddg",
    label: "Stability change · ΔΔG",
    labelZh: "稳定性变化 · ΔΔG",
    color: "var(--score-stability)",
    stage: 4,
  },
] as const;
const number = (value: unknown, digits = 3) =>
  typeof value === "number" && Number.isFinite(value)
    ? value.toFixed(digits)
    : "—";

export function Workbench() {
  const { lang, sessionId, runRequest, setRunRequest } = useApp();
  const zh = lang === "zh";
  const [params, setParams] = useSearchParams();
  const jobId = params.get("job");
  const run = useRun(jobId);
  const [sequence, setSequence] = useState("");
  const [name, setName] = useState("");
  const [ligand, setLigand] = useState("");
  const [size, setSize] = useState(1000);
  const [order, setOrder] = useState(1);
  const [tab, setTab] = useState(0);
  const [selection, setSelection] = useState(defaultSelection);
  const [host, setHost] = useState("e_coli");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [exportFormat, setExportFormat] = useState("json");
  const [picked, setPicked] = useState<string | null>(null);
  const [view, setView] = useState(1);
  const [rate, setRate] = useState(0);
  const lastCount = useRef<{ n: number; t: number } | null>(null);
  const scroll = useRef<HTMLElement>(null);
  const sequenceView = useRef<HTMLDivElement>(null);
  const navigation = useRef({frame:0,active:false,target:1});
  const goRef = useRef<(phase:number)=>void>(()=>{});
  const viewRef = useRef(1);
  useEffect(() => setPicked(null), [jobId]);
  useEffect(() => {
    if (!notice) return;
    const timer = window.setTimeout(() => setNotice(""), 5000);
    return () => window.clearTimeout(timer);
  }, [notice]);
  const active =
    run.meta?.status === "running" || run.meta?.status === "queued";
  const progress = run.meta?.progress;
  const phase = progress?.phase ?? 1;
  const waiting = progress?.status === "awaiting_next";
  const target = Number(
    run.generation.target_sequences ??
      progress?.target_sequences ??
      (jobId ? runRequest?.candidate.top_k : size) ??
      size,
  );
  const saved = Number(
    run.generation.saved_sequences ??
      progress?.saved_sequences ??
      run.meta?.stages.find((stage) => stage.index === 1)?.stats
        ?.saved_sequences ??
      0,
  );
  const selected = run.result?.selected ?? [];
  const current = picked
    ? (selected.find((c) => c.mutation === picked) ??
      run.rawCandidates.find((c) => c.mutation === picked))
    : selected[0];
  useEffect(() => {
    const area = sequenceView.current;
    if (!area) return;
    const residue = area.querySelector("mark");
    if (!residue) {
      area.scrollTop = 0;
      return;
    }
    const box = area.getBoundingClientRect();
    const mutation = residue.getBoundingClientRect();
    area.scrollTop += mutation.top - box.top - (area.clientHeight - mutation.height) / 2;
  }, [current?.mutation, current?.mutant_sequence]);
  const weights = selection.weights,
    confidence = selection.confidence!;
  const total = Object.keys(weights).reduce(
    (n, k) =>
      n +
      weights[k as keyof typeof weights] *
        confidence[k as keyof typeof weights],
    0,
  );
  useEffect(() => {
    if (runRequest) {
      setSequence(runRequest.sequence);
      setName(runRequest.sequence_name);
      setLigand(runRequest.activity?.ligand_smiles ?? "");
      setSelection(runRequest.selection);
      setSize(runRequest.candidate.top_k);
      const orders = Object.keys(runRequest.mutation?.counts ?? {});
      if (orders.length === 1) setOrder(Number(orders[0]));
    }
  }, [runRequest]);
  useEffect(() => {
    const t = performance.now(),
      previous = lastCount.current;
    if (previous && saved > previous.n)
      setRate((saved - previous.n) / ((t - previous.t) / 1000));
    lastCount.current = { n: saved, t };
    if (!active || waiting) setRate(0);
  }, [saved, active, waiting]);
  useEffect(() => {
    const root = scroll.current!;
    const sections = [...root.querySelectorAll<HTMLElement>("[data-phase]")];
    const update = () => {
      const anchor =
        root.getBoundingClientRect().top + root.clientHeight * 0.35;
      const current = sections.find((section) => {
        const box = section.getBoundingClientRect();
        return box.top <= anchor && box.bottom > anchor;
      });
      if (current) {viewRef.current=Number(current.dataset.phase);setView(viewRef.current);}
    };
    root.addEventListener("scroll", update, { passive: true });
    window.addEventListener("resize", update);
    update();
    return () => {
      root.removeEventListener("scroll", update);
      window.removeEventListener("resize", update);
    };
  }, []);
  const go = (next: number) => {
    const root=scroll.current,section=document.getElementById(`phase-${next}`);
    if(!root||!section||next<1||next>3)return;
    const state=navigation.current;
    cancelAnimationFrame(state.frame);
    const start=root.scrollTop;
    const target=start+section.getBoundingClientRect().top-root.getBoundingClientRect().top;
    const duration=matchMedia('(prefers-reduced-motion: reduce)').matches?0:Math.min(900,620+Math.abs(target-start)/root.clientHeight*110);
    state.active=true;state.target=next;root.dataset.transition='true';root.style.scrollBehavior='auto';
    const began=performance.now();
    const tick=(time:number)=>{
      const t=duration===0?1:Math.min(1,(time-began)/duration);
      const eased=t*t*t*(t*(t*6-15)+10);
      root.scrollTop=start+(target-start)*eased;
      if(t<1)state.frame=requestAnimationFrame(tick);
      else {state.active=false;delete root.dataset.transition;root.style.scrollBehavior='';viewRef.current=next;setView(next);}
    };
    state.frame=requestAnimationFrame(tick);
  };
  goRef.current=go;
  useEffect(()=>{
    const root=scroll.current!;
    const gesture={last:0,total:0,latched:false};
    const wheel=(event:WheelEvent)=>{
      if(event.defaultPrevented||event.ctrlKey||Math.abs(event.deltaX)>Math.abs(event.deltaY))return;
      if(event.target instanceof Element&&event.target.closest('.mutation-atlas:not(.is-empty)'))return;
      // Editors, result lists and the molecular map retain their own scrolling.
      let element=event.target instanceof HTMLElement?event.target:null;
      while(element&&element!==root){
        const style=getComputedStyle(element);
        if(/auto|scroll/.test(style.overflowY)&&element.scrollHeight>element.clientHeight+1){
          const canScroll=event.deltaY>0?element.scrollTop+element.clientHeight<element.scrollHeight-2:element.scrollTop>2;
          if(canScroll)return;
        }
        element=element.parentElement;
      }
      if(innerWidth<=1000)return;
      event.preventDefault();
      const now=performance.now();
      if(now-gesture.last>180){gesture.total=0;gesture.latched=false;}
      gesture.last=now;
      if(navigation.current.active||gesture.latched)return;
      const delta=event.deltaY*(event.deltaMode===1?16:event.deltaMode===2?root.clientHeight:1);
      if(Math.sign(delta)!==Math.sign(gesture.total))gesture.total=0;
      gesture.total+=delta;
      if(Math.abs(gesture.total)<42)return;
      gesture.latched=true;
      goRef.current(Math.max(1,Math.min(3,viewRef.current+Math.sign(gesture.total))));
    };
    const keyboard=(event:KeyboardEvent)=>{
      if((event.target as HTMLElement).closest('input,textarea,select,button,a,[contenteditable=true]'))return;
      const target=event.key==='PageDown'?viewRef.current+1:event.key==='PageUp'?viewRef.current-1:event.key==='Home'?1:event.key==='End'?3:null;
      if(target!==null){event.preventDefault();goRef.current(Math.max(1,Math.min(3,target)));}
    };
    root.addEventListener('wheel',wheel,{passive:false});window.addEventListener('keydown',keyboard);
    return()=>{root.removeEventListener('wheel',wheel);window.removeEventListener('keydown',keyboard);cancelAnimationFrame(navigation.current.frame);};
  },[]);
  const perform = async (action: () => Promise<unknown>) => {
    setBusy(true);
    setError("");
    setNotice("");
    try {
      await action();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };
  const start = () =>
    void perform(async () => {
      if (sequenceError(sequence))
        throw new Error(
          zh
            ? "请输入 1–1000 位标准氨基酸序列。"
            : "Enter 1–1000 canonical amino-acid residues.",
        );
      if (!ligand.trim())
        throw new Error(
          zh
            ? "Phenom 需要底物 SMILES。"
            : "Phenom requires the substrate SMILES.",
        );
      if (selectionError(selection))
        throw new Error(
          zh
            ? "请检查评分目标、权重和阈值。"
            : "Check the scoring targets, weights and thresholds.",
        );
      const candidate = {
        top_k: size,
        allowed_positions: null,
        excluded_positions: [],
      };
      if (
        candidateError(candidate, selection, sequence.length) ||
        !Number.isInteger(order) ||
        order < 1 ||
        order > sequence.length
      )
        throw new Error(
          zh
            ? "建库数量或突变阶数无效。"
            : "Invalid library size or mutation order.",
        );
      const request = reviewedRequest({
        name,
        sequence,
        goal: `Codon optimization for ${host === "e_coli" ? "Escherichia coli" : "Bacillus subtilis"}.`,
        candidate,
        selection,
        aiReady: !isDemo(),
        lang,
        ligandSmiles: ligand.trim(),
        mutationOrder: order,
      });
      const { job_id } = await api.submitRun(request, sessionId);
      rememberRequest(job_id, request);
      lastCount.current = null;
      setPicked(null);
      setParams(
        (p) => {
          const next = new URLSearchParams(p);
          next.set("job", job_id);
          return next;
        },
        { replace: true },
      );
    });
  const applyBalance = async () => {
    if (!jobId) throw new Error("Start a run first.");
    await api.balanceRun(jobId, weights, confidence);
    if (runRequest) {
      const updated = {
        ...runRequest,
        selection: { ...runRequest.selection, weights, confidence },
      };
      setRunRequest(updated);
      rememberRequest(jobId, updated);
    }
    setNotice(zh ? "配平已应用。" : "Balance applied.");
  };
  const next = () =>
    void perform(async () => {
      if (!jobId || !waiting || !progress?.next_phase) return;
      if (phase === 2) await applyBalance();
      await api.advanceRun(jobId, progress.next_phase);
      go(progress.next_phase);
      run.retry();
    });
  const exportReport = () =>
    void perform(async () => {
      if (!jobId) return;
      const ref = artifactFilename(run.result?.markdown_report, "report.md");
      downloadText(
        isDemo()
          ? run.result!.markdown_report!
          : await api.readArtifact(jobId, ref),
        "enzora-report.md",
        "text/markdown;charset=utf-8",
      );
    });
  const exportCodons = () =>
    void perform(async () => {
      const rows: Record<string, unknown>[] = [];
      for (const row of [{ mutation: "WT", mutant_sequence: run.wtSequence }, ...selected]) {
        if (!row.mutant_sequence)
          throw new Error(`Missing protein sequence: ${row.mutation}`);
        const result = await api.codonOptimize(row.mutant_sequence, [host]);
        if (result.errors.length)
          throw new Error(result.errors.map((e) => e.error).join("; "));
        rows.push(
          ...result.results.map((r) => ({ ...r, mutation: row.mutation })),
        );
      }
      downloadText(
        serializeCsv(rows, [
          "mutation",
          "scientific_name",
          "optimized_cds_with_stop",
          "gc_fraction",
        ]),
        "enzora-codons.csv",
        "text/csv;charset=utf-8",
      );
      setNotice(
        zh
          ? "宿主密码子优化已完成并导出。"
          : "Host codons optimized and exported.",
      );
    });
  const sample = () => {
    setSequence(
      "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQANMKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAN",
    );
    setName("Demo enzyme");
    setLigand("CCO");
    setSize(1000);
    setOrder(1);
    setSelection(defaultSelection());
  };
  const balanceLocked = !active || phase > 2;
  const balanceDirty =
    !!runRequest &&
    scorers.some(
      ({ key }) =>
        weights[key] !== runRequest.selection.weights[key] ||
        confidence[key] !== runRequest.selection.confidence?.[key],
    );
  return (
    <>
      <main ref={scroll} id="main-content" className="process-scroll">
        <div className="process-canvas">
          <section
            id="phase-1"
            data-phase="1"
            className="process-section library-section"
          >
            <img
              className="protein-decoration stage-decoration stage-decoration-library"
              src={proteinIllustration}
              alt=""
              aria-hidden="true"
            />
            <div className="section-heading">
              <div>
                <h1>Enzora</h1>
                <h2 className="library-stage-title">{zh ? "酶突变库构建" : "Enzyme Mutation Database Construction"}</h2>
                <p>
                  {zh
                    ? "输入起始酶序列和底物，设置要生成多少条突变序列。三个模型共同提出突变，建立下一步用于筛选的候选库。"
                    : "Enter your enzyme and substrate, then choose how many variants to make. Three models propose mutations to build a library for screening."}
                </p>
              </div>
            </div>
            <div className="library-composition">
              <div className="library-controls">
                <div
                  className="flip-tabs"
                  role="tablist"
                  aria-label={zh ? "建库步骤" : "Library views"}
                >
                  {[
                    zh ? "输入" : "Input",
                    zh ? "设置" : "Settings",
                  ].map((label, i) => (
                    <button
                      id={`library-tab-${i}`}
                      role="tab"
                      key={i}
                      aria-selected={tab === i}
                      aria-controls={`library-panel-${i}`}
                      tabIndex={tab === i ? 0 : -1}
                      onKeyDown={(e) => {
                        if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
                          const next =
                            (i + 1) % 2;
                          setTab(next);
                          document
                            .getElementById(`library-tab-${next}`)
                            ?.focus();
                        }
                      }}
                      onClick={() => setTab(i)}
                    >
                      {label}
                    </button>
                  ))}
                </div>
                <div
                  className="flip-content"
                  key={tab}
                  role="tabpanel"
                  id={`library-panel-${tab}`}
                  aria-labelledby={`library-tab-${tab}`}
                >
                  {tab === 0 && (
                    <>
                      <div className="input-heading">
                        <h2>{zh ? "输入酶和底物" : "Enter enzyme & substrate"}</h2>
                        {isDemo() && (
                          <button
                            className="text-button"
                            onClick={sample}
                            disabled={active}
                          >
                            {zh ? "载入示例" : "Load sample"} ↗
                          </button>
                        )}
                      </div>
                      <label className="field">
                        {zh ? "实验名称" : "Experiment name"}
                        <input
                          value={name}
                          onChange={(e) => setName(e.target.value)}
                          disabled={active}
                          placeholder="Enzyme design"
                        />
                      </label>
                      <label className="field">
                        {zh ? "蛋白序列" : "Protein sequence"}
                        <textarea
                          className="protein-input"
                          spellCheck={false}
                          value={sequence}
                          onChange={(e) => setSequence(e.target.value)}
                          disabled={active}
                          placeholder="MKTAYIAKQRQISFVK…"
                          rows={4}
                        />
                      </label>
                      <div className="input-meta">
                        <span>
                          {sequence.length} {zh ? "氨基酸" : "residues"}
                        </span>
                        <Link to="/new">
                          {zh
                            ? "FASTA / 核酸 / 数据库导入"
                            : "FASTA / DNA / Database import"}{" "}
                          ↗
                        </Link>
                      </div>
                      <label className="field">
                        {zh
                          ? "底物 SMILES · Phenom 必填"
                          : "Substrate SMILES · Required for Phenom"}
                        <input
                          value={ligand}
                          onChange={(e) => setLigand(e.target.value)}
                          disabled={active}
                          placeholder={
                            zh ? "输入实际底物" : "Enter the actual substrate"
                          }
                        />
                      </label>
                      <p className="small dim">
                        {isDemo()
                          ? zh
                            ? "示例蛋白与底物仅用于演示，不代表真实实验。"
                            : "The sample protein and substrate are illustrative only."
                          : zh
                            ? "活性预测对应当前酶序列与底物的组合。"
                            : "Activity predictions are specific to this enzyme–substrate pair."}
                      </p>
                    </>
                  )}
                  {tab === 1 && (
                    <>
                      <h2>
                        {zh ? "设置突变库" : "Library settings"}
                      </h2>
                      <label className="field">
                        {zh ? "突变库序列数量" : "Library size"}
                        <input
                          type="number"
                          min={1}
                          step={1}
                          value={size}
                          onChange={(e) => setSize(Number(e.target.value))}
                          disabled={active}
                        />
                      </label>
                      <label className="field">
                        {zh
                          ? "每条序列的替换位点数"
                          : "Substitutions per sequence"}
                        <input
                          type="number"
                          min={1}
                          max={sequence.length || 1000}
                          step={1}
                          value={order}
                          onChange={(e) => setOrder(Number(e.target.value))}
                          disabled={active}
                        />
                      </label>
                      <label className="field">
                        Top-K
                        <input
                          type="number"
                          min={1}
                          max={50}
                          value={selection.top_k}
                          onChange={(e) =>
                            setSelection({
                              ...selection,
                              top_k: Number(e.target.value),
                            })
                          }
                          disabled={active}
                        />
                      </label>
                      <p>
                        {zh
                          ? "按指定数量建立突变库，默认生成 1000 条单点突变；不再强制全量枚举。"
                          : "Generate the requested quota: 1,000 single substitutions by default, with configurable mutation order."}
                      </p>
                      <p className="small dim">
                        {zh
                          ? "当前生成实现使用 ProteinDPO、EnzGFM 与 VESM3；酶活性由 Phenom 预测。"
                          : "Generation uses ProteinDPO, EnzGFM and VESM3. Phenom predicts enzyme activity."}
                      </p>
                    </>
                  )}
                </div>
                <div className="library-actions">
                  <button
                    className="primary"
                    disabled={active || busy}
                    onClick={start}
                  >
                    {busy
                      ? zh
                        ? "处理中…"
                        : "Working…"
                      : zh
                        ? "构建突变库"
                        : "Build library"}{" "}
                    <span>↗</span>
                  </button>
                  {active && (
                    <button
                      className="ghost"
                      disabled={busy}
                      onClick={() =>
                        void perform(async () => {
                          await api.cancelRun(jobId!);
                          run.retry();
                        })
                      }
                    >
                      {zh ? "取消运行" : "Cancel run"}
                    </button>
                  )}
                </div>
              </div>
              <WorkspaceIntro
                runKey={jobId ?? 'idle'}
                landscape={run.generation.llr_landscape as Parameters<typeof WorkspaceIntro>[0]['landscape']}
                saved={saved}
                target={target}
                running={active && !waiting && phase === 1}
                rate={rate}
                zh={zh}
                contributions={
                  run.generation.limiting_model_counts as
                    Record<string, number> | undefined
                }
                model={
                  run.generation.status === "scoring" &&
                  typeof run.generation.model === "string"
                    ? run.generation.model
                    : undefined
                }
              />
            </div>
          </section>
          <section
            id="phase-2"
            data-phase="2"
            className="process-section screening-section"
          >
            <img className="stage-decoration stage-decoration-screening" src={moleculeDecoration} alt="" aria-hidden="true" draggable={false} />
            <div className="section-heading">
              <div>
                <h2>
                  {zh
                    ? "多目标虚拟筛选"
                    : "Multi-objective virtual screening"}
                </h2>
                <p>
                  {zh
                    ? "预测每条候选的活性、最适 pH 和稳定性。在右侧设置你更看重的性质，助手会参考预测的不确定性，帮助调整最终排序中各项分数的占比。"
                    : "Predict activity, optimal pH and stability for each candidate. Set your priorities on the right; the Assistant reviews prediction uncertainty to help balance the final ranking."}
                </p>
              </div>
            </div>
            <div className="screening-composition">
              <div className="scorer-stack">
                {scorers.map((model) => {
                  const values = run.rawCandidates
                    .map((c) => c[model.field])
                    .filter(
                      (n): n is number =>
                        typeof n === "number" && Number.isFinite(n),
                    );
                  const completed = run.meta?.stages.some(
                    (s) => s.index === model.stage && s.finished_at != null,
                  );
                  return (
                    <article className="scorer-row" key={model.key}>
                      <div className="scorer-name">
                        <h3>{model.name}</h3>
                        <span style={{ color: model.color }}>
                          {zh ? model.labelZh : model.label}
                        </span>
                        <span className="scorer-state">
                          {completed
                            ? zh
                              ? "预测完成"
                              : "Scored"
                            : phase === 2 &&
                                !waiting &&
                                model.stage ===
                                  Math.max(
                                    1,
                                    ...(run.meta?.stages.map((s) => s.index) ??
                                      []),
                                  ) +
                                    1
                              ? zh
                                ? "运行中"
                                : "Scoring"
                              : zh
                                ? "等待"
                                : "Waiting"}
                        </span>
                      </div>
                      <ScoreDistribution
                        rows={run.rawCandidates}
                        field={model.field}
                        color={model.color}
                        zh={zh}
                      />
                      <div className="scorer-summary">
                        <strong>
                          {values.length
                            ? number(
                                values.reduce((a, b) => a + b, 0) /
                                  values.length,
                                2,
                              )
                            : "—"}
                        </strong>
                        <span>
                          {zh ? "均值" : "Mean"} ·{" "}
                          {values.length.toLocaleString()}{" "}
                          {zh ? "条预测" : "predictions"}
                        </span>
                      </div>
                    </article>
                  );
                })}
                <ScoringWaterfall rows={run.rawCandidates} weights={weights} confidence={confidence} zh={zh} active={view===2}/>
                <details className="phenom-model-details">
                  <summary>{zh ? "Phenom · 模型架构与不确定性" : "Phenom · Architecture & uncertainty"}</summary>
                  <p>{zh
                    ? "以蛋白质序列和底物分子图为输入，分层图网络聚合残基、口袋、原子与 motif；双向注意力构造可解释的酶–底物交互指纹，预测 log₁₀(kcat) 与 log₁₀(Km)。"
                    : "Protein sequences and substrate molecular graphs feed a hierarchical graph network over residues, pockets, atoms and motifs. Bidirectional attention constructs an interpretable interaction fingerprint to predict log₁₀(kcat) and log₁₀(Km)."}</p>
                  <p>{zh
                    ? "原生 Gaussian head 同时输出均值 μ 与标准差 σ。注意力揭示模型关注位置，不代表实测催化机制；σ 不等同于校准置信度。Confidence 是用户设置的配平参数。"
                    : "Native Gaussian heads output a mean μ and standard deviation σ. Attention indicates model focus rather than experimentally established catalytic mechanisms. σ is not calibrated confidence; Confidence sets user-defined balancing preferences."}</p>
                </details>
                <div className="agent-decision">
                  <span className="copilot-orb" />
                  <div>
                    <strong>
                      {zh ? "助手正在做什么" : "Assistant review"}
                    </strong>
                    <p>
                      {run.meta?.status === 'failed' || run.meta?.status === 'cancelled'
                        ? zh
                          ? "本轮已停止。请检查运行提示与已保存的预测输出。"
                          : "This run has stopped. Check the run message and saved prediction outputs."
                        : waiting && phase === 2
                        ? zh
                          ? "评分已就绪。调整配平后，到结果页筛选 Top-K。"
                          : "Scores are ready. Review the balance, then select Top-K on the Output page."
                        : run.meta?.status === "success"
                          ? zh
                            ? "Top-K 与报告已生成。可在结果页检查序列并导出。"
                            : "Top-K and report are ready. Inspect sequences and export in Output."
                          : phase >= 3
                            ? zh
                              ? "已确认配平，正在筛选与生成报告。"
                              : "Balance confirmed. Selecting candidates and preparing the report."
                            : phase === 2
                              ? zh
                                ? "模型正在逐条预测。结果返回后，助手会检查评分与不确定性，提出配平建议。"
                                : "The models are scoring candidates. The Assistant will review their predictions and uncertainty to suggest a balance."
                            : zh
                              ? "建库完成后，点击左下角启动筛选；预测结果返回后，助手会检查各模型的评分与不确定性。"
                              : "Start screening after the library is ready. The Assistant will review the returned predictions and their uncertainty."}
                    </p>
                  </div>
                </div>
              </div>
              <div className="design-space">
                <div className="balance-settings">
                  <div className="input-heading">
                    <h3>{zh ? "评分配平" : "Score balance"}</h3>
                    <span
                      className="balance-status"
                      data-dirty={balanceDirty}
                      role="status"
                    >
                      {!jobId || !active
                        ? zh
                          ? "下轮运行设置"
                          : "Next run settings"
                        : balanceDirty
                          ? zh
                            ? "有未应用修改"
                            : "Unapplied changes"
                          : zh
                            ? "已应用"
                            : "Applied"}
                    </span>
                  </div>
                  <p className="balance-formula">
                    {zh
                      ? "偏好表示你有多看重这项性质；降低可信度，就会减少该模型对最终排序的影响。"
                      : "Preferences set your priorities. Lower confidence gives a predictor less influence on the final ranking."}
                  </p>
                  <div className="balance-bar">
                    {scorers.map((m) => (
                      <span
                        key={m.key}
                        style={{
                          width: `${total ? ((weights[m.key] * confidence[m.key]) / total) * 100 : 0}%`,
                          background: m.color,
                        }}
                      />
                    ))}
                  </div>
                  {scorers.map((m) => (
                    <div className="balance-control" key={m.key}>
                      <div>
                        <strong>{m.name}</strong>
                        <span>
                          {total
                            ? Math.round(
                                ((weights[m.key] * confidence[m.key]) / total) *
                                  100,
                              )
                            : 0}
                          %
                        </span>
                      </div>
                      <div className="balance-inputs">
                        <label>
                          {zh ? "偏好权重" : "Preference"}
                          <input
                            aria-label={`${m.name} preference`}
                            type="number"
                            min={0}
                            step={0.05}
                            disabled={busy || (active && phase > 2)}
                            value={weights[m.key]}
                            onChange={(e) =>
                              setSelection({
                                ...selection,
                                weights: {
                                  ...weights,
                                  [m.key]: Number(e.target.value),
                                },
                              })
                            }
                          />
                        </label>
                        <label>
                          {zh ? "可信度" : "Confidence"}{" "}
                          <span>{Math.round(confidence[m.key] * 100)}%</span>
                          <input
                            aria-label={`${m.name} confidence`}
                            type="range"
                            min={0}
                            max={1}
                            step={0.01}
                            disabled={busy || (active && phase > 2)}
                            value={confidence[m.key]}
                            style={
                              {
                                "--range-fill": `${confidence[m.key] * 100}%`,
                                "--range-color": m.color,
                              } as React.CSSProperties
                            }
                            onChange={(e) =>
                              setSelection({
                                ...selection,
                                confidence: {
                                  ...confidence,
                                  [m.key]: Number(e.target.value),
                                },
                              })
                            }
                          />
                        </label>
                      </div>
                    </div>
                  ))}
                  <div className="threshold-fields">
                    <label className="field">
                      {zh ? "目标 pH（可选）" : "Target pH (optional)"}
                      <input
                        type="number"
                        min={0}
                        max={14}
                        step={0.1}
                        disabled={active}
                        value={selection.target_ph ?? ""}
                        onChange={(e) =>
                          setSelection({
                            ...selection,
                            target_ph:
                              e.target.value === ""
                                ? null
                                : Number(e.target.value),
                            ph_tolerance: e.target.value === "" ? null : 1,
                          })
                        }
                      />
                    </label>
                    <label className="field">
                      {zh
                        ? "活性下限 log₁₀（可选）"
                        : "Min log₁₀ activity (optional)"}
                      <input
                        type="number"
                        step={0.1}
                        disabled={active}
                        value={selection.min_activity_log10 ?? ""}
                        onChange={(e) =>
                          setSelection({
                            ...selection,
                            min_activity_log10:
                              e.target.value === ""
                                ? null
                                : Number(e.target.value),
                          })
                        }
                      />
                    </label>
                  </div>
                  <p className="small dim">
                    {zh
                      ? "可信度为配平系数，不是校准概率。目标与阈值仅在启动前设置。"
                      : "Confidence adjusts influence, not a probability. Set targets before screening."}
                  </p>
                </div>
                <button
                  disabled={busy || balanceLocked}
                  onClick={() => void perform(applyBalance)}
                >
                  {zh ? "应用配平" : "Apply balance"} ↗
                </button>
              </div>
            </div>
            <div className="section-bottom">
<button
                className="primary"
                disabled={busy || !(waiting && phase === 1)}
                onClick={next}
              >
                {zh ? "启动虚拟筛选" : "Start Screening"} ↗
              </button>

            </div>
          </section>
          <section
            id="phase-3"
            data-phase="3"
            className="process-section output-section"
          >
            <img className="stage-decoration stage-decoration-output" src={dnaDecoration} alt="" aria-hidden="true" draggable={false} />
            <div className="output-main">
              <MutationAtlas
                data={run.atlas}
                picked={picked}
                onPick={setPicked}
                zh={zh}
              />
              <div className="output-intro">
                <h2>{zh?'突变分布 UMAP 可视化':'Mutation distribution · UMAP'}</h2>
                <p>{zh?'拖动或缩放地图，查看候选序列的分布。筛选 Top-K 后，检查蓝色标出的突变位点，并导出报告、蛋白序列或适合目标宿主的编码 DNA。':'Drag or zoom the map to explore your candidates. Select Top-K, inspect the mutations highlighted in blue, and export the report, protein sequences or host-optimized coding DNA.'}</p>
                {waiting&&phase===2&&<button className="primary output-select" disabled={busy} onClick={next}>{zh?'筛选 Top-K 并生成报告':'Select Top-K & generate report'} ↗</button>}
                <div className="output-metrics">
                  <div>
                    <strong>
                      {run.atlas?.points.length.toLocaleString() ?? "—"}
                    </strong>
                    <span>{zh ? "条库内序列" : "library sequences"}</span>
                  </div>
                  <div>
                    <strong>{selected.length || "—"}</strong>
                    <span>{zh ? "条精选序列" : "selected candidates"}</span>
                  </div>
                </div>
                <div className="host-actions">
                  <label className="field">
                    {zh ? "密码子优化宿主" : "Expression host"}
                    <select
                      value={host}
                      onChange={(e) => setHost(e.target.value)}
                      disabled={active}
                    >
                      <option value="e_coli">Escherichia coli</option>
                      <option value="b_subtilis">Bacillus subtilis</option>
                    </select>
                  </label>
                  <button
                    disabled={busy || !selected.length}
                    onClick={exportCodons}
                  >
                    {zh ? "优化并导出 CDS" : "Optimize & export CDS"} ↗
                  </button>
                </div>
                <p className="small dim">
                  {isDemo()
                    ? zh
                      ? "演示投影不是 UMAP，全部分数均为模拟。"
                      : "Demo projection is not UMAP. All scores are simulated."
                    : zh
                      ? "真实 UMAP 保留坐标几何；同义密码子优化不改变蛋白序列。"
                      : "UMAP preserves coordinate geometry. Synonymous codon optimization preserves the protein."}
                </p>
              </div>
            </div>
            <div className="output-report">
              <div className="report-copy">
                <h3>{zh ? "Agent 分析报告" : "Agent report"}</h3>
                <p>
                  {run.analysisText ??
                    (run.meta?.status === "failed" || run.meta?.status === "cancelled"
                      ? (zh
                        ? "本次流程已停止，尚未生成最终报告。请查看运行日志与已保存的预测结果。"
                        : "This run stopped before producing a final report. Inspect the run log and saved predictions.")
                      : (zh
                        ? "完成评分后，点击本页的“筛选 Top-K 并生成报告”。"
                        : "After scoring, select Top-K on this page to generate the report."))}
                </p>
                <button
                  className="text-button"
                  disabled={!run.result?.markdown_report || busy}
                  onClick={exportReport}
                >
                  {zh ? "导出完整报告" : "Export full report"} ↗
                </button>
              </div>
              <div className="top-results">
                <div className="input-heading">
                  <h3>
                    Top-{selected.length || selection.top_k}{" "}
                    {zh ? "候选" : "candidates"}
                  </h3>
                  <div>
                    <select aria-label={zh?'结果导出格式':'Result export format'} value={exportFormat} onChange={e=>setExportFormat(e.target.value)}>
                      {['json','csv','fasta','genbank','sbol'].map(format=><option key={format} value={format}>{format.toUpperCase()}</option>)}
                    </select>
                    <button className="text-button" disabled={!selected.length||busy} onClick={()=>void perform(async()=>{
                      const bioartifact={schema:'enzora.bioartifact/1' as const, records:selected.map(c=>{if(typeof c.mutant_sequence!=='string')throw new Error('Missing candidate sequence');return {id:c.mutation,name:c.mutation,type:'protein' as const,sequence:c.mutant_sequence};}),
                        predictions:{candidates:selected},metadata:{job_id:jobId,selection:run.meta?.applied_selection,reference_protein:run.wtSequence},data:{}};
                      const result=await api.exportArtifact(bioartifact,exportFormat);
                      downloadText(result.content,`enzora-topk.${exportFormat==='genbank'?'gb':exportFormat==='sbol'?'xml':exportFormat}`);
                      if(result.warnings.length)setNotice(result.warnings.join(' '));
                    })}>{zh?'导出结果':'Export results'} ↗</button>
                  </div>
                </div>
                <div className="top-candidate-list">
                  {!!selected.length && (
                    <table>
                      <caption className="sr-only">
                        {zh
                          ? "Top-K 模型预测，点击突变名称定位序列"
                          : "Top-K model predictions; select a mutation to inspect its sequence"}
                      </caption>
                      <thead>
                        <tr>
                          <th>{zh ? "突变" : "Mutation"}</th>
                          <th>{zh ? "综合分" : "Score"}</th>
                          <th>log₁₀ kcat</th>
                          <th>pH</th>
                          <th>ΔΔG</th>
                        </tr>
                      </thead>
                      <tbody>
                        {selected.map((c, i) => (
                          <tr
                            key={c.mutation}
                            className={
                              current?.mutation === c.mutation ? "selected" : ""
                            }
                          >
                            <th scope="row">
                              <button
                                aria-pressed={current?.mutation === c.mutation}
                                onClick={() => setPicked(c.mutation)}
                              >
                                <span>{i + 1}</span> {c.mutation}
                              </button>
                            </th>
                            <td>{number(c.final_score)}</td>
                            <td>{number(c.activity_log10, 2)}</td>
                            <td>{number(c.ph_opt, 2)}</td>
                            <td>{number(c.ddg, 2)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  )}
                  {!selected.length && (
                    <p className="dim">
                      {zh ? "尚无筛选结果" : "No candidates selected yet."}
                    </p>
                  )}
                </div>
                {current && (
                  <div className="sequence-caption">
                    <strong>{current.mutation}</strong>
                    {!selected.some(
                      (row) => row.mutation === current.mutation,
                    ) && (
                      <span>
                        {zh
                          ? "库内序列 · 未入选 Top-K"
                          : "Library sequence · outside Top-K"}{" "}
                        · {zh ? "综合分" : "Score"}{" "}
                        {number(current.final_score)}
                      </span>
                    )}
                    <span>
                      {zh
                        ? "深蓝 = 相对野生型的替换残基"
                        : "Dark blue = substitutions relative to wild type"}
                    </span>
                  </div>
                )}
                <div
                  className="selected-sequence"
                  ref={sequenceView}
                  aria-label={
                    zh
                      ? "突变残基以深蓝色标记"
                      : "Mutated residues highlighted in dark blue"
                  }
                >
                  {current?.mutant_sequence &&
                    Array.from(current.mutant_sequence).map((aa, i) =>
                      aa !== run.wtSequence[i] ? (
                        <mark
                          key={i}
                          title={`${run.wtSequence[i]}${i + 1}${aa}`}
                        >
                          {aa}
                        </mark>
                      ) : (
                        <span key={i}>{aa}</span>
                      ),
                    )}
                </div>
              </div>
            </div>
          </section>
        </div>
      </main>
      <nav
        className="process-nav"
        aria-label={zh ? "流程阶段" : "Process stages"}
      >
        {[
          zh ? "建库" : "Library",
          zh ? "评分" : "Screening",
          zh ? "结果" : "Output",
        ].map((label, i) => (
          <button
            key={label}
            className={view === i + 1 ? "active" : ""}
            onClick={() => go(i + 1)}
            aria-current={view === i + 1 ? "step" : undefined}
          >
            <span />
            {label}
          </button>
        ))}
      </nav>
      {(error || run.error || notice || run.reconnecting) && (
        <div
          className="process-notice"
          role={error || run.error ? "alert" : "status"}
        >
          {error ||
            run.error ||
            notice ||
            (zh ? "实时连接重连中…" : "Reconnecting to live updates…")}
          <button
            className="ghost"
            aria-label={zh ? "关闭通知" : "Dismiss notice"}
            onClick={() => {
              setError("");
              setNotice("");
              if (run.error) run.retry();
            }}
          >
            ×
          </button>
        </div>
      )}
      <ChatPanel
        sequence={sequence}
        jobId={jobId}
        onChange={run.retry}
        phase={phase}
        status={run.meta?.status}
      />
    </>
  );
}

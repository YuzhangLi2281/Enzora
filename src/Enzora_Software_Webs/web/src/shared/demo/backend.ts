/** Repeatable browser workspace: the permanent carbonic-anhydrase case uses
 * saved model predictions and UMAP coordinates. Custom inputs use labelled
 * illustrative simulation. Only Assistant conversation reaches the public API. */

import { applyMutation, cleanSequence, translateNucleotide } from "./sequence";
import savedCase from "../../../../../../resource/demo-case.json";
import { candidateScoreEvidence, rerankCandidates } from "../../features/results/rerank";
import type {
  AtlasRecord,
  ProteinStructure,
  BioArtifact,
  Candidate,
  ClassifyResult,
  CodonOptimizeResponse,
  HostInfo,
  LookupCandidate,
  LookupResolved,
  ParseResult,
  RunEvent,
  RunMeta,
  RunRequest,
  RunResult,
  ServerSettings,
  ToolInfo,
  TranslationResult,
} from "../api/contracts";

const WT = "MRNSTKTSLLLAGLCTAAQMVFVTHASAAAMNTRNTRSKQRAPFGVSSSVKPRLDLIEQAPNPAYDRHPACITLPERTCRHPLTDLEANEQLGRCEDSVKNRFDRVIPFLQVVAGIPLGLDYVTRVQELAQSSLGHTLPEELLKDNWISGHNLKGIFGYATAKALTAATEQFSRKIMSEKDDSASAIGFFLDCGFHAVDISPCADGRLKGLLPYILRLPLTAFTYRKAYAGSMFDIEDDLAQWEKNELRRYREGVPNTADQPTRYLKIAVYHFSTSDPTHSGCAAHGSNDRAALEAALTQLMKFREAVENAHCCGASIDILLIGVDTDTDAIRVHIPDSKGFLNPYRYVDNTVTYAQTLHLAPDEARVIIHEAILNANRSDGWAKGNGVASEGMRRFIGQLLINNLSQIDYVVNRHGGRYPPNDIGHAERYISVGDGFDEVQIRNLAYYAHLDTVEENAIDVDVGIKIFTKLNLSRGLPIPIAIHYRYDPNVPGSRERTVVKARRIYNAIKERFSSLDEQNLLQFRLSVQAQDIGSPIEEVASA";
export { WT as DEMO_ENZYME_SEQUENCE };
export const DEMO_CASE_ID = "demo-recorded-1000";
const recorded = savedCase as unknown as {request:RunRequest;pool:Candidate[];umap:{method:string;points:Record<string,unknown>[]};generation:Record<string,unknown>;analysis:string;report:string;balance_decision:{reason:string};structure:ProteinStructure;atlas:AtlasRecord};

// Kyte-Doolittle hydropathy (subset used by the demo), residue volumes, charge.
const HYDRO: Record<string, number> = {
  A: 1.8,
  R: -4.5,
  N: -3.5,
  D: -3.5,
  C: 2.5,
  Q: -3.5,
  E: -3.5,
  G: -0.4,
  H: -3.2,
  I: 4.5,
  L: 3.8,
  K: -3.9,
  M: 1.9,
  F: 2.8,
  P: -1.6,
  S: -0.8,
  T: -0.7,
  W: -0.9,
  Y: -1.3,
  V: 4.2,
};
const VOLUME: Record<string, number> = {
  A: 88.6,
  R: 173.4,
  N: 114.1,
  D: 111.1,
  C: 108.5,
  Q: 143.8,
  E: 138.4,
  G: 60.1,
  H: 153.2,
  I: 166.7,
  L: 166.7,
  K: 168.6,
  M: 162.9,
  F: 189.9,
  P: 112.7,
  S: 89,
  T: 116.1,
  W: 227.8,
  Y: 193.6,
  V: 140,
};
const CHARGE: Record<string, number> = { D: -1, E: -1, K: 1, R: 1 };
const CLASS: Record<string, string> = {
  A: "nonpolar",
  V: "nonpolar",
  L: "nonpolar",
  I: "nonpolar",
  M: "nonpolar",
  G: "nonpolar",
  P: "nonpolar",
  F: "nonpolar",
  W: "nonpolar",
  S: "polar_uncharged",
  T: "polar_uncharged",
  C: "polar_uncharged",
  N: "polar_uncharged",
  Q: "polar_uncharged",
  Y: "polar_uncharged",
  K: "positive",
  R: "positive",
  H: "positive",
  D: "negative",
  E: "negative",
};

const E_COLI_CODONS: Record<string, string> = {
  A: "GCG",
  C: "TGC",
  D: "GAT",
  E: "GAA",
  F: "TTT",
  G: "GGC",
  H: "CAT",
  I: "ATT",
  K: "AAA",
  L: "CTG",
  M: "ATG",
  N: "AAC",
  P: "CCG",
  Q: "CAG",
  R: "CGC",
  S: "AGC",
  T: "ACC",
  V: "GTG",
  W: "TGG",
  Y: "TAT",
};
const YEAST_CODONS: Record<string, string> = {
  A: "GCT",
  C: "TGT",
  D: "GAT",
  E: "GAA",
  F: "TTT",
  G: "GGT",
  H: "CAT",
  I: "ATT",
  K: "AAA",
  L: "TTG",
  M: "ATG",
  N: "AAT",
  P: "CCA",
  Q: "CAA",
  R: "AGA",
  S: "TCT",
  T: "ACT",
  V: "GTT",
  W: "TGG",
  Y: "TAT",
};
const BACILLUS_CODONS: Record<string, string> = {
  ...E_COLI_CODONS,
  A: "GCA",
  N: "AAT",
  Q: "CAA",
  R: "AGA",
  S: "TCA",
  T: "ACA",
  V: "GTT",
};
const CODON_MAPS: Record<string, Record<string, string>> = {
  e_coli: E_COLI_CODONS,
  s_cerevisiae: YEAST_CODONS,
  b_subtilis: BACILLUS_CODONS,
};

const STAGES = [
  "resolve_user_request",
  "vesm_library_construction",
  "phenom_activity_prediction",
  "ephod_ph_prediction",
  "unistab_stability_prediction",
  "personalized_filter_and_rank",
  "point_mutation_feature_analysis",
  "codon_optimization",
  "visualize_results",
  "gemini_result_analysis",
  "write_markdown_report",
];

type RawCandidate = [string, number, number, number, boolean, string?];
const RAW_CANDIDATES: RawCandidate[] = [
  ["A3V", 1.42, 7.6, -1.2, true],
  ["A3L", 1.55, 7.9, -0.6, true],
  ["Y6F", 0.98, 7.2, -2.1, true],
  ["I8V", 1.21, 7.8, -0.9, true],
  ["K10R", 0.85, 8.3, -0.4, true],
  ["Q12E", 1.33, 6.9, -1.4, true],
  ["R13K", 1.02, 8.6, 0.2, true],
  ["F16Y", 0.76, 7.4, -2.6, true],
  ["F16W", 0.52, 7.1, 1.8, false, "UniStab ddG above threshold"],
  ["V17I", 1.18, 7.7, -1.1, true],
  ["S19T", 1.05, 7.5, -0.8, true],
  ["K21R", 0.91, 8.1, -0.3, true],
];
const BLOSUM_DEMO: Record<string, number> = {
  A3V: 0,
  A3L: -1,
  Y6F: 3,
  I8V: 3,
  K10R: 2,
  Q12E: 2,
  R13K: 2,
  F16Y: 3,
  F16W: 1,
  V17I: 3,
  S19T: 1,
  K21R: 2,
};

function buildCandidate(
  [exampleMutation, activity, ph, ddg, passed, reason]: RawCandidate,
  sequence = WT,
): Candidate {
  const position = Number(exampleMutation.match(/\d+/)?.[0]);
  const original = sequence[position - 1];
  let substitute = exampleMutation.slice(-1);
  if (original === substitute) substitute = original === "A" ? "V" : "A";
  const mutation = `${original}${position}${substitute}`;
  const match = mutation.match(/^([A-Z])(\d+)([A-Z])$/);
  const wt = match?.[1] ?? "";
  const mt = match?.[3] ?? "";
  return {
    mutation,
    position: Number(match?.[2]),
    wt_aa: wt,
    mutant_aa: mt,
    activity_log10: activity,
    ph_opt: ph,
    ddg,
    delta_ph_from_wt: Number((ph - 7.5).toFixed(2)),
    passed,
    reasons: reason ?? "",
    blosum62: BLOSUM_DEMO[mutation] ?? null,
    conservative_substitution:
      BLOSUM_DEMO[mutation] == null ? null : BLOSUM_DEMO[mutation] >= 0,
    residue_class_change: `${CLASS[wt] ?? "?"}->${CLASS[mt] ?? "?"}`,
    delta_hydrophobicity: Number(
      ((HYDRO[mt] ?? 0) - (HYDRO[wt] ?? 0)).toFixed(2),
    ),
    delta_volume: Number(((VOLUME[mt] ?? 0) - (VOLUME[wt] ?? 0)).toFixed(1)),
    delta_charge: Number(((CHARGE[mt] ?? 0) - (CHARGE[wt] ?? 0)).toFixed(1)),
    mutant_sequence: applyMutation(sequence, mutation),
    activity_component: 0,
    ph_component: 0,
    stability_component: 0,
    final_score: 0,
    pareto_optimal: false,
  };
}

function rankedPool(
  weights: { ph: number; activity: number; stability: number },
  request: RunRequest,
): Candidate[] {
  if (request.sequence === WT && Object.keys(request.mutation?.counts ?? {}).join() === '1' && Number(request.mutation?.counts[1]) <= 1000) {
    request.selection = {...request.selection, reference:recorded.request.selection.reference,
      confidence:request.selection.confidence ?? recorded.request.selection.confidence,
      score_policy:{...recorded.request.selection.score_policy!,target_ph:request.selection.target_ph}};
    const rows = recorded.pool.slice(0,request.mutation!.counts[1]).map(row=>{
      const candidate = {...row};
      const evidence = candidateScoreEvidence(candidate,{...request.selection,weights})!;
      candidate.final_score=evidence.score;
      candidate.passed=(request.selection.min_activity_log10===null||Number(candidate.activity_log10)>=request.selection.min_activity_log10)
        &&(request.selection.target_ph===null||request.selection.ph_tolerance===null||Math.abs(Number(candidate.ph_opt)-request.selection.target_ph)<=request.selection.ph_tolerance)
        &&(request.selection.stability_operator==='none'||request.selection.stability_threshold===null||
          (request.selection.stability_operator==='le'?Number(candidate.ddg)<=request.selection.stability_threshold:Number(candidate.ddg)>=request.selection.stability_threshold));
      return candidate;
    });
    return rows.sort((a,b)=>Number(b.passed)-Number(a.passed)||Number(b.final_score)-Number(a.final_score));
  }
  const requested = Object.values(
    request.mutation?.counts ?? { 1: RAW_CANDIDATES.length },
  ).reduce((a, b) => a + b, 0);
  const raw: Candidate[] = [];
  const alphabet = "ACDEFGHIKLMNPQRSTVWY";
  const positions =
    request.candidate.allowed_positions ??
    Array.from({ length: request.sequence.length }, (_, i) => i + 1);
  for (const [orderText, count] of Object.entries(
    request.mutation?.counts ?? { 1: requested },
  )) {
    const order = Number(orderText);
    const seen = new Set<string>();
    for (let index = 0; seen.size < count && index < count * 100; index++) {
      const pos = positions[index % positions.length];
      if (request.candidate.excluded_positions.includes(pos)) continue;
      const aa = alphabet[Math.floor(index / positions.length) % 20];
      if (aa === request.sequence[pos - 1]) continue;
      const substitutions = [`${request.sequence[pos - 1]}${pos}${aa}`];
      for (let j = 1; j < order; j++) {
        const extra = positions[(index + j) % positions.length];
        if (
          extra === pos ||
          request.candidate.excluded_positions.includes(extra)
        )
          break;
        const original = request.sequence[extra - 1];
        substitutions.push(
          `${original}${extra}${alphabet[(alphabet.indexOf(original) + j) % 20]}`
        );
      }
      if (substitutions.length !== order) continue;
      const mutation = substitutions.join(":");
      if (seen.has(mutation)) continue;
      seen.add(mutation);
      const row = buildCandidate(
        [
          substitutions[0],
          1 + Math.sin(index * 0.37) * 0.8,
          7 + Math.sin(index * 0.19),
          Math.cos(index * 0.11) * 2 - 0.8,
          true,
        ],
        request.sequence
      );
      row.mutation = mutation;
      let seq = request.sequence;
      for (const sub of substitutions) seq = applyMutation(seq, sub);
      row.mutant_sequence = seq;
      raw.push(row);
    }
    if (seen.size !== count)
      throw new Error(
        "Demo mutation quota exceeds the available design space."
      );
  }
  for (const row of raw) {
    row.passed =
      (request.selection.min_activity_log10 === null ||
        Number(row.activity_log10) >= request.selection.min_activity_log10) &&
      (request.selection.target_ph === null ||
        request.selection.ph_tolerance === null ||
        Math.abs(Number(row.ph_opt) - request.selection.target_ph) <=
          request.selection.ph_tolerance);
  }
  const ranked = rerankCandidates(raw, weights, request.selection);
  // deterministic Pareto demo: mark top performers on the 3 components
  const eligible = ranked.filter((c) => c.passed);
  const paretoIdx = new Set<number>();
  for (let i = 0; i < eligible.length; i++) {
    let dominated = false;
    for (let j = 0; j < eligible.length; j++) {
      if (i === j) continue;
      const a = eligible[i];
      const b = eligible[j];
      const aV = [
        a.activity_component ?? 0,
        a.ph_component ?? 0,
        a.stability_component ?? 0,
      ];
      const bV = [
        b.activity_component ?? 0,
        b.ph_component ?? 0,
        b.stability_component ?? 0,
      ];
      if (bV.every((v, k) => v >= aV[k]) && bV.some((v, k) => v > aV[k])) {
        dominated = true;
        break;
      }
    }
    if (!dominated) paretoIdx.add(i);
  }
  eligible.forEach((c, i) => {
    c.pareto_optimal = paretoIdx.has(i);
  });
  return ranked;
}

const DEFAULT_WEIGHTS = { ph: 0.3, activity: 0.4, stability: 0.3 };

// ------------------------------------------------------------------ demo jobs
interface DemoJob {
  request: RunRequest;
  meta: RunMeta;
  result: RunResult;
  pool: Candidate[];
  weights: { ph: number; activity: number; stability: number };
  recorded?: boolean;
}
const jobs = new Map<string, DemoJob>();
try {
  const saved = JSON.parse(
    sessionStorage.getItem("fea_demo_jobs") ?? "[]",
  ) as DemoJob[];
  for (const job of saved.slice(-3)) {
    if (!job.request || !job.meta?.id || !Array.isArray(job.pool)) continue;
    // A browser reload stops local timers. Restore an honest terminal state.
    if (job.meta.status === "queued" || job.meta.status === "running") {
      job.meta.status = "cancelled";
      job.meta.error =
        "Demo interrupted by page reload. Start a new demo to replay the workflow.";
    }
    jobs.set(job.meta.id, job);
  }
} catch {
  /* A private browser or an older demo cache can start fresh. */
}
function persistJobs() {
  for(const [id,job]of jobs)if(id!==DEMO_CASE_ID&&job.meta.status!=='running'&&job.meta.status!=='queued'&&jobs.size>4){jobs.delete(id);listeners.delete(id);}

  try {
    sessionStorage.setItem(
      "fea_demo_jobs",
      JSON.stringify([...jobs.values()].filter(job=>!job.recorded).slice(-3))
    );
  } catch {
    /* optional cache */
  }
}
const listeners = new Map<string, ((evt: RunEvent) => void)[]>();
let jobCounter = 0;


function metaFor(
  jobId: string,
  request: RunRequest,
  status: RunMeta["status"],
): RunMeta {
  return {
    id: jobId,
    status,
    created_at: Date.now() / 1000,
    started_at: null,
    finished_at: null,
    error: null,
    force_cpu: request.runtime?.force_cpu ?? false,
    config_summary: {
      sequence_name: request.sequence_name || "enzyme",
      sequence_length: request.sequence.length,
      user_request: request.user_request,
      report_language: request.report?.language ?? "en",
    },
    stages: [],
    run_dir: `jobs/${jobId}/outputs`,
    result: null,
  };
}

function emit(
  jobId: string,
  event: string,
  data: Record<string, unknown>,
): void {
  persistJobs();
  (listeners.get(jobId) ?? []).forEach((cb) => cb({ event, data }));
}

function codonOptimizeFor(
  protein: string,
  keys: string[],
): CodonOptimizeResponse {
  const results: CodonOptimizeResponse["results"] = [];
  const errors: CodonOptimizeResponse["errors"] = [];
  for (const key of keys) {
    const map = CODON_MAPS[key];
    if (!map) {
      errors.push({
        organism_key: key,
        error: "demo codon table not included for this host",
      });
      continue;
    }
    if (!protein || [...protein].some((aa) => !map[aa]))
      throw new Error("Canonical protein sequence required.");
    const cds = [...protein].map((aa) => map[aa]).join("");
    const withStop = cds + "TAA";
    results.push({
      organism_key: key,
      scientific_name:
        key === "e_coli"
          ? "Escherichia coli K-12 W3110"
          : key === "b_subtilis"
            ? "Bacillus subtilis"
            : "Saccharomyces cerevisiae",
      display_name_zh:
        key === "e_coli"
          ? "大肠杆菌"
          : key === "b_subtilis"
            ? "枯草芽孢杆菌"
            : "酿酒酵母",
      taxid:
        key === "e_coli" ? "316407" : key === "b_subtilis" ? "1423" : "4932",
      codon_source: "builtin_kazusa",
      optimized_cds: cds,
      optimized_cds_with_stop: withStop,
      stop_codon: "TAA",
      protein_length: protein.length,
      cds_length: cds.length,
      gc_fraction: cds ? (cds.match(/[GC]/g)?.length ?? 0) / cds.length : 0,
    });
  }
  return {
    results,
    errors,
    method: "deterministic_most_frequent_synonymous_codon",
  };
}

function demoResult(job: DemoJob, request: RunRequest): RunResult {
  const selected = job.pool
    .filter((c) => c.passed)
    .slice(0, request.selection.top_k ?? 10);
  const wantCodon =
    /codon|密码子/i.test(request.user_request ?? "") ||
    /codon|密码子/i.test(request.sequence_name ?? "");
  return {
    status: "success",
    stages: [],
    selected,
    candidate_pool_size: job.pool.length,
    wt_ph: job.recorded ? recorded.request.selection.reference!.ph_opt : 7.5,
    markdown_report: job.recorded ? `# Enzora saved-case replay\n\nSaved model predictions from run 626ac5ab7ce0; current selection and ranking are replayed locally, not new inference.\n\nSelected: ${selected.map(row=>row.mutation).join(", ")}.\n\nAlkaline stability at pH 8–10 has not been experimentally verified.` : `# Enzora Demo Report\n\nDemo run for **${request.sequence_name}**. All numbers are illustrative UI preview data, not real predictions.`,
    figures: [
      "01_final_score.svg",
      "02_activity_vs_stability.svg",
      "03_property_heatmap.svg",
      "04_mutation_sequence_map.svg",
      "05_pareto_front.svg",
    ],
    analysis_file: "analysis.txt",
    codon_optimization: wantCodon
      ? {
          enabled: true,
          status: "success",
          host_count: 1,
          count: (selected.length + 1) * 1,
          organisms: ["Escherichia coli K-12 W3110"],
          resolved_hosts: [
            {
              organism_key: "e_coli",
              scientific_name: "Escherichia coli K-12 W3110",
              display_name_zh: "大肠杆菌",
              taxid: "316407",
              codon_source: "builtin_kazusa",
            },
          ],
          failed_hosts: [],
          warnings: [],
        }
      : {
          enabled: false,
          status: "not_requested",
          host_count: 0,
          count: 0,
          resolved_hosts: [],
          failed_hosts: [],
          warnings: [],
        },
  };
}

/** Preview follows the same explicit Next barriers as the real run. */
function playPhase(job: DemoJob, phase: number) {
  const id = job.meta.id;
  job.meta.stages = job.meta.stages.filter(stage=>!((phase===1&&stage.index<=1)||(phase===2&&stage.index>=2&&stage.index<=4)||(phase===3&&stage.index>=5)));
  job.meta.status = "running";
  job.meta.started_at ??= Date.now() / 1000;
  job.meta.progress = {
    phase,
    status: "running",
    saved_sequences: phase === 1 ? 0 : job.pool.length,
    target_sequences: job.pool.length,
  };
  if(phase===1)job.meta.generation={saved_sequences:0,target_sequences:job.pool.length};
  emit(id, "progress", job.meta.progress);
  const indices =
    phase === 1 ? [0, 1] : phase === 2 ? [2, 3, 4] : [5, 6, 7, 8, 9, 10];
  if (phase === 1)
    for (let i = 1; i <= 10; i++)
      void (
        setTimeout(() => {
          if (job.meta.status !== "running") return;
          job.meta.progress = {
            phase,
            status: "saving",
            saved_sequences: Math.round((job.pool.length * i) / 10),
            target_sequences: job.pool.length,
          };
          if(job.recorded) {
            const landscape=recorded.generation.llr_landscape as {bins:number[];frames:{saved:number}[]};
            job.meta.generation=job.pool.length===1000?{...recorded.generation,saved_sequences:job.meta.progress.saved_sequences,
              llr_landscape:{...landscape,frames:landscape.frames.filter(frame=>frame.saved<=job.meta.progress!.saved_sequences!)}}:
              {saved_sequences:job.meta.progress.saved_sequences,target_sequences:job.pool.length};
          }
          emit(id, "progress", job.meta.progress);
        }, i * 330)
      );
  indices.forEach((index, i) =>
    void (
      setTimeout(
        () => {
          if (job.meta.status !== "running") return;
          job.meta.stages.push({
            index,
            stage: STAGES[index],
            finished_at: Date.now() / 1000,
            seconds: 1,
          });
          emit(id, "stage_completed", { index, stage: STAGES[index] });
        },
        phase === 1 ? 3600 + i * 100 : (i + 1) * 800,
      )
    )
  );
  void (
    setTimeout(
      () => {
        if (job.meta.status !== "running") return;
        if(job.recorded&&job.pool.length===1000) job.meta.generation={...recorded.generation};
        if (phase < 3) {
          job.meta.progress = {
            phase,
            status: "awaiting_next",
            next_phase: phase + 1,
            saved_sequences: job.pool.length,
            target_sequences: job.pool.length,
          };
          emit(id, "progress", job.meta.progress);
        } else {
          job.result = demoResult(job, job.request);
          job.meta.status = "success";
          job.meta.result = job.result;
          job.meta.finished_at = Date.now() / 1000;
          job.meta.progress = {
            phase: 3,
            status: "success",
            saved_sequences: job.pool.length,
            target_sequences: job.pool.length,
          };
          emit(id, "completed", { result: job.result });
        }
      },
      phase === 1 ? 4000 : indices.length * 800 + 300,
    )
  );
}

const ANALYSIS_TEXT = `Demo analysis: compare the illustrative activity, pH and stability scores in the candidate table. Adjusting the weights changes the local ranking without rerunning models. These values demonstrate interface behavior only; they are not biological predictions or experimental evidence.`;

// This permanent baseline is reconstructed on every page load, independent of session cache.
const baselineRequest=JSON.parse(JSON.stringify(recorded.request)) as RunRequest;
const baseline:DemoJob={request:baselineRequest,meta:metaFor(DEMO_CASE_ID,baselineRequest,'success'),
 pool:recorded.pool.map(row=>({...row})),weights:{...baselineRequest.selection.weights},recorded:true,result:null as unknown as RunResult};
baseline.result=demoResult(baseline,baselineRequest);
baseline.meta={...baseline.meta,result:baseline.result,applied_selection:baselineRequest.selection,
 generation:{...recorded.generation},progress:{phase:3,status:'success',saved_sequences:1000,target_sequences:1000},
 stages:STAGES.map((stage,index)=>({stage,index,finished_at:Date.now()/1000,seconds:0}))};
jobs.set(DEMO_CASE_ID,baseline);

export const demo = {
  submitRun(request: RunRequest): Promise<{ job_id: string }> {
    request=JSON.parse(JSON.stringify(request)) as RunRequest;
    jobCounter += 1;
    const jobId = `demo-${Date.now()}-${jobCounter}`;
    const weights = { ...(request.selection?.weights ?? DEFAULT_WEIGHTS) };
    const pool = rankedPool(weights, request);
    const job: DemoJob = {
      request,
      meta: metaFor(jobId, request, "queued"),
      pool,
      recorded:request.sequence===WT && pool.every(row=>row.score_normalization==="input_reference_three_model_llm_v3"),
      weights,
      result: null as unknown as RunResult,
    };
    job.result = demoResult(job, request);
    job.meta.applied_selection=job.request.selection;
    jobs.set(jobId, job);
    persistJobs();
    if (request.runtime.phase_control) {
      playPhase(job, 1);
      return Promise.resolve({ job_id: jobId });
    }
    // Legacy automatic preview is used only when phase control is explicitly absent.
    void (
      setTimeout(() => {
        if (job.meta.status === "cancelled") return;
        job.meta.status = "running";
        job.meta.started_at = Date.now() / 1000;
        emit(jobId, "started", { job_id: jobId });
      }, 600)
    );
    const stageStats: Record<string, Record<string, unknown>> = {
      generate_mutations: { count: request.sequence.length * 19 },
      vesm_library_construction: {
        full_scan: request.sequence.length * 19,
        pool_size: pool.length,
      },
      ephod_ph_prediction: { wt_ph: 7.5, count: 13 },
      unistab_stability_prediction: { count: 12 },
      personalized_filter_and_rank: { ranked: 12, passed: 11 },
      point_mutation_feature_analysis: {
        feature_count: 12,
        pareto_optimal: 4,
        selected: job.result.selected.length,
      },
      visualize_results: { figures: 5 },
      gemini_result_analysis: { enabled: true },
    };
    const startedAt = Date.now() / 1000;
    STAGES.forEach((stage, index) => {
      void (
        setTimeout(
          () => {
            if (job.meta.status === "cancelled") return;
            const finishedAt = Date.now() / 1000;
            job.meta.stages.push({
              index,
              stage,
              started_at: startedAt + index * 1.1,
              finished_at: finishedAt,
              seconds: 1.1,
              stats: stageStats[stage] ?? null,
            });
            emit(jobId, "stage_completed", {
              stage,
              index,
              total: STAGES.length,
              stats: stageStats[stage] ?? null,
            });
          },
          1200 + index * 1100,
        )
      );
    });
    void (
      setTimeout(
        () =>
          emit(jobId, "log_line", {
            source: "phenom",
            line: "[demo] scanning 684 single substitutions (GPU)",
          }),
        2400,
      )
    );
    void (
      setTimeout(
        () =>
          emit(jobId, "log_line", {
            source: "unistab",
            line: "[demo] predicting ddG, batch=1",
          }),
        5600,
      )
    );
    void (
      setTimeout(
        () => {
          if (job.meta.status === "cancelled") return;
          job.meta.status = "success";
          job.meta.finished_at = Date.now() / 1000;
          job.meta.result = job.result;
          emit(jobId, "completed", { result: job.result });
        },
        1300 + STAGES.length * 1100,
      )
    );
    return Promise.resolve({ job_id: jobId });
  },

  replayRun(id:string,phase:number,selection:RunRequest['selection']):{job_id:string} {
    const original=jobs.get(id);
    if(!original||original.meta.status!=='success'||![2,3].includes(phase))throw new Error('Choose a completed Demo case and phase 2 or 3.');
    const request=JSON.parse(JSON.stringify(original.request)) as RunRequest;
    request.selection=JSON.parse(JSON.stringify(selection));
    const job_id=`demo-replay-${Date.now()}-${++jobCounter}`;
    const pool=rankedPool(request.selection.weights,request);
    const replay:DemoJob={request,pool,weights:{...request.selection.weights},recorded:original.recorded,
      meta:metaFor(job_id,request,'running'),result:null as unknown as RunResult};
    replay.meta.stages=original.meta.stages.filter(stage=>stage.index<(phase===2?2:5)).map(stage=>({...stage}));
    replay.meta.applied_selection=request.selection;
    replay.result=demoResult(replay,request);
    jobs.set(job_id,replay);playPhase(replay,phase);
    return {job_id};
  },
  advanceRun(id: string, phase: number) {
    const job = jobs.get(id);
    if (
      !job ||
      job.meta.status !== "running" ||
      job.meta.progress?.status !== "awaiting_next" ||
      job.meta.progress.next_phase !== phase
    )
      throw new Error("This phase is not ready.");
    playPhase(job, phase);
  },
  balanceRun(
    id: string,
    weights: DemoJob["weights"],
    confidence: DemoJob["weights"],
  ) {
    const job = jobs.get(id);
    if (
      !job ||
      job.meta.status !== "running" ||
      (job.meta.progress?.phase ?? 3) > 2
    )
      throw new Error("Balancing is closed for this run.");
    const values = Object.keys(weights) as (keyof typeof weights)[];
    if (
      values.some(
        (k) =>
          !Number.isFinite(weights[k]) ||
          weights[k] < 0 || weights[k] > 1 ||
          !Number.isFinite(confidence[k]) ||
          confidence[k] < 0 ||
          confidence[k] > 1,
      )
    )
      throw new Error("Invalid balance parameters.");
    if (Math.abs(values.reduce((n,k)=>n+weights[k],0)-1)>1e-8) throw new Error("Preference weights must sum to 1.");
    const total = values.reduce((n, k) => n + weights[k] * confidence[k], 0);
    if (
      total <= 0 ||
      (weights.ph > 0 && job.request.selection.target_ph === null)
    )
      throw new Error("Set a pH target and a positive effective weight.");
    job.request.selection.weights = {...weights};
    job.request.selection.confidence = {...confidence};
    const effective_weights = {
      ph: (weights.ph * confidence.ph) / total,
      activity: (weights.activity * confidence.activity) / total,
      stability: (weights.stability * confidence.stability) / total,
    };
    job.weights={...weights};
    job.pool = rankedPool(weights, job.request);
    job.meta.applied_selection=job.request.selection;
    persistJobs();
    return { effective_weights };
  },
  listRuns(): RunMeta[] {
    return [...jobs.values()].map((j) => j.meta).reverse();
  },

  readArtifact(jobId: string, rel: string): string {
    const job = jobs.get(jobId);
    if (!job) throw new Error("Unknown demo job");
    if (rel === "00_resolved_request.json")
      return JSON.stringify({
        ...job.request,
        resolved_candidate: job.request.candidate,
        resolved_selection: job.request.selection,
      });
    if(rel==='llm_balance_decisions.jsonl')return JSON.stringify(job.recorded?recorded.balance_decision:{reason:'Local illustrative replay; no new model inference.'});
    if(rel==='umap.json' && job.recorded) {
      const byMutation=new Map(job.pool.map((row,index)=>[row.mutation,{row,index}]));
      return JSON.stringify({...recorded.umap,points:recorded.umap.points.filter(point=>byMutation.has(String(point.mutation))).map(point=>{
        const {row,index}=byMutation.get(String(point.mutation))!;
        return {...point,score:row.final_score,rank:row.passed&&index<job.request.selection.top_k?index+1:null};
      })});
    }
    if (rel === "umap.json")
      return JSON.stringify({
        method: "Demo projection (not UMAP)",
        points: job.pool.map((row, i) => {
          // Illustrative branched projection, never substituted for real UMAP coordinates.
          const branch = i % 19,
            t = (Math.floor(i / 19) + 0.5) / Math.ceil(job.pool.length / 19);
          const angle = branch * 2.39996 + 0.18 * Math.sin(t * 9 + branch);
          const radius = 0.25 + 3.2 * Math.sqrt(t);
          const twig = Math.sin(i * 12.9898) * (0.025 + 0.1 * t);
          return {
            mutation: row.mutation,
            x: Math.cos(angle) * radius + Math.cos(angle + Math.PI / 2) * twig,
            y: Math.sin(angle) * radius + Math.sin(angle + Math.PI / 2) * twig,
            score: row.final_score,
            activity_log10: row.activity_log10,
            rank: i < job.request.selection.top_k ? i + 1 : null,
          };
        }),
      });
    if (rel === "analysis.txt" && job.recorded)return "Saved-case explanation; not new inference.\n"+recorded.analysis;
    if (rel === "analysis.txt")
      return job.request.report.language === "zh"
        ? "演示分析：候选表展示模拟活性、最适 pH 与稳定性分数。配平系数经偏好 × 可信度归一化后参与排序；通过阈值的 Top-K 保留全部替换标记。可导出蛋白 FASTA 与宿主偏好的 CDS。此处数据仅用于检验交互，不构成模型预测或实验依据。"
        : ANALYSIS_TEXT;
    if (rel === "report.md" || rel === "report_en.md") return job.result.markdown_report ?? "";
    throw new Error("Artifact not available in demo mode");
  },

  getRun(jobId: string): RunMeta {
    const job = jobs.get(jobId);
    if (!job) throw new Error("unknown demo job");
    return job.meta;
  },

  cancelRun(jobId: string): void {
    const job = jobs.get(jobId);
    if (
      job &&
      (job.meta.status === "running" || job.meta.status === "queued")
    ) {
      job.meta.status = "cancelled";
      job.meta.finished_at = Date.now() / 1000;
      emit(jobId, "cancelled", {});
    }
  },

  subscribeRun(
    jobId: string,
    onEvent: (evt: RunEvent) => void,
    onEnd: () => void,
  ): () => void {
    const list = listeners.get(jobId) ?? [];
    list.push(onEvent);
    listeners.set(jobId, list);
    const job = jobs.get(jobId);
    if (job && job.meta.status === "success") {
      setTimeout(
        () => onEvent({ event: "completed", data: { result: job.result } }),
        300
      );
      setTimeout(onEnd, 500);
    }
    return () => {
      const current = listeners.get(jobId) ?? [];
      listeners.set(
        jobId,
        current.filter((cb) => cb !== onEvent)
      );
    };
  },

  getCandidates(
    jobId: string,
    params: {
      pool?: string;
      page?: number;
      page_size?: number;
      sort_by?: string;
      order?: string;
      filter?: string;
      q?: string;
    },
  ): {
    total: number;
    page: number;
    columns: string[];
    rows: Record<string, unknown>[];
  } {
    const job = jobs.get(jobId);
    if (!job) return { total: 0, page: 1, columns: [], rows: [] };
    let rows: Record<string, unknown>[] = job.pool as unknown as Record<
      string,
      unknown
    >[];
    if (
      job.request.runtime.phase_control &&
      (params.pool === "scan" || params.pool === "scores")
    ) {
      const ready = new Set(job.meta.stages.map((stage) => stage.index));
      rows = rows.map((row) => {
        const visible = { ...row };
        for (const [index, fields] of [
          [2, ["activity_log10", "activity_component"]],
          [3, ["ph_opt", "ph_component", "delta_ph_from_wt"]],
          [4, ["ddg", "stability_component"]],
          [5, ["final_score", "passed", "reasons", "pareto_optimal"]],
        ] as const) {
          if (params.pool === "scan" || !ready.has(index))
            for (const field of fields) delete visible[field];
        }
        return visible;
      });
    }
    if (params.filter === "passed")
      rows = rows.filter((r) => String(r.passed) === "true");
    if (params.filter === "pareto")
      rows = rows.filter((r) => String(r.pareto_optimal) === "true");
    if (params.q)
      rows = rows.filter((r) =>
        String(r.mutation)
          .toLowerCase()
          .includes(String(params.q).toLowerCase())
      );
    if (params.sort_by) {
      rows = [...rows].sort((a, b) => {
        const av = a[params.sort_by!];
        const bv = b[params.sort_by!];
        const cmp =
          typeof av === "number" && typeof bv === "number"
            ? av - bv
            : String(av).localeCompare(String(bv));
        return params.order === "asc" ? cmp : -cmp;
      });
    }
    const page = params.page ?? 1;
    const size = params.page_size ?? 50;
    return {
      total: rows.length,
      page,
      columns: Object.keys(rows[0] ?? {}),
      rows: rows.slice((page - 1) * size, page * size),
    };
  },

  rerankRun(
    jobId: string,
    weights: { ph: number; activity: number; stability: number },
  ): Record<string, unknown>[] {
    const job = jobs.get(jobId);
    if (!job) return [];
    return rankedPool(weights,{...job.request,selection:{...job.request.selection,weights}}) as unknown as Record<string,unknown>[];
  },

  classify(text: string, typeOverride?: string | null): ClassifyResult {
    const sampleGoal =
      "Find human carbonic anhydrase II and optimize it near pH 7.5, select 10 candidates.";
    const type =
      typeOverride ||
      (text.trim() === WT
        ? "protein"
        : text.trim().toLowerCase() === "carbonic anhydrase" ||
            text.trim() === sampleGoal
          ? "text"
          : null);
    if (!type)
      throw new Error(
        "AI input recognition needs the backend. In Demo, choose an explicit input type or load the sample."
      );
    const base: ClassifyResult = {
      type: "text",
      sequence: null,
      sequences: [],
      request_text: text,
      fasta_name: null,
      warnings: ["Demo input preview; no AI request was made."],
      translation: null,
      ambiguous: false,
      suggested_type: null,
      records: [],
      interpretation: "Demo input preview",
      requires_confirmation: true,
    };
    if (type === "text")
      return {
        ...base,
        lookup_query: text.trim() === sampleGoal ? "carbonic anhydrase" : text,
      };
    const { body, name } = cleanSequence(text);
    const alphabet =
      type === "protein"
        ? "ACDEFGHIKLMNPQRSTVWY"
        : type === "rna"
          ? "ACGUN"
          : "ACGTN";
    if (!body || [...body].some((letter) => !alphabet.includes(letter)))
      throw new Error("Invalid sequence for the selected input type.");
    const translation = type === "protein" ? null : translateNucleotide(body);
    const record={id:"demo-1",name:name||"Demo sequence",type:type as "protein"|"dna"|"rna",sequence:body};
    return {
      ...base,
      bioartifact:{schema:"enzora.bioartifact/1",records:[record],predictions:{},metadata:{source:"local_demo"},data:{}},
      type: type as ClassifyResult["type"],
      sequence: body,
      sequences: [body],
      request_text: null,
      fasta_name: name,
      translation,
      records: [
        {
          id: "demo-1",
          name: name || "Demo sequence",
          type: type as "protein" | "dna" | "rna",
          sequence: body,
        },
      ],
    };
  },
  translate(sequence: string, _type: string): TranslationResult {
    return translateNucleotide(sequence);
  },

  codonOptimize(protein: string, keys: string[]): CodonOptimizeResponse {
    return codonOptimizeFor(protein, keys);
  },

  lookup(query: string): { candidates: LookupCandidate[]; warning?: string } {
    if(!/碳酸酐酶|carbonic|ca2|P00918/i.test(query))return {candidates:[],warning:'The saved Demo database case is P00918 (human carbonic anhydrase 2).'};
    return {candidates:[{source:'uniprot',accession:'P00918',name:'Carbonic anhydrase 2',gene:'CA2',organism:'Homo sapiens',taxid:9606,length:260,ec_numbers:['4.2.1.1'],reviewed:true,url:'https://www.uniprot.org/uniprotkb/P00918/entry',license:'CC BY 4.0'}]};
  },
  lookupResolve(accession:string,_source:string):LookupResolved {
    if(accession!=='P00918')throw new Error('The saved Demo database record is P00918.');
    return {accession,name:'Carbonic anhydrase 2',organism:'Homo sapiens',taxid:9606,
      sequence:String(recorded.atlas.metadata!.sequence),url:'https://www.uniprot.org/uniprotkb/P00918/entry',license:'CC BY 4.0'};
  },

  atlas(query:string,similar=false):AtlasRecord {
    const q=query.trim();
    if(!["P00918",String(recorded.atlas.metadata?.protein_hash),String(recorded.atlas.metadata?.sequence),"carbonic anhydrase"].some(value=>value.toLowerCase()===q.toLowerCase()))throw new Error("Demo includes the saved P00918 Atlas reference. Enter P00918; use an invitation for other live queries.");
    if(similar)throw new Error("The saved P00918 case has no recorded similarity hits. Turn off similarity to inspect its metadata and SAE features.");
    return JSON.parse(JSON.stringify(recorded.atlas));
  },
  sequenceTool(tool:string,sequence:string,args:Record<string,number>={}):{prepared_sequence?:{start:number;end:number;sequence:string};structure?:ProteinStructure} {
    const protein=cleanSequence(sequence).body;
    if(tool==='inspect_structure') {
      if(protein!==WT)throw new Error("The saved WT structure matches only the 544-residue example. Load the example to inspect it; no new folding occurs in Demo.");
      return {structure:recorded.structure};
    }
    if(tool==='prepare_mature_protein') {
      const {start,end}=args;
      if(!Number.isInteger(start)||!Number.isInteger(end)||start<1||end<start||end>protein.length)throw new Error("Provide a valid inclusive 1-based interval.");
      return {prepared_sequence:{start,end,sequence:protein.slice(start-1,end)}};
    }
    throw new Error("New structure inference requires an invited live workspace.");
  },
  inputFile(filename:string,content:string,type?:string|null):ClassifyResult {
    let text=content;
    if(/\.json$/i.test(filename)) {
      const parsed=JSON.parse(content);const records=Array.isArray(parsed)?parsed:parsed.records??parsed.bioartifact?.records??[parsed];
      if(records.length!==1)throw new Error("Import one sequence at a time in Demo.");
      text=records[0].sequence??records[0].mutant_sequence;
      if(typeof text!=='string')throw new Error("JSON must contain a sequence or mutant_sequence.");
    }else if(/\.csv$/i.test(filename)) {
      const lines=content.trim().split(/\r?\n/);const headers=lines[0].replace(/^\uFEFF/,'').split(',');const index=headers.findIndex(x=>['sequence','mutant_sequence'].includes(x.replace(/"/g,'')));
      if(index<0||lines.length!==2)throw new Error("Import a CSV containing exactly one sequence or mutant_sequence row.");
      text=lines[1].split(',')[index].replace(/^"|"$/g,'');
    }else if(!/\.(fasta|fa|faa|fna|ffn|frn|txt|seq)$/i.test(filename))throw new Error("Demo imports FASTA, sequence text, single-record CSV and JSON. Other formats require live mode.");
    if((text.match(/^>/gm)||[]).length>1)throw new Error("Import one FASTA sequence at a time in Demo.");
    return demo.classify(text,type||'protein');
  },
  exportArtifact(artifact:BioArtifact,format:string):{content:string;warnings:string[]} {
    if(format==='json')return {content:JSON.stringify(artifact,null,2),warnings:[]};
    if(format==='fasta')return {content:artifact.records.map(r=>`>${r.id} ${r.name}\n${r.sequence}\n`).join(''),warnings:[]};
    if(format==='csv')return {content:'id,name,type,sequence\n'+artifact.records.map(r=>[r.id,r.name,r.type,r.sequence].map(v=>'"'+v.replace(/"/g,'""')+'"').join(',')).join('\n'),warnings:[]};
    throw new Error("Choose FASTA, CSV or JSON for local Demo export.");
  },

  tools(): ToolInfo[] {
    const builtin: [string, string, string[]][] = [
      [
        "phenom",
        "Mutation PHENOM log10 kcat provider",
        ["activity_prediction", "activity_ranking"],
      ],
      ["ephod", "Protein optimum pH prediction provider", ["ph_prediction"]],
      [
        "unistab",
        "Mutation stability prediction provider",
        ["stability_prediction"],
      ],
      [
        "codon_optimizer",
        "Multi-host synonymous codon optimization provider",
        ["codon_optimization"],
      ],
      [
        "sequence_translate",
        "DNA/mRNA to protein translation (plugin)",
        ["sequence_translation"],
      ],
    ];
    return builtin.map(([name, description, capabilities]) => ({
      name,
      version: "1.0",
      description,
      capabilities,
      inputs:
        name === "phenom"
          ? ["protein_sequence", "mutation_list"]
          : name === "sequence_translate"
            ? ["nucleotide_sequence", "sequence_type"]
            : ["protein_sequence"],
      outputs:
        name === "phenom"
          ? ["activity_predictions"]
          : name === "ephod"
            ? ["ph_predictions"]
            : name === "unistab"
              ? ["stability_predictions"]
              : name === "sequence_translate"
                ? ["protein_sequence", "translation_metadata"]
                : ["optimized_cds_set"],
      priority: 100,
      properties: { offline: true, gpu: name !== "ephod", batch: true },
      health: { available: true, details: [] },
    }));
  },

  health(kind: string): Record<string, unknown> {
    if (kind === "check-paths") return { ok: true, errors: [] };
    if (kind === "check-gpu")
      return {
        paths: { ok: true, details: [] },
        ephod: { ok: true, details: "demo" },
        phenom: { ok: true, details: "demo" },
        unistab: { ok: true, details: "demo" },
      };
    return {
      passed: true,
      mutation: "A3V",
      phenom_activity_log10: 1.42,
      ephod_wt_ph: 7.5,
      ephod_mutant_ph: 7.6,
      unistab_ddg: -1.2,
    };
  },

  settings(): ServerSettings {
    return {
      models: {
        ephod: {
          repo: "../external/EpHod",
          python: "../external/EpHod/.env/bin/python",
          enabled: true,
        },
        phenom: {
          repo: "../src/Enzora_Screening/PheNom",
          python: "python3",
          model_dir: "../model_weights/enzora_screen/PheNom",
          esmc_model_dir: "../model_weights/enzora_screen/ESMC",
          molt5_model_dir: "../model_weights/enzora_screen/MolT5",
        },
        unistab: {
          repo: "../external/UniStab",
          python: "../external/UniStab/.env/bin/python",
          checkpoint: "../external/UniStab/checkpoints/model.ckpt",
          enabled: true,
        },
      },
      llm: {
        base_url: "https://api.minimax.cn/v1",
        parameter_model: "MiniMax-M2.5",
        analysis_model: "MiniMax-M2.5",
      },
      runtime: {
        gpu_id: 0,
        timeout_seconds: 7200,
        cache_dir: "../cache",
        max_gpu_jobs: 1,
      },
      plugin_roots: ["../plugins"],
      jobs_dir: "jobs",
      report: { language: "en" },
    };
  },

  parseRequest(
    requestText: string,
    selection: Record<string, unknown>,
    candidate: Record<string, unknown> = {},
    _context?: unknown,
  ): ParseResult {
    const phMatch = requestText.match(/pH\s*([\d.]+)/i);
    const topMatch = requestText.match(/(?:top|返回|前)\s*(\d+)/i);
    const weights = phMatch
      ? { ph: 0.25, activity: 0.55, stability: 0.2 }
      : { ph: 0, activity: 0.7, stability: 0.3 };
    const resolvedSelection = {
      ...selection,
      target_ph: phMatch ? Number(phMatch[1]) : null,
      ph_tolerance: phMatch ? 1 : null,
      weights,
      top_k: topMatch ? Math.min(50, Number(topMatch[1])) : 10,
    };
    const resolvedCandidate = {
      allowed_positions: null,
      excluded_positions: [],
      ...candidate,
      top_k: Math.max(Number(candidate.top_k || 32), resolvedSelection.top_k),
    };
    const diff: ParseResult["diff"] = Object.entries(resolvedSelection)
      .filter(
        ([key, value]) =>
          JSON.stringify(selection[key]) !== JSON.stringify(value),
      )
      .map(([key, value]) => ({
        path: `selection.${key}`,
        before: selection[key],
        after: value,
      }));
    return {
      source: "demo",
      interpretation:
        "演示参数建议：偏重活性，并兼顾稳定性。实际 AI 建议需连接服务端。",
      resolved_candidate: resolvedCandidate,
      resolved_selection: resolvedSelection,
      diff,
      notes: ["演示建议使用模拟参数，没有调用 MiniMax。"],
      explicit_fields: [],
      suggested_fields: [
        "selection.target_ph",
        "selection.ph_tolerance",
        "selection.top_k",
        "selection.weights.ph",
        "selection.weights.activity",
        "selection.weights.stability",
      ],
      rationale: {
        "selection.weights": "演示配置优先考虑活性，同时保留稳定性权重。",
      },
    };
  },

  plan(
    capabilities: string[],
    useLlm: boolean,
    request = "",
    initialArtifacts = ["protein_sequence"],
  ): Record<string, unknown> {
    const byCap: Record<string, string> = {
      activity_prediction: "phenom",
      ph_prediction: "ephod",
      stability_prediction: "unistab",
      codon_optimization: "codon_optimizer",
    };
    const steps: { id: string; tool: string; depends_on: string[] }[] = [];
    let source: string | null = null;
    if (initialArtifacts.includes("nucleotide_sequence")) {
      steps.push({
        id: "translate",
        tool: "sequence_translate",
        depends_on: [],
      });
      source = "translate";
    } else if (capabilities.includes("sequence_translation")) {
      throw new Error("核酸翻译需要 DNA/RNA 输入，请修改已有输入类型。");
    }
    const needsPrepared = capabilities.some((cap) =>
      ["activity_prediction", "ph_prediction", "stability_prediction"].includes(
        cap,
      )
    );
    if (needsPrepared)
      steps.push({
        id: "prepare",
        tool: "preview_candidate_preparation",
        depends_on: source ? [source] : [],
      });
    capabilities
      .filter((cap) => cap !== "sequence_translation")
      .forEach((cap, index) => {
        steps.push({
          id: `step_${index + 1}`,
          tool: byCap[cap] || "phenom",
          depends_on:
            cap === "codon_optimization"
              ? source
                ? [source]
                : []
              : ["prepare"],
        });
      });
    return {
      requested_capabilities: capabilities,
      initial_artifacts: initialArtifacts,
      rationale: useLlm
        ? `演示 MiniMax 方案：${request || "综合比较活性、pH 和稳定性"}`
        : "演示基础方案：按工具输入依赖安排步骤。",
      source: "demo",
      preview_only: true,
      notes: ["演示规划使用本地模拟数据，没有调用 MiniMax 或运行模型。"],
      steps,
    };
  },

  hosts(): HostInfo[] {
    const list: [string, string, string, string][] = [
      ["e_coli", "Escherichia coli K-12 W3110", "大肠杆菌", "316407"],
      ["s_cerevisiae", "Saccharomyces cerevisiae", "酿酒酵母", "4932"],
      ["h_sapiens", "Homo sapiens", "人源细胞", "9606"],
      ["p_putida", "Pseudomonas putida KT2440", "恶臭假单胞菌", "160488"],
      [
        "k_pastoris",
        "Komagataella pastoris (Pichia pastoris)",
        "毕赤酵母",
        "4922",
      ],
      ["b_subtilis", "Bacillus subtilis", "枯草芽孢杆菌", "1423"],
    ];
    return list.filter(([key])=>Boolean(CODON_MAPS[key])).map(([key, scientific_name, display_name_zh, taxid]) => ({
      key,
      scientific_name,
      display_name_zh,
      taxid,
      source: "builtin_kazusa",
    }));
  },

  analysisText(): string {
    return ANALYSIS_TEXT;
  },
  wtSequence(): string {
    return WT;
  },
};

import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import { fileURLToPath } from "node:url";
import ts from "typescript";

const root = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../src",
);
// Compile existing TypeScript in memory using the project's compiler. Each
// harness isolates the browser storage, network, module cache and demo clock.
function harness({ saved = new Map(), query = "?demo=1", fetchImpl } = {}) {
  const cache = new Map();
  const tasks = [];
  const storage = {
    getItem: (key) => saved.get(key) ?? null,
    setItem: (key, value) => saved.set(key, value),
    removeItem: (key) => saved.delete(key),
  };
  const context = vm.createContext({
    URL,
    URLSearchParams,
    AbortController,
    TextDecoder,
    Response,
    Date,
    console,
    localStorage: storage,
    sessionStorage: storage,
    __testEnv: {},
    window: { location: { search: query }, setTimeout, clearTimeout },
    fetch:
      fetchImpl ??
      (() => {
        throw new Error("Unexpected network request");
      }),
    setTimeout: (callback, delay) => {
      tasks.push({ callback, delay });
      return tasks.length;
    },
    clearTimeout: () => {},
  });
  function load(file) {
    if(file.endsWith(".json"))return {default:JSON.parse(fs.readFileSync(path.resolve(root,file),"utf8"))};
    const filename = path.resolve(
      root,
      file.endsWith(".ts") ? file : `${file}.ts`,
    );
    if (cache.has(filename)) return cache.get(filename).exports;
    const source = fs
      .readFileSync(filename, "utf8")
      .replaceAll("import.meta.env", "__testEnv");
    const compiled = ts.transpileModule(source, {
      compilerOptions: {
        target: ts.ScriptTarget.ES2020,
        module: ts.ModuleKind.CommonJS,
      },
    }).outputText;
    const module = { exports: {} };
    cache.set(filename, module);
    const run = vm.runInContext(
      `(function(exports, require, module) { ${compiled}\n})`,
      context,
    );
    run(
      module.exports,
      (name) => load(path.resolve(path.dirname(filename), name)),
      module,
    );
    return module.exports;
  }
  return {
    load,
    saved,
    finish: () => {
      tasks
        .sort((a, b) => a.delay - b.delay)
        .splice(0)
        .forEach((task) => task.callback());
    },
  };
}

const request = {
  sequence_name: "frontend-test",
  sequence: "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAN",
  user_request: null,
  candidate: { top_k: 32, allowed_positions: null, excluded_positions: [] },
  selection: {
    target_ph: 8,
    ph_tolerance: 1,
    min_activity_log10: null,
    stability_direction: "lower_is_better",
    stability_operator: "none",
    stability_threshold: null,
    weights: { ph: 0.3, activity: 0.4, stability: 0.3 },
    top_k: 5,
  },
  llm: { enabled: false },
  visualization: { enabled: true },
  report: { language: "en" },
  runtime: {},
};

test("deployment URL handles blank values and rejects credentials or unsupported schemes", () => {
  const { normalizeApiBase } = harness().load("shared/api/config");
  assert.equal(normalizeApiBase(""), "/api");
  assert.equal(
    normalizeApiBase(" https://example.org/api/// "),
    "https://example.org/api",
  );
  assert.equal(normalizeApiBase("/gateway/api/"), "/gateway/api");
  for (const value of [
    "javascript:alert(1)",
    "//example.org/api",
    "https://user:secret@example.org/api",
    "https://example.org/api?key=secret",
  ])
    assert.throws(() => normalizeApiBase(value));
});
test("explicit demo=false overrides build defaults; arbitrary query values do not enable it", () => {
  const { resolveDemoMode } = harness().load("shared/api/config");
  assert.equal(resolveDemoMode("?demo=0", "1", true), false);
  assert.equal(resolveDemoMode("", "0", true), false);
  assert.equal(resolveDemoMode("?demo=false", null, false), false);
  assert.equal(resolveDemoMode("?demo=1", "0", false), true);
});
test("cancelled demos never become successful or gain completed stages later", async () => {
  const h = harness();
  const { demo } = h.load("shared/demo/backend");
  const { job_id } = await demo.submitRun(request);
  demo.cancelRun(job_id);
  h.finish();
  assert.equal(demo.getRun(job_id).status, "cancelled");
  assert.equal(demo.getRun(job_id).stages.length, 0);
});
test("demo completion, selected count, sequence mutations and reload recovery", async () => {
  const h = harness();
  const { demo } = h.load("shared/demo/backend");
  const { job_id } = await demo.submitRun(request);
  h.finish();
  const meta = demo.getRun(job_id);
  assert.equal(meta.status, "success");
  assert.equal(meta.stages.length, 11);
  assert.equal(meta.result.selected.length, 5);
  for (const row of demo.getCandidates(job_id, {}).rows) {
    const [, original, position, substitute] = row.mutation.match(
      /^([A-Z])(\d+)([A-Z])$/,
    );
    assert.equal(request.sequence[Number(position) - 1], original);
    assert.equal(row.mutant_sequence[Number(position) - 1], substitute);
    assert.equal(row.mutant_sequence.length, request.sequence.length);
  }
  const restored = harness({ saved: h.saved }).load("shared/demo/backend").demo;
  assert.equal(restored.getRun(job_id).status, "success");
  assert.equal(
    JSON.parse(restored.readArtifact(job_id, "00_resolved_request.json"))
      .sequence,
    request.sequence,
  );
});
test("reloading an in-progress demo reports cancellation, not a forever-running job", async () => {
  const h = harness();
  const { demo } = h.load("shared/demo/backend");
  const { job_id } = await demo.submitRun(request);
  assert.equal(
    harness({ saved: h.saved }).load("shared/demo/backend").demo.getRun(job_id)
      .status,
    "cancelled",
  );
});
test("demo artifacts use only local data and unknown files fail clearly", async () => {
  const h = harness();
  const { api } = h.load("shared/api/client");
  const { job_id } = await api.submitRun(request, null);
  h.finish();
  assert.match(await api.readArtifact(job_id, "report.md"), /Demo/);
  assert.equal((await api.ping()).ok, true);
  await assert.rejects(
    api.readArtifact(job_id, "missing.txt"),
    /not available/,
  );
});
test("HTTP artifact errors are not rendered as reports", async () => {
  const { api } = harness({
    query: "?demo=0",
    fetchImpl: async () =>
      new Response('{"detail":"missing artifact"}', { status: 404 }),
  }).load("shared/api/client");
  await assert.rejects(
    api.readArtifact("run", "analysis.txt"),
    (error) => error.status === 404 && error.message === "missing artifact",
  );
});
test("network errors are actionable and GET does not send unnecessary JSON headers", async () => {
  let headers;
  const { api } = harness({
    query: "?demo=0",
    fetchImpl: async (_url, options) => {
      headers = options.headers;
      throw new Error("offline");
    },
  }).load("shared/api/client");
  await assert.rejects(api.tools(), /Cannot reach the server/);
  assert.equal(headers["Content-Type"], undefined);
});
test("CSV export escapes text and keeps missing or negative numeric values intact", () => {
  const { serializeCsv } = harness().load("shared/lib/download");
  const csv = serializeCsv(
    [
      {
        mutation: "A1V",
        ddg: -1.5,
        missing: null,
        note: "=1+1",
        text: 'a,"b"',
      },
    ],
    ["mutation", "ddg", "missing", "note", "text"],
  );
  assert.match(csv, /"-1.5","","'=1\+1","a,""b"""/);
});
test("report downloads use the generated language filename from server result paths", async () => {
  const urls = [];
  const h = harness({
    query: "?demo=0",
    fetchImpl: async (url) => {
      urls.push(url);
      return new Response("# 实验报告", { status: 200 });
    },
  });
  const { artifactFilename } = h.load("shared/lib/download");
  const { api } = h.load("shared/api/client");
  const cases = [
    [
      "/home/user/project/server/jobs/run/outputs/try1/report_zh.md",
      "report_zh.md",
    ],
    [
      "D:\\project\\server\\jobs\\run\\outputs\\try2\\report_en.md",
      "report_en.md",
    ],
    ["report.md", "report.md"],
    ["# Enzora Demo\nIllustrative report", "report.md"],
    [undefined, "report.md"],
  ];
  for (const [reference, expected] of cases) {
    const filename = artifactFilename(reference, "report.md");
    assert.equal(filename, expected);
    assert.equal(await api.readArtifact("run", filename), "# 实验报告");
    assert.equal(urls.at(-1), `/api/runs/run/files/${expected}`);
  }
  assert.equal(
    artifactFilename("/outputs/try1/analysis.txt", "analysis.txt"),
    "analysis.txt",
  );
});
test("zero pH tolerance is rejected and inputs stay unchanged", () => {
  const { rerankCandidates } = harness().load("features/results/rerank");
  const rows = [
    {
      mutation: "A1V",
      activity_log10: 1,
      ddg: -1,
      ph_opt: 8,
      final_score: 0.1,
    },
  ];
  assert.throws(
    () =>
      rerankCandidates(rows, request.selection.weights, {
        target_ph: 8,
        ph_tolerance: 0,
      }),
    /tolerance/,
  );
  assert.equal(rows[0].final_score, 0.1);
});

test("environment-backed planner sends the goal and initial artifacts without a browser key session", async () => {
  const calls = [];
  const { api } = harness({
    query: "?demo=0",
    fetchImpl: async (url, options) => {
      calls.push({ url, body: options.body ? JSON.parse(options.body) : null });
      return new Response(
        JSON.stringify(
          url.endsWith("/session/status")
            ? { configured: true, source: "environment" }
            : {
                requested_capabilities: ["ph_prediction"],
                initial_artifacts: ["protein_sequence"],
                steps: [],
                rationale: "test",
              },
        ),
        { status: 200, headers: { "Content-Type": "application/json" } },
      );
    },
  }).load("shared/api/client");
  assert.equal((await api.sessionStatus(null)).configured, true);
  await api.plan(["ph_prediction"], true, null, "Target pH 7.5", [
    "protein_sequence",
  ], { sequence_name: "enzyme", sequence_type: "protein", sequence_length: 544, current_goal: "Target pH 7.5" });
  const body = calls[1].body;
  assert.equal(body.request, "Target pH 7.5");
  assert.equal(body.input_summary.sequence_length, 544);
  assert.equal(body.input_summary.sequence_name, "enzyme");
  assert.equal("sequence" in body.input_summary, false);
  assert.equal(body.session_id, null);
  assert.equal(body.use_llm, true);
  assert.deepEqual(body.initial_artifacts, ["protein_sequence"]);
  assert.equal("api_key" in body, false);
  assert.equal("model" in body, false);
});

test("AI parameter context and source metadata cross the API unchanged", async () => {
  let body;
  const { api } = harness({
    query: "?demo=0",
    fetchImpl: async (_url, options) => {
      body = JSON.parse(options.body);
      return new Response(
        JSON.stringify({
          source: "ai",
          resolved_candidate: request.candidate,
          resolved_selection: request.selection,
          suggested_fields: ["selection.weights.activity"],
          explicit_fields: ["selection.target_ph"],
          rationale: { "selection.weights": "Prioritize activity" },
          diff: [],
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      );
    },
  }).load("shared/api/client");
  const result = await api.parseRequest(
    "Prioritize activity at pH 8",
    request.candidate,
    request.selection,
    null,
    { sequence_name: "enzyme", sequence_length: 35, sequence_type: "protein" },
  );
  assert.equal(body.context.sequence_length, 35);
  assert.equal("sequence" in body.context, false);
  assert.equal(result.source, "ai");
  assert.equal(result.rationale["selection.weights"], "Prioritize activity");
  assert.equal(result.explicit_fields[0], "selection.target_ph");
});

test("confidence-adjusted preview uses PHENOM activity and rejects missing predictions", () => {
  const { rerankCandidates } = harness().load("features/results/rerank");
  const rows = [
    { mutation: "A1V", activity_log10: 2, ddg: 1, ph_opt: 7 },
    { mutation: "A1C", activity_log10: 1, ddg: -1, ph_opt: 7 },
  ];
  const result = rerankCandidates(
    rows,
    { ph: 0, activity: 0.5, stability: 0.5 },
    {
      confidence: { ph: 1, activity: 0, stability: 1 },
      stability_direction: "lower_is_better",
    },
  );
  assert.equal(result[0].mutation, "A1C");
  assert.throws(
    () =>
      rerankCandidates([{ mutation: "A1V", ddg: 0 }], {
        ph: 0,
        activity: 1,
        stability: 0,
      }),
    /predictions/,
  );
});

test("manual demo phases wait for explicit Next; 1000 unique sequences and local APIs stay offline", async () => {
  const h = harness();
  const { api } = h.load("shared/api/client");
  const draft = structuredClone(request);
  draft.sequence = request.sequence.repeat(2);
  draft.candidate.top_k = 1000;
  draft.mutation = { counts: { 1: 1000 } };
  draft.runtime.phase_control = true;
  draft.selection.confidence = { ph: 1, activity: 1, stability: 1 };
  const { job_id } = await api.submitRun(draft, null);
  h.finish();
  assert.equal((await api.getRun(job_id)).progress.status, "awaiting_next");
  assert.equal((await api.getRun(job_id)).progress.next_phase, 2);
  assert.equal((await api.getRun(job_id)).stages.length, 2);
  const rows = (
    await api.getCandidates(job_id, { pool: "scan", page_size: 1000 })
  ).rows;
  assert.equal(rows.length, 1000);
  assert.equal(new Set(rows.map((r) => r.mutant_sequence)).size, 1000);
  assert.ok(
    rows.every(
      (r) =>
        r.activity_log10 === undefined &&
        r.ddg === undefined &&
        r.ph_opt === undefined,
    ),
  );
  assert.throws(() => api.advanceRun(job_id, 3), /not ready/);
  await api.advanceRun(job_id, 2);
  h.finish();
  assert.equal((await api.getRun(job_id)).progress.next_phase, 3);
  const scores = (
    await api.getCandidates(job_id, { pool: "scores", page_size: 1000 })
  ).rows;
  assert.ok(
    scores.every(
      (r) =>
        typeof r.activity_log10 === "number" &&
        typeof r.ddg === "number" &&
        typeof r.ph_opt === "number",
    ),
  );
  const balanced = await api.balanceRun(
    job_id,
    { ph: 0, activity: 0.5, stability: 0.5 },
    { ph: 1, activity: 0.2, stability: 1 },
  );
  assert.ok(Math.abs(balanced.effective_weights.activity - 1 / 6) < 1e-12);
  await assert.rejects(api.chat("explain", job_id, null), /Cannot reach/);
  await api.advanceRun(job_id, 3);
  h.finish();
  assert.equal((await api.getRun(job_id)).status, "success");
  const atlas = JSON.parse(await api.readArtifact(job_id, "umap.json"));
  assert.equal(atlas.points.length, 1000);
  assert.match(atlas.method, /not UMAP/);
  assert.equal(atlas.points.filter((p) => p.rank).length, 5);
  assert.throws(
    () =>
      api.balanceRun(
        job_id,
        draft.selection.weights,
        draft.selection.confidence,
      ),
    /closed/,
  );
});

test("copilot is available before a run and keeps returned codon artifacts", async () => {
  let sent;
  const h = harness({
    query: "?demo=0",
    fetchImpl: async (url, init) => {
      sent = { url, body: JSON.parse(init.body) };
      return new Response(
        ': assistant working\n\ndata: '+JSON.stringify({event:'complete',data:{message:'Ready',codon_rows:[{mutation:'M1A',optimized_cds_with_stop:'GCGTAA'}]}})+'\n\n',
        { status: 200,headers:{"Content-Type":"text/event-stream"} },
      );
    },
  });
  const { api } = h.load("shared/api/client");
  const response = await api.chat("Explain PheNom", null, null);
  assert.equal(sent.url, "/api/chat/stream");
  assert.equal(sent.body.job_id, null);
  assert.equal(response.codon_rows[0].optimized_cds_with_stop, "GCGTAA");
  const local = harness().load("shared/api/client");
  await assert.rejects(local.api.chat("Explain", null, null), /Cannot reach/);
});
test("demo host export rejects unknown residues and supports Bacillus explicitly", async () => {
  const { api } = harness().load("shared/api/client");
  const response = await api.codonOptimize("MANT", ["b_subtilis"]);
  assert.equal(response.results[0].scientific_name, "Bacillus subtilis");
  assert.equal(
    response.results[0].optimized_cds,
    "ATGGC AAATACA".replaceAll(" ", ""),
  );
  assert.throws(
    () => api.codonOptimize("MAX", ["e_coli"]),
    /Canonical protein/,
  );
});

test("double-mutation preview preserves the VESM delimiter and exact substitution count", async () => {
  const h = harness();
  const { api } = h.load("shared/api/client");
  const draft = structuredClone(request);
  draft.candidate.top_k = 20;
  draft.mutation = { counts: { 2: 20 } };
  draft.runtime.phase_control = true;
  const { job_id } = await api.submitRun(draft, null);
  h.finish();
  const { rows } = await api.getCandidates(job_id, {
    pool: "scan",
    page_size: 100,
  });
  assert.equal(rows.length, 20);
  for (const row of rows) {
    assert.match(row.mutation, /^[A-Z]\d+[A-Z]:[A-Z]\d+[A-Z]$/);
    assert.equal(
      [...row.mutant_sequence].filter((aa, i) => aa !== draft.sequence[i])
        .length,
      2,
    );
  }
});


test('BioArtifact export preserves schema, predictions and format warnings across HTTP', async () => {
  const artifact = { schema: 'enzora.bioartifact/1', records: [{id:'WT',name:'enzyme',type:'protein',sequence:'ACD'}], predictions: {kcat:{WT:2}}, metadata: {source:'upload'}, data:{} };
  let sent;
  const h = harness({query:'?demo=0', fetchImpl: async (url, options) => {
    assert.ok(String(url).endsWith('/input/export'));
    sent = JSON.parse(options.body);
    return new Response(JSON.stringify({content:'>WT enzyme\nACD\n', warnings:['FASTA annotation loss']}), {status:200,headers:{'Content-Type':'application/json'}});
  }});
  const result = await h.load('shared/api/client').api.exportArtifact(artifact, 'fasta');
  assert.deepEqual(sent, {bioartifact:artifact,format:'fasta'});
  assert.equal(result.warnings[0], 'FASTA annotation loss');
});

test('WT-reference reweighting preserves frozen components instead of pool min-max', () => {
  const h = harness();
  const rows = [{mutation:'a',activity_log10:1,ph_opt:7,ddg:0,activity_component:.7,ph_component:.6,stability_component:.5,score_normalization:'input_reference_three_model_llm_v3'},
                {mutation:'b',activity_log10:100,ph_opt:7,ddg:0,activity_component:.4,ph_component:.6,stability_component:.5,score_normalization:'input_reference_three_model_llm_v3'}];
  const ranked = h.load('features/results/rerank').rerankCandidates(rows,{activity:1,ph:0,stability:0});
  assert.equal(ranked[0].mutation,'a');
  assert.equal(ranked[0].final_score,.7);
  assert.equal(ranked[1].final_score,.4);
});


test("workspace login adds bearer headers to REST and streaming progress without URL credentials", async () => {
  const calls = [];
  const saved = new Map();
  const client = harness({saved, query: "?demo=0", fetchImpl: async (url, options = {}) => {
    calls.push({url, options});
    if (url.endsWith("/events")) return new Response('data: {"event":"scheduler","data":{"state":"RUNNING"}}\n\ndata: {"event":"stream_end","data":{}}\n\n');
    return new Response(JSON.stringify({configured: true}), {headers: {"Content-Type":"application/json"}});
  }}).load("shared/api/client");
  await client.login("test-workspace-code");
  await client.api.sessionStatus(null);
  const events = [];
  await new Promise((resolve) => client.api.subscribeRun("run123", (event) => events.push(event), resolve));
  assert.equal(events[0].event, "scheduler");
  assert.ok(calls.every(({options}) => options.headers.Authorization === "Bearer test-workspace-code"));
  assert.ok(calls.every(({url}) => !url.includes("test-workspace-code")));
});


test('Assistant stream reports a provider failure instead of hanging or inventing a reply', async()=>{
 const h=harness({query:'?demo=0',fetchImpl:async()=>new Response(': heartbeat\n\ndata: '+JSON.stringify({event:'error',data:{status:502,message:'MiniMax HTTP 429'}})+'\n\n')});
 await assert.rejects(()=>h.load('shared/api/client').api.chat('Hello',null,null),/MiniMax HTTP 429/);
});

test('Sequence and Atlas tools send explicit inputs only to the Enzora API',async()=>{
 const sent=[];
 const h=harness({query:'?demo=0',fetchImpl:async(url,init)=>{sent.push([url,JSON.parse(init.body)]);return new Response(JSON.stringify({source:'ESM Atlas Alpha'}));}});
 const {api}=h.load('shared/api/client');
 await api.atlas('P01308',false);
 await api.sequenceTool('prepare_mature_protein','ACDE',{start:2,end:4});
 assert.equal(sent[0][0],'/api/tools/atlas');assert.deepEqual(sent[0][1],{query:'P01308',similar:false});
 assert.equal(sent[1][0],'/api/tools/sequence');assert.deepEqual(sent[1][1],{tool:'prepare_mature_protein',sequence:'ACDE',arguments:{start:2,end:4}});
});

test('Assistant forwards real progress and bounded conversation context before the final reply',async()=>{
 const progress=[];let sent;
 const event={phase:'tool_running',tool:'query_esm_atlas'};
 const h=harness({query:'?demo=0',fetchImpl:async(url,init)=>{sent=JSON.parse(init.body);return new Response('data: '+JSON.stringify({event:'progress',data:event})+'\n\ndata: '+JSON.stringify({event:'complete',data:{message:'Retrieved protein'}})+'\n\n');}});
 const history=[{role:'user',text:'Find P01308'}];
 const result=await h.load('shared/api/client').api.chat('Show its metadata',null,null,undefined,e=>progress.push(e),history);
 assert.deepEqual(JSON.parse(JSON.stringify(progress)),[event]);assert.deepEqual(sent.history,history);assert.equal(result.message,'Retrieved protein');
});


test("candidate evidence changes with uncertainty and reconstructs the frozen score", () => {
  const {candidateScoreEvidence, rerankCandidates} = harness().load("features/results/rerank");
  const selection = {...request.selection,confidence:{activity:.8,ph:.9,stability:1},
    reference:{activity_log10:1,ph_opt:6},score_policy:{method:'input_reference_three_model_llm_v3',target_ph:8,
      scales:{kcat:1,ph:1,stability:1},uncertainty_scales:{kcat:1,ph:.5,stability:2}}};
  const candidate = {mutation:'A1G',activity_log10:2,ph_opt:7,ddg:-1,
    activity_sigma_log10:.1,ph_uncertainty:.5,stability_uncertainty:2};
  const a = candidateScoreEvidence(candidate,selection);
  const b = candidateScoreEvidence({...candidate,activity_sigma_log10:10},selection);
  assert.ok(a.models[0].influence > b.models[0].influence);
  assert.ok(a.score > b.score);
  assert.ok(b.neutral > a.neutral);
  const components = Object.fromEntries(a.models.map(m=>[`${m.key}_component`,.5+m.reliability*(m.utility-.5)]));
  const ranked = rerankCandidates([{...candidate,...components,score_normalization:'input_reference_three_model_llm_v3'}],selection.weights,selection);
  assert.ok(Math.abs(a.score-ranked[0].final_score)<1e-12);
  assert.equal(candidate.activity_log10,2);
  assert.throws(()=>candidateScoreEvidence({...candidate,ph_uncertainty:null},selection));
  assert.throws(()=>candidateScoreEvidence({...candidate,activity_log10:null},selection));
  assert.throws(()=>candidateScoreEvidence({...candidate,stability_uncertainty:-1},selection));
});

test('public entry defaults to Demo even when an old visitor saved live mode without an invitation', () => {
  const h=harness({query:'',saved:new Map([['fea_demo','0']])});
  assert.equal(h.load('shared/api/client').isDemo(),true);
  assert.equal(harness({query:'?demo=0'}).load('shared/api/client').isDemo(),false);
});

test('permanent real-case Demo is restored independently of browser cache', () => {
  const {demo,DEMO_CASE_ID}=harness().load('shared/demo/backend');
  const baseline=demo.getRun(DEMO_CASE_ID);
  assert.equal(baseline.status,'success');
  const rows=demo.getCandidates(DEMO_CASE_ID,{page_size:1000}).rows;
  assert.equal(rows.length,1000);assert.equal(new Set(rows.map(r=>r.mutant_sequence)).size,1000);
  assert.equal(baseline.result.selected[0].mutation,'V368R');
  const umap=JSON.parse(demo.readArtifact(DEMO_CASE_ID,'umap.json'));
  assert.equal(umap.method,'UMAP');assert.equal(umap.points.length,1000);
  assert.equal(harness().load('shared/demo/backend').demo.getRun(DEMO_CASE_ID).result.selected[0].mutation,'V368R');
});

test('saved-case Build, Screening and Top-K can be repeated without corrupting the permanent case', async () => {
  const h=harness();const {demo,DEMO_CASE_ID}=h.load('shared/demo/backend');
  const original=JSON.stringify(demo.getCandidates(DEMO_CASE_ID,{page_size:1000}).rows);
  const request=JSON.parse(demo.readArtifact(DEMO_CASE_ID,'00_resolved_request.json'));
  request.runtime.phase_control=true;
  for(let pass=0;pass<3;pass++){
    const {job_id}=await demo.submitRun(request);h.finish();
    assert.equal(demo.getRun(job_id).progress.next_phase,2);
    demo.balanceRun(job_id,request.selection.weights,request.selection.confidence);
    demo.advanceRun(job_id,2);h.finish();assert.equal(demo.getRun(job_id).progress.next_phase,3);
    demo.advanceRun(job_id,3);h.finish();assert.equal(demo.getRun(job_id).status,'success');
    assert.equal(demo.getRun(job_id).result.selected[0].mutation,'V368R');
    assert.equal(JSON.parse(demo.readArtifact(job_id,'umap.json')).points.length,1000);
    assert.match(demo.readArtifact(job_id,'llm_balance_decisions.jsonl'),/reason/);
  }
  for(const phase of [2,3,2,3]){const replay=demo.replayRun(DEMO_CASE_ID,phase,request.selection);h.finish();assert.equal(demo.getRun(replay.job_id).progress.status,phase===2?'awaiting_next':'success');}
  assert.equal(JSON.stringify(demo.getCandidates(DEMO_CASE_ID,{page_size:1000}).rows),original);
});

test('Demo local tools and file roundtrip remain repeatable, and cannot fold arbitrary proteins', () => {
  const {demo,DEMO_ENZYME_SEQUENCE:wt}=harness().load('shared/demo/backend');
  for(let i=0;i<3;i++){
    assert.equal(demo.sequenceTool('inspect_structure',wt).structure.points.length,544);
    assert.equal(demo.sequenceTool('prepare_mature_protein',wt,{start:65,end:544}).prepared_sequence.sequence,wt.slice(64));
    assert.equal(demo.atlas('P00918').metadata.protein_name,'Carbonic anhydrase 2');
    const cds=demo.codonOptimize(wt,['e_coli']).results[0].optimized_cds_with_stop;
    assert.equal(demo.translate(cds,'dna').protein_sequence,wt);
    const imported=demo.inputFile('selected.fasta','>WT\n'+wt,'protein');
    assert.equal(JSON.parse(demo.exportArtifact(imported.bioartifact,'json').content).records[0].sequence,wt);
  }
  assert.throws(()=>demo.sequenceTool('predict_structure',wt),/invited/);
  assert.throws(()=>demo.sequenceTool('inspect_structure','AAAA'),/only/);
});

test('Demo Assistant calls only the public read-only endpoint and forwards bounded conversation', async () => {
  const calls=[];const h=harness({fetchImpl:async(url,init)=>{calls.push({url,body:JSON.parse(init.body)});return new Response(JSON.stringify({message:'Saved case explanation',source:'minimax',mode:'read_only_saved_case'}),{status:200});}});
  const answer=await h.load('shared/api/client').api.chat('Why V368R?',null,null,undefined,undefined,[{role:'user',text:'Explain the saved case'}]);
  assert.equal(answer.message,'Saved case explanation');assert.equal(calls.length,1);assert.equal(calls[0].url,'/api/demo/chat');
  assert.equal(calls[0].body.history.length,1);assert.equal(calls[0].body.job_id,undefined);
});


test('default Atlas demonstration has an actual stored matching structure and reloads repeatedly',()=>{
 const {demo,DEMO_ATLAS_QUERY,DEMO_ATLAS_SEQUENCE}=harness().load('shared/demo/backend');
 for(let i=0;i<3;i++){
   const result=demo.atlas(DEMO_ATLAS_QUERY);
   assert.equal(result.metadata.accession,'UPI000036401E');
   assert.equal(result.metadata.sequence,DEMO_ATLAS_SEQUENCE);
   assert.equal(result.structure.sequence,DEMO_ATLAS_SEQUENCE);
   assert.equal(result.structure.points.length,260);
   assert.match(result.structure.pdb,/ATOM/);
   assert.match(result.structure.pdb,/Attribution 4.0/);
   assert.equal(result.fold_on_miss,false);
   assert.equal(result.metadata.folded_on_demand,false);
   assert.equal(demo.sequenceTool('inspect_structure',DEMO_ATLAS_SEQUENCE).structure.points.length,260);
 }
 assert.equal(demo.atlas('P00918').structure,undefined);
 assert.equal(demo.lookupResolve('P00918','uniprot').sequence,demo.atlas('P00918').metadata.sequence);
});

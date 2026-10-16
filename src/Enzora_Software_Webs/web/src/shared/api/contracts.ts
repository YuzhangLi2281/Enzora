/** Shared data contracts mirroring the backend (server/) and the agent's request.json. */

export type Lang = "en" | "zh";
export type Theme = "deep-space" | "macaron";
export type InputType = "protein" | "dna" | "rna" | "text" | "mixed";

export interface RunRequest {
  sequence_name: string;
  sequence: string;
  user_request: string | null;
  candidate: {
    top_k: number;
    allowed_positions: number[] | null;
    excluded_positions: number[];
  };
  mutation?: {
    counts: Record<number, number>;
    random_seed?: number;
    chain_id?: string;
  };
  activity?: { task: "kcat"; ligand_smiles: string };
  selection: {
    target_ph: number | null;
    ph_tolerance: number | null;
    min_activity_log10: number | null;
    stability_direction: "lower_is_better" | "higher_is_better";
    stability_operator: "none" | "le" | "ge";
    stability_threshold: number | null;
    confidence?: { ph: number; activity: number; stability: number };
    weights: { ph: number; activity: number; stability: number };
    top_k: number;
  };
  llm: {
    enabled: boolean;
    parameter_parsing: boolean;
    result_analysis: boolean;
    analysis_fail_open: boolean;
  };
  visualization: { enabled: boolean; umap?: boolean };
  report: { language: Lang };
  runtime: {
    gpu_id?: number;
    force_cpu?: boolean;
    phase_control?: boolean;
    timeout_seconds?: number;
    keep_temp?: boolean;
  };
}

export interface Candidate {
  mutation: string;
  position?: number | null;
  wt_aa?: string | null;
  mutant_aa?: string | null;
  activity_log10?: number | null;
  ph_opt?: number | null;
  delta_ph_from_wt?: number | null;
  ddg?: number | null;
  activity_component?: number | null;
  ph_component?: number | null;
  stability_component?: number | null;
  final_score?: number | null;
  passed?: boolean;
  reasons?: string;
  blosum62?: number | null;
  conservative_substitution?: boolean | null;
  pareto_optimal?: boolean | null;
  residue_class_change?: string | null;
  delta_hydrophobicity?: number | null;
  delta_volume?: number | null;
  delta_charge?: number | null;
  mutant_sequence?: string | null;
  [key: string]: unknown;
}

export interface StageRecord {
  index: number;
  stage: string;
  started_at?: number | null;
  finished_at?: number | null;
  seconds?: number | null;
  stats?: Record<string, unknown> | null;
}

export interface RunMeta {
  id: string;
  status: "queued" | "running" | "success" | "failed" | "cancelled";
  created_at: number;
  started_at?: number | null;
  finished_at?: number | null;
  error?: string | null;
  force_cpu?: boolean;
  config_summary?: {
    sequence_name: string;
    sequence_length: number;
    user_request: string | null;
    report_language: Lang;
  };
  stages: StageRecord[];
  run_dir: string;
  generation?: Record<string, unknown>;
  applied_selection?: RunRequest['selection'];
  progress?: {
    phase: number;
    status: string;
    next_phase?: number;
    saved_sequences?: number;
    target_sequences?: number;
  };
  result?: RunResult | null;
}

export interface RunResult {
  status: string;
  stages: Record<string, unknown>[];
  selected: Candidate[];
  candidate_pool_size?: number;
  wt_ph?: number;
  markdown_report?: string;
  figures?: string[];
  analysis_file?: string;
  codon_optimization?: CodonOptimizationInfo;
  error?: string;
}

export interface CodonOptimizationInfo {
  enabled?: boolean;
  status?: string;
  host_count?: number;
  count?: number;
  organisms?: string[];
  resolved_hosts?: HostProfile[];
  failed_hosts?: { query: string; error: string }[];
  warnings?: string[];
  csv?: string | null;
  fasta?: string | null;
  per_host_fastas?: Record<string, string>;
}

export interface HostProfile {
  organism_key: string;
  scientific_name: string;
  display_name_zh?: string | null;
  taxid: string;
  codon_source?: string;
  gc_fraction?: number;
  optimized_cds?: string;
  optimized_cds_with_stop?: string;
  stop_codon?: string;
}

export interface HostInfo {
  key: string;
  scientific_name: string;
  display_name_zh: string;
  taxid: string;
  source?: string;
}

export interface LookupCandidate {
  source: "uniprot" | "ncbi";
  accession: string;
  name: string | null;
  gene?: string | null;
  organism: string | null;
  taxid?: number | null;
  length?: number | null;
  ec_numbers?: string[];
  reviewed: boolean;
  url: string;
  license: string;
}

export interface LookupResolved {
  accession: string;
  name: string | null;
  organism: string | null;
  taxid?: number | null;
  sequence: string;
  url: string;
  license: string;
}

export interface BioArtifact {
  schema: 'enzora.bioartifact/1';
  records: InputRecord[];
  predictions: Record<string, unknown>;
  metadata: Record<string, unknown>;
  data: Record<string, unknown>;
}

export interface ClassifyResult {
  bioartifact?: BioArtifact;
  action?: "new_input" | "update_goal" | "clarify";
  type: InputType;
  sequence: string | null;
  sequences: string[];
  request_text: string | null;
  fasta_name: string | null;
  warnings: string[];
  translation: TranslationResult | null;
  ambiguous: boolean;
  suggested_type: "dna" | "protein" | null;
  records?: InputRecord[];
  interpretation?: string;
  lookup_query?: string | null;
  requires_confirmation?: boolean;
}

export interface InputContext {
  sequence_name: string;
  sequence_type: string;
  sequence_length: number;
  current_goal: string;
}

export interface InputRecord {
  id: string;
  name: string;
  type: "protein" | "dna" | "rna";
  sequence: string;
}

export interface SessionStatus {
  configured: boolean;
  source: "environment" | "session" | "none" | "demo";
}

export interface TranslationResult {
  protein_sequence: string;
  orf_start_1based: number;
  orf_end_1based: number;
  cds_length: number;
  warnings: string[];
}

export interface ToolInfo {
  name: string;
  version: string;
  description: string;
  capabilities: string[];
  inputs: string[];
  outputs: string[];
  priority: number;
  properties: Record<string, unknown>;
  health: { available: boolean; details: string[] };
}

export interface ServerSettings {
  models: Record<string, Record<string, unknown>>;
  llm: Record<string, unknown>;
  runtime: Record<string, unknown>;
  plugin_roots: string[];
  jobs_dir: string;
  report: { language: Lang };
}

export type RunEventName =
  | "queued"
  | "started"
  | "stage_completed"
  | "log_line"
  | "llm_parsed"
  | "completed"
  | "failed"
  | "cancelled"
  | "stream_end"
  | "heartbeat"
  | "scheduler"
  | "scheduler_warning";

export interface RunEvent {
  event: RunEventName | string;
  data: Record<string, unknown>;
}

export interface ParseResult {
  source?: "ai" | "demo";
  suggested_fields?: string[];
  explicit_fields?: string[];
  rationale?: Record<string, string>;
  interpretation?: string;
  resolved_candidate: Record<string, unknown>;
  resolved_selection: Record<string, unknown>;
  diff: { path: string; before: unknown; after: unknown }[];
  notes?: string[];
}

export interface CodonOptimizeRow {
  organism_key: string;
  scientific_name: string;
  display_name_zh: string | null;
  taxid: string;
  codon_source: string;
  optimized_cds: string;
  optimized_cds_with_stop: string;
  stop_codon: string;
  protein_length: number;
  cds_length: number;
  gc_fraction: number;
}

export interface CodonOptimizeResponse {
  results: CodonOptimizeRow[];
  errors: { organism_key: string; error: string }[];
  method: string;
}

export interface ParameterContext {
  sequence_name?: string;
  sequence_type?: string;
  sequence_length?: number;
  current_goal?: string;
}

export interface PlanPreview {
  requested_capabilities: string[];
  initial_artifacts: string[];
  rationale: string;
  steps: { id: string; tool: string; depends_on: string[] }[];
  preview_only?: boolean;
  source?: string;
  notes?: string[];
}

export interface ProteinStructure {
  source: string;
  sequence: string;
  pdb?: string;
  points: {residue:number;x:number;y:number;z:number}[];
}

export interface AtlasRecord {
  source: string;
  source_url: string;
  fold_on_miss?: boolean;
  metadata?: Record<string,unknown> & {sequence?:string;protein_hash?:string};
  structure?: ProteinStructure;
  similar_proteins?: {protein_hash:string;protein_accession:string;similarity_score:number;protein_name?:string;sequence_length?:number}[];
}

export interface AssistantProgress {
  phase:'planning'|'tool_running'|'tool_complete'|'explaining';
  tool?:string;
}

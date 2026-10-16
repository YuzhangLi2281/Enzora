import { useEffect, useState } from "react";
import { api, isDemo, ApiError } from "../../shared/api/client";
import { parseCsv } from "../../shared/lib/csv";
import { recallRequest } from "./requestStorage";
import { useApp } from "../../app/store";
import type {
  Candidate,
  HostProfile,
  RunMeta,
  RunRequest,
} from "../../shared/api/contracts";

const active = (meta: RunMeta) =>
  meta.status === "queued" || meta.status === "running";
const message = (error: unknown) =>
  error instanceof Error ? error.message : String(error);

export function useRun(jobId: string | null) {
  const [atlas, setAtlas] = useState<{
    method: string;
    points: {
      mutation: string;
      x: number;
      y: number;
      rank: number | null;
      score: number | null;
    }[];
  } | null>(null);
  const [generation, setGeneration] = useState<Record<string, unknown>>({});
  const [revision, retry] = useState(0);
  const [meta, setMeta] = useState<RunMeta | null>(null);
  const [rawCandidates, setRawCandidates] = useState<Candidate[]>([]);
  const [wtSequence, setWtSequence] = useState("");
  const [codonRows, setCodonRows] = useState<HostProfile[]>([]);
  const [analysisText, setAnalysisText] = useState<string | null>(null);
  const [analysisError, setAnalysisError] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [reconnecting, setReconnecting] = useState(false);
  const [loading, setLoading] = useState(false);
  const [logs, setLogs] = useState<{ source: string; line: string }[]>([]);

  useEffect(() => {
    let disposed = false;
    let refreshing = false;
    let finished = false;
    let requestLoaded = false;
    let unsubscribe: (() => void) | undefined;
    const store = useApp.getState();
    store.setActiveJob(jobId);
    store.setCandidates([]);
    store.setSelectedMutation(null);
    const cached = jobId ? recallRequest(jobId) : null;
    store.setRunRequest(cached);
    setAtlas(null);
    setGeneration({});
    setMeta(null);
    setRawCandidates([]);
    setCodonRows([]);
    setLogs([]);
    setWtSequence(cached?.sequence ?? "");
    setAnalysisText(null);
    setAnalysisError(null);
    setError(null);
    setReconnecting(false);
    setLoading(Boolean(jobId));
    if (!jobId) return;
    const id = jobId;

    async function loadRequest(run: RunMeta) {
      if (requestLoaded) return;
      try {
        const resolved = JSON.parse(
          await api.readArtifact(id, "00_resolved_request.json"),
        );
        if (disposed) return;
        const sequence = resolved.sequence;
        if (
          typeof sequence !== "string" ||
          !resolved.resolved_candidate ||
          !resolved.resolved_selection
        )
          throw new Error("Resolved request contract is incomplete.");
        const request: RunRequest = cached
          ? cached
          : {
              sequence_name: run.config_summary!.sequence_name,
              sequence,
              user_request: resolved.user_request,
              candidate: resolved.resolved_candidate,
              selection: resolved.resolved_selection,
              mutation: resolved.mutation,
              activity: resolved.activity,
              llm: {
                enabled: false,
                parameter_parsing: false,
                result_analysis: false,
                analysis_fail_open: true,
              },
              visualization: { enabled: true, umap: true },
              report: { language: run.config_summary!.report_language },
              runtime: { phase_control: true },
            };
        if (run.applied_selection) request.selection = run.applied_selection;
        setWtSequence(sequence);
        useApp.getState().setRunRequest(request);
        requestLoaded = Boolean(sequence);
      } catch (failure) {
        if (!(failure instanceof ApiError && failure.status === 404))
          throw failure;
      }
    }

    async function refresh() {
      if (disposed || refreshing || finished) return;
      refreshing = true;
      try {
        const run = await api.getRun(id);
        if (disposed) return;
        setMeta({ ...run, stages: [...run.stages] });
        if (run.generation) setGeneration(run.generation);
        setError(null);
        setLoading(false);
        if (active(run)) {
          await loadRequest(run);
          const pool = run.stages.some((stage) => stage.index >= 2)
            ? "scores"
            : run.stages.some((stage) => stage.index >= 1)
              ? "scan"
              : null;
          if (pool) {
            const rows: Candidate[] = [];
            let page = 1;
            try {
              let total: number;
              do {
                const batch = await api.getCandidates(id, {
                  pool,
                  page,
                  page_size: 500,
                });
                rows.push(
                  ...(batch.rows.map((row) => ({
                    ...row,
                    mutation: row.mutation ?? row.mutations,
                  })) as Candidate[]),
                );
                total = batch.total;
                page++;
                if (!batch.rows.length) break;
              } while (rows.length < total);
              if (!disposed) setRawCandidates(rows);
            } catch (failure) {
              if (!(failure instanceof ApiError && failure.status === 404))
                throw failure;
            }
          }
          if (!unsubscribe && !disposed)
            unsubscribe = api.subscribeRun(
              id,
              (event) => {
                if (disposed) return;
                if (event.event === "scheduler" || event.event === "scheduler_warning") {
                  setLogs((previous) => [...previous, {
                    source: "Slurm",
                    line: event.event === "scheduler" ? `Job ${event.data.job_id}: ${event.data.state}` : String(event.data.message ?? event.data.error ?? "Scheduler temporarily unavailable"),
                  }].slice(-120));
                  void refresh();
                } else if (event.event === "log_line")
                  setLogs((previous) =>
                    [
                      ...previous,
                      {
                        source: String(event.data.source ?? ""),
                        line: String(event.data.line ?? ""),
                      },
                    ].slice(-120),
                  );
                else {
                  if (event.event === "progress")
                    setGeneration((previous) => ({
                      ...previous,
                      ...event.data,
                    }));
                  void refresh();
                }
              },
              () => {
                void refresh();
              },
              (connected) => {
                if (!disposed) setReconnecting(!connected);
              },
            );
          return;
        }
        unsubscribe?.();
        unsubscribe = undefined;
        setReconnecting(false);
        let rows: Candidate[] = [];
        if (run.status === "success") {
          let page = 1;
          let total = 0;
          do {
            const result = await api.getCandidates(id, {
              pool: "ranked",
              page,
              page_size: 500,
            });
            if (disposed) return;
            rows.push(...(result.rows as unknown as Candidate[]));
            total = result.total;
            if (!result.rows.length) break;
            page += 1;
          } while (rows.length < total);
          const selected = new Map(
            (run.result?.selected ?? []).map((row) => [row.mutation, row]),
          );
          rows = rows.map((row) => ({ ...row, ...selected.get(row.mutation) }));
          setRawCandidates(rows);
          const atlasData = JSON.parse(await api.readArtifact(id, "umap.json"));
          if (
            !Array.isArray(atlasData.points) ||
            atlasData.points.some(
              (p: { x: number; y: number }) =>
                !Number.isFinite(p.x) || !Number.isFinite(p.y),
            )
          )
            throw new Error("Invalid atlas coordinates.");
          if (!disposed) setAtlas(atlasData);
          const artifacts = await Promise.allSettled([
            api.readArtifact(id, "analysis.txt"),
            run.result?.codon_optimization?.enabled && !isDemo()
              ? api.readArtifact(id, "05_codon_optimized_sequences.csv")
              : Promise.resolve(""),
          ]);
          if (disposed) return;
          setAnalysisText(
            artifacts[0].status === "fulfilled" ? artifacts[0].value : null,
          );
          setAnalysisError(
            artifacts[0].status === "rejected"
              ? message(artifacts[0].reason)
              : null,
          );
          if (artifacts[1].status === "fulfilled" && artifacts[1].value)
            setCodonRows(
              parseCsv(artifacts[1].value) as unknown as HostProfile[],
            );
          if (isDemo() && run.result?.codon_optimization?.enabled) {
            const resolved = JSON.parse(
              await api.readArtifact(id, "00_resolved_request.json"),
            );
            const hosts =
              run.result.codon_optimization.resolved_hosts?.map(
                (host) => host.organism_key,
              ) ?? [];
            const codon = await api.codonOptimize(resolved.sequence, hosts);
            if (disposed) return;
            setCodonRows(codon.results as unknown as HostProfile[]);
          }
        }
        await loadRequest(run);
        finished = true;
      } catch (failure) {
        if (!disposed) {
          setError(message(failure));
          setLoading(false);
        }
      } finally {
        refreshing = false;
      }
    }
    void refresh();
    // Reconcile missed SSE events, reconnects, and simultaneous browser tabs.
    const timer = window.setInterval(() => {
      void refresh();
    }, 5000);
    return () => {
      disposed = true;
      unsubscribe?.();
      window.clearInterval(timer);
    };
  }, [jobId, revision]);

  return {
    atlas,
    generation,
    meta,
    result: meta?.result ?? null,
    rawCandidates,
    wtSequence,
    codonRows,
    logs,
    analysisText,
    analysisError,
    error,
    loading,
    reconnecting,
    retry: () => retry((value) => value + 1),
  };
}

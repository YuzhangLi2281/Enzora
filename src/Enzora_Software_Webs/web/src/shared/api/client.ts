/** REST + SSE client for the FastAPI backend.
 *  - Dev: Vite proxies /api to VITE_PROXY_TARGET (GPU server).
 *  - Prod: VITE_API_BASE_URL, or same-origin when served by the backend.
 *  - Demo mode (VITE_DEMO=1, URL ?demo=1, or header toggle) routes every call
 *    to simulated data so the whole UI works without a backend.
 *  All third-party network access happens on the backend; this client only
 *  ever talks to the agent's own API. */

import { demo } from "../demo/backend";
import { normalizeApiBase, resolveDemoMode } from "./config";
import type {
  ClassifyResult,
  CodonOptimizeResponse,
  HostInfo,
  InputContext,
  LookupCandidate,
  LookupResolved,
  ParseResult,
  RunEvent,
  RunMeta,
  RunRequest,
  ServerSettings,
  ToolInfo,
  SessionStatus,
  TranslationResult,
  ParameterContext,
  PlanPreview,
} from "./contracts";

export const API_BASE = normalizeApiBase(
  window.__ENZYME_CONFIG__?.apiBaseUrl || import.meta.env.VITE_API_BASE_URL,
);

const demoMode = (() => {
  let saved: string | null = null;
  try {
    saved = localStorage.getItem("fea_demo");
    const override = new URLSearchParams(window.location.search).get("demo");
    if (override === "0" || override === "1")
      localStorage.setItem("fea_demo", override);
  } catch {
    /* storage may be disabled */
  }
  return resolveDemoMode(
    window.location.search,
    saved,
    window.__ENZYME_CONFIG__?.demo === true ||
      import.meta.env.VITE_DEMO === "1",
  );
})();
export const isDemo = (): boolean => demoMode;

export function setDemo(on: boolean): void {
  try {
    localStorage.setItem("fea_demo", on ? "1" : "0");
  } catch {
    /* URL remains authoritative */
  }
  const url = new URL(window.location.href);
  url.searchParams.set("demo", on ? "1" : "0");
  url.searchParams.delete("job");
  window.location.assign(url.toString());
}

export function accessToken(): string {
  try { return sessionStorage.getItem("enzora_access_token") || ""; }
  catch { return ""; }
}

export async function login(token: string): Promise<void> {
  const response = await fetch(`${API_BASE}/session/status`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!response.ok) throw new Error(response.status === 401 ? "访问码不正确 / Invalid access code" : "无法连接服务 / Cannot connect to service");
  sessionStorage.setItem("enzora_access_token", token);
}

class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function responseFor(
  path: string,
  init?: RequestInit,
): Promise<Response> {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), 150_000);
  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, {
      ...init,
      headers: {
        ...(init?.body ? { "Content-Type": "application/json" } : {}),
        ...(accessToken() ? { Authorization: `Bearer ${accessToken()}` } : {}),
        ...(init?.headers ?? {}),
      },
      signal: controller.signal,
    });
  } catch (error) {
    throw new ApiError(
      0,
      controller.signal.aborted
        ? "Request timed out. Check the server connection and try again."
        : "Cannot reach the server. Check the API address and network connection.",
    );
  } finally {
    window.clearTimeout(timeout);
  }
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail =
        typeof body.detail === "string"
          ? body.detail
          : JSON.stringify(body.detail ?? body);
    } catch {
      /* keep statusText */
    }
    throw new ApiError(response.status, detail);
  }
  return response;
}

async function http<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await responseFor(path, init);
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

const post = <T>(path: string, body: unknown): Promise<T> =>
  http<T>(path, { method: "POST", body: JSON.stringify(body) });

export const api = {
  chat(
    message: string,
    jobId: string | null,
    sessionId: string | null,
  ): Promise<{
    message: string;
    codon_rows?: Record<string, unknown>[];
    tool_result?: {
      weights: { ph: number; activity: number; stability: number };
      confidence: { ph: number; activity: number; stability: number };
    };
  }> {
    if (isDemo())
      return Promise.resolve({
        message: /[\u3400-\u9fff]/.test(message)
          ? "这是本地演示回复，没有调用 Gemini。你可以设置突变数量与阶数、体验三评分器配平，再通过 Next Process 查看 Top-K 和宿主密码子导出。连接真实后端后，助手才会根据自然语言调用相应工具。"
          : "Demo assistant: no Gemini request was made. Adjust preferences and confidence in the control panel, apply the balance, then use Next Process to advance.",
      });
    return post("/chat", { message, job_id: jobId, session_id: sessionId });
  },
  advanceRun(jobId: string, phase: number): Promise<unknown> {
    if (isDemo()) return Promise.resolve(demo.advanceRun(jobId, phase));
    return post(`/runs/${encodeURIComponent(jobId)}/advance`, { phase });
  },
  balanceRun(
    jobId: string,
    weights: { ph: number; activity: number; stability: number },
    confidence: { ph: number; activity: number; stability: number },
  ): Promise<{ effective_weights: typeof weights }> {
    if (isDemo())
      return Promise.resolve(demo.balanceRun(jobId, weights, confidence));
    return post(`/runs/${encodeURIComponent(jobId)}/balance`, {
      weights,
      confidence,
    });
  },
  // ---- runs ----
  submitRun(
    request: RunRequest,
    sessionId: string | null,
  ): Promise<{ job_id: string }> {
    if (isDemo()) return demo.submitRun(request);
    return post("/runs", { request, session_id: sessionId });
  },
  listRuns(): Promise<RunMeta[]> {
    if (isDemo()) return Promise.resolve(demo.listRuns());
    return http("/runs");
  },
  getRun(jobId: string): Promise<RunMeta> {
    if (isDemo()) return Promise.resolve(demo.getRun(jobId));
    return http(`/runs/${encodeURIComponent(jobId)}`);
  },
  cancelRun(jobId: string): Promise<unknown> {
    if (isDemo()) return Promise.resolve(demo.cancelRun(jobId));
    return post(`/runs/${encodeURIComponent(jobId)}/cancel`, {});
  },
  /** Subscribe to SSE events; returns an unsubscribe function. */
  subscribeRun(
    jobId: string,
    onEvent: (evt: RunEvent) => void,
    onEnd: () => void,
    onConnection?: (connected: boolean) => void,
  ): () => void {
    if (isDemo()) return demo.subscribeRun(jobId, onEvent, onEnd);
    if (window.__ENZYME_CONFIG__?.eventTransport === "poll") {
      let active = true;
      let timer: number | undefined;
      const poll = async () => {
        try {
          const batch = await http<{events: RunEvent[]; finished: boolean}>(`/runs/${encodeURIComponent(jobId)}/events?transport=poll`);
          if (!active) return;
          onConnection?.(true);
          batch.events.forEach(onEvent);
          if (batch.finished) { active = false; onEnd(); return; }
        } catch { if (active) onConnection?.(false); }
        if (active) timer = window.setTimeout(() => void poll(), 1500);
      };
      void poll();
      return () => { active = false; window.clearTimeout(timer); };
    }
    if (accessToken()) {
      const controller = new AbortController();
      void (async () => {
        try {
          const response = await fetch(`${API_BASE}/runs/${encodeURIComponent(jobId)}/events`, {
            headers: { Authorization: `Bearer ${accessToken()}` }, signal: controller.signal,
          });
          if (!response.ok || !response.body) throw new Error("Cannot open progress stream");
          onConnection?.(true);
          const reader = response.body.getReader();
          const decoder = new TextDecoder();
          let buffer = "";
          while (!controller.signal.aborted) {
            const { value, done } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            buffer = buffer.replace(/\r\n/g, "\n");
            let end: number;
            while ((end = buffer.indexOf("\n\n")) >= 0) {
              const frame = buffer.slice(0, end); buffer = buffer.slice(end + 2);
              const data = frame.split("\n").filter((line) => line.startsWith("data:")).map((line) => line.slice(5).trimStart()).join("\n");
              if (!data) continue;
              const payload = JSON.parse(data) as RunEvent;
              if (payload.event === "stream_end") { controller.abort(); onEnd(); return; }
              onEvent(payload);
            }
          }
          if (!controller.signal.aborted) onConnection?.(false);
        } catch { if (!controller.signal.aborted) onConnection?.(false); }
      })();
      return () => controller.abort();
    }
    const source = new EventSource(
      `${API_BASE}/runs/${encodeURIComponent(jobId)}/events`,
    );
    source.onopen = () => onConnection?.(true);
    source.onerror = () => onConnection?.(false);
    const handler = (e: MessageEvent) => {
      try {
        const payload = JSON.parse(e.data) as RunEvent;
        if (payload.event === "stream_end") {
          source.close();
          onEnd();
          return;
        }
        onEvent(payload);
      } catch {
        /* ignore malformed frames */
      }
    };
    source.onmessage = handler;
    return () => source.close();
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
    } = {},
  ): Promise<{
    total: number;
    page: number;
    columns: string[];
    rows: Record<string, unknown>[];
  }> {
    if (isDemo()) return Promise.resolve(demo.getCandidates(jobId, params));
    const query = new URLSearchParams();
    Object.entries(params).forEach(([k, v]) => {
      if (v !== undefined && v !== "") query.set(k, String(v));
    });
    return http(
      `/runs/${encodeURIComponent(jobId)}/candidates?${query.toString()}`,
    );
  },
  rerankRun(
    jobId: string,
    weights: { ph: number; activity: number; stability: number },
    sessionId: string | null,
  ): Promise<{ rows: Record<string, unknown>[] }> {
    if (isDemo())
      return Promise.resolve({ rows: demo.rerankRun(jobId, weights) });
    return post(`/runs/${encodeURIComponent(jobId)}/rerank`, {
      weights,
      session_id: sessionId,
    });
  },
  fileUrl(jobId: string, rel: string): string {
    return `${API_BASE}/runs/${encodeURIComponent(jobId)}/files/${rel.split("/").map(encodeURIComponent).join("/")}`;
  },
  async readArtifact(jobId: string, rel: string): Promise<string> {
    if (isDemo()) return demo.readArtifact(jobId, rel);
    const response = await responseFor(
      `/runs/${encodeURIComponent(jobId)}/files/${rel.split("/").map(encodeURIComponent).join("/")}`,
    );
    return response.text();
  },
  figureUrl(jobId: string, name: string): string {
    return `${API_BASE}/runs/${encodeURIComponent(jobId)}/figures/${encodeURIComponent(name)}`;
  },
  runMetaUrl(jobId: string): string {
    return `${API_BASE}/runs/${encodeURIComponent(jobId)}`;
  },
  // ---- input routing ----
  classify(
    text: string,
    sessionId: string | null = null,
    typeOverride?: string | null,
    context?: InputContext,
  ): Promise<ClassifyResult> {
    if (isDemo()) return Promise.resolve(demo.classify(text, typeOverride));
    return post("/input/classify", {
      text,
      session_id: sessionId,
      type_override: typeOverride || null,
      context: context || null,
    });
  },
  inputFile(
    filename: string,
    content: string,
    sessionId: string | null,
    typeOverride?: string | null,
  ): Promise<ClassifyResult> {
    if (isDemo())
      return Promise.reject(
        new ApiError(
          400,
          "File import requires a live backend. Turn off demo mode.",
        ),
      );
    return post("/input/file", {
      filename,
      content,
      session_id: sessionId,
      type_override: typeOverride || null,
    });
  },
  exportArtifact(bioartifact: NonNullable<ClassifyResult['bioartifact']>, format: string): Promise<{ content: string; warnings: string[] }> {
    return post('/input/export', { bioartifact, format });
  },
  translate(
    sequence: string,
    sequenceType: string,
  ): Promise<TranslationResult> {
    if (isDemo())
      return Promise.resolve(demo.translate(sequence, sequenceType));
    return post("/translate", { sequence, sequence_type: sequenceType });
  },
  codonOptimize(
    proteinSequence: string,
    organismKeys: string[],
  ): Promise<CodonOptimizeResponse> {
    if (isDemo())
      return Promise.resolve(demo.codonOptimize(proteinSequence, organismKeys));
    return post("/codon-optimize", {
      protein_sequence: proteinSequence,
      organism_keys: organismKeys,
    });
  },
  // ---- lookup ----
  lookup(
    query: string,
  ): Promise<{ candidates: LookupCandidate[]; warning?: string }> {
    if (isDemo()) return Promise.resolve(demo.lookup(query));
    return post("/lookup", { query });
  },
  lookupResolve(accession: string, source: string): Promise<LookupResolved> {
    if (isDemo()) return Promise.resolve(demo.lookupResolve(accession, source));
    return post("/lookup/resolve", { accession, source });
  },
  // ---- tools & health ----
  tools(): Promise<ToolInfo[]> {
    if (isDemo()) return Promise.resolve(demo.tools());
    return http("/tools");
  },
  health(
    kind: "check-paths" | "check-gpu" | "validate-cpu",
  ): Promise<Record<string, unknown>> {
    if (isDemo()) return Promise.resolve(demo.health(kind));
    return post(`/health/${kind}`, {});
  },
  ping(): Promise<{ ok: boolean }> {
    if (isDemo()) return Promise.resolve({ ok: true });
    return http("/ping");
  },
  // ---- settings & session ----
  getSettings(): Promise<ServerSettings> {
    if (isDemo()) return Promise.resolve(demo.settings());
    return http("/settings");
  },
  putSettings(partial: Record<string, unknown>): Promise<ServerSettings> {
    if (isDemo()) return Promise.resolve(demo.settings());
    return http("/settings", { method: "PUT", body: JSON.stringify(partial) });
  },
  sessionKey(apiKey: string): Promise<{ session_id: string }> {
    if (isDemo()) return Promise.resolve({ session_id: "demo-session" });
    return post("/session/key", { api_key: apiKey });
  },
  sessionStatus(sessionId: string | null): Promise<SessionStatus> {
    if (isDemo()) return Promise.resolve({ configured: true, source: "demo" });
    const query = sessionId
      ? `?session_id=${encodeURIComponent(sessionId)}`
      : "";
    return http(`/session/status${query}`);
  },
  // ---- llm previews ----
  parseRequest(
    requestText: string,
    candidate: Record<string, unknown>,
    selection: Record<string, unknown>,
    sessionId: string | null,
    context?: ParameterContext,
  ): Promise<ParseResult> {
    if (isDemo())
      return Promise.resolve(
        demo.parseRequest(requestText, selection, candidate, context),
      );
    return post("/parse", {
      request_text: requestText,
      candidate,
      selection,
      session_id: sessionId,
      context,
    });
  },
  plan(
    capabilities: string[],
    useLlm: boolean,
    sessionId: string | null,
    request = "",
    initialArtifacts = ["protein_sequence"],
    inputSummary?: Partial<InputContext>,
  ): Promise<PlanPreview> {
    if (isDemo())
      return Promise.resolve(
        demo.plan(
          capabilities,
          useLlm,
          request,
          initialArtifacts,
        ) as unknown as PlanPreview,
      );
    return post("/plan", {
      capabilities,
      initial_artifacts: initialArtifacts,
      ...(inputSummary ? { input_summary: inputSummary } : {}),
      request,
      use_llm: useLlm,
      session_id: sessionId,
    });
  },
  // ---- hosts ----
  hosts(): Promise<HostInfo[]> {
    if (isDemo()) return Promise.resolve(demo.hosts());
    return http("/hosts");
  },
};

export { ApiError };

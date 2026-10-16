/** Global UI state (Zustand): theme, language, session, selection, run state. */

import { create } from "zustand";
import type {
  Candidate,
  Lang,
  RunRequest,
  Theme,
} from "../shared/api/contracts";

export const DEFAULT_THEME: Theme =
  (import.meta.env.VITE_DEFAULT_THEME as Theme | undefined) === "deep-space"
    ? "deep-space"
    : "macaron";
export const DEFAULT_LANG: Lang =
  (import.meta.env.VITE_DEFAULT_LANG as Lang | undefined) === "zh"
    ? "zh"
    : "en";

const requestedTheme = new URLSearchParams(window.location.search).get("theme");

const THEME_KEY = "fea_theme";
const LANG_KEY = "fea_lang";
const SESSION_KEY = "fea_session";

function readStorage<T>(key: string, fallback: T): T {
  try {
    const value = localStorage.getItem(key);
    return value != null ? (JSON.parse(value) as T) : fallback;
  } catch {
    return fallback;
  }
}

interface AppState {
  theme: Theme;
  lang: Lang;
  sessionId: string | null;
  activeJobId: string | null;
  runRequest: RunRequest | null;
  candidates: Candidate[];
  rawCandidates: Candidate[];
  selectedMutation: string | null;
  weights: { ph: number; activity: number; stability: number };
  weightsDirty: boolean;
  setTheme: (t: Theme) => void;
  setLang: (l: Lang) => void;
  setSessionId: (id: string | null) => void;
  setActiveJob: (id: string | null) => void;
  setRunRequest: (r: RunRequest | null) => void;
  setCandidates: (rows: Candidate[], raw?: Candidate[]) => void;
  setSelectedMutation: (m: string | null) => void;
  setWeights: (
    w: { ph: number; activity: number; stability: number },
    dirty?: boolean,
  ) => void;
  resetWeights: () => void;
}

export const useApp = create<AppState>((set, get) => ({
  theme:
    requestedTheme === "macaron" || requestedTheme === "deep-space"
      ? requestedTheme
      : readStorage<Theme>(THEME_KEY, DEFAULT_THEME),
  lang: readStorage<Lang>(LANG_KEY, DEFAULT_LANG),
  sessionId: sessionStorage.getItem(SESSION_KEY),
  activeJobId: null,
  runRequest: null,
  candidates: [],
  rawCandidates: [],
  selectedMutation: null,
  weights: { ph: 0.3, activity: 0.4, stability: 0.3 },
  weightsDirty: false,

  setTheme: (theme) => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem(THEME_KEY, JSON.stringify(theme));
    set({ theme });
  },
  setLang: (lang) => {
    localStorage.setItem(LANG_KEY, JSON.stringify(lang));
    set({ lang });
  },
  setSessionId: (id) => {
    if (id) sessionStorage.setItem(SESSION_KEY, id);
    else sessionStorage.removeItem(SESSION_KEY);
    set({ sessionId: id });
  },
  setActiveJob: (id) => set({ activeJobId: id }),
  setRunRequest: (r) =>
    set({
      runRequest: r,
      weights: r?.selection.weights ?? get().weights,
      weightsDirty: false,
    }),
  setCandidates: (rows, raw) =>
    set({
      candidates: rows,
      rawCandidates: raw ?? rows,
    }),
  setSelectedMutation: (m) => set({ selectedMutation: m }),
  setWeights: (w, dirty = true) => set({ weights: w, weightsDirty: dirty }),
  resetWeights: () => {
    const request = get().runRequest;
    set({
      weights: request?.selection.weights ?? {
        ph: 0.3,
        activity: 0.4,
        stability: 0.3,
      },
      weightsDirty: false,
    });
  },
}));

export function applyTheme(theme: Theme): void {
  document.documentElement.dataset.theme = theme;
}

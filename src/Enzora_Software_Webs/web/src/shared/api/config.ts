export interface FrontendConfig {
  apiBaseUrl?: string;
  demo?: boolean;
  eventTransport?: "sse" | "poll";
}

declare global {
  interface Window { __ENZYME_CONFIG__?: FrontendConfig }
}

export function normalizeApiBase(value: string | undefined): string {
  const base = value?.trim().replace(/\/+$/, '') || '/api';
  if (base.startsWith('/') && !base.startsWith('//') && !/[?#]/.test(base)) return base;
  const url = new URL(base);
  if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password || url.search || url.hash) {
    throw new Error('API address must be an HTTP(S) URL or a same-origin path, without credentials, query or fragment.');
  }
  return base;
}

export function resolveDemoMode(query: string, saved: string | null, defaultOn: boolean): boolean {
  const override = new URLSearchParams(query).get('demo');
  if (override === '1' || override === '0') return override === '1';
  if (saved === '1' || saved === '0') return saved === '1';
  return defaultOn;
}

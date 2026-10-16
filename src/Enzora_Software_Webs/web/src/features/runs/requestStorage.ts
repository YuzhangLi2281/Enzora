import type { RunRequest } from '../../shared/api/contracts';

// Scientific parameters only. Never cache session IDs or API keys here.
export function rememberRequest(id: string, request: RunRequest) {
  try { sessionStorage.setItem(`fea_request_${id}`, JSON.stringify(request)); } catch { /* optional */ }
}
export function recallRequest(id: string): RunRequest | null {
  try { return JSON.parse(sessionStorage.getItem(`fea_request_${id}`) ?? 'null'); } catch { return null; }
}

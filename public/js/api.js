// Thin fetch wrapper: demo flag, timeouts, one retry on transient errors, GET de-duplication.
import { store } from './store.js';

export class ApiError extends Error {
  constructor(message, status = 0, code = '') { super(message); this.status = status; this.code = code; }
}
const TIMEOUT_MS = 75000;
const inflight = new Map();
const memo = new Map(); // short-lived result cache so revisiting a tab is instant
const MEMO_MS = 60000;

function qs(params = {}) {
  const u = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) if (v !== undefined && v !== null && v !== '') u.set(k, v);
  if (store.settings().demo) u.set('demo', '1');
  const s = u.toString();
  return s ? '?' + s : '';
}

async function request(url, opts, retry) {
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), TIMEOUT_MS);
  try {
    const res = await fetch(url, { ...opts, signal: ctl.signal });
    let body = null;
    try { body = await res.json(); } catch { /* not JSON */ }
    if (!res.ok) {
      if (retry && [502, 504].includes(res.status)) return request(url, opts, false);
      throw new ApiError(body?.error || `Request failed (${res.status})`, res.status, body?.code || '');
    }
    return body;
  } catch (e) {
    if (e instanceof ApiError) throw e;
    if (e.name === 'AbortError') throw new ApiError('This is taking too long. Try fewer stocks or a shorter window.', 0, 'timeout');
    if (retry) return request(url, opts, false);
    throw new ApiError('Network problem. Check your connection and try again.', 0, 'network');
  } finally { clearTimeout(timer); }
}

export function get(path, params, { fresh = false } = {}) {
  const url = '/api' + path + qs(params);
  if (!fresh) {
    const hit = memo.get(url);
    if (hit && Date.now() - hit.t < MEMO_MS) return Promise.resolve(hit.v);
    if (inflight.has(url)) return inflight.get(url);
  }
  const p = request(url, {}, true).then((v) => { memo.set(url, { t: Date.now(), v }); return v; })
    .finally(() => inflight.delete(url));
  inflight.set(url, p);
  return p;
}
export const post = (path, body) => request('/api' + path + qs(), {
  method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
}, false);
export const clearCache = () => memo.clear();

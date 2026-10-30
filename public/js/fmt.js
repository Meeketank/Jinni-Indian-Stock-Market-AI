// Client-side formatting for table cells (the server sends raw numbers + a format name).
const nf = (d) => new Intl.NumberFormat('en-IN', { minimumFractionDigits: d, maximumFractionDigits: d });
const NF = { 0: nf(0), 1: nf(1), 2: nf(2), 3: nf(3) };
const MON = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

export const isNum = (v) => typeof v === 'number' && Number.isFinite(v);
export const dash = '—';

export function inr(v, d = 0) {
  if (!isNum(v)) return dash;
  const a = Math.abs(v), s = v < 0 ? '-' : '';
  if (a >= 1e7) return `${s}₹${NF[2].format(a / 1e7)} Cr`;
  if (a >= 1e5) return `${s}₹${NF[2].format(a / 1e5)} L`;
  return `${s}₹${NF[d].format(a)}`;
}
export const pct = (v, d = 1, signed = false) => !isNum(v) ? dash : `${signed && v > 0 ? '+' : ''}${NF[d].format(v * 100)}%`;
export const num = (v, d = 2) => !isNum(v) ? dash : NF[d].format(v);

export function date(v) {
  if (!v) return dash;
  const m = /^(\d{4})-(\d{2})-(\d{2})/.exec(String(v));
  return m ? `${+m[3]} ${MON[+m[2] - 1]} ${m[1]}` : String(v);
}

// Map status-like words to a badge colour class.
const GREEN = new Set(['STRONG BUY', 'BUY', 'BULLISH', 'TARGET HIT', 'CONFIRMED', 'ACTIVE']);
const RED = new Set(['STRONG SELL', 'SELL', 'BEARISH', 'STOP HIT', 'FAILED']);
const GOLD = new Set(['HOLD', 'NEUTRAL', 'OPEN', 'EXPIRED', 'PRINTED', 'FORMING', 'FILLED']);
export function badgeClass(v) {
  const k = String(v).toUpperCase();
  if (k === 'STRONG BUY') return 'b-sbuy'; if (k === 'STRONG SELL') return 'b-ssell';
  if (GREEN.has(k)) return 'b-buy'; if (RED.has(k)) return 'b-sell'; if (GOLD.has(k)) return 'b-hold';
  return 'b-mute';
}
export const gradeClass = (g) => ({ A: 'b-sbuy', B: 'b-buy', C: 'b-hold', D: 'b-ssell' }[g] || 'b-mute');
export const toneClass = (v) => (isNum(v) ? (v > 0 ? 'up' : v < 0 ? 'down' : '') : '');

// Plain-text version of a cell (used for sorting, CSV export and tooltips).
export function plain(kind, v) {
  if (v === null || v === undefined || v === '') return dash;
  switch (kind) {
    case 'int': case 'n0': return isNum(v) ? NF[0].format(v) : dash;
    case 'n1': return num(v, 1); case 'n2': return num(v, 2); case 'n3': return num(v, 3);
    case 'pct0': case 'bar01': return pct(v, 0); case 'pct1': return pct(v, 1); case 'pct2': return pct(v, 2);
    case 'spct1': return pct(v, 1, true); case 'spct2': return pct(v, 2, true);
    case 'inr0': return inr(v, 0); case 'inr2': return inr(v, 2);
    case 'bar100': return isNum(v) ? NF[0].format(v) : dash;
    case 'tone': return isNum(v) ? (v > 0 ? '+' : '') + NF[2].format(v) : dash;
    case 'date': return date(v);
    case 'bool': return v ? '✓' : '—';
    default: return String(v);
  }
}

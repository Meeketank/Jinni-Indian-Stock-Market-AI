// Everything the user owns lives in this browser (localStorage). The server is stateless.
const P = 'jinni.';
const SAMPLE_HOLDINGS = [
  ['RELIANCE.NS', 40, 1250], ['HDFCBANK.NS', 50, 800], ['ICICIBANK.NS', 60, 1150], ['INFY.NS', 45, 1500],
  ['TCS.NS', 18, 3600], ['BHARTIARTL.NS', 35, 1500], ['LT.NS', 20, 3300], ['ITC.NS', 150, 420],
  ['SUNPHARMA.NS', 40, 1600], ['MARUTI.NS', 5, 11500],
].map(([ticker, quantity, avg_cost]) => ({ ticker, quantity, avg_cost }));

export const DEFAULT_RULES = {
  max_loss_per_trade: 10000, max_position_pct: 0.10, max_sector_pct: 0.35, min_reward_risk: 1.5,
  cash_buffer_pct: 0.05, max_open_trades: 8, min_probability: 0.0, require_stop: true,
};
const DEFAULT_SETTINGS = { rf: 0.065, capital: 1000000, risk_pct: 0.01, demo: false };

function read(key, fallback) {
  try { const v = localStorage.getItem(P + key); return v === null ? fallback : JSON.parse(v); } catch { return fallback; }
}
function write(key, value) {
  try { localStorage.setItem(P + key, JSON.stringify(value)); return true; } catch { return false; }
}

export const store = {
  settings: () => ({ ...DEFAULT_SETTINGS, ...read('settings', {}) }),
  saveSettings(patch) { const s = { ...this.settings(), ...patch }; write('settings', s); return s; },
  rules: () => ({ ...DEFAULT_RULES, ...read('rules', {}) }),
  saveRules(r) { write('rules', r); },
  holdings: () => read('holdings', null) || SAMPLE_HOLDINGS.map((h) => ({ ...h })),
  hasOwnHoldings: () => read('holdings', null) !== null,
  saveHoldings(h) { write('holdings', h); },
  resetHoldings() { try { localStorage.removeItem(P + 'holdings'); } catch { /* ignore */ } },
  watchlist: () => read('watchlist', []),
  addWatch(t) { const w = this.watchlist(); if (!w.includes(t)) { w.push(t); write('watchlist', w); } return w; },
  removeWatch(t) { const w = this.watchlist().filter((x) => x !== t); write('watchlist', w); return w; },
  journal: () => read('journal', []),
  addPredictions(rows) {
    const j = this.journal(); let n = 0;
    let id = j.reduce((m, p) => Math.max(m, +p.id || 0), 0);
    for (const r of rows) {
      if (j.some((p) => p.ticker === r.ticker && p.entry_date === r.entry_date && p.target_pct === r.target_pct && p.horizon === r.horizon)) continue;
      j.push({ ...r, id: ++id }); n++;
    }
    write('journal', j); return n;
  },
  removePrediction(id) { write('journal', this.journal().filter((p) => String(p.id) !== String(id))); },
  clearJournal() { write('journal', []); },
  last: (key, fb) => read('last.' + key, fb),
  saveLast: (key, v) => write('last.' + key, v),
  exportAll() {
    return { version: 1, exported: new Date().toISOString(), settings: this.settings(), rules: this.rules(),
      holdings: read('holdings', null), watchlist: this.watchlist(), journal: this.journal() };
  },
  importAll(d) {
    if (!d || typeof d !== 'object') throw new Error('Not a JINNI backup file.');
    if (d.settings) write('settings', d.settings); if (d.rules) write('rules', d.rules);
    if (d.holdings) write('holdings', d.holdings); if (Array.isArray(d.watchlist)) write('watchlist', d.watchlist);
    if (Array.isArray(d.journal)) write('journal', d.journal);
  },
};

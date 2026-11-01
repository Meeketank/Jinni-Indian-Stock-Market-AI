import { h, pageHead, Segmented, Select, Span, SPAN_UNITS, tradingDays, NumberInput, TextInput, Slider, Toggle, FileText, Results, card, formGrid, button, disclaimer,
  catalog, toast, icon, renderBlocks, md } from '../ui.js';
import { get, post } from '../api.js';
import { store } from '../store.js';
import { ctxNav, go, showIf } from './kit.js';

const CHUNK = 100, PARALLEL = 4;

function symbolsFromCsv(text) {
  const lines = text.split(/\r?\n/).filter(Boolean); if (!lines.length) return [];
  const head = lines[0].split(',').map((x) => x.replace(/"/g, '').trim().toUpperCase());
  const idx = head.indexOf('SYMBOL');
  return lines.slice(idx >= 0 ? 1 : 0).map((l) => l.split(',')[idx >= 0 ? idx : 0].replace(/"/g, '').trim()).filter(Boolean);
}
const openTrades = () => store.journal().filter((p) => p.source === 'plan' &&
  new Date(p.entry_date).getTime() + p.horizon * 1.45 * 864e5 >= Date.now()).length;

export async function render(view) {
  const cat = await catalog();
  const last = store.last('finder', {});
  const s = store.settings();
  view.append(pageHead('Return Finder', 'Pick a goal. JINNI estimates, for every stock, the odds of reaching it before the stop-loss using three independent engines, and says plainly when it is a coin flip.'));

  const target = Segmented({ label: 'Target gain', options: [['0.02', '2%'], ['0.05', '5%'], ['0.07', '7%'], ['0.1', '10%'], ['0.15', '15%'], ['custom', 'Custom']], value: last.target || '0.05' });
  const targetC = NumberInput({ label: 'Custom target (%)', value: last.targetC || 7, min: 0.5, max: 100, step: 0.5 });
  const lw = last.within || {};
  const within = Span({ label: 'Within', n: lw.n || 1, unit: lw.unit || 'month', units: SPAN_UNITS.slice(0, 5), max: 250,
    presets: [[1, 'week'], [2, 'week'], [1, 'month'], [3, 'month'], [6, 'month'], [1, 'year']], help: 'Any count of days, weeks, months, quarters or years (converted to trading days).' });
  const stop = Segmented({ label: 'Stop-loss', options: [['auto', 'Auto'], ['fixed', 'Fixed %'], ['none', 'None']], value: last.stop || 'auto',
    help: 'Auto: volatility-based (at least 1×ATR), no wider than target ÷ 1.5. None: just the odds of touching the target.' });
  const stopF = NumberInput({ label: 'Stop (%)', value: last.stopF || 3, min: 0.5, max: 50, step: 0.5 });
  const universe = Select({ label: 'Search in', value: last.universe || 'nifty50', options: [['market', 'All listed stocks (NSE + BSE)'], ['nifty50', 'NIFTY 50'],
    ['holdings', 'My holdings'], ['watchlist', 'My watchlist'], ['custom', 'Custom list']] });
  const custom = TextInput({ label: 'Symbols', value: last.custom || '', placeholder: 'e.g. IRCTC, HAL, TRENT', help: 'Comma separated NSE symbols, up to 70.' });
  const budget = NumberInput({ label: 'My budget (₹, 0 = no limit)', value: last.budget ?? 0, min: 0, step: 500, help: 'Only stocks you can afford at least one share of are considered, and JINNI builds a whole-share plan.' });
  const maxNames = Slider({ label: 'Split the budget across up to', min: 1, max: 5, value: last.maxNames || 3, fmt: (v) => `${v} stock${v > 1 ? 's' : ''}` });
  const deepN = Slider({ label: 'Market scan: deep-analyse the top', min: 20, max: 60, step: 10, value: last.deepN || 40, fmt: (v) => `${v} stocks`,
    help: 'Stage 1 screens every listed stock; the best N go through all three engines.' });
  const ml = Toggle({ label: 'Include ML engine (slower, more thorough)', value: last.ml ?? true });
  const minPx = NumberInput({ label: 'Min price (₹)', value: last.minPx || 20, min: 1, step: 5 });
  const minVal = NumberInput({ label: 'Min daily traded value (₹ Cr)', value: last.minVal || 1, min: 0.05, step: 0.25, help: 'Filters out thin stocks you cannot enter or exit at the shown price.' });
  const sme = Toggle({ label: 'Include SME board', value: false });
  let uploaded = null; const upNote = h('div', { class: 'help' }, 'Optional: upload NSE EQUITY_L.csv to override the stock list.');
  const upload = FileText({ label: 'Stock list override', button: 'Upload CSV', onText: (t, name) => { uploaded = symbolsFromCsv(t); upNote.textContent = `${name}: ${uploaded.length} symbols loaded.`; } });
  const filters = h('details', { class: 'acc' }, h('summary', {}, 'Market-scan filters'), h('div', { class: 'in' }, formGrid(minPx, minVal, sme, upload, upNote)));

  const sync = () => {
    showIf(targetC, () => target.get() === 'custom'); 
    showIf(stopF, () => stop.get() === 'fixed'); showIf(custom, () => universe.get() === 'custom');
    showIf(deepN, () => universe.get() === 'market'); filters.classList.toggle('hidden', universe.get() !== 'market');
  };
  [target, stop, universe].forEach((c) => c.onChange(sync)); sync();
  const runBtn = button('Find opportunities', () => run(), 'primary block', 'search');
  view.append(card(formGrid(target, targetC, within, stop, stopF, universe, custom, budget, maxNames, deepN, ml), filters, h('div', { style: { marginTop: '12px' } }, runBtn)));

  const res = new Results(); const out = h('div'); const detail = new Results();
  view.append(res.el, out, detail.el);
  let state = null;

  const params = () => {
    const t = target.get() === 'custom' ? (targetC.get() || 7) / 100 : +target.get();
    const hz = Math.min(250, Math.max(2, tradingDays(within.get())));
    const st = stop.get() === 'fixed' ? String((stopF.get() || 3) / 100) : stop.get();
    return { t, hz, st };
  };

  async function marketScan(p) {
    res.loading('Scanning the entire NSE market…', 'Loading the stock list');
    const uni = await get('/market/universe', { sme: sme.get() ? 1 : 0, min_val: minVal.get() || 1, min_px: minPx.get() || 20 });
    let tickers = uni.rows.map((r) => r[0]);
    if (uploaded?.length) tickers = uploaded.map((x) => (x.includes('.') ? x : `${x}.NS`));
    const chunks = []; for (let i = 0; i < tickers.length; i += CHUNK) chunks.push(tickers.slice(i, i + CHUNK));
    const rows = []; let done = 0, failed = 0, next = 0, lastErr = null;
    const worker = async () => {
      while (next < chunks.length) {
        const c = chunks[next++];
        try {
          const d = await get('/market/scan', { tickers: c.join(','), target: p.t, hz: p.hz, budget: budget.get() || 0, min_px: minPx.get() || 20, min_val: minVal.get() || 1 });
          rows.push(...d.rows);
        } catch (e) { failed++; lastErr = e; }
        done++; res.progress(done / chunks.length, `Screened ${Math.min(done * CHUNK, tickers.length).toLocaleString('en-IN')} of ${tickers.length.toLocaleString('en-IN')} stocks`);
      }
    };
    await Promise.all(Array.from({ length: PARALLEL }, worker));
    if (!rows.length) throw lastErr || new Error('No price data could be loaded.');
    const ok = rows.filter((r) => !r.why).sort((a, b) => b.q - a.q);
    const reasons = {}; rows.filter((r) => r.why).forEach((r) => r.why.split('; ').forEach((w) => { reasons[w] = (reasons[w] || 0) + 1; }));
    const top = ok.slice(0, deepN.get()).map((r) => r.t);
    state.funnel = { listed: uni.listed || uni.count, source: uni.source, scanned: rows.length, passed: ok.length, deep: top.length, reasons, failed: failed * CHUNK };
    if (!top.length) throw new Error('No stock passed the filters. Try a bigger budget, a lower liquidity filter or a longer window.');
    return top;
  }

  async function run() {
    detail.clear(); out.replaceChildren(); state = { funnel: null };
    const p = params(); let tickers;
    store.saveLast('finder', { target: target.get(), targetC: targetC.get(), within: within.get(), stop: stop.get(), stopF: stopF.get(),
      universe: universe.get(), custom: custom.get(), budget: budget.get(), maxNames: maxNames.get(), deepN: deepN.get(), ml: ml.get(), minPx: minPx.get(), minVal: minVal.get() });
    runBtn.disabled = true;
    try {
      const u = universe.get();
      if (u === 'market') tickers = await marketScan(p);
      else {
        tickers = u === 'nifty50' ? cat.nifty50.map((x) => x.t) : u === 'holdings' ? store.holdings().map((x) => x.ticker)
          : u === 'watchlist' ? store.watchlist() : custom.get().split(/[\s,]+/).filter(Boolean);
        if (!tickers.length) { res.error(new Error(u === 'watchlist' ? 'Your watchlist is empty. Add stocks from the Analyzer.' : 'No stocks to search.')); return; }
        tickers = tickers.slice(0, 70);
      }
      res.loading(`Running ${tickers.length} stocks through 3 engines…`, 'Time Machine · history replay · ML model');
      const d = await get('/finder', { target: p.t, hz: p.hz, stop: p.st, tickers: tickers.join(','), ml: ml.get() ? 1 : 0, budget: budget.get() || 0, max_names: maxNames.get() });
      state.doc = d; state.p = p; show();
    } catch (e) { res.error(e, run); } finally { runBtn.disabled = false; }
  }

  function show() {
    const { doc: d, p, funnel } = state;
    res.clear();
    const byTicker = Object.fromEntries((d.data?.rows || []).map((r) => [r.ticker, r]));
    const ctx = { onSelect: (action, key) => { if (action === 'detail') openDetail(key, byTicker[key]); else ctxNav.onSelect(action, key); } };
    out.replaceChildren();
    if (funnel) {
      out.append(h('div', { class: 'callout info' }, h('div', { html: md(`**Whole-market scan:** ${funnel.listed.toLocaleString('en-IN')} listed → ${funnel.passed.toLocaleString('en-IN')} passed filters → ${funnel.deep} deep-analysed.`) }),
        h('div', { class: 'caption' }, `Stock list: ${funnel.source}. ` + (funnel.failed ? `${funnel.failed} symbols could not be loaded (rate limit or delisted). ` : '') +
          (Object.keys(funnel.reasons).length ? 'Filtered out: ' + Object.entries(funnel.reasons).map(([k, v]) => `${k} (${v.toLocaleString('en-IN')})`).join(', ') + '.' : ''))));
    }
    out.append(renderBlocks(d.blocks, ctx));
    const act = h('div', { class: 'row', style: { margin: '10px 0' } });
    const rows = (d.data?.rows || []).filter((r) => ['A', 'B'].includes(r.Grade)).slice(0, 5);
    act.append(button('Track top 5 (A/B) in Journal', () => {
      const n = store.addPredictions(rows.map((r) => trackRow(r, d.data, 0, 'pick'))); toast(n ? `Tracking ${n} new predictions` : 'No new A/B picks to track');
    }, '', 'plus'));
    for (const [name, items] of Object.entries(d.data?.plans || {})) {
      act.append(button(`Track “${name}” in Journal`, () => {
        const n = store.addPredictions(items.map((it) => trackRow(byTicker[it.ticker], d.data, it.shares, 'plan'))); toast(`Tracking ${n} positions`);
      }, '', 'plus'));
    }
    out.append(act);
    if (!detail.el.childNodes.length) out.append(h('div', { class: 'caption' }, 'Tap a stock in the table for its price cone, history, sizing and risk check.'));
  }

  function trackRow(r, d, qty, source) {
    return { ticker: r.ticker, entry: r.Price, target_pct: d.target, stop_pct: r['Stop %'], target_price: r['Target ₹'], stop_price: r['Stop ₹'],
      horizon: d.hz, p_target: r.p_target, p_stop: r['P(stop first)'], grade: r.Grade, verdict: r.Verdict, qty, entry_date: r.as_of, source };
  }

  async function openDetail(ticker, row) {
    const { p } = state; const rules = store.rules(); const st = store.settings();
    detail.loading(`Loading ${ticker.replace('.NS', '')}…`); detail.el.scrollIntoView({ behavior: 'smooth', block: 'start' });
    try {
      const d = await post('/finder/detail', { ticker, target: p.t, hz: p.hz, stop: p.st, capital: st.capital, rules, holdings: store.holdings().filter((x) => +x.quantity > 0).map((x) => ({ ticker: x.ticker, quantity: +x.quantity, avg_cost: x.avg_cost ? +x.avg_cost : null })),
        open_trades: openTrades(), row });
      detail.show([{ type: 'section', title: 'Deep dive & trade plan' }, ...d.blocks], ctxNav);
      const bar = h('div', { class: 'row', style: { margin: '8px 0 24px' } });
      if (d.data?.track) bar.append(button('Track this trade plan', () => { const n = store.addPredictions([{ ...d.data.track, allowed: undefined }]); toast(n ? 'Tracked. See Journal.' : 'Already tracked today.'); }, 'primary', 'plus'));
      if (d.data?.track && !d.data.track.allowed) bar.lastChild.disabled = true;
      bar.append(h('a', { class: 'btn', href: `#/stock/${encodeURIComponent(ticker)}` }, icon('chart'), 'Open full analysis'));
      detail.el.append(bar);
    } catch (e) { detail.error(e, () => openDetail(ticker, row)); }
  }
  view.append(disclaimer());
}

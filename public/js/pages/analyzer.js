import { h, icon, pageHead, SymbolInput, Segmented, Select, Span, SPAN_UNITS, tradingDays, NumberInput, Slider, Toggle, Chips, MultiPick, tabsUI, disclaimer,
  catalog, niftySuggest, button, toast, card } from '../ui.js';
import { get } from '../api.js';
import { store } from '../store.js';
import { livePanel, go } from './kit.js';

const LAYER_DEFAULT = ['Candlestick patterns', 'Chart patterns', 'Trendlines', 'Support/resistance zones'];

export async function render(view, params) {
  const [cat, sugg] = await Promise.all([catalog(), niftySuggest()]);
  const ticker = decodeURIComponent(params[0] || store.last('stock', 'RELIANCE.NS'));
  store.saveLast('stock', ticker);
  view.append(pageHead('Analyzer', 'Charts with auto-detected patterns, 87 indicators, odds, seasonality and a trade plan for any NSE/BSE stock.'));
  const sym = SymbolInput({ label: 'Stock', value: ticker, suggestions: sugg });
  sym.onChange(() => { const v = sym.get(); if (v) go(`#/stock/${encodeURIComponent(v.toUpperCase())}`); });
  const watch = () => {
    const on = store.watchlist().includes(ticker);
    if (on) { store.removeWatch(ticker); toast('Removed from watchlist'); } else { store.addWatch(ticker); toast('Added to watchlist'); }
    wbtn.replaceChildren(icon('star'), store.watchlist().includes(ticker) ? 'On watchlist' : 'Add to watchlist');
  };
  const wbtn = button(store.watchlist().includes(ticker) ? 'On watchlist' : 'Add to watchlist', watch, '', 'star');
  view.append(card(h('div', { class: 'row', style: { alignItems: 'flex-end' } }, h('div', { class: 'grow' }, sym.el), wbtn)));

  const T = encodeURIComponent(ticker);
  const api = (tab, q) => get(`/stock/${T}/${tab}`, q);
  const settings = store.settings();
  const group = (kind) => cat.categories.map((c) => ({ name: c, items: cat.indicators.filter((i) => i.kind === kind && i.cat === c).map((i) => ({ name: i.name, desc: i.desc })) })).filter((g) => g.items.length);
  const tabs = [
    { label: '360° view', load: (p) => livePanel(p, [], () => api('overview')) },
    { label: 'Chart', load: (p) => {
      const ov = MultiPick({ label: 'Overlays', groups: group('overlay'), value: ['EMA 20', 'SMA 50', 'SMA 200'], max: 14, noun: 'overlays' });
      const pn = MultiPick({ label: 'Indicator panels', groups: group('panel'), value: ['RSI 14', 'MACD'], max: 4, noun: 'indicators' });
      const ly = Chips({ label: 'Auto-detected patterns to draw', options: cat.layers, value: LAYER_DEFAULT, cls: 'full' });
      const zm = Span({ label: 'Time range', n: 6, unit: 'month' });
      const hk = Toggle({ label: 'Heikin-Ashi candles', value: false });
      livePanel(p, [zm, ov, pn, ly, hk], () => api('chart', { overlays: ov.get().join(','), panels: pn.get().join(','),
        layers: ly.get().join(','), ...zm.get(), heikin: hk.get() ? 1 : 0 }));
    } },
    { label: 'Patterns', load: (p) => { const zm = Span({ label: 'Look for patterns in the last', n: 6, unit: 'month' });
      const hz = Segmented({ label: 'Look-ahead for history stats (trading days)', options: [5, 10, 20], value: 10 });
      livePanel(p, [zm, hz], () => api('patterns', { h: hz.get(), ...zm.get() })); } },
    { label: 'Indicators', load: (p) => livePanel(p, [], () => api('indicators')) },
    { label: 'Outlook', load: (p) => { const hz = Segmented({ label: 'Horizon', options: [[1, 'Next session'], [5, '5 sessions'], [10, '10 sessions']], value: 1 });
      livePanel(p, [hz], () => api('outlook', { h: hz.get() })); } },
    { label: 'Return odds', load: (p) => {
      const g = Segmented({ label: 'Goal', options: [[0.02, '+2%'], [0.03, '+3%'], [0.05, '+5%'], [0.07, '+7%'], [0.1, '+10%'], [0.15, '+15%'], [0.2, '+20%']], value: 0.05 });
      const w = Span({ label: 'Within', n: 1, unit: 'month', units: SPAN_UNITS.slice(0, 5), presets: [[1, 'week'], [2, 'week'], [1, 'month'], [3, 'month'], [6, 'month'], [1, 'year']], max: 250 });
      const wrap = livePanel(p, [g, w], () => api('odds', { target: g.get(), hz: Math.min(250, tradingDays(w.get())) }));
      p.append(h('a', { href: '#/finder', class: 'btn small' }, icon('target'), 'Compare against every stock in Return Finder'));
    } },
    { label: 'Seasonality', load: (p) => livePanel(p, [], () => api('seasonality')) },
    { label: 'Trade plan', load: (p) => {
      const side = Segmented({ label: 'Side', options: ['LONG', 'SHORT'], value: 'LONG' });
      const cap = NumberInput({ label: 'Capital (₹)', value: settings.capital, min: 1000, step: 50000 });
      const rk = Slider({ label: 'Risk per trade', min: 0.25, max: 3, step: 0.25, value: settings.risk_pct * 100, fmt: (v) => `${v}%` });
      livePanel(p, [side, cap, rk], () => api('plan', { side: side.get(), capital: cap.get() || settings.capital, risk_pct: rk.get() / 100 }));
    } },
    { label: 'Fundamentals', load: (p) => livePanel(p, [], () => api('fundamentals')) },
    { label: 'News & sentiment', load: (p) => livePanel(p, [], () => api('news')) },
  ];
  const t = tabsUI(tabs, { onShow: (i, panel, it) => it.load(panel) });
  view.append(t.el, disclaimer());
}

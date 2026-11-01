import { h, pageHead, Select, Segmented, TextInput, NumberInput, Chips, MultiPick, Span, SPAN_UNITS, tradingDays, tabsUI, disclaimer, catalog } from '../ui.js';
import { get } from '../api.js';
import { livePanel } from './kit.js';

export async function render(view) {
  const cat = await catalog();
  view.append(pageHead('Strategy Lab', 'Backtest simple rules after Indian costs, and see how an ML model really performs on data it has never seen.'));
  const stocks = cat.nifty50.map((x) => [x.t, x.name]);
  const tabs = [
    { label: 'System accuracy', load: (p) => {
      p.append(h('p', { class: 'page-sub' }, 'Walk-forward test of the "chance of reaching my target" model on real price history: what it said, and what then happened.'));
      const g = Segmented({ label: 'Target gain', options: [[0.02, '2%'], [0.05, '5%'], [0.1, '10%']], value: 0.05 });
      const w = Span({ label: 'Within', n: 1, unit: 'month', units: SPAN_UNITS.slice(0, 4), max: 63, presets: [[1, 'week'], [2, 'week'], [1, 'month'], [3, 'month']] });
      livePanel(p, [g, w], () => get('/accuracy', { target: g.get(), h: Math.min(63, Math.max(3, tradingDays(w.get()))) }));
    } },
    { label: 'Backtest rules', load: (p) => {
      const t = Select({ label: 'Stock', options: stocks, value: stocks[0]?.[0] });
      const per = Segmented({ label: 'Period', options: ['3y', '5y', '10y'], value: '5y' });
      const cost = NumberInput({ label: 'Cost per side (bps)', value: 15, min: 0, max: 100, step: 1 });
      const st = Chips({ label: 'Strategies', options: Object.entries(cat.strategies).map(([k, v]) => [k, v.split(':')[0]]), value: Object.keys(cat.strategies), cls: 'full' });
      livePanel(p, [t, per, cost, st], () => get('/lab/backtest', { ticker: t.get(), period: per.get(), cost: cost.get() ?? 15, strategies: st.get().join(',') }), { runLabel: 'Run backtest' });
    } },
    { label: 'ML model (walk-forward)', load: (p) => {
      p.append(h('p', { class: 'page-sub' }, 'Gradient-boosted trees predict whether a stock will be up over the next h days. Validation is walk-forward with a purge gap: train only on the past, test on the following unseen block. Compare accuracy with the base rate. Daily stock direction is close to a coin flip, and an honest model shows that.'));
      const pool = MultiPick({ label: 'Train on', groups: [{ name: 'NIFTY 50', items: cat.nifty50.map((x) => ({ name: x.t, desc: x.name })) }], value: ['RELIANCE.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'INFY.NS', 'TCS.NS', 'ITC.NS'], max: 8, noun: 'stocks' });
      const extra = TextInput({ label: 'Extra symbols', value: '', placeholder: 'e.g. IRCTC, HAL' });
      const hz = Select({ label: 'Horizon (days)', options: [1, 5, 10, 20], value: 5 });
      const tg = Select({ label: 'Target', options: [[0, 'Absolute (stock up)'], [1, 'Relative (beats NIFTY)']], value: 0 });
      const fo = Select({ label: 'Folds', options: [3, 4, 5, 6], value: 5 });
      livePanel(p, [pool, extra, hz, tg, fo], () => get('/lab/ml', { tickers: [...pool.get(), ...extra.get().split(/[\s,]+/).filter(Boolean)].slice(0, 10).join(','), hz: hz.get(), relative: tg.get(), folds: fo.get() }),
        { runLabel: 'Train & validate' });
    } },
  ];
  view.append(tabsUI(tabs, { onShow: (i, panel, it) => it.load(panel) }).el, disclaimer());
}

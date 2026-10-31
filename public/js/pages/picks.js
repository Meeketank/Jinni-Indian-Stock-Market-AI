import { h, pageHead, NumberInput, Segmented, Slider, Select, Span, SPAN_UNITS, tradingDays, disclaimer, card } from '../ui.js';
import { get } from '../api.js';
import { store } from '../store.js';
import { livePanel } from './kit.js';

export async function render(view) {
  view.append(pageHead('Smart Picks', 'A systematic, validated stock selection: every stock is ranked on proven signals, tested walk-forward against the whole market, then built into a risk-aware basket for your budget.'));
  const last = store.last('picks', {});
  const budget = NumberInput({ label: 'Budget (₹)', value: last.budget || store.settings().capital / 10, min: 1000, step: 5000 });
  const profile = Segmented({ label: 'Risk style', options: [['conservative', 'Conservative'], ['balanced', 'Balanced'], ['aggressive', 'Aggressive']], value: last.profile || 'balanced',
    help: 'Conservative minimises volatility, balanced equalises each stock\'s risk, aggressive leans into the top scores.' });
  const k = Slider({ label: 'Number of stocks', min: 3, max: 20, value: last.k || 8, fmt: (v) => `${v}` });
  const hold = Span({ label: 'Holding / rebalance period', n: last.hn || 1, unit: last.hu || 'month', units: SPAN_UNITS.slice(0, 3), max: 90,
    presets: [[2, 'week'], [1, 'month'], [2, 'month'], [3, 'month']] });
  const uni = Select({ label: 'Universe (largest NSE stocks by market cap)', options: [[60, 'Top 60'], [120, 'Top 120'], [200, 'Top 200']], value: last.n || 120 });
  const save = () => store.saveLast('picks', { budget: budget.get(), profile: profile.get(), k: k.get(), hn: hold.get().n, hu: hold.get().unit, n: uni.get() });
  livePanel(view, [budget, profile, k, hold, uni], () => {
    save();
    return get('/select', { budget: budget.get() || 100000, profile: profile.get(), k: k.get(), n: uni.get(), h: Math.min(63, Math.max(5, tradingDays(hold.get()))) });
  }, { runLabel: 'Build my picks' });
  view.append(disclaimer());
}

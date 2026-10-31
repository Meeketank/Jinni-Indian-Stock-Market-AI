import { h, pageHead, Select, Segmented, TextInput, Slider, NumberInput, Toggle, Results, card, formGrid, button, disclaimer, catalog } from '../ui.js';
import { post } from '../api.js';
import { store } from '../store.js';
import { ctxNav, holdingsForApi, showIf } from './kit.js';

export async function render(view) {
  const cat = await catalog();
  view.append(pageHead('Optimizer', 'Build portfolios for best risk-adjusted return, lowest volatility or equal risk, with caps per stock and sector.'));
  const s = store.settings();
  const src = Segmented({ label: 'Universe', options: [['holdings', 'My holdings'], ['nifty50', 'NIFTY 50'], ['custom', 'Custom']], value: 'holdings' });
  const custom = TextInput({ label: 'Symbols', value: 'RELIANCE, HDFCBANK, INFY, ITC, LT, SUNPHARMA, TITAN, NTPC, BHARTIARTL', help: 'Comma separated, 2–60.' });
  const obj = Select({ label: 'Objective', options: Object.entries(cat.objectives), value: 'max_sharpe' });
  const mu = Select({ label: 'Expected returns', options: [['capm', 'CAPM equilibrium (beta × 7% ERP)'], ['shrunk', 'Historical, shrunk 50% to the mean'], ['historical', 'Historical (noisy)']], value: 'capm' });
  const maxW = Slider({ label: 'Max weight per stock', min: 5, max: 100, value: 25, fmt: (v) => `${v}%` });
  const minW = Slider({ label: 'Min weight per stock', min: 0, max: 5, step: 0.5, value: 0, fmt: (v) => `${v}%` });
  const capOn = Toggle({ label: 'Cap sector weight', value: true });
  const cap = Slider({ label: 'Sector cap', min: 10, max: 100, step: 5, value: 30, fmt: (v) => `${v}%` });
  const look = Select({ label: 'Estimation window', options: ['1y', '2y', '3y', '5y'], value: '3y' });
  const capital = NumberInput({ label: 'Capital to deploy (₹)', value: s.capital, min: 1000, step: 100000 });
  const sync = () => { showIf(custom, () => src.get() === 'custom'); showIf(cap, () => capOn.get()); }; [src, capOn].forEach((c) => c.onChange(sync)); sync();
  const res = new Results();
  const run = async () => {
    const hold = holdingsForApi();
    const tickers = src.get() === 'nifty50' ? cat.nifty50.map((x) => x.t) : src.get() === 'holdings' ? hold.map((x) => x.ticker) : custom.get().split(/[\s,]+/).filter(Boolean);
    if (tickers.length < 2) { res.error(new Error('Pick at least two stocks.')); return; }
    res.loading('Estimating covariance and solving…', 'Ledoit-Wolf shrinkage · constrained optimisation · efficient frontier');
    try {
      const d = await post('/optimizer', { tickers, holdings: src.get() === 'holdings' ? hold : [], objective: obj.get(), mu_method: mu.get(), max_w: maxW.get() / 100,
        min_w: minW.get() / 100, sector_cap: capOn.get() ? cap.get() / 100 : null, lookback: look.get(), capital: capital.get() || s.capital, rf: s.rf });
      res.show(d.blocks, ctxNav);
    } catch (e) { res.error(e, run); }
  };
  view.append(card(formGrid(src, custom, obj, mu, maxW, minW, capOn, cap, look, capital), h('div', { style: { marginTop: '12px' } }, button('Optimise', run, 'primary block', 'tune'))), res.el, disclaimer());
}

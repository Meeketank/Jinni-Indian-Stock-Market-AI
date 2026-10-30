import { h, pageHead, Select, Slider, Results, card, formGrid, button, disclaimer, FileText, toast, download, icon } from '../ui.js';
import { post } from '../api.js';
import { store } from '../store.js';
import { ctxNav, holdingsForApi } from './kit.js';

export async function render(view) {
  view.append(pageHead('Portfolio Risk', 'VaR and CVaR four ways, stress tests, risk contribution, beta and liquidity for your holdings.'));
  let rows = store.holdings().map((x) => ({ ...x }));
  const list = h('div'); const note = h('div', { class: 'caption' });
  const paint = () => {
    list.replaceChildren(...rows.map((r, i) => h('div', { class: 'edit-row' },
      h('input', { type: 'text', value: r.ticker, placeholder: 'Symbol e.g. TCS', 'aria-label': 'Symbol', autocapitalize: 'characters', oninput: (e) => { r.ticker = e.target.value.trim().toUpperCase(); } }),
      h('input', { type: 'number', value: r.quantity, min: 0, placeholder: 'Qty', inputmode: 'decimal', 'aria-label': 'Quantity', oninput: (e) => { r.quantity = e.target.value; } }),
      h('input', { type: 'number', value: r.avg_cost ?? '', min: 0, placeholder: 'Avg cost ₹', inputmode: 'decimal', 'aria-label': 'Average cost', oninput: (e) => { r.avg_cost = e.target.value; } }),
      h('button', { type: 'button', class: 'btn small danger del', 'aria-label': 'Remove row', onclick: () => { rows.splice(i, 1); paint(); } }, icon('trash')))));
    note.textContent = store.hasOwnHoldings() ? '' : 'Sample portfolio shown until you save your own.';
  };
  paint();
  const clean = () => rows.filter((r) => r.ticker && +r.quantity > 0).map((r) => ({ ticker: r.ticker, quantity: +r.quantity, avg_cost: r.avg_cost ? +r.avg_cost : null }));
  const lookback = Select({ label: 'Risk lookback', options: ['1y', '2y', '3y', '5y'], value: '2y' });
  const conf = Select({ label: 'VaR confidence', options: [[0.95, '95%'], [0.975, '97.5%'], [0.99, '99%']], value: 0.95 });
  const horizon = Select({ label: 'VaR horizon', options: [[1, '1 day'], [5, '5 days'], [10, '10 days'], [21, '21 days']], value: 1 });
  const shock = Slider({ label: 'Stress test: NIFTY move', min: -40, max: 20, value: -15, fmt: (v) => `${v}%` });
  const part = Slider({ label: 'Liquidity: max share of daily volume', min: 5, max: 50, step: 5, value: 20, fmt: (v) => `${v}%` });
  const res = new Results();
  const save = () => { const c = clean(); if (!c.length) { toast('Add at least one holding.'); return false; } store.saveHoldings(c); rows = c.map((x) => ({ ...x })); return true; };
  const run = async () => {
    const c = clean(); if (!c.length) { res.error(new Error('Add at least one holding.')); return; }
    res.loading('Pricing your portfolio and running risk models…', 'Downloading history · Ledoit-Wolf covariance · Monte Carlo');
    try {
      const d = await post('/portfolio/risk', { holdings: c, lookback: lookback.get(), conf: +conf.get(), horizon: +horizon.get(), rf: store.settings().rf, shock: shock.get() / 100, participation: part.get() / 100 });
      res.show(d.blocks, ctxNav);
    } catch (e) { res.error(e, run); }
  };
  const csvText = () => 'ticker,quantity,avg_cost\n' + clean().map((r) => `${r.ticker},${r.quantity},${r.avg_cost ?? ''}`).join('\n');
  const imp = FileText({ label: '', button: 'Import CSV', onText: (t) => {
    const lines = t.split(/\r?\n/).filter(Boolean); const head = lines[0].toLowerCase().split(',').map((x) => x.trim());
    const ti = head.indexOf('ticker') >= 0 ? head.indexOf('ticker') : head.indexOf('symbol'), qi = head.indexOf('quantity'), ci = head.indexOf('avg_cost');
    if (ti < 0 || qi < 0) { toast('CSV needs columns: ticker, quantity (avg_cost optional).'); return; }
    rows = lines.slice(1).map((l) => { const c = l.split(','); return { ticker: c[ti].trim().toUpperCase(), quantity: +c[qi], avg_cost: ci >= 0 && c[ci] ? +c[ci] : null }; }).filter((r) => r.ticker);
    paint(); toast(`Imported ${rows.length} holdings`);
  } });
  view.append(card(h('div', { class: 'section-title', style: { marginTop: 0 } }, 'Holdings'), list, note,
    h('div', { class: 'row', style: { marginTop: '8px' } }, button('Add row', () => { rows.push({ ticker: '', quantity: '', avg_cost: '' }); paint(); }, 'small', 'plus'),
      imp, button('Export CSV', () => download('jinni_holdings.csv', csvText(), 'text/csv'), 'small', 'download'),
      button('Reset to sample', () => { store.resetHoldings(); rows = store.holdings().map((x) => ({ ...x })); paint(); toast('Sample portfolio restored'); }, 'small'))),
    card(formGrid(lookback, conf, horizon, shock, part), h('div', { class: 'row', style: { marginTop: '12px' } }, button('Save & analyze', () => { if (save()) run(); }, 'primary', 'shield'))),
    res.el, disclaimer());
  run();
}

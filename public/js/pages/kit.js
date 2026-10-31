// Shared page helpers.
import { h, Results, card, formGrid, button } from '../ui.js';
import { store } from '../store.js';

export const go = (hash) => { location.hash = hash; };
export const ctxNav = { onSelect: (action, key) => { if (action === 'analyze') go(`#/stock/${encodeURIComponent(key)}`); } };
export const debounce = (fn, ms = 300) => { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; };
export const holdingsForApi = () => store.holdings().filter((x) => x.ticker && +x.quantity > 0)
  .map((x) => ({ ticker: String(x.ticker).trim(), quantity: +x.quantity, avg_cost: x.avg_cost ? +x.avg_cost : null }));

// A panel with optional controls that refetches when they change.
export function livePanel(panel, controls, fetcher, { ctx = ctxNav, auto = true, runLabel } = {}) {
  const res = new Results();
  let seq = 0;
  const run = async () => {
    const my = ++seq; res.loading();
    try { const d = await fetcher(); if (my === seq) res.show(d.blocks, ctx); return d; }
    catch (e) { if (my === seq) res.error(e, run); }
  };
  if (controls.length) {
    const form = formGrid(controls);
    panel.append(card(form, runLabel ? h('div', { class: 'row', style: { marginTop: '12px' } }, button(runLabel, run, 'primary')) : null));
    if (!runLabel) controls.forEach((c) => c.onChange(debounce(run, 250)));
  }
  panel.append(res.el);
  if (auto) run();
  return { run, res };
}
export const showIf = (ctl, test) => ctl.el.classList.toggle('hidden', !test());

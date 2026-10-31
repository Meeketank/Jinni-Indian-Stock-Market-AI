import { h, Results, icon, disclaimer, button } from '../ui.js';
import { get } from '../api.js';
import { ctxNav, go } from './kit.js';

const CARDS = [
  ['picks', 'star', 'Smart Picks', 'Validated stock selection: ranked on proven signals, tested walk-forward, built into a risk-aware basket for your budget.'],
  ['finder', 'target', 'Return Finder', 'Pick a goal (e.g. +7% in a week) and a budget. Get ranked stocks with honest odds.'],
  ['stock', 'chart', 'Analyzer', '87 indicators, auto-detected patterns, 360° view, forecast and trade plan.'],
  ['portfolio', 'shield', 'Portfolio Risk', 'VaR / CVaR, stress tests, beta, risk contribution, liquidity.'],
  ['screener', 'filter', 'Screener', 'Rank the NIFTY 50 or your own list by five factors.'],
  ['events', 'event', 'Sentiment & Events', 'Market mood, news tone and an event lab: "does the market always react to X?"'],
  ['optimizer', 'tune', 'Optimizer', 'Max-Sharpe, min-vol, risk parity, with a trade list.'],
  ['lab', 'flask', 'Strategy Lab', 'Backtests with Indian costs and walk-forward ML validation.'],
  ['journal', 'journal', 'Journal', 'Every tracked prediction scored against what really happened.'],
];

export async function render(view) {
  view.append(h('div', { class: 'hero' },
    h('div', { class: 'brandline' }, h('img', { src: '/favicon.svg', width: 40, height: 40, alt: '' }), 'JINNI'),
    h('h1', {}, 'Indian market risk & analytics terminal'),
    h('p', {}, 'Multi-factor ratings, odds of reaching your target, portfolio VaR and stress tests, auto-detected chart patterns and event studies for NSE & BSE, in one place.'),
    h('div', { class: 'cta' }, button('Find stocks for my target', () => go('#/finder'), 'primary', 'target'),
      button('Analyze a stock', () => go('#/stock'), '', 'chart'))));
  // Each section loads on its own, so one slow or failing feed never blanks the page.
  const section = (title, path, loadingMsg, ttl) => {
    const res = new Results(); const head = h('div', { class: 'section-title' }, title);
    const load = async (fresh) => {
      const had = res.el.children.length > 0 && !res.el.querySelector('.callout.error');
      if (!had) res.loading(loadingMsg);
      try { const d = await get(path, {}, fresh ? { fresh: true } : {}); res.show(d.blocks, ctxNav); }
      catch (e) { if (!had) res.error(e, () => load(true)); }
    };
    view.append(head, res.el); load();
    if (ttl) { const t = setInterval(() => { if (!res.el.isConnected) clearInterval(t); else if (!document.hidden) load(true); }, ttl); }
  };
  view.append(h('div', { class: 'section-title' }, 'Market pulse'));
  const pulse = new Results(); view.append(pulse.el);
  const loadPulse = async (fresh) => {
    const had = pulse.el.children.length > 0 && !pulse.el.querySelector('.callout.error');
    if (!had) pulse.loading('Loading indices…');
    try { const d = await get('/home/pulse', {}, fresh ? { fresh: true } : {}); pulse.show(d.blocks, ctxNav); } catch (e) { if (!had) pulse.error(e, () => loadPulse(true)); }
  };
  loadPulse();
  const timer = setInterval(() => { if (!pulse.el.isConnected) clearInterval(timer); else if (!document.hidden) loadPulse(true); }, 60000);
  section('Today’s movers · all NSE stocks', '/home/movers', 'Reading live quotes for the whole exchange…', 60000);
  section('Market outlook · ML estimate', '/outlook/market?h=1', 'Training the outlook model on 10 years of global cues…');
  section('Sector rotation', '/home/sectors', 'Reading sector indices…');
  section('NIFTY 50 stocks', '/home/nifty50', 'Reading all 50 constituents…');
  view.append(h('div', { class: 'section-title' }, 'Workspaces'),
    h('div', { class: 'grid-cards' }, CARDS.map(([id, ic, t, d]) => h('a', { class: 'nav-card', href: `#/${id}` },
      h('div', { class: 't' }, icon(ic), t), h('div', { class: 'd' }, d)))), disclaimer());
}

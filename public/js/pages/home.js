import { h, Results, icon, disclaimer, button } from '../ui.js';
import { get } from '../api.js';
import { ctxNav, go } from './kit.js';

const CARDS = [
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
  const res = new Results(); view.append(res.el);
  const load = async () => {
    res.loading('Loading market pulse…', 'Reading indices, sectors and the NIFTY 50');
    try { const d = await get('/home'); res.show(d.blocks, ctxNav); } catch (e) { res.error(e, load); }
  };
  load();
  view.append(h('div', { class: 'section-title' }, 'Workspaces'),
    h('div', { class: 'grid-cards' }, CARDS.map(([id, ic, t, d]) => h('a', { class: 'nav-card', href: `#/${id}` },
      h('div', { class: 't' }, icon(ic), t), h('div', { class: 'd' }, d)))), disclaimer());
}

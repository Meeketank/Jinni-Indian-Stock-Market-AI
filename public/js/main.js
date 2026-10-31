// App shell: hash router, navigation (bottom tabs on phones, top bar on desktop), splash.
import { h, icon, sheet, button } from './ui.js';
import { store } from './store.js';
import { toggle, isDark } from './theme.js';

const NAV = [
  { id: 'home', label: 'Home', icon: 'home', primary: true, desc: 'Market pulse' },
  { id: 'finder', label: 'Finder', icon: 'target', primary: true, desc: 'Which stocks can hit my target?' },
  { id: 'stock', label: 'Analyze', icon: 'chart', primary: true, desc: 'Charts, patterns, 87 indicators' },
  { id: 'portfolio', label: 'Portfolio', icon: 'shield', primary: true, desc: 'VaR, stress tests, risk' },
  { id: 'screener', label: 'Screener', icon: 'filter', desc: 'Rank stocks by 5 factors' },
  { id: 'events', label: 'Sentiment & Events', icon: 'event', desc: 'Market mood, event lab, calendar' },
  { id: 'optimizer', label: 'Optimizer', icon: 'tune', desc: 'Best weights, rebalance trades' },
  { id: 'lab', label: 'Lab', icon: 'flask', desc: 'Backtests and ML validation' },
  { id: 'journal', label: 'Journal', icon: 'journal', desc: 'Your predictions vs reality' },
  { id: 'settings', label: 'Settings', icon: 'gear', desc: 'Assumptions, rules, backup' },
];
const PAGES = {
  home: () => import('./pages/home.js'), finder: () => import('./pages/finder.js'), stock: () => import('./pages/analyzer.js'),
  portfolio: () => import('./pages/portfolio.js'), screener: () => import('./pages/screener.js'),
  events: () => import('./pages/events.js'), optimizer: () => import('./pages/optimizer.js'),
  lab: () => import('./pages/lab.js'), journal: () => import('./pages/journal.js'), settings: () => import('./pages/settings.js'),
};
const view = document.getElementById('view');
let cleanup = null; let navToken = 0;

function parse() {
  const raw = (location.hash || '#/home').replace(/^#\/?/, '');
  const [id, ...rest] = raw.split('/').filter((x) => x !== '');
  return { id: PAGES[id] ? id : 'home', params: rest.map(decodeURIComponent) };
}

function buildNav(active) {
  const top = document.getElementById('topnav'); const bar = document.getElementById('tabbar');
  top.replaceChildren(); bar.replaceChildren();
  const more = () => {
    const items = h('div', { class: 'items' }, NAV.filter((n) => !n.primary).map((n) => h('a', { class: 'item', href: `#/${n.id}`, onclick: () => s.close() },
      icon(n.icon), h('span', {}, n.label, h('small', {}, n.desc)))));
    const s = sheet('More', items);
  };
  for (const n of NAV.filter((x) => x.primary)) {
    const cls = n.id === active ? 'active' : '';
    top.append(h('a', { href: `#/${n.id}`, class: cls }, icon(n.icon), n.label));
    bar.append(h('a', { href: `#/${n.id}`, class: cls }, icon(n.icon), n.label));
  }
  const secondaryActive = NAV.find((n) => n.id === active && !n.primary);
  top.append(h('button', { class: `navlink ${secondaryActive ? 'active' : ''}`, type: 'button', onclick: more }, icon('more'), secondaryActive ? secondaryActive.label : 'More'));
  bar.append(h('button', { type: 'button', class: secondaryActive ? 'active' : '', onclick: more }, icon('more'), secondaryActive ? secondaryActive.label.split(' ')[0] : 'More'));
}

async function route() {
  const token = ++navToken; const { id, params } = parse();
  buildNav(id); try { cleanup?.(); } catch { /* ignore */ } cleanup = null;
  view.replaceChildren(h('div', { class: 'loading-box' }, h('div', { class: 'spinner' })));
  try {
    const mod = await PAGES[id]();
    if (token !== navToken) return;
    view.replaceChildren();
    cleanup = (await mod.render(view, params)) || null;
  } catch (e) {
    console.error(e);
    if (token === navToken) view.replaceChildren(h('div', { class: 'callout error' }, 'This page failed to load. ', h('a', { href: '#/home' }, 'Go home'), ' or reload.'));
  }
  window.scrollTo(0, 0); view.focus({ preventScroll: true });
}

function splash() {
  const el = document.getElementById('splash'); if (!el) return;
  let seen = false; try { seen = sessionStorage.getItem('jinni.splash') === '1'; sessionStorage.setItem('jinni.splash', '1'); } catch { /* ignore */ }
  if (seen) { el.remove(); return; }
  const done = () => el.remove(); el.addEventListener('animationend', (e) => { if (e.animationName === 'splash-out') done(); });
  setTimeout(done, 3200);
}

function themeBtn() {
  const b = document.getElementById('theme-btn'); if (!b) return;
  const paint = () => { b.replaceChildren(icon(isDark() ? 'sun' : 'moon')); b.title = isDark() ? 'Switch to light mode' : 'Switch to dark mode'; };
  b.addEventListener('click', toggle); window.addEventListener('jinni:theme', paint); paint();
}

themeBtn(); splash();
window.addEventListener('hashchange', route);
route();

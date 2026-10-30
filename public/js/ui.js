// UI toolkit: DOM helper, safe markdown, block renderer (tables, KPIs, charts, …) and form controls.
import * as F from './fmt.js';
import { mountPlot } from './plot.js';
import { get } from './api.js';
import { store } from './store.js';

// ------------------------------------------------------------------ DOM
export function h(tag, attrs, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === undefined || v === null || v === false) continue;
    if (k === 'class') el.className = v;
    else if (k === 'style' && typeof v === 'object') Object.assign(el.style, v);
    else if (k.startsWith('on') && typeof v === 'function') el.addEventListener(k.slice(2).toLowerCase(), v);
    else if (k === 'html') el.innerHTML = v;
    else el.setAttribute(k, v === true ? '' : v);
  }
  add(el, kids);
  return el;
}
function add(el, kids) {
  for (const k of kids.flat(Infinity)) {
    if (k === null || k === undefined || k === false) continue;
    el.append(k instanceof Node ? k : document.createTextNode(String(k)));
  }
}
export const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
export const safeUrl = (u) => (/^https?:\/\//i.test(u || '') ? u : '#');

// Minimal, XSS-safe markdown: **bold**, `code`, [text](https://…), "- " lists, "#### " headings.
export function md(src) {
  const inline = (s) => esc(s)
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>').replace(/`(.+?)`/g, '<code>$1</code>')
    .replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g, (_, t, u) => `<a href="${esc(u)}" target="_blank" rel="noopener noreferrer">${t}</a>`);
  const out = []; let list = false;
  for (const raw of String(src ?? '').split('\n')) {
    const line = raw.trimEnd();
    const hd = /^(#{2,4})\s+(.*)$/.exec(line);
    if (/^\s*-\s+/.test(line)) { if (!list) { out.push('<ul>'); list = true; } out.push(`<li>${inline(line.replace(/^\s*-\s+/, ''))}</li>`); continue; }
    if (list) { out.push('</ul>'); list = false; }
    if (hd) out.push(`<h4>${inline(hd[2])}</h4>`); else if (line) out.push(`<p>${inline(line)}</p>`);
  }
  if (list) out.push('</ul>');
  return out.join('');
}

// ---------------------------------------------------------------- icons
const ICONS = {
  moon: '<path d="M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5z"/>',
  sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M2 12h2M20 12h2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>',
  home: '<path d="M3 11l9-8 9 8v9a1 1 0 0 1-1 1h-5v-6H9v6H4a1 1 0 0 1-1-1z"/>',
  target: '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1.4"/>',
  chart: '<path d="M7 3v18M7 8H4.5v7H7M17 2v19M17 6h-2.5v9H17"/><path d="M7 8h2.5v7H7M17 6h2.5v9H17"/>',
  shield: '<path d="M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6z"/>',
  more: '<rect x="4" y="4" width="6.5" height="6.5" rx="1.5"/><rect x="13.5" y="4" width="6.5" height="6.5" rx="1.5"/><rect x="4" y="13.5" width="6.5" height="6.5" rx="1.5"/><rect x="13.5" y="13.5" width="6.5" height="6.5" rx="1.5"/>',
  filter: '<path d="M3 5h18l-7 8v6l-4 2v-8z"/>',
  event: '<rect x="3.5" y="5" width="17" height="15" rx="2"/><path d="M8 3v4M16 3v4M3.5 10h17"/>',
  tune: '<path d="M4 7h10M18 7h2M4 17h2M10 17h10"/><circle cx="16" cy="7" r="2"/><circle cx="8" cy="17" r="2"/>',
  flask: '<path d="M9 3h6M10 3v6l-5.5 9.5A2 2 0 0 0 6.2 21.5h11.6a2 2 0 0 0 1.7-3L14 9V3"/>',
  journal: '<path d="M9 11.5l2.2 2.2L16 9M4 5h3M4 12h1.5M4 19h6"/><rect x="9" y="3" width="11" height="18" rx="2"/>',
  gear: '<circle cx="12" cy="12" r="3"/><path d="M12 2v3M12 19v3M2 12h3M19 12h3M4.9 4.9l2.1 2.1M17 17l2.1 2.1M4.9 19.1L7 17M17 7l2.1-2.1"/>',
  search: '<circle cx="11" cy="11" r="7"/><path d="M20 20l-4-4"/>',
  download: '<path d="M12 4v11M7 11l5 5 5-5M5 20h14"/>',
  star: '<path d="M12 3l2.7 5.6 6.1.9-4.4 4.3 1 6.1L12 17l-5.4 2.9 1-6.1L3.2 9.5l6.1-.9z"/>',
  plus: '<path d="M12 5v14M5 12h14"/>',
  trash: '<path d="M4 7h16M9 7V4h6v3M6 7l1 13h10l1-13"/>',
};
export function icon(name) {
  const s = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  s.setAttribute('viewBox', '0 0 24 24'); s.setAttribute('class', 'ico'); s.setAttribute('aria-hidden', 'true');
  s.innerHTML = ICONS[name] || ''; return s;
}

// -------------------------------------------------------------- feedback
export function toast(msg, ms = 2600) {
  const root = document.getElementById('toast-root');
  const t = h('div', { class: 'toast' }, msg); root.append(t);
  setTimeout(() => t.remove(), ms);
}
export function sheet(title, content, { onClose } = {}) {
  const root = document.getElementById('sheet-root');
  const close = () => { back.remove(); document.body.style.overflow = ''; onClose?.(); };
  const back = h('div', { class: 'sheet-back', onclick: (e) => { if (e.target === back) close(); } },
    h('div', { class: 'sheet', role: 'dialog', 'aria-label': title }, h('div', { class: 'row' }, h('h3', { class: 'grow' }, title),
      h('button', { class: 'btn small', onclick: close }, 'Done')), content));
  root.append(back); document.body.style.overflow = 'hidden';
  return { close, el: back };
}
export function download(name, text, mime = 'text/plain') {
  const a = h('a', { href: URL.createObjectURL(new Blob([text], { type: mime })), download: name });
  document.body.append(a); a.click(); a.remove();
}
export const loadingBox = (msg = 'Working…', steps = '') =>
  h('div', { class: 'loading-box' }, h('div', { class: 'spinner' }), h('div', {}, msg), h('div', { class: 'steps' }, steps),
    h('div', { class: 'progress hidden' }, h('i')));

// Results container with loading / error / content states.
export class Results {
  constructor() { this.el = h('div', { class: 'results', 'aria-live': 'polite' }); this._tick = null; }
  clear() { this.el.replaceChildren(); clearInterval(this._tick); }
  loading(msg, steps) {
    clearInterval(this._tick);
    const box = loadingBox(msg, steps); this.el.replaceChildren(box);
    const t0 = Date.now(); const stepEl = box.querySelector('.steps');
    this._steps = steps; this._tick = setInterval(() => {
      const s = Math.round((Date.now() - t0) / 1000);
      if (s >= 4) stepEl.textContent = `${this._steps || ''} ${s}s`.trim();
    }, 1000);
    return box;
  }
  progress(f, msg) {
    const box = this.el.querySelector('.loading-box'); if (!box) return;
    const bar = box.querySelector('.progress'); bar.classList.remove('hidden'); bar.firstChild.style.width = `${Math.round(f * 100)}%`;
    if (msg) box.querySelector('.steps').textContent = this._steps = msg;
  }
  error(e, retry) {
    clearInterval(this._tick);
    const actions = h('div', { class: 'row', style: { marginTop: '10px' } });
    if (retry) actions.append(h('button', { class: 'btn small', onclick: retry }, 'Try again'));
    if (e?.code === 'data_unavailable' && !store.settings().demo) {
      actions.append(h('button', { class: 'btn small', onclick: () => { store.saveSettings({ demo: true }); location.reload(); } },
        'Use demo data instead'));
    }
    this.el.replaceChildren(h('div', { class: 'callout error' }, h('strong', {}, e?.message || 'Something went wrong.'), actions));
  }
  show(blocks, ctx) { clearInterval(this._tick); this.el.replaceChildren(renderBlocks(blocks, ctx)); }
}

// ----------------------------------------------------------------- tabs
export function tabsUI(items, { onShow, initial = 0 } = {}) {
  const bar = h('div', { class: 'tabs', role: 'tablist' });
  const panels = items.map((it) => h('div', { class: 'tabpanel hidden', role: 'tabpanel' }));
  const shown = new Set();
  const select = (i) => {
    [...bar.children].forEach((b, j) => { b.classList.toggle('on', i === j); b.setAttribute('aria-selected', i === j); });
    panels.forEach((p, j) => p.classList.toggle('hidden', i !== j));
    if (!shown.has(i)) { shown.add(i); onShow?.(i, panels[i], items[i]); }
    bar.children[i].scrollIntoView?.({ block: 'nearest', inline: 'center' });
  };
  items.forEach((it, i) => bar.append(h('button', { role: 'tab', onclick: () => select(i) }, it.label)));
  const root = h('div', { class: 'tabs-root' }, bar, ...panels);
  queueMicrotask(() => select(initial));
  return { el: root, select, panels };
}

// ---------------------------------------------------------- block renderer
export function renderBlocks(blocks, ctx = {}) {
  const frag = document.createDocumentFragment();
  for (const b of blocks || []) { const n = block(b, ctx); if (n) frag.append(n); }
  return frag;
}
const wrapBlocks = (blocks, ctx) => { const d = h('div'); d.append(renderBlocks(blocks, ctx)); return d; };

function block(b, ctx) {
  switch (b.type) {
    case 'kpis': return h('div', { class: 'kpi-grid' }, b.items.map((k) => {
      const t = k.tone ? ` ${k.tone}` : '';
      return h('div', { class: 'kpi' }, h('div', { class: 'lbl', title: k.label }, k.label),
        h('div', { class: `val${k.sub ? '' : t}` }, k.value), k.sub ? h('div', { class: `sub${t}` }, k.sub) : null);
    }));
    case 'section': return h('div', { class: 'section-title' }, b.title);
    case 'text': return h('div', { class: `prose${b.muted ? ' muted' : ''}`, html: md(b.text) });
    case 'callout': return h('div', { class: `callout ${b.kind}`, html: md(b.text) });
    case 'plot': { const d = h('div'); mountPlot(d, b.fig, b.height || 340); return d; }
    case 'cols': {
      const r = b.ratio || b.cols.map(() => 1);
      return h('div', { class: 'cols', style: { gridTemplateColumns: r.map((x) => `${x}fr`).join(' ') } },
        b.cols.map((c) => wrapBlocks(c, ctx)));
    }
    case 'tabs': {
      const t = tabsUI(b.tabs.map((x) => ({ label: x.label, blocks: x.blocks })), {
        onShow: (i, panel, it) => panel.append(renderBlocks(it.blocks, ctx)),
      });
      return t.el;
    }
    case 'details': return h('details', { class: 'acc' }, h('summary', {}, b.title), h('div', { class: 'in' }, wrapBlocks(b.blocks, ctx)));
    case 'table': return tableBlock(b, ctx);
    case 'tape': return tape(b.items);
    case 'news': return newsList(b.items);
    case 'badge': return h('span', { class: `badge ${b.cls}` }, b.label);
    case 'demo': return h('div', { class: 'callout info' }, h('span', { class: 'badge b-demo' }, 'DEMO DATA'), ' ',
      'Synthetic prices, for illustration only. Turn demo mode off in Settings to use live data.');
    case 'stockhead': return h('div', { class: 'stockhead' }, h('h3', {}, b.name, h('span', { class: `badge ${b.cls}` }, b.rating)),
      h('div', { class: 'sub' }, [b.ticker, b.sector, b.asof ? `as of ${b.asof}` : ''].filter(Boolean).join(' · ')),
      b.sub ? h('div', { class: 'sub' }, b.sub) : null);
    case 'score': return scoreRing(b);
    case 'gate': return gate(b);
    default: return null;
  }
}

function scoreRing(b) {
  const v = Math.max(0, Math.min(100, b.value)), r = 36, c = 2 * Math.PI * r;
  const col = v >= 60 ? 'var(--up)' : v < 42 ? 'var(--down)' : 'var(--gold)';
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('viewBox', '0 0 84 84'); svg.setAttribute('class', 'ring');
  svg.innerHTML = `<circle cx="42" cy="42" r="${r}" fill="none" style="stroke:var(--border)" stroke-width="8"/>
    <circle cx="42" cy="42" r="${r}" fill="none" style="stroke:${col}" stroke-width="8" stroke-linecap="round"
      stroke-dasharray="${(c * v) / 100} ${c}" transform="rotate(-90 42 42)"/>
    <text x="42" y="50" text-anchor="middle">${Math.round(v)}</text>`;
  return h('div', { class: 'score' }, svg, h('div', {}, h('div', { class: 'muted', style: { fontSize: '.8rem' } }, b.label),
    h('span', { class: `badge ${b.cls}` }, b.rating)));
}
function gate(b) {
  const ic = { PASS: '✅', WARN: '⚠️', BLOCK: '⛔' };
  return h('div', { class: 'gate' }, h('h4', {}, `Risk check: ${ic[b.status]} ${b.status}`),
    h('ul', {}, b.checks.map(([s, rule, d]) => h('li', {}, `${ic[s]} `, h('strong', {}, rule), `: ${d}`))));
}
function tape(items) {
  const one = items.map(([n, p, c]) => h('span', { class: 'item' }, h('b', {}, n), F.num(p, 2), ' ',
    h('span', { class: c >= 0 ? 'up' : 'down' }, F.pct(c, 2, true))));
  return h('div', { class: 'tape' }, h('div', { class: 'tape-inner' }, one, one.map((n) => n.cloneNode(true))));
}
function newsList(items) {
  return h('ul', { class: 'news' }, items.map((n) => {
    const col = n.score > 0.1 ? 'var(--up)' : n.score < -0.1 ? 'var(--down)' : 'var(--muted)';
    return h('li', {}, h('span', { class: 'dot', style: { background: col }, title: `tone ${n.score?.toFixed?.(2)}` }),
      h('div', {}, h('div', { class: 'meta' }, [n.time ? F.date(n.time) + (String(n.time).length > 10 ? ' ' + String(n.time).slice(11, 16) : '') : '', n.ticker].filter(Boolean).join(' · ')),
        h('a', { href: safeUrl(n.link), target: '_blank', rel: 'noopener noreferrer' }, n.headline),
        n.tags ? h('div', {}, h('span', { class: 'badge b-hold' }, n.tags)) : null));
  }));
}

// --------------------------------------------------------------- tables
const LEFT = new Set(['text', 'badge', 'grade', 'date', 'link']);
const phone = window.matchMedia('(max-width: 640px)');
function tableBlock(b, ctx) {
  const cols = b.columns; let rows = (b.rows || []).slice();
  if (!rows.length) return h('div', { class: 'caption' }, b.empty || 'Nothing to show.');
  let sort = b.sort ? { key: b.sort[0], desc: !!b.sort[1] } : null; let selected = null;
  const longText = (c) => c.fmt === 'text' && rows.slice(0, 20).some((r) => String(r[c.key] ?? '').length > 45);
  const wrapCols = new Set(cols.filter(longText).map((c) => c.key));
  const tbody = h('tbody'); const headRow = h('tr'); const cards = h('div', { class: 'tcards' });
  const sorted = () => (sort ? rows.slice().sort((a, bb) => {
    const x = a[sort.key], y = bb[sort.key];
    if (x === null || x === undefined) return 1; if (y === null || y === undefined) return -1;
    const d = typeof x === 'number' && typeof y === 'number' ? x - y : String(x).localeCompare(String(y));
    return sort.desc ? -d : d;
  }) : rows);
  const pick = (r, el, root) => {
    if (!b.select) return;
    selected = r[b.select.key]; root.querySelectorAll('.sel').forEach((x) => x.classList.remove('sel')); el.classList.add('sel');
    ctx.onSelect?.(b.select.action, r[b.select.key], r);
  };
  const drawTable = () => {
    tbody.replaceChildren(...sorted().map((r) => {
      const tr = h('tr', { class: b.select ? 'click' : '' }, cols.map((c, i) => cellFor(c, r, i, wrapCols.has(c.key))));
      if (b.select) tr.addEventListener('click', () => pick(r, tr, tbody));
      if (selected !== null && r[b.select?.key] === selected) tr.classList.add('sel');
      return tr;
    }));
    [...headRow.children].forEach((th, i) => {
      const arr = th.querySelector('.arr'); if (arr) arr.remove();
      if (sort && sort.key === cols[i].key) th.append(h('span', { class: 'arr' }, sort.desc ? '▼' : '▲'));
    });
  };
  // Phones: one card per row (title = first column, the rest as label/value pairs) instead of a sideways-scrolling table.
  const drawCards = () => {
    cards.replaceChildren(...sorted().map((r) => {
      const [first, ...rest] = cols;
      const card = h('div', { class: `tcard${b.select ? ' click' : ''}` });
      const head = cellFor(first, r, 0, false); head.className = 'tc-title';
      card.append(h('div', { class: 'tc-head' }, h('div', { class: 'tc-title' }, ...head.childNodes)));
      const grid = h('div', { class: 'tc-grid' });
      for (const c of rest) {
        const v = r[c.key]; if (v === null || v === undefined || v === '') continue;
        const td = cellFor(c, r, 1, false);
        const long = c.fmt === 'text' && String(v).length > 28;
        grid.append(h('div', { class: `tc-item${long ? ' long' : ''}` }, h('span', { class: 'tc-l' }, c.label), h('span', { class: 'tc-v' }, ...td.childNodes)));
      }
      card.append(grid);
      if (b.select) card.addEventListener('click', () => pick(r, card, cards));
      if (selected !== null && r[b.select?.key] === selected) card.classList.add('sel');
      return card;
    }));
  };
  const draw = () => (phone.matches ? drawCards() : drawTable());
  cols.forEach((c, i) => headRow.append(h('th', { class: LEFT.has(c.fmt) || i === 0 ? 'l' : '', title: c.help || '',
    onclick: () => { sort = sort && sort.key === c.key ? { key: c.key, desc: !sort.desc } : { key: c.key, desc: !LEFT.has(c.fmt) }; draw(); } }, c.label)));
  const table = h('div', { class: 'tbl-wrap', style: b.height ? { maxHeight: `${b.height}px` } : {} },
    h('table', { class: 'tbl' }, h('thead', {}, headRow), tbody));
  const sortSel = h('select', { class: 'tc-sort', 'aria-label': 'Sort by', onchange: (e) => {
    const [k, d] = e.target.value.split('|'); sort = k ? { key: k, desc: d === 'd' } : null; drawCards(); } },
  h('option', { value: '' }, 'Sort: default'),
  cols.flatMap((c) => [h('option', { value: `${c.key}|d` }, `${c.label} ↓`), h('option', { value: `${c.key}|a` }, `${c.label} ↑`)]));
  const wrap = h('div', { class: 'tblock' }, h('div', { class: 'tc-bar' }, sortSel), cards, table);
  const apply = () => { wrap.classList.toggle('as-cards', phone.matches); draw(); };
  phone.addEventListener?.('change', apply); apply();
  return wrap;
}
function cellFor(c, r, i, wrap) {
  const v = r[c.key]; const k = c.fmt;
  const td = h('td', { class: `${LEFT.has(k) || i === 0 ? 'l' : ''}${wrap ? ' wrap' : ''}` });
  if (v === null || v === undefined || v === '') { td.textContent = F.dash; return td; }
  switch (k) {
    case 'badge': td.append(h('span', { class: `badge ${F.badgeClass(v)}` }, v)); break;
    case 'grade': td.append(h('span', { class: `badge ${F.gradeClass(v)}` }, v)); break;
    case 'bar01': case 'bar100': {
      const f = k === 'bar01' ? v : v / 100;
      td.append(h('span', { class: 'pbar' }, h('span', { class: 'track' }, h('i', { style: { width: `${Math.max(0, Math.min(1, f)) * 100}%` } })),
        h('span', {}, F.plain(k, v)))); break;
    }
    case 'tone': {
      const w = Math.min(Math.abs(v), 1) * 50;
      td.append(h('span', { class: 'tonebar' }, h('span', { class: 'track' }, h('i', { style: { left: v >= 0 ? '50%' : `${50 - w}%`, width: `${w}%`, background: v >= 0 ? 'var(--up)' : 'var(--down)' } })),
        h('span', { class: F.toneClass(v) }, F.plain(k, v)))); break;
    }
    case 'spct1': case 'spct2': case 'n1signed': td.append(h('span', { class: F.toneClass(v) }, F.plain(k, v))); break;
    case 'bool': td.append(h('span', { class: v ? 'up' : 'muted' }, F.plain(k, v))); break;
    default: td.textContent = F.plain(k, v);
  }
  return td;
}

// ------------------------------------------------------------- controls
// Each control returns { el, get(), set(v), onChange(fn) }.
function wire(el, getter, setter) {
  const subs = [];
  const api = { el, get: getter, set: (v) => { setter(v); }, onChange: (fn) => { subs.push(fn); return api; }, _emit: () => subs.forEach((f) => f(getter())) };
  return api;
}
const opts = (list) => list.map((o) => (Array.isArray(o) ? { v: o[0], l: o[1] } : typeof o === 'object' ? o : { v: o, l: String(o) }));
const field = (label, inner, help, cls = '') => h('div', { class: `field ${cls}` }, label ? h('label', {}, label) : null, inner, help ? h('div', { class: 'help' }, help) : null);

export function Select({ label, options, value, help, cls }) {
  const o = opts(options);
  const sel = h('select', {}, o.map((x) => h('option', { value: x.v }, x.l)));
  sel.value = String(value ?? o[0]?.v);
  const c = wire(field(label, sel, help, cls), () => { const x = o.find((y) => String(y.v) === sel.value); return x ? x.v : sel.value; }, (v) => { sel.value = String(v); });
  sel.addEventListener('change', c._emit); return c;
}
export function Segmented({ label, options, value, help, cls }) {
  const o = opts(options); let cur = value ?? o[0]?.v;
  const bar = h('div', { class: 'seg', role: 'group' });
  const paint = () => [...bar.children].forEach((b, i) => b.classList.toggle('on', o[i].v === cur));
  o.forEach((x) => bar.append(h('button', { type: 'button', onclick: () => { cur = x.v; paint(); c._emit(); } }, x.l)));
  const c = wire(field(label, bar, help, cls), () => cur, (v) => { cur = v; paint(); }); paint(); return c;
}
export function NumberInput({ label, value, min, max, step = 1, help, cls, placeholder }) {
  const i = h('input', { type: 'number', inputmode: 'decimal', min, max, step, value, placeholder });
  const c = wire(field(label, i, help, cls), () => (i.value === '' ? null : Number(i.value)), (v) => { i.value = v; });
  i.addEventListener('change', c._emit); return c;
}
// Time span = a count plus a unit ("3" + "months"); quick chips fill both. get() -> { n, unit }.
export const SPAN_UNITS = [['day', 'Days'], ['week', 'Weeks'], ['month', 'Months'], ['quarter', 'Quarters'], ['year', 'Years'], ['fy', 'Financial years']];
const TRADING = { day: 1, week: 5, month: 21, quarter: 63, year: 252, fy: 252 };
/** Approximate trading days in a span, for endpoints that take a horizon in trading days. */
export const tradingDays = ({ n, unit }) => Math.max(1, Math.round(n * (TRADING[unit] || 1)));
export function Span({ label = 'Time range', n = 6, unit = 'month', units = SPAN_UNITS, presets = [[1, 'month'], [3, 'month'], [6, 'month'], [1, 'year'], [3, 'year'], [5, 'year']], max = 600, help, cls }) {
  const num = h('input', { type: 'number', inputmode: 'numeric', min: 1, max, step: 1, value: n, 'aria-label': `${label}: count` });
  const sel = h('select', { 'aria-label': `${label}: unit` }, units.map(([v, l]) => h('option', { value: v }, l))); sel.value = unit;
  const short = { day: 'D', week: 'W', month: 'M', quarter: 'Q', year: 'Y', fy: 'FY' };
  const chips = h('div', { class: 'seg' }, presets.map(([pn, pu]) => h('button', { type: 'button', 'data-n': pn, 'data-u': pu,
    onclick: () => { num.value = pn; sel.value = pu; paint(); c._emit(); } }, `${pn}${short[pu]}`)));
  const read = () => { let v = Math.round(Number(num.value)); if (!Number.isFinite(v) || v < 1) v = 1; if (v > max) v = max; return { n: v, unit: sel.value }; };
  const paint = () => { const cur = read(); [...chips.children].forEach((b) => b.classList.toggle('on', +b.dataset.n === cur.n && b.dataset.u === cur.unit)); };
  const fire = () => { const cur = read(); num.value = cur.n; paint(); c._emit(); };
  num.addEventListener('change', fire); sel.addEventListener('change', fire);
  num.addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); num.blur(); } });
  const c = wire(field(label, h('div', {}, h('div', { class: 'span-row' }, num, sel), chips), help, cls), read, (v) => { num.value = v.n; sel.value = v.unit; paint(); });
  paint(); return c;
}
export function TextInput({ label, value = '', placeholder, help, cls }) {
  const i = h('input', { type: 'text', value, placeholder, autocomplete: 'off', autocapitalize: 'off', spellcheck: 'false' });
  const c = wire(field(label, i, help, cls), () => i.value.trim(), (v) => { i.value = v; });
  i.addEventListener('change', c._emit); return c;
}
export function TextArea({ label, value = '', rows = 4, help, cls, placeholder }) {
  const i = h('textarea', { rows, placeholder }); i.value = value;
  const c = wire(field(label, i, help, cls), () => i.value, (v) => { i.value = v; });
  i.addEventListener('change', c._emit); return c;
}
export function Toggle({ label, value = false, help, cls }) {
  const i = h('input', { type: 'checkbox' }); i.checked = !!value;
  const lab = h('label', { class: 'toggle' }, i, h('span', { class: 'sw' }), h('span', {}, label));
  const c = wire(h('div', { class: `field ${cls || ''}` }, lab, help ? h('div', { class: 'help' }, help) : null), () => i.checked, (v) => { i.checked = !!v; });
  i.addEventListener('change', c._emit); return c;
}
export function Slider({ label, min, max, step = 1, value, fmt = (v) => v, help, cls }) {
  const i = h('input', { type: 'range', min, max, step, value }); const val = h('span', { class: 'slider-val' }, fmt(value));
  const c = wire(field(null, h('div', {}, h('label', { class: 'lab', style: { display: 'block', marginBottom: '2px', color: 'var(--muted)', fontSize: '.82rem', fontWeight: 600 } }, label, val), i), help, cls),
    () => Number(i.value), (v) => { i.value = v; val.textContent = fmt(Number(v)); });
  i.addEventListener('input', () => { val.textContent = fmt(Number(i.value)); }); i.addEventListener('change', c._emit); return c;
}
// Toggle-chips for a short list (multi-select).
export function Chips({ label, options, value = [], help, cls }) {
  const o = opts(options); const on = new Set(value);
  const box = h('div', { class: 'chips' });
  const paint = () => [...box.children].forEach((b, i) => b.classList.toggle('off', !on.has(o[i].v)));
  o.forEach((x) => box.append(h('button', { type: 'button', class: 'chip', onclick: () => { on.has(x.v) ? on.delete(x.v) : on.add(x.v); paint(); c._emit(); } }, x.l)));
  const c = wire(field(label, box, help, cls), () => o.map((x) => x.v).filter((v) => on.has(v)), (v) => { on.clear(); v.forEach((x) => on.add(x)); paint(); });
  paint(); return c;
}
// Searchable multi-select in a sheet, for long grouped lists (indicators).
export function MultiPick({ label, groups, value = [], max = 99, help, cls, noun = 'items' }) {
  let sel = [...value]; const summary = h('div', { class: 'chips' });
  const paint = () => {
    summary.replaceChildren(...sel.map((v) => h('span', { class: 'chip' }, v, h('button', { type: 'button', 'aria-label': `Remove ${v}`, onclick: (e) => { e.stopPropagation(); sel = sel.filter((x) => x !== v); paint(); c._emit(); } }, '×'))),
      h('button', { type: 'button', class: 'btn small', onclick: openSheet }, sel.length ? 'Edit' : `Choose ${noun}`));
  };
  const openSheet = () => {
    const q = h('input', { type: 'search', placeholder: `Search ${noun}…` }); const list = h('div');
    const draw = () => {
      const needle = q.value.toLowerCase(); list.replaceChildren();
      for (const g of groups) {
        const items = g.items.filter((it) => !needle || it.name.toLowerCase().includes(needle) || (it.desc || '').toLowerCase().includes(needle));
        if (!items.length) continue;
        list.append(h('div', { class: 'pick-group' }, g.name));
        for (const it of items) {
          const cb = h('input', { type: 'checkbox' }); cb.checked = sel.includes(it.name);
          cb.addEventListener('change', () => {
            if (cb.checked) { if (sel.length >= max) { cb.checked = false; toast(`You can pick up to ${max} ${noun}.`); return; } sel.push(it.name); }
            else sel = sel.filter((x) => x !== it.name);
          });
          list.append(h('label', { class: 'pick-row' }, cb, h('span', {}, it.name, it.desc ? h('small', {}, it.desc) : null)));
        }
      }
    };
    q.addEventListener('input', draw); draw();
    sheet(label, h('div', {}, h('div', { class: 'help' }, `Pick up to ${max}.`), q, list), { onClose: () => { paint(); c._emit(); } });
  };
  const c = wire(field(label, summary, help, cls), () => [...sel], (v) => { sel = [...v]; paint(); }); paint(); return c;
}
// Symbol box with NIFTY 50 suggestions and server-side search of all listed stocks.
/** Turn whatever was typed ("vedanta", "TCS", "hdfc bank") into a listed ticker, or null if nothing matches. */
export async function resolveSymbol(text) {
  const q = String(text || '').trim(); if (!q) return null;
  if (q.startsWith('^') || q.includes('=')) return q.toUpperCase();
  const r = await get('/search', { q });
  const res = r.results || []; const bare = q.toUpperCase().replace(/\s+/g, '').replace(/\.(NS|BO)$/, '');
  const exact = res.find((x) => x.symbol.toUpperCase() === bare && (!/\.(NS|BO)$/i.test(q) || x.t.toUpperCase() === q.toUpperCase().replace(/\s+/g, '')));
  return (exact || res[0])?.t || null;
}
export function SymbolInput({ label, value = '', help, cls, suggestions = [] }) {
  const i = h('input', { type: 'text', value, placeholder: 'Search any listed stock, e.g. Vedanta or IRCTC', autocomplete: 'off', autocapitalize: 'none', autocorrect: 'off', spellcheck: 'false', enterkeyhint: 'search' });
  const list = h('div', { class: 'list hidden' }); let timer; let seq = 0; let items = []; let cur = -1;
  const hide = () => { list.classList.add('hidden'); cur = -1; };
  const paint = () => [...list.children].forEach((e, k) => e.classList?.toggle('act', k === cur));
  const finish = (t) => { i.value = t; hide(); c._emit(); };
  const show = (rows, note) => {
    items = rows.slice(0, 12);
    if (!items.length) { list.replaceChildren(h('div', { class: 'opt none' }, note || 'No listed stock matches')); list.classList.remove('hidden'); return; }
    list.replaceChildren(...items.map((x) => h('div', { class: 'opt', onmousedown: (e) => { e.preventDefault(); finish(x.t); }, ontouchend: (e) => { e.preventDefault(); finish(x.t); } },
      h('span', {}, x.t), h('small', {}, [x.name, x.src].filter(Boolean).join(' · ')))));
    list.classList.remove('hidden');
  };
  const query = (q) => {
    clearTimeout(timer); const my = ++seq;
    if (!q) { show(suggestions); return; }
    show([], 'Searching all NSE/BSE listings…');
    timer = setTimeout(async () => {
      try { const r = await get('/search', { q }); if (my === seq) show(r.results || []); }
      catch { if (my === seq) show([], 'Search is unavailable. Type the full symbol, e.g. TCS.NS'); }
    }, 160);
  };
  i.addEventListener('input', () => query(i.value.trim()));
  i.addEventListener('focus', () => { if (!i.value) show(suggestions); i.select?.(); });
  i.addEventListener('blur', () => setTimeout(hide, 150));
  i.addEventListener('keydown', async (e) => {
    if (e.key === 'ArrowDown' || e.key === 'ArrowUp') { e.preventDefault(); if (items.length) { cur = (cur + (e.key === 'ArrowDown' ? 1 : -1) + items.length) % items.length; paint(); } return; }
    if (e.key !== 'Enter') return;
    e.preventDefault();
    if (cur >= 0 && items[cur]) { finish(items[cur].t); return; }
    const txt = i.value.trim(); if (!txt) return; hide();
    try { const t = await resolveSymbol(txt); if (t) finish(t); else show([], `No listed stock matches “${txt}”`); } catch { finish(txt.toUpperCase()); }
  });
  const c = wire(field(label, h('div', { class: 'combo' }, i, list), help, cls), () => i.value.trim(), (v) => { i.value = v; });
  return c;
}
export function FileText({ label, accept = '.csv,text/csv', button = 'Choose file', onText, help, cls }) {
  const i = h('input', { type: 'file', accept, class: 'hidden' });
  i.addEventListener('change', async () => { const f = i.files[0]; if (f) { onText(await f.text(), f.name); i.value = ''; } });
  return field(label, h('div', {}, h('button', { type: 'button', class: 'btn small', onclick: () => i.click() }, button), i), help, cls);
}
export const formGrid = (...c) => h('div', { class: 'form' }, c.flat().filter(Boolean).map((x) => x.el || x));
export const card = (...c) => h('div', { class: 'card' }, c);
export const pageHead = (title, sub) => h('div', {}, h('h1', { class: 'page-title' }, title), sub ? h('p', { class: 'page-sub' }, sub) : null);
export const disclaimer = () => h('div', { class: 'disclaimer' }, 'JINNI is an analytics and education tool, not investment advice and not a SEBI-registered adviser. ' +
  'Market data from Yahoo Finance may be delayed. Model outputs are probabilistic and past performance does not guarantee future results.');
export const button = (label, onclick, cls = '', ic) => h('button', { type: 'button', class: `btn ${cls}`, onclick }, ic ? icon(ic) : null, label);

let _cat; export const catalog = () => (_cat ||= get('/catalog'));
export const niftySuggest = async () => (await catalog()).nifty50.map((x) => ({ t: x.t, name: `${x.name} · ${x.sector}` }));

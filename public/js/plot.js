// Plotly wrapper: applies the active theme to server-built figures and renders lazily when visible.
// Figures reference theme colours as var(--token); bright dark-theme accents are swapped for deeper ones in light mode.
import { isDark } from './theme.js';

const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const palette = () => ({ text: css('--text'), muted: css('--muted'), grid: css('--grid'), line: css('--border'), panel: css('--panel') });
const COLORWAY_DARK = ['#4C8DFF', '#F5B841', '#22C55E', '#E86AF0', '#2EC4B6', '#FF8A4C', '#A78BFA', '#94A3B8'];
const COLORWAY_LIGHT = ['#2563EB', '#C98A0C', '#16A34A', '#C026D3', '#0F9488', '#EA580C', '#7C3AED', '#64748B'];
const LIGHT_MAP = {
  '#F5B841': '#C98A0C', '#F7C25A': '#C98A0C', '#E39B1B': '#B4740A', '#22C55E': '#16A34A', '#EF4444': '#DC2626', '#4C8DFF': '#2563EB',
  '#8A94A6': '#64748B', '#E86AF0': '#C026D3', '#2EC4B6': '#0F9488', '#FF8A4C': '#EA580C', '#A78BFA': '#7C3AED', '#94A3B8': '#64748B',
  '#E6EAF2': '#192033', '#26A69A': '#16A34A', '#EF5350': '#DC2626', '#2DD4BF': '#0F9488', '#F87171': '#DC2626', '#4ADE80': '#16A34A',
  '#60A5FA': '#2563EB', '#FACC15': '#CA8A04', '#FBBF24': '#D97706', '#C084FC': '#9333EA', '#38BDF8': '#0284C7',
};

// Replace var(--x) with the live colour and, in light mode, swap pale accents for readable ones.
function resolveColours(fig) {
  let txt = JSON.stringify(fig);
  txt = txt.replace(/var\(--([a-z0-9-]+)\)/gi, (_, n) => css('--' + n) || '#888888');
  if (!isDark()) txt = txt.replace(/#[0-9a-f]{6}\b/gi, (m) => LIGHT_MAP[m.toUpperCase()] || m);
  return JSON.parse(txt);
}

function axisDefaults(C) { return { gridcolor: C.grid, zeroline: false, linecolor: C.line, tickfont: { color: C.muted }, automargin: true }; }
const isObj = (o) => o && typeof o === 'object' && !Array.isArray(o);
function merge(base, over) {
  const out = { ...base };
  for (const [k, v] of Object.entries(over || {})) out[k] = isObj(v) && isObj(base[k]) ? merge(base[k], v) : v;
  return out;
}

export function themed(figIn, height) {
  const C = palette(); const COLORWAY = isDark() ? COLORWAY_DARK : COLORWAY_LIGHT;
  const fig = resolveColours(figIn);
  const lay = { ...(fig.layout || {}) };
  const small = window.innerWidth < 640;
  const base = {
    paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
    font: { family: 'Inter, system-ui, sans-serif', color: C.text, size: small ? 11 : 12 }, colorway: COLORWAY,
    legend: { orientation: 'h', yanchor: 'top', y: -0.1, x: 0, bgcolor: 'rgba(0,0,0,0)', font: { size: 11 } },
    margin: { l: small ? 6 : 10, r: small ? 6 : 12, t: 40, b: 12 },
    hoverlabel: { bgcolor: C.panel, bordercolor: C.line, font: { color: C.text } },
    xaxis: axisDefaults(C), yaxis: axisDefaults(C), hovermode: 'closest', dragmode: false,
  };
  for (const k of Object.keys(lay)) if (/^[xy]axis\d+$/.test(k)) base[k] = axisDefaults(C);
  const out = merge(base, lay);
  if (typeof out.title === 'string') out.title = { text: out.title };
  out.title = merge({ font: { size: 14, color: C.text }, x: 0, xanchor: 'left' }, out.title || {});
  if (!out.title.text) out.title.text = '';
  out.height = height;
  return { data: fig.data || [], layout: out };
}

const CONFIG = { displayModeBar: false, responsive: true, scrollZoom: false, doubleClick: 'reset', staticPlot: false };
let io = null;
const pending = new WeakMap();
const jobs = new Map();

function draw(el) {
  const job = pending.get(el);
  if (!job) return;
  pending.delete(el);
  if (!window.Plotly) { el.textContent = 'Charts are unavailable (could not load the chart library).'; el.classList.remove('loading'); return; }
  try {
    const { data, layout } = themed(job.fig, job.height);
    el.style.height = job.height + 'px';
    window.Plotly.react(el, data, layout, CONFIG).then(() => el.classList.remove('loading'));
  } catch (e) { el.textContent = 'This chart could not be drawn.'; el.classList.remove('loading'); console.error(e); }
}

export function mountPlot(el, fig, height = 360) {
  el.classList.add('plot', 'loading');
  el.style.minHeight = Math.min(height, 200) + 'px';
  const job = { fig, height }; pending.set(el, job); jobs.set(el, job);
  if (!('IntersectionObserver' in window)) { requestAnimationFrame(() => draw(el)); return; }
  io = io || new IntersectionObserver((entries) => {
    for (const e of entries) if (e.isIntersecting) { io.unobserve(e.target); draw(e.target); }
  }, { rootMargin: '300px 0px' });
  io.observe(el);
}

window.addEventListener('jinni:theme', () => {
  for (const [el, job] of jobs) {
    if (!el.isConnected) { jobs.delete(el); continue; }
    if (el.querySelector('.js-plotly-plot') || el.classList.contains('js-plotly-plot')) { pending.set(el, job); draw(el); }
  }
});

let t;
window.addEventListener('resize', () => {
  clearTimeout(t);
  t = setTimeout(() => document.querySelectorAll('.plot .js-plotly-plot').forEach((p) => window.Plotly?.Plots.resize(p)), 150);
});

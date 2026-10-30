// Plotly wrapper: merges the dark theme into server-built figures and renders lazily when visible.
const C = { text: '#E6EAF2', muted: '#8A94A6', grid: '#1A2433', line: '#222D3F', panel: '#162030' };
const COLORWAY = ['#4C8DFF', '#F5B841', '#22C55E', '#E86AF0', '#2EC4B6', '#FF8A4C', '#A78BFA', '#94A3B8'];

function axisDefaults() { return { gridcolor: C.grid, zeroline: false, linecolor: C.line, tickfont: { color: C.muted }, automargin: true }; }
const isObj = (o) => o && typeof o === 'object' && !Array.isArray(o);
function merge(base, over) {
  const out = { ...base };
  for (const [k, v] of Object.entries(over || {})) out[k] = isObj(v) && isObj(base[k]) ? merge(base[k], v) : v;
  return out;
}

export function themed(fig, height) {
  const lay = { ...(fig.layout || {}) };
  const small = window.innerWidth < 640;
  const base = {
    paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
    font: { family: 'Inter, system-ui, sans-serif', color: C.text, size: small ? 11 : 12 }, colorway: COLORWAY,
    legend: { orientation: 'h', yanchor: 'top', y: -0.1, x: 0, bgcolor: 'rgba(0,0,0,0)', font: { size: 11 } },
    margin: { l: small ? 6 : 10, r: small ? 6 : 12, t: 40, b: 12 },
    hoverlabel: { bgcolor: C.panel, bordercolor: C.line, font: { color: C.text } },
    xaxis: axisDefaults(), yaxis: axisDefaults(), hovermode: 'closest', dragmode: false,
  };
  for (const k of Object.keys(lay)) if (/^[xy]axis\d+$/.test(k)) base[k] = axisDefaults();
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
  pending.set(el, { fig, height });
  if (!('IntersectionObserver' in window)) { requestAnimationFrame(() => draw(el)); return; }
  io = io || new IntersectionObserver((entries) => {
    for (const e of entries) if (e.isIntersecting) { io.unobserve(e.target); draw(e.target); }
  }, { rootMargin: '300px 0px' });
  io.observe(el);
}

let t;
window.addEventListener('resize', () => {
  clearTimeout(t);
  t = setTimeout(() => document.querySelectorAll('.plot .js-plotly-plot').forEach((p) => window.Plotly?.Plots.resize(p)), 150);
});

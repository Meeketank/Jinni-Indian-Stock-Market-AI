// Theme control: light (default), dark, or follow the device. Emits `jinni:theme` so charts can redraw.
const KEY = 'jinni.theme';
export const MODES = ['light', 'dark', 'auto'];

export function mode() {
  try { const m = localStorage.getItem(KEY); return MODES.includes(m) ? m : 'light'; } catch { return 'light'; }
}
export const isDark = () => document.documentElement.getAttribute('data-theme') === 'dark';

function resolve(m) {
  return m === 'dark' || (m === 'auto' && window.matchMedia?.('(prefers-color-scheme: dark)').matches);
}
function apply() {
  const dark = resolve(mode());
  const root = document.documentElement;
  if (dark) root.setAttribute('data-theme', 'dark'); else root.removeAttribute('data-theme');
  document.querySelector('meta[name="theme-color"]')?.setAttribute('content', dark ? '#0A0E14' : '#F4F6FA');
  window.dispatchEvent(new CustomEvent('jinni:theme', { detail: { dark } }));
}
export function setMode(m) {
  try { localStorage.setItem(KEY, m); } catch { /* ignore */ }
  apply();
}
export function toggle() { setMode(isDark() ? 'light' : 'dark'); }
window.matchMedia?.('(prefers-color-scheme: dark)').addEventListener?.('change', () => { if (mode() === 'auto') apply(); });

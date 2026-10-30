// Runs before first paint so there is no flash of the wrong theme. Light is the default.
(function () {
  var mode = 'light';
  try { mode = localStorage.getItem('jinni.theme') || 'light'; } catch (e) { /* ignore */ }
  var dark = mode === 'dark' || (mode === 'auto' && window.matchMedia && matchMedia('(prefers-color-scheme: dark)').matches);
  var root = document.documentElement;
  if (dark) root.setAttribute('data-theme', 'dark'); else root.removeAttribute('data-theme');
  var m = document.querySelector('meta[name="theme-color"]');
  if (m) m.setAttribute('content', dark ? '#0A0E14' : '#F4F6FA');
})();

import { h, pageHead, NumberInput, Slider, Toggle, card, formGrid, button, disclaimer, toast, download, FileText, Results, renderBlocks } from '../ui.js';
import { get, clearCache } from '../api.js';
import { store, DEFAULT_RULES } from '../store.js';
import { mode, setMode } from '../theme.js';

export async function render(view) {
  view.append(pageHead('Settings', 'Assumptions and rules apply across the app. Everything is stored in this browser only.'));
  const s = store.settings();
  const cur = mode();
  const seg = h('div', { class: 'row' }, [['light', 'Light'], ['dark', 'Dark'], ['auto', 'Match device']].map(([m, t]) => {
    const b = button(t, () => { setMode(m); seg.querySelectorAll('button').forEach((x) => x.classList.toggle('primary', x === b)); }, m === cur ? 'primary' : '');
    return b;
  }));
  view.append(card(h('h3', {}, 'Appearance'), seg));
  const rf = NumberInput({ label: 'Risk-free rate (annual %)', value: +(s.rf * 100).toFixed(2), min: 0, max: 15, step: 0.05, help: 'Used for Sharpe, Sortino and alpha. Default ≈ India 91-day T-bill / 10Y G-Sec.' });
  const cap = NumberInput({ label: 'Default capital (₹)', value: s.capital, min: 10000, step: 50000 });
  const rk = NumberInput({ label: 'Risk per trade (%)', value: +(s.risk_pct * 100).toFixed(2), min: 0.1, max: 5, step: 0.1 });
  view.append(card(h('h3', {}, 'Assumptions'), formGrid(rf, cap, rk), h('div', { class: 'row', style: { marginTop: '12px' } },
    button('Save assumptions', () => { store.saveSettings({ rf: (rf.get() ?? 6.5) / 100, capital: cap.get() || 1e6, risk_pct: (rk.get() ?? 1) / 100 }); toast('Saved'); }, 'primary'))));

  const r = store.rules();
  const R = {
    loss: NumberInput({ label: 'Max loss per trade (₹)', value: r.max_loss_per_trade, min: 100, step: 1000, help: 'Rupees lost if the stop fills. Decides position size.' }),
    rr: NumberInput({ label: 'Min reward : risk', value: r.min_reward_risk, min: 0.5, max: 5, step: 0.25 }),
    pos: Slider({ label: 'Max position (% of capital)', min: 1, max: 50, value: Math.round(r.max_position_pct * 100), fmt: (v) => `${v}%` }),
    sec: Slider({ label: 'Max sector (% of capital)', min: 5, max: 100, value: Math.round(r.max_sector_pct * 100), fmt: (v) => `${v}%` }),
    cash: Slider({ label: 'Cash buffer', min: 0, max: 50, value: Math.round(r.cash_buffer_pct * 100), fmt: (v) => `${v}%` }),
    open: NumberInput({ label: 'Max open trades', value: r.max_open_trades, min: 1, max: 100 }),
    prob: Slider({ label: 'Min P(target first)', min: 0, max: 90, value: Math.round(r.min_probability * 100), fmt: (v) => `${v}%` }),
    stop: Toggle({ label: 'Require a stop-loss', value: r.require_stop }),
  };
  view.append(card(h('h3', {}, 'Risk rules'), h('p', { class: 'hint' }, 'Every trade plan in Return Finder is checked against these.'), formGrid(...Object.values(R)),
    h('div', { class: 'row', style: { marginTop: '12px' } }, button('Save rules', () => {
      store.saveRules({ max_loss_per_trade: R.loss.get() ?? DEFAULT_RULES.max_loss_per_trade, min_reward_risk: R.rr.get() ?? 1.5, max_position_pct: R.pos.get() / 100,
        max_sector_pct: R.sec.get() / 100, cash_buffer_pct: R.cash.get() / 100, max_open_trades: R.open.get() ?? 8, min_probability: R.prob.get() / 100, require_stop: R.stop.get() });
      toast('Rules saved');
    }, 'primary'), button('Reset to defaults', () => { store.saveRules(DEFAULT_RULES); toast('Defaults restored'); location.reload(); }))));

  const wl = h('div'); const paintWl = () => {
    const w = store.watchlist(); wl.replaceChildren(...(w.length ? w.map((t) => h('div', { class: 'row', style: { marginBottom: '6px' } }, h('a', { class: 'grow', href: `#/stock/${encodeURIComponent(t)}` }, t),
      button('Remove', () => { store.removeWatch(t); paintWl(); }, 'small'))) : [h('div', { class: 'hint' }, 'Empty. Add stocks from the Analyzer.')]));
  }; paintWl();
  view.append(card(h('h3', {}, 'Watchlist'), wl));

  const dm = Toggle({ label: 'Demo mode (synthetic prices, for previews and offline use)', value: s.demo });
  const status = new Results();
  const check = async () => {
    status.loading('Checking the data feed…');
    try { const d = await get('/health', { deep: 1 }, { fresh: true });
      status.el.replaceChildren(h('div', { class: `callout ${d.live_data || d.demo ? 'ok' : 'error'}` }, d.demo ? 'Demo mode is on: no live request was made.' :
        d.live_data ? `Live data is working. Latest NIFTY close: ${d.latest}.` : `Live data is not reachable right now (${d.error || 'no response from the data provider'}).`));
    } catch (e) { status.error(e); }
  };
  const importer = FileText({ label: '', accept: '.json,application/json', button: 'Restore backup', onText: (t) => { try { store.importAll(JSON.parse(t)); toast('Backup restored'); setTimeout(() => location.reload(), 600); } catch (e) { toast(e.message); } } });
  view.append(card(h('h3', {}, 'Data'), formGrid(dm), h('div', { class: 'row', style: { margin: '10px 0' } },
    button('Save', () => { store.saveSettings({ demo: dm.get() }); location.reload(); }, 'primary'), button('Check live data', check), button('Clear cached results', () => { clearCache(); toast('Cache cleared'); })),
    status.el, h('p', { class: 'hint' }, 'Prices come from Yahoo Finance and are cached for a few minutes to respect rate limits. If live data is unavailable, JINNI says so instead of showing made-up numbers.')),
    card(h('h3', {}, 'Backup'), h('p', { class: 'hint' }, 'Your holdings, watchlist, journal and settings live only in this browser. Export a backup before clearing browser data or switching devices.'),
      h('div', { class: 'row' }, button('Export backup', () => download('jinni_backup.json', JSON.stringify(store.exportAll(), null, 2), 'application/json'), '', 'download'), importer)),
    disclaimer());
}

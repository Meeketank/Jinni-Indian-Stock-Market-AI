import { h, pageHead, Select, Segmented, TextInput, TextArea, Slider, tabsUI, disclaimer, catalog, button } from '../ui.js';
import { get } from '../api.js';
import { store } from '../store.js';
import { livePanel, showIf } from './kit.js';

export async function render(view) {
  const cat = await catalog();
  view.append(pageHead('Sentiment & Events', 'ML outlook, live multi-source news sentiment, market mood and an event lab that answers "when X happens, does the market always react?".'));
  const measure = [...Object.entries(cat.indices), ...cat.nifty50.map((x) => [x.t, x.name])];
  const tabs = [
    { label: 'Outlook', load: (p) => {
      p.append(h('p', { class: 'page-sub' }, 'A machine-learning estimate of where NIFTY heads next, built from global cues (US close, Asia, crude, gold, dollar, rupee, yields, VIX) plus a small tilt from live news. It always shows how well it did on data it had never seen.'));
      const hz = Segmented({ label: 'Horizon', options: [[1, 'Next session'], [5, 'Next 5 sessions']], value: 1 });
      livePanel(p, [hz], () => get('/outlook/market', { h: hz.get() }));
    } },
    { label: 'Live news', load: (p) => livePanel(p, [], () => get('/sentiment/live')) },
    { label: 'Market mood', load: (p) => livePanel(p, [], () => get('/mood')) },
    { label: 'News tone', load: (p) => {
      const sy = TextInput({ label: 'Symbols', value: ['^NSEI', ...store.watchlist().slice(0, 5)].join(', ') || '^NSEI', help: 'Comma separated, e.g. ^NSEI, RELIANCE, TCS' });
      livePanel(p, [sy], () => get('/news', { tickers: sy.get() || '^NSEI' }), { runLabel: 'Read the news' });
    } },
    { label: 'Event lab', load: (p) => {
      p.append(h('p', { class: 'page-sub' }, 'Pick what to measure and define the event, either as a recurring rule or by pasting dates. JINNI checks every past occurrence and answers honestly.'));
      const what = Select({ label: 'Measure', options: measure, value: '^NSEI' });
      const years = Segmented({ label: 'History', options: ['3y', '5y', '10y'], value: '10y' });
      const mode = Segmented({ label: 'Event', options: [['rule', 'Recurring rule'], ['builtin', 'Built-in event'], ['dates', 'Paste dates']], value: 'rule' });
      const nth = Select({ label: 'Which', options: ['1st', '2nd', '3rd', '4th', 'Last', 'Every'], value: '2nd' });
      const day = Select({ label: 'Day', options: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'], value: 'Friday' });
      const ev = Select({ label: 'Event', options: cat.events, value: cat.events[0], help: 'Built-in dates cover recent years; please verify any you rely on.' });
      const dates = TextArea({ label: 'Dates (YYYY-MM-DD, one per line or comma separated)', value: '', placeholder: 'e.g. 2025-04-02', rows: 4, cls: 'full' });
      const off = Segmented({ label: 'Reaction measured on', options: [['0', 'Same session'], ['1', 'Next session (event after 3:30 pm IST)']], value: '1' });
      const win = Slider({ label: 'Window around the event (trading days)', min: 1, max: 15, value: 5 });
      const sync = () => { const m = mode.get(); showIf(nth, () => m === 'rule'); showIf(day, () => m === 'rule'); showIf(ev, () => m === 'builtin'); showIf(dates, () => m === 'dates'); showIf(off, () => m === 'dates'); };
      mode.onChange(sync); sync();
      livePanel(p, [what, years, mode, nth, day, ev, dates, off, win], () => get('/events/study', { symbol: what.get(), years: years.get(), mode: mode.get(), nth: nth.get(), day: day.get(),
        event: ev.get(), dates: dates.get().replace(/\n/g, ','), offset: off.get(), window: win.get() }), { runLabel: 'Run event study' });
    } },
    { label: 'Calendar', load: (p) => { const s = Select({ label: 'Market / stock', options: measure, value: '^NSEI' }); livePanel(p, [s], () => get('/events/calendar', { symbol: s.get() })); } },
  ];
  view.append(tabsUI(tabs, { onShow: (i, panel, it) => it.load(panel) }).el, disclaimer());
}

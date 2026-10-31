import { pageHead, Select, TextInput, Chips, Slider, MultiPick, disclaimer, catalog, h } from '../ui.js';
import { get } from '../api.js';
import { store } from '../store.js';
import { livePanel, showIf } from './kit.js';

export async function render(view) {
  const cat = await catalog();
  view.append(pageHead('Screener', 'Rank stocks by trend, momentum, timing, volume and risk, and filter by active chart setups.'));
  const uni = Select({ label: 'Universe', options: [['nifty50', 'NIFTY 50'], ['watchlist', 'My watchlist'], ['custom', 'Custom list']], value: 'nifty50' });
  const custom = TextInput({ label: 'Symbols', value: 'IRCTC, TATAELXSI, POLYCAB, DIXON, HAL, BEL', help: 'Comma separated, up to 60.' });
  const ratings = Chips({ label: 'Rating', options: ['STRONG BUY', 'BUY', 'HOLD', 'SELL', 'STRONG SELL'], value: ['STRONG BUY', 'BUY', 'HOLD', 'SELL', 'STRONG SELL'], cls: 'full' });
  const minScore = Slider({ label: 'Min score', min: 0, max: 100, value: 0 });
  const maxRsi = Slider({ label: 'Max RSI', min: 30, max: 100, value: 100 });
  const sectors = MultiPick({ label: 'Sectors (NIFTY 50 only)', groups: [{ name: 'Sectors', items: cat.sectors.map((s) => ({ name: s })) }], noun: 'sectors' });
  const setups = MultiPick({ label: 'Chart setup active today', groups: [{ name: 'Setups', items: cat.setups.map((s) => ({ name: s })) }], noun: 'setups' });
  const sync = () => showIf(custom, () => uni.get() === 'custom'); uni.onChange(sync); sync();
  const tickers = () => { const u = uni.get(); if (u === 'nifty50') return 'nifty50';
    const l = u === 'watchlist' ? store.watchlist().join(',') : custom.get(); return l || 'nifty50'; };
  livePanel(view, [uni, custom, ratings, minScore, maxRsi, sectors, setups], () => {
    if (uni.get() === 'watchlist' && !store.watchlist().length) return Promise.resolve({ blocks: [{ type: 'callout', kind: 'info', text: 'Your watchlist is empty. Add stocks from the Analyzer.' }] });
    return get('/screen', { tickers: tickers(), ratings: ratings.get().join(','), min_score: minScore.get(), max_rsi: maxRsi.get(), sectors: sectors.get().join(','), setups: setups.get().join(',') });
  });
  view.append(disclaimer());
}

import { h, pageHead, Results, card, button, disclaimer, download, toast, icon } from '../ui.js';
import { post } from '../api.js';
import { store } from '../store.js';

export async function render(view) {
  view.append(pageHead('Journal', "JINNI's real report card. Every prediction you track is checked against the actual price path after it was made. This is the only accuracy number that counts."));
  const res = new Results(); const tools = h('div', { class: 'row', style: { margin: '10px 0' } });
  const load = async () => {
    const preds = store.journal();
    tools.replaceChildren();
    if (!preds.length) { res.show([{ type: 'callout', kind: 'info', text: 'Nothing tracked yet. In **Return Finder**, use **Track top 5** or **Track this trade plan**.' }]); return; }
    res.loading('Scoring predictions against prices…');
    try {
      const d = await post('/journal/score', { predictions: preds.map(({ id, ticker, entry_date, entry, target_price, stop_price, target_pct, horizon, p_target, grade, qty, source }) =>
        ({ id, ticker, entry_date, entry, target_price, stop_price, target_pct, horizon, p_target: p_target ?? 0, grade: grade || '', qty: qty || 0, source: source || 'pick' })) });
      res.show(d.blocks, { onSelect: (action, id) => { if (action === 'journal-row') pickRow(id); } });
      tools.append(button('Export JSON', () => download('jinni_journal.json', JSON.stringify(preds, null, 2), 'application/json'), 'small', 'download'),
        button('Clear all', () => { if (confirm('Delete every tracked prediction?')) { store.clearJournal(); load(); } }, 'small danger', 'trash'));
    } catch (e) { res.error(e, load); }
  };
  const pickRow = (id) => { if (confirm('Remove this prediction from the journal?')) { store.removePrediction(id); toast('Removed'); load(); } };
  view.append(res.el, tools, h('div', { class: 'caption' }, 'Predictions are stored in this browser. Export a backup in Settings to keep them across devices.'), disclaimer());
  load();
}

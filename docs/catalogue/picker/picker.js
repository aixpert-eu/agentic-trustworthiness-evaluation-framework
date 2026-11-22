// ---- METRIC PICKER (added to the original page) ----
// Lets people choose which metrics a use case needs. The choice is a list of metric ids that can be downloaded as
// JSON and loaded again here or in the demo dashboards. Selection state lives in this page (and, when the browser
// allows it, localStorage); nothing is sent anywhere.
const SELECTION_SCHEMA = 'aixpert-metric-selection/1';
const STORE_KEY = 'aixpert-catalogue-selection';
const picked = new Set();
const byId = new Map(metrics.map(m => [m.id, m]));
const displayName = m => ({ bleu: 'BLEU', rouge_l: 'ROUGE-L', meteor: 'METEOR', bertscore: 'BERTScore',
  ece: 'Expected calibration error', pass_at_k_agent: 'Pass@k (agent)', pass_hat_k_agent: 'Pass^k (agent)',
  judge_elo: 'Judge Elo' }[m.id] || m.name);
const pickEl = id => document.getElementById(id);

try {
  const saved = JSON.parse(localStorage.getItem(STORE_KEY) || '{}');
  (saved.metrics || []).forEach(id => byId.has(id) && picked.add(id));
  if (saved.label) pickEl('sel-label').value = saved.label;
} catch (e) { /* storage unavailable: the page still works, the selection just is not remembered */ }

function persist() {
  try { localStorage.setItem(STORE_KEY, JSON.stringify({ metrics: [...picked], label: pickEl('sel-label').value })); } catch (e) {}
}
function selectionPayload() {
  return { schema: SELECTION_SCHEMA, label: pickEl('sel-label').value.trim(), created: new Date().toISOString(),
    source: 'AIXPERT evaluation framework catalogue', metrics: metrics.filter(m => picked.has(m.id)).map(m => m.id) };
}
function setStatus(text) { pickEl('sel-status').textContent = text; }

function toggle(id, force) {
  const on = force === undefined ? !picked.has(id) : force;
  on ? picked.add(id) : picked.delete(id);
  refresh();
}
function pickable(card, m) {
  const w = document.createElement('div');
  w.className = 'cell-wrap';
  w.dataset.id = m.id;
  const b = document.createElement('button');
  b.type = 'button';
  b.className = 'pick';
  const mark = document.createElement('span');
  mark.className = 'pick-mark';
  mark.setAttribute('aria-hidden', 'true');
  b.append(mark, document.createTextNode(''));
  b.onclick = () => toggle(m.id);
  w.append(card, b);
  paintCard(w);
  return w;
}
function paintCard(w) {
  const m = byId.get(w.dataset.id), on = picked.has(m.id), b = w.querySelector('.pick');
  w.classList.toggle('is-picked', on);
  b.setAttribute('aria-pressed', String(on));
  b.setAttribute('aria-label', (on ? 'Remove ' : 'Add ') + displayName(m) + (on ? ' from' : ' to') + ' my selection');
  b.querySelector('.pick-mark').textContent = on ? '✓' : '';
  b.lastChild.textContent = on ? 'In selection' : 'Add to selection';
}
function syncDialogPick(m) {
  const b = pickEl('dialog-pick'), on = picked.has(m.id);
  b.dataset.id = m.id;
  b.setAttribute('aria-pressed', String(on));
  b.textContent = on ? '✓ In my selection' : '+ Add to my selection';
}

function refresh() {
  document.querySelectorAll('.cell-wrap').forEach(paintCard);
  const cur = pickEl('dialog-pick').dataset.id;
  if (cur) syncDialogPick(byId.get(cur));
  const n = picked.size;
  pickEl('tray').hidden = n === 0;
  document.body.classList.toggle('has-tray', n > 0);
  pickEl('tray-count').textContent = n + (n === 1 ? ' metric selected' : ' metrics selected');
  // dimension coverage of the selection
  const box = pickEl('dimcards');
  box.replaceChildren();
  dimensions.forEach(d => {
    const inDim = metrics.filter(m => m.dims.includes(d.name)), sel = inDim.filter(m => picked.has(m.id));
    const card = document.createElement('div');
    card.className = 'dimcard' + (inDim.length === 0 ? ' empty-dim' : '');
    const h = document.createElement('h3');
    h.textContent = d.name;
    const cov = document.createElement('div');
    cov.className = 'cov';
    cov.innerHTML = inDim.length ? '<b>' + sel.length + '</b> selected of ' + inDim.length + ' available'
      : 'No registered metric yet (a taxonomy dimension without an implementation)';
    const chips = document.createElement('div');
    chips.className = 'chips';
    sel.forEach(m => {
      const c = document.createElement('span');
      c.className = 'chip';
      c.append(document.createTextNode(displayName(m)));
      const x = document.createElement('button');
      x.type = 'button';
      x.textContent = '×';
      x.setAttribute('aria-label', 'Remove ' + displayName(m) + ' from my selection');
      x.onclick = () => toggle(m.id, false);
      c.append(x);
      chips.append(c);
    });
    card.append(h, cov, chips);
    box.append(card);
  });
  persist();
}

pickEl('dialog-pick').onclick = () => { const id = pickEl('dialog-pick').dataset.id; if (id) toggle(id); };
pickEl('sel-all').onclick = () => {
  document.querySelectorAll('.cell-wrap').forEach(w => picked.add(w.dataset.id));
  refresh();
  setStatus('Added every metric currently shown. Total selected: ' + picked.size + '.');
};
const clearAll = () => { picked.clear(); refresh(); setStatus('Selection cleared.'); };
pickEl('sel-clear').onclick = clearAll;
pickEl('tray-clear').onclick = clearAll;
pickEl('sel-label').addEventListener('input', persist);

function download() {
  const payload = selectionPayload();
  if (!payload.metrics.length) { setStatus('Nothing selected yet. Add metrics from the cards first.'); return; }
  const slug = (payload.label || 'metric-selection').toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '') || 'metric-selection';
  const url = URL.createObjectURL(new Blob([JSON.stringify(payload, null, 2) + '\n'], { type: 'application/json' }));
  const a = document.createElement('a');
  a.href = url;
  a.download = slug + '.json';
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
  setStatus('Downloaded ' + slug + '.json with ' + payload.metrics.length + ' metric ids. Load it in a demo dashboard with "Metrics shown".');
}
pickEl('sel-download').onclick = download;
pickEl('tray-download').onclick = download;

pickEl('sel-copy').onclick = async () => {
  const ids = selectionPayload().metrics.join('\n');
  if (!ids) { setStatus('Nothing selected yet.'); return; }
  try { await navigator.clipboard.writeText(ids); setStatus('Copied ' + picked.size + ' metric ids to the clipboard.'); }
  catch (e) {
    const t = pickEl('sel-ids');
    t.hidden = false; t.value = ids; t.select();
    setStatus('Copy is blocked in this browser. The ids are in the box below; press Ctrl/Cmd+C.');
  }
};

// Load a selection: the JSON written by this page, a plain JSON array of ids, or ids separated by commas/spaces/lines.
function parseSelection(text) {
  let ids;
  try {
    const v = JSON.parse(text);
    ids = Array.isArray(v) ? v : (v && Array.isArray(v.metrics) ? v.metrics : null);
    if (v && v.label && !Array.isArray(v)) pickEl('sel-label').value = String(v.label);
  } catch (e) { ids = text.split(/[\s,;]+/); }
  return (ids || []).map(String).map(s => s.trim()).filter(Boolean);
}
pickEl('sel-load-btn').onclick = () => pickEl('sel-load').click();
pickEl('sel-load').addEventListener('change', async e => {
  const f = e.target.files[0];
  if (!f) return;
  const ids = parseSelection(await f.text());
  const known = ids.filter(id => byId.has(id)), unknown = ids.filter(id => !byId.has(id));
  picked.clear();
  known.forEach(id => picked.add(id));
  refresh();
  setStatus('Loaded ' + known.length + ' metrics from ' + f.name + (unknown.length ? '. Not in this catalogue: ' + unknown.join(', ') + '.' : '.'));
  e.target.value = '';
});

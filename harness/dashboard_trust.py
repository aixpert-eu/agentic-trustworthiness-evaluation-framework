"""Trustworthiness-by-dimension section shared by the demo dashboards.

The dashboards open with the general trustworthiness metrics, grouped by the ten framework dimensions, and
follow with the task-specific results. This module owns everything that is the same for both:

    taxonomy()          the ordered dimensions and what the metric registry offers per dimension (built at page
                        build time, so old results.json files get the same view)
    load_selection()    reads a metric selection (the JSON exported by the catalogue picker, a JSON list of ids, or
                        ids separated by commas or lines)
    apply_trust()       fills three template placeholders: __TRUST_CSS__, __TRUST_SECTION__, __TRUST_JS__

A dashboard template calls ``trustInit({rows, rowHtml, metrics})`` from its script. Each metric is
``{id, dim, label, val, res, html, missing}``:

    val(c)     the config's value, or null when this run has none (the metric then counts as "not measured")
    run        true for a metric of the whole comparison (for example discriminative power), shown under the
               table instead of as a column; its val() ignores the configuration
    res(c)     the banded result object (with band_level / band_label) or null, used for the optional dimension band
    html(c)    the cell content
    missing    why the value can be absent, shown next to the metric when it is

A page never hides a gap: a dimension without a measured metric says so, a metric with no value in this run is
listed as not measured, and metrics picked in the catalogue that the demo does not compute are listed under the
table with what they would need.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SELECTION_SCHEMA = "aixpert-metric-selection/1"

# The ten framework dimensions, in catalogue order.
DIMENSIONS = (
    "Accuracy",
    "Robustness",
    "Safety, Security & Privacy",
    "Fairness, Ethical & Legal Aspects",
    "Transparency & Explainability",
    "Accountability",
    "Autonomy",
    "Auditability / Verifiability",
    "Efficiency",
    "Human-centricity",
)
# Tags found in the registry that are not a framework dimension name.
ALIASES = {"Transparency": "Transparency & Explainability"}


def canonical_dimension(name: str) -> str:
    return ALIASES.get(name, name)


def taxonomy() -> dict[str, Any]:
    """Dimensions with their registered-metric counts, plus the registry facts the page needs for unmeasured metrics."""
    counts = {d: 0 for d in DIMENSIONS}
    registry: dict[str, dict[str, Any]] = {}
    try:
        from contracts.registry import REGISTRY
    except ImportError:  # the page still renders; it just cannot say what else the registry offers
        REGISTRY = []
    for spec in REGISTRY:
        dims = list(dict.fromkeys(canonical_dimension(d) for d in spec.dimensions))
        registry[spec.metric_id] = {"dims": dims, "cat": spec.applicability, "requires": list(spec.requires)}
        for d in dims:
            counts[d] = counts.get(d, 0) + 1
    return {"dimensions": [{"name": d, "registered": n} for d, n in counts.items()], "registry": registry}


def load_selection(source: str | Path | list | dict | None) -> dict[str, Any] | None:
    """A selection as ``{"label": str, "metrics": [ids]}``, or None when nothing was given.

    ``source`` may be a path (JSON object with a ``metrics`` list, JSON list of ids, or text with ids separated
    by commas, semicolons, spaces or lines) or an already parsed list or dict.
    """
    if source is None:
        return None
    label = ""
    if isinstance(source, (str, Path)) and Path(source).is_file():
        text = Path(source).read_text(encoding="utf-8")
        try:
            source = json.loads(text)
        except json.JSONDecodeError:
            source = [t for t in text.replace(",", " ").replace(";", " ").split() if t]
    if isinstance(source, dict):
        label = str(source.get("label") or "")
        source = source.get("metrics")
    if not isinstance(source, list):
        raise ValueError("a metric selection is a JSON list of ids or an object with a 'metrics' list")
    ids = list(dict.fromkeys(str(i).strip() for i in source if str(i).strip()))
    if not ids:
        raise ValueError("the metric selection is empty")
    return {"label": label, "metrics": ids}


TRUST_CSS = r"""
.part{font:11px/1.6 ui-monospace,monospace;letter-spacing:1.5px;text-transform:uppercase;color:var(--coral);margin:46px 0 -22px;padding-top:16px;border-top:1px solid var(--line)}
.top+.part{margin-top:26px;border-top:0;padding-top:0}
.cover{display:grid;grid-template-columns:repeat(auto-fill,minmax(185px,1fr));gap:10px;margin:8px 0 14px}
.tile{border:1px solid var(--line);border-radius:12px;padding:9px 12px;font-size:12px;line-height:1.45;color:var(--mute);background:#ffffff06}
.tile b{display:block;color:var(--ink);font-weight:600;font-size:13px;line-height:1.3;margin-bottom:2px}
.tile.on{border-color:var(--coral)}.tile.gap{border-style:dashed}.tile.none{border-style:dotted;opacity:.8}
.picker{border:1px solid var(--line);border-radius:12px;padding:9px 14px;margin:0 0 12px;background:#ffffff06}
.picker summary{font-size:13px}.picker-body{padding-top:10px}
.picker fieldset{border:1px solid var(--line);border-radius:10px;margin:0 0 10px;padding:6px 12px 8px}
.picker legend{font-size:12px;color:var(--mute);padding:0 6px}
.picker label{display:inline-block;margin:2px 16px 2px 0;font-size:13px}
.picker-actions{display:flex;flex-wrap:wrap;gap:10px 14px;align-items:center;margin:8px 0}
.btn{border:1px solid #ffffff3a;background:#ffffff08;color:var(--ink);border-radius:8px;padding:6px 12px;font-size:12px;cursor:pointer}
.btn:hover{border-color:var(--coral)}
button:focus-visible,input:focus-visible{outline:3px solid var(--coral);outline-offset:3px}
th.dimh{text-align:center;border-left:1px solid var(--line);border-bottom:1px solid var(--coral);color:var(--ink)}
th.dimh small{display:block;font-weight:400;color:var(--mute);font-size:11px}
.gl{border-left:1px solid var(--line)}
.nmg{margin:6px 0;font-size:13px}.nmg b{margin-right:8px;font-weight:600}
.nm{display:inline-block;border:1px dashed var(--line);border-radius:99px;padding:1px 10px;margin:2px 4px 2px 0;font-size:12px;color:var(--mute)}
.nm small{margin-left:4px;font-size:11px}
@media(max-width:760px){.cover{grid-template-columns:repeat(2,1fr)}.tile{padding:8px 10px}}
.missing{margin:12px 0 0;color:var(--mute);font-size:13px}.missing summary{font-size:13px}.missing h3{font:600 14px Georgia,serif;margin:0 0 6px;color:var(--ink)}
"""

TRUST_SECTION = r"""
<div class="part">Part 1 &middot; Trustworthiness metrics, by dimension</div>
<h2>Trustworthiness profile (WP3 framework)</h2>
<p class="sub">The general metrics of the WP3 registry, grouped by trustworthiness dimension. Band chips follow the framework banding (4 Excellent to 0 Unacceptable). A dimension this run does not measure is shown as a gap, never as a pass.</p>
<div class="cover" id="tcover"></div>
<details class="picker" id="tpicker"><summary id="tpick-sum"></summary><div class="picker-body">
<p class="sub">Choose which measured metrics to show, or load a selection made in the catalogue picker. Selected metrics this run did not measure are listed under the table instead of being hidden.</p>
<div id="tpick-checks"></div>
<div class="picker-actions">
<label><input type="checkbox" id="tband"> Show a dimension band (weakest measured metric in the dimension)</label>
<button class="btn" id="tall" type="button">Show all</button>
<button class="btn" id="tload" type="button">Load selection file&hellip;</button>
<input type="file" id="tfile" accept=".json,.txt,application/json,text/plain" hidden aria-label="Load a metric selection file">
</div>
<p class="sub" id="tstatus" aria-live="polite"></p>
</div></details>
<div class="panel"><table id="trust"></table></div>
<p class="sub" id="tnote"></p>
<div class="panel" id="trun" hidden></div>
<div class="missing" id="tmissing"></div>
"""

TRUST_JS = r"""
function ciText(r) {
  return r && r.ci95 ? `95% interval ${r.ci95[0].toFixed(2)} to ${r.ci95[1].toFixed(2)}` : '';
}
function ciBar(ci) {
  return ci ? `<div class="ciw" style="left:${100 * ci[0]}%;width:${Math.max(0, 100 * (ci[1] - ci[0]))}%"></div>` : '';
}
function ciPct(ci) {
  return ci ? `95% interval ${(100 * ci[0]).toFixed(0)}% to ${(100 * ci[1]).toFixed(0)}%` : '';
}
function trustInit(o) {
  const tax = R.taxonomy || {dimensions: [], registry: {}}, reg = tax.registry || {};
  const rows = o.rows || [], all = o.metrics;
  const meta = R.meta || {};
  const NEEDS = {common: 'output + reference or checks', trajectory: 'a tool-call trace', reflective: 'trials within one run',
    governance: 'governance events + labeled cases', efficiency: 'resource measurements + a budget', human_centricity: 'participant responses',
    probabilistic: 'probabilities or confidence data', aggregate: 'repeated runs or case sets', meta_evaluation: 'evaluation scores or reference data'};
  const nice = id => id.replace(/_/g, ' ').replace(/^./, ch => ch.toUpperCase());
  const dimNames = tax.dimensions.map(d => d.name);
  all.forEach(t => { if (!dimNames.includes(t.dim)) dimNames.push(t.dim); });
  const registered = Object.fromEntries(tax.dimensions.map(d => [d.name, d.registered]));
  const mById = Object.fromEntries(all.map(t => [t.id, t]));
  const measured = all.filter(t => t.run ? t.val() != null : rows.some(c => t.val(c) != null));
  const isMeasured = id => measured.some(t => t.id === id);
  const regDims = id => (reg[id] && reg[id].dims) || (mById[id] ? [mById[id].dim] : []);
  const home = id => (mById[id] && mById[id].dim) || regDims(id)[0] || 'Not in this build\'s registry';
  const nameOf = id => mById[id] ? mById[id].label : nice(id);
  const why = id => mById[id] ? (mById[id].missing || 'no value in this run') :
    reg[id] ? 'not computed by this demo (needs ' + (NEEDS[reg[id].cat] || reg[id].cat) + (reg[id].requires.length ? '; ' + reg[id].requires.join(', ') : '') + ')' :
    'not in this build\'s metric registry';
  const bandOf = (t, c) => { const r = t.res && t.res(c); return r && !r.error && r.band_level != null ? r : null; };
  const banded = t => rows.some(c => bandOf(t, c));
  let sel = R.selection ? new Set(R.selection.metrics) : null, label = R.selection ? (R.selection.label || '') : '', band = false;

  // picker checkboxes, built once so focus survives changes
  $('tpick-checks').innerHTML = dimNames.filter(d => measured.some(t => t.dim === d)).map(d =>
    `<fieldset><legend>${esc(d)}</legend>${measured.filter(t => t.dim === d).map(t => `<label><input type="checkbox" data-id="${esc(t.id)}"> ${esc(t.label)}</label>`).join('')}</fieldset>`).join('') ||
    '<p class="mute">No metric has a value in this run yet.</p>';
  const syncChecks = () => $('tpick-checks').querySelectorAll('input[data-id]').forEach(i => { i.checked = !sel || sel.has(i.dataset.id); });
  const setStatus = s => { $('tstatus').textContent = s; };

  function draw() {
    const shownAll = measured.filter(t => !sel || sel.has(t.id));
    const shown = shownAll.filter(t => !t.run), shownRun = shownAll.filter(t => t.run);
    const shownDims = dimNames.filter(d => shown.some(t => t.dim === d));
    // coverage tiles
    $('tcover').innerHTML = dimNames.map(d => {
      const k = measured.filter(t => t.dim === d).length, n = registered[d] ?? 0, s = sel ? [...sel].filter(id => regDims(id).includes(d)).length : null;
      const cls = k ? 'on' : n ? 'gap' : 'none';
      const line = k ? `${k} measured of ${n} registered` : n ? `not measured in this run (${n} registered)` : 'no registered metric yet';
      return `<div class="tile ${cls}"><b>${esc(d)}</b>${line}${s == null ? '' : `<br>${s} in your selection`}</div>`;
    }).join('');
    // picker summary
    $('tpick-sum').textContent = sel ? `Metrics shown: ${label ? '“' + label + '”' : 'your selection'} (${sel.size} chosen, ${shown.length} of them measured). Change` : `Metrics shown: all ${measured.length} measured. Choose which to show`;
    syncChecks();
    // table
    if (!rows.length) { $('trust').innerHTML = '<tr><td class="mute">No runs to show yet.</td></tr>'; }
    else if (!shown.length && !shownRun.length) { $('trust').innerHTML = '<tr><td class="mute">None of the selected metrics has a value in this run. They are listed below.</td></tr>'; }
    else if (!shown.length) { $('trust').innerHTML = '<tr><td class="mute">The selected metrics apply to the whole comparison; see below.</td></tr>'; }
    else {
      let h1 = '<tr><th rowspan="2">Configuration</th>', h2 = '<tr>';
      shownDims.forEach(d => {
        const ts = shown.filter(t => t.dim === d), k = measured.filter(t => t.dim === d).length, withBand = band && ts.some(banded);
        h1 += `<th class="dimh" colspan="${ts.length + (withBand ? 1 : 0)}">${esc(d)}<small>${k} measured of ${registered[d] ?? 0} registered</small></th>`;
        h2 += ts.map((t, i) => `<th class="${i === 0 ? 'gl' : ''}" title="${esc(t.id)}">${esc(t.label)}</th>`).join('') + (withBand ? `<th title="Lowest band among the banded metrics shown for this dimension (weakest link). Metrics without a framework band are not included.">Dimension band<small class="mute" style="display:block;font-weight:400">${ts.filter(banded).length} of ${ts.length} banded</small></th>` : '');
      });
      let body = '';
      rows.forEach(c => {
        body += `<tr><td>${o.rowHtml(c)}</td>`;
        shownDims.forEach(d => {
          const ts = shown.filter(t => t.dim === d);
          body += ts.map((t, i) => `<td class="${i === 0 ? 'gl' : ''}">${t.html(c)}</td>`).join('');
          if (band && ts.some(banded)) {
            let worst = null;
            ts.forEach(t => { const r = bandOf(t, c); if (r && (!worst || r.band_level < worst.r.band_level)) worst = {r, t}; });
            body += worst ? `<td><span class="chip b${worst.r.band_level}" title="weakest: ${esc(worst.t.label)}">${esc(worst.r.band_label)}</span></td>` : '<td class="mute">n/a</td>';
          }
        });
        body += '</tr>';
      });
      $('trust').innerHTML = h1 + '</tr>' + h2 + '</tr>' + body;
    }
    // note on k and intervals, run-level metrics
    const notes = [];
    if (meta.pass_k) notes.push(`pass^k and pass@k use k = ${meta.pass_k} for every configuration.`);
    if (meta.bootstrap) notes.push(`Hover a band chip, or read the whiskers on the bars below, for the 95% interval (${meta.bootstrap.method}, ${meta.bootstrap.n_boot} draws).`);
    $('tnote').textContent = notes.join(' ');
    $('trun').hidden = !shownRun.length;
    $('trun').innerHTML = shownRun.length ? '<div class="grp">Whole-comparison metrics</div><table><tr><th>Dimension</th><th>Metric</th><th>Result</th></tr>' +
      shownRun.map(t => `<tr><td>${esc(t.dim)}</td><td title="${esc(t.id)}">${esc(t.label)}</td><td>${t.html()}</td></tr>`).join('') + '</table>' : '';
    // what is not measured
    const dormant = sel ? [] : all.filter(t => !isMeasured(t.id));  // this demo computes them, this run has no value
    const dormantIds = new Set(dormant.map(t => t.id));
    const ids = sel ? [...sel].filter(id => !isMeasured(id)) : Object.keys(reg).filter(id => !isMeasured(id) && !dormantIds.has(id));
    const groups = {};
    ids.forEach(id => (groups[home(id)] = groups[home(id)] || []).push(id));
    const order = [...dimNames, ...Object.keys(groups).filter(g => !dimNames.includes(g))].filter(g => groups[g]);
    const list = order.map(g => `<div class="nmg"><b>${esc(g)}</b>${groups[g].map(id => `<span class="nm" title="${esc(why(id))}">${esc(nameOf(id))}${sel ? `<small>${esc(why(id))}</small>` : ''}</span>`).join('')}</div>`).join('');
    const dormantHtml = dormant.length ? `<div class="nmg"><b>The demo computes these, but this run has no value for them</b>${dormant.map(t => `<span class="nm">${esc(t.label)}<small>${esc(t.missing || 'no value in this run')}</small></span>`).join('')}</div>` : '';
    $('tmissing').innerHTML = sel ? (ids.length ? `<h3>Selected but not measured in this run (${ids.length})</h3>${list}` : '') :
      dormantHtml + (ids.length ? `<details><summary>Other registered metrics this demo does not compute (${ids.length})</summary>${list}</details>` : '');
  }

  $('tpick-checks').addEventListener('change', () => {
    const checked = [...$('tpick-checks').querySelectorAll('input[data-id]')].filter(i => i.checked).map(i => i.dataset.id);
    const extra = sel ? [...sel].filter(id => !isMeasured(id)) : [];
    if (checked.length === measured.length && !extra.length) { sel = null; label = ''; } else sel = new Set([...checked, ...extra]);
    draw();
  });
  $('tband').addEventListener('change', e => { band = e.target.checked; draw(); });
  $('tall').addEventListener('click', () => { sel = null; label = ''; setStatus('Showing every measured metric.'); draw(); });
  $('tload').addEventListener('click', () => $('tfile').click());
  $('tfile').addEventListener('change', async e => {
    const f = e.target.files[0]; if (!f) return;
    const text = await f.text(); let ids, name = '';
    try { const v = JSON.parse(text); ids = Array.isArray(v) ? v : (v && Array.isArray(v.metrics) ? v.metrics : null); name = v && !Array.isArray(v) && v.label ? String(v.label) : ''; }
    catch (err) { ids = text.split(/[\s,;]+/); }
    ids = (ids || []).map(s => String(s).trim()).filter(Boolean);
    if (!ids.length) { setStatus('That file holds no metric ids, so nothing changed.'); e.target.value = ''; return; }
    sel = new Set(ids); label = name;
    const unknown = ids.filter(id => !reg[id] && !mById[id]);
    setStatus(`Loaded ${sel.size} metric ids from ${f.name}${unknown.length ? '. Not in this build\'s registry: ' + unknown.join(', ') : ''}.`);
    e.target.value = ''; draw();
  });
  draw();
}
"""


def apply_trust(template: str) -> str:
    """Fill the trust placeholders. Call this BEFORE inserting the run data."""
    return (
        template.replace("__TRUST_CSS__", TRUST_CSS)
        .replace("__TRUST_SECTION__", TRUST_SECTION)
        .replace("__TRUST_JS__", TRUST_JS)
    )


def page_data(results: dict, selection: dict | None = None) -> dict:
    """The run data plus what the trust section needs, as one JSON-ready object."""
    data = {**results, "taxonomy": taxonomy()}
    if selection:
        data["selection"] = selection
    return data

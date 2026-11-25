"""Self-contained HTML dashboard for the DiagFlowBench demo (no network, no libraries)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from harness.dashboard_theme import apply_theme
from harness.dashboard_trust import apply_trust, load_selection, page_data

TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AIXPERT WP3 DiagFlowBench demo</title>
<style>
__THEME_CSS__
__TRUST_CSS__
</style></head><body><main>
__HEADER__
__TRUST_SECTION__

<div class="part">Part 2 &middot; Task-specific metrics (DiagFlowBench)</div>
<h2>Off-procedure behaviour</h2>
<p class="sub">When the operator reports something the procedure does not cover. Right: correct abstention (the assistant recognises the gap). Left: forced mapping (a real step the input does not license) and fabrication (a step that is not in the procedure).</p>
<div class="panel" id="off"></div>

<h2>On-procedure capability</h2>
<p class="sub">When the operator reports something the procedure covers. Clean conversations.</p>
<div class="panel" id="on"></div>

<div class="part">Part 3 &middot; Details</div>
<h2>Resource use</h2><p class="sub">Per conversation turn, averaged over all runs. Cost uses the price table given at evaluation time: a model id ending in :free costs 0 (shown as free), and any other model without a price is shown as not priced and its cost is not scored. The banded latency and cost also appear under the Efficiency dimension in Part 1.</p>
<div class="panel"><table id="eff"></table></div>

<h2>Conversation drill-down</h2>
<p class="sub">Pick a configuration to see each conversation turn by turn. Highlighted rows are off-procedure injections.</p>
<div class="panel"><select id="cfg"></select><div id="items"></div></div>
<p class="notes" id="notes"></p>
__FOOTER__
</main>
<script id="data" type="application/json">__DATA__</script>
<script>
const R = JSON.parse(document.getElementById('data').textContent);
const C = R.configs, $ = id => document.getElementById(id), live = C.filter(c => !c.reference), ref = C.filter(c => c.reference);
const key = c => c.model + (c.reference ? ' (paper)' : ' | ' + c.pattern);
const color = c => ({react: 'var(--react)', basic_rag: 'var(--rag)', baseline: 'var(--base)'})[c.pattern] || 'var(--paper)';
const pct = x => x == null ? 'n/a' : (100 * x).toFixed(0) + '%', f2 = x => x == null ? 'n/a' : x.toFixed(2);
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const m = (c, id) => (c.metrics || {})[id], rate = (c, id) => m(c, id) ? m(c, id).rate : null;
const chip = r => !r ? 'n/a' : r.error ? '<span class="chip b0">error</span>' : `<span class="chip b${r.band_level}" title="${esc(ciText(r))}">${esc(r.band_label)}</span> <span class="n">${r.rate != null ? pct(r.rate) : f2(r.score)}</span>`;
$('sub').textContent = `${R.meta.dataset}. ${live.length} configuration(s)${ref.length ? ' plus ' + ref.length + ' published reference model(s)' : ''}, generated ${R.meta.generated_at.slice(0, 16).replace('T', ' ')} UTC.`;

const sorted = [...live, ...ref.slice().sort((a, b) => (rate(b, 'correct_abstention_rate') ?? 0) - (rate(a, 'correct_abstention_rate') ?? 0))];
$('off').innerHTML = '<div class="legend"><span class="sw" style="background:var(--fa)"></span>fabrication<span class="sw" style="background:var(--fm)"></span>forced mapping<span class="sw" style="background:var(--ca)"></span>correct abstention</div>' +
  sorted.map(c => {
    const fa = rate(c, 'fabrication_rate'), fm = rate(c, 'forced_mapping_rate'), ca = rate(c, 'correct_abstention_rate');
    if (ca == null) return `<div class="row"><div class="lbl">${esc(key(c))}</div><div class="mute">no judged off-procedure turns</div><div></div></div>`;
    const tip = [['correct abstention', 'correct_abstention_rate'], ['forced mapping', 'forced_mapping_rate'], ['fabrication', 'fabrication_rate']]
      .map(([l, id]) => { const r = m(c, id); return r && r.ci95 ? `${l} ${pct(r.rate)} (${ciPct(r.ci95)})` : ''; }).filter(Boolean).join('; ');
    return `<div class="div" title="${esc(tip)}" style="opacity:${c.reference ? .8 : 1}"><div class="lbl" title="${esc(key(c))}"><span class="dot" style="background:${color(c)}"></span>${esc(key(c))}</div>
      <div class="left"><div class="seg" style="width:${100 * fm}%;background:var(--fm)"></div><div class="seg" style="width:${100 * fa}%;background:var(--fa)"></div></div>
      <div class="right"><div class="seg" style="width:${100 * ca}%;background:var(--ca)"></div></div>
      <div class="n mute">&nbsp;${pct(fm + fa)} | ${pct(ca)}</div></div>`;
  }).join('');

const bars = [['step_accuracy', 'Step accuracy (reply names the next step)'], ['branch_following', 'Branch following (takes the branch the operator\'s answer selects)'], ['termination_recognition', 'Termination recognition (no further step at the end)']];
const anyCi = sorted.some(c => bars.some(([id]) => (m(c, id) || {}).ci95));
$('on').innerHTML = (anyCi ? '<div class="legend">Whisker on a bar: 95% interval, bootstrap over conversations (the published reference rows have none).</div>' : '') +
  bars.map(([id, title]) => `<div class="grp">${title}</div>` + sorted.map(c => {
  const v = rate(c, id); if (v == null) return '';
  const ci = (m(c, id) || {}).ci95;
  return `<div class="row" title="${esc(ciPct(ci))}"><div class="lbl" title="${esc(key(c))}"><span class="dot" style="background:${color(c)}"></span>${esc(key(c))}</div><div class="track"><div class="fill" style="width:${100 * v}%;background:${color(c)}"></div>${ciBar(ci)}</div><div class="n">${pct(v)}</div></div>`;
}).join('')).join('');

__TRUST_JS__
const NEEDS_RUNS = '<span class="mute">needs 2+ runs</span>', bandOnly = r => r && !r.error ? r : null;
const costText = c => c.cost_mean_usd == null ? 'not priced' : c.price_basis === 'free' ? 'free' : '$' + c.cost_mean_usd.toFixed(4);
trustInit({rows: live, rowHtml: c => `<span class="dot" style="background:${color(c)}"></span>${esc(key(c))}`, metrics: [
  {id: 'pass_hat_k_agent', dim: 'Accuracy', label: 'Turn-level reliability (pass^k)', val: c => c.pass_hat_k, res: c => c.pass_hat_k, html: c => chip(c.pass_hat_k)},
  {id: 'pass_at_k_agent', dim: 'Accuracy', label: 'Capability within k tries (pass@k)', val: c => c.pass_at_k, res: c => c.pass_at_k, html: c => chip(c.pass_at_k),
    missing: 'needs the evaluate stage of this version (rerun it on the same runs)'},
  {id: 'cross_run_consistency', dim: 'Robustness', label: 'Repeatability (cross-run)', val: c => c.cross_run_consistency, res: c => c.cross_run_consistency,
    html: c => c.cross_run_consistency ? chip(c.cross_run_consistency) : NEEDS_RUNS, missing: 'needs 2 or more runs per conversation'},
  {id: 'error_recovery_rate', dim: 'Robustness', label: 'Recovery after a failed off-procedure turn', val: c => m(c, 'error_recovery_rate'), res: c => m(c, 'error_recovery_rate'), html: c => chip(m(c, 'error_recovery_rate')),
    missing: 'needs judged off-procedure turns'},
  {id: 'over_abstention_rate', dim: 'Autonomy', label: 'Over-abstention (valid input declined)', val: c => m(c, 'over_abstention_rate'), res: c => m(c, 'over_abstention_rate'), html: c => chip(m(c, 'over_abstention_rate')),
    missing: 'needs judged turns'},
  {id: 'escalation_appropriateness', dim: 'Autonomy', label: 'Abstains only when it should (F1)', val: c => m(c, 'escalation_appropriateness'), res: c => m(c, 'escalation_appropriateness'),
    html: c => { const e = m(c, 'escalation_appropriateness'); return e ? `<span class="chip b${e.band_level}">${esc(e.band_label)}</span> <span class="n">${f2(e.f1)}</span>` : 'n/a'; }, missing: 'needs judged turns'},
  {id: 'log_completeness', dim: 'Transparency & Explainability', label: 'Tool-call log completeness', val: c => c.log_completeness, res: () => null,
    html: c => c.log_completeness == null ? '<span class="mute">no tool calls</span>' : `<span class="n">${f2(c.log_completeness)}</span>`, missing: 'needs a pattern that makes tool calls (and the evaluate stage of this version)'},
  {id: 'latency', dim: 'Efficiency', label: 'Latency per turn', val: c => c.latency_mean_s, res: c => bandOnly(c.latency),
    html: c => c.latency_mean_s == null ? 'n/a' : c.latency_mean_s.toFixed(1) + ' s ' + (bandOnly(c.latency) ? chip({...c.latency, rate: null}) : '')},
  {id: 'cost_per_query', dim: 'Efficiency', label: 'Cost per turn', val: c => c.cost_mean_usd, res: c => bandOnly(c.cost),
    html: c => costText(c) + ' ' + (bandOnly(c.cost) ? chip({...c.cost, rate: null}) : '')},
  {id: 'discriminative_power', dim: 'Auditability / Verifiability', run: true, label: 'Discriminative power (does the score separate the configurations)',
    val: () => ((R.meta.meta_evaluation || {}).discriminative_power), res: () => null, missing: 'needs two or more configurations (and the evaluate stage of this version)',
    html: () => { const r = R.meta.meta_evaluation.discriminative_power; return r.error ? '<span class="chip b0">error</span>' :
      `<span class="chip b${r.band_level}">${esc(r.band_label)}</span> <span class="mute">${esc(r.explanation)} Score used: ${esc(r.score_used)}.</span>`; }}
]});

$('eff').innerHTML = '<tr><th>Configuration</th><th>Latency per turn</th><th>Cost per turn</th><th>Tokens in / out per turn</th><th>Tool calls per turn</th><th>Errored runs</th></tr>' +
  live.map(c => `<tr><td><span class="dot" style="background:${color(c)}"></span>${esc(key(c))}</td><td>${c.latency_mean_s == null ? 'n/a' : c.latency_mean_s.toFixed(1) + ' s'} ${c.latency && !c.latency.error ? chip({...c.latency, rate: null}) : ''}</td><td>${costText(c)} ${c.cost && !c.cost.error ? chip({...c.cost, rate: null}) : ''}</td><td class="n">${Math.round(c.tokens_in_mean ?? 0)} / ${Math.round(c.tokens_out_mean ?? 0)}</td><td class="n">${f2(c.tool_calls_mean)}</td><td class="n">${c.n_errors} of ${c.n_runs}</td></tr>`).join('');

live.forEach((c, i) => $('cfg').insertAdjacentHTML('beforeend', `<option value="${i}">${esc(key(c))}</option>`));
const outChip = t => t.outcome ? `<span class="chip ${t.outcome === 'correct_abstention' ? 'b4' : 'b0'}">${esc(t.outcome.replace('_', ' '))}</span>` : '';
function items() {
  const c = live[+$('cfg').value]; if (!c) { $('items').innerHTML = ''; return; }
  $('items').innerHTML = c.items.map(it => {
    const first = it.runs[0], ok = first.turns.filter(t => t.ok).length;
    const runs = it.runs.map(r => `<details><summary>run ${r.run}: ${r.turns.filter(t => t.ok).length} of ${r.n_gold_turns} turns ok${r.error ? ' <span class="no">(error: ' + esc(r.error) + ')</span>' : ''}</summary>
      <table><tr><th>#</th><th>Operator</th><th>Assistant reply</th><th>Reply matches step</th><th></th></tr>${r.turns.map(t => `<tr class="${t.is_injection ? 'inj' : ''}"><td class="n">${t.turn_index}</td><td>${esc(t.operator_utterance)}${t.is_injection ? '<br><span class="mute">off-procedure: ' + esc(t.category) + '</span>' : ''}</td><td>${esc(t.model_response)}</td><td class="mute">${esc(t.matched_text ?? 'no step')}${!t.is_injection && t.expected_text != null ? '<br>expected: ' + esc(t.expected_text) : ''}</td><td>${t.ok == null ? '' : `<span class="${t.ok ? 'ok' : 'no'}">${t.ok ? '&#10003;' : '&#10007;'}</span>`} ${outChip(t)}</td></tr>`).join('')}</table></details>`).join('');
    return `<details><summary><b>${esc(it.regime)}</b> &nbsp; ${esc(it.conversation_id)} &nbsp; <span class="mute">${ok} of ${first.n_gold_turns} turns ok (run ${first.run})</span></summary>${runs}</details>`;
  }).join('');
}
$('cfg').onchange = items; items();
$('notes').innerHTML = R.meta.notes.map(esc).join('<br>');
</script></body></html>
"""


def build_html(results: dict, selection: dict | None = None) -> str:
    """``selection`` is an optional metric selection (see harness.dashboard_trust.load_selection) shown on open."""
    themed = apply_trust(apply_theme(TEMPLATE, line1="AIXPERT WP3 evaluation framework", line2="DiagFlowBench demo",
                                     footer="AIXPERT WP3 evaluation framework: DiagFlowBench demo"))
    return themed.replace("__DATA__", json.dumps(page_data(results, selection)).replace("</", "<\\/"))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results_diagflow.json")
    ap.add_argument("--out", default="dashboard_diagflow.html")
    ap.add_argument("--metrics", help="metric selection file exported by the catalogue picker; the page opens showing only these")
    args = ap.parse_args(argv)
    results = json.loads(Path(args.results).read_text(encoding="utf-8"))
    Path(args.out).write_text(build_html(results, load_selection(args.metrics)), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

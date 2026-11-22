"""Self-contained HTML dashboard for the DiagFlowBench demo (no network, no libraries)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from harness.dashboard_theme import apply_theme

TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AIXPERT WP3 DiagFlowBench demo</title>
<style>
__THEME_CSS__
</style></head><body><main>
__HEADER__

<h2>Off-procedure behaviour</h2>
<p class="sub">When the operator reports something the procedure does not cover. Right: correct abstention (the assistant recognises the gap). Left: forced mapping (a real step the input does not license) and fabrication (a step that is not in the procedure).</p>
<div class="panel" id="off"></div>

<h2>On-procedure capability</h2>
<p class="sub">When the operator reports something the procedure covers. Clean conversations.</p>
<div class="panel" id="on"></div>

<h2>Trustworthiness metrics (WP3 framework)</h2>
<p class="sub">Band chips follow the framework banding (4 Excellent to 0 Unacceptable). Reliability metrics need several runs per conversation to differ from a single run.</p>
<div class="panel"><table id="trust"></table></div>

<h2>Efficiency</h2><p class="sub">Per conversation turn, averaged over all runs. Cost uses the price table given at evaluation time (free models cost 0).</p>
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
const chip = r => !r ? 'n/a' : r.error ? '<span class="chip b0">error</span>' : `<span class="chip b${r.band_level}">${esc(r.band_label)}</span> <span class="n">${r.rate != null ? pct(r.rate) : f2(r.score)}</span>`;
$('sub').textContent = `${R.meta.dataset}. ${live.length} configuration(s)${ref.length ? ' plus ' + ref.length + ' published reference model(s)' : ''}, generated ${R.meta.generated_at.slice(0, 16).replace('T', ' ')} UTC.`;

const sorted = [...live, ...ref.slice().sort((a, b) => (rate(b, 'correct_abstention_rate') ?? 0) - (rate(a, 'correct_abstention_rate') ?? 0))];
$('off').innerHTML = '<div class="legend"><span class="sw" style="background:var(--fa)"></span>fabrication<span class="sw" style="background:var(--fm)"></span>forced mapping<span class="sw" style="background:var(--ca)"></span>correct abstention</div>' +
  sorted.map(c => {
    const fa = rate(c, 'fabrication_rate'), fm = rate(c, 'forced_mapping_rate'), ca = rate(c, 'correct_abstention_rate');
    if (ca == null) return `<div class="row"><div class="lbl">${esc(key(c))}</div><div class="mute">no judged off-procedure turns</div><div></div></div>`;
    return `<div class="div" style="opacity:${c.reference ? .8 : 1}"><div class="lbl" title="${esc(key(c))}"><span class="dot" style="background:${color(c)}"></span>${esc(key(c))}</div>
      <div class="left"><div class="seg" style="width:${100 * fm}%;background:var(--fm)"></div><div class="seg" style="width:${100 * fa}%;background:var(--fa)"></div></div>
      <div class="right"><div class="seg" style="width:${100 * ca}%;background:var(--ca)"></div></div>
      <div class="n mute">&nbsp;${pct(fm + fa)} | ${pct(ca)}</div></div>`;
  }).join('');

const bars = [['step_accuracy', 'Step accuracy (reply names the next step)'], ['branch_following', 'Branch following (takes the branch the operator\'s answer selects)'], ['termination_recognition', 'Termination recognition (no further step at the end)']];
$('on').innerHTML = bars.map(([id, title]) => `<div class="grp">${title}</div>` + sorted.map(c => {
  const v = rate(c, id); if (v == null) return '';
  return `<div class="row"><div class="lbl" title="${esc(key(c))}"><span class="dot" style="background:${color(c)}"></span>${esc(key(c))}</div><div class="track"><div class="fill" style="width:${100 * v}%;background:${color(c)}"></div></div><div class="n">${pct(v)}</div></div>`;
}).join('')).join('');

$('trust').innerHTML = '<tr><th>Configuration</th><th>Turn-level reliability (pass^k)</th><th>Repeatability (cross-run)</th><th>Recovery after a failed off-procedure turn</th><th>Over-abstention (valid input declined)</th><th>Abstains only when it should (F1)</th></tr>' +
  live.map(c => { const e = m(c, 'escalation_appropriateness');
    return `<tr><td><span class="dot" style="background:${color(c)}"></span>${esc(key(c))}</td><td>${chip(c.pass_hat_k)}</td><td>${chip(c.cross_run_consistency)}</td><td>${chip(m(c, 'error_recovery_rate'))}</td><td>${chip(m(c, 'over_abstention_rate'))}</td><td>${e ? `<span class="chip b${e.band_level}">${esc(e.band_label)}</span> <span class="n">${f2(e.f1)}</span>` : 'n/a'}</td></tr>`; }).join('');

$('eff').innerHTML = '<tr><th>Configuration</th><th>Latency per turn</th><th>Cost per turn</th><th>Tokens in / out per turn</th><th>Tool calls per turn</th><th>Errored runs</th></tr>' +
  live.map(c => `<tr><td><span class="dot" style="background:${color(c)}"></span>${esc(key(c))}</td><td>${c.latency_mean_s == null ? 'n/a' : c.latency_mean_s.toFixed(1) + ' s'} ${c.latency && !c.latency.error ? chip({...c.latency, rate: null}) : ''}</td><td>$${(c.cost_mean_usd ?? 0).toFixed(4)}</td><td class="n">${Math.round(c.tokens_in_mean ?? 0)} / ${Math.round(c.tokens_out_mean ?? 0)}</td><td class="n">${f2(c.tool_calls_mean)}</td><td class="n">${c.n_errors} of ${c.n_runs}</td></tr>`).join('');

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


def build_html(results: dict) -> str:
    themed = apply_theme(TEMPLATE, line1="AIXPERT WP3 evaluation framework", line2="DiagFlowBench demo",
                         footer="AIXPERT WP3 evaluation framework: DiagFlowBench demo")
    return themed.replace("__DATA__", json.dumps(results).replace("</", "<\\/"))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results_diagflow.json")
    ap.add_argument("--out", default="dashboard_diagflow.html")
    args = ap.parse_args(argv)
    Path(args.out).write_text(build_html(json.loads(Path(args.results).read_text(encoding="utf-8"))), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

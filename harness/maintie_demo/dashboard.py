"""Self-contained HTML dashboard from results.json (no network, no libraries)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from harness.dashboard_theme import apply_theme

TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AIXPERT WP3 live demo</title>
<style>
__THEME_CSS__
</style></head><body><main>
__HEADER__

<h2>Task quality</h2><p class="sub">How well each agent annotated the work orders, scored against MaintIE gold labels.</p>
<div class="panel" id="quality"></div>

<h2>Trustworthiness metrics (WP3 framework)</h2>
<p class="sub">Computed by the existing WP3 metrics. Band chips follow the framework banding (4 Excellent to 0 Unacceptable).</p>
<div class="panel"><table id="trust"></table></div>

<h2>Efficiency</h2><p class="sub">Per query, averaged over all runs. Cost uses the price table given at evaluation time (free models cost 0).</p>
<div class="panel"><table id="eff"></table></div>

<h2>Work-order drill-down</h2>
<p class="sub">Pick a configuration to see, per work order, the gold annotation and what the agent returned in each run.</p>
<div class="panel"><select id="cfg"></select><div id="items"></div></div>
<p class="notes" id="notes"></p>
__FOOTER__
</main>
<script id="data" type="application/json">__DATA__</script>
<script>
const R = JSON.parse(document.getElementById('data').textContent);
const C = R.configs, $ = id => document.getElementById(id);
const key = c => c.model + ' | ' + c.pattern, color = c => c.pattern === 'react' ? 'var(--react)' : 'var(--rag)';
const pct = x => x == null ? 'n/a' : (100 * x).toFixed(0) + '%', f2 = x => x == null ? 'n/a' : x.toFixed(2);
const esc = s => String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const chip = r => !r ? 'n/a' : r.error ? '<span class="chip b0">error</span>' :
  `<span class="chip b${r.band_level ?? (r.score >= 1 ? 4 : 0)}">${r.band_label ?? (r.score >= 1 ? 'all conform' : 'violations')}</span> <span class="n">${f2(r.score)}</span>`;
$('sub').textContent = `${R.meta.dataset}. ${C.length} configuration(s), generated ${R.meta.generated_at.slice(0, 16).replace('T', ' ')} UTC.`;

const bars = [['task_success_rate', 'Task success (all gold entities found, right class)', c => c.task_success_rate],
  ['subgoal_progress', 'Sub-goal progress (share of gold entities and relations found)', c => c.subgoal_progress],
  ['entity_f1', 'Entity F1', c => c.entity.f1], ['relation_f1', 'Relation F1', c => c.relation.f1]];
$('quality').innerHTML = '<div class="legend"><span class="dot" style="background:var(--react)"></span>ReAct <span class="dot" style="background:var(--rag);margin-left:12px"></span>Basic RAG</div>' +
  bars.map(([, title, fn]) => `<div class="grp">${title}</div>` + C.map(c => {
    const v = fn(c) ?? 0;
    return `<div class="row"><div class="lbl" title="${esc(key(c))}">${esc(key(c))}</div><div class="track"><div class="fill" style="width:${100 * v}%;background:${color(c)}"></div></div><div class="n">${pct(v)}</div></div>`;
  }).join('')).join('');

$('trust').innerHTML = '<tr><th>Configuration</th><th>Outcome reliability (pass^k)</th><th>Repeatability (cross-run)</th><th>Schema conformance</th><th>Log completeness</th><th>Step efficiency</th></tr>' +
  C.map(c => `<tr><td><span class="dot" style="background:${color(c)}"></span>${esc(key(c))}</td><td>${chip(c.pass_hat_k)}</td><td>${chip(c.cross_run_consistency)}</td><td>${chip(c.schema_conformance)}</td><td class="n">${f2(c.log_completeness)}</td><td class="n">${f2(c.step_efficiency)}</td></tr>`).join('');

$('eff').innerHTML = '<tr><th>Configuration</th><th>Latency (mean)</th><th>Cost per query</th><th>Tokens in / out</th><th>Tool calls</th><th>Errored runs</th></tr>' +
  C.map(c => `<tr><td><span class="dot" style="background:${color(c)}"></span>${esc(key(c))}</td><td>${c.latency_mean_s == null ? 'n/a' : c.latency_mean_s.toFixed(1) + ' s'} ${c.latency && !c.latency.error ? chip(c.latency) : ''}</td><td>$${(c.cost_mean_usd ?? 0).toFixed(4)} ${c.cost && !c.cost.error ? chip(c.cost) : ''}</td><td class="n">${Math.round(c.tokens_in_mean ?? 0)} / ${Math.round(c.tokens_out_mean ?? 0)}</td><td class="n">${f2(c.tool_calls_mean)}</td><td class="n">${c.n_errors} of ${c.n_runs}</td></tr>`).join('');

C.forEach((c, i) => $('cfg').insertAdjacentHTML('beforeend', `<option value="${i}">${esc(key(c))}</option>`));
const pill = e => `<span class="mono">${esc(e.text)} [${esc(e.type)}]</span>`;
const rel = r => `<span class="mono">${esc(r.head)} -${esc(r.type)}-> ${esc(r.tail)}</span>`;
function items() {
  const c = C[+$('cfg').value];
  $('items').innerHTML = c.items.map(it => {
    const ok = it.runs.map(r => `<span class="${r.task_success >= 1 ? 'ok' : 'no'}">${r.task_success >= 1 ? '&#10003;' : '&#10007;'}</span>`).join(' ');
    const runs = it.runs.map(r => `<p><b>run ${r.run}</b> ${r.error ? '<span class="no">error: ' + esc(r.error) + '</span>' :
      r.predicted ? `entities: ${r.predicted.entities.map(pill).join(', ') || 'none'}<br>relations: ${r.predicted.relations.map(rel).join(', ') || 'none'}${r.predicted.n_invalid ? `<br><span class="no">${r.predicted.n_invalid} out-of-scheme item(s) dropped</span>` : ''}` : '<span class="no">output did not match the schema</span>'}</p>`).join('');
    return `<details><summary>${ok} &nbsp; ${esc(it.text)}</summary><p><b>gold</b> entities: ${it.gold.entities.map(pill).join(', ')}<br>relations: ${it.gold.relations.map(rel).join(', ') || 'none'}</p>${runs}</details>`;
  }).join('');
}
$('cfg').onchange = items; items();
$('notes').innerHTML = R.meta.notes.map(esc).join('<br>');
</script></body></html>
"""


def build_html(results: dict) -> str:
    payload = json.dumps(results).replace("</", "<\\/")
    themed = apply_theme(TEMPLATE, line1="AIXPERT WP3 evaluation framework", line2="Live demo (MaintIE)",
                         footer="AIXPERT WP3 evaluation framework: live demo")
    return themed.replace("__DATA__", payload)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results.json")
    ap.add_argument("--out", default="dashboard.html")
    args = ap.parse_args(argv)
    Path(args.out).write_text(build_html(json.loads(Path(args.results).read_text(encoding="utf-8"))), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

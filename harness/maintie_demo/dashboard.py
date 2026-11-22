"""Self-contained HTML dashboard from results.json (no network, no libraries)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AIXPERT WP3 live demo</title>
<style>
:root{--bg:#fafaf8;--panel:#fff;--ink:#1c1c1a;--mute:#6b6b66;--line:#e4e3dd;--react:#2d6cdf;--rag:#d9822b;--good:#2f855a;--mid:#b7791f;--bad:#c53030}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#141413;--panel:#1d1d1b;--ink:#ecebe6;--mute:#9a9a93;--line:#2e2e2b;--react:#6ea0ff;--rag:#f0a255;--good:#5fc08a;--mid:#e0b04f;--bad:#f07a7a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:32px 0 4px}p.sub{margin:0 0 12px;color:var(--mute);font-size:13px}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 16px;overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);vertical-align:middle}
th{color:var(--mute);font-weight:600}td.n{font-variant-numeric:tabular-nums;white-space:nowrap}
.chip{display:inline-block;padding:1px 8px;border-radius:99px;font-size:12px;border:1px solid var(--line)}
.b4,.b3{color:var(--good);border-color:var(--good)}.b2{color:var(--mid);border-color:var(--mid)}.b1,.b0{color:var(--bad);border-color:var(--bad)}
.row{display:grid;grid-template-columns:minmax(150px,230px) 1fr 52px;gap:8px;align-items:center;margin:3px 0;font-size:13px}
.track{background:var(--line);height:10px;border-radius:5px;overflow:hidden}.fill{height:100%;border-radius:5px}
.lbl{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px}
.grp{margin:14px 0 4px;font-weight:600;font-size:13px}.legend{font-size:12px;color:var(--mute);margin-bottom:6px}
select{font:inherit;padding:4px 6px;background:var(--panel);color:var(--ink);border:1px solid var(--line);border-radius:6px}
.ok{color:var(--good)}.no{color:var(--bad)}.mono{font-family:ui-monospace,monospace;font-size:12px}
details{border-bottom:1px solid var(--line);padding:6px 0}summary{cursor:pointer}
.notes{font-size:12px;color:var(--mute);margin-top:8px}
</style></head><body><main>
<h1>AIXPERT WP3 evaluation framework: live demo</h1>
<p class="sub" id="sub"></p>

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
    return TEMPLATE.replace("__DATA__", payload)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results.json")
    ap.add_argument("--out", default="dashboard.html")
    args = ap.parse_args(argv)
    Path(args.out).write_text(build_html(json.loads(Path(args.results).read_text(encoding="utf-8"))), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

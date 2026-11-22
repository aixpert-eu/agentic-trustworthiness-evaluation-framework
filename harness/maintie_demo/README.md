# MaintIE live demo

Runs real WP4 agents (ReAct, Basic RAG) on MaintIE work orders, scores the runs with the WP3 metrics, and renders a self-contained HTML dashboard.

Dataset: [nlp-tlp/maintie](https://github.com/nlp-tlp/maintie) (MIT), clone it anywhere and pass the path as `--maintie`. Task: extract entities (5 top-level classes) and relations (6 types) from a work order.

## Pipeline

```
capture  ->  runs/<model>/<pattern>/<item>__r<k>.json   (live, costs money, resumable)
evaluate ->  results.json                                (offline, re-runnable, no model calls)
dashboard -> dashboard.html                              (self-contained)
```

Capture and evaluation are separate on purpose: live runs are written once with provenance (WP4 commit, model, temperature, token usage, wall clock, errors) and every metric is computed from the saved files.

## Run it

```bash
export OPENROUTER_API_KEY=...        # environment only, never in a file
M=/path/to/maintie
for P in react basic_rag; do
  python -m harness.maintie_demo.capture --maintie $M --pattern $P --model google/gemma-4-31b-it:free \
      --n 50 --runs 3 --wp4-commit $(git -C /path/to/aixpert-inobo rev-parse --short HEAD)
done
python -m harness.maintie_demo.evaluate --maintie $M --prices prices.json
python -m harness.maintie_demo.dashboard
```

Repeat capture for each model to compare (free and paid). `prices.json` maps a model to `[usd_per_1k_input, usd_per_1k_output]`; models missing from it cost 0. Add `--dry-run` to capture to exercise the whole pipeline with a fake agent and no API calls. Rate-limited or failed calls are retried, then recorded as errored runs (counted as failures); `--retry-errors` re-runs them.

## The WP4 side (local file)

`harness/maintie_demo/wp4_agents.py` builds the two patterns and imports WP4's private `inobo` package, so it is gitignored and never committed. Keep a local copy next to `capture.py`. It needs WP4 main plus `langchain[mcp]>=1.4.0`.

## What is measured

| Output | Source |
|---|---|
| task_success (every gold entity found with the right class), subgoal_progress (share of gold entities and relations found), log_completeness, step_efficiency | existing single-trajectory metrics via `harness.evaluate_trajectory` |
| pass^k (outcome reliability), cross_run_consistency (repeatability on per-run entity F1) | existing aggregate metrics, over runs per work order |
| schema_violation_rate | existing metric; a run conforms when its output parses and every label is in the scheme |
| latency, cost_per_query | existing efficiency metrics on mean wall clock and token cost |
| entity / relation precision, recall, F1 | demo scorer (micro-averaged), task quality rather than a trustworthiness metric |

Matching is on normalised text, not character offsets: entity = (text, top-level class), relation = (head text, tail text, type).

## Demo choices to know about

- `step_efficiency` needs a labelled optimal step count. Here ReAct = one lookup per gold entity, Basic RAG = 1. This is a demo choice, change it in `evaluate.optimal_steps`.
- The retriever and the ontology lookup are built from the 7,000 silver work orders and never contain the evaluation items.
- Basic RAG has no structured output, so its final answer is the text of the last AI message, parsed as JSON; the answer prompt is overridden to demand JSON only.
- Work orders contain anonymisation tokens such as `<num>`; they are left as they appear in the data.
- Models: pin one model per capture so a configuration is comparable across runs.

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


### Reading the dashboard

The page opens with the general trustworthiness metrics grouped by framework dimension (Part 1), then the task-specific results (Part 2), then details (Part 3). A tile per dimension says how many registered metrics this run measured; a dimension without a measured metric is shown as a gap. Metrics the demo computes but that have no value in the run (for example repeatability with one run per item) are listed under the table with what they need.

Under "Metrics shown" you can untick metrics, switch on a weakest-metric band per dimension (only metrics that carry a framework band count), or load a selection file made with the catalogue picker. Metrics in the selection that the run did not measure are listed, not hidden. To open the page with a selection already applied:

```bash
python -m harness.maintie_demo.dashboard --metrics uc3_selection.json
```

The selection file is JSON written by the catalogue picker (`{"label": ..., "metrics": [ids]}`); a plain list of ids or a text file of ids also works.

**How far can a number move.** Bars carry a whisker and every band chip a tooltip with the 95% interval: a percentile bootstrap that resamples the work order (never the run, because repeated runs of one work order are not independent), 1000 draws, fixed seed. With 10 work orders read it as an indication, not an exact coverage. pass^k and pass@k are computed with the same k for every configuration (default: the fewest runs any work order has, at most 5; `--k` to choose, an error if it exceeds the runs captured). pass@k is never below pass^k; for a deterministic agent they are equal. Log completeness is not scored for a run that made no tool calls (nothing to log is not a pass). Discriminative power (does the score separate the configurations at all) describes the comparison, not one configuration, so it has its own panel.

Repeat capture for each model to compare (free and paid). `prices.json` maps a model to `[usd_per_1k_input, usd_per_1k_output]`; models missing from it cost 0. Add `--dry-run` to capture to exercise the whole pipeline with a fake agent and no API calls. Rate-limited or failed calls are retried, then recorded as errored runs (counted as failures); `--retry-errors` re-runs them.

### Long unattended runs

- **Authentication or credit errors stop capture at once** (exit code 2) and write nothing for that run, so a bad key in a batch job cannot fill the output with error files; fix the key and re-run, finished runs are kept.
- `--timeout SECONDS` caps one attempt. A slower run is recorded as an error (`RunTimeout`) and not retried.
- `--min-free-requests N` stops cleanly (exit code 3) when fewer than N free-model requests remain today (read from OpenRouter `/api/v1/key`, checked every 10 runs, `:free` models only). Re-run later; capture resumes.

### Comparing like with like

Capture can be staged (for example `--n 10 --runs 1`, later `--n 10 --runs 3`, later `--n 20`). The first N work orders for a given `--seed` are a prefix of the first M for M > N, so deeper captures only add missing files. To score every model on identical work orders and runs, filter at evaluation time:

```bash
python -m harness.maintie_demo.evaluate --maintie $M --n 10 --max-runs 3 --common
# --models ernie,lightning   case-insensitive substrings of model ids
# --patterns basic_rag       exact pattern names
```

`--common` keeps only the (work order, run) pairs that every configuration has. `--min-runs M` keeps only the work orders that have runs 0 to M-1 in every configuration, which is what you want after an extension that has only reached some work orders (otherwise the work order with the fewest runs sets k for everyone). `--k N` and `--bootstrap N` set the k of pass^k / pass@k and the number of bootstrap draws. For adding work orders and runs on the cluster in resumable rounds, with a request budget, see `harness/slurm/README.md`. Without a filter, evaluate warns when configurations differ in work orders or runs, and the dashboard notes carry the warning or the subset used.

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

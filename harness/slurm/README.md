# Extra runs for the two demos (SLURM)

Two job scripts that add runs to the captures you already have, so the numbers on the dashboards stand on more data:

- `diagflow_extra_runs.sbatch`: DiagFlowBench, one model, three patterns (baseline, Basic RAG, ReAct).
- `maintie_extra_runs.sbatch`: MaintIE, five models, two patterns (Basic RAG, ReAct).

Both are resumable. Runs already on disk are skipped, a run cut short by the walltime is redone, and the job stops cleanly when only `RESERVE` free requests are left for the day. Submit the same command again tomorrow and it continues.

## What buys what

The unit the statistics resample is the work order (MaintIE) or the conversation (DiagFlowBench), not the run. So:

- **More work orders or conversations** narrow the 95% intervals. For a rate near 50% the worst-case half-width is about 0.31 with 10 items, 0.22 with 20 and 0.18 with 30.
- **More runs of the same item** make pass^k, pass@k and repeatability meaningful. They do not narrow the intervals much. DiagFlowBench has one run per conversation today, so repeatability and pass^k beyond k = 1 are not measured there yet.
- The runs captured so far use temperature 0, so repeats of the same item mostly give the same answer. Extra runs then say how stable the serving is, not how stable the model is under sampling. If you want the second kind of repeatability, capture a separate sampled arm (`TEMPERATURE=0.7 RUNS_DIR=runs_t07 ...`) and evaluate it on its own. Do not mix temperatures in one directory.

## Rounds

The work is done in rounds so that a day's quota never leaves one configuration far ahead of the others. A round is `pairs:runs` (DiagFlowBench) or `n:runs` (MaintIE) and means: after this round every configuration has `runs` runs of the first `pairs` pairs / `n` work orders. The selection is deterministic and the first `n` are a prefix of the first `m > n`, so the items you already have are kept.

```
# DiagFlowBench: 5 -> 15 pairs (10 -> 30 conversations), one run each   (default)
ROUNDS="10:1 15:1" sbatch harness/slurm/diagflow_extra_runs.sbatch
# DiagFlowBench: runs 2 and 3 of the 5 pairs you have
ROUNDS="5:2 5:3"   sbatch harness/slurm/diagflow_extra_runs.sbatch
# MaintIE: 10 -> 30 work orders, three runs each                       (default)
ROUNDS="15:3 20:3 25:3 30:3" sbatch harness/slurm/maintie_extra_runs.sbatch
```

Other settings are environment variables too (`MODEL`, `MODELS`, `PATTERNS`, `RESERVE`, `TIMEOUT`, `RUNS_DIR`, `OUT_DIR`, `EVALUATE`, `JUDGE`, `PRICES`, `TEMPERATURE`; see the top of each script). Set `REPO`, `DIAGFLOW` or `MAINTIE` if your paths differ from `/scratch/$USER/...`. Two things to check once: the environment lines near the top (copy the ones from your existing job script) and that `OPENROUTER_API_KEY` is exported the way your existing job does it.

Rehearse without any API call and without touching your real runs: `DRY_RUN=1 bash harness/slurm/diagflow_extra_runs.sbatch`. It uses the built-in fake agent and a separate `*_dryrun` directory.

## What it costs (estimates)

Free-model quota is 1000 requests per day, of which about 800 to 900 are usable with the default reserve (days of quota below use 800). The figures come from the first captures and are rough (plus or minus 40%). The job prints the free requests used by each block, so the first block tells you the real number.

| Round | Requests | Compute | Days of quota |
|---|---|---|---|
| DiagFlowBench, +5 pairs (10 conversations), 1 run: baseline | about 140 | 5 min | |
| same, Basic RAG | about 280 | 13 min | |
| same, ReAct | about 600 | 20 min | |
| **DiagFlowBench, one such round, all patterns** | **about 1000** | 40 min | 1.3 |
| DiagFlowBench, 5 -> 15 pairs (two rounds) | about 2000 | 80 min | 2.5 |
| DiagFlowBench, runs 2 and 3 of the 5 pairs (two rounds) | about 2000 | 80 min | 2.5 |
| **MaintIE, +5 work orders x 3 runs, all 10 configurations** | **about 700** | 70 min | 0.9 |
| MaintIE, 10 -> 20 work orders (two rounds) | about 1400 | 2.3 h | 1.8 |
| MaintIE, 10 -> 30 work orders (four rounds) | about 2800 | 4.5 h | 3.5 |

The judge (Claude Haiku 4.5 through OpenRouter) is paid and not part of the free quota. It labels the injection turns of the new conversations; every label is cached in `judge_cache_demo.json`, so it is paid once. `JUDGE=cache` evaluates with cached labels only.

Keep the live demo in mind: the default `RESERVE` leaves 200 (DiagFlowBench) or 100 (MaintIE) requests untouched each day, and a live run stays at one run per item.

### Suggested order

1. DiagFlowBench `ROUNDS="10:1"` (about 1000 requests): 20 conversations per pattern.
2. MaintIE `ROUNDS="15:3 20:3"` (about 1400): 20 work orders, three runs each.
3. DiagFlowBench `ROUNDS="15:1"` (about 1000): 30 conversations.

Steps 1 to 3 are about 3400 requests, roughly four and a half days of quota. Then, only if there is time: MaintIE `25:3 30:3`; DiagFlowBench repeats (`5:2 5:3`).

Run `diagflow_extra_runs.sbatch` and `maintie_extra_runs.sbatch` one after the other, not at the same time: they share one daily quota and one rate limit.

## Reading the result

Each job ends by evaluating with `--common --min-runs R`, where R is the runs value of the last round that finished for every configuration, and by rebuilding the dashboard (`results_<dataset>_extra_<last round>.json`, `dashboard_<dataset>_extra_<last round>.html`). That keeps only the items and runs every configuration has, so an unfinished round cannot skew the comparison. The note under the matrix on the dashboard says which k was used for pass^k and pass@k.

To score differently by hand, for example the 10 items that have 3 runs while 30 have one:

```
python -m harness.maintie_demo.evaluate --maintie <clone> --runs runs --common --min-runs 3 --out results_3runs.json
```

`--k` fixes k (it must not exceed the fewest runs any item has) and `--bootstrap N` changes the number of resamples (default 1000, fixed seed, so the same data always gives the same intervals).

## Metrics

New in the evaluation stage, computed from runs you already have once you evaluate again: pass@k, 95% intervals (bootstrap over items), tool-call log completeness for DiagFlowBench (shown as "no tool calls" for baseline, which makes none), and discriminative power across the configurations (a property of the comparison, shown in its own panel).

Not computable from these runs, however many you add: unsafe action rate and prompt injection resistance (they need labelled case sets and governance events). The dashboards list them with that reason.

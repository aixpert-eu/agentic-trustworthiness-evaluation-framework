# DiagFlowBench demo (UC3-style diagnostic dialogue)

Evaluates procedure-grounded diagnostic assistants with the WP3 metrics, using WP4's ReAct and Basic RAG patterns on [DiagFlowBench](https://github.com/aixpert-rug/DiagFlowBench) (Gil de Avalle et al., EMNLP 2026 Industry Track; code MIT, data CC-BY-4.0). It mirrors `harness/maintie_demo`: capture (live, resumable), evaluate (offline), dashboard.

## The task being evaluated

A shopfloor operator reports observations while working through a documented troubleshooting procedure, stored as a flowchart (nodes: terminator, process or decision; edges labelled Yes or No). At every turn the assistant gets the procedure, the conversation so far and the operator's latest message, and must:

- **on-procedure input:** suggest exactly one next action that is a step of the procedure (1 to 3 sentences), following the branch the operator's answer selects, and say so when the procedure has ended;
- **off-procedure input** (a symptom the procedure has no branch for, a fault in an undocumented component, or an unrelated question): not map it to a step. It should recognise the gap and redirect, ask for clarification or escalate.

Off-procedure turns are scored as correct abstention, forced mapping (a real step the input does not license) or fabrication (a step that is not in the procedure). The dataset has 50 flowcharts and 838 matched pairs of conversations: a clean one, and a mixed one with off-procedure turns injected.

## Run it

```bash
export OPENROUTER_API_KEY=...        # environment only, never in a file
R=/path/to/DiagFlowBench             # cloned repository
for P in baseline react basic_rag; do
  python -m harness.diagflow_demo.capture --diagflow $R --pattern $P --model baidu/ernie-5.5-flash-preview:free \
      --pairs 5 --runs 1 --timeout 600 --min-free-requests 100 --wp4-commit $(git -C /path/to/aixpert-inobo rev-parse --short HEAD)
done
python -m harness.diagflow_demo.evaluate --diagflow $R --reference
python -m harness.diagflow_demo.dashboard
```


### Reading the dashboard

The page opens with the general trustworthiness metrics grouped by framework dimension (Part 1), then the task-specific results (Part 2), then details (Part 3). A tile per dimension says how many registered metrics this run measured; a dimension without a measured metric is shown as a gap. Metrics the demo computes but that have no value in the run (for example repeatability with one run per item) are listed under the table with what they need.

Under "Metrics shown" you can untick metrics, switch on a weakest-metric band per dimension (only metrics that carry a framework band count), or load a selection file made with the catalogue picker. Metrics in the selection that the run did not measure are listed, not hidden. To open the page with a selection already applied:

```bash
python -m harness.diagflow_demo.dashboard --metrics uc3_selection.json
```

The selection file is JSON written by the catalogue picker (`{"label": ..., "metrics": [ids]}`); a plain list of ids or a text file of ids also works.

`--pattern baseline` is the benchmark's own setup (whole graph as JSON in the system prompt, no WP4), so patterns can be compared with it and with the published models. `--dry-run` exercises everything with a fake agent. Capture options (resume, `--retry-errors`, `--timeout`, `--min-free-requests`, authentication abort) work as in the MaintIE demo.

**Request budget.** A conversation has about 13 (clean) to 15 (mixed) operator turns, so one run costs roughly 14 requests for `baseline`, 28 for Basic RAG and 60 or more for ReAct. `--pairs 5` is 10 conversations. `--max-turns` (default 14) skips the longest conversations.

**More conversations and runs.** The first N pairs for a seed are a prefix of the first M for M > N, so `--pairs 15` after `--pairs 5` only adds the missing conversations, and `--runs 3` only adds runs 1 and 2. `harness/slurm/` has resumable job scripts that do this in rounds with a daily request guard, and a README with the budget. At evaluation, `--common --min-runs M` keeps the conversations and runs every pattern has (use it after an extension that has only reached part of the way), `--k N` sets the k of pass^k and pass@k for every configuration (default: the fewest runs any conversation has, at most 5) and `--bootstrap N` the number of draws of the 95% intervals, which resample the conversation (never the run) with a fixed seed. One run per conversation means k = 1: repeatability is then not measured, and the dashboard says so.

## Scoring

Metrics live in `metrics/diagnostic_dialogue/` and are registered in `contracts/registry.py`: `step_accuracy`, `branch_following`, `termination_recognition` (Jaccard node matching, threshold 0.05, as DiagFlowBench), and `correct_abstention_rate`, `forced_mapping_rate`, `fabrication_rate`, `over_abstention_rate` (labels from the DiagFlowBench judge, Claude Haiku 4.5 via OpenRouter, temperature 0, same prompts). Existing metrics are reused for recovery (`error_recovery_rate`), abstention decisions (`escalation_appropriateness`), reliability (`pass_hat_k`, `cross_run_consistency`) and efficiency.

The judge needs `JUDGE_API_KEY` or `OPENROUTER_API_KEY` at evaluation time. It is a paid model; for this sample the cost is a few cents. Labels are cached in `judge_cache_demo.json` (same key format as the original cache), so re-evaluating costs nothing. Without a key, evaluate serves labels from the cache only and warns about turns with none.

### Parity with the paper

```bash
python -m harness.diagflow_demo.parity --diagflow $R
```

Re-scores the released per-model results and judge cache with this code and compares with the released `analysis_report.json`. All values match except termination recognition (see below). This takes seconds and needs no API.

### Demo choices and known differences

- **Step accuracy compares with the next step.** The benchmark's code compares the reply with the step of the turn just answered, while a correct reply names the next step. On the released Llama 3.3 70B results that gives 4.6% exact against 59.8% for the next step. `parity.py` uses the benchmark's way (`expected="gold"`) to reproduce its numbers; the demo uses the next step.
- **Termination recognition.** The released report and the released scorer disagree (Llama 3.3 70B: 83.4% in the report, 0.8% from the code), so it cannot be reproduced. The code's rule also fails a reply that quotes the terminal node ("the procedure has ended at 'End, Notify maintenance team'"). The demo accepts a reply that resolves to a terminal node (`accept_terminal_node=True`); only a reply that proposes a non-terminal step fails.
- **History** is the assistant's own earlier replies, as in the benchmark; errors can compound across turns.
- **Turn ok**: right step (on-procedure) or abstention (off-procedure). pass^k is over turns, repeatability over each conversation's per-run share of ok turns. Latency and cost budgets are per turn and are demo defaults.

## The WP4 side

`harness/diagflow_demo/diagflow_agents.py` builds the two patterns on one procedure graph (ReAct with step lookup tools, Basic RAG over the steps) and imports WP4's `inobo` package (currently private; `uv sync --extra inobo`, see the top-level README for the SSH access it needs). `capture` imports it only for the `react` and `basic_rag` patterns without `--dry-run`, so the dry run, the `baseline` pattern, `evaluate` and `dashboard` work without `inobo`. Record the WP4 commit you ran with `--wp4-commit`.

#!/bin/bash
#SBATCH --job-name=diagflow_extra
#SBATCH --partition=regularshort
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=%x_%j.out
#
# More DiagFlowBench runs for tighter statistics: more conversations, more runs per conversation, or both.
# Read harness/slurm/README.md first (what it costs, in which order to run it, how to read the result).
#
# The capture is resumable. Runs already on disk are skipped, a run cut short by the walltime is redone,
# and the job stops cleanly when only RESERVE free requests are left today. Submit it again tomorrow with
# the same settings and it carries on where it stopped.
#
# The work is done in ROUNDS. A round is "pairs:runs" and means: after this round every pattern has
# `runs` runs of the first `pairs` matched clean/mixed pairs. Rounds finish for all patterns before the
# next one starts, so a day's quota never leaves one pattern far ahead of the others.
#
#   ROUNDS="10:1 15:1"   5 -> 15 pairs, one run each (more conversations; the default)
#   ROUNDS="5:2 5:3"     runs 2 and 3 of the 5 pairs captured so far (repeatability)
#
# Settings are environment variables, set when submitting:   ROUNDS="5:2 5:3" sbatch harness/slurm/diagflow_extra_runs.sbatch
# Rehearse the whole thing without any API call or any real file touched:   DRY_RUN=1 bash harness/slurm/diagflow_extra_runs.sbatch

set -uo pipefail

REPO=${REPO:-/scratch/$USER/ageval}                  # checkout of this framework
DIAGFLOW=${DIAGFLOW:-/scratch/$USER/DiagFlowBench}   # clone of the DiagFlowBench repository
MODEL=${MODEL:-baidu/ernie-5.5-flash-preview:free}
PATTERNS=${PATTERNS:-"baseline basic_rag react"}     # cheapest first: about 14, 28 and 60 requests per conversation-run
ROUNDS=${ROUNDS:-"10:1 15:1"}
RESERVE=${RESERVE:-200}        # free requests left untouched today (rehearsal, live demo); one conversation-run can use ~100
TEMPERATURE=${TEMPERATURE:-0}  # keep 0 to match the runs already captured; a sampled arm needs its own RUNS_DIR
TIMEOUT=${TIMEOUT:-300}        # seconds allowed per agent call
STOP_AFTER_MIN=${STOP_AFTER_MIN:-150}   # no new capture block after this many minutes (the job has 180)
OUT_DIR=${OUT_DIR:-.}          # where results and dashboards are written (relative to REPO)
RUNS_DIR=${RUNS_DIR:-runs_diagflow}     # relative to REPO; must be the directory of the runs captured so far
WP4_COMMIT=${WP4_COMMIT:-52aa9ed}       # recorded in each run file; the value used for the runs already captured
EVALUATE=${EVALUATE:-1}        # 1: evaluate and rebuild the dashboard at the end
JUDGE=${JUDGE:-auto}           # auto: label new turns with Claude Haiku through OpenRouter (paid, cached); cache: only labels already cached
JUDGE_CACHE=${JUDGE_CACHE:-judge_cache_demo.json}
PRICES=${PRICES:-}             # optional prices.json
DRY_RUN=${DRY_RUN:-0}

# --- environment: use the lines from your existing diagflow_patterns job script here ----------------------
SKIP_ENV_SETUP=${SKIP_ENV_SETUP:-0}   # 1: skip the lines below (the environment is already active in this shell)
if [ "$SKIP_ENV_SETUP" != 1 ] && command -v module >/dev/null 2>&1; then
  module load Miniconda3
  source activate aixpert-unified
fi
# -------------------------------------------------------------------------------------------------------

cd "$REPO" || { echo "REPO $REPO not found"; exit 2; }
START=$(date +%s)
TAG="${ROUNDS##* }"; TAG="${TAG/:/p_}r"; TAG="extra_${TAG}"   # e.g. extra_15p_1r for the last round "15:1"
DRY_FLAG=""
if [ "$DRY_RUN" = 1 ]; then
  DRY_FLAG="--dry-run"; RUNS_DIR="${RUNS_DIR}_dryrun"; TAG="${TAG}_dryrun"; JUDGE=rule
elif [ -z "${OPENROUTER_API_KEY:-}" ]; then
  echo "OPENROUTER_API_KEY is not set. Export it the way your existing job script does."; exit 2
fi
[ -d "$DIAGFLOW" ] || { echo "DIAGFLOW $DIAGFLOW not found"; exit 2; }
python -c "import harness.diagflow_demo.capture" || { echo "cannot import the framework: check the environment lines above"; exit 2; }

quota() {
  [ "$DRY_RUN" = 1 ] && { echo "-"; return; }
  python - <<'PY'
from harness.maintie_demo.capture import free_requests_remaining
r = free_requests_remaining()
print("?" if r is None else r)
PY
}
elapsed_min() { echo $(( ($(date +%s) - START) / 60 )); }

echo "model $MODEL | rounds $ROUNDS | patterns $PATTERNS | runs in $RUNS_DIR | free requests now: $(quota)"
DONE_RUNS=1      # every conversation already on disk has at least one run
STATUS=complete
for ROUND in $ROUNDS; do
  PAIRS=${ROUND%%:*}; RUNS=${ROUND##*:}
  ROUND_OK=1
  for PATTERN in $PATTERNS; do
    if [ "$(elapsed_min)" -ge "$STOP_AFTER_MIN" ]; then STATUS="time (submit again to continue)"; ROUND_OK=0; break; fi
    echo "=== round ${PAIRS} pairs x ${RUNS} runs | $PATTERN | $(date +%H:%M) | free requests before: $(quota) ==="
    before=$(quota)
    python -m harness.diagflow_demo.capture --diagflow "$DIAGFLOW" --pattern "$PATTERN" --model "$MODEL" \
      --pairs "$PAIRS" --runs "$RUNS" --min-turns 4 --max-turns 14 --seed 0 --temperature "$TEMPERATURE" \
      --out "$RUNS_DIR" --wp4-commit "$WP4_COMMIT" --timeout "$TIMEOUT" --min-free-requests "$RESERVE" $DRY_FLAG
    rc=$?
    after=$(quota)
    if [[ "$before" =~ ^[0-9]+$ && "$after" =~ ^[0-9]+$ ]]; then echo "free requests used by this block: $((before - after)) (left: $after)"; fi
    if [ $rc -eq 3 ]; then STATUS="daily free-request quota (resume tomorrow)"; ROUND_OK=0; break; fi
    if [ $rc -ne 0 ]; then STATUS="capture failed with exit code $rc"; ROUND_OK=0; break; fi
  done
  [ $ROUND_OK -eq 1 ] && DONE_RUNS=$RUNS || break
done
echo "=== capture finished: $STATUS | $(elapsed_min) min | free requests left: $(quota) ==="

FAILED=0; [[ "$STATUS" == "capture failed"* ]] && FAILED=1
# a failed capture (bad key, no credit) is not evaluated: the judge calls are paid and the data would be incomplete
if [ "$EVALUATE" = 1 ] && [ $FAILED -eq 0 ]; then
  mkdir -p "$OUT_DIR"
  OUT="$OUT_DIR/results_diagflow_${TAG}.json"
  # --common and --min-runs keep only the conversations and runs every pattern has, so an unfinished round cannot skew the comparison
  echo "=== evaluate (--common --min-runs $DONE_RUNS) ==="
  python -m harness.diagflow_demo.evaluate --diagflow "$DIAGFLOW" --runs "$RUNS_DIR" --out "$OUT" \
    --judge "$JUDGE" --judge-cache "$JUDGE_CACHE" --reference --common --min-runs "$DONE_RUNS" ${PRICES:+--prices "$PRICES"} \
  && python -m harness.diagflow_demo.dashboard --results "$OUT" --out "$OUT_DIR/dashboard_diagflow_${TAG}.html" \
  && echo "wrote $OUT and $OUT_DIR/dashboard_diagflow_${TAG}.html"
fi
exit $FAILED

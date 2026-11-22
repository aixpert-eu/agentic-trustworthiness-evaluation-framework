#!/bin/bash
#SBATCH --job-name=maintie_extra
#SBATCH --partition=regularshort
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=%x_%j.out
#
# More MaintIE runs for tighter statistics: more work orders, more runs per work order, or both.
# Read harness/slurm/README.md first (what it costs, in which order to run it, how to read the result).
#
# The capture is resumable. Runs already on disk are skipped, a run cut short by the walltime is redone,
# and the job stops cleanly when only RESERVE free requests are left today. Submit it again tomorrow with
# the same settings and it carries on where it stopped.
#
# The work is done in ROUNDS. A round is "n:runs" and means: after this round every model and pattern has
# `runs` runs of the first `n` work orders (the same selection as the runs already captured, so the
# earlier ones are kept). Rounds finish for all configurations before the next one starts, so a day's quota
# never leaves one model far ahead of the others.
#
#   ROUNDS="15:3 20:3 25:3 30:3"   10 -> 30 work orders, three runs each (more work orders; the default)
#   ROUNDS="10:4 10:5"             runs 4 and 5 of the 10 work orders captured so far (repeatability)
#
# Settings are environment variables, set when submitting:   ROUNDS="15:3 20:3" sbatch harness/slurm/maintie_extra_runs.sbatch
# Rehearse the whole thing without any API call or any real file touched:   DRY_RUN=1 bash harness/slurm/maintie_extra_runs.sbatch

set -uo pipefail

REPO=${REPO:-/scratch/$USER/ageval}              # checkout of this framework
MAINTIE=${MAINTIE:-/scratch/$USER/maintie}       # clone of nlp-tlp/maintie
MODELS=${MODELS:-"baidu/ernie-5.5-flash-preview:free cohere/north-mini-code:free liquid/lfm-3-1.7b:free nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free nvidia/nemotron-3.5-lightning:free"}
PATTERNS=${PATTERNS:-"basic_rag react"}          # about 3 and 6 requests per work order and run
ROUNDS=${ROUNDS:-"15:3 20:3 25:3 30:3"}
RESERVE=${RESERVE:-100}        # free requests left untouched today (rehearsal, live demo)
TEMPERATURE=${TEMPERATURE:-0}  # keep 0 to match the runs already captured; a sampled arm needs its own RUNS_DIR
TIMEOUT=${TIMEOUT:-300}        # seconds allowed per agent call
STOP_AFTER_MIN=${STOP_AFTER_MIN:-150}   # no new capture block after this many minutes (the job has 180)
OUT_DIR=${OUT_DIR:-.}          # where results and dashboards are written (relative to REPO)
RUNS_DIR=${RUNS_DIR:-runs}     # relative to REPO; must be the directory of the runs captured so far
WP4_COMMIT=${WP4_COMMIT:-52aa9ed}       # recorded in each run file; the value used for the runs already captured
EVALUATE=${EVALUATE:-1}        # 1: evaluate and rebuild the dashboard at the end
PRICES=${PRICES:-}             # optional prices.json
DRY_RUN=${DRY_RUN:-0}

# --- environment: use the lines from your existing job script here ---------------------------------------
SKIP_ENV_SETUP=${SKIP_ENV_SETUP:-0}   # 1: skip the lines below (the environment is already active in this shell)
if [ "$SKIP_ENV_SETUP" != 1 ] && command -v module >/dev/null 2>&1; then
  module load Miniconda3
  source activate aixpert-unified
fi
# -------------------------------------------------------------------------------------------------------

cd "$REPO" || { echo "REPO $REPO not found"; exit 2; }
START=$(date +%s)
TAG="${ROUNDS##* }"; TAG="${TAG/:/n_}r"; TAG="extra_${TAG}"   # e.g. extra_30n_3r for the last round "30:3"
DRY_FLAG=""
if [ "$DRY_RUN" = 1 ]; then
  DRY_FLAG="--dry-run"; RUNS_DIR="${RUNS_DIR}_dryrun"; TAG="${TAG}_dryrun"
elif [ -z "${OPENROUTER_API_KEY:-}" ]; then
  echo "OPENROUTER_API_KEY is not set. Export it the way your existing job script does."; exit 2
fi
[ -d "$MAINTIE" ] || { echo "MAINTIE $MAINTIE not found"; exit 2; }
python -c "import harness.maintie_demo.capture" || { echo "cannot import the framework: check the environment lines above"; exit 2; }

quota() {
  [ "$DRY_RUN" = 1 ] && { echo "-"; return; }
  python - <<'PY'
from harness.maintie_demo.capture import free_requests_remaining
r = free_requests_remaining()
print("?" if r is None else r)
PY
}
elapsed_min() { echo $(( ($(date +%s) - START) / 60 )); }

echo "rounds $ROUNDS | patterns $PATTERNS | runs in $RUNS_DIR | free requests now: $(quota)"
DONE_RUNS=1      # every work order already on disk has at least one run
STATUS=complete
for ROUND in $ROUNDS; do
  N=${ROUND%%:*}; RUNS=${ROUND##*:}
  ROUND_OK=1
  for PATTERN in $PATTERNS; do
    for MODEL in $MODELS; do
      if [ "$(elapsed_min)" -ge "$STOP_AFTER_MIN" ]; then STATUS="time (submit again to continue)"; ROUND_OK=0; break 2; fi
      echo "=== round ${N} work orders x ${RUNS} runs | $MODEL | $PATTERN | $(date +%H:%M) ==="
      before=$(quota)
      python -m harness.maintie_demo.capture --maintie "$MAINTIE" --pattern "$PATTERN" --model "$MODEL" \
        --n "$N" --runs "$RUNS" --seed 0 --temperature "$TEMPERATURE" \
        --out "$RUNS_DIR" --wp4-commit "$WP4_COMMIT" --timeout "$TIMEOUT" --min-free-requests "$RESERVE" $DRY_FLAG
      rc=$?
      after=$(quota)
      if [[ "$before" =~ ^[0-9]+$ && "$after" =~ ^[0-9]+$ ]]; then echo "free requests used by this block: $((before - after)) (left: $after)"; fi
      if [ $rc -eq 3 ]; then STATUS="daily free-request quota (resume tomorrow)"; ROUND_OK=0; break 2; fi
      if [ $rc -ne 0 ]; then STATUS="capture failed with exit code $rc"; ROUND_OK=0; break 2; fi
    done
  done
  [ $ROUND_OK -eq 1 ] && DONE_RUNS=$RUNS || break
done
echo "=== capture finished: $STATUS | $(elapsed_min) min | free requests left: $(quota) ==="

FAILED=0; [[ "$STATUS" == "capture failed"* ]] && FAILED=1
# a failed capture (bad key, no credit) is not evaluated: the judge calls are paid and the data would be incomplete
if [ "$EVALUATE" = 1 ] && [ $FAILED -eq 0 ]; then
  mkdir -p "$OUT_DIR"
  OUT="$OUT_DIR/results_maintie_${TAG}.json"
  # --common and --min-runs keep only the work orders and runs every configuration has, so an unfinished round cannot skew the comparison
  echo "=== evaluate (--common --min-runs $DONE_RUNS) ==="
  python -m harness.maintie_demo.evaluate --maintie "$MAINTIE" --runs "$RUNS_DIR" --out "$OUT" \
    --common --min-runs "$DONE_RUNS" ${PRICES:+--prices "$PRICES"} \
  && python -m harness.maintie_demo.dashboard --results "$OUT" --out "$OUT_DIR/dashboard_maintie_${TAG}.html" \
  && echo "wrote $OUT and $OUT_DIR/dashboard_maintie_${TAG}.html"
fi
exit $FAILED

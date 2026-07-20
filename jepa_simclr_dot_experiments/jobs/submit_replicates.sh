#!/bin/bash
# Submit best-HP replicate runs.
# Reads configs/hp_best/index.csv to pick YAMLs, expands to (yaml × seed) pairs,
# then submits a SLURM array where each task runs ONE (yaml, seed) combo.
#
# Usage:
#   bash jobs/submit_replicates.sh [AFTER_JOB_ID] [SKIP_BATCHES]
#
# AFTER_JOB_ID  — optional: first batch waits for this job (via afterany).
# SKIP_BATCHES  — optional int: skip the first N batches (default 0).
#                 Use to resume after a submit-limit failure, e.g.:
#                   bash jobs/submit_replicates.sh 485146 1
#
# Edit the BATCHES array below to control which YAMLs and seeds to use.

SCRIPT="jobs/submit_array_replicates.sh"
INDEX_CSV="configs/hp_best/index.csv"

# ── Per-batch configuration ──────────────────────────────────────────────────
# Each batch specifies:
#   FILTER            pandas .query() expression (use '' for all)
#   SEEDS             space-separated list of integer seeds
#   ORDER             "seed_first" (one seed of every yaml, then next seed)
#                     or "yaml_first" (all seeds of yaml 0, then yaml 1, ...)
#   RUN_GROUP_PREFIX  label prefix for W&B run groups
BATCHES=(
    "FILTER=rank == 1 and variant == 'baseline'           SEEDS='42 1337 2024'  ORDER=seed_first  RUN_GROUP_PREFIX=best_hp_baseline"
    "FILTER=rank == 1 and variant == 'baseline_traj_neg'  SEEDS='42 1337 2024'  ORDER=seed_first  RUN_GROUP_PREFIX=best_hp_traj_neg"
)

# Set to your W&B project name.
WANDB_PROJECT_DEFAULT="YOUR_PROJECT_NAME"
MAX_CONCURRENT="${MAX_CONCURRENT:-5}"
# ─────────────────────────────────────────────────────────────────────────────

mkdir -p .replicate_pairs

PREV_JOB="${1:-}"
SKIP_BATCHES="${2:-0}"

for i in "${!BATCHES[@]}"; do
    if (( i < SKIP_BATCHES )); then
        echo "Skipping batch $i (< SKIP_BATCHES=$SKIP_BATCHES)"
        continue
    fi
    BATCH="${BATCHES[$i]}"
    FILTER=$(echo "$BATCH" | sed -nE "s/.*FILTER=([^[:space:]].*[^[:space:]]?)[[:space:]]+SEEDS=.*/\1/p")
    SEEDS_VAL=$(echo "$BATCH" | grep -oP "SEEDS='[^']*'" | sed "s/SEEDS='//" | sed "s/'$//")
    ORDER=$(echo "$BATCH" | grep -oP 'ORDER=\S+' | cut -d= -f2)
    RGP=$(echo "$BATCH" | grep -oP 'RUN_GROUP_PREFIX=\S+' | cut -d= -f2)

    ORDER="${ORDER:-seed_first}"
    PAIRS_FILE=".replicate_pairs/pairs_$(echo "$RGP" | tr '/' '_').tsv"

    FILTER_VAL="$FILTER" SEEDS_PY="$SEEDS_VAL" ORDER_PY="$ORDER" INDEX_CSV_PY="$INDEX_CSV" \
    python - <<'PY' > "$PAIRS_FILE"
import os, pandas as pd
df = pd.read_csv(os.environ["INDEX_CSV_PY"])
flt = os.environ["FILTER_VAL"].strip()
if flt:
    df = df.query(flt)
yamls = [os.path.abspath(y) for y in df["config_path"].tolist()]
seeds = os.environ["SEEDS_PY"].split()
order = os.environ["ORDER_PY"]
if order == "seed_first":
    pairs = [(y, s) for s in seeds for y in yamls]
elif order == "yaml_first":
    pairs = [(y, s) for y in yamls for s in seeds]
else:
    raise SystemExit(f"Unknown ORDER: {order}")
for y, s in pairs:
    print(f"{y}\t{s}")
PY

    N=$(wc -l < "$PAIRS_FILE")
    if (( N == 0 )); then
        echo "WARN: no pairs match filter: $FILTER  (skipping)"
        continue
    fi

    DEP=""
    [[ -n "$PREV_JOB" ]] && DEP="--dependency=afterany:${PREV_JOB}"

    JOB_ID=$(env \
        PAIRS_FILE="$(pwd)/$PAIRS_FILE" \
        WANDB_PROJECT="$WANDB_PROJECT_DEFAULT" \
        RUN_GROUP_PREFIX="$RGP" \
        sbatch --parsable --array=0-$((N-1))%${MAX_CONCURRENT} $DEP "$SCRIPT")

    if [[ -z "$JOB_ID" ]]; then
        echo "ERROR: sbatch returned no JOB_ID for batch $i (likely submit-limit hit)." >&2
        echo "       Resume with: bash $0 ${PREV_JOB:-} $i" >&2
        exit 1
    fi

    echo "Submitted $JOB_ID | ${N} (yaml,seed) pairs | order=${ORDER} | seeds=[$SEEDS_VAL] | dep=${PREV_JOB:-none}"
    PREV_JOB="$JOB_ID"
done

echo ""
echo "Chain submitted. Monitor with: squeue -r -u \$USER"

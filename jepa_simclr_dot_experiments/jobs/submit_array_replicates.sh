#!/bin/bash
#SBATCH --job-name=best_hp_replicates
#SBATCH --nodes=1
#SBATCH --partition=<YOUR_PARTITION>
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --gpus=1
#SBATCH --mem=24G
#SBATCH --array=0-0   # overridden by submit_replicates.sh

set -euo pipefail
shopt -s nullglob
PS4='+ [${BASH_SOURCE##*/}:${LINENO}] '
set -x

conda init bash > /dev/null 2>&1
set +u
source ~/.bashrc
set -u
conda activate "${CONDA_ENV:-YOUR_CONDA_ENV}"
trap 'rc=$?; echo "[ERROR] line ${LINENO}: ${BASH_COMMAND} (exit ${rc})" >&2; exit ${rc}' ERR

# ── Parameters — override via env from submit_replicates.sh ──────────────────
# PAIRS_FILE     : TAB-separated file, one line per (yaml_path, seed) combo (required)
# WANDB_PROJECT  : destination W&B project
PAIRS_FILE="${PAIRS_FILE:?PAIRS_FILE must be set}"
WANDB_PROJECT="${WANDB_PROJECT:-YOUR_PROJECT_NAME}"
RUN_GROUP_PREFIX="${RUN_GROUP_PREFIX:-best_hp_replicate}"
# ─────────────────────────────────────────────────────────────────────────────

export WANDB_PROJECT

TOTAL=$(wc -l < "$PAIRS_FILE")
if (( SLURM_ARRAY_TASK_ID >= TOTAL )); then
    echo "Array index $SLURM_ARRAY_TASK_ID out of range (total=$TOTAL)"
    exit 0
fi

LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$PAIRS_FILE")
YAML_PATH=$(echo "$LINE" | cut -f1)
SEED=$(echo "$LINE" | cut -f2)

if [[ ! -f "$YAML_PATH" ]]; then
    echo "YAML not found: $YAML_PATH" >&2
    exit 1
fi

YAML_NAME=$(basename "$YAML_PATH" .yaml)
VARIANT_DIR=$(basename "$(dirname "$YAML_PATH")")
RUN_GROUP="${RUN_GROUP_PREFIX}_${VARIANT_DIR}_${YAML_NAME}"
RUN_NAME="${YAML_NAME}_seed${SEED}"

echo "Host: $(hostname) | Task: $SLURM_ARRAY_TASK_ID / $((TOTAL-1))"
echo "YAML: $YAML_PATH  Seed: $SEED  W&B project: $WANDB_PROJECT  Run group: $RUN_GROUP"

python train.py --configs "$YAML_PATH" --values \
    seed="$SEED" \
    run_group="$RUN_GROUP" \
    run_name="$RUN_NAME"

#!/bin/bash
#SBATCH --job-name=ou_sweep_arr
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --partition=<YOUR_PARTITION>
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --gpus=1
#SBATCH --mem=40G

# Array submission for run_sweep.py.
#
# Usage:
#   sbatch --array=0-N jobs/submit_array.sh jobs/params_paper.txt
#
# Each non-comment line of the param file is one array task's full arg list for
# run_sweep.py (including --outfile so tasks do not race on the same HDF5).
# Override the --array= range on the command line to match the number of
# non-empty param lines (N = number of lines - 1).

set -euo pipefail

# Conda activation.
if ! command -v conda >/dev/null 2>&1; then
  set +u; source ~/.bashrc; set -u
fi
CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-YOUR_CONDA_ENV}"

PARAM_FILE="${1:?param file required}"
[[ -f "$PARAM_FILE" ]] || { echo "param file not found: $PARAM_FILE" >&2; exit 1; }

# Strip comments and blank lines.
mapfile -t PARAMS < <(grep -Ev '^[[:space:]]*(#|$)' "$PARAM_FILE")

TASK_IDX="${SLURM_ARRAY_TASK_ID:?must be submitted as an array}"
if (( TASK_IDX >= ${#PARAMS[@]} )); then
  echo "TASK_IDX=$TASK_IDX out of range (only ${#PARAMS[@]} param lines)" >&2
  exit 2
fi

LINE="${PARAMS[$TASK_IDX]}"
echo "[task $TASK_IDX] $LINE"
read -r -a ARGS <<< "$LINE"

python -u run_sweep.py "${ARGS[@]}"

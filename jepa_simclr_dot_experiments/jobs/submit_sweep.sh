#!/bin/bash
#SBATCH --job-name=ou_tau_sweep
#SBATCH --nodes=1
#SBATCH --partition=<YOUR_PARTITION>
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --gpus=1
#SBATCH --mem=24G
#SBATCH --array=0-9    # must match length of ALPHAS below minus 1

set -euo pipefail
shopt -s nullglob

conda init bash > /dev/null 2>&1
set +u
source ~/.bashrc
set -u

CONDA_ENV="${CONDA_ENV:-YOUR_CONDA_ENV}"
conda activate "$CONDA_ENV"
trap 'rc=$?; echo "[ERROR] line ${LINENO}: ${BASH_COMMAND} (exit ${rc})" >&2; exit ${rc}' ERR

# ── Parameters — edit these ──────────────────────────────────────────────────
# Defined AFTER bashrc to prevent sourcing from overwriting env overrides.
# alpha = exp(-1/tau); approx tau values: 0, 1, 4.6, 10, 22, 47, 100, 500, 1000, 100000
if [[ -n "${ALPHAS_STR:-}" ]]; then
    IFS=':' read -ra ALPHAS <<< "$ALPHAS_STR"
else
    ALPHAS=(0.0 0.368 0.806 0.905 0.955 0.979 0.99 0.998 0.999 0.99999)
fi

VARIANT="${VARIANT:-baseline_traj_neg}"
NOISE_SIGMA="${NOISE_SIGMA:-3.0}"
DOT_STD="${DOT_STD:-1.3}"
MAX_STEP="${MAX_STEP:-4.0}"
N_STEPS="${N_STEPS:-17}"
SAMPLES="${SAMPLES:-20}"
COMBO_SEED="${COMBO_SEED:-12345}"
BATCH_SIZES="${BATCH_SIZES:-64:128:256:512}"

# Set WANDB_PROJECT to your W&B project name (entity/project will be used as-is).
# To specify an entity, export WANDB_ENTITY before submitting or edit wandb.init() in train.py.
WANDB_PROJECT="${WANDB_PROJECT:-YOUR_PROJECT_NAME}"
PROJECTOR="id"
OU_CENTER_MODE="global_half"
OU_MU=0.5
# ─────────────────────────────────────────────────────────────────────────────

export WANDB_PROJECT

NOISE_ALPHA="${ALPHAS[$SLURM_ARRAY_TASK_ID]}"
ALPHA_TAG="${NOISE_ALPHA//./p}"
SIGMA_TAG="${NOISE_SIGMA//./}"
SIGMA_CFG="${NOISE_SIGMA%.0}"
BASE_CONFIG="${BASE_CONFIG:-configs/sweep_changing_uniform.(${SIGMA_CFG}).simclr.best.yaml}"
RUN_GROUP="${RUN_GROUP:-ou_tau_${SIGMA_TAG}_a${ALPHA_TAG}_${VARIANT}_array}"

echo "Host: $(hostname) | Task: $SLURM_ARRAY_TASK_ID / $((${#ALPHAS[@]}-1))"
echo "alpha=$NOISE_ALPHA | dot_std=$DOT_STD | max_step=$MAX_STEP | T=$N_STEPS | samples=$SAMPLES"

COMBO_FILE="$(mktemp "${PWD}/.combo_tau_${VARIANT}_XXXXXX.tsv")"
FINISHED_FILE="$(mktemp "${PWD}/.finished_runs_XXXXXX.txt")"
trap 'rm -f "$COMBO_FILE" "$FINISHED_FILE"' EXIT

# ── Fetch already-finished runs for this group from W&B ──────────────────────
# Replace YOUR_WANDB_ENTITY with your W&B username or organisation slug.
SKIP_FINISHED="${SKIP_FINISHED:-true}"
if [[ "$SKIP_FINISHED" == "true" ]]; then
    echo "Fetching finished runs for group: $RUN_GROUP"
    WANDB_PROJECT_ENV="$WANDB_PROJECT" RUN_GROUP_ENV="$RUN_GROUP" N_STEPS_ENV="$N_STEPS" python - <<'PY' > "$FINISHED_FILE"
import os, wandb
api = wandb.Api(timeout=60)
entity  = os.environ.get("WANDB_ENTITY", "YOUR_WANDB_ENTITY")
project = os.environ["WANDB_PROJECT_ENV"]
group   = os.environ["RUN_GROUP_ENV"]
n_steps = int(os.environ["N_STEPS_ENV"])
runs = api.runs(
    f"{entity}/{project}",
    filters={"group": group, "state": "finished"},
)
for r in runs:
    if r.config.get("n_steps") == n_steps:
        print(r.name)
PY
    echo "Found $(wc -l < "$FINISHED_FILE") already-finished runs to skip."
fi

SAMPLES_ENV="$SAMPLES" SEED_ENV="$COMBO_SEED" N_STEPS_ENV="$N_STEPS" BATCH_SIZES_ENV="$BATCH_SIZES" python - <<'PY' > "$COMBO_FILE"
import itertools, os, random

sample_n          = int(os.environ["SAMPLES_ENV"])
seed              = int(os.environ["SEED_ENV"])
n_steps           = int(os.environ["N_STEPS_ENV"])
batch_size_values = [int(x) for x in os.environ["BATCH_SIZES_ENV"].split(":")]

base_lr_values     = [0.00025175077582298053, 0.00397036372419493, 0.005468628141751146,
                      0.0058349968994208246,  0.006679431807619572, 0.007883452914848325,
                      0.6934333890743459,     1.197242776802477,    1.7954588193057708,
                      1.9765958136776631,     2.166530826654367,    2.684365337702064,
                      2.9154523385452946,     3.1919744486444204,   3.35896624765643,
                      3.892307395784348]
loss_temp_values   = [0.1, 0.125, 0.5, 1.0, 4.0, 8.0, 16.0, 32.0, 128.0]
normalize_z_values = [True, False]

combos = list(itertools.product(base_lr_values, loss_temp_values, batch_size_values, [n_steps], normalize_z_values))
rng = random.Random(seed)
rng.shuffle(combos)
for i, (lr, temp, bs, ns, normz) in enumerate(combos[:sample_n], 1):
    print(f"{i}\t{lr}\t{temp}\t{bs}\t{ns}\t{'true' if normz else 'false'}")
PY

TOTAL="$(wc -l < "$COMBO_FILE")"
echo "Running $TOTAL HP combinations"

SUCCESS=0
FAILED=0

while IFS=$'\t' read -r IDX LR TEMP BS NSTEPS NORMZ; do
    RUN_NAME="ou_tau_${SIGMA_TAG}_a${ALPHA_TAG}_${VARIANT}_${IDX}"
    if [[ "$SKIP_FINISHED" == "true" ]] && grep -qxF "$RUN_NAME" "$FINISHED_FILE"; then
        echo "Skipping already-finished combo $IDX ($RUN_NAME)"
        SUCCESS=$((SUCCESS + 1))
        continue
    fi
    if python train.py --configs "$BASE_CONFIG" --values \
        simclr.variant="$VARIANT" \
        simclr.projector="$PROJECTOR" \
        simclr.base_lr="$LR" \
        simclr.loss_temp="$TEMP" \
        simclr.normalize_z="$NORMZ" \
        dataset_batch_size="$BS" \
        n_steps="$NSTEPS" \
        val_n_steps="$NSTEPS" \
        dataset_type=Single \
        dataset_static_noise=0.0 \
        dataset_noise="$NOISE_SIGMA" \
        dataset_noise_alpha="$NOISE_ALPHA" \
        dataset_ou_center_mode="$OU_CENTER_MODE" \
        dataset_ou_mu="$OU_MU" \
        dataset_max_step="$MAX_STEP" \
        dataset_dot_std="$DOT_STD" \
        run_group="$RUN_GROUP" \
        run_name="$RUN_NAME" \
        seed=null
    then
        SUCCESS=$((SUCCESS + 1))
    else
        FAILED=$((FAILED + 1))
        echo "Combo $IDX failed (exit $?). Continuing."
    fi
done < "$COMBO_FILE"

echo "Done. success=$SUCCESS failed=$FAILED total=$TOTAL"

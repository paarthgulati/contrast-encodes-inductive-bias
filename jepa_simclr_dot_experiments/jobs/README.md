# jepa_simclr_dot_experiments/jobs — SLURM Submission Scripts

## Overview

Three scripts cover the two training phases used in the paper:

| Script | Purpose |
|---|---|
| `submit_sweep.sh` | Phase 1: HP search — one SLURM array task per OU α value |
| `submit_replicates.sh` | Phase 2 orchestrator: submit a chained array over (YAML, seed) pairs |
| `submit_array_replicates.sh` | Phase 2 worker: each array task runs one (YAML, seed) combo |

## Before submitting

Edit the following placeholders in each script:

| Placeholder | Replace with |
|---|---|
| `<YOUR_PARTITION>` | Your cluster's GPU partition name |
| `YOUR_CONDA_ENV` | Your conda environment name (must have all requirements installed) |
| `YOUR_PROJECT_NAME` | Your W&B project name |
| `YOUR_WANDB_ENTITY` | Your W&B username or organisation slug (in `submit_sweep.sh`) |

## Phase 1 — HP sweep

```bash
cd jepa_simclr_dot_experiments
mkdir -p logs

# baseline_traj_neg, sigma=3.0, 10 alpha values:
VARIANT=baseline_traj_neg NOISE_SIGMA=3.0 sbatch --array=0-9 jobs/submit_sweep.sh

# baseline, same conditions:
VARIANT=baseline NOISE_SIGMA=3.0 sbatch --array=0-9 jobs/submit_sweep.sh
```

Each array task runs `SAMPLES=20` random (LR, temperature, batch_size, normalize_z) combos for one α value and logs them to W&B.

## Phase 2 — Best-HP replicates

After identifying the best HP per condition (stored in `configs/hp_best/`):

```bash
bash jobs/submit_replicates.sh
```

This reads `configs/hp_best/index.csv`, materialises (YAML, seed) pairs, and submits a chained SLURM array. Edit the `BATCHES` array inside `submit_replicates.sh` to control which variants/seeds to run.

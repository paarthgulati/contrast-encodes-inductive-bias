# dvsib_pendulum_experiments/jobs — SLURM Submission Scripts

## Overview

| File | Purpose |
|---|---|
| `submit_array.sh` | SLURM array launcher: one task per non-comment line in a param file |
| `params_paper.txt` | Canonical paper sweep: 4 tasks covering all (σ, α, neg_mode) conditions |

## Before submitting

1. **Download and preprocess the data** (once, from inside `dvsib_pendulum_experiments/`):
   ```bash
   python data.py --all --root data/
   export DATA_ROOT=data
   ```
2. Edit `submit_array.sh`:
   - Replace `<YOUR_PARTITION>` with your cluster's GPU partition.
   - Replace `YOUR_CONDA_ENV` with your conda environment name.
3. Create an output directory: `mkdir -p dvsib_pendulum_experiments/results dvsib_pendulum_experiments/logs`

## Submitting the paper sweep

```bash
cd dvsib_pendulum_experiments
sbatch --array=0-3 jobs/submit_array.sh jobs/params_paper.txt
```

This runs 4 × 15 = 60 individual training runs (15 replicates × 4 condition groups), each writing to a separate HDF5 file in `results/`. Monitor with `squeue -u $USER`.

## Custom sweeps

Create a new param file following the `params_paper.txt` format (one `run_sweep.py` arg string per line, comments with `#`), then:

```bash
# N = number of non-comment lines - 1
sbatch --array=0-N jobs/submit_array.sh jobs/your_params.txt
```

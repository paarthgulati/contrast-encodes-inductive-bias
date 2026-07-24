# jepa_simclr_dot_experiments — SimCLR on Moving-Dot Dataset

## What this component does

This sub-project trains a **SimCLR** contrastive self-supervised learning model on a synthetic 28×28 moving-dot dataset. The model must predict the next frame's representation given the current representation and the intervening action (a 2-D velocity vector), using an RNN predictor. Two contrastive objectives are studied:

- **`baseline`** (`neg_mode=time`): negatives are drawn from all other trajectories in the batch at each time step.
- **`baseline_traj_neg`** (`neg_mode=traj`): negatives are drawn only from other time steps within the *same* trajectory.

Noise is injected as an Ornstein-Uhlenbeck (OU/AR-1) process with memory parameter α ∈ [0,1] (`global_half` mode: mean-reverts to μ=0.5). Results are logged to **Weights & Biases (W&B)**; no local model checkpoints are required for the paper analysis.

Adapted from the SimCLR-predictor JEPA of Sobal et al., "Joint Embedding
Predictive Architectures Focus on Slow Features"
([arXiv:2211.10831](https://arxiv.org/abs/2211.10831)), originally released
under MIT (notice preserved in [`../THIRD_PARTY_LICENSES.md`](../THIRD_PARTY_LICENSES.md)).

---

## Reproducing paper figures

Figures 2, 4, 5, S2, S3, S4, S5 in the paper are produced from `results/*.csv`
(an aggregated export of this component's W&B runs) by the top-level
[`notebooks/reproduce_figures.ipynb`](../notebooks/reproduce_figures.ipynb) —
no need to retrain or query W&B directly. This directory's own training code
below is for reproducing the sweeps that generated those CSVs.

---

## Installation / dependencies

```bash
pip install -r ../requirements.txt   # from repo root
# Requires Python >= 3.9, PyTorch >= 2.0
```

---

## Data

The moving-dot dataset is generated **on the fly** — no download is needed. `ContinuousMotionDataset` (in `data/single.py`) renders Gaussian blobs on a 28×28 canvas and samples random walk trajectories at each training step. Key generation parameters (dot std, max step size, noise σ, OU α) are set via YAML config or command-line flags.

---

## How to run

### Single training run
```bash
cd jepa_simclr_dot_experiments
python train.py --configs configs/sweep_changing_uniform.\(3\).simclr.best.yaml \
    --values simclr.variant=baseline_traj_neg \
             dataset_noise_alpha=0.905 \
             seed=42
```

### Full OU τ sweep
Edit `jobs/submit_sweep.sh` to set `<YOUR_PARTITION>`, `YOUR_CONDA_ENV`, and `YOUR_PROJECT_NAME`, then:
```bash
mkdir -p logs
# 10-alpha array, one task per α value:
sbatch --array=0-9 jobs/submit_sweep.sh
# To run both variants (baseline and baseline_traj_neg):
VARIANT=baseline      sbatch --array=0-9 jobs/submit_sweep.sh
VARIANT=baseline_traj_neg sbatch --array=0-9 jobs/submit_sweep.sh
```

### Best-HP replicates
```bash
# Submits a chained array of all (yaml, seed) combos from configs/hp_best/index.csv:
bash jobs/submit_replicates.sh
```

### Supervised baseline
```bash
python supervised_baseline.py --values \
    dataset_noise=3.0 \
    dataset_noise_alpha=0.905 \
    wandb_project=YOUR_PROJECT_NAME
```

---

## Expected outputs

Each training run logs the following to W&B:

| Key | Description | Logged every |
|---|---|---|
| `loss` | InfoNCE contrastive loss | 100 gradient steps |
| `learning_rate` | Current LARS learning rate | 100 gradient steps |
| `avg_eval_enc_loss` | Position-probe MSE on validation set | `safe_every_n_epochs` epochs (default 10), plus once before training and once at the last epoch |
| `avg_eval_enc_loss_rmse` | RMSE (pixels) of position probe | same cadence as `avg_eval_enc_loss` |

Runs are grouped by `run_group` in W&B. Aggregate results by querying the project via `wandb.Api()`.

---

## Configuration

### Paper hyperparameters (per YAML in `configs/`)

| Parameter | Paper value | Description |
|---|---|---|
| `n_steps` | 17 | Trajectory length T |
| `dataset_noise` | 3.0 | OU noise amplitude σ_U |
| `dataset_noise_alpha` | varies (see sweep) | OU memory α ∈ [0,1] |
| `dataset_ou_center_mode` | `global_half` | OU mean-reversion target (μ=0.5) |
| `simclr.variant` | `baseline` or `baseline_traj_neg` | Negative-sampling strategy |
| `simclr.arch` | `menet5` | CNN backbone for 28×28 images |
| `simclr.embedding_size` | 512 | Encoder output dimension |
| `simclr.backbone_width_factor` | 2 | Channel multiplier for MeNet5 |
| `simclr.projector` | `id` | Projection head (`id` = identity) |
| `simclr.predictor` | `rnn` | Predictor type (GRU) |
| `simclr.epochs` | 100 | Training epochs |
| `simclr.normalize_z` | varies | L2-normalize representations before InfoNCE |
| `simclr.loss_temp` | varies | NT-Xent temperature |
| `dataset_batch_size` | varies | Batch size |

Best hyperparameters per (T, σ, α) condition are recorded in `configs/hp_best/index.csv`.

---

## I/O mechanism

**Output: Weights & Biases.** All metrics are logged live to a W&B project. No local model checkpoints are saved by default (set `output_path` in the config to enable checkpoint saving).

**W&B configuration:**
- Set the `WANDB_PROJECT` environment variable before training, or the run defaults to `YOUR_PROJECT_NAME`.
- Set `WANDB_ENTITY` to your W&B username/org (used in `submit_sweep.sh` for the skip-finished check).
- Downstream consumers query W&B via `wandb.Api()` using the same project name.

---

## Directory layout

```
jepa_simclr_dot_experiments/
├── train.py               # Main training script
├── supervised_baseline.py # Supervised comparison baseline
├── simclr.py              # SimCLR model (baseline + baseline_traj_neg)
├── configs.py             # OmegaConf/dataclass config infrastructure
├── models.py              # Backbone, projector, RNN predictor, probe architectures
├── lars.py                # LARS optimizer + cosine-warmup LR schedule
├── diagnostics.py         # Representation analysis utilities
├── probing.py             # Downstream position-probing
├── resnet.py              # ResNet18 backbone
├── data/
│   └── single.py          # On-the-fly moving-dot dataset with OU noise
├── configs/
│   ├── sweep_changing_uniform.(*).simclr.best.yaml  # Best HP per noise level (changing noise)
│   ├── sweep_fixed_uniform.(*).simclr.best.yaml     # Best HP per noise level (fixed noise)
│   └── hp_best/           # Best HP per (T, σ, α) condition + index.csv
├── jobs/
│   ├── README.md
│   ├── submit_sweep.sh              # OU τ sweep (HP search)
│   ├── submit_replicates.sh         # Best-HP replicate chain orchestrator
│   └── submit_array_replicates.sh   # Per-task array worker
└── results/               # Aggregated W&B export (CSVs), consumed by
                            # notebooks/reproduce_figures.ipynb — not regenerated
                            # by this sub-project's own code
```

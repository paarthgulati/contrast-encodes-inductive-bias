# dvsib_pendulum_experiments — DVSIB on Pendulum Dataset

## What this component does

This sub-project trains a **Deep Variational Symmetric Information Bottleneck (DVSIB)** model on a simulated pendulum dataset (1200 trials × 60 frames of 28×28 pixel observations). The model learns to compress a sliding window of past frames into a low-dimensional latent code z that is maximally informative about future frames, subject to a KL compression penalty (controlled by γ).

Two contrastive objectives are studied:

- **`neg_mode=traj`**: negatives are drawn from other time-windows *within the same trajectory*.
- **`neg_mode=time`**: negatives are drawn from *other trajectories at the same time point*.

OU-correlated pixel noise (parameters σ and α) is added to the clean pendulum pixel data. Trained models are stored in **HDF5 files** alongside training history, probe metrics, and latent rollout arrays. No W&B is used.

Adapted from the DySIB architecture of Martini, Abdelaleem, Gulati, and
Nemenman, "Information bottleneck for learning the phase space of dynamics
from high-dimensional experimental data" (2026,
[arXiv:2604.24662](https://arxiv.org/abs/2604.24662)), originally released
under MIT (notice preserved in [`../THIRD_PARTY_LICENSES.md`](../THIRD_PARTY_LICENSES.md)).

---

## Reproducing paper figures

Figures 2, 3, S1, S2 in the paper are produced from `figure_data/*.csv` and
`checkpoints/*.pt` (small extracts of this component's full sweep) by the
top-level [`notebooks/reproduce_figures.ipynb`](../notebooks/reproduce_figures.ipynb)
— no need to retrain. This directory's own training code below is for
reproducing the full sweep from scratch.

---

## Installation / dependencies

```bash
pip install -r ../requirements.txt   # from repo root
# Requires Python >= 3.9, PyTorch >= 2.0
```

---

## Data

The pendulum dataset must be downloaded and preprocessed before training. Run from inside `dvsib_pendulum_experiments/`:

```bash
python data.py --all --root data/
```

This fetches the [Chen et al. (2022)](https://zenodo.org/records/6653856) pendulum video dataset from Zenodo (~167 MB), downsamples 1200 trajectories from 128×128 RGB to 28×28 grayscale, and writes two files to `data/`:

(`run_sweep.py` and `run_supervised_sweep.py`'s own `load_data()` will also run
this same download+preprocess step automatically if `data/` isn't populated
yet — pre-running it explicitly is only needed to control where/when the
~167 MB download happens, e.g. on an HPC login node before submitting a job
to a compute node without network egress.)

- `data/single_pendulum_all` — PyTorch tensor **(1200, 60, 784)**, pixels ∈ [0,1]
- `data/phys_vars.npy` — NumPy dict with physical labels (`theta`, `vel_theta`, `kinetic energy`, `potential energy`, …)

Set `DATA_ROOT` to the output directory before running sweeps:
```bash
export DATA_ROOT=dvsib_pendulum_experiments/data
```

Data splits are fixed: training trials `0:samples_n`, test trials `1000:1200`.

---

## How to run

### Single sweep (paper conditions)
```bash
cd dvsib_pendulum_experiments
# Set DATA_ROOT to the directory containing single_pendulum_all and phys_vars.npy
export DATA_ROOT=/path/to/pendulum/data
mkdir -p results logs

# Run all 4 paper tasks as a SLURM array (edit partition/conda env in jobs/submit_array.sh first):
sbatch --array=0-3 jobs/submit_array.sh jobs/params_paper.txt
```

### Direct Python invocation (single condition)
```bash
python run_sweep.py \
    --root_dir $DATA_ROOT \
    --outfile results/test.h5 \
    --ndims 8 \
    --num_frames 2 \
    --samples_n 1000 \
    --sigmas 0.0 0.1 0.2 0.3 0.4 0.5 \
    --ou_alphas 0.0 1.0 \
    --neg_modes traj time \
    --reps 15 \
    --epochs 100 \
    --traj_sampler full \
    --probe_every 1
    # --prober_type rf is the default (Appendix B.4); pass --prober_type mlp
    # for the legacy trained-MLP prober instead.
```

### Supervised baseline
```bash
python run_supervised_sweep.py \
    --root_dir $DATA_ROOT \
    --outfile results/supervised.h5 \
    --ndims 8 \
    --num_frames 2 \
    --samples_n 1000 \
    --sigmas 0.0 0.2 \
    --ou_alphas 0.0 1.0 \
    --reps 5 \
    --epochs 100
```

---

## Expected outputs

`run_sweep.py` writes one HDF5 file per invocation. The file layout is:

```
results.h5
└── runs/
    └── dz{kz}_numF{T}_sN{N}_sig{σ}_oua{α}_neg{mode}_rep{r}/
        attrs:        all hyperparameters (incl. prober_type, rf_expand_dim, rf_alpha)
        training/     train_loss, train_kl, train_mi, test_mi        [epochs]
                      probe_epochs, probe_theta_train_deg,
                      probe_theta_test_deg                           [every probe_every epochs, plus one pre-training entry]
        probers/      theta_train_deg, theta_test_deg          (attrs)
        rollout/      what_frames, delta_greedy, delta_pred,
                      rms_theta_greedy, rms_theta_pred          [frames]
        model_weights serialized state_dict (uint8)
```

Helper functions in `run_sweep.py`:
- `load_model(h5_path, run_name, device)` — reconstruct DVSIB from saved weights.
- `load_summary(h5_path)` — returns a pandas DataFrame of scalar metrics for all runs.
- `load_summary_many(h5_paths)` — concatenate summaries from multiple files.

Approximate runtime: ~3 min/run on an A100 GPU (100 epochs, 1000 training trajectories, kz=8).

---

## Configuration

### Paper hyperparameters

| Parameter | Paper value | Description |
|---|---|---|
| `--ndims` | 2 or 8 | Latent dimensionality kz |
| `--num_frames` | 2 | Frames per sliding window |
| `--samples_n` | 1000 | Training trajectories (must be ≤ 1000: trials 1000-1200 are the fixed test split) |
| `--sigmas` | 0.0, 0.1, 0.2, 0.3, 0.4, 0.5 | OU noise amplitude σ |
| `--ou_alphas` | 0.0, 1.0 | OU memory α (0=white noise, 1=static) |
| `--neg_modes` | `traj`, `time` | InfoNCE negative-sampling strategy |
| `--reps` | 15 | Independent random repetitions |
| `--epochs` | 100 | Training epochs |
| `--gamma` | 0.01 | KL compression weight γ |
| `--lr` | 1e-4 | Adam learning rate |
| `--traj_sampler` | `full` | Use all windows per trajectory |
| `--noise_dist` | `gaussian` | Noise distribution (gaussian or uniform) |
| `--probe_every` | 1 | Fit a fresh theta prober every epoch (Appendix B.4) |
| `--prober_type` | `rf` | Theta prober: `rf` (default, Appendix B.4) = fixed random-feature expansion + closed-form ridge-regression head; `mlp` = legacy trained MLP |
| `--rf_expand_dim` | 512 | `[rf]` Random-feature projection dimension |
| `--rf_alpha` | 1e-3 | `[rf]` Ridge regularization strength |

---

## I/O mechanism

**Output: HDF5 files.** Each run is stored as a group inside a single `.h5` file with embedded model weights, training history, probe results, and latent rollout arrays. No W&B is used.

Downstream consumers call `load_model` and `load_summary` from `run_sweep.py` to load trained models and aggregate metrics.

**Note on reproducibility:** Noise seeding and training seeding are decoupled so that the same noise is applied across different `neg_modes` for the same `(rep, sigma, ou_alpha)` tuple.

---

## Directory layout

```
dvsib_pendulum_experiments/
├── data.py                # Download + preprocess pendulum dataset from Zenodo
├── run_sweep.py           # Main CLI: sweep, train, probe, save to HDF5
├── run_supervised_sweep.py # Supervised baseline CLI
├── models.py              # DVSIB model, Encoder, DeltaDecoder, Prober,
│                          # RandomFeatureProber, datasets
├── noise.py               # OU/AR-1 noise generators
├── training.py            # train_dvsib, train_prober, fit_ridge_head,
│                          # encode_dataset, rollout_latent
├── supervised_models.py   # SupervisedModel (Encoder + linear head for θ)
├── checkpoints/           # Small trained-model checkpoints (kz=2), consumed by
│                          # notebooks/reproduce_figures.ipynb — not regenerated
│                          # by this sub-project's own code
├── figure_data/            # Extracted sweep metrics (CSVs), consumed by
│                          # notebooks/reproduce_figures.ipynb
└── jobs/
    ├── README.md
    ├── submit_array.sh    # SLURM array submission script
    └── params_paper.txt   # Canonical paper sweep parameters (4 array tasks)
```

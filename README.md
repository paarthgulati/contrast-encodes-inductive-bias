# Contrast encodes inductive bias: separating slow noise from dynamics in predictive representation learning

Code, data, and trained-model artifacts accompanying:

> Paarth Gulati and Ilya Nemenman. "Contrast encodes inductive bias: separating
> slow noise from dynamics in predictive representation learning." 2026.
> [arXiv:2606.07770](https://arxiv.org/abs/2606.07770)

---

## Overview

This repository contains code for two independent sets of experiments studying how the choice of negative-sampling strategy in contrastive predictive objectives determines whether a model learns the true latent dynamics or instead encodes slowly varying observation noise. Noise temporal correlation is controlled by an Ornstein-Uhlenbeck (OU) process with memory parameter α ∈ [0,1]:

- **α = 0**: i.i.d. (white) noise, independent per frame.
- **α = 1**: fully static noise, identical across all frames in a trajectory.
- **0 < α < 1**: temporally correlated OU noise with correlation time τ = −1/ln(α).

When negatives are sampled *across* trajectories (ATN), slowly varying (high-α) noise provides an easy predictive shortcut and the learned representation is dominated by trajectory-specific noise rather than dynamics. Sampling negatives *within* a single trajectory (WTN) removes this shortcut and forces the encoder to capture the underlying dynamical variables.

The two sub-projects demonstrate this in different settings and use different datasets, models, and I/O pipelines. They are treated as independent codebases and do not share code, data, or checkpoints:

- **`jepa_simclr_dot_experiments/`** — SimCLR-predictor JEPA on a synthetic moving-dot dataset
- **`dvsib_pendulum_experiments/`** — DySIB on real experimental video of a rigid pendulum

Both directories work the same way for a reader: run `notebooks/reproduce_figures.ipynb` to regenerate every figure from shipped data (no GPU, no training), or use the sub-project's own training code to reproduce the underlying sweeps from scratch.

---

## Reproducing the paper's figures (no training needed)

```bash
pip install -r requirements.txt
jupyter lab notebooks/reproduce_figures.ipynb
```

The notebook reads only from data already shipped in this repository — aggregated
sweep-metric CSVs (`jepa_simclr_dot_experiments/results/`,
`dvsib_pendulum_experiments/figure_data/`) and small trained-model checkpoints
(`dvsib_pendulum_experiments/checkpoints/`). No W&B account and no live HDF5 sweep
files are needed. The one exception: it will download the ~167 MB pendulum video
dataset from Zenodo on first run if `dvsib_pendulum_experiments/data/` isn't
already populated (needed to re-encode a few example trajectories for the
embedding figures).

---

## Training from scratch

See each sub-project's own README for full details, hyperparameters, and expected outputs.

### SimCLR (moving-dot)
```bash
cd jepa_simclr_dot_experiments
# Single run — trajectory-contrast variant, σ=3.0, α≈0.905 (τ≈10 frames)
python train.py \
    --configs configs/sweep_changing_uniform.\(3\).simclr.best.yaml \
    --values simclr.variant=baseline_traj_neg dataset_noise_alpha=0.905 seed=42
```
Results appear in your W&B project (set `WANDB_PROJECT` env var).

### DVSIB (pendulum)
```bash
cd dvsib_pendulum_experiments
# Step 1 — download and preprocess the pendulum dataset (~167 MB from Zenodo):
python data.py --all --root data/
export DATA_ROOT=data
# Step 2 — run the paper sweep as a SLURM array (edit partition/env in jobs/submit_array.sh first):
mkdir -p results logs
sbatch --array=0-3 jobs/submit_array.sh jobs/params_paper.txt
```
Results are saved as HDF5 in `results/`.

---

## Repository structure

```
contrast-encodes-inductive-bias/
├── README.md                 ← this file
├── LICENSE                   GPL-3.0
├── THIRD_PARTY_LICENSES.md   Upstream MIT notices (see Code credits below)
├── requirements.txt
├── notebooks/
│   └── reproduce_figures.ipynb   Regenerates every figure from shipped data
├── jepa_simclr_dot_experiments/         # SimCLR on 28×28 moving-dot dataset (W&B-based I/O)
│   ├── README.md
│   ├── train.py, simclr.py, supervised_baseline.py, data/single.py, ...
│   ├── configs/               # Best YAML configs + hp_best/ index
│   ├── jobs/                  # SLURM sweep and replicate scripts
│   └── results/                # Aggregated W&B export (CSVs), used by the notebook
└── dvsib_pendulum_experiments/          # DVSIB on 28×28 pendulum dataset (HDF5-based I/O)
    ├── README.md
    ├── data.py, run_sweep.py, run_supervised_sweep.py, models.py, noise.py, training.py
    ├── checkpoints/            # Small trained-model checkpoints, used by the notebook
    ├── figure_data/             # Extracted sweep metrics (CSVs), used by the notebook
    └── jobs/                   # SLURM array script + canonical param file
```

See each subdirectory's `README.md` for full details on data, configuration, expected outputs, and I/O mechanisms.

---

## Installation

```bash
pip install -r requirements.txt   # Python >= 3.9, PyTorch >= 2.0
```

---

## Noise model

The two sub-projects use different OU noise implementations:

**SimCLR (moving-dot):** Uniform[0,1] innovations, mean-reverting to μ = 0.5:
```
η_t = μ + α·(η_{t-1} − μ) + β·(U_t − μ),   β = √(1 − α²),   U_t ~ Uniform[0,1]
x_noisy = x_clean + σ · η_t · max(x_clean)
```
No clipping is applied.

**DVSIB (pendulum):** Gaussian innovations, zero-mean:
```
η_t = α·η_{t-1} + β·ξ_t,   β = √(1 − α²),   ξ_t ~ N(0,1)
x_noisy = clip(x_clean + σ · η_t,  0, 1)
```

---

## Code credits

Both sub-projects are adapted from previously released open-source codebases,
originally distributed under the MIT license (see [`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md)
for the full original notices):

- **`jepa_simclr_dot_experiments/`** adapts the SimCLR-predictor JEPA of:
  > V. Sobal, J. S V, S. Jalagam, N. Carion, K. Cho, and Y. LeCun. "Joint
  > Embedding Predictive Architectures Focus on Slow Features." SSL Theory and
  > Practice Workshop, NeurIPS 2022. [arXiv:2211.10831](https://arxiv.org/abs/2211.10831)

- **`dvsib_pendulum_experiments/`** adapts the DySIB architecture of:
  > K. M. Martini, E. Abdelaleem, P. Gulati, and I. Nemenman. "Information
  > bottleneck for learning the phase space of dynamics from high-dimensional
  > experimental data." 2026. [arXiv:2604.24662](https://arxiv.org/abs/2604.24662)

- The pendulum video dataset itself is from:
  > B. Chen, K. Huang, S. Raghupathi, I. Chandratreya, Q. Du, and H. Lipson.
  > "Automated discovery of fundamental variables hidden in experimental
  > data." *Nature Computational Science* 2, 433–442 (2022).
  > [doi:10.1038/s43588-022-00281-6](https://doi.org/10.1038/s43588-022-00281-6)

---

## License

This repository is licensed under GPL-3.0 — see [`LICENSE`](LICENSE). Both
sub-projects derive from MIT-licensed upstream code; the original MIT
copyright notices are preserved in [`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md)
as required by their terms.

---

## Citation

```bibtex
@article{gulati2026contrast,
  title   = {Contrast encodes inductive bias: separating slow noise from
             dynamics in predictive representation learning},
  author  = {Gulati, Paarth and Nemenman, Ilya},
  journal = {arXiv preprint arXiv:2606.07770},
  year    = {2026},
}
```

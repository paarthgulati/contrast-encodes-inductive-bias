"""
Sweep script: DVSIB on pendulum pixel data with OU-correlated observation noise.

Sweeps over:
    ndims, num_frames, samples_n, sigma, ou_alpha, neg_mode, rep.

Each sweep point trains one DVSIB model on pixel trajectories corrupted by
Ornstein-Uhlenbeck noise (sigma = marginal std, ou_alpha = temporal memory).
Noise is applied to all 1200 trajectories (train + test) with a
parameter-dependent seed so two sweep points with the same (rep, sigma,
ou_alpha) share the same noisy dataset.

The DVSIB predictive objective is always the residual form
`pred_mean = zx + delta`.

`neg_mode` controls which batch entries serve as InfoNCE negatives; see
`models.DVSIB` for details:
    "all"   : standard InfoNCE -- all other batch entries are negatives.
    "traj"  : within-trajectory negatives only -- for row i, the denominator
              sums over batch entries j with trial(j) == trial(i). Forces the
              critic to distinguish timepoints of the *same* trajectory
              rather than different trajectories.
    "time"  : cross-trajectory negatives only -- for row i, the denominator
              sums over batch entries j with time(j) == time(i). Forces the
              critic to distinguish *different trajectories at the same
              time step* rather than different timepoints.

All results are written to a single HDF5 file. Runs are identified by a
unique key derived from their parameters; completed runs are skipped on
restart unless --overwrite is passed.

HDF5 layout
-----------
results.h5
└── runs/
    └── {run_name}/
        attrs:  ndims, num_frames, samples_n, rep,
                sigma, ou_alpha, neg_mode,
                seed, noise_seed, epochs, gamma, batch_size, input_dim,
                prober_type, rf_expand_dim, rf_alpha
        ├── training/   { train_loss, train_kl, train_mi, test_mi,
                          probe_epochs, probe_theta_train_deg,
                          probe_theta_test_deg }
        ├── probers/    attrs: theta_{train,test}_deg
        ├── rollout/    { what_frames, delta_greedy, delta_pred,
                          rms_theta_{greedy,pred} }
        └── model_weights   (uint8 1-D, serialised state_dict)

Usage example
-------------
    python run_sweep.py \\
        --root_dir /path/to/data \\
        --outfile  results/results.h5 \\
        --ndims 2 --num_frames 2 --samples_n 256 1000 \\
        --sigmas 0.0 0.1 0.2 --ou_alphas 0.0 0.8 0.95 \\
        --neg_modes all traj time \\
        --reps 3 --epochs 300
"""

import argparse
import io
from pathlib import Path

import h5py
import numpy as np
import torch

from models import (DVSIB, Prober, RandomFeatureProber, FrameDataset,
                    TrajGroupedBatchSampler, TimeGroupedBatchSampler,
                    DatasetTrajFull, collate_traj_full,
                    VALID_NEG_MODES)
from noise import add_ou_noise_per_trajectory, add_ou_noise_uniform
from training import (
    train_dvsib,
    train_prober,
    fit_ridge_head,
    predict_prober,
    wrapped_theta_rmse_deg_from_cos_sin,
    encode_dataset,
    rollout_latent,
    rollout_theta_rmse_wrapped,
)


# ─────────────────────────────────────────────────────────────────────────────
# HDF5 I/O utilities (importable from notebooks)
# ─────────────────────────────────────────────────────────────────────────────

def list_runs(h5_path):
    """Return a sorted list of run names stored in the HDF5 file."""
    h5_path = Path(h5_path)
    if not h5_path.exists():
        return []
    with h5py.File(h5_path, "r") as f:
        return sorted(f.get("runs", {}).keys())


def load_summary(h5_path):
    """
    Load scalar metrics for every completed run into a pandas DataFrame.
    Columns include all run parameters plus all prober metrics and final MI.
    An `h5_file` column records the source path, handy when concatenating
    summaries from multiple sweep files.
    """
    import pandas as pd

    h5_path = Path(h5_path)
    rows = []
    with h5py.File(h5_path, "r") as f:
        runs_grp = f.get("runs", {})
        for run_name, grp in runs_grp.items():
            row = {"run_name": run_name, "h5_file": str(h5_path)}
            row.update({k: _attr_to_py(v) for k, v in grp.attrs.items()})
            row.update({k: float(v) for k, v in grp["probers"].attrs.items()})
            row["final_train_mi"] = float(grp["training/train_mi"][-1])
            row["final_test_mi"] = float(grp["training/test_mi"][-1])
            rows.append(row)
    return pd.DataFrame(rows)


def load_summary_many(h5_paths, *, drop_duplicates=True):
    """
    Concatenate `load_summary(p)` over several HDF5 files into one DataFrame.

    Args:
        h5_paths        : iterable of HDF5 paths. Missing paths are skipped.
        drop_duplicates : if True (default) and the same run_name appears in
                          more than one file, keep the first occurrence and
                          drop the rest.

    Returns:
        pandas.DataFrame with an `h5_file` column tracing each row to its
        source file.
    """
    import pandas as pd

    frames = []
    for p in h5_paths:
        if Path(p).exists():
            frames.append(load_summary(p))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    if drop_duplicates and "run_name" in df.columns:
        df = df.drop_duplicates(subset="run_name", keep="first").reset_index(drop=True)
    return df


def find_run_file(h5_paths, run_name):
    """
    Return the first HDF5 path in `h5_paths` that contains `runs/{run_name}`,
    or raise FileNotFoundError listing what was searched.
    """
    searched = []
    for p in h5_paths:
        p = Path(p)
        searched.append(str(p))
        if not p.exists():
            continue
        with h5py.File(p, "r") as f:
            if f"runs/{run_name}" in f:
                return p
    raise FileNotFoundError(
        f"Run {run_name!r} not found in any of: {searched}"
    )


def _attr_to_py(v):
    """Normalise HDF5 attr values (bytes -> str, numpy scalars -> python)."""
    if isinstance(v, bytes):
        return v.decode("utf-8")
    if isinstance(v, np.generic):
        return v.item()
    return v


def load_run(h5_path, run_name):
    """Load all stored data for a single run."""
    with h5py.File(h5_path, "r") as f:
        grp = f[f"runs/{run_name}"]
        return {
            "params":   {k: _attr_to_py(v) for k, v in grp.attrs.items()},
            "training": {k: grp[f"training/{k}"][:] for k in grp["training"]},
            "probers":  {k: float(v) for k, v in grp["probers"].attrs.items()},
            "rollout":  {k: grp[f"rollout/{k}"][:] for k in grp["rollout"]},
        }


def load_model(h5_path, run_name, device="cpu"):
    """Reconstruct a trained DVSIB model from serialised weights."""
    with h5py.File(h5_path, "r") as f:
        grp = f[f"runs/{run_name}"]
        params = {k: _attr_to_py(v) for k, v in grp.attrs.items()}
        raw = grp["model_weights"][:]

    buf = io.BytesIO(raw.tobytes())
    state_dict = torch.load(buf, map_location=device)

    model = DVSIB(
        input_dim=int(params["input_dim"]),
        dz=int(params["ndims"]),
        dzf=32,
        num_frames=int(params["num_frames"]),
        gamma=float(params["gamma"]),
        alpha=1.0,
        hidden=256,
        decoder_hdim=64,
        neg_mode=str(params["neg_mode"]),
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _run_exists(h5_path, run_name):
    if not Path(h5_path).exists():
        return False
    with h5py.File(h5_path, "a") as f:
        return f"runs/{run_name}" in f


def _delete_run(h5_path, run_name):
    with h5py.File(h5_path, "a") as f:
        key = f"runs/{run_name}"
        if key in f:
            del f[key]


def _save_run(h5_path, run_name, params, hist, probe_metrics, rollout_arrays,
              model_state_bytes):
    Path(h5_path).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(h5_path, "a") as f:
        grp = f.require_group(f"runs/{run_name}")
        for k, v in params.items():
            grp.attrs[k] = v
        tr = grp.require_group("training")
        for key, arr in hist.items():
            tr.create_dataset(key, data=np.asarray(arr, dtype=np.float32))
        pb = grp.require_group("probers")
        for k, v in probe_metrics.items():
            pb.attrs[k] = float(v)
        ro = grp.require_group("rollout")
        for k, v in rollout_arrays.items():
            ro.create_dataset(k, data=np.asarray(v, dtype=np.float32))
        grp.create_dataset(
            "model_weights",
            data=np.frombuffer(model_state_bytes, dtype=np.uint8),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pendulum_data_module():
    """Load dvsib_pendulum_experiments/data.py by file path (not `import data`) so this
    doesn't collide with jepa_simclr_dot_experiments' own `data` package when both
    sub-projects are on sys.path at once, e.g. inside notebooks/reproduce_figures.ipynb."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_dvsib_pendulum_data", Path(__file__).resolve().parent / "data.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_data(root_dir):
    """
    Load pixel trajectories and physical variables.

    Expects in root_dir:
        single_pendulum_all   - torch-saved tensor (1200, 60, 784), clean pixels
        phys_vars.npy         - numpy dict with aligned physical variables

    If either file is missing, downloads and preprocesses the pendulum
    dataset from Zenodo into root_dir first (see data.py).
    """
    root = Path(root_dir).resolve()
    tensor_path = root / "single_pendulum_all"
    phys_path = root / "phys_vars.npy"
    if not (tensor_path.exists() and phys_path.exists()):
        data_mod = _pendulum_data_module()
        raw_dir = data_mod.extract(data_mod.download(root=root), root=root)
        data_mod.preprocess(raw_dir, root=root)
    data = torch.load(tensor_path, weights_only=False)
    data = torch.as_tensor(data).float()
    phys = np.load(phys_path, allow_pickle=True).item()
    return data, phys


def make_loaders(data, num_frames, samples_n, batch_size,
                 neg_mode="all", traj_batch=64, k_per_traj=16,
                 traj_sampler="grouped", traj_chunk_size=0,
                 time_batch=None, k_per_time=18,
                 standard_sampler=False):
    """
    Build training DataLoader and a capped test batch for DVSIB eval.

    neg_mode='all' (or standard_sampler=True):
        Standard shuffled DataLoader with batch_size.

    neg_mode='traj', traj_sampler='grouped' (default for traj):
        TrajGroupedBatchSampler -- randomly selects traj_batch trajectories
        and k_per_traj windows per trajectory each step.
        Effective batch size = traj_batch * k_per_traj.

    neg_mode='traj', traj_sampler='full':
        DatasetTrajFull -- one dataset item = all numD windows of one
        trajectory. DataLoader batches traj_chunk_size trajectories at a
        time; collate_traj_full flattens to (traj_chunk_size*numD, nF, D).
        Every row sees all numD-1 same-trajectory windows as negatives.
        traj_chunk_size=0 (default) auto-sets to max(1, batch_size // numD).

    The test batch is always the first `batch_size` windows of the
    sequentially-ordered test FrameDataset so same-trajectory windows are
    grouped together -- required for neg_mode='traj' test-MI eval.
    """
    numD = data.shape[1] - 2 * num_frames + 1

    if not standard_sampler:
        if neg_mode == "traj":
            if traj_sampler == "full":
                chunk = traj_chunk_size if traj_chunk_size > 0 else max(1, batch_size // numD)
                train_loader = torch.utils.data.DataLoader(
                    DatasetTrajFull(data[:samples_n], num_frames),
                    batch_size=chunk,
                    shuffle=True,
                    collate_fn=collate_traj_full,
                )
            else:  # "grouped"
                sampler = TrajGroupedBatchSampler(
                    n_trials=samples_n,
                    numFD=numD,
                    traj_batch=traj_batch,
                    k_per_traj=k_per_traj,
                )
                train_loader = torch.utils.data.DataLoader(
                    FrameDataset(data[:samples_n], num_frames),
                    batch_sampler=sampler,
                )
        elif neg_mode == "time":
            if traj_sampler == "full":
                # DatasetTrajFull batches full trajectories; after flattening
                # every time point has k trajectories as cross-traj negatives.
                chunk = traj_chunk_size if traj_chunk_size > 0 else max(1, batch_size // numD)
                train_loader = torch.utils.data.DataLoader(
                    DatasetTrajFull(data[:samples_n], num_frames),
                    batch_size=chunk,
                    shuffle=True,
                    collate_fn=collate_traj_full,
                )
            else:  # "grouped" -> TimeGroupedBatchSampler
                sampler = TimeGroupedBatchSampler(
                    n_trials=samples_n,
                    numFD=numD,
                    time_batch=time_batch,
                    k_per_time=k_per_time,
                )
                train_loader = torch.utils.data.DataLoader(
                    FrameDataset(data[:samples_n], num_frames),
                    batch_sampler=sampler,
                )
        else:  # "all"
            train_loader = torch.utils.data.DataLoader(
                FrameDataset(data[:samples_n], num_frames),
                batch_size=batch_size,
                shuffle=True,
            )
    else:
        train_loader = torch.utils.data.DataLoader(
            FrameDataset(data[:samples_n], num_frames),
            batch_size=batch_size,
            shuffle=True,
        )

    test_all = FrameDataset(data[1000:], num_frames)
    test_loader = torch.utils.data.DataLoader(
        test_all, batch_size=len(test_all), shuffle=False
    )
    x_test, y_test, tid_test, tmi_test = next(iter(test_loader))
    x_test    = x_test[:batch_size]
    y_test    = y_test[:batch_size]
    tid_test  = tid_test[:batch_size]
    tmi_test  = tmi_test[:batch_size]

    return train_loader, x_test, y_test, tid_test, tmi_test, numD


def make_phys_tensors(phys, num_frames):
    """
    Flatten physical variable arrays to match FrameDataset (trial, window) order.
    Physical variables are tied to the *clean* underlying pendulum state.
    """
    sl = slice(None, -(2 * num_frames - 1))

    def flat(arr):
        return torch.from_numpy(arr[:, sl].flatten()).float()

    return {
        "theta":   flat(phys["theta"]),
        "omega":   flat(phys["vel_theta"]),
        "kinetic": flat(phys["kinetic energy"]),
        "pot":     flat(phys["potential energy"]),
    }


def _fmt(x):
    """Compact numeric tag used in run names (0.1 -> '0p1')."""
    return str(x).replace(".", "p").replace("-", "m")


_DEFAULT_GAMMA = 0.01
_DEFAULT_LR    = 1e-4


def make_run_name(ndims, num_frames, samples_n, sigma, ou_alpha, neg_mode, rep,
                  noise_dist="gaussian", gamma=_DEFAULT_GAMMA, lr=_DEFAULT_LR):
    # NOTE: this name does not encode traj_sampler, prober_type, rf_expand_dim,
    # rf_alpha, k_per_traj, k_per_time, or standard_sampler. Re-running the
    # same (ndims, num_frames, samples_n, sigma, ou_alpha, neg_mode, rep) with
    # a different value for any of those produces the SAME run_name, so it
    # will be silently skipped (without --overwrite) or silently clobber the
    # previous run (with --overwrite). Be mindful when sweeping those params.
    name = (
        f"dz{ndims}_numF{num_frames}_sN{samples_n}"
        f"_sig{_fmt(sigma)}_oua{_fmt(ou_alpha)}_neg{neg_mode}_rep{rep}"
    )
    if noise_dist != "gaussian":
        name += f"_ndist{noise_dist}"
    if abs(gamma - _DEFAULT_GAMMA) > 1e-9:
        name += f"_gam{_fmt(gamma)}"
    if abs(lr - _DEFAULT_LR) > 1e-10:
        name += f"_lr{_fmt(lr)}"
    return name


def make_seed(base_seed, rep):
    """Deterministic training seed; only depends on rep, matching No_Added_Noise."""
    return base_seed + (rep - 1)


def make_noise_seed(base_noise_seed, rep, sigma, ou_alpha):
    """
    Deterministic OU-noise seed. The same (rep, sigma, ou_alpha) produces the
    same noisy dataset across different `neg_mode`/`ndims`/`num_frames` sweep
    points, so neg_mode comparisons are apples-to-apples on identical noise.
    """
    return int(
        base_noise_seed
        + 1000 * (rep - 1)
        + int(round(100 * sigma))
        + int(round(10_000 * ou_alpha))
    )


# ─────────────────────────────────────────────────────────────────────────────
# Single-run computation
# ─────────────────────────────────────────────────────────────────────────────

def run_one(*, data_clean, phys, ndims, num_frames, samples_n,
            sigma, ou_alpha, neg_mode, rep, seed, noise_seed,
            epochs, batch_size, gamma, lr, device,
            traj_batch=64, k_per_traj=16,
            traj_sampler="grouped", traj_chunk_size=0,
            time_batch=None, k_per_time=18,
            standard_sampler=False, noise_dist="gaussian",
            probe_every=1, no_clip=False,
            prober_type="rf", rf_expand_dim=512, rf_alpha=1e-3):
    """
    Execute one sweep point:
        1. Set seeds.
        2. Add OU noise to the *entire* pixel tensor.
        3. Train DVSIB on noisy windows.
        4. Encode all (noisy) data to latent codes.
        5. Train probers from clean physical variables.
        6. Roll out the latent dynamics; decode physical quantities per step.
    """
    run_name = make_run_name(
        ndims, num_frames, samples_n, sigma, ou_alpha, neg_mode, rep,
        noise_dist=noise_dist, gamma=gamma, lr=lr,
    )
    print(f"\n{'='*60}\n{run_name}  (seed={seed}, noise_seed={noise_seed})\n{'='*60}")

    # 1. Seeds for training stochasticity. Noise RNG uses noise_seed separately.
    torch.manual_seed(seed)
    np.random.seed(seed)

    T = data_clean.shape[1]
    D = data_clean.shape[2]
    numD = T - 2 * num_frames + 1

    # 2. Apply OU noise.
    _noise_fn = (add_ou_noise_uniform if noise_dist == "uniform"
                 else add_ou_noise_per_trajectory)
    data_noisy = _noise_fn(
        data_clean, sigma=sigma, ou_alpha=ou_alpha,
        clip=None if no_clip else (0.0, 1.0), seed=noise_seed, init="stationary",
    )

    # 3. Train DVSIB on the noisy data
    train_loader, x_test, y_test, tid_test, tmi_test, _ = make_loaders(
        data_noisy, num_frames, samples_n, batch_size,
        neg_mode=neg_mode, traj_batch=traj_batch, k_per_traj=k_per_traj,
        traj_sampler=traj_sampler, traj_chunk_size=traj_chunk_size,
        time_batch=time_batch, k_per_time=k_per_time,
        standard_sampler=standard_sampler,
    )
    x_test = x_test.to(device)
    y_test = y_test.to(device)
    tid_test = tid_test.to(device)

    model = DVSIB(
        input_dim=D, dz=ndims, dzf=32, num_frames=num_frames,
        gamma=gamma, alpha=1.0, hidden=256, decoder_hdim=64,
        neg_mode=neg_mode,
    ).to(device)

    # Precompute physical variable tensors (used by probe callback and final probers)
    fvars = make_phys_tensors(phys, num_frames)

    def tr(f): return f[:samples_n * numD]
    def ts(f): return f[1000 * numD:]

    # Pre-compute theta targets once; captured by the probe callback closure
    _theta_train    = tr(fvars["theta"])
    _theta_test     = ts(fvars["theta"])
    _theta_train_xy = torch.stack([torch.cos(_theta_train), torch.sin(_theta_train)], dim=1)
    _theta_test_xy  = torch.stack([torch.cos(_theta_test),  torch.sin(_theta_test)],  dim=1)

    def _probe_fn(model):
        """Encode current representations, fit a fresh theta prober, return RMSE."""
        zx_all_cb, _ = encode_dataset(model, data_noisy, num_frames, device=device)
        zx_tr_cb = zx_all_cb[:samples_n * numD]
        zx_ts_cb = zx_all_cb[1000 * numD:]
        if prober_type == "rf":
            p = RandomFeatureProber(dz=ndims, output_dim=2,
                                    expand_dim=rf_expand_dim).to(device)
            fit_ridge_head(p, zx_tr_cb, _theta_train_xy,
                           alpha=rf_alpha, device=device)
        else:  # "mlp"
            p = Prober(dz=ndims, output_dim=2, hdim=1024).to(device)
            p, _, _ = train_prober(
                p,
                z_train=zx_tr_cb, y_train=_theta_train_xy,
                z_test=zx_ts_cb,  y_test=_theta_test_xy,
                epochs=50, lr=1e-4, batch_size=batch_size, device=device,
            )
        theta_tr_pred = predict_prober(p, zx_tr_cb, batch_size=batch_size, device=device)
        theta_ts_pred = predict_prober(p, zx_ts_cb, batch_size=batch_size, device=device)
        return {
            "theta_train_deg": wrapped_theta_rmse_deg_from_cos_sin(theta_tr_pred, _theta_train),
            "theta_test_deg":  wrapped_theta_rmse_deg_from_cos_sin(theta_ts_pred, _theta_test),
        }

    model, hist = train_dvsib(
        model, train_loader, x_test, y_test, tid_test,
        tmi_test=tmi_test.to(device),
        epochs=epochs, lr=lr, device=device,
        probe_fn=_probe_fn if probe_every > 0 else None,
        probe_every=probe_every,
    )

    # 4. Encode all data (noisy) into latent means
    zx_all, _ = encode_dataset(model, data_noisy, num_frames, device=device)
    zx_train = zx_all[:samples_n * numD]
    zx_test = zx_all[1000 * numD:]

    # 5. Probers — targets are clean physical variables

    probe_metrics = {}

    theta_train = tr(fvars["theta"])
    theta_test = ts(fvars["theta"])
    theta_train_xy = torch.stack(
        [torch.cos(theta_train), torch.sin(theta_train)], dim=1
    )
    theta_test_xy = torch.stack(
        [torch.cos(theta_test), torch.sin(theta_test)], dim=1
    )

    if prober_type == "rf":
        p_theta = RandomFeatureProber(dz=ndims, output_dim=2,
                                      expand_dim=rf_expand_dim).to(device)
        fit_ridge_head(p_theta, zx_train, theta_train_xy,
                       alpha=rf_alpha, device=device)
    else:  # "mlp"
        p_theta = Prober(dz=ndims, output_dim=2, hdim=1024).to(device)
        p_theta, _, _ = train_prober(
            p_theta,
            z_train=zx_train, y_train=theta_train_xy,
            z_test=zx_test,   y_test=theta_test_xy,
            epochs=100, lr=1e-4, batch_size=batch_size, device=device,
        )
    theta_train_pred = predict_prober(p_theta, zx_train, batch_size=batch_size, device=device)
    theta_test_pred = predict_prober(p_theta, zx_test, batch_size=batch_size, device=device)
    probe_metrics["theta_train_deg"] = wrapped_theta_rmse_deg_from_cos_sin(
        theta_train_pred, theta_train
    )
    probe_metrics["theta_test_deg"] = wrapped_theta_rmse_deg_from_cos_sin(
        theta_test_pred, theta_test
    )
    print(f"  theta   train={probe_metrics['theta_train_deg']:.2f} deg"
          f"  test={probe_metrics['theta_test_deg']:.2f} deg")

    # 6. Rollout and theta RMS
    rollout = rollout_latent(model, zx_test, num_frames, numD, device=device)

    rms_theta_g, rms_theta_p = rollout_theta_rmse_wrapped(
        p_theta, rollout, ts(fvars["theta"]), numD, num_frames,
        batch_size=batch_size, device=device,
    )

    rollout_arrays = {
        "what_frames":      rollout["what_frames"],
        "delta_greedy":     rollout["delta_greedy"],
        "delta_pred":       rollout["delta_pred"],
        "rms_theta_greedy": rms_theta_g,
        "rms_theta_pred":   rms_theta_p,
    }

    params = {
        "ndims":       ndims,
        "num_frames":  num_frames,
        "samples_n":   samples_n,
        "rep":         rep,
        "sigma":       float(sigma),
        "ou_alpha":    float(ou_alpha),
        "neg_mode":    neg_mode,
        "seed":        seed,
        "noise_seed":  noise_seed,
        "epochs":           epochs,
        "gamma":            gamma,
        "lr":               lr,
        "batch_size":       batch_size,
        "traj_batch":       traj_batch,
        "k_per_traj":       k_per_traj,
        "traj_sampler":     traj_sampler,
        "traj_chunk_size":  traj_chunk_size,
        "time_batch":       time_batch if time_batch is not None else 0,
        "k_per_time":       k_per_time,
        "standard_sampler": int(standard_sampler),
        "noise_dist":       noise_dist,
        "input_dim":        D,
        "probe_every":      probe_every,
        "prober_type":      prober_type,
        "rf_expand_dim":    rf_expand_dim,
        "rf_alpha":         rf_alpha,
    }

    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    model_state_bytes = buf.getvalue()

    return params, hist, probe_metrics, rollout_arrays, model_state_bytes


# ─────────────────────────────────────────────────────────────────────────────
# Summary plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_theta_vs_ou_alpha(df, outdir):
    """theta test RMSE vs ou_alpha, one curve per (sigma, neg_mode)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(dpi=120)
    cmap = plt.cm.tab10.colors
    for i, ((sigma, neg_mode), grp) in enumerate(df.groupby(["sigma", "neg_mode"])):
        s = (
            grp.groupby("ou_alpha")
            .agg(mean=("theta_test_deg", "mean"),
                 std=("theta_test_deg", "std"))
            .reset_index().sort_values("ou_alpha")
        )
        ax.errorbar(
            s["ou_alpha"], s["mean"], yerr=s["std"],
            marker="o", capsize=3,
            label=f"sigma={sigma}, neg={neg_mode}",
            color=cmap[i % len(cmap)],
        )
    ax.set_xlabel(r"OU memory $\alpha$")
    ax.set_ylabel(r"$\theta$ test RMSE (deg)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(Path(outdir) / "theta_vs_ou_alpha.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_theta_vs_sigma(df, outdir):
    """theta test RMSE vs sigma, one curve per (ou_alpha, neg_mode)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(dpi=120)
    cmap = plt.cm.tab10.colors
    for i, ((ou_alpha, neg_mode), grp) in enumerate(df.groupby(["ou_alpha", "neg_mode"])):
        s = (
            grp.groupby("sigma")
            .agg(mean=("theta_test_deg", "mean"),
                 std=("theta_test_deg", "std"))
            .reset_index().sort_values("sigma")
        )
        ax.errorbar(
            s["sigma"], s["mean"], yerr=s["std"],
            marker="o", capsize=3,
            label=f"ou_alpha={ou_alpha}, neg={neg_mode}",
            color=cmap[i % len(cmap)],
        )
    ax.set_xlabel(r"OU noise std $\sigma$")
    ax.set_ylabel(r"$\theta$ test RMSE (deg)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(Path(outdir) / "theta_vs_sigma.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description=(
            "Sweep DVSIB with OU-correlated observation noise over "
            "ndims x num_frames x samples_n x sigma x ou_alpha x neg_mode x rep."
        )
    )
    # sweep dimensions
    p.add_argument("--ndims", nargs="+", type=int, default=[2])
    p.add_argument("--num_frames", nargs="+", type=int, default=[2])
    p.add_argument("--samples_n", nargs="+", type=int,
                   default=[2, 4, 8, 16, 32, 64, 128, 256, 512, 1000])
    p.add_argument("--sigmas", nargs="+", type=float, default=[0.1],
                   help="OU noise marginal std values to sweep")
    p.add_argument("--ou_alphas", nargs="+", type=float, default=[0.0],
                   help="OU memory values to sweep (0=white, 1=constant drift)")
    p.add_argument("--neg_modes", nargs="+", type=str, default=["all"],
                   choices=list(VALID_NEG_MODES),
                   help="InfoNCE negative sets to compare: "
                        "'all' = in-batch negatives from all trajectories, "
                        "'traj' = within-trajectory negatives only.")
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--rep_offset", type=int, default=0,
                   help="Rep indices will be rep_offset+1 .. rep_offset+reps")

    # seeds
    p.add_argument("--base_seed", type=int, default=42,
                   help="Training seed for rep 1; rep k uses base_seed + (k-1).")
    p.add_argument("--base_noise_seed", type=int, default=12345,
                   help="Base seed for the OU-noise RNG (deterministic per "
                        "(rep, sigma, ou_alpha)).")

    # training
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--gamma", type=float, default=0.01,
                   help="KL weight in DVSIB loss (compression strength)")
    p.add_argument("--lr", type=float, default=1e-4,
                   help="Adam learning rate for DVSIB training")
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--traj_sampler", default="grouped",
                   choices=["grouped", "full"],
                   help="Sampler for neg_mode='traj'. "
                        "'grouped' (default): TrajGroupedBatchSampler -- traj_batch "
                        "trajectories x k_per_traj random windows per step. "
                        "'full': DatasetTrajFull -- all numD windows per trajectory "
                        "per step, giving the maximum numD-1 within-traj negatives.")
    p.add_argument("--traj_batch", type=int, default=64,
                   help="[grouped] Trajectories per step. "
                        "Effective batch = traj_batch * k_per_traj.")
    p.add_argument("--k_per_traj", type=int, default=16,
                   help="[grouped] Windows per trajectory per step. "
                        "Effective batch = traj_batch * k_per_traj.")
    p.add_argument("--traj_chunk_size", type=int, default=0,
                   help="[full] Trajectories per DataLoader batch. "
                        "0 (default) = auto as max(1, batch_size // numD).")
    p.add_argument("--time_batch", type=int, default=0,
                   help="[time/grouped] Time points sampled per step. "
                        "0 (default) = all numD time points.")
    p.add_argument("--k_per_time", type=int, default=18,
                   help="[time/grouped] Trajectories sampled per time point. "
                        "Effective batch = time_batch * k_per_time.")
    p.add_argument("--standard_sampler", action="store_true",
                   help="Use a standard shuffled DataLoader even for "
                        "neg_mode='traj'. Disables traj_sampler entirely.")
    p.add_argument("--noise_dist", type=str, default="gaussian",
                   choices=["gaussian", "uniform"],
                   help="Noise innovation distribution. 'gaussian' (default) = "
                        "original OU noise, sigma is marginal std. "
                        "'uniform' = Uniform[0,1] OU overlay centered at 0.5, "
                        "sigma is an amplitude ceiling not a std.")
    p.add_argument("--probe_every", type=int, default=1,
                   help="Fit theta prober every this many epochs to track "
                        "representation quality during training (the paper "
                        "fits a fresh prober every epoch). "
                        "0 disables periodic probing.")
    p.add_argument("--prober_type", type=str, default="rf",
                   choices=["rf", "mlp"],
                   help="Theta prober type: 'rf' (default) = RandomFeatureProber "
                        "fitted by closed-form ridge regression (Appendix B.4); "
                        "'mlp' = trainable MLP fitted by Adam (legacy behaviour).")
    p.add_argument("--rf_expand_dim", type=int, default=512,
                   help="[rf] Random-feature projection dimension (default 512).")
    p.add_argument("--rf_alpha", type=float, default=1e-3,
                   help="[rf] Ridge regularisation strength (default 1e-3).")

    # paths
    p.add_argument("--root_dir", type=str, required=True,
                   help="Directory with single_pendulum_all and phys_vars.npy")
    p.add_argument("--outfile", type=str, default="results/results.h5")

    # control
    p.add_argument("--device", type=str, default=None, choices=["cuda", "cpu"])
    p.add_argument("--overwrite", action="store_true",
                   help="Re-run and overwrite runs already present in the HDF5 file")
    p.add_argument("--no_clip", action="store_true",
                   help="Skip pixel-range clamping after adding noise (pass clip=None)")
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    outfile = Path(args.outfile).resolve()
    outfile.parent.mkdir(parents=True, exist_ok=True)

    assert all(sn <= 1000 for sn in args.samples_n), (
        "samples_n must be <= 1000: trials 1000:1200 are the fixed test "
        "split, and training on data[:samples_n] would otherwise leak "
        f"test trajectories into training (got {args.samples_n})"
    )

    print("Loading data...")
    data_clean, phys = load_data(args.root_dir)
    print(f"  pixel data: {tuple(data_clean.shape)}")

    for rep in range(args.rep_offset + 1, args.rep_offset + args.reps + 1):
        for ndims in args.ndims:
            for num_frames in args.num_frames:
                for samples_n in args.samples_n:
                    for sigma in args.sigmas:
                        for ou_alpha in args.ou_alphas:
                            for neg_mode in args.neg_modes:
                                run_name = make_run_name(
                                    ndims, num_frames, samples_n,
                                    sigma, ou_alpha, neg_mode, rep,
                                    noise_dist=args.noise_dist,
                                    gamma=args.gamma,
                                    lr=args.lr,
                                )
                                seed = make_seed(args.base_seed, rep)
                                noise_seed = make_noise_seed(
                                    args.base_noise_seed, rep, sigma, ou_alpha
                                )

                                if _run_exists(outfile, run_name):
                                    if args.overwrite:
                                        print(f"  overwriting existing run: {run_name}")
                                        _delete_run(outfile, run_name)
                                    else:
                                        print(f"  skipping (already done): {run_name}")
                                        continue

                                (params, hist, probe_metrics,
                                 rollout_arrays, model_bytes) = run_one(
                                    data_clean=data_clean, phys=phys,
                                    ndims=ndims, num_frames=num_frames,
                                    samples_n=samples_n,
                                    sigma=sigma, ou_alpha=ou_alpha,
                                    neg_mode=neg_mode,
                                    rep=rep, seed=seed, noise_seed=noise_seed,
                                    epochs=args.epochs,
                                    batch_size=args.batch_size,
                                    gamma=args.gamma,
                                    lr=args.lr,
                                    device=device,
                                    traj_batch=args.traj_batch,
                                    k_per_traj=args.k_per_traj,
                                    traj_sampler=args.traj_sampler,
                                    traj_chunk_size=args.traj_chunk_size,
                                    time_batch=args.time_batch or None,
                                    k_per_time=args.k_per_time,
                                    standard_sampler=args.standard_sampler,
                                    noise_dist=args.noise_dist,
                                    probe_every=args.probe_every,
                                    no_clip=args.no_clip,
                                    prober_type=args.prober_type,
                                    rf_expand_dim=args.rf_expand_dim,
                                    rf_alpha=args.rf_alpha,
                                )

                                _save_run(outfile, run_name, params, hist,
                                          probe_metrics, rollout_arrays,
                                          model_bytes)
                                print(f"  saved -> {outfile}:runs/{run_name}")

    print(f"\nDone.  Output: {outfile}")
    print(f"  runs completed: {list_runs(outfile)}")

    df = load_summary(outfile)
    if df["ou_alpha"].nunique() > 1:
        plot_theta_vs_ou_alpha(df, outfile.parent)
    if df["sigma"].nunique() > 1:
        plot_theta_vs_sigma(df, outfile.parent)

    print("\nFinal summary:")
    cols = ["ndims", "num_frames", "samples_n", "sigma", "ou_alpha",
            "neg_mode", "rep", "theta_test_deg", "final_test_mi"]
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()

"""
Sweep script: supervised baseline for DVSIB on pendulum pixel data with
OU-correlated observation noise.

Trains a SupervisedModel (same Encoder architecture as DVSIB + linear head)
end-to-end on [cos(theta), sin(theta)] labels derived from noisy pixel windows.
No dynamics component, no KL term, no two-stage prober training.

Sweeps over: ndims, num_frames, samples_n, sigma, ou_alpha, rep.

Results are written to a separate HDF5 file (default supervised_results.h5)
and never touch the DVSIB results file.

HDF5 layout
-----------
supervised_results.h5
└── runs/
    └── {run_name}/
        attrs:  ndims, num_frames, samples_n, rep,
                sigma, ou_alpha,
                seed, noise_seed, epochs, lr, batch_size, input_dim,
                noise_dist
        ├── training/   { train_loss, test_loss,
                          theta_train_deg, theta_test_deg }
        └── model_weights   (uint8 1-D, serialised state_dict)

Run naming
----------
    sup_dz{ndims}_numF{num_frames}_sN{samples_n}_sig{sigma}_oua{ou_alpha}_rep{rep}

Usage example
-------------
    python run_supervised_sweep.py \\
        --root_dir /path/to/data \\
        --outfile  results/supervised_results.h5 \\
        --ndims 2 --num_frames 2 --samples_n 256 1000 \\
        --sigmas 0.0 0.1 0.2 --ou_alphas 0.0 0.8 0.95 \\
        --reps 3 --epochs 300
"""

import argparse
import io
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

from supervised_models import SupervisedModel, LabeledWindowDataset
from noise import add_ou_noise_per_trajectory, add_ou_noise_uniform
from training import wrapped_theta_rmse_deg_from_cos_sin
from run_sweep import (
    load_data, make_phys_tensors,
    make_seed, make_noise_seed, _fmt, _attr_to_py,
)


# ─────────────────────────────────────────────────────────────────────────────
# Run naming
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_LR = 1e-4


def make_run_name_sup(ndims, num_frames, samples_n, sigma, ou_alpha, rep,
                      noise_dist="gaussian", lr=_DEFAULT_LR):
    name = (
        f"sup_dz{ndims}_numF{num_frames}_sN{samples_n}"
        f"_sig{_fmt(sigma)}_oua{_fmt(ou_alpha)}_rep{rep}"
    )
    if noise_dist != "gaussian":
        name += f"_ndist{noise_dist}"
    if abs(lr - _DEFAULT_LR) > 1e-10:
        name += f"_lr{_fmt(lr)}"
    return name


# ─────────────────────────────────────────────────────────────────────────────
# HDF5 I/O
# ─────────────────────────────────────────────────────────────────────────────

def list_runs_sup(h5_path):
    """Return a sorted list of supervised run names stored in the HDF5 file."""
    h5_path = Path(h5_path)
    if not h5_path.exists():
        return []
    with h5py.File(h5_path, "r") as f:
        return sorted(f.get("runs", {}).keys())


def load_summary_sup(h5_path):
    """
    Load scalar metrics for every completed supervised run into a DataFrame.
    Columns: all run parameters + final train/test loss + final theta RMSE.
    """
    import pandas as pd

    h5_path = Path(h5_path)
    rows = []
    with h5py.File(h5_path, "r") as f:
        runs_grp = f.get("runs", {})
        for run_name, grp in runs_grp.items():
            row = {"run_name": run_name, "h5_file": str(h5_path)}
            row.update({k: _attr_to_py(v) for k, v in grp.attrs.items()})
            tr = grp["training"]
            row["final_train_loss"]     = float(tr["train_loss"][-1])
            row["final_test_loss"]      = float(tr["test_loss"][-1])
            row["final_theta_train_deg"] = float(tr["theta_train_deg"][-1])
            row["final_theta_test_deg"]  = float(tr["theta_test_deg"][-1])
            rows.append(row)
    return pd.DataFrame(rows)


def load_run_sup(h5_path, run_name):
    """Load all stored data for a single supervised run."""
    with h5py.File(h5_path, "r") as f:
        grp = f[f"runs/{run_name}"]
        return {
            "params":   {k: _attr_to_py(v) for k, v in grp.attrs.items()},
            "training": {k: grp[f"training/{k}"][:] for k in grp["training"]},
        }


def load_model_sup(h5_path, run_name, device="cpu"):
    """Reconstruct a trained SupervisedModel from serialised weights."""
    with h5py.File(h5_path, "r") as f:
        grp = f[f"runs/{run_name}"]
        params = {k: _attr_to_py(v) for k, v in grp.attrs.items()}
        raw = grp["model_weights"][:]

    buf = io.BytesIO(raw.tobytes())
    state_dict = torch.load(buf, map_location=device)

    model = SupervisedModel(
        input_dim=int(params["input_dim"]),
        dz=int(params["ndims"]),
        dzf=32,
        num_frames=int(params["num_frames"]),
        hidden=256,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _run_exists_sup(h5_path, run_name):
    if not Path(h5_path).exists():
        return False
    with h5py.File(h5_path, "a") as f:
        return f"runs/{run_name}" in f


def _delete_run_sup(h5_path, run_name):
    with h5py.File(h5_path, "a") as f:
        key = f"runs/{run_name}"
        if key in f:
            del f[key]


def _save_run_sup(h5_path, run_name, params, hist, model_state_bytes):
    Path(h5_path).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(h5_path, "a") as f:
        grp = f.require_group(f"runs/{run_name}")
        for k, v in params.items():
            grp.attrs[k] = v
        tr = grp.require_group("training")
        for key, arr in hist.items():
            tr.create_dataset(key, data=np.asarray(arr, dtype=np.float32))
        grp.create_dataset(
            "model_weights",
            data=np.frombuffer(model_state_bytes, dtype=np.uint8),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_supervised(model, train_loader, x_test, labels_test,
                     epochs=300, lr=1e-4, device="cuda"):
    """
    Train a SupervisedModel end-to-end with MSE loss on [cos(theta), sin(theta)].

    Args:
        model        : SupervisedModel (already on device)
        train_loader : DataLoader yielding (past_window, cos_sin_label)
        x_test       : (N, num_frames, D) test windows (on device)
        labels_test  : (N, 2) test [cos, sin] labels (on device)
        epochs       : training epochs
        lr           : Adam learning rate
        device       : torch device string

    Returns:
        model : trained model
        hist  : dict with per-epoch lists:
                  train_loss, test_loss, theta_train_deg, theta_test_deg
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist = {
        "train_loss": [], "test_loss": [],
        "theta_train_deg": [], "theta_test_deg": [],
    }

    # Precompute true theta from test labels for RMSE reporting
    theta_test_true = torch.atan2(labels_test[:, 1], labels_test[:, 0]).cpu()

    for epoch in (pbar := tqdm(range(epochs))):
        model.train()
        run_loss = 0.0
        n_batches = 0
        train_preds = []
        train_labels = []

        for x, lbl in train_loader:
            x   = x.to(device)
            lbl = lbl.to(device)
            opt.zero_grad()
            pred = model(x)
            loss = torch.mean(torch.sum((lbl - pred).pow(2), dim=1))

            if torch.isnan(loss):
                print(f"NaN loss at epoch {epoch + 1}; stopping.")
                return model, hist

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            run_loss += loss.item()
            n_batches += 1
            train_preds.append(pred.detach().cpu())
            train_labels.append(lbl.cpu())

        model.eval()
        with torch.no_grad():
            test_pred = model(x_test)
            test_loss = torch.mean(
                torch.sum((labels_test - test_pred).pow(2), dim=1)
            ).item()

        tr_loss = run_loss / n_batches

        # Theta RMSE from [cos, sin] predictions via arctan2
        all_train_pred   = torch.cat(train_preds,  dim=0)
        all_train_labels = torch.cat(train_labels, dim=0)
        theta_train_true = torch.atan2(all_train_labels[:, 1], all_train_labels[:, 0])
        theta_train_deg  = wrapped_theta_rmse_deg_from_cos_sin(all_train_pred, theta_train_true)

        test_pred_cpu    = test_pred.cpu()
        theta_test_deg   = wrapped_theta_rmse_deg_from_cos_sin(test_pred_cpu, theta_test_true)

        hist["train_loss"].append(tr_loss)
        hist["test_loss"].append(test_loss)
        hist["theta_train_deg"].append(theta_train_deg)
        hist["theta_test_deg"].append(theta_test_deg)

        pbar.set_postfix({
            "loss_tr": f"{tr_loss:.4f}",
            "loss_ts": f"{test_loss:.4f}",
            "θ_ts°":   f"{theta_test_deg:.2f}",
        })

    return model, hist


# ─────────────────────────────────────────────────────────────────────────────
# Single-run computation
# ─────────────────────────────────────────────────────────────────────────────

def run_one_supervised(*, data_clean, phys, ndims, num_frames, samples_n,
                       sigma, ou_alpha, rep, seed, noise_seed,
                       epochs, batch_size, lr, device,
                       noise_dist="gaussian", no_clip=False):
    """
    Execute one supervised sweep point:
        1. Set seeds.
        2. Add OU noise to the pixel tensor.
        3. Build LabeledWindowDatasets from noisy pixels + clean theta labels.
        4. Train SupervisedModel.
        5. Return params, history, and serialised model weights.
    """
    run_name = make_run_name_sup(
        ndims, num_frames, samples_n, sigma, ou_alpha, rep,
        noise_dist=noise_dist, lr=lr,
    )
    print(f"\n{'='*60}\n{run_name}  (seed={seed}, noise_seed={noise_seed})\n{'='*60}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    T = data_clean.shape[1]
    D = data_clean.shape[2]
    numD = T - 2 * num_frames + 1

    # Apply OU noise
    _noise_fn = (add_ou_noise_uniform if noise_dist == "uniform"
                 else add_ou_noise_per_trajectory)
    data_noisy = _noise_fn(
        data_clean, sigma=sigma, ou_alpha=ou_alpha,
        clip=None if no_clip else (0.0, 1.0), seed=noise_seed, init="stationary",
    )

    # Ground-truth theta for the labelled windows.
    # LabeledWindowDataset accesses theta[trial, t] for t in 0..numD-1 only,
    # so passing the full per-trial theta array is safe and correct.
    theta_all = torch.from_numpy(phys["theta"]).float()   # (1200, T)

    train_ds = LabeledWindowDataset(data_noisy[:samples_n], theta_all[:samples_n], num_frames)
    test_ds  = LabeledWindowDataset(data_noisy[1000:],      theta_all[1000:],      num_frames)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True
    )

    # Full test batch (all test windows at once for eval)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=len(test_ds), shuffle=False
    )
    x_test, labels_test = next(iter(test_loader))
    x_test      = x_test.to(device)
    labels_test = labels_test.to(device)

    model = SupervisedModel(
        input_dim=D, dz=ndims, dzf=32, num_frames=num_frames, hidden=256,
    ).to(device)

    model, hist = train_supervised(
        model, train_loader, x_test, labels_test,
        epochs=epochs, lr=lr, device=device,
    )

    params = {
        "ndims":       ndims,
        "num_frames":  num_frames,
        "samples_n":   samples_n,
        "rep":         rep,
        "sigma":       float(sigma),
        "ou_alpha":    float(ou_alpha),
        "seed":        seed,
        "noise_seed":  noise_seed,
        "epochs":      epochs,
        "lr":          lr,
        "batch_size":  batch_size,
        "input_dim":   D,
        "noise_dist":  noise_dist,
    }

    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    model_state_bytes = buf.getvalue()

    return run_name, params, hist, model_state_bytes


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description=(
            "Sweep supervised baseline (same Encoder as DVSIB + linear head) "
            "over ndims x num_frames x samples_n x sigma x ou_alpha x rep."
        )
    )

    # sweep dimensions
    p.add_argument("--ndims", nargs="+", type=int, default=[2])
    p.add_argument("--num_frames", nargs="+", type=int, default=[2])
    p.add_argument("--samples_n", nargs="+", type=int,
                   default=[2, 4, 8, 16, 32, 64, 128, 256, 512, 1000])
    p.add_argument("--sigmas", nargs="+", type=float, default=[0.1])
    p.add_argument("--ou_alphas", nargs="+", type=float, default=[0.0])
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--rep_offset", type=int, default=0)

    # seeds
    p.add_argument("--base_seed", type=int, default=42)
    p.add_argument("--base_noise_seed", type=int, default=12345)

    # training
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--noise_dist", type=str, default="gaussian",
                   choices=["gaussian", "uniform"])

    # paths
    p.add_argument("--root_dir", type=str, required=True)
    p.add_argument("--outfile", type=str, default="results/supervised_results.h5")

    # control
    p.add_argument("--device", type=str, default=None, choices=["cuda", "cpu"])
    p.add_argument("--overwrite", action="store_true")
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
                            run_name = make_run_name_sup(
                                ndims, num_frames, samples_n,
                                sigma, ou_alpha, rep,
                                noise_dist=args.noise_dist,
                                lr=args.lr,
                            )
                            seed = make_seed(args.base_seed, rep)
                            noise_seed = make_noise_seed(
                                args.base_noise_seed, rep, sigma, ou_alpha
                            )

                            if _run_exists_sup(outfile, run_name):
                                if args.overwrite:
                                    print(f"  overwriting existing run: {run_name}")
                                    _delete_run_sup(outfile, run_name)
                                else:
                                    print(f"  skipping (already done): {run_name}")
                                    continue

                            run_name, params, hist, model_bytes = run_one_supervised(
                                data_clean=data_clean, phys=phys,
                                ndims=ndims, num_frames=num_frames,
                                samples_n=samples_n,
                                sigma=sigma, ou_alpha=ou_alpha,
                                rep=rep, seed=seed, noise_seed=noise_seed,
                                epochs=args.epochs,
                                batch_size=args.batch_size,
                                lr=args.lr,
                                device=device,
                                noise_dist=args.noise_dist,
                                no_clip=args.no_clip,
                            )

                            _save_run_sup(outfile, run_name, params, hist, model_bytes)
                            print(f"  saved -> {outfile}:runs/{run_name}")

    print(f"\nDone.  Output: {outfile}")
    print(f"  runs completed: {list_runs_sup(outfile)}")


if __name__ == "__main__":
    main()

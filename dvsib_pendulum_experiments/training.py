"""
Training and evaluation for DVSIB on pendulum dynamics with OU-noisy pixels.

Functions:
    train_dvsib                  -- train a DVSIB model, return (model, history dict)
    train_prober                 -- train a Prober with MSE loss, return (prober, train_mse, test_mse)
    fit_ridge_head                -- fit RandomFeatureProber head via closed-form ridge regression
    predict_prober               -- run a trained prober in mini-batches
    wrapped_theta_rmse_deg_from_cos_sin
                                 -- wrapped angular RMSE from [cos(theta), sin(theta)] predictions
    encode_dataset               -- encode all (trial, window) pairs, return latent codes
    rollout_latent               -- greedy + stochastic rollout in latent space
    rollout_rms                  -- physical-variable RMS along a rollout vs ground truth
    rollout_theta_rmse_wrapped   -- wrapped angular RMSE along a rollout for theta

The rollout uses `model.predict(z) -> (next_mean, next_logvar)` so it works
unchanged across all DVSIB objective versions (residual / delta / direct /
intermediate).  The decoder interpretation is owned by the model.
"""

import numpy as np
import torch
from tqdm import tqdm

from models import FrameDataset, ProbeDataset


# ─────────────────────────────────────────────────────────────────────────────
# DVSIB training
# ─────────────────────────────────────────────────────────────────────────────

def train_dvsib(model, loader, x_test, y_test, tid_test, tmi_test=None,
                epochs=300, lr=1e-4, device="cuda",
                probe_fn=None, probe_every=5):
    """
    Train a DVSIB model.

    Args:
        model             : DVSIB instance (already on device)
        loader            : DataLoader yielding (past, future, trial_idx, time_idx)
        x_test, y_test    : test windows (already on device) for per-epoch eval
        tid_test          : test trial-index tensor (on device)
        tmi_test          : test time-index tensor (on device); required when
                            model.neg_mode == 'time', ignored otherwise
        epochs            : number of training epochs
        lr                : Adam learning rate
        device            : torch device string
        probe_fn          : optional callable(model) -> {"theta_train_deg": float,
                            "theta_test_deg": float}. Called every probe_every epochs.
        probe_every       : interval (in epochs) at which probe_fn is called.

    Returns:
        model : trained model
        hist  : dict with per-epoch lists
                  train_loss, train_kl, train_mi, test_mi
                and per-checkpoint lists (length epochs // probe_every):
                  probe_epochs, probe_theta_train_deg, probe_theta_test_deg
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist = {
        "train_loss": [], "train_kl": [], "train_mi": [], "test_mi": [],
        "probe_epochs": [], "probe_theta_train_deg": [], "probe_theta_test_deg": [],
    }

    if probe_fn is not None:
        probe_result = probe_fn(model)
        hist["probe_epochs"].append(0.0)
        hist["probe_theta_train_deg"].append(probe_result["theta_train_deg"])
        hist["probe_theta_test_deg"].append(probe_result["theta_test_deg"])

    for epoch in (pbar := tqdm(range(epochs))):
        model.train()
        run_loss = run_kl = run_mi = 0.0
        n_batches = 0

        for x, y, tid, tmi in loader:
            x   = x.to(device)
            y   = y.to(device)
            tid = tid.to(device)
            tmi = tmi.to(device)
            opt.zero_grad()
            loss, kl, mi_term, mi = model(x, y, trial_idx=tid, time_idx=tmi)

            if torch.isnan(loss):
                print(f"NaN loss at epoch {epoch + 1}; stopping.")
                return model, hist

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            run_loss += loss.item()
            run_kl   += kl.item()
            run_mi   += mi.item()
            n_batches += 1

        model.eval()
        with torch.no_grad():
            _, _, _, test_mi = model(x_test, y_test,
                                     trial_idx=tid_test, time_idx=tmi_test)

        tr_loss = run_loss / n_batches
        tr_kl   = run_kl   / n_batches
        tr_mi   = run_mi   / n_batches
        ts_mi   = test_mi.item()

        hist["train_loss"].append(tr_loss)
        hist["train_kl"].append(tr_kl)
        hist["train_mi"].append(tr_mi)
        hist["test_mi"].append(ts_mi)

        if probe_fn is not None and (epoch + 1) % probe_every == 0:
            probe_result = probe_fn(model)
            hist["probe_epochs"].append(float(epoch + 1))
            hist["probe_theta_train_deg"].append(probe_result["theta_train_deg"])
            hist["probe_theta_test_deg"].append(probe_result["theta_test_deg"])

        pbar.set_postfix({
            "loss":  f"{tr_loss:.3f}",
            "MI_tr": f"{tr_mi:.3f}",
            "MI_ts": f"{ts_mi:.3f}",
        })

    return model, hist


# ─────────────────────────────────────────────────────────────────────────────
# Prober training
# ─────────────────────────────────────────────────────────────────────────────

def train_prober(prober, z_train, y_train, z_test, y_test,
                 epochs=100, lr=1e-4, batch_size=1024, device="cuda"):
    """
    Train a Prober to predict a physical variable from latent codes.
    The DVSIB model is NOT touched here; z_train / z_test are precomputed.
    """
    loader = torch.utils.data.DataLoader(
        ProbeDataset(z_train, y_train), batch_size=batch_size, shuffle=True
    )
    z_test = z_test.to(device)
    y_test = y_test.to(device)
    opt = torch.optim.Adam(prober.parameters(), lr=lr)

    tr_mse = float("nan")
    for _ in range(epochs):
        prober.train()
        run_loss = 0.0
        n = 0
        for z, y in loader:
            z, y = z.to(device), y.to(device)
            opt.zero_grad()
            loss = torch.mean(torch.sum((y - prober(z)).pow(2), dim=1))
            if torch.isnan(loss):
                break
            loss.backward()
            opt.step()
            run_loss += loss.item()
            n += 1
        if n > 0:
            tr_mse = run_loss / n

    prober.eval()
    with torch.no_grad():
        ts_mse = torch.mean(
            torch.sum((y_test - prober(z_test)).pow(2), dim=1)
        ).item()

    return prober, tr_mse, ts_mse


def fit_ridge_head(prober, zx_train, y_train, alpha=1e-3,
                   feat_batch=4096, device="cuda"):
    """
    Fit the linear head of a RandomFeatureProber via closed-form ridge regression.

    The frozen expansion weights are never touched. Normal equations are
    accumulated in float64 mini-batches so the full feature matrix (N x d) is
    never materialised. Bias is fitted without regularisation via centering:

        W = (XtX_c + alpha*I)^-1 Xty_c
        b = ybar - Xbar @ W

    where Xbar / ybar are the column means of features / targets.

    Args:
        prober    : RandomFeatureProber (already on device is fine; inputs moved
                    to device inside the function)
        zx_train  : (N, dz) float tensor of latent codes (CPU or device)
        y_train   : (N, output_dim) float tensor of targets (CPU or device)
        alpha     : L2 regularisation strength on weights (default 1e-3)
        feat_batch: mini-batch size for Gram matrix accumulation
        device    : torch device string

    Modifies prober.head.weight and prober.head.bias in-place.
    Returns the prober (modified in-place, for chaining).
    """
    prober.eval()
    expand_dim = prober.expand.out_features
    output_dim = prober.head.out_features

    XtX   = torch.zeros(expand_dim, expand_dim, dtype=torch.float64, device=device)
    Xty   = torch.zeros(expand_dim, output_dim, dtype=torch.float64, device=device)
    X_sum = torch.zeros(expand_dim,             dtype=torch.float64, device=device)
    y_sum = torch.zeros(output_dim,             dtype=torch.float64, device=device)
    n     = 0

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(zx_train, y_train),
        batch_size=feat_batch, shuffle=False,
    )

    with torch.no_grad():
        for z_b, y_b in loader:
            z_b  = z_b.to(device)
            y_b  = y_b.to(device)
            feat  = prober.act(prober.expand(z_b)).double()
            y_b   = y_b.double()
            XtX  += feat.T @ feat
            Xty  += feat.T @ y_b
            X_sum += feat.sum(0)
            y_sum += y_b.sum(0)
            n     += feat.shape[0]

    X_mean = X_sum / n
    y_mean = y_sum / n
    XtX_c  = XtX - n * torch.outer(X_mean, X_mean)
    Xty_c  = Xty - n * torch.outer(X_mean, y_mean)

    reg = alpha * torch.eye(expand_dim, dtype=torch.float64, device=device)
    W   = torch.linalg.solve(XtX_c + reg, Xty_c)   # (expand_dim, output_dim)
    b   = y_mean - X_mean @ W                        # (output_dim,)

    prober.head.weight.data = W.T.float()
    prober.head.bias.data   = b.float()
    return prober


def predict_prober(prober, z, batch_size=4096, device="cuda"):
    """Run a trained prober on latent codes in mini-batches."""
    prober.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, z.shape[0], batch_size):
            zb = z[start: start + batch_size].to(device)
            parts.append(prober(zb).cpu())
    return torch.cat(parts, dim=0)


def wrapped_theta_rmse_deg_from_cos_sin(pred_cos_sin, true_theta):
    """
    Convert predicted [cos(theta), sin(theta)] vectors to angles and compute
    wrapped angular RMSE in degrees.
    """
    pred_cos_sin = torch.as_tensor(pred_cos_sin).float()
    true_theta = torch.as_tensor(true_theta).float().reshape(-1)

    pred_theta = torch.atan2(pred_cos_sin[:, 1], pred_cos_sin[:, 0])
    dtheta = torch.atan2(
        torch.sin(pred_theta - true_theta),
        torch.cos(pred_theta - true_theta),
    )
    return float(torch.rad2deg(torch.sqrt(torch.mean(dtheta.pow(2)))).item())


# ─────────────────────────────────────────────────────────────────────────────
# Encoding
# ─────────────────────────────────────────────────────────────────────────────

def encode_dataset(model, data, num_frames, device="cuda", batch_size=4096):
    """
    Encode all (trial, window) pairs from frame data through the DVSIB encoder.
    Returns the posterior MEAN (not a sample) for downstream probing / rollout.

    Returns:
        zx   : (n_trials * numD, dz) CPU tensor  — mean latent codes, ordered
               by (trial, window) with trial as the slow index
        numD : number of windows per trial = T - 2*num_frames + 1
    """
    numD = data.shape[1] - 2 * num_frames + 1
    dataset = FrameDataset(data, num_frames)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False
    )

    model.eval()
    zx_parts = []
    with torch.no_grad():
        for x, *_ in loader:
            mean_z, _, _ = model.encoder(x.to(device))
            zx_parts.append(mean_z.cpu())

    return torch.cat(zx_parts, dim=0), numD


# ─────────────────────────────────────────────────────────────────────────────
# Latent rollout
# ─────────────────────────────────────────────────────────────────────────────

def rollout_latent(model, zx, num_frames, numD, device="cuda",
                   num_repeats=40, extra_steps=0):
    """
    Roll out the latent dynamics forward in time from the first window of each
    test trajectory, comparing against the ground-truth latent codes.

    The step is delegated to `model.predict(z) -> (next_mean, next_logvar)` so
    this routine is version-agnostic (residual / delta / direct / intermediate).

    Greedy rollout  : z_{t+1} = next_mean(z_t)
    Stochastic      : z_{t+1} = next_mean(z_t) + exp(0.5 * next_logvar) * eps

    Returns dict with:
        what_frames  : (n_steps+1,) int — frame index of each rollout step
        greedy_path  : (n_steps+1, n_trials, dz)
        pred_mean    : (n_steps+1, n_trials, dz) — mean over stochastic samples
        pred_std     : (n_steps+1, n_trials, dz) — std  over stochastic samples
        delta_greedy : (n_compare,) latent-space RMS per step, greedy
        delta_pred   : (n_compare,) latent-space RMS per step, stochastic mean
    """
    model.eval()
    n_trials = zx.shape[0] // numD
    dz = zx.shape[1]
    n_steps = numD // num_frames + extra_steps

    what_frames = np.arange(n_steps + 1) * num_frames

    truth = zx.reshape(n_trials, numD, dz)[:, ::num_frames, :].numpy()

    start = zx[::numD, :].to(device)
    greedy_steps = [start.cpu().numpy()]

    s = start.unsqueeze(1).expand(-1, num_repeats, -1).reshape(-1, dz).to(device)
    stoch_steps = [s.cpu().numpy().reshape(n_trials, num_repeats, dz)]

    g = start.clone()

    with torch.no_grad():
        for _ in range(n_steps):
            g_next_mean, _ = model.predict(g)
            g = g_next_mean
            greedy_steps.append(g.cpu().numpy())

            s_next_mean, s_next_logvar = model.predict(s)
            noise = torch.randn_like(s)
            s = s_next_mean + (0.5 * s_next_logvar).exp() * noise
            stoch_steps.append(s.cpu().numpy().reshape(n_trials, num_repeats, dz))

    greedy = np.stack(greedy_steps, axis=0)
    stoch = np.stack(stoch_steps, axis=0)

    pred_mean = stoch.mean(axis=2)
    pred_std = stoch.std(axis=2)

    n_compare = min(n_steps + 1, truth.shape[1])
    truth_c = truth[:, :n_compare, :]
    greedy_c = greedy[:n_compare].transpose(1, 0, 2)
    pred_c = pred_mean[:n_compare].transpose(1, 0, 2)

    delta_greedy = np.mean(
        np.sqrt(np.sum((truth_c - greedy_c) ** 2, axis=2)), axis=0
    )
    delta_pred = np.mean(
        np.sqrt(np.sum((truth_c - pred_c) ** 2, axis=2)), axis=0
    )

    return {
        "what_frames":  what_frames[:n_compare],
        "greedy_path":  greedy,
        "pred_mean":    pred_mean,
        "pred_std":     pred_std,
        "delta_greedy": delta_greedy,
        "delta_pred":   delta_pred,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Physical-variable RMS along a rollout
# ─────────────────────────────────────────────────────────────────────────────

def rollout_rms(prober, rollout, target_flat, numD, num_frames, device="cuda"):
    """Per-step RMSE of a scalar physical variable along a rollout."""
    prober.eval()
    greedy_path = rollout["greedy_path"]
    pred_mean = rollout["pred_mean"]
    n_steps1, n_trials, dz = greedy_path.shape

    truth_2d = target_flat.reshape(n_trials, numD)[:, ::num_frames].numpy()
    n_compare = min(n_steps1, truth_2d.shape[1])

    def _decode(path):
        z = torch.from_numpy(
            path[:n_compare].reshape(-1, dz)
        ).float().to(device)
        with torch.no_grad():
            vals = prober(z).cpu().numpy()
        return vals.reshape(n_compare, n_trials)

    gv = _decode(greedy_path)
    pv = _decode(pred_mean)

    truth_c = truth_2d[:, :n_compare].T
    rms_greedy = np.sqrt(np.mean((truth_c - gv) ** 2, axis=1))
    rms_pred = np.sqrt(np.mean((truth_c - pv) ** 2, axis=1))
    return rms_greedy, rms_pred


def rollout_theta_rmse_wrapped(
    prober, rollout, target_flat, numD, num_frames, batch_size=4096, device="cuda"
):
    """Per-step wrapped angular RMSE (in degrees) along a rollout."""
    prober.eval()
    greedy_path = rollout["greedy_path"]
    pred_mean = rollout["pred_mean"]
    n_steps1, n_trials, dz = greedy_path.shape

    truth_theta = torch.as_tensor(target_flat).float() \
                    .reshape(n_trials, numD)[:, ::num_frames]
    n_compare = min(n_steps1, truth_theta.shape[1])
    truth_theta = truth_theta[:, :n_compare].T

    def _decode_theta(path):
        z = torch.from_numpy(path[:n_compare].reshape(-1, dz)).float()
        pred_xy = predict_prober(prober, z, batch_size=batch_size, device=device)
        pred_theta = torch.atan2(pred_xy[:, 1], pred_xy[:, 0])
        return pred_theta.reshape(n_compare, n_trials)

    theta_g = _decode_theta(greedy_path)
    theta_p = _decode_theta(pred_mean)

    dtheta_g = torch.atan2(
        torch.sin(theta_g - truth_theta),
        torch.cos(theta_g - truth_theta),
    )
    dtheta_p = torch.atan2(
        torch.sin(theta_p - truth_theta),
        torch.cos(theta_p - truth_theta),
    )

    rms_greedy = torch.rad2deg(torch.sqrt(torch.mean(dtheta_g.pow(2), dim=1))).cpu().numpy()
    rms_pred = torch.rad2deg(torch.sqrt(torch.mean(dtheta_p.pow(2), dim=1))).cpu().numpy()
    return rms_greedy, rms_pred

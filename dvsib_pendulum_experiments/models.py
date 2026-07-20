"""
Neural network architectures for DVSIB on pendulum dynamics with OU-correlated
observation noise.

Classes:
    Encoder        -- per-frame MLP VAE encoder
    DeltaDecoder   -- predicts (mean_delta, logvar) from zx
    DVSIB          -- shared encoder + delta decoder + NCE MI estimator
                      with a `neg_mode` switch selecting the InfoNCE negatives
    Prober              -- small MLP to decode a physical variable from z (legacy)
    RandomFeatureProber -- random-feature linear probe fitted by ridge regression
    FrameDataset        -- sliding (past_window, future_window, trial_idx) triples
    ProbeDataset        -- (z, y) regression dataset for prober training

InfoNCE negative modes
----------------------
`neg_mode` controls which batch entries contribute as negatives to the
Gaussian InfoNCE denominator, holding the residual predictive objective
`pred_mean = zx + delta` fixed:

    "all"   -- standard InfoNCE: every other entry in the batch is a negative.
    "traj"  -- within-trajectory negatives only: for row i, the denominator
               sums over batch entries j whose trial index matches trial(i).
               This forces the critic to distinguish different *timepoints of
               the same trajectory* rather than different trajectories, which
               is the harder task when trajectories differ mostly by OU noise
               realisation or by a slow drift.
    "time"  -- cross-trajectory negatives only: for row i, the denominator
               sums over batch entries j whose time index matches time(i).
               This forces the critic to distinguish *different trajectories
               at the same time step*.

All modes use the identical encoder, decoder, and residual `zx + delta`
predictor. Only the score-matrix masking differs.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


LOG2PI = 1.8378770664093453

VALID_NEG_MODES = ("all", "traj", "time")


class Encoder(nn.Module):
    """
    Per-frame MLP encoder.

    Each of the num_frames input frames is passed through the same 3-layer MLP
    independently (Linear layers broadcast over the frame dimension). The
    per-frame embeddings are concatenated, then two linear heads produce the
    mean and log-variance of the VAE latent code z ~ N(mean, exp(logvar)).
    """

    def __init__(self, dz, dzf, input_dim, num_frames, hidden=256):
        super().__init__()
        self.dz = dz
        self.dzf = dzf
        self.num_frames = num_frames
        self.flat_dim = dzf * num_frames

        self.dense1 = nn.Linear(input_dim, hidden)
        self.dense2 = nn.Linear(hidden, hidden)
        self.dense3 = nn.Linear(hidden, dzf)

        self.mean_head = nn.Linear(self.flat_dim, dz)
        self.logvar_head = nn.Linear(self.flat_dim, dz)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        x : (batch, num_frames, D)
        returns (mean_z, logvar_z, z_sample) each of shape (batch, dz)
        """
        h = F.relu(self.dense1(x))
        h = F.relu(self.dense2(h))
        h = F.relu(self.dense3(h)).reshape(x.shape[0], -1)
        mean_z = self.mean_head(h)
        logvar_z = self.logvar_head(h)
        eps = torch.randn_like(mean_z)
        z = mean_z + (0.5 * logvar_z).exp() * eps
        return mean_z, logvar_z, z


class DeltaDecoder(nn.Module):
    """
    Latent dynamics decoder: zx -> (mean_delta, logvar).
    Interpreted as a residual prediction z_future ≈ zx + mean_delta.
    """

    def __init__(self, dz, hdim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dz, hdim), nn.ReLU(),
            nn.Linear(hdim, hdim), nn.ReLU(),
            nn.Linear(hdim, hdim), nn.ReLU(),
        )
        self.mean_head = nn.Linear(hdim, dz)
        self.logvar_head = nn.Linear(hdim, dz)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z):
        h = self.mlp(z)
        return self.mean_head(h), self.logvar_head(h)


class DVSIB(nn.Module):
    """
    Deep Variational Sufficient Information Bottleneck for frame-sequence data.

    Residual predictive objective (fixed):
        pred_mean = zx + delta       (delta ~ DeltaDecoder(zx))

    Loss (minimise):
        loss = gamma * (KL(q(z|X)) + KL(q(z|Y))) - alpha * I_NCE

    The InfoNCE bound is
        I_NCE = mean_i [ scores[i, i] - logsumexp_{j in S(i)} scores[i, j]
                                      + log |S(i)| ]
    with a Gaussian critic
        scores[i, j] = log N(zy_j ; pred_mean_i, exp(pred_logvar_i)).

    The negative set S(i) is controlled by `neg_mode`:
        "all"  : S(i) = {0, ..., batch-1}
        "traj" : S(i) = { j : trial(j) == trial(i) }
                 (requires a trial-index tensor to be passed to forward)
        "time" : S(i) = { j : time(j) == time(i) }
                 (requires a time-index tensor to be passed to forward)

    Args:
        input_dim    : per-frame pixel dimension
        dz           : latent dimension
        dzf          : per-frame embedding size inside encoder
        num_frames   : frames per window
        gamma        : weight on the KL compression term (recommend 0.01)
        alpha        : weight on the MI maximisation term (typically 1.0)
        hidden       : encoder MLP hidden width
        decoder_hdim : DeltaDecoder hidden width
        neg_mode     : "all", "traj", or "time"
    """

    def __init__(
        self,
        input_dim,
        dz=2,
        dzf=32,
        num_frames=2,
        gamma=0.01,
        alpha=1.0,
        hidden=256,
        decoder_hdim=64,
        neg_mode="all",
    ):
        super().__init__()
        if neg_mode not in VALID_NEG_MODES:
            raise ValueError(
                f"neg_mode must be one of {VALID_NEG_MODES}, got {neg_mode!r}"
            )

        self.encoder = Encoder(
            dz=dz, dzf=dzf, input_dim=input_dim,
            num_frames=num_frames, hidden=hidden,
        )
        self.delta_decoder = DeltaDecoder(dz=dz, hdim=decoder_hdim)
        self.gamma = gamma
        self.alpha = alpha
        self.neg_mode = neg_mode

    def _score_matrix(self, zy, pred_mean, pred_logvar):
        """
        Gaussian log-critic score matrix.

        scores[i, j] = log N(zy_j ; pred_mean_i, exp(pred_logvar_i))
                     = -0.5 * sum_d [ (zy_{j,d} - mu_{i,d})^2 / exp(lv_{i,d})
                                      + lv_{i,d} + log(2*pi) ]
        Returns a (batch, batch) tensor.
        """
        diff2 = (zy[None, :, :] - pred_mean[:, None, :]).pow(2)     # (B, B, dz)
        inv_var = torch.exp(-pred_logvar)[:, None, :]               # (B, 1, dz)
        lv = pred_logvar[:, None, :]                                # (B, 1, dz)
        return -0.5 * (diff2 * inv_var + lv + LOG2PI).sum(dim=2)    # (B, B)

    def _info_all(self, scores):
        """Standard InfoNCE: every in-batch entry is a negative."""
        batch = scores.shape[0]
        return (
            scores.diagonal()
            - scores.logsumexp(dim=1)
            + torch.log(torch.tensor(float(batch), device=scores.device))
        ).mean()

    def _info_traj_mask(self, scores, trial_idx):
        """
        Within-trajectory InfoNCE: for row i, the denominator sums only over
        batch entries j with trial_idx[j] == trial_idx[i]. The `log |S(i)|`
        term uses the per-row count so I remains a valid lower bound on the
        within-trajectory mutual information.
        """
        mask = trial_idx[:, None] == trial_idx[None, :]                     # (B, B)
        neg_inf = torch.finfo(scores.dtype).min
        masked = scores.masked_fill(~mask, neg_inf)
        log_denom = masked.logsumexp(dim=1)                                 # (B,)
        counts = mask.sum(dim=1).clamp_min(1).to(scores.dtype)              # (B,)
        return (scores.diagonal() - log_denom + torch.log(counts)).mean()

    def _info_time_mask(self, scores, trial_idx, time_idx):
        """
        Cross-trajectory, same-timepoint InfoNCE: for row i, the denominator
        sums over batch entries j where time_idx[j] == time_idx[i] (same window
        position) but trial_idx[j] != trial_idx[i] (different trajectory).
        The diagonal (positive) is always included.

        Rows where no other trajectory shares the same time point (count == 1
        after excluding self) contribute zero to the MI estimate -- the
        `clamp_min(1)` on counts prevents log(0) and those rows cancel out.
        """
        same_time  = time_idx[:, None] == time_idx[None, :]    # (B, B)
        diff_trial = trial_idx[:, None] != trial_idx[None, :]  # (B, B)
        mask = same_time & diff_trial
        mask.fill_diagonal_(True)                               # always include positive
        neg_inf = torch.finfo(scores.dtype).min
        masked = scores.masked_fill(~mask, neg_inf)
        log_denom = masked.logsumexp(dim=1)                     # (B,)
        counts = mask.sum(dim=1).clamp_min(1).to(scores.dtype) # (B,)
        return (scores.diagonal() - log_denom + torch.log(counts)).mean()

    def predict(self, zx):
        """
        Next-latent (mean, logvar) for rollout. Since the only objective is
        residual, predict is always zx + delta.
        """
        delta, logvar = self.delta_decoder(zx)
        return zx + delta, logvar

    def forward(self, x, y, trial_idx=None, time_idx=None):
        """
        x, y      : (batch, num_frames, D) -- past and future windows
        trial_idx : (batch,) long tensor -- trajectory index per window
                    Required for neg_mode 'traj' and 'time'.
        time_idx  : (batch,) long tensor -- window-start position per window
                    Required for neg_mode 'time'.

        Returns (loss, kl_term, mi_term, mi_estimate).
        """
        mean_zx, logvar_zx, zx = self.encoder(x)
        kl_x = torch.mean(torch.sum(
            -0.5 * (1 + logvar_zx - mean_zx.pow(2) - logvar_zx.exp()), dim=1
        ))
        mean_zy, logvar_zy, zy = self.encoder(y)
        kl_y = torch.mean(torch.sum(
            -0.5 * (1 + logvar_zy - mean_zy.pow(2) - logvar_zy.exp()), dim=1
        ))

        delta, delta_logvar = self.delta_decoder(zx)
        pred_mean = zx + delta

        scores = self._score_matrix(zy, pred_mean, delta_logvar)

        if self.neg_mode == "all":
            mi = self._info_all(scores)
        elif self.neg_mode == "traj":
            if trial_idx is None:
                raise ValueError(
                    "neg_mode='traj' requires `trial_idx` to be passed to forward()."
                )
            mi = self._info_traj_mask(scores, trial_idx)
        else:  # "time"
            if trial_idx is None or time_idx is None:
                raise ValueError(
                    "neg_mode='time' requires both `trial_idx` and `time_idx`."
                )
            mi = self._info_time_mask(scores, trial_idx, time_idx)

        kl = self.gamma * (kl_x + kl_y)
        mi_term = self.alpha * mi
        loss = kl - mi_term
        return loss, kl, mi_term, mi


class Prober(nn.Module):
    """Small MLP that decodes a scalar physical variable from a latent code (legacy)."""

    def __init__(self, dz, output_dim=1, hdim=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dz, hdim),   nn.ReLU(),
            nn.Linear(hdim, hdim), nn.ReLU(),
            nn.Linear(hdim, hdim), nn.ReLU(),
            nn.Linear(hdim, output_dim),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z):
        return self.net(z)


class RandomFeatureProber(nn.Module):
    """
    Random-feature linear probe.

    Projects z into a high-dimensional space with a single *fixed* (untrained)
    linear layer followed by a nonlinearity, then trains only a linear head on
    top via closed-form ridge regression (see `fit_ridge_head` in training.py).

    The frozen expansion acts as a random kernel approximation, so any
    improvement over a purely linear probe comes from the geometry of the
    representation rather than from an optimised nonlinear mapping.

    Args:
        dz          : latent dimension (input size)
        output_dim  : number of targets (1 for scalar, 2 for cos/sin angle)
        expand_dim  : random-projection dimension (default 512)
        activation  : nn.Module to apply after expansion; None -> nn.Softplus()
        weight_init : initialisation for the frozen expansion weights:
                        "softplus_critical" (default) -- N(0, 2/sqrt(dz)), variance-
                            preserving for Softplus at the critical point
                        "he"      -- Kaiming normal (good for ReLU)
                        "xavier"  -- Glorot normal
                        callable  -- called directly on the weight tensor
    """

    def __init__(self, dz, output_dim=1, expand_dim=512,
                 activation=None, weight_init="softplus_critical"):
        super().__init__()

        self.expand = nn.Linear(dz, expand_dim)
        nn.init.zeros_(self.expand.bias)

        if weight_init == "softplus_critical":
            nn.init.normal_(self.expand.weight, std=2.0 / math.sqrt(dz))
        elif weight_init == "he":
            nn.init.kaiming_normal_(self.expand.weight, nonlinearity="relu")
        elif weight_init == "xavier":
            nn.init.xavier_normal_(self.expand.weight)
        elif callable(weight_init):
            weight_init(self.expand.weight)
        else:
            raise ValueError(f"Unknown weight_init: {weight_init!r}")

        # Freeze the expansion -- never updated by any optimiser
        self.expand.requires_grad_(False)

        self.act  = nn.Softplus() if activation is None else activation
        self.head = nn.Linear(expand_dim, output_dim)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, z):
        return self.head(self.act(self.expand(z)))


class FrameDataset(torch.utils.data.Dataset):
    """
    Sliding-window dataset over trajectory data.

    Each item is a 3-tuple:
        (past_window, future_window, trial_idx)
    with past_window, future_window shape (num_frames, D) and trial_idx a
    0-D long tensor holding the trajectory index within `data`.

    The trial_idx is required by `neg_mode='traj'` and harmless under 'all'
    (the training loop simply ignores it). Trial indices are relative to the
    tensor passed in, so the training split uses 0 .. samples_n - 1 and the
    test split uses 0 .. 199.
    """

    def __init__(self, data, num_frames):
        self.data = data
        self.num_frames = num_frames
        self.n_trials = data.shape[0]
        self.T = data.shape[1]
        self.numD = self.T - 2 * num_frames + 1
        assert self.numD > 0, f"Not enough frames: T={self.T}, num_frames={num_frames}"

    def __len__(self):
        return self.n_trials * self.numD

    def __getitem__(self, idx):
        trial = idx // self.numD
        t = idx % self.numD
        past = self.data[trial, t: t + self.num_frames, :]
        future = self.data[trial, t + self.num_frames: t + 2 * self.num_frames, :]
        return (past, future,
                torch.tensor(trial, dtype=torch.long),
                torch.tensor(t,     dtype=torch.long))


class TrajGroupedBatchSampler(torch.utils.data.Sampler):
    """
    Batch sampler that guarantees k_per_traj windows from each of traj_batch
    randomly-selected trajectories per step.

    This is the default sampler for neg_mode='traj'. With a standard shuffled
    DataLoader and large samples_n, each batch would contain ~1 window per
    trajectory, collapsing the within-trajectory InfoNCE bound to zero.
    This sampler ensures at least k_per_traj same-trajectory negatives per row.

    Effective batch size = traj_batch * k_per_traj.

    Args:
        n_trials        : number of training trajectories (samples_n)
        numFD           : number of valid window start positions per trajectory
                          (= T - 2*num_frames + 1)
        traj_batch      : trajectories sampled per step (default 64)
        k_per_traj      : windows sampled per selected trajectory (default 16)
        steps_per_epoch : gradient steps per epoch; if None, set to
                          max(1, n_trials * numFD // (traj_batch * k_per_traj))
                          to match the data volume of a standard loader
    """

    def __init__(self, n_trials, numFD, traj_batch=64, k_per_traj=16,
                 steps_per_epoch=None):
        self.n_trials = int(n_trials)
        self.numFD = int(numFD)
        self.traj_batch = int(traj_batch)
        self.k_per_traj = int(k_per_traj)
        if steps_per_epoch is None:
            steps_per_epoch = max(1, self.n_trials * self.numFD
                                  // (self.traj_batch * self.k_per_traj))
        self.steps_per_epoch = int(steps_per_epoch)

    def __iter__(self):
        for _ in range(self.steps_per_epoch):
            traj = torch.randint(0, self.n_trials, (self.traj_batch,))
            t_idx = torch.randint(0, self.numFD, (self.traj_batch, self.k_per_traj))
            idx = traj[:, None] * self.numFD + t_idx
            yield idx.reshape(-1).tolist()

    def __len__(self):
        return self.steps_per_epoch


class TimeGroupedBatchSampler(torch.utils.data.Sampler):
    """
    Batch sampler for neg_mode='time': guarantees k_per_time trajectories at
    each of time_batch randomly-selected time points per step.

    Each batch entry shares its time point with k_per_time - 1 other entries
    from different trajectories, giving k_per_time - 1 same-time negatives per
    row for the cross-trajectory InfoNCE.

    Effective batch size = time_batch * k_per_time.

    Args:
        n_trials        : number of training trajectories (samples_n)
        numFD           : number of valid window start positions per trajectory
                          (= T - 2*num_frames + 1)
        time_batch      : time points sampled per step (default 57 = all)
        k_per_time      : trajectories sampled per time point (default 18)
        steps_per_epoch : gradient steps per epoch; if None, set to
                          max(1, n_trials * numFD // (time_batch * k_per_time))
    """

    def __init__(self, n_trials, numFD, time_batch=None, k_per_time=18,
                 steps_per_epoch=None):
        self.n_trials = int(n_trials)
        self.numFD    = int(numFD)
        self.time_batch  = int(time_batch) if time_batch is not None else self.numFD
        self.k_per_time  = int(k_per_time)
        if steps_per_epoch is None:
            steps_per_epoch = max(1, self.n_trials * self.numFD
                                  // (self.time_batch * self.k_per_time))
        self.steps_per_epoch = int(steps_per_epoch)

    def __iter__(self):
        for _ in range(self.steps_per_epoch):
            times = torch.randint(0, self.numFD,   (self.time_batch,))
            trajs = torch.randint(0, self.n_trials, (self.time_batch, self.k_per_time))
            # flat index: trial * numFD + time
            idx = trajs * self.numFD + times[:, None]
            yield idx.reshape(-1).tolist()

    def __len__(self):
        return self.steps_per_epoch


class DatasetTrajFull(torch.utils.data.Dataset):
    """
    One item = all sliding windows for one trajectory.

    Returns (x, y, traj_id) where:
        x, y   : (numD, num_frames, D)  -- all numD window pairs for the trial
        traj_id: scalar long tensor

    Used with neg_mode='traj' and traj_sampler='full'. A DataLoader batching
    traj_chunk_size trajectories at a time, combined with collate_traj_full,
    yields (traj_chunk_size * numD, num_frames, D) batches where every row has
    numD - 1 same-trajectory negatives -- the maximum possible.
    """

    def __init__(self, data, num_frames):
        self.data = data
        self.num_frames = num_frames
        self.n_trials = data.shape[0]
        self.T = data.shape[1]
        self.numD = self.T - 2 * num_frames + 1
        assert self.numD > 0, f"Not enough frames: T={self.T}, num_frames={num_frames}"
        starts = torch.arange(self.numD)
        offsets = torch.arange(num_frames)
        self._idx_x = starts[:, None] + offsets[None, :]   # (numD, num_frames)
        self._idx_y = self._idx_x + num_frames

    def __len__(self):
        return self.n_trials

    def __getitem__(self, traj):
        td = self.data[traj]                               # (T, D)
        return (td[self._idx_x],                           # (numD, num_frames, D)
                td[self._idx_y],
                torch.tensor(traj, dtype=torch.long),
                torch.arange(self.numD, dtype=torch.long)) # time indices 0..numD-1


def collate_traj_full(batch):
    """
    Collate fn for DatasetTrajFull: flatten trajectory-major batches into
    the standard window-major format expected by the model and training loop.

    Input:  list of (x:[numD,nF,D], y:[numD,nF,D], traj_id:scalar, tmi:[numD])
    Output: (x:[B*numD,nF,D], y:[B*numD,nF,D], traj_id:[B*numD], tmi:[B*numD])
    """
    xs, ys, tids, tmis = zip(*batch)
    x = torch.cat(xs, dim=0)
    y = torch.cat(ys, dim=0)
    tid = torch.cat(
        [tid.unsqueeze(0).expand(xi.shape[0]) for tid, xi in zip(tids, xs)], dim=0
    )
    tmi = torch.cat(tmis, dim=0)   # each tmi is already (numD,); just concatenate
    return x, y, tid, tmi


class ProbeDataset(torch.utils.data.Dataset):
    """Simple (z, y) regression dataset for prober training."""

    def __init__(self, z, y):
        self.z = z
        self.y = y

    def __len__(self):
        return self.z.shape[0]

    def __getitem__(self, idx):
        return self.z[idx], self.y[idx]

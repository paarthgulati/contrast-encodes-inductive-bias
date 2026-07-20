"""
Supervised baseline for DVSIB on pendulum dynamics with OU-correlated
observation noise.

Classes:
    SupervisedModel     -- Encoder + linear head, trained end-to-end on
                           [cos(theta), sin(theta)] labels; uses mean_z only
                           (deterministic, no KL term)
    LabeledWindowDataset -- sliding past-window dataset with [cos, sin] labels
                            aligned to window start (same convention as probers)
"""

import torch
import torch.nn as nn

from models import Encoder


class SupervisedModel(nn.Module):
    """
    Supervised encoder: same Encoder architecture as DVSIB + a linear head.

    Forward pass uses mean_z only (deterministic).  No sampling, no KL term.

    Args:
        input_dim  : per-frame pixel dimension
        dz         : latent dimension
        dzf        : per-frame embedding size inside encoder (default 32)
        num_frames : frames per window
        hidden     : encoder MLP hidden width (default 256)
    """

    def __init__(self, input_dim, dz=2, dzf=32, num_frames=2, hidden=256):
        super().__init__()
        self.encoder = Encoder(
            dz=dz, dzf=dzf, input_dim=input_dim,
            num_frames=num_frames, hidden=hidden,
        )
        self.head = nn.Linear(dz, 2)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        """
        x : (batch, num_frames, D)
        returns pred : (batch, 2)  -- predicted [cos(theta), sin(theta)]
        """
        mean_z, _, _ = self.encoder(x)
        return self.head(mean_z)


class LabeledWindowDataset(torch.utils.data.Dataset):
    """
    Sliding-window dataset pairing each past window with its [cos, sin] label.

    Each item is a 2-tuple:
        (past_window, cos_sin_label)
    with past_window shape (num_frames, D) and cos_sin_label shape (2,).

    Label alignment: window starting at frame t in trajectory `trial` is
    labelled with [cos(theta[trial, t]), sin(theta[trial, t])], matching the
    convention used by the existing probers in make_phys_tensors / encode_dataset.

    Args:
        data   : (n_trials, T, D) float tensor -- pixel data (noisy)
        theta  : (n_trials, T) float array/tensor -- ground-truth angles (rad)
        num_frames : frames per past window
    """

    def __init__(self, data, theta, num_frames):
        self.data = data
        self.theta = torch.as_tensor(theta).float()
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
        th = self.theta[trial, t]
        label = torch.stack([torch.cos(th), torch.sin(th)])
        return past, label

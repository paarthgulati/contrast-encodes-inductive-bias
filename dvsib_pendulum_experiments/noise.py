"""
Temporally-correlated observation noise for pendulum pixel trajectories.

Public functions
----------------
add_ou_noise_per_trajectory
    Gaussian OU/AR(1) noise (original).  sigma = marginal noise std.

add_ou_noise_uniform
    Uniform OU noise.  Innovations drawn from Uniform[0,1] centered at
    mu=0.5; process stays near [0,1].  sigma = amplitude ceiling, not std.
"""

from typing import Optional, Tuple, Union

import numpy as np
import torch


def add_ou_noise_per_trajectory(
    data: Union[torch.Tensor, np.ndarray],
    sigma: float,
    ou_alpha: float,
    clip: Optional[Tuple[float, float]] = (0.0, 1.0),
    seed: int = 0,
    init: str = "stationary",
):
    """
    Add per-pixel AR(1) (Ornstein-Uhlenbeck) noise to each trajectory.

    For each trajectory and each pixel the noise process is:

        eta_0 ~ N(0, 1)           if init == "stationary"   (recommended)
        eta_0 = 0                 if init == "zero"
        eta_t = ou_alpha * eta_{t-1} + sqrt(1 - ou_alpha^2) * xi_t,  xi_t ~ N(0,1)

    Output:
        x_noisy = clamp(x + sigma * eta, clip)

    Parametrisation:
        * ou_alpha = 0 reproduces independent Gaussian noise per frame.
        * ou_alpha → 1 makes the noise slow-varying within a trajectory.
        * The `sqrt(1 - ou_alpha^2)` innovation keeps Var(eta_t) = 1 for all t,
          so sigma directly controls the marginal noise std regardless of
          ou_alpha.

    Args:
        data     : (n_trials, T, D) torch.Tensor or np.ndarray
        sigma    : noise standard deviation added to x
        ou_alpha : memory parameter, in [0, 1]
        clip     : (lo, hi) clamp range, or None to skip clamping.
                   Default (0, 1) matches pendulum-pixel range.
        seed     : RNG seed for reproducibility
        init     : "stationary" or "zero"

    Returns:
        x_noisy : torch.Tensor with the same shape as `data`
    """
    x = torch.as_tensor(data).clone()
    if x.ndim != 3:
        raise ValueError(f"Expected (n_trials, T, D), got {tuple(x.shape)}")
    if not (0.0 <= ou_alpha <= 1.0):
        # Paper's Eq. 1 only defines the OU memory parameter on [0, 1]
        # (0=white noise, 1=static); negative alpha is mathematically
        # well-defined (anti-correlated noise) but outside the paper's model
        # and never meaningfully used, so it's rejected rather than silently
        # accepted.
        raise ValueError(f"ou_alpha must be in [0, 1], got {ou_alpha}")

    n_trials, T, D = x.shape
    g = torch.Generator(device=x.device)
    g.manual_seed(int(seed))

    beta = (1.0 - ou_alpha * ou_alpha) ** 0.5
    eta = torch.empty((n_trials, T, D), device=x.device, dtype=x.dtype)

    if init == "stationary":
        eta[:, 0, :] = torch.randn(
            (n_trials, D), generator=g, device=x.device, dtype=x.dtype
        )
    elif init == "zero":
        eta[:, 0, :] = 0.0
    else:
        raise ValueError(f"init must be 'stationary' or 'zero', got {init}")

    for t in range(1, T):
        xi = torch.randn(
            (n_trials, D), generator=g, device=x.device, dtype=x.dtype
        )
        eta[:, t, :] = ou_alpha * eta[:, t - 1, :] + beta * xi

    x_noisy = x + sigma * eta
    if clip is not None:
        lo, hi = clip
        x_noisy = x_noisy.clamp(lo, hi)
    return x_noisy


def add_ou_noise_uniform(
    data: Union[torch.Tensor, np.ndarray],
    sigma: float,
    ou_alpha: float,
    clip: Optional[Tuple[float, float]] = (0.0, 1.0),
    seed: int = 0,
    init: str = "stationary",
):
    """
    Add per-pixel Uniform OU noise to each trajectory.

    The overlay process is centered at mu = 0.5 (the mean of Uniform[0, 1]):

        o_0 ~ Uniform[0, 1]        if init == "stationary"   (recommended)
        o_0 = 0.5                  if init == "zero"
        o_t = mu + ou_alpha*(o_{t-1} - mu) + beta*(xi_t - mu),  xi_t ~ U[0,1]

    where beta = sqrt(1 - ou_alpha^2) and mu = 0.5.

    Output:
        x_noisy = clamp(x + sigma * o_t, clip)

    Parametrisation:
        * ou_alpha = 0 reproduces i.i.d. Uniform[0, 1] noise per frame.
        * ou_alpha -> 1 makes the overlay slow-varying (frozen across frames).
        * sigma is an amplitude ceiling (o_t in [0, 1]), NOT a std dev.
          The mean overlay per pixel is ~sigma * 0.5, systematically
          brightening the image.
        * The sqrt(1 - ou_alpha^2) scaling preserves the stationary variance
          of the process regardless of ou_alpha.

    Args:
        data     : (n_trials, T, D) torch.Tensor or np.ndarray
        sigma    : noise amplitude ceiling (x_noisy = x + sigma * o_t)
        ou_alpha : memory parameter, in [0, 1]
        clip     : (lo, hi) clamp range, or None to skip clamping.
                   Default (0, 1) matches pendulum-pixel range.
        seed     : RNG seed for reproducibility
        init     : "stationary" draws o_0 ~ Uniform[0,1];
                   "zero" sets o_0 = 0.5 (the process mean)

    Returns:
        x_noisy : torch.Tensor with the same shape as `data`
    """
    x = torch.as_tensor(data).clone()
    if x.ndim != 3:
        raise ValueError(f"Expected (n_trials, T, D), got {tuple(x.shape)}")
    if not (0.0 <= ou_alpha <= 1.0):
        # See add_ou_noise_per_trajectory: alpha < 0 is outside the paper's
        # model and not meaningful here, so it's rejected rather than
        # silently accepted.
        raise ValueError(f"ou_alpha must be in [0, 1], got {ou_alpha}")

    n_trials, T, D = x.shape
    g = torch.Generator(device=x.device)
    g.manual_seed(int(seed))

    mu = 0.5
    beta = (1.0 - ou_alpha * ou_alpha) ** 0.5
    o = torch.empty((n_trials, T, D), device=x.device, dtype=x.dtype)

    if init == "stationary":
        o[:, 0, :] = torch.rand(
            (n_trials, D), generator=g, device=x.device, dtype=x.dtype
        )
    elif init == "zero":
        o[:, 0, :] = mu
    else:
        raise ValueError(f"init must be 'stationary' or 'zero', got {init}")

    for t in range(1, T):
        xi = torch.rand(
            (n_trials, D), generator=g, device=x.device, dtype=x.dtype
        )
        o[:, t, :] = mu + ou_alpha * (o[:, t - 1, :] - mu) + beta * (xi - mu)

    x_noisy = x + sigma * o
    if clip is not None:
        lo, hi = clip
        x_noisy = x_noisy.clamp(lo, hi)
    return x_noisy

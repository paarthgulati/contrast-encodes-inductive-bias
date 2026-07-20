from typing import NamedTuple, Any, Optional
import math

import torch
import numpy as np


class Sample(NamedTuple):
    states: torch.Tensor    # [(batch_size), T, 1, 28, 28]
    locations: torch.Tensor # [(batch_size), T, 2]
    actions: torch.Tensor   # [(batch_size), T, 2]


class ContinuousMotionDataset:
    def __init__(
        self,
        size,
        batch_size,
        n_steps=2,
        concentration: float = 0.2,
        max_step: float = 4.0,
        std: float = 1.3,
        noise: float = 0.0,
        noise_alpha: Optional[float] = None,
        ou_center_mode: str = "global_half",
        ou_mu: float = 0.5,
        ou_clip: bool = False,
        static_noise: float = 0.0,
        static_noise_speed: float = 0.0,
        img_size: int = 28,
        normalize: bool = False,
        device: Any = torch.device("cpu"),
        train: bool = True,
    ):
        self.size = size
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.concentration = concentration
        self.std = std
        self.max_step = max_step
        self.noise = noise
        self.noise_alpha = (
            None if noise_alpha is None else float(noise_alpha)
        )
        if self.noise_alpha is not None and not (0.0 <= self.noise_alpha <= 1.0):
            raise ValueError(
                f"noise_alpha must be in [0, 1], got {self.noise_alpha}"
            )
        self.ou_center_mode = str(ou_center_mode)
        if self.ou_center_mode != "global_half":
            raise ValueError(
                f"ou_center_mode must be 'global_half', got {self.ou_center_mode!r}"
            )
        self.ou_mu = float(ou_mu)
        self.ou_clip = bool(ou_clip)
        self.static_noise = static_noise
        self.static_noise_speed = static_noise_speed
        self.img_size = img_size
        self.device = device

        self.normalize = False
        self.padding = 2 * self.std

        if normalize:
            self.n_mean, self.n_std = self.estimate_mean_std()
            self.normalize = True

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        return self.generate_multistep_sample()

    def __iter__(self):
        for _ in range(self.size):
            yield self.generate_multistep_sample()

    def render_location(self, locations: torch.Tensor):
        x = torch.linspace(
            0, self.img_size - 1, steps=self.img_size, device=self.device
        )
        y = torch.linspace(
            0, self.img_size - 1, steps=self.img_size, device=self.device
        )
        xx, yy = torch.meshgrid(x, y, indexing="xy")
        c = torch.stack([xx, yy], dim=-1)
        c = c.view(*([1] * (len(locations.shape) - 1)), *c.shape).repeat(
            *locations.shape[:-1], *([1] * len(c.shape))
        )
        locations = locations.unsqueeze(-2).unsqueeze(-2)
        img = torch.exp(
            -(c - locations).norm(dim=-1).pow(2) / (2 * self.std * self.std)
        ) / (2 * math.pi * self.std * self.std)
        return img

    def generate_state(self):
        effective_range = (self.img_size - 1) - 2 * self.padding
        location = (
            torch.rand(size=(self.batch_size, 2), device=self.device) * effective_range
            + self.padding
        )
        return location

    def generate_static_overlay(self, shape):
        # Shape is BS x T x H x W; returns a single random frame tiled over time.
        static_noise_overlay = (
            torch.rand(shape[0], *shape[2:], device=self.device)
        ).unsqueeze(1)
        repeat = [1] * len(static_noise_overlay.shape)
        repeat[1] = shape[1]
        return static_noise_overlay.repeat(*repeat)

    def generate_rnd_overlay(self, shape):
        return torch.rand(shape, device=self.device)

    def generate_ou_overlay(self, shape):
        """OU (AR-1) noise overlay centered at global mean mu (global_half mode).

        alpha=0 produces i.i.d. uniform noise; alpha=1 produces a static frame.
        """
        if self.noise_alpha is None:
            raise RuntimeError("generate_ou_overlay called with noise_alpha=None")

        alpha = self.noise_alpha
        if alpha <= 0.0:
            overlay = self.generate_rnd_overlay(shape)
        elif alpha >= 1.0:
            overlay = self.generate_static_overlay(shape)
        else:
            rnd_overlay = self.generate_rnd_overlay(shape)
            overlay = torch.empty_like(rnd_overlay)
            overlay[:, 0] = rnd_overlay[:, 0]
            beta = math.sqrt(max(0.0, 1.0 - alpha * alpha))
            mu = rnd_overlay.new_tensor(self.ou_mu)

            for i in range(1, shape[1]):
                overlay[:, i] = (
                    mu
                    + alpha * (overlay[:, i - 1] - mu)
                    + beta * (rnd_overlay[:, i] - mu)
                )

        if self.ou_clip:
            overlay = overlay.clamp(0.0, 1.0)
        return overlay

    def generate_multistep_sample(self):
        actions = self.generate_actions(self.n_steps)
        start_location = self.generate_state()
        sample = self.generate_transitions(start_location, actions)
        if self.static_noise > 0 or self.noise > 0:
            static_noise_overlay = (
                self.generate_static_overlay(sample.states.shape) * sample.states.max()
            )
            if self.static_noise_speed > 0:
                for i in range(static_noise_overlay.shape[1]):
                    static_noise_overlay[:, i] = torch.roll(
                        static_noise_overlay[:, i],
                        shifts=int(i * self.static_noise_speed),
                        dims=-1,
                    )
            if self.noise_alpha is None:
                rnd_noise_overlay = (
                    self.generate_rnd_overlay(sample.states.shape)
                    * sample.states.max()
                )
            else:
                rnd_noise_overlay = (
                    self.generate_ou_overlay(sample.states.shape)
                    * sample.states.max()
                )
            sample = Sample(
                states=(
                    sample.states
                    + static_noise_overlay * self.static_noise
                    + rnd_noise_overlay * self.noise
                ),
                locations=sample.locations,
                actions=sample.actions,
            )
        if self.normalize:
            sample = Sample(
                states=(sample.states - self.n_mean) / self.n_std,
                locations=(
                    (sample.locations - self.padding)
                    / (self.img_size - 1 - 2 * self.padding)
                    - 0.5
                )
                / np.sqrt(1 / 12),
                actions=sample.actions / (0.4102 * self.max_step + 1e-7),
            )
        return sample

    def normalize_location(self, location):
        return (
            (location - self.padding) / (self.img_size - 1 - 2 * self.padding) - 0.5
        ) / np.sqrt(1 / 12)

    def generate_transitions(self, location, actions):
        locations = [location]
        for i in range(actions.shape[1]):
            next_location = self.generate_transition(locations[-1], actions[:, i])
            locations.append(next_location)

        locations = torch.stack(locations, dim=1).unsqueeze(dim=-2)
        actions = actions.unsqueeze(dim=-2)
        states = self.render_location(locations)
        return Sample(states, locations, actions)

    def generate_transition(self, location, action):
        next_location = location + action
        next_location = torch.clamp(
            next_location, min=self.padding, max=self.img_size - 1 - self.padding
        )
        return next_location

    def generate_actions(self, n_steps: int):
        x = torch.rand(self.batch_size, device=self.device) * 2 * math.pi
        d = torch.distributions.VonMises(
            x, concentration=self.concentration * torch.ones_like(x)
        )
        a = (
            d.sample((n_steps - 1,)).permute(1, 0).to(self.device)
        )
        step_sizes = (
            torch.rand(self.batch_size, n_steps - 1, 1, device=self.device)
            * self.max_step
        )
        vec = ContinuousMotionDataset.angle_to_vec(a)
        actions = vec * step_sizes
        return actions

    @staticmethod
    def angle_to_vec(a):
        return torch.stack([torch.cos(a), torch.sin(a)], dim=-1)

    @torch.no_grad()
    def estimate_mean_std(self):
        N = 100
        means = []
        stds = []
        for i in range(N):
            imgs = self[i].states.flatten()
            means.append(imgs.mean().item())
            stds.append(imgs.std().item())
        return np.mean(means), np.mean(stds)

    @torch.no_grad()
    def unnormalize_mse(self, mse):
        if not self.normalize:
            return mse
        return mse * (
            (np.sqrt(1 / 12) * (self.img_size - 1 - 2 * self.padding)) ** 2
        )

    @torch.no_grad()
    def unnormalize_location(self, locations):
        if not self.normalize:
            return locations
        return ((locations * np.sqrt(1 / 12)) + 0.5) * (
            self.img_size - 1 - 2 * self.padding
        ) + self.padding


class FixedIndexedMotionDataset(ContinuousMotionDataset):
    """Deterministic dataset: each index maps to a fixed sample across epochs."""

    def __init__(self, *args, base_seed: int = 0, split_seed_offset: int = 0, **kwargs):
        self.base_seed = int(base_seed)
        self.split_seed_offset = int(split_seed_offset)
        super().__init__(*args, **kwargs)

    def __iter__(self):
        for i in range(self.size):
            yield self[i]

    def _seed_for_index(self, idx: int) -> int:
        return (
            self.base_seed + self.split_seed_offset + int(idx) * 1_000_003
        ) % (2**63 - 1)

    def _fork_rng_devices(self):
        if self.device.type == "cuda":
            if self.device.index is None:
                return [torch.cuda.current_device()]
            return [self.device.index]
        return []

    def __getitem__(self, i):
        seed = self._seed_for_index(i)
        with torch.random.fork_rng(devices=self._fork_rng_devices()):
            torch.manual_seed(seed)
            return self.generate_multistep_sample()

import data
from typing import Optional
import os
from dataclasses import dataclass, field
import dataclasses
from pathlib import Path
import random
from enum import Enum, auto

import torch
import numpy as np
from tqdm.auto import tqdm
import wandb
from omegaconf import OmegaConf, MISSING
from matplotlib import pyplot as plt
import matplotlib
import enum
from torch import nn

from configs import ConfigBase
from simclr import SimCLRConfig, SimCLR
from lars import LARS, exclude_bias_and_norm, adjust_learning_rate
import probing


def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


class ModelType(enum.Enum):
    SimCLR = enum.auto()


class DatasetType(Enum):
    Single = auto()
    SingleFixed = auto()


@dataclass
class TrainConfig(ConfigBase):
    model_type: ModelType = MISSING
    n_steps: int = 17
    val_n_steps: int = 17
    dataset_size: int = 10000
    dataset_noise: float = 0.0
    dataset_noise_alpha: Optional[float] = None
    dataset_ou_center_mode: str = "global_half"
    dataset_ou_mu: float = 0.5
    dataset_ou_clip: bool = False
    val_dataset_size: int = 10000
    wandb: bool = True
    run_name: Optional[str] = None
    run_group: Optional[str] = None
    output_path: Optional[str] = None
    safe_every_n_epochs: int = 10
    eval_mpcs: int = 20
    quick_debug: bool = False
    seed: Optional[int] = None
    load_checkpoint_path: Optional[str] = None
    eval_only: bool = False
    probe_mpc: bool = False
    dataset_static_noise: float = 0.0
    dataset_batch_size: int = 128
    dataset_static_noise_speed: float = 0.0
    dataset_dot_std: float = 1.3
    dataset_max_step: float = 4.0
    dataset_normalize: bool = False
    dataset_fixed_seed: Optional[int] = None
    simclr: SimCLRConfig = field(default_factory=SimCLRConfig)
    eval_at_the_end_only: bool = False
    dataset_type: DatasetType = DatasetType.Single

    probing_cfg: probing.ProbingConfig = field(default_factory=probing.ProbingConfig)


class Trainer:
    def __init__(self, config: TrainConfig):
        self.config = config
        self.model_config = self.config.simclr

        if config.seed is None:
            seed = random.SystemRandom().randint(0, 2**32 - 1)
        else:
            seed = int(config.seed)
        seed_everything(seed)
        config.seed = seed

        self.pred_ms = SimCLR(config.simclr)
        self.pred_ms = self.pred_ms.cuda()

        if config.wandb:
            # Set WANDB_PROJECT env var or pass --wandb_project to control where
            # runs are logged. Replace YOUR_WANDB_ENTITY with your W&B username/org.
            project_name = os.getenv("WANDB_PROJECT", "YOUR_PROJECT_NAME")
            wandb.init(
                project=project_name,
                name=config.run_name,
                group=config.run_group,
                config=dataclasses.asdict(config),
            )
            wandb.config.update({"seed": seed}, allow_val_change=True)
            wandb.run.summary["seed"] = seed

        self.sample_step = 0
        self.epoch = 0
        self.step = 0

        self.save_config()
        self.init_dataset()

        load_result = self.maybe_load_model()
        if (
            config.eval_only
            and not config.probing_cfg.full_finetune
            and not load_result
        ):
            print("WARN: probing a random network. Is that intentional?")

        self.n_parameters = sum(
            p.numel() for p in self.pred_ms.parameters() if p.requires_grad
        )
        print("number of params:", self.n_parameters)
        if config.wandb:
            wandb.run.summary["n_params"] = self.n_parameters
            wandb.run.summary["actual_repr_size"] = self.pred_ms.embedding

    def init_dataset(self):
        assert self.pred_ms.args.channels == 1, (
            "encoder and dataset type are incompatible: "
            f"single dot dataset provides one channel, while {self.pred_ms.args.channels} were expected"
        )
        dataset_cls = data.ContinuousMotionDataset
        train_seed_kwargs = {}
        val_seed_kwargs = {}

        if self.config.dataset_type == DatasetType.SingleFixed:
            if self.config.dataset_fixed_seed is not None:
                fixed_seed = int(self.config.dataset_fixed_seed)
            elif self.config.seed is not None:
                fixed_seed = int(self.config.seed)
            else:
                fixed_seed = 0

            dataset_cls = data.FixedIndexedMotionDataset
            train_seed_kwargs = {
                "base_seed": fixed_seed,
                "split_seed_offset": 0,
            }
            val_seed_kwargs = {
                "base_seed": fixed_seed,
                "split_seed_offset": 10_000_000,
            }

        common_kwargs = dict(
            n_steps=self.config.n_steps + self.pred_ms.args.rnn_burnin - 1,
            noise=self.config.dataset_noise,
            noise_alpha=self.config.dataset_noise_alpha,
            ou_center_mode=self.config.dataset_ou_center_mode,
            ou_mu=self.config.dataset_ou_mu,
            ou_clip=self.config.dataset_ou_clip,
            static_noise=self.config.dataset_static_noise,
            static_noise_speed=self.config.dataset_static_noise_speed,
            std=self.config.dataset_dot_std,
            max_step=self.config.dataset_max_step,
            normalize=self.config.dataset_normalize,
            device=torch.device("cuda"),
        )

        self.ds = dataset_cls(
            self.config.dataset_size // self.config.dataset_batch_size,
            batch_size=self.config.dataset_batch_size,
            train=True,
            **common_kwargs,
            **train_seed_kwargs,
        )
        self.val_ds = dataset_cls(
            self.config.val_dataset_size // self.config.dataset_batch_size,
            batch_size=self.config.dataset_batch_size,
            train=False,
            **{**common_kwargs, "n_steps": self.config.val_n_steps + self.pred_ms.args.rnn_burnin - 1},
            **val_seed_kwargs,
        )

    def maybe_load_model(self):
        if self.config.load_checkpoint_path is not None:
            checkpoint = torch.load(self.config.load_checkpoint_path)
            self.pred_ms.load_state_dict(checkpoint["model_state_dict"])
            return True
        return False

    def save_config(self):
        if self.config.output_path is not None:
            out_dir = (
                os.path.join(self.config.output_path, self.config.run_group)
                if self.config.run_group
                else self.config.output_path
            )
            os.makedirs(out_dir, exist_ok=True)
            p = Path(out_dir) / "config.yaml"
            with p.open("w") as f:
                OmegaConf.save(config=self.config, f=f)
                print("saved config")

    def train(self):
        self.optimizer = LARS(
            self.pred_ms.parameters(),
            lr=0,
            weight_decay=1e-6,
            weight_decay_filter=exclude_bias_and_norm,
            lars_adaptation_filter=exclude_bias_and_norm,
        )

        self.save_model()

        if not self.config.eval_at_the_end_only or self.config.quick_debug:
            self.validate()

        for epoch in tqdm(range(1, self.pred_ms.args.epochs + 1)):
            self.epoch = epoch
            for step, batch in enumerate(self.ds, start=epoch * len(self.ds)):
                s = batch.states.cuda().permute(1, 0, 2, 3, 4)
                a = batch.actions.cuda().permute(1, 0, 2, 3)
                a = a[:, :, 0]  # actions of the first dot

                lr = adjust_learning_rate(
                    self.model_config, self.optimizer, self.ds, step
                )

                self.sample_step += s.shape[1]
                self.step = step

                self.optimizer.zero_grad()
                loss_info = self.pred_ms.forward(s, a, step=self.step)
                loss_info.total_loss.backward()
                self.optimizer.step()

                if step % 100 == 0 and wandb.run is not None:
                    wandb.log(
                        {
                            "sample_step": self.sample_step,
                            "loss": loss_info.total_loss.item(),
                            "learning_rate": lr,
                            "custom_step": step,
                        },
                        commit=False,
                    )

                    diags = loss_info.diagnostics_info
                    if diags is not None:
                        if isinstance(diags, dict):
                            wandb.log(diags)
                        else:
                            wandb.log(diags.build_log_dict())

                if self.config.quick_debug:
                    return

            if (
                self.epoch % self.config.safe_every_n_epochs == 0
                and not self.config.eval_at_the_end_only
            ) or self.epoch >= self.model_config.epochs:
                self.save_model()
                self.validate()

    def validate(self):
        if not self.config.probing_cfg.full_finetune:
            self.pred_ms.eval()

        probing_enc_result = probing.probe_enc_position(
            backbone=self.pred_ms.backbone,
            embedding=self.pred_ms.embedding,
            dataset=self.val_ds,
            quick_debug=self.config.quick_debug,
            config=self.config.probing_cfg,
            name_suffix=f"_{self.epoch}",
        )

        log_dict = {
            "avg_eval_enc_loss": probing_enc_result,
            "avg_eval_enc_loss_rmse": np.sqrt(probing_enc_result),
            "epoch": self.epoch,
            "sample_step": self.sample_step,
            "custom_step": self.step,
        }

        if self.config.wandb:
            wandb.log(log_dict)
            wandb.run.summary.update(log_dict)

        for v in log_dict.values():
            if isinstance(v, matplotlib.figure.Figure):
                plt.close(v)

        self.pred_ms.train()

    def save_model(self):
        if self.config.output_path is not None:
            out_dir = (
                os.path.join(self.config.output_path, self.config.run_group)
                if self.config.run_group
                else self.config.output_path
            )
            os.makedirs(out_dir, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": self.pred_ms.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                },
                os.path.join(
                    out_dir,
                    f"epoch={self.epoch}_sample_step={self.sample_step}.ckpt",
                ),
            )


def main(config: TrainConfig):
    torch.set_num_threads(1)
    trainer = Trainer(config)
    if not config.eval_only:
        trainer.train()
    else:
        trainer.validate()


if __name__ == "__main__":
    cfg = TrainConfig.parse_from_command_line()
    main(cfg)

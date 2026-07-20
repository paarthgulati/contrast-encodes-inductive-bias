import dataclasses
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import wandb

import data
import models
import probing
from configs import ConfigBase


def seed_everything(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


@dataclass
class SupervisedBaselineConfig(ConfigBase):
    # Dataset — match the same parameter names as train.py for clarity
    dataset_noise: float = 0.0
    dataset_noise_alpha: Optional[float] = None
    dataset_ou_center_mode: str = "global_half"
    dataset_ou_mu: float = 0.5
    dataset_static_noise: float = 0.0
    dataset_dot_std: float = 1.3
    dataset_max_step: float = 4.0
    val_dataset_size: int = 10000
    dataset_batch_size: int = 512
    dataset_normalize: bool = True
    n_steps: int = 17
    # Backbone — same arch/size as SSL runs for fair comparison
    arch: str = "menet5"
    embedding_size: int = 512
    backbone_width_factor: int = 2
    # full_finetune=True trains backbone+prober jointly from random init
    probing_cfg: probing.ProbingConfig = field(
        default_factory=lambda: probing.ProbingConfig(
            full_finetune=True, lr=1e-3, epochs_enc=100
        )
    )
    # W&B — set WANDB_PROJECT env var or pass --wandb_project on the command line
    wandb_project: str = "YOUR_PROJECT_NAME"
    wandb: bool = True
    run_name: Optional[str] = None
    run_group: Optional[str] = None
    seed: int = 42
    quick_debug: bool = False


def main(config: SupervisedBaselineConfig):
    seed_everything(config.seed)

    if config.wandb:
        wandb.init(
            project=config.wandb_project,
            name=config.run_name,
            group=config.run_group,
            config=dataclasses.asdict(config),
        )

    dataset = data.ContinuousMotionDataset(
        config.val_dataset_size // config.dataset_batch_size,
        batch_size=config.dataset_batch_size,
        n_steps=config.n_steps,
        noise=config.dataset_noise,
        noise_alpha=config.dataset_noise_alpha,
        ou_center_mode=config.dataset_ou_center_mode,
        ou_mu=config.dataset_ou_mu,
        static_noise=config.dataset_static_noise,
        std=config.dataset_dot_std,
        max_step=config.dataset_max_step,
        normalize=config.dataset_normalize,
        device=torch.device("cuda"),
    )

    # Fresh random-init backbone — no SSL pre-training
    backbone, embedding = models.build_backbone(
        arch=config.arch,
        embedding_size=config.embedding_size,
        backbone_mlp=None,
        backbone_width_factor=config.backbone_width_factor,
        channels=1,
    )
    backbone = backbone.cuda()

    # full_finetune=True: backbone + prober head trained jointly on position MSE
    avg_mse = probing.probe_enc_position(
        backbone=backbone,
        embedding=embedding,
        dataset=dataset,
        quick_debug=config.quick_debug,
        config=config.probing_cfg,
    )

    rmse = float(np.sqrt(avg_mse))

    log_dict = {
        "avg_eval_enc_loss": float(avg_mse),
        "avg_eval_enc_loss_rmse": rmse,
        "dataset_noise": config.dataset_noise,
        "dataset_noise_alpha": config.dataset_noise_alpha,
        "dataset_static_noise": config.dataset_static_noise,
    }

    print(f"sigma={config.dataset_noise:.3f}  RMSE={rmse:.4f}")

    if config.wandb:
        wandb.log(log_dict)
        wandb.run.summary.update(log_dict)
        wandb.finish()


if __name__ == "__main__":
    cfg = SupervisedBaselineConfig.parse_from_command_line()
    main(cfg)

from dataclasses import dataclass
from typing import Optional, Any

import torch
from torch.nn import functional as F

from configs import ConfigBase
import models


@dataclass
class SimCLRConfig(ConfigBase):
    variant: str = "baseline"
    projector: str = "512-512-512"
    random_projector: bool = False
    predictor: str = "rnn"
    backbone_mlp: Optional[str] = None
    epochs: int = 100
    base_lr: float = 0.2
    decoding_coeff: float = 0.0
    action_coeff: float = 0.0
    arch: str = "menet5"
    embedding_size: Optional[int] = 64  # only for MeNet5
    action_dim: int = 2
    rnn_layers: int = 1
    rnn_burnin: int = 1
    # NOTE: must be 2 to match the paper's Appendix B.2 channel counts
    # (32, 64, 64); every shipped config sets this explicitly. A width_factor
    # of 1 silently trains a half-width encoder instead.
    backbone_width_factor: int = 1
    normalize_z: bool = False
    loss_temp: float = 1.0
    channels: int = 1


@dataclass
class LossInfo:
    total_loss: torch.Tensor
    diagnostics_info: Any


class SimCLR(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.variant = getattr(args, "variant", "baseline")

        self.backbone, self.embedding = models.build_backbone(
            args.arch,
            args.embedding_size,
            args.backbone_mlp,
            args.backbone_width_factor,
            channels=args.channels,
        )

        print(f"backbone: {args.arch}  variant: {args.variant}")

        if self.variant == "baseline":
            self.projector = models.Projector(
                args.projector, self.embedding, random=args.random_projector
            )
            self.num_features = self.projector.output_dim
            self.predictor = models.build_predictor(
                args.predictor, self.embedding, self.args.action_dim, self.args.rnn_layers
            )
            self._forward_impl = self._forward_baseline

        elif self.variant == "baseline_traj_neg":
            # Same architecture as baseline; negatives are drawn within each trajectory
            # rather than across the full batch.
            self.projector = models.Projector(
                args.projector, self.embedding, random=args.random_projector
            )
            self.num_features = self.projector.output_dim
            self.predictor = models.build_predictor(
                args.predictor, self.embedding, self.args.action_dim, self.args.rnn_layers
            )
            self._forward_impl = self._forward_baseline_traj_neg

        else:
            raise ValueError(
                f"Unknown SimCLR variant: {self.variant!r}. "
                "Must be 'baseline' or 'baseline_traj_neg'."
            )

    def forward(self, states, actions, step=None):
        return self._forward_impl(states, actions, step)

    def _forward_baseline(self, states, actions, step=None):
        """states [T, batch_size, 1, 28, 28]
        actions [T-1, batch_size]
        """
        T = states.shape[0]

        states_flatten = states.reshape(-1, *states.shape[2:])
        states_enc = self.backbone(states_flatten)
        states_proj = self.projector(states_enc)

        states_enc = states_enc.view(*states.shape[:2], -1)
        states_proj = states_proj.view(*states.shape[:2], -1)

        current_enc = states_enc[0]
        pred_encs = self.predictor.predict_sequence(
            enc=current_enc, actions=actions, h=None
        )

        info_nce_loss = torch.tensor(0.0).cuda()

        for i in range(T - 1):
            pred_enc = pred_encs[i]
            pred_proj = self.projector(pred_enc)
            info_nce_loss += self.info_nce_simclr(pred_proj, states_proj[i + 1])

        info_nce_loss /= T - 1
        return LossInfo(total_loss=info_nce_loss, diagnostics_info=None)

    def _forward_baseline_traj_neg(self, states, actions, step=None):
        """Baseline with trajectory-local negatives.

        states [T, B, 1, 28, 28]
        actions [T-1, B, A]
        """
        T, B = states.shape[0], states.shape[1]

        states_flatten = states.reshape(-1, *states.shape[2:])
        states_enc = self.backbone(states_flatten)
        states_proj = self.projector(states_enc)

        states_enc = states_enc.view(T, B, -1)
        states_proj = states_proj.view(T, B, -1)

        current_enc = states_enc[0]
        pred_encs = self.predictor.predict_sequence(
            enc=current_enc, actions=actions, h=None
        )  # [T-1, B, E]

        pred_projs = self.projector(
            pred_encs.reshape(-1, pred_encs.shape[-1])
        ).view(T - 1, B, -1)
        target_projs = states_proj[1:]  # [T-1, B, D]

        # Reindex to [B, T-1, D] so negatives are only different time points
        # within the same trajectory.
        loss = self.info_nce_simclr_traj(
            pred_projs.permute(1, 0, 2), target_projs.permute(1, 0, 2)
        )
        return LossInfo(total_loss=loss, diagnostics_info=None)

    def info_nce_simclr(self, pred, encs, reduction="mean"):
        if self.args.normalize_z:
            pred = F.normalize(pred, dim=-1)
            encs = F.normalize(encs, dim=-1)

        features = torch.cat([pred, encs], dim=0)

        labels = torch.cat(
            [torch.arange(pred.shape[0]) for i in range(2)],
            dim=0,
        )
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(pred.device)

        similarity_matrix = torch.matmul(features, features.T)

        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(pred.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(
            similarity_matrix.shape[0], -1
        )

        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        negatives = similarity_matrix[~labels.bool()].view(
            similarity_matrix.shape[0], -1
        )

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(pred.device)

        logits = logits / self.args.loss_temp
        return F.cross_entropy(logits, labels, reduction=reduction)

    def info_nce_simclr_traj(self, pred, encs):
        """Trajectory-local InfoNCE.

        pred, encs: [B, N, D], where N is the number of time points.
        For each trajectory, positives are paired time steps and negatives
        are all other time steps within that trajectory.
        """
        if self.args.normalize_z:
            pred = F.normalize(pred, dim=-1)
            encs = F.normalize(encs, dim=-1)

        B, N, _ = pred.shape
        features = torch.cat([pred, encs], dim=1)  # [B, 2N, D]
        similarity = torch.matmul(features, features.transpose(1, 2))  # [B, 2N, 2N]

        labels = torch.cat(
            [torch.arange(N, device=pred.device) for _ in range(2)],
            dim=0,
        )
        labels = labels.unsqueeze(0) == labels.unsqueeze(1)  # [2N, 2N]

        mask = torch.eye(2 * N, dtype=torch.bool, device=pred.device)
        labels_wo_diag = labels[~mask].view(2 * N, 2 * N - 1)
        similarity_wo_diag = similarity[:, ~mask].view(B, 2 * N, 2 * N - 1)

        positive_mask = labels_wo_diag.unsqueeze(0).expand(B, -1, -1)
        positives = similarity_wo_diag[positive_mask].view(B, 2 * N, -1)
        negatives = similarity_wo_diag[~positive_mask].view(B, 2 * N, -1)

        logits = torch.cat([positives, negatives], dim=-1)
        logits = logits / self.args.loss_temp
        labels = torch.zeros(B * 2 * N, dtype=torch.long, device=pred.device)
        return F.cross_entropy(logits.reshape(B * 2 * N, -1), labels, reduction="mean")

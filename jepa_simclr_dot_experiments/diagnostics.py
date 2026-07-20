import torch
from torch.nn import functional as F
import numpy as np


@torch.no_grad()
def get_cov_std_corr(r: torch.Tensor):
    cov = torch.einsum("bf, be -> ef", r, r) / (r.shape[0] - 1)
    stds = diagonal(cov) ** 0.5
    corr = cov / (torch.einsum("i,j -> ij", stds, stds) + 1e-8)
    avg_corr = (off_diagonal(corr).abs()).mean().item()
    avg_cov = (off_diagonal(cov).abs()).mean().item()
    avg_std = (stds).sum().div(r.shape[1]).item()
    return avg_cov, avg_std, avg_corr


@torch.no_grad()
def get_cropped_std_diff(r: torch.Tensor) -> float:
    r_range = r.max(dim=0).values - r.min(dim=0).values + 1e-8
    r_adjusted = r - r.min(dim=0).values / r_range
    std_diff = (r_adjusted.std(dim=0) - np.sqrt(1 / 12)).abs().mean()
    return std_diff


def off_diagonal(x):
    n, m = x.shape[-2:]
    assert n == m
    return (
        x.flatten(-2)[..., :-1]
        .view(*x.shape[:-2], n - 1, n + 1)[..., :, 1:]
        .flatten(-2)
    )


def diagonal(x):
    n, m = x.shape[-2:]
    assert n == m
    return torch.einsum("...ii->...i", x)


@torch.no_grad()
def analyze_repr(gt: torch.Tensor, pred: torch.Tensor, normalize: bool = True):
    """Normalize tensors, compare their std, similarities and covariance.
    Not used for training, only for diagnostics."""
    gt_unit = gt.float() - gt.mean(dim=0)
    pred_unit = pred.float() - pred.mean(dim=0)

    if normalize:
        gt_unit = F.normalize(gt_unit.float(), dim=-1)
        pred_unit = F.normalize(pred_unit.float(), dim=-1)

    avg_cov_gt, avg_std_gt, avg_corr_gt = get_cov_std_corr(gt_unit)
    avg_cov_pred, avg_std_pred, avg_corr_pred = get_cov_std_corr(pred_unit)

    diff = ((gt - pred).float() ** 2).mean()
    assert len(diff.shape) == 0

    avg_cos_sim = (gt_unit * pred_unit).sum(dim=-1).mean()

    avg_gt_norm = gt.norm(p=2, dim=-1).mean()
    avg_pred_norm = pred.norm(p=2, dim=-1).mean()

    diff_sum = ((gt_unit - pred_unit).float() ** 2).sum()

    avg_std_feature_gt = torch.sqrt(gt_unit.var(dim=-1) + 0.0001).mean()
    avg_std_feature_pred = torch.sqrt(pred_unit.var(dim=-1) + 0.0001).mean()
    avg_std_feature = (avg_std_feature_gt + avg_std_feature_pred) / 2

    feature_std_target = 1 / np.sqrt(gt.shape[-1])
    avg_std_gap_feature_gt = (
        feature_std_target - avg_std_feature_gt
    ) / feature_std_target
    avg_std_gap_feature_pred = (
        feature_std_target - avg_std_feature_pred
    ) / feature_std_target

    batch_std_target = 1 / np.sqrt(gt.shape[0])
    avg_std_gap_batch_gt = (batch_std_target - avg_std_gt) / batch_std_target
    avg_std_gap_batch_pred = (batch_std_target - avg_std_pred) / batch_std_target

    avg_gt_std_diff = get_cropped_std_diff(gt)
    avg_pred_std_diff = get_cropped_std_diff(pred)

    return dict(
        diff=diff.item(),
        diff_sum=diff_sum.item(),
        avg_std_feature=avg_std_feature.item(),
        avg_std_feature_gt=avg_std_feature_gt.item(),
        avg_std_feature_pred=avg_std_feature_pred.item(),
        avg_cos_sim=avg_cos_sim.item(),
        avg_gt_norm=avg_gt_norm.item(),
        avg_pred_norm=avg_pred_norm.item(),
        avg_std_gap_feature_gt=avg_std_gap_feature_gt.item(),
        avg_std_gap_feature_pred=avg_std_gap_feature_pred.item(),
        avg_std_gap_batch_gt=avg_std_gap_batch_gt.item(),
        avg_std_gap_batch_pred=avg_std_gap_batch_pred.item(),
        avg_cov_gt=avg_cov_gt,
        avg_cov_pred=avg_cov_pred,
        avg_std_gt=avg_std_gt,
        avg_std_pred=avg_std_pred,
        avg_corr_gt=avg_corr_gt,
        avg_corr_pred=avg_corr_pred,
        avg_gt_std_diff=avg_gt_std_diff,
        avg_pred_std_diff=avg_pred_std_diff,
        # eigenvals_histogram=svd.detach().cpu().tolist(),
    )



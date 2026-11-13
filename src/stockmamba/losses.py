from __future__ import annotations

import torch


def paper_rank_position_loss(prediction: torch.Tensor, target: torch.Tensor, *, alpha: float = 0.5, gamma: float = 1.0, epsilon: float = 1e-8):
    valid = torch.isfinite(prediction) & torch.isfinite(target)
    prediction, target = prediction[valid], target[valid]
    if prediction.numel() < 30:
        loss = (prediction - target).square().mean()
        return loss, {"wmse": float(loss.detach()), "ic_loss": 0.0}
    ranks = torch.argsort(torch.argsort(target)).to(prediction.dtype) / float(prediction.numel() - 1)
    weights = 1.0 + gamma * (2.0 * ranks - 1.0).square()
    weights = weights / weights.mean().clamp_min(epsilon)
    wmse = (weights * (prediction - target).square()).mean()
    p = prediction - prediction.mean(); y = target - target.mean()
    ic_loss = 1.0 - (p * y).sum() / (torch.sqrt(p.square().sum() * y.square().sum()) + epsilon)
    loss = alpha * wmse + (1.0 - alpha) * ic_loss
    return loss, {"wmse": float(wmse.detach()), "ic_loss": float(ic_loss.detach())}

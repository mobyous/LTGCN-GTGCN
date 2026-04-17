import torch
from typing import Optional


def rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return torch.sqrt(torch.mean((pred - target) ** 2)).item()


def mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    return torch.abs(pred - target).mean().item()


def smape(pred: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> float:
    """
    Symmetric MAPE, masked to non-zero actual days.

    Zero-actual days (true case count = 0) are excluded: when actual=0 and pred>0,
    the ratio blows up to 200% and dominates the average, giving a meaningless number.
    This is common in COVID data (zero-case days in early pandemic, post-wave troughs).

    Returns 0-200 (%). NaN-safe: returns 0.0 if all actuals are zero.
    """
    mask = target > 0                                              # only non-zero actual days
    if mask.sum() == 0:
        return 0.0
    p = pred[mask]
    t = target[mask]
    denom = ((torch.abs(p) + torch.abs(t)) / 2).clamp(min=eps)
    return (torch.abs(p - t) / denom).mean().item() * 100


def mda(pred: torch.Tensor, target: torch.Tensor, last_known: torch.Tensor) -> float:
    """
    Mean Directional Accuracy.

    Measures how often the model correctly predicts the direction of change
    (up or down) relative to the last known value in the input window.

    Args:
        pred:       [S, N] — predicted case counts (real scale)
        target:     [S, N] — actual case counts (real scale)
        last_known: [S, N] — last timestep of input window (real scale)

    Returns:
        Fraction of (sample, node) pairs where predicted direction == actual direction.
        Range: 0.0 – 1.0.

    Notes:
        Zero-change forecasts are treated as a valid direction. This matters for
        persistence baselines, where pred == last_known by construction.
    """
    pred_dir   = (pred   - last_known).sign()
    actual_dir = (target - last_known).sign()
    return (pred_dir == actual_dir).float().mean().item()


def population_weighted_rmse(
    pred:       torch.Tensor,          # [S, N]
    target:     torch.Tensor,          # [S, N]
    pop_weights: torch.Tensor,         # [N] — raw population counts
) -> float:
    """
    Population-weighted average RMSE across nodes.

    Per-node RMSE weighted by population share, so high-population cities
    (São Paulo, Barcelona) count more than small provinces.

    Returns a single scalar in the same unit as pred/target (daily cases).
    """
    w = pop_weights.float() / pop_weights.float().sum()   # [N] normalised weights
    per_node_rmse = torch.sqrt(((pred - target) ** 2).mean(dim=0))  # [N]
    return (per_node_rmse * w).sum().item()


def compute_all(
    pred:        torch.Tensor,
    target:      torch.Tensor,
    last_known:  Optional[torch.Tensor] = None,
    pop_weights: Optional[torch.Tensor] = None,
) -> dict:
    metrics = {
        "rmse":     rmse(pred, target),
        "mae":      mae(pred, target),
        "smape":    smape(pred, target),
    }
    if last_known is not None:
        metrics["mda"] = mda(pred, target, last_known)
    if pop_weights is not None:
        metrics["pop_weighted_rmse"] = population_weighted_rmse(pred, target, pop_weights)
    return metrics

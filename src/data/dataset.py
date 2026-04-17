import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List
import pandas as pd


class CovidGraphDataset(Dataset):
    def __init__(self, X: torch.Tensor, Y: torch.Tensor):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def make_temporal_splits(
    df: pd.DataFrame,
    date_col: str,
    id_col: str,
    value_col: str,
    node_order: list,
    input_window: int = 14,
    output_window: int = 1,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    rolling_folds: int = 1,
    rolling_fold_index: int = 1,
    device: torch.device = torch.device("cpu"),
    log_transform: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Correct temporal splitting that prevents data leakage:

      1. Pivot df → [days, N] aligned to node_order.
      2. Split at the DAY level: train / val / test date ranges.
      3. Fit normalization statistics on TRAIN days only.
      4. Normalize all splits using train stats.
      5. Window each split independently — zero temporal overlap across boundaries.

    Returns:
        X_tr, Y_tr, X_vl, Y_vl, X_te, Y_te  — shape [S, window, N, 1]
        means, stds  — shape [N], train-split statistics for later inverse transform
    """
    pivot = (
        df.pivot(index=date_col, columns=id_col, values=value_col)
        .reindex(columns=node_order)
        .sort_index()
        .fillna(0.0)
    )
    data = pivot.to_numpy().astype(np.float32)  # [days, N]

    # Log1p compresses multi-wave dynamic range before z-scoring.
    # Prevents z-score blow-up on cities with near-zero training cases that spike in val/test.
    if log_transform:
        data = np.log1p(data)

    days, N = data.shape

    if rolling_folds < 1:
        raise ValueError(f"rolling_folds must be >= 1, got {rolling_folds}")
    if not (1 <= rolling_fold_index <= rolling_folds):
        raise ValueError(
            f"rolling_fold_index must be in [1, {rolling_folds}], got {rolling_fold_index}"
        )

    test_ratio = 1.0 - train_ratio - val_ratio
    if test_ratio <= 0:
        raise ValueError(
            f"Invalid split ratios: train={train_ratio}, val={val_ratio}, test={test_ratio}"
        )

    if rolling_folds == 1:
        train_end = int(days * train_ratio)
        val_end = int(days * (train_ratio + val_ratio))
        test_end = days
    else:
        initial_train_ratio = train_ratio - (rolling_folds - 1) * test_ratio
        if initial_train_ratio <= 0:
            raise ValueError(
                "rolling_folds is too large for the current train/val/test ratios. "
                f"Got initial_train_ratio={initial_train_ratio:.4f}."
            )
        train_end_ratio = initial_train_ratio + (rolling_fold_index - 1) * test_ratio
        val_end_ratio = train_end_ratio + val_ratio
        test_end_ratio = val_end_ratio + test_ratio

        train_end = int(days * train_end_ratio)
        val_end = int(days * val_end_ratio)
        test_end = min(days, int(days * test_end_ratio))

    train_data = data[:train_end]
    val_data   = data[train_end:val_end]
    test_data  = data[val_end:test_end]

    # Normalize: fit ONLY on train split
    means = train_data.mean(axis=0)                      # [N]
    stds  = train_data.std(axis=0)                       # [N]
    stds  = np.where(stds < 1e-8, 1.0, stds)            # constant nodes → unit std

    train_norm = (train_data - means) / stds
    val_norm   = (val_data   - means) / stds
    test_norm  = (test_data  - means) / stds

    def _window(arr: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        xs, ys = [], []
        max_i = len(arr) - input_window - output_window
        for i in range(max_i + 1):
            xs.append(arr[i : i + input_window])
            ys.append(arr[i + input_window : i + input_window + output_window])
        if not xs:
            return (
                torch.zeros(0, input_window, N, 1, dtype=torch.float32),
                torch.zeros(0, output_window, N, 1, dtype=torch.float32),
            )
        X = torch.tensor(np.expand_dims(np.stack(xs), -1), dtype=torch.float32)
        Y = torch.tensor(np.expand_dims(np.stack(ys), -1), dtype=torch.float32)
        return X.to(device), Y.to(device)

    X_tr, Y_tr = _window(train_norm)
    X_vl, Y_vl = _window(val_norm)
    X_te, Y_te = _window(test_norm)

    fold_suffix = (
        f" [rolling fold {rolling_fold_index}/{rolling_folds}]"
        if rolling_folds > 1 else ""
    )
    print(
        f"[dataset] train={len(X_tr)}, val={len(X_vl)}, test={len(X_te)} windows "
        f"(days: {train_end} / {val_end - train_end} / {test_end - val_end})"
        f"{fold_suffix}"
    )

    means_t = torch.tensor(means, dtype=torch.float32)
    stds_t  = torch.tensor(stds,  dtype=torch.float32)
    return X_tr, Y_tr, X_vl, Y_vl, X_te, Y_te, means_t, stds_t


def make_loaders(
    X_tr, Y_tr, X_vl, Y_vl, X_te, Y_te,
    batch_size: int = 32,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_loader = DataLoader(CovidGraphDataset(X_tr, Y_tr), batch_size=batch_size, shuffle=True,  drop_last=False)
    val_loader   = DataLoader(CovidGraphDataset(X_vl, Y_vl), batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader  = DataLoader(CovidGraphDataset(X_te, Y_te), batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, val_loader, test_loader

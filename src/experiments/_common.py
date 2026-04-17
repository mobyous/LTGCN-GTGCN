"""
Shared setup utilities for all experiments.
"""
import os
import random
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, List

from src.config import TrainConfig

ROOT = Path(__file__).resolve().parent.parent.parent


def get_device(preference: str = "auto") -> torch.device:
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(preference)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    os.environ["PYTHONHASHSEED"]        = str(seed)


def build_optimizer_and_scheduler(
    model: torch.nn.Module,
    cfg: TrainConfig,
) -> Tuple[torch.optim.Optimizer, object]:
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    # ReduceLROnPlateau: halves LR when EMA-smoothed val loss stops improving.
    # Better than StepLR for noisy val sets (small N → few val batches → high variance).
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg.scheduler_gamma,        # multiply LR by this on plateau
        patience=cfg.scheduler_step,       # epochs to wait before reducing
        min_lr=1e-6,
    )
    return optimizer, scheduler


def load_brazil_pop_weights(node_order: List[int]) -> torch.Tensor:
    """Return [N] population tensor aligned to node_order (Brazil ibgeIDs)."""
    pop_df  = pd.read_csv(ROOT / "data" / "cleaned_population_2022.csv")
    pop_map = dict(zip(pop_df["ibgeID"].astype(int), pop_df["population"].astype(float)))
    return torch.tensor([pop_map.get(c, 1.0) for c in node_order], dtype=torch.float32)


def load_spain_pop_weights(node_order: List[int], name_to_cod: dict) -> torch.Tensor:
    """Return [N] population tensor aligned to node_order (Spain cod_ine ints)."""
    pop_df     = pd.read_csv(ROOT / "data" / "Spain" / "final_cleaned_population_by_province_2025.csv")
    pop_map    = dict(zip(pop_df["Province"], pop_df["Population_2025"].astype(float)))
    cod_to_name = {v: k for k, v in name_to_cod.items()}
    return torch.tensor(
        [pop_map.get(cod_to_name.get(c, ""), 1.0) for c in node_order],
        dtype=torch.float32,
    )


def maybe_compile(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    """
    Apply torch.compile on CUDA only.
    MPS already provides Metal GPU acceleration; inductor backend doesn't support MPS.
    CPU gets no benefit from compilation overhead.
    """
    if device.type != "cuda":
        print(f"[compile] skipped (device={device}, inductor requires CUDA)", flush=True)
        return model
    try:
        compiled = torch.compile(model, dynamic=False)
        print(f"[compile] torch.compile enabled on {device}", flush=True)
        return compiled
    except Exception as e:
        print(f"[compile] skipped: {e}", flush=True)
        return model


def get_edge_tensors(pyg_data) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    edge_index  = pyg_data.edge_index
    edge_weight = pyg_data.edge_attr.squeeze(-1) if pyg_data.edge_attr is not None else None
    return edge_index, edge_weight

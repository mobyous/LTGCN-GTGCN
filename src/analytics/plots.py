import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional


def plot_losses(
    train_losses: List[float],
    val_losses: List[float],
    title: str = "Training",
    save_path: Optional[str] = None,
):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(train_losses, label="train", linewidth=1.5)
    ax.plot(val_losses,   label="val",   linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss on Normalized Targets")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[plot] {save_path}")
    plt.close(fig)


def plot_city_predictions(
    preds: np.ndarray,
    actuals: np.ndarray,
    city_name: str,
    save_path: Optional[str] = None,
):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(actuals, label="actual",    linewidth=1.5)
    ax.plot(preds,   label="predicted", linewidth=1.5, linestyle="--")
    ax.set_title(f"One-Step Forecast on {city_name}")
    ax.set_xlabel("Forecast Step in Test Period (Days)")
    ax.set_ylabel("Daily New COVID-19 Cases (Original Scale)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[plot] {save_path}")
    plt.close(fig)


def plot_per_city_metrics(
    city_names: list,
    rmses: list,
    smapes: list,
    title: str = "",
    save_path: Optional[str] = None,
):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    idx = np.argsort(rmses)[::-1][:20]          # top-20 worst by RMSE
    names = [city_names[i] for i in idx]

    axes[0].barh(names, [rmses[i] for i in idx])
    axes[0].set_xlabel("RMSE (Daily Cases)")
    axes[0].set_ylabel("City / Province")
    axes[0].set_title(f"Top-20 worst RMSE {title}")

    axes[1].barh(names, [smapes[i] for i in idx])
    axes[1].set_xlabel("SMAPE (%)")
    axes[1].set_ylabel("City / Province")
    axes[1].set_title(f"Top-20 worst SMAPE {title}")

    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[plot] {save_path}")
    plt.close(fig)

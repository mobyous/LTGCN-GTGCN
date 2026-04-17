import time
import sys
import os
import resource
import subprocess
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, List, Tuple

from src.training.metrics import compute_all


class Trainer:
    """
    Unified trainer for all GNN COVID models.

    Fixes vs original notebooks:
    - Explicit model.train() / model.eval() on every epoch.
    - Validation loss on val set (not test set) during training.
    - Test metrics on DENORMALIZED real case counts using train-split stats only.
    - MDA computed from last known input timestep (also denormalized).
    - Population-weighted RMSE when pop_weights provided.
    - Gradient clipping applied.
    """

    def __init__(
        self,
        model: nn.Module,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor],
        optimizer: torch.optim.Optimizer,
        scheduler,
        device: torch.device,
        means: torch.Tensor,                        # [N] train-split means
        stds:  torch.Tensor,                        # [N] train-split stds
        node_features: Optional[torch.Tensor] = None,
        pop_weights:   Optional[torch.Tensor] = None,   # [N] raw population per node
        grad_clip: float = 1.0,
        output_window: int = 1,
        log_transform: bool = False,                # must match dataset log_transform flag
    ):
        self.model         = model
        self.edge_index    = edge_index.to(device)
        self.edge_weight   = edge_weight.to(device) if edge_weight is not None else None
        self.optimizer     = optimizer
        self.scheduler     = scheduler
        self.device        = device
        self.means         = means.to(device)
        self.stds          = stds.to(device)
        self.node_features = node_features.to(device) if node_features is not None else None
        self.pop_weights   = pop_weights.cpu()      if pop_weights   is not None else None
        self.grad_clip     = grad_clip
        self.output_window = output_window
        self.log_transform = log_transform
        self.criterion     = nn.MSELoss()

        self.train_losses: List[float] = []
        self.val_losses:   List[float] = []
        self.epoch_times:  List[float] = []
        self.total_train_time_sec: float = 0.0
        self.avg_epoch_time_sec: float = 0.0
        self.epochs_ran: int = 0
        self.peak_torch_allocated_bytes: int = 0
        self.peak_torch_allocated_gb: float = 0.0
        self.peak_gpu_vram_bytes: int = 0
        self.peak_gpu_vram_gb: float = 0.0
        self.peak_tracked_memory_kind: str = "unknown"
        self.best_val_loss: float = float("inf")
        self.best_epoch: int = 0

    def _process_rss_bytes(self) -> int:
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS reports bytes, Linux reports kilobytes
        return int(usage if sys.platform == "darwin" else usage * 1024)

    def _reset_peak_memory_stats(self):
        self.peak_torch_allocated_bytes = 0
        self.peak_gpu_vram_bytes = 0
        if self.device.type == "cuda" and torch.cuda.is_available():
            self.peak_tracked_memory_kind = "torch_cuda_allocated"
            torch.cuda.reset_peak_memory_stats(self.device)
        elif self.device.type == "mps":
            self.peak_tracked_memory_kind = "torch_mps_allocated"
            if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                try:
                    torch.mps.empty_cache()
                except Exception:
                    pass
        else:
            self.peak_tracked_memory_kind = "process_rss"
        self._update_peak_memory_stats()

    def _current_mps_memory_bytes(self) -> int:
        if not hasattr(torch, "mps"):
            return 0
        current = 0
        if hasattr(torch.mps, "current_allocated_memory"):
            try:
                current = max(current, int(torch.mps.current_allocated_memory()))
            except Exception:
                pass
        if hasattr(torch.mps, "driver_allocated_memory"):
            try:
                current = max(current, int(torch.mps.driver_allocated_memory()))
            except Exception:
                pass
        return current

    def _current_cuda_process_vram_bytes(self) -> int:
        if self.device.type != "cuda" or not torch.cuda.is_available():
            return 0

        gpu_index = self.device.index if self.device.index is not None else torch.cuda.current_device()
        pid = os.getpid()
        cmd = [
            "nvidia-smi",
            f"--id={gpu_index}",
            "--query-compute-apps=pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ]
        try:
            proc = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            return 0

        total_mib = 0
        for line in proc.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 2:
                continue
            try:
                row_pid = int(parts[0])
                used_mib = int(parts[1])
            except ValueError:
                continue
            if row_pid == pid:
                total_mib += used_mib
        return total_mib * 1024 * 1024

    def _update_peak_memory_stats(self):
        if self.device.type == "cuda" and torch.cuda.is_available():
            self.peak_torch_allocated_bytes = max(
                self.peak_torch_allocated_bytes,
                int(torch.cuda.max_memory_allocated(self.device)),
            )
            self.peak_gpu_vram_bytes = max(
                self.peak_gpu_vram_bytes,
                self._current_cuda_process_vram_bytes(),
            )
        elif self.device.type == "mps":
            self.peak_torch_allocated_bytes = max(
                self.peak_torch_allocated_bytes,
                self._current_mps_memory_bytes(),
            )
        else:
            self.peak_torch_allocated_bytes = max(
                self.peak_torch_allocated_bytes,
                self._process_rss_bytes(),
            )

    def _forward(self, batch_X: torch.Tensor) -> torch.Tensor:
        kwargs = dict(edge_index=self.edge_index, edge_weight=self.edge_weight)
        if self.node_features is not None:
            kwargs["node_features"] = self.node_features
        return self.model(batch_X, **kwargs)

    def _capture_model_state(self):
        return {
            key: value.detach().cpu().clone()
            for key, value in self.model.state_dict().items()
        }

    def _restore_model_state(self, state_dict):
        self.model.load_state_dict(state_dict)

    def _squeeze_y(self, by: torch.Tensor) -> torch.Tensor:
        return by.squeeze(1) if self.output_window == 1 else by

    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total, count = 0.0, 0
        for bx, by in loader:
            bx = bx.to(self.device)
            by = self._squeeze_y(by.to(self.device))
            self.optimizer.zero_grad(set_to_none=True)
            loss = self.criterion(self._forward(bx), by)
            loss.backward()
            if self.grad_clip:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            self._update_peak_memory_stats()
            total += loss.item()
            count += 1
        return total / max(count, 1)

    @torch.no_grad()
    def _eval_epoch(self, loader: DataLoader) -> float:
        self.model.eval()
        total, count = 0.0, 0
        for bx, by in loader:
            bx = bx.to(self.device)
            by = self._squeeze_y(by.to(self.device))
            total += self.criterion(self._forward(bx), by).item()
            self._update_peak_memory_stats()
            count += 1
        return total / max(count, 1)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int):
        print(f"[trainer] starting: {epochs} epochs, {len(train_loader)} train batches, "
              f"{len(val_loader)} val batches  (first epoch may be slow — MPS compiles Metal shaders on first use)",
              flush=True)
        fit_start = time.time()
        self._reset_peak_memory_stats()
        ema_val    = None
        best_state = None
        is_plateau = isinstance(self.scheduler,
                                torch.optim.lr_scheduler.ReduceLROnPlateau)
        for epoch in range(1, epochs + 1):
            t0         = time.time()
            train_loss = self._train_epoch(train_loader)
            val_loss   = self._eval_epoch(val_loader)
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            # EMA-smoothed val fed to plateau scheduler — less reactive to single-epoch spikes
            ema_val = val_loss if ema_val is None else 0.3 * val_loss + 0.7 * ema_val

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                best_state = self._capture_model_state()

            if is_plateau:
                self.scheduler.step(ema_val)
            else:
                self.scheduler.step()

            lr = self.optimizer.param_groups[0]["lr"]
            epoch_time = time.time() - t0
            self.epoch_times.append(epoch_time)
            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"train={train_loss:.4f} | val={val_loss:.4f} | "
                f"ema={ema_val:.4f} | lr={lr:.2e} | "
                f"{epoch_time:.1f}s",
                flush=True,
            )
        if best_state is not None:
            self._restore_model_state(best_state)
        self.total_train_time_sec = time.time() - fit_start
        self.epochs_ran = epochs
        self.avg_epoch_time_sec = self.total_train_time_sec / epochs if epochs > 0 else 0.0
        self.peak_torch_allocated_gb = self.peak_torch_allocated_bytes / (1024 ** 3)
        self.peak_gpu_vram_gb = self.peak_gpu_vram_bytes / (1024 ** 3)

    @torch.no_grad()
    def test(self, test_loader: DataLoader) -> Tuple[dict, torch.Tensor, torch.Tensor]:
        """
        Evaluate on test set.
        Metrics reported on DENORMALIZED real case counts (train-split stats only).
        Includes MDA (directional accuracy) and population-weighted RMSE if available.
        """
        self.model.eval()
        all_pred, all_true, all_last = [], [], []

        for bx, by in test_loader:
            bx = bx.to(self.device)                      # [B, T, N, 1]
            by = self._squeeze_y(by.to(self.device))
            all_pred.append(self._forward(bx).cpu())
            all_true.append(by.cpu())
            # Last known value: final timestep of input window, shape [B, N]
            all_last.append(bx[:, -1, :, 0].cpu())

        preds = torch.cat(all_pred, dim=0).squeeze(-1)   # [S, N]
        trues = torch.cat(all_true, dim=0).squeeze(-1)   # [S, N]
        lasts = torch.cat(all_last, dim=0)               # [S, N] — normalized scale

        # Denormalize with TRAIN-ONLY stats → real case counts
        means = self.means.cpu()
        stds  = self.stds.cpu()
        preds_real = preds * stds + means
        trues_real = trues * stds + means
        lasts_real = lasts * stds + means
        if self.log_transform:
            # Invert log1p: expm1 = exp(x) - 1
            preds_real = torch.expm1(preds_real)
            trues_real = torch.expm1(trues_real)
            lasts_real = torch.expm1(lasts_real)
        preds_real = preds_real.clamp(min=0)
        trues_real = trues_real.clamp(min=0)
        lasts_real = lasts_real.clamp(min=0)

        metrics = compute_all(
            pred=preds_real,
            target=trues_real,
            last_known=lasts_real,
            pop_weights=self.pop_weights,
        )

        print(
            f"\n[TEST] RMSE={metrics['rmse']:.2f} | "
            f"MAE={metrics['mae']:.2f} | "
            f"SMAPE={metrics['smape']:.2f}% | "
            f"MDA={metrics.get('mda', float('nan')):.3f}",
            flush=True,
        )
        if "pop_weighted_rmse" in metrics:
            print(
                f"[TEST] Pop-weighted RMSE={metrics['pop_weighted_rmse']:.2f}",
                flush=True,
            )

        return metrics, preds_real, trues_real

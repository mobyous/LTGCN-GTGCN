import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


def _batch_edge_index(edge_index: torch.Tensor, edge_weight, B: int, N: int):
    """
    Replicate a single graph's edge_index for a batch of B identical graphs.
    Offsets node IDs by i*N so all B graphs sit in one [B*N]-node graph.
    Returns: batched_edge_index [2, B*E], batched_edge_weight [B*E] or None.
    """
    E       = edge_index.size(1)
    offsets = torch.arange(B, device=edge_index.device).repeat_interleave(E) * N
    ei_b    = edge_index.repeat(1, B) + offsets.unsqueeze(0)   # [2, B*E]
    ew_b    = edge_weight.repeat(B) if edge_weight is not None else None
    return ei_b, ew_b


class GConvGRUCell(nn.Module):
    """
    Graph-Convolutional GRU cell.

    Optimisations vs naive implementation:
    - Vectorised batch: one GCNConv call over [B*N] instead of B separate calls.
    - Fused z/r gate: conv_z and conv_r share identical input [x,h],
      so one GCNConv(C, 2H) replaces two GCNConv(C, H) calls.
      Net result: 2 GCN propagations per timestep instead of 3.
    """

    def __init__(self, in_channels: int, hidden_channels: int):
        super().__init__()
        self.hidden_channels = hidden_channels
        C = in_channels + hidden_channels
        # z and r share input [x, h]  → fuse into one conv
        self.conv_zr = GCNConv(C, 2 * hidden_channels)
        # h_tilde needs [x, r*h]      → separate conv
        self.conv_h  = GCNConv(C, hidden_channels)

    # Threshold above which the B*N super-graph becomes too large for MPS scatter ops.
    # Below this: vectorised (one GCNConv call). Above: loop over B (safe for any N).
    _VECTORISE_THRESHOLD = 50_000

    def forward(self, x, h, edge_index, edge_weight=None):
        B, N, _ = x.size()
        H       = self.hidden_channels

        if B * N <= self._VECTORISE_THRESHOLD:
            # ── Vectorised path: one GCNConv call over the B*N super-graph ────
            ei_b, ew_b = _batch_edge_index(edge_index, edge_weight, B, N)
            x_flat = x.reshape(B * N, -1)
            h_flat = h.reshape(B * N, -1)
            xh     = torch.cat([x_flat, h_flat], dim=-1)
            zr     = self.conv_zr(xh, ei_b, ew_b)
            z      = torch.sigmoid(zr[:, :H])
            r      = torch.sigmoid(zr[:, H:])
            h_tilde = torch.tanh(self.conv_h(
                torch.cat([x_flat, r * h_flat], dim=-1), ei_b, ew_b
            ))
            return ((1 - z) * h_flat + z * h_tilde).reshape(B, N, H)
        else:
            # ── Loop path: one GCNConv call per sample (safe for large N) ─────
            h_next = []
            for b in range(B):
                xh = torch.cat([x[b], h[b]], dim=-1)          # [N, C]
                zr = self.conv_zr(xh, edge_index, edge_weight) # [N, 2H]
                z  = torch.sigmoid(zr[:, :H])
                r  = torch.sigmoid(zr[:, H:])
                h_tilde = torch.tanh(self.conv_h(
                    torch.cat([x[b], r * h[b]], dim=-1), edge_index, edge_weight
                ))
                h_next.append((1 - z) * h[b] + z * h_tilde)
            return torch.stack(h_next, dim=0)                  # [B, N, H]


class GCRN(nn.Module):
    def __init__(self, in_channels: int = 1, hidden_channels: int = 64, out_channels: int = 1):
        super().__init__()
        self.cell = GConvGRUCell(in_channels, hidden_channels)
        self.head = nn.Sequential(nn.ReLU(), nn.Linear(hidden_channels, out_channels))

    def forward(self, x_seq, edge_index, edge_weight=None, **_):
        if x_seq.dim() == 3:
            x_seq = x_seq.unsqueeze(0)
        B, T, N, _ = x_seq.size()
        h = torch.zeros(B, N, self.cell.hidden_channels, device=x_seq.device)
        for t in range(T):
            h = self.cell(x_seq[:, t], h, edge_index, edge_weight)
        return self.head(h)   # [B, N, out_channels]

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from src.models.gcrn import _batch_edge_index


class GConvLSTMCell(nn.Module):
    """
    Graph-Convolutional LSTM cell.

    Optimisations:
    - Vectorised batch: one call over [B*N] nodes.
    - ALL four gates (i, f, o, g) share the same input [x, h],
      so one GCNConv(C, 4H) replaces four GCNConv(C, H) calls.
      Net result: 1 GCN propagation per timestep instead of 4.
    """

    def __init__(self, in_channels: int, hidden_channels: int):
        super().__init__()
        self.hidden_channels = hidden_channels
        C = in_channels + hidden_channels
        # All 4 gates share input [x, h] → one fused conv
        self.conv_ifog = GCNConv(C, 4 * hidden_channels)

    _VECTORISE_THRESHOLD = 50_000

    def forward(self, x, h, c, edge_index, edge_weight=None):
        B, N, _ = x.size()
        H       = self.hidden_channels

        if B * N <= self._VECTORISE_THRESHOLD:
            # ── Vectorised path ───────────────────────────────────────────────
            ei_b, ew_b = _batch_edge_index(edge_index, edge_weight, B, N)
            xh   = torch.cat([x.reshape(B * N, -1), h.reshape(B * N, -1)], dim=-1)
            ifog = self.conv_ifog(xh, ei_b, ew_b)
            i = torch.sigmoid(ifog[:, :H])
            f = torch.sigmoid(ifog[:, H:2*H])
            o = torch.sigmoid(ifog[:, 2*H:3*H])
            g = torch.tanh(   ifog[:, 3*H:])
            c_flat = c.reshape(B * N, -1)
            c_next = f * c_flat + i * g
            h_next = o * torch.tanh(c_next)
            return h_next.reshape(B, N, H), c_next.reshape(B, N, H)
        else:
            # ── Loop path ─────────────────────────────────────────────────────
            h_out, c_out = [], []
            for b in range(B):
                xh   = torch.cat([x[b], h[b]], dim=-1)
                ifog = self.conv_ifog(xh, edge_index, edge_weight)
                i = torch.sigmoid(ifog[:, :H])
                f = torch.sigmoid(ifog[:, H:2*H])
                o = torch.sigmoid(ifog[:, 2*H:3*H])
                g = torch.tanh(   ifog[:, 3*H:])
                cn = f * c[b] + i * g
                hn = o * torch.tanh(cn)
                h_out.append(hn);  c_out.append(cn)
            return torch.stack(h_out, dim=0), torch.stack(c_out, dim=0)


class GCRNLSTM(nn.Module):
    def __init__(self, in_channels: int = 1, hidden_channels: int = 64, out_channels: int = 1):
        super().__init__()
        self.cell = GConvLSTMCell(in_channels, hidden_channels)
        self.head = nn.Sequential(nn.ReLU(), nn.Linear(hidden_channels, out_channels))

    def forward(self, x_seq, edge_index, edge_weight=None, **_):
        if x_seq.dim() == 3:
            x_seq = x_seq.unsqueeze(0)
        B, T, N, _ = x_seq.size()
        h = torch.zeros(B, N, self.cell.hidden_channels, device=x_seq.device)
        c = torch.zeros_like(h)
        for t in range(T):
            h, c = self.cell(x_seq[:, t], h, c, edge_index, edge_weight)
        return self.head(h)   # [B, N, out_channels]

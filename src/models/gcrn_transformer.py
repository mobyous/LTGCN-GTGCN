import torch
import torch.nn as nn
from src.models.gcrn import GConvGRUCell


class GCRNTransformer(nn.Module):
    """Per-node temporal Transformer encoder → GConvGRU spatial recurrence."""

    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 64,
        out_channels: int = 1,
        transformer_dim: int = 64,
        nhead: int = 4,
        num_layers: int = 1,
    ):
        super().__init__()
        self.proj_in  = nn.Linear(in_channels, transformer_dim)
        enc = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=nhead,
            dim_feedforward=transformer_dim * 4,
            batch_first=True,
            dropout=0.1,
        )
        self.transformer = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.proj_out = nn.Linear(transformer_dim, in_channels)
        self.gru  = GConvGRUCell(in_channels, hidden_channels)
        self.head = nn.Sequential(nn.ReLU(), nn.Linear(hidden_channels, out_channels))

    def forward(self, x_seq, edge_index, edge_weight=None, **_):
        if x_seq.dim() == 3:
            x_seq = x_seq.unsqueeze(0)
        B, T, N, F = x_seq.size()

        # Temporal encoding per node: [B*N, T, F] → [B*N, T, F]
        x_flat = x_seq.permute(0, 2, 1, 3).reshape(B * N, T, F)
        x_flat = self.proj_out(self.transformer(self.proj_in(x_flat)))  # [BN, T, F]
        x_seq  = x_flat.reshape(B, N, T, F).permute(0, 2, 1, 3)        # [B, T, N, F]

        # Spatial recurrence
        h = torch.zeros(B, N, self.gru.hidden_channels, device=x_seq.device)
        for t in range(T):
            h = self.gru(x_seq[:, t], h, edge_index, edge_weight)
        return self.head(h)  # [B, N, out_channels]

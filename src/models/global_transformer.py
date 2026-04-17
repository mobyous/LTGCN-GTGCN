import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class _GCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, gat_heads=2):
        super().__init__()
        self.gcn1 = GCNConv(in_channels, hidden_channels)
        self.gcn2 = GCNConv(hidden_channels, hidden_channels)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, edge_weight=None):
        x = self.relu(self.gcn1(x, edge_index, edge_weight))
        return self.relu(self.gcn2(x, edge_index, edge_weight))  # [N, D]


class GlobalTransformer(nn.Module):
    """
    Global spatiotemporal Transformer.

    Fixes applied vs original:
    - CORRECT causal mask: allows all nodes at the same timestep to attend to each other,
      only blocks attention to strictly future timesteps.
    - Sinusoidal (deterministic) positional encoding instead of random nn.Parameter.
    - Decoder dimensions scale with hidden_dim instead of being hardcoded.
    """

    def __init__(
        self,
        input_dim: int = 1,
        gcn_dim: int = 1,
        hidden_dim: int = 64,
        nhead: int = 4,
        num_layers: int = 1,
        num_nodes: int = None,
        forecast_dim: int = 1,
        time_pos_len: int = 1000,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.2,
        gat_heads: int = 2,
    ):
        super().__init__()
        assert num_nodes, "num_nodes required"
        assert hidden_dim % nhead == 0, "hidden_dim must be divisible by nhead"

        self.num_nodes  = num_nodes
        self.hidden_dim = hidden_dim
        self.use_gcn    = gcn_dim > 0

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        if self.use_gcn:
            self.spatial_encoder = _GCNEncoder(gcn_dim, hidden_dim, gat_heads)
        else:
            self.register_buffer("_dummy", torch.zeros(1))

        # Sinusoidal positional encoding — fully deterministic, not a learnable parameter
        pos = torch.zeros(time_pos_len, hidden_dim)
        position = torch.arange(time_pos_len).float().unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / hidden_dim)
        )
        pos[:, 0::2] = torch.sin(position * div_term)
        pos[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("time_pos", pos)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, dim_feedforward=4*hidden_dim,
            dropout=ff_dropout, batch_first=True,
        )
        if attn_dropout > 0:
            enc_layer.self_attn.dropout = nn.Dropout(attn_dropout)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # Decoder scales with hidden_dim
        mid = max(hidden_dim // 2, 32)
        low = max(hidden_dim // 8, 16)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, mid),
            nn.ReLU(),
            nn.Linear(mid, low),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(low, forecast_dim),
        )

    def _make_causal_mask(self, T: int, N: int, device) -> torch.Tensor:
        """
        Correct spatiotemporal causal mask.

        Positions are ordered: [t=0,n=0 ... t=0,n=N-1, t=1,n=0 ... t=T-1,n=N-1].
        mask[i, j] = True → position i CANNOT attend to position j.

        Rule: block when j's timestep is strictly greater than i's timestep.
        Within the same timestep all N nodes can freely attend to each other.
        """
        seq   = T * N
        t_idx = torch.arange(seq, device=device) // N   # [seq] — timestep per position
        # t_idx[j] > t_idx[i]  ←→  t_idx[i] < t_idx[j]
        mask  = t_idx.unsqueeze(1) < t_idx.unsqueeze(0) # [seq, seq]: mask[i,j] = (t_i < t_j)
        return mask

    def forward(self, x_seq, edge_index, edge_weight=None, node_features=None, **_):
        B, T, N, F = x_seq.shape
        assert N == self.num_nodes, f"Expected {self.num_nodes} nodes, got {N}"

        x = self.input_proj(x_seq)  # [B, T, N, D]

        if self.use_gcn:
            assert node_features is not None, "node_features required when gcn_dim > 0"
            sp = self.spatial_encoder(node_features, edge_index, edge_weight)  # [N, D]
            sp = sp.unsqueeze(0).unsqueeze(1).expand(B, T, -1, -1)             # [B, T, N, D]
        else:
            sp = torch.zeros(B, T, N, self.hidden_dim, device=x_seq.device)

        # Temporal positional encoding: same D-vector for all N nodes at each timestep
        tp = self.time_pos[:T].unsqueeze(1).expand(-1, N, -1)  # [T, N, D]
        tp = tp.unsqueeze(0).expand(B, -1, -1, -1)             # [B, T, N, D]

        x = (x + sp + tp).reshape(B, T * N, self.hidden_dim)   # [B, T*N, D]

        mask = self._make_causal_mask(T, N, x.device)
        x    = self.transformer(x, mask=mask)                   # [B, T*N, D]
        x    = x.reshape(B, T, N, self.hidden_dim)
        x    = x[:, -1, :, :]                                   # [B, N, D] — last timestep only
        return self.decoder(x)                                  # [B, N, forecast_dim]

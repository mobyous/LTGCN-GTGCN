import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class _GCNEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.gcn1 = GCNConv(in_channels, out_channels)
        self.gcn2 = GCNConv(out_channels, out_channels)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, edge_weight=None):
        x = self.relu(self.gcn1(x, edge_index, edge_weight))
        return self.relu(self.gcn2(x, edge_index, edge_weight))  # [N, D]


class _TemporalTransformer(nn.Module):
    """
    Per-node causal temporal transformer. Processes each node's time series independently
    using shared transformer weights.

    For large graphs (N >> 100) the [B*N, T, D] tensor would OOM. We process it in
    chunks of at most `node_chunk` rows so peak memory is bounded regardless of N.
    """

    def __init__(self, in_channels, hidden_dim=64, nhead=4, num_layers=1,
                 max_len=200, node_chunk=1024):
        super().__init__()
        self.proj       = nn.Linear(in_channels, hidden_dim)
        self.hidden_dim = hidden_dim
        self.node_chunk = node_chunk

        # Sinusoidal position encoding — deterministic, no random init
        pos = torch.zeros(max_len, hidden_dim)
        position = torch.arange(max_len).float().unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / hidden_dim)
        )
        pos[:, 0::2] = torch.sin(position * div_term)
        pos[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pos_enc", pos)

        enc = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, dim_feedforward=4*hidden_dim,
            batch_first=True, dropout=0.1,
        )
        self.transformer = nn.TransformerEncoder(enc, num_layers=num_layers)

    def forward(self, x):
        # x: [B, T, N, F]
        B, T, N, F = x.size()
        # Flatten to [B*N, T, F] then project + add positional encoding
        x_flat = x.permute(0, 2, 1, 3).reshape(B * N, T, F)   # [BN, T, F]
        x_flat = self.proj(x_flat) + self.pos_enc[:T]           # [BN, T, D]

        # Chunked forward: keeps peak FFN memory = node_chunk * T * 4D bytes
        # instead of B*N * T * 4D bytes — safe for large N (e.g. Brazil N≈5K)
        outs = []
        for start in range(0, B * N, self.node_chunk):
            chunk = x_flat[start : start + self.node_chunk]     # [c, T, D]
            outs.append(self.transformer(chunk)[:, -1, :])      # [c, D]
        return torch.cat(outs, dim=0).reshape(B, N, self.hidden_dim)  # [B, N, D]


class _FusionAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)

    def forward(self, temporal, spatial):
        # Both [B, N, D]
        attn = torch.softmax((self.q(temporal) * self.k(spatial)).sum(-1, keepdim=True), dim=1)
        return temporal + attn * self.v(spatial)


class LocalTransformer(nn.Module):
    """
    Local spatiotemporal model:
    - Per-node temporal Transformer (independent time series encoding)
    - GCN spatial encoder on learned node embeddings
    - Attention-based fusion of temporal and spatial representations
    """

    def __init__(
        self,
        in_channels: int = 1,
        graph_feat_dim: int = 1,
        trans_hidden: int = 64,
        out_channels: int = 1,
        num_nodes: int = None,
        nhead: int = 4,
        num_layers: int = 1,
    ):
        super().__init__()
        assert num_nodes, "num_nodes required for node embeddings"
        self.trans_hidden = trans_hidden

        self.node_emb  = nn.Embedding(num_nodes, graph_feat_dim)
        self.spatial   = _GCNEncoder(graph_feat_dim, trans_hidden)
        self.temporal  = _TemporalTransformer(in_channels, trans_hidden, nhead=nhead, num_layers=num_layers)
        self.fusion    = _FusionAttention(trans_hidden)
        self.head      = nn.Sequential(
            nn.Linear(trans_hidden, 32),
            nn.ReLU(),
            nn.Linear(32, out_channels),
        )

    def forward(self, x_seq, edge_index, edge_weight=None, node_indices=None, **_):
        B = x_seq.size(0)
        N = x_seq.size(2)

        temporal_repr = self.temporal(x_seq)  # [B, N, D]

        if node_indices is None:
            node_indices = torch.arange(N, device=x_seq.device)
        node_feats   = self.node_emb(node_indices)                       # [N, feat_dim]
        spatial_repr = self.spatial(node_feats, edge_index, edge_weight) # [N, D]
        spatial_repr = spatial_repr.unsqueeze(0).expand(B, -1, -1)       # [B, N, D]

        fused = self.fusion(temporal_repr, spatial_repr)
        return self.head(fused)  # [B, N, out_channels]

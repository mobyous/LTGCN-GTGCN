import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class _GCNEncoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.gcn1 = GCNConv(in_channels, out_channels)
        self.gcn2 = GCNConv(out_channels, out_channels)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, edge_weight=None):
        x = self.relu(self.gcn1(x, edge_index, edge_weight))
        return self.relu(self.gcn2(x, edge_index, edge_weight))


class _TemporalTransformer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 64,
        nhead: int = 4,
        num_layers: int = 1,
        max_len: int = 200,
        node_chunk: int = 1024,
    ):
        super().__init__()
        self.proj = nn.Linear(in_channels, hidden_dim)
        self.hidden_dim = hidden_dim
        self.node_chunk = node_chunk

        pos = torch.zeros(max_len, hidden_dim)
        position = torch.arange(max_len).float().unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / hidden_dim)
        )
        pos[:, 0::2] = torch.sin(position * div_term)
        pos[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pos_enc", pos)

        enc = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=4 * hidden_dim,
            batch_first=True,
            dropout=0.1,
        )
        self.transformer = nn.TransformerEncoder(enc, num_layers=num_layers)

    def forward(self, x):
        bsz, timesteps, num_nodes, num_features = x.size()
        x_flat = x.permute(0, 2, 1, 3).reshape(bsz * num_nodes, timesteps, num_features)
        x_flat = self.proj(x_flat) + self.pos_enc[:timesteps]

        outs = []
        for start in range(0, bsz * num_nodes, self.node_chunk):
            chunk = x_flat[start : start + self.node_chunk]
            outs.append(self.transformer(chunk)[:, -1, :])
        return torch.cat(outs, dim=0).reshape(bsz, num_nodes, self.hidden_dim)


class _LinearTemporalEncoder(nn.Module):
    """
    Per-node linear map from lag window to hidden representation.
    This keeps temporal modeling simple and removes self-attention.
    """

    def __init__(self, input_window: int, in_channels: int, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(input_window * in_channels, hidden_dim)

    def forward(self, x):
        bsz, timesteps, num_nodes, num_features = x.size()
        x_flat = x.permute(0, 2, 1, 3).reshape(bsz, num_nodes, timesteps * num_features)
        return self.proj(x_flat)


class _FusionAttention(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)

    def forward(self, temporal, spatial):
        attn = torch.softmax((self.q(temporal) * self.k(spatial)).sum(-1, keepdim=True), dim=1)
        return temporal + attn * self.v(spatial)


class _ConcatFusion(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
        )

    def forward(self, temporal, spatial):
        return self.proj(torch.cat([temporal, spatial], dim=-1))


class LocalTransformerAblation(nn.Module):
    """
    Spain LTGCN ablation model.

    Supported variants:
    - temporal-only: transformer temporal encoder only
    - spatial-only: GCN over last observed case count, no temporal encoder
    - no-fusion: transformer temporal encoder + GCN spatial encoder, concat fusion
    - linear-temporal-gcn: linear temporal encoder + GCN spatial encoder, attention fusion
    """

    def __init__(
        self,
        *,
        variant: str,
        input_window: int,
        in_channels: int = 1,
        graph_feat_dim: int = 1,
        hidden_dim: int = 64,
        out_channels: int = 1,
        num_nodes: int,
        nhead: int = 4,
        num_layers: int = 1,
    ):
        super().__init__()
        self.variant = variant
        self.use_temporal = variant in {"temporal_only", "no_fusion", "linear_temporal_gcn"}
        self.use_spatial = variant in {"spatial_only", "no_fusion", "linear_temporal_gcn"}
        self.spatial_from_signal = variant == "spatial_only"

        if self.use_temporal:
            if variant == "linear_temporal_gcn":
                self.temporal = _LinearTemporalEncoder(input_window, in_channels, hidden_dim)
            else:
                self.temporal = _TemporalTransformer(
                    in_channels=in_channels,
                    hidden_dim=hidden_dim,
                    nhead=nhead,
                    num_layers=num_layers,
                )

        if self.use_spatial:
            spatial_in = in_channels if self.spatial_from_signal else graph_feat_dim
            self.spatial = _GCNEncoder(spatial_in, hidden_dim)
            if not self.spatial_from_signal:
                self.node_emb = nn.Embedding(num_nodes, graph_feat_dim)

        if self.use_temporal and self.use_spatial:
            self.fusion = _ConcatFusion(hidden_dim) if variant == "no_fusion" else _FusionAttention(hidden_dim)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, out_channels),
        )

    def _spatial_from_last_value(self, x_seq, edge_index, edge_weight):
        last_values = x_seq[:, -1, :, :]  # [B, N, F]
        outputs = []
        for batch_idx in range(last_values.size(0)):
            outputs.append(self.spatial(last_values[batch_idx], edge_index, edge_weight))
        return torch.stack(outputs, dim=0)

    def _spatial_from_node_embeddings(self, x_seq, edge_index, edge_weight, node_indices):
        num_nodes = x_seq.size(2)
        if node_indices is None:
            node_indices = torch.arange(num_nodes, device=x_seq.device)
        node_feats = self.node_emb(node_indices)
        spatial_repr = self.spatial(node_feats, edge_index, edge_weight)
        return spatial_repr.unsqueeze(0).expand(x_seq.size(0), -1, -1)

    def forward(self, x_seq, edge_index, edge_weight=None, node_indices=None, **_):
        temporal_repr = None
        spatial_repr = None

        if self.use_temporal:
            temporal_repr = self.temporal(x_seq)

        if self.use_spatial:
            if self.spatial_from_signal:
                spatial_repr = self._spatial_from_last_value(x_seq, edge_index, edge_weight)
            else:
                spatial_repr = self._spatial_from_node_embeddings(x_seq, edge_index, edge_weight, node_indices)

        if self.use_temporal and self.use_spatial:
            fused = self.fusion(temporal_repr, spatial_repr)
        elif self.use_temporal:
            fused = temporal_repr
        elif self.use_spatial:
            fused = spatial_repr
        else:
            raise ValueError("Ablation model requires at least one active branch.")

        return self.head(fused)

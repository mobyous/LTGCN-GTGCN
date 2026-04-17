import torch
import torch.nn as nn
import torch.nn.functional as F


class _NConv(nn.Module):
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: [B, C, N, T]
        if adj.dim() == 2:
            x = torch.einsum("bcnt,nm->bcmt", x, adj)
        else:
            x = torch.einsum("bcnt,bnm->bcmt", x, adj)
        return x.contiguous()


class _Linear(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.mlp = nn.Conv2d(c_in, c_out, kernel_size=(1, 1), bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class _GCN(nn.Module):
    def __init__(self, c_in: int, c_out: int, dropout: float, support_len: int = 3, order: int = 2):
        super().__init__()
        self.nconv = _NConv()
        self.order = order
        self.dropout = dropout
        total_in = (order * support_len + 1) * c_in
        self.mlp = _Linear(total_in, c_out)

    def forward(self, x: torch.Tensor, supports: list[torch.Tensor]) -> torch.Tensor:
        out = [x]
        for adj in supports:
            x1 = self.nconv(x, adj)
            out.append(x1)
            for _ in range(2, self.order + 1):
                x2 = self.nconv(x1, adj)
                out.append(x2)
                x1 = x2
        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        return F.dropout(h, self.dropout, training=self.training)


class GraphWaveNet(nn.Module):
    """
    Port of the official Graph WaveNet architecture to this repo's API.

    Inputs:
    - x_seq: [B, T, N, F]
    - edge_index / edge_weight: optional fixed graph, converted to dense transition supports

    Output:
    - [B, N, forecast_dim] when output_window == 1
    """

    def __init__(
        self,
        num_nodes: int,
        in_dim: int = 1,
        out_dim: int = 1,
        dropout: float = 0.3,
        gcn_bool: bool = True,
        addaptadj: bool = True,
        residual_channels: int = 32,
        dilation_channels: int = 32,
        skip_channels: int = 256,
        end_channels: int = 512,
        kernel_size: int = 2,
        blocks: int = 4,
        layers: int = 2,
        adapt_dim: int = 10,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj
        self.adapt_dim = adapt_dim

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_channels=in_dim, out_channels=residual_channels, kernel_size=(1, 1))

        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, adapt_dim))
        self.nodevec2 = nn.Parameter(torch.randn(adapt_dim, num_nodes))

        receptive_field = 1
        self.fixed_supports_len = 2  # forward and reverse transitions from provided graph
        supports_len = self.fixed_supports_len + (1 if (gcn_bool and addaptadj) else 0)

        for _block in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for _layer in range(layers):
                self.filter_convs.append(
                    nn.Conv2d(
                        in_channels=residual_channels,
                        out_channels=dilation_channels,
                        kernel_size=(1, kernel_size),
                        dilation=(1, new_dilation),
                    )
                )
                self.gate_convs.append(
                    nn.Conv2d(
                        in_channels=residual_channels,
                        out_channels=dilation_channels,
                        kernel_size=(1, kernel_size),
                        dilation=(1, new_dilation),
                    )
                )
                self.residual_convs.append(
                    nn.Conv2d(in_channels=dilation_channels, out_channels=residual_channels, kernel_size=(1, 1))
                )
                self.skip_convs.append(
                    nn.Conv2d(in_channels=dilation_channels, out_channels=skip_channels, kernel_size=(1, 1))
                )
                self.bn.append(nn.BatchNorm2d(residual_channels))
                if gcn_bool:
                    self.gconv.append(_GCN(dilation_channels, residual_channels, dropout, support_len=supports_len))
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2

        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels, out_channels=end_channels, kernel_size=(1, 1))
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels, out_channels=out_dim, kernel_size=(1, 1))
        self.receptive_field = receptive_field

    def _dense_supports(
        self,
        edge_index: torch.Tensor | None,
        edge_weight: torch.Tensor | None,
        device: torch.device,
    ) -> list[torch.Tensor]:
        if edge_index is None:
            eye = torch.eye(self.num_nodes, device=device)
            return [eye, eye]

        adj = torch.zeros(self.num_nodes, self.num_nodes, device=device)
        if edge_weight is None:
            values = torch.ones(edge_index.size(1), device=device)
        else:
            values = edge_weight.to(device)
        adj[edge_index[0], edge_index[1]] = values

        # Add reverse direction if graph is directed / weighted asymmetrically.
        adj_t = adj.transpose(0, 1)

        def _row_normalize(mat: torch.Tensor) -> torch.Tensor:
            denom = mat.sum(dim=1, keepdim=True).clamp_min(1e-8)
            return mat / denom

        return [_row_normalize(adj), _row_normalize(adj_t)]

    def forward(self, x_seq: torch.Tensor, edge_index=None, edge_weight=None, **_) -> torch.Tensor:
        # Convert [B, T, N, F] -> [B, F, N, T]
        if x_seq.dim() == 3:
            x_seq = x_seq.unsqueeze(0)
        x = x_seq.permute(0, 3, 2, 1)

        in_len = x.size(3)
        if in_len < self.receptive_field:
            x = F.pad(x, (self.receptive_field - in_len, 0, 0, 0))

        x = self.start_conv(x)
        skip = None

        supports = self._dense_supports(edge_index, edge_weight, x.device)
        if self.gcn_bool and self.addaptadj:
            adp = F.softmax(F.relu(self.nodevec1 @ self.nodevec2), dim=1)
            supports = supports + [adp]

        for idx in range(self.blocks * self.layers):
            residual = x
            filt = torch.tanh(self.filter_convs[idx](residual))
            gate = torch.sigmoid(self.gate_convs[idx](residual))
            x = filt * gate

            s = self.skip_convs[idx](x)
            if skip is None:
                skip = s
            else:
                skip = skip[:, :, :, -s.size(3):] + s

            if self.gcn_bool:
                x = self.gconv[idx](x, supports)
            else:
                x = self.residual_convs[idx](x)

            x = x + residual[:, :, :, -x.size(3):]
            x = self.bn[idx](x)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)  # [B, out_dim, N, T']
        x = x[..., -1]  # [B, out_dim, N]
        x = x.permute(0, 2, 1)  # [B, N, out_dim]
        return x

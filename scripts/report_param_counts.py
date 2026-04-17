#!/usr/bin/env python
"""
Report trainable parameter counts for the current controlled model comparison.

This script is intentionally script-first: it instantiates the repo's current
model variants and prints a comparable parameter-count table instead of relying
on ad hoc interactive counting.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.graph import build_brazil_graph, build_spain_graph, extract_brazil_backbone, extract_spain_backbone, load_spain_mobility, top_k_brazil_cities
from src.data.loader import load_brazil_covid
from src.data.preprocess import drop_constant_nodes_brazil, filter_brazil_covid
from src.models.gcrn import GCRN
from src.models.global_transformer import GlobalTransformer
from src.models.graph_wavenet import GraphWaveNet
from src.models.local_transformer import LocalTransformer


@dataclass
class ModelSpec:
    name: str
    model: torch.nn.Module


@dataclass
class ReferenceSetup:
    num_nodes: int
    gcrn_hidden: int
    ltgcn_hidden: int
    gtgcn_hidden: int


def count_trainable_params(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def _spain_num_nodes() -> int:
    centrality_df = pd.read_csv("data/Spain/centrality_provinces.csv")
    cities = sorted(centrality_df["nomemun"].unique())
    mobility_df = load_spain_mobility()
    backbone_df = extract_spain_backbone(mobility_df=mobility_df, cities=cities, alpha=0.01, top_k=5)
    _, node_order = build_spain_graph(backbone_df)
    return len(node_order)


def _brazil_num_nodes(city_top_k: int) -> int:
    covid_df = load_brazil_covid()
    filtered_df = drop_constant_nodes_brazil(filter_brazil_covid(covid_df))
    candidate_ids = set(filtered_df["ibgeID"].unique())
    if city_top_k > 0:
        candidate_ids &= top_k_brazil_cities(city_top_k, candidate_ids)
    backbone_df, _ = extract_brazil_backbone(
        city_whitelist=candidate_ids,
        alpha=0.01,
        top_k=5,
    )
    _, node_order = build_brazil_graph(backbone_df)
    return len(node_order)


def _build_spain_current() -> list[ModelSpec]:
    setup = ReferenceSetup(
        num_nodes=_spain_num_nodes(),
        gcrn_hidden=288,
        ltgcn_hidden=128,
        gtgcn_hidden=128,
    )
    return [
        ModelSpec("GCRN", GCRN(in_channels=1, hidden_channels=setup.gcrn_hidden, out_channels=1)),
        ModelSpec(
            "LTGCN",
            LocalTransformer(
                in_channels=1,
                graph_feat_dim=1,
                trans_hidden=setup.ltgcn_hidden,
                out_channels=1,
                num_nodes=setup.num_nodes,
                nhead=4,
                num_layers=1,
            ),
        ),
        ModelSpec(
            "GraphWaveNet",
            GraphWaveNet(
                num_nodes=setup.num_nodes,
                in_dim=1,
                out_dim=1,
                residual_channels=29,
                dilation_channels=29,
                skip_channels=232,
                end_channels=464,
                blocks=4,
                layers=2,
            ),
        ),
        ModelSpec(
            "GTGCN",
            GlobalTransformer(
                input_dim=1,
                gcn_dim=1,
                hidden_dim=setup.gtgcn_hidden,
                nhead=4,
                num_layers=1,
                num_nodes=setup.num_nodes,
                forecast_dim=1,
                attn_dropout=0.0,
                ff_dropout=0.2,
                gat_heads=2,
            ),
        ),
    ]


def _build_brazil_scale(scale: str) -> list[ModelSpec]:
    scale_to_top_k = {
        "top40": 40,
        "cleaned": 1305,
        "full": 0,
    }
    if scale not in scale_to_top_k:
        raise ValueError(f"Unsupported Brazil scale: {scale}")
    setup = ReferenceSetup(
        num_nodes=_brazil_num_nodes(scale_to_top_k[scale]),
        gcrn_hidden=144,
        ltgcn_hidden=64,
        gtgcn_hidden=64,
    )
    return [
        ModelSpec("GCRN", GCRN(in_channels=1, hidden_channels=setup.gcrn_hidden, out_channels=1)),
        ModelSpec(
            "LTGCN",
            LocalTransformer(
                in_channels=1,
                graph_feat_dim=1,
                trans_hidden=setup.ltgcn_hidden,
                out_channels=1,
                num_nodes=setup.num_nodes,
                nhead=4,
                num_layers=1,
            ),
        ),
        ModelSpec(
            "GraphWaveNet",
            GraphWaveNet(
                num_nodes=setup.num_nodes,
                in_dim=1,
                out_dim=1,
                residual_channels=14 if scale == "top40" else (11 if scale == "cleaned" else 32),
                dilation_channels=14 if scale == "top40" else (11 if scale == "cleaned" else 32),
                skip_channels=112 if scale == "top40" else (88 if scale == "cleaned" else 256),
                end_channels=224 if scale == "top40" else (176 if scale == "cleaned" else 512),
                blocks=4,
                layers=2,
            ),
        ),
        ModelSpec(
            "GTGCN",
            GlobalTransformer(
                input_dim=1,
                gcn_dim=1,
                hidden_dim=setup.gtgcn_hidden,
                nhead=4,
                num_layers=1,
                num_nodes=setup.num_nodes,
                forecast_dim=1,
                attn_dropout=0.0,
                ff_dropout=0.2,
                gat_heads=2,
            ),
        ),
    ]


def _format_report(model_specs: list[ModelSpec]) -> str:
    rows = []
    for spec in model_specs:
        params = count_trainable_params(spec.model)
        rows.append((spec.name, params))

    min_params = min(params for _, params in rows)
    max_params = max(params for _, params in rows)
    lines = []
    lines.append(f"{'Model':<8} {'Params':>12} {'x min':>8} {'vs min %':>10}")
    lines.append("-" * 42)
    for name, params in rows:
        multiple = params / min_params
        pct = (params - min_params) / min_params * 100
        lines.append(f"{name:<8} {params:>12,} {multiple:>8.2f} {pct:>9.1f}%")
    lines.append("-" * 42)
    lines.append(f"Max/min ratio: {max_params / min_params:.2f}x")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Report trainable parameter counts for current model comparisons")
    parser.add_argument(
        "--target",
        choices=["spain-current", "brazil-top40", "brazil-cleaned", "brazil-full"],
        default="spain-current",
    )
    args = parser.parse_args()

    if args.target == "spain-current":
        specs = _build_spain_current()
    elif args.target == "brazil-top40":
        specs = _build_brazil_scale("top40")
    elif args.target == "brazil-cleaned":
        specs = _build_brazil_scale("cleaned")
    else:
        specs = _build_brazil_scale("full")

    print(f"Target: {args.target}")
    print(_format_report(specs))


if __name__ == "__main__":
    main()

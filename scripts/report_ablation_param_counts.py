#!/usr/bin/env python
"""
Report LTGCN ablation parameter counts for the current Spain/Brazil experiment setups.
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

from src.data.graph import (  # noqa: E402
    build_brazil_graph,
    build_spain_graph,
    extract_brazil_backbone,
    extract_spain_backbone,
    load_spain_mobility,
    top_k_brazil_cities,
)
from src.data.loader import load_brazil_covid  # noqa: E402
from src.data.preprocess import drop_constant_nodes_brazil, filter_brazil_covid  # noqa: E402
from src.models.local_transformer import LocalTransformer  # noqa: E402
from src.models.local_transformer_ablation import LocalTransformerAblation  # noqa: E402


@dataclass
class AblationSetup:
    target_name: str
    num_nodes: int
    full_hidden: int
    ablation_hidden: dict[str, int]


def count_trainable_params(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def _spain_num_nodes() -> int:
    centrality_df = pd.read_csv(ROOT / "data" / "Spain" / "centrality_provinces.csv")
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
    backbone_df, _ = extract_brazil_backbone(city_whitelist=candidate_ids, alpha=0.01, top_k=5)
    _, node_order = build_brazil_graph(backbone_df)
    return len(node_order)


def build_setup(target: str) -> AblationSetup:
    if target == "spain":
        return AblationSetup(
            target_name=target,
            num_nodes=_spain_num_nodes(),
            full_hidden=128,
            ablation_hidden={
                "temporal_only": 148,
                "spatial_only": 500,
                "no_fusion": 132,
                "linear_temporal_gcn": 252,
            },
        )
    if target == "brazil-top40":
        return AblationSetup(
            target_name=target,
            num_nodes=_brazil_num_nodes(40),
            full_hidden=64,
            ablation_hidden={
                "temporal_only": 72,
                "spatial_only": 244,
                "no_fusion": 68,
                "linear_temporal_gcn": 124,
            },
        )
    if target == "brazil-cleaned":
        return AblationSetup(
            target_name=target,
            num_nodes=_brazil_num_nodes(1305),
            full_hidden=64,
            ablation_hidden={
                "temporal_only": 76,
                "spatial_only": 248,
                "no_fusion": 68,
                "linear_temporal_gcn": 124,
            },
        )
    if target == "brazil-full":
        return AblationSetup(
            target_name=target,
            num_nodes=_brazil_num_nodes(0),
            full_hidden=64,
            ablation_hidden={
                "temporal_only": 76,
                "spatial_only": 256,
                "no_fusion": 68,
                "linear_temporal_gcn": 124,
            },
        )
    raise ValueError(f"Unsupported target: {target}")


def build_models(setup: AblationSetup) -> list[tuple[str, torch.nn.Module]]:
    models: list[tuple[str, torch.nn.Module]] = []
    models.append(
        (
            "full_ltgcn",
            LocalTransformer(
                in_channels=1,
                graph_feat_dim=1,
                trans_hidden=setup.full_hidden,
                out_channels=1,
                num_nodes=setup.num_nodes,
                nhead=4,
                num_layers=1,
            ),
        )
    )
    for name, variant in (
        ("temporal_only", "temporal_only"),
        ("spatial_only", "spatial_only"),
        ("no_fusion", "no_fusion"),
        ("linear_temporal_gcn", "linear_temporal_gcn"),
    ):
        models.append(
            (
                name,
                LocalTransformerAblation(
                    variant=variant,
                    input_window=14,
                    in_channels=1,
                    graph_feat_dim=1,
                    hidden_dim=setup.ablation_hidden[variant],
                    out_channels=1,
                    num_nodes=setup.num_nodes,
                    nhead=4,
                    num_layers=1,
                ),
            )
        )
    return models


def main():
    parser = argparse.ArgumentParser(description="Report LTGCN ablation parameter counts")
    parser.add_argument("--target", choices=["spain", "brazil-top40", "brazil-cleaned", "brazil-full"], default="spain")
    args = parser.parse_args()

    setup = build_setup(args.target)
    rows = [(name, count_trainable_params(model)) for name, model in build_models(setup)]
    min_params = min(params for _, params in rows)

    print(f"Target: {setup.target_name}")
    print(f"Nodes: {setup.num_nodes}")
    print(f"Full LTGCN hidden: {setup.full_hidden}")
    print(f"Ablation hidden: {setup.ablation_hidden}")
    print()
    print(f"{'Model':<20} {'Params':>12} {'x min':>8} {'vs min %':>10}")
    print("-" * 54)
    for name, params in rows:
        multiple = params / min_params
        pct = (params - min_params) / min_params * 100
        print(f"{name:<20} {params:>12,} {multiple:>8.2f} {pct:>9.1f}%")


if __name__ == "__main__":
    main()

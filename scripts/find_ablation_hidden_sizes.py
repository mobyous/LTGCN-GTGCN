#!/usr/bin/env python
"""
Search hidden sizes for LTGCN ablation variants so they match the full LTGCN parameter budget.
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
class Setup:
    target_name: str
    num_nodes: int
    full_hidden: int


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


def build_setup(target: str) -> Setup:
    if target == "spain":
        return Setup(target_name=target, num_nodes=_spain_num_nodes(), full_hidden=128)
    if target == "brazil-top40":
        return Setup(target_name=target, num_nodes=_brazil_num_nodes(40), full_hidden=64)
    if target == "brazil-cleaned":
        return Setup(target_name=target, num_nodes=_brazil_num_nodes(1305), full_hidden=64)
    if target == "brazil-full":
        return Setup(target_name=target, num_nodes=_brazil_num_nodes(0), full_hidden=64)
    raise ValueError(f"Unsupported target: {target}")


def build_full_model(setup: Setup) -> LocalTransformer:
    return LocalTransformer(
        in_channels=1,
        graph_feat_dim=1,
        trans_hidden=setup.full_hidden,
        out_channels=1,
        num_nodes=setup.num_nodes,
        nhead=4,
        num_layers=1,
    )


def build_ablation_model(setup: Setup, variant: str, hidden_dim: int) -> LocalTransformerAblation:
    return LocalTransformerAblation(
        variant=variant,
        input_window=14,
        in_channels=1,
        graph_feat_dim=1,
        hidden_dim=hidden_dim,
        out_channels=1,
        num_nodes=setup.num_nodes,
        nhead=4,
        num_layers=1,
    )


def search_variant(
    *,
    setup: Setup,
    variant: str,
    target_params: int,
    min_hidden: int,
    max_hidden: int,
    step: int,
    top_k: int,
) -> list[tuple[int, int, int, float]]:
    candidates = []
    for hidden in range(min_hidden, max_hidden + 1, step):
        if variant in {"temporal_only", "no_fusion"} and hidden % 4 != 0:
            continue
        params = count_trainable_params(build_ablation_model(setup, variant, hidden))
        diff = abs(params - target_params)
        pct = diff / target_params * 100
        candidates.append((hidden, params, diff, pct))
    candidates.sort(key=lambda row: (row[2], row[0]))
    return candidates[:top_k]


def main():
    parser = argparse.ArgumentParser(description="Find hidden sizes that match full LTGCN parameter budgets")
    parser.add_argument("--target", choices=["spain", "brazil-top40", "brazil-cleaned", "brazil-full"], default="spain")
    parser.add_argument("--min-hidden", type=int, default=16)
    parser.add_argument("--max-hidden", type=int, default=512)
    parser.add_argument("--step", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    setup = build_setup(args.target)
    full_params = count_trainable_params(build_full_model(setup))

    print(f"Target: {setup.target_name}")
    print(f"Nodes: {setup.num_nodes}")
    print(f"Full LTGCN hidden: {setup.full_hidden}")
    print(f"Full LTGCN params: {full_params:,}")
    print()

    for variant in ("temporal_only", "spatial_only", "no_fusion", "linear_temporal_gcn"):
        print(f"[{variant}]")
        print(f"{'Hidden':>8} {'Params':>12} {'Abs diff':>12} {'Diff %':>10}")
        print("-" * 48)
        for hidden, params, diff, pct in search_variant(
            setup=setup,
            variant=variant,
            target_params=full_params,
            min_hidden=args.min_hidden,
            max_hidden=args.max_hidden,
            step=args.step,
            top_k=args.top_k,
        ):
            print(f"{hidden:>8} {params:>12,} {diff:>12,} {pct:>9.2f}%")
        print()


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Search for a GCRN hidden size that matches the current LTGCN / GTGCN parameter budget.

This is intended for the "match parameter budgets" revision item. It uses the
repo's current model implementations and dataset-specific node counts rather
than hand-derived formulas.
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
from src.models.gcrn import GCRN  # noqa: E402
from src.models.global_transformer import GlobalTransformer  # noqa: E402
from src.models.local_transformer import LocalTransformer  # noqa: E402


def count_trainable_params(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


@dataclass
class ReferenceSetup:
    target_name: str
    num_nodes: int
    ltgcn_hidden: int
    gtgcn_hidden: int


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


def build_reference_setup(target: str) -> ReferenceSetup:
    if target == "spain-current":
        return ReferenceSetup(
            target_name=target,
            num_nodes=_spain_num_nodes(),
            ltgcn_hidden=128,
            gtgcn_hidden=128,
        )
    if target == "brazil-top40":
        return ReferenceSetup(
            target_name=target,
            num_nodes=_brazil_num_nodes(40),
            ltgcn_hidden=64,
            gtgcn_hidden=64,
        )
    if target == "brazil-cleaned":
        return ReferenceSetup(
            target_name=target,
            num_nodes=_brazil_num_nodes(1305),
            ltgcn_hidden=64,
            gtgcn_hidden=64,
        )
    if target == "brazil-full":
        return ReferenceSetup(
            target_name=target,
            num_nodes=_brazil_num_nodes(0),
            ltgcn_hidden=64,
            gtgcn_hidden=64,
        )
    raise ValueError(f"Unsupported target: {target}")


def build_ltgcn(setup: ReferenceSetup) -> LocalTransformer:
    return LocalTransformer(
        in_channels=1,
        graph_feat_dim=1,
        trans_hidden=setup.ltgcn_hidden,
        out_channels=1,
        num_nodes=setup.num_nodes,
        nhead=4,
        num_layers=1,
    )


def build_gtgcn(setup: ReferenceSetup) -> GlobalTransformer:
    return GlobalTransformer(
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
    )


def target_count(ltgcn_params: int, gtgcn_params: int, match_to: str) -> int:
    if match_to == "average":
        return round((ltgcn_params + gtgcn_params) / 2)
    if match_to == "ltgcn":
        return ltgcn_params
    if match_to == "gtgcn":
        return gtgcn_params
    if match_to == "min":
        return min(ltgcn_params, gtgcn_params)
    if match_to == "max":
        return max(ltgcn_params, gtgcn_params)
    raise ValueError(f"Unsupported match_to={match_to}")


def search_gcrn_hidden_sizes(
    *,
    target_params: int,
    min_hidden: int,
    max_hidden: int,
    step: int,
    top_k: int,
) -> list[tuple[int, int, int, float]]:
    candidates = []
    for hidden in range(min_hidden, max_hidden + 1, step):
        params = count_trainable_params(GCRN(in_channels=1, hidden_channels=hidden, out_channels=1))
        diff = abs(params - target_params)
        pct = diff / target_params * 100
        candidates.append((hidden, params, diff, pct))
    candidates.sort(key=lambda row: (row[2], row[0]))
    return candidates[:top_k]


def main():
    parser = argparse.ArgumentParser(description="Find a GCRN hidden size that matches LTGCN/GTGCN parameter budgets")
    parser.add_argument(
        "--target",
        choices=["spain-current", "brazil-top40", "brazil-cleaned", "brazil-full"],
        default="spain-current",
    )
    parser.add_argument(
        "--match-to",
        choices=["average", "ltgcn", "gtgcn", "min", "max"],
        default="average",
        help="Which transformer budget to match against",
    )
    parser.add_argument("--min-hidden", type=int, default=32)
    parser.add_argument("--max-hidden", type=int, default=1024)
    parser.add_argument("--step", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    setup = build_reference_setup(args.target)
    ltgcn_params = count_trainable_params(build_ltgcn(setup))
    gtgcn_params = count_trainable_params(build_gtgcn(setup))
    target_params = target_count(ltgcn_params, gtgcn_params, args.match_to)

    results = search_gcrn_hidden_sizes(
        target_params=target_params,
        min_hidden=args.min_hidden,
        max_hidden=args.max_hidden,
        step=args.step,
        top_k=args.top_k,
    )

    print(f"Target: {args.target}")
    print(f"Nodes: {setup.num_nodes}")
    print(f"LTGCN params: {ltgcn_params:,}")
    print(f"GTGCN params: {gtgcn_params:,}")
    print(f"Matching mode: {args.match_to}")
    print(f"Target params: {target_params:,}")
    print()
    print(f"{'Hidden':>8} {'GCRN params':>12} {'Abs diff':>12} {'Diff %':>10}")
    print("-" * 48)
    for hidden, params, diff, pct in results:
        print(f"{hidden:>8} {params:>12,} {diff:>12,} {pct:>9.2f}%")


if __name__ == "__main__":
    main()

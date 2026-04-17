#!/usr/bin/env python
"""
Export current parameter counts for main models and ablations to a JSON file.

This is the single source-of-truth artifact for parameter budgets so we don't
need to rerun multiple reporting scripts just to check counts.
"""
from __future__ import annotations

import argparse
import json
import sys
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
from src.models.graph_wavenet import GraphWaveNet  # noqa: E402
from src.models.local_transformer import LocalTransformer  # noqa: E402
from src.models.local_transformer_ablation import LocalTransformerAblation  # noqa: E402


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


def summarize_budget(models: dict) -> dict:
    trainable = {name: spec["params"] for name, spec in models.items() if spec["params"] > 0}
    min_params = min(trainable.values())
    max_params = max(trainable.values())
    return {
        "min_trainable_params": min_params,
        "max_trainable_params": max_params,
        "max_min_ratio": round(max_params / min_params, 4),
    }


def export_main_models() -> dict:
    spain_nodes = _spain_num_nodes()
    brazil_top40_nodes = _brazil_num_nodes(40)
    brazil_cleaned_nodes = _brazil_num_nodes(1305)
    brazil_full_nodes = _brazil_num_nodes(0)

    datasets = {
        "spain": {
            "num_nodes": spain_nodes,
            "models": {
                "gcrn": {
                    "hidden_channels": 288,
                    "params": count_trainable_params(GCRN(in_channels=1, hidden_channels=288, out_channels=1)),
                },
                "ltgcn": {
                    "trans_hidden": 128,
                    "params": count_trainable_params(
                        LocalTransformer(
                            in_channels=1,
                            graph_feat_dim=1,
                            trans_hidden=128,
                            out_channels=1,
                            num_nodes=spain_nodes,
                            nhead=4,
                            num_layers=1,
                        )
                    ),
                },
                "gtgcn": {
                    "hidden_dim": 128,
                    "params": count_trainable_params(
                        GlobalTransformer(
                            input_dim=1,
                            gcn_dim=1,
                            hidden_dim=128,
                            nhead=4,
                            num_layers=1,
                            num_nodes=spain_nodes,
                            forecast_dim=1,
                            attn_dropout=0.0,
                            ff_dropout=0.2,
                            gat_heads=2,
                        )
                    ),
                },
                "graph_wavenet": {
                    "residual_channels": 29,
                    "dilation_channels": 29,
                    "skip_channels": 232,
                    "end_channels": 464,
                    "blocks": 4,
                    "num_layers": 2,
                    "params": count_trainable_params(
                        GraphWaveNet(
                            num_nodes=spain_nodes,
                            in_dim=1,
                            out_dim=1,
                            residual_channels=29,
                            dilation_channels=29,
                            skip_channels=232,
                            end_channels=464,
                            blocks=4,
                            layers=2,
                        )
                    ),
                },
                "persistence": {
                    "params": 0,
                },
            },
        },
        "brazil_top40": {
            "num_nodes": brazil_top40_nodes,
            "models": {
                "gcrn": {
                    "hidden_channels": 144,
                    "params": count_trainable_params(GCRN(in_channels=1, hidden_channels=144, out_channels=1)),
                },
                "ltgcn": {
                    "trans_hidden": 64,
                    "params": count_trainable_params(
                        LocalTransformer(
                            in_channels=1,
                            graph_feat_dim=1,
                            trans_hidden=64,
                            out_channels=1,
                            num_nodes=brazil_top40_nodes,
                            nhead=4,
                            num_layers=1,
                        )
                    ),
                },
                "gtgcn": {
                    "hidden_dim": 64,
                    "params": count_trainable_params(
                        GlobalTransformer(
                            input_dim=1,
                            gcn_dim=1,
                            hidden_dim=64,
                            nhead=4,
                            num_layers=1,
                            num_nodes=brazil_top40_nodes,
                            forecast_dim=1,
                            attn_dropout=0.0,
                            ff_dropout=0.2,
                            gat_heads=2,
                        )
                    ),
                },
                "graph_wavenet": {
                    "residual_channels": 14,
                    "dilation_channels": 14,
                    "skip_channels": 112,
                    "end_channels": 224,
                    "blocks": 4,
                    "num_layers": 2,
                    "params": count_trainable_params(
                        GraphWaveNet(
                            num_nodes=brazil_top40_nodes,
                            in_dim=1,
                            out_dim=1,
                            residual_channels=14,
                            dilation_channels=14,
                            skip_channels=112,
                            end_channels=224,
                            blocks=4,
                            layers=2,
                        )
                    ),
                },
                "persistence": {
                    "params": 0,
                },
            },
        },
        "brazil_cleaned": {
            "num_nodes": brazil_cleaned_nodes,
            "models": {
                "gcrn": {
                    "hidden_channels": 144,
                    "params": count_trainable_params(GCRN(in_channels=1, hidden_channels=144, out_channels=1)),
                },
                "ltgcn": {
                    "trans_hidden": 64,
                    "params": count_trainable_params(
                        LocalTransformer(
                            in_channels=1,
                            graph_feat_dim=1,
                            trans_hidden=64,
                            out_channels=1,
                            num_nodes=brazil_cleaned_nodes,
                            nhead=4,
                            num_layers=1,
                        )
                    ),
                },
                "gtgcn": {
                    "hidden_dim": 64,
                    "params": count_trainable_params(
                        GlobalTransformer(
                            input_dim=1,
                            gcn_dim=1,
                            hidden_dim=64,
                            nhead=4,
                            num_layers=1,
                            num_nodes=brazil_cleaned_nodes,
                            forecast_dim=1,
                            attn_dropout=0.0,
                            ff_dropout=0.2,
                            gat_heads=2,
                        )
                    ),
                },
                "graph_wavenet": {
                    "residual_channels": 11,
                    "dilation_channels": 11,
                    "skip_channels": 88,
                    "end_channels": 176,
                    "blocks": 4,
                    "num_layers": 2,
                    "params": count_trainable_params(
                        GraphWaveNet(
                            num_nodes=brazil_cleaned_nodes,
                            in_dim=1,
                            out_dim=1,
                            residual_channels=11,
                            dilation_channels=11,
                            skip_channels=88,
                            end_channels=176,
                            blocks=4,
                            layers=2,
                        )
                    ),
                },
                "persistence": {
                    "params": 0,
                },
            },
        },
        "brazil_full": {
            "num_nodes": brazil_full_nodes,
            "models": {
                "gcrn": {
                    "hidden_channels": 144,
                    "params": count_trainable_params(GCRN(in_channels=1, hidden_channels=144, out_channels=1)),
                },
                "ltgcn": {
                    "trans_hidden": 64,
                    "params": count_trainable_params(
                        LocalTransformer(
                            in_channels=1,
                            graph_feat_dim=1,
                            trans_hidden=64,
                            out_channels=1,
                            num_nodes=brazil_full_nodes,
                            nhead=4,
                            num_layers=1,
                        )
                    ),
                },
                "gtgcn": {
                    "hidden_dim": 64,
                    "params": count_trainable_params(
                        GlobalTransformer(
                            input_dim=1,
                            gcn_dim=1,
                            hidden_dim=64,
                            nhead=4,
                            num_layers=1,
                            num_nodes=brazil_full_nodes,
                            forecast_dim=1,
                            attn_dropout=0.0,
                            ff_dropout=0.2,
                            gat_heads=2,
                        )
                    ),
                },
                "graph_wavenet": {
                    "residual_channels": 32,
                    "dilation_channels": 32,
                    "skip_channels": 256,
                    "end_channels": 512,
                    "blocks": 4,
                    "num_layers": 2,
                    "params": count_trainable_params(
                        GraphWaveNet(
                            num_nodes=brazil_full_nodes,
                            in_dim=1,
                            out_dim=1,
                            residual_channels=32,
                            dilation_channels=32,
                            skip_channels=256,
                            end_channels=512,
                            blocks=4,
                            layers=2,
                        )
                    ),
                },
                "persistence": {
                    "params": 0,
                },
            },
        },
    }

    for payload in datasets.values():
        payload["budget_summary"] = summarize_budget(payload["models"])
    return datasets


def export_ablations() -> dict:
    spain_nodes = _spain_num_nodes()
    brazil_top40_nodes = _brazil_num_nodes(40)
    brazil_cleaned_nodes = _brazil_num_nodes(1305)
    brazil_full_nodes = _brazil_num_nodes(0)

    def full_ltgcn(num_nodes: int, hidden: int) -> int:
        return count_trainable_params(
            LocalTransformer(
                in_channels=1,
                graph_feat_dim=1,
                trans_hidden=hidden,
                out_channels=1,
                num_nodes=num_nodes,
                nhead=4,
                num_layers=1,
            )
        )

    def ablation(num_nodes: int, variant: str, hidden: int) -> int:
        return count_trainable_params(
            LocalTransformerAblation(
                variant=variant,
                input_window=14,
                in_channels=1,
                graph_feat_dim=1,
                hidden_dim=hidden,
                out_channels=1,
                num_nodes=num_nodes,
                nhead=4,
                num_layers=1,
            )
        )

    datasets = {
        "spain": {
            "num_nodes": spain_nodes,
            "models": {
                "full_ltgcn": {"trans_hidden": 128, "params": full_ltgcn(spain_nodes, 128)},
                "temporal_only": {"hidden_dim": 148, "params": ablation(spain_nodes, "temporal_only", 148)},
                "spatial_only": {"hidden_dim": 500, "params": ablation(spain_nodes, "spatial_only", 500)},
                "no_fusion": {"hidden_dim": 132, "params": ablation(spain_nodes, "no_fusion", 132)},
                "linear_temporal_gcn": {"hidden_dim": 252, "params": ablation(spain_nodes, "linear_temporal_gcn", 252)},
                "persistence": {"params": 0},
            },
        },
        "brazil_top40": {
            "num_nodes": brazil_top40_nodes,
            "models": {
                "full_ltgcn": {"trans_hidden": 64, "params": full_ltgcn(brazil_top40_nodes, 64)},
                "temporal_only": {"hidden_dim": 72, "params": ablation(brazil_top40_nodes, "temporal_only", 72)},
                "spatial_only": {"hidden_dim": 244, "params": ablation(brazil_top40_nodes, "spatial_only", 244)},
                "no_fusion": {"hidden_dim": 68, "params": ablation(brazil_top40_nodes, "no_fusion", 68)},
                "linear_temporal_gcn": {"hidden_dim": 124, "params": ablation(brazil_top40_nodes, "linear_temporal_gcn", 124)},
                "persistence": {"params": 0},
            },
        },
        "brazil_cleaned": {
            "num_nodes": brazil_cleaned_nodes,
            "models": {
                "full_ltgcn": {"trans_hidden": 64, "params": full_ltgcn(brazil_cleaned_nodes, 64)},
                "temporal_only": {"hidden_dim": 76, "params": ablation(brazil_cleaned_nodes, "temporal_only", 76)},
                "spatial_only": {"hidden_dim": 248, "params": ablation(brazil_cleaned_nodes, "spatial_only", 248)},
                "no_fusion": {"hidden_dim": 68, "params": ablation(brazil_cleaned_nodes, "no_fusion", 68)},
                "linear_temporal_gcn": {"hidden_dim": 124, "params": ablation(brazil_cleaned_nodes, "linear_temporal_gcn", 124)},
                "persistence": {"params": 0},
            },
        },
        "brazil_full": {
            "num_nodes": brazil_full_nodes,
            "models": {
                "full_ltgcn": {"trans_hidden": 64, "params": full_ltgcn(brazil_full_nodes, 64)},
                "temporal_only": {"hidden_dim": 76, "params": ablation(brazil_full_nodes, "temporal_only", 76)},
                "spatial_only": {"hidden_dim": 256, "params": ablation(brazil_full_nodes, "spatial_only", 256)},
                "no_fusion": {"hidden_dim": 68, "params": ablation(brazil_full_nodes, "no_fusion", 68)},
                "linear_temporal_gcn": {"hidden_dim": 124, "params": ablation(brazil_full_nodes, "linear_temporal_gcn", 124)},
                "persistence": {"params": 0},
            },
        },
    }

    for payload in datasets.values():
        payload["budget_summary"] = summarize_budget(payload["models"])
    return datasets


def main():
    parser = argparse.ArgumentParser(description="Export current parameter counts to JSON")
    parser.add_argument(
        "--out",
        default=str(ROOT / "parameter_counts.json"),
        help="Output JSON path",
    )
    args = parser.parse_args()

    payload = {
        "main_models": export_main_models(),
        "ablations": export_ablations(),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[done] parameter counts saved -> {out_path}")


if __name__ == "__main__":
    main()

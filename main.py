"""
Main experiment runner.

Usage:
    python main.py --experiment brazil_gcrn_top40
    python main.py --experiment brazil_persistence_top40
    python main.py --experiment brazil_graph_wavenet_top40
    python main.py --experiment brazil_ltgcn_temporal_only_top40
    python main.py --experiment spain_global_transformer
    python main.py --experiment spain_graph_wavenet
    python main.py --experiment spain_persistence
    python main.py --experiment spain_ltgcn_temporal_only
    python main.py --all-spain
    python main.py --all-brazil
    python main.py --experiment all
    python main.py --list

Brazil city-scale variants (suffix applies to all Brazil experiment families):
    _full      all backbone cities  (~5,300+)  — slow, full coverage
    _cleaned   top-1305 by pop      (~1,305)   — matches cleaned-notebook scale
    _top40     top-40 by pop        (~38)      — fast, matches old notebooks

Examples:
    python main.py --experiment brazil_gcrn_full
    python main.py --experiment brazil_persistence_top40
    python main.py --experiment brazil_local_transformer_cleaned
    python main.py --experiment brazil_global_transformer_top40
    python main.py --experiment brazil_graph_wavenet_cleaned
    python main.py --experiment brazil_linear_temporal_gcn_cleaned
    python main.py --experiment spain_persistence
    python main.py --experiment spain_ltgcn_no_fusion
    python main.py --all-spain
    python main.py --all-brazil
"""
import argparse
import csv
import json
import copy
from pathlib import Path

from src.config import DataConfig, ModelConfig, TrainConfig, ExperimentConfig

# ── Brazil city-scale configs ──────────────────────────────────────────────────
# Each variant is (city_top_k, display_label).
# (city_top_k, log_transform, display_label)
_BRAZIL_SCALES = {
    "full":    (0,    True,  "~5,300+ cities  [all backbone cities, log-transform on]"),
    "cleaned": (1305, True,  "~1,305 cities   [top-1305 by population, log-transform on]"),
    "top40":   (40,   False, "~38 cities      [top-40 by population, matches old notebooks]"),
}

_BRAZIL_ABLATION_HIDDEN = {
    "temporal_only": {0: 76, 1305: 76, 40: 72},
    "spatial_only": {0: 256, 1305: 248, 40: 244},
    "no_fusion": {0: 68, 1305: 68, 40: 68},
    "linear_temporal_gcn": {0: 124, 1305: 124, 40: 124},
}

_BRAZIL_GRAPH_WAVENET_CONFIG = {
    40: {"residual_channels": 14, "dilation_channels": 14, "skip_channels": 112, "end_channels": 224, "blocks": 4, "num_layers": 2},
    1305: {"residual_channels": 11, "dilation_channels": 11, "skip_channels": 88, "end_channels": 176, "blocks": 4, "num_layers": 2},
    0: {"residual_channels": 32, "dilation_channels": 32, "skip_channels": 256, "end_channels": 512, "blocks": 4, "num_layers": 2},
}

_SPAIN_ABLATION_HIDDEN = {
    "temporal_only": 148,
    "spatial_only": 500,
    "no_fusion": 132,
    "linear_temporal_gcn": 252,
}


def _brazil_gcrn_cfg(name: str, city_top_k: int, log_transform: bool) -> ExperimentConfig:
    return ExperimentConfig(
        name=name,
        data=DataConfig(country="brazil", input_window=14, output_window=1,
                        brazil_city_top_k=city_top_k, log_transform=log_transform),
        model=ModelConfig(model_type="gcrn", hidden_channels=144),
        train=TrainConfig(epochs=50, batch_size=32, lr=1e-3,
                          scheduler_step=10, scheduler_gamma=0.7),
    )


def _brazil_local_transformer_cfg(name: str, city_top_k: int, log_transform: bool) -> ExperimentConfig:
    return ExperimentConfig(
        name=name,
        data=DataConfig(country="brazil", input_window=14, output_window=1,
                        brazil_city_top_k=city_top_k, log_transform=log_transform),
        model=ModelConfig(model_type="local_transformer", trans_hidden=64,
                          nhead=4, graph_feat_dim=1),
        train=TrainConfig(epochs=50, batch_size=32, lr=1e-3,
                          scheduler_step=10, scheduler_gamma=0.7),
    )


def _brazil_graph_wavenet_cfg(name: str, city_top_k: int, log_transform: bool) -> ExperimentConfig:
    gwn = _BRAZIL_GRAPH_WAVENET_CONFIG[city_top_k]
    return ExperimentConfig(
        name=name,
        data=DataConfig(country="brazil", input_window=14, output_window=1,
                        brazil_city_top_k=city_top_k, log_transform=log_transform),
        model=ModelConfig(
            model_type="graph_wavenet",
            residual_channels=gwn["residual_channels"],
            dilation_channels=gwn["dilation_channels"],
            skip_channels=gwn["skip_channels"],
            end_channels=gwn["end_channels"],
            blocks=gwn["blocks"],
            num_layers=gwn["num_layers"],
        ),
        train=TrainConfig(epochs=50, batch_size=32, lr=1e-3,
                          scheduler_step=10, scheduler_gamma=0.7),
    )


def _brazil_global_transformer_cfg(name: str, city_top_k: int, log_transform: bool) -> ExperimentConfig:
    return ExperimentConfig(
        name=name,
        data=DataConfig(country="brazil", input_window=14, output_window=1,
                        brazil_city_top_k=city_top_k, log_transform=log_transform),
        model=ModelConfig(model_type="global_transformer", hidden_dim=64,
                          nhead=4, num_layers=1, ff_dropout=0.2, gat_heads=2),
        train=TrainConfig(epochs=50, batch_size=32, lr=1e-3,
                          scheduler_step=10, scheduler_gamma=0.95),
    )


def _brazil_persistence_cfg(name: str, city_top_k: int, log_transform: bool) -> ExperimentConfig:
    return ExperimentConfig(
        name=name,
        data=DataConfig(country="brazil", input_window=14, output_window=1,
                        brazil_city_top_k=city_top_k, log_transform=log_transform),
        model=ModelConfig(model_type="persistence"),
        train=TrainConfig(epochs=0),
    )


def _brazil_ltgcn_temporal_only_cfg(name: str, city_top_k: int, log_transform: bool) -> ExperimentConfig:
    return ExperimentConfig(
        name=name,
        data=DataConfig(country="brazil", input_window=14, output_window=1,
                        brazil_city_top_k=city_top_k, log_transform=log_transform),
        model=ModelConfig(
            model_type="temporal_only",
            trans_hidden=_BRAZIL_ABLATION_HIDDEN["temporal_only"][city_top_k],
            nhead=4,
            graph_feat_dim=1,
        ),
        train=TrainConfig(epochs=50, batch_size=32, lr=1e-3,
                          scheduler_step=10, scheduler_gamma=0.7),
    )


def _brazil_ltgcn_spatial_only_cfg(name: str, city_top_k: int, log_transform: bool) -> ExperimentConfig:
    return ExperimentConfig(
        name=name,
        data=DataConfig(country="brazil", input_window=14, output_window=1,
                        brazil_city_top_k=city_top_k, log_transform=log_transform),
        model=ModelConfig(
            model_type="spatial_only",
            trans_hidden=_BRAZIL_ABLATION_HIDDEN["spatial_only"][city_top_k],
            nhead=4,
            graph_feat_dim=1,
        ),
        train=TrainConfig(epochs=50, batch_size=32, lr=1e-3,
                          scheduler_step=10, scheduler_gamma=0.7),
    )


def _brazil_ltgcn_no_fusion_cfg(name: str, city_top_k: int, log_transform: bool) -> ExperimentConfig:
    return ExperimentConfig(
        name=name,
        data=DataConfig(country="brazil", input_window=14, output_window=1,
                        brazil_city_top_k=city_top_k, log_transform=log_transform),
        model=ModelConfig(
            model_type="no_fusion",
            trans_hidden=_BRAZIL_ABLATION_HIDDEN["no_fusion"][city_top_k],
            nhead=4,
            graph_feat_dim=1,
        ),
        train=TrainConfig(epochs=50, batch_size=32, lr=1e-3,
                          scheduler_step=10, scheduler_gamma=0.7),
    )


def _brazil_linear_temporal_gcn_cfg(name: str, city_top_k: int, log_transform: bool) -> ExperimentConfig:
    return ExperimentConfig(
        name=name,
        data=DataConfig(country="brazil", input_window=14, output_window=1,
                        brazil_city_top_k=city_top_k, log_transform=log_transform),
        model=ModelConfig(
            model_type="linear_temporal_gcn",
            trans_hidden=_BRAZIL_ABLATION_HIDDEN["linear_temporal_gcn"][city_top_k],
            nhead=4,
            graph_feat_dim=1,
        ),
        train=TrainConfig(epochs=50, batch_size=32, lr=1e-3,
                          scheduler_step=10, scheduler_gamma=0.7),
    )


_BRAZIL_CFG_FACTORIES = {
    "brazil_gcrn":               _brazil_gcrn_cfg,
    "brazil_local_transformer":  _brazil_local_transformer_cfg,
    "brazil_graph_wavenet":      _brazil_graph_wavenet_cfg,
    "brazil_global_transformer": _brazil_global_transformer_cfg,
    "brazil_persistence":        _brazil_persistence_cfg,
    "brazil_ltgcn_temporal_only": _brazil_ltgcn_temporal_only_cfg,
    "brazil_ltgcn_spatial_only":  _brazil_ltgcn_spatial_only_cfg,
    "brazil_ltgcn_no_fusion":     _brazil_ltgcn_no_fusion_cfg,
    "brazil_linear_temporal_gcn": _brazil_linear_temporal_gcn_cfg,
}

# ── Experiment registry ────────────────────────────────────────────────────────
# Values: (module_path, cfg_or_None)  — None means use the module's own default.

EXPERIMENTS: dict = {}

# Brazil: one entry per model × scale
for _model, _factory in _BRAZIL_CFG_FACTORIES.items():
    for _scale, (_top_k, _log, _label) in _BRAZIL_SCALES.items():
        _exp_name = f"{_model}_{_scale}"
        if _model == "brazil_persistence":
            module_path = "src.experiments.persistence_baseline"
        elif _model in {
            "brazil_ltgcn_temporal_only",
            "brazil_ltgcn_spatial_only",
            "brazil_ltgcn_no_fusion",
            "brazil_linear_temporal_gcn",
        }:
            module_path = "src.experiments.brazil_local_transformer_ablations"
        else:
            module_path = f"src.experiments.{_model}"
        EXPERIMENTS[_exp_name] = (module_path, _factory(_exp_name, _top_k, _log))

# Spain: single scale, module uses its own default config
EXPERIMENTS["spain_gcrn"] = (
    "src.experiments.spain_gcrn",
    ExperimentConfig(
        name="spain_gcrn",
        data=DataConfig(country="spain", input_window=14, output_window=1),
        model=ModelConfig(model_type="gcrn", hidden_channels=288),
        train=TrainConfig(epochs=50, batch_size=32, lr=1e-3, scheduler_step=5, scheduler_gamma=0.5),
    ),
)
EXPERIMENTS["spain_local_transformer"] = (
    "src.experiments.spain_local_transformer",
    ExperimentConfig(
        name="spain_local_transformer",
        data=DataConfig(country="spain", input_window=14, output_window=1),
        model=ModelConfig(model_type="local_transformer", trans_hidden=128, nhead=4, graph_feat_dim=1),
        train=TrainConfig(epochs=50, batch_size=32, lr=3e-4, scheduler_step=5, scheduler_gamma=0.5),
    ),
)
EXPERIMENTS["spain_graph_wavenet"] = (
    "src.experiments.spain_graph_wavenet",
    ExperimentConfig(
        name="spain_graph_wavenet",
        data=DataConfig(country="spain", input_window=14, output_window=1),
        model=ModelConfig(
            model_type="graph_wavenet",
            residual_channels=29,
            dilation_channels=29,
            skip_channels=232,
            end_channels=464,
            blocks=4,
            num_layers=2,
        ),
        train=TrainConfig(epochs=50, batch_size=32, lr=1e-3, scheduler_step=5, scheduler_gamma=0.5),
    ),
)
EXPERIMENTS["spain_global_transformer"] = (
    "src.experiments.spain_global_transformer",
    ExperimentConfig(
        name="spain_global_transformer",
        data=DataConfig(country="spain", input_window=14, output_window=1),
        model=ModelConfig(model_type="global_transformer", hidden_dim=128, nhead=4, num_layers=1, ff_dropout=0.2, gat_heads=2),
        train=TrainConfig(epochs=50, batch_size=16, lr=3e-4, scheduler_step=5, scheduler_gamma=0.5),
    ),
)
EXPERIMENTS["spain_persistence"] = (
    "src.experiments.persistence_baseline",
    ExperimentConfig(
        name="spain_persistence",
        data=DataConfig(country="spain", input_window=14, output_window=1),
        model=ModelConfig(model_type="persistence"),
        train=TrainConfig(epochs=0),
    ),
)
for _name, _variant in (
    ("spain_ltgcn_temporal_only", "temporal_only"),
    ("spain_ltgcn_spatial_only", "spatial_only"),
    ("spain_ltgcn_no_fusion", "no_fusion"),
    ("spain_linear_temporal_gcn", "linear_temporal_gcn"),
):
    EXPERIMENTS[_name] = (
        "src.experiments.spain_local_transformer_ablations",
        ExperimentConfig(
            name=_name,
            data=DataConfig(country="spain", input_window=14, output_window=1),
            model=ModelConfig(
                model_type=_variant,
                trans_hidden=_SPAIN_ABLATION_HIDDEN[_variant],
                nhead=4,
                graph_feat_dim=1,
            ),
            train=TrainConfig(
                epochs=50,
                batch_size=32,
                lr=3e-4,
                weight_decay=1e-4,
                scheduler_step=5,
                scheduler_gamma=0.5,
            ),
        ),
    )


# ── Runner ────────────────────────────────────────────────────────────────────

def ordered_all_experiments() -> list[str]:
    """
    Explicit execution order for `--experiment all`.

    Requested order:
    1. Spain main experiments
    2. Spain LTGCN ablations
    3. Brazil top40 main experiments
    4. Brazil top40 LTGCN ablations
    5. Brazil cleaned main experiments
    6. Brazil cleaned LTGCN ablations
    7. Brazil full: GCRN + LTGCN only
    8. Brazil full LTGCN ablations only

    Brazil full intentionally excludes GTGCN, and also excludes the persistence
    baseline to keep the full-scale sweep limited to GCRN/LTGCN families.
    """
    ordered = [
        # Spain main
        "spain_gcrn",
        "spain_local_transformer",
        "spain_graph_wavenet",
        "spain_global_transformer",
        "spain_persistence",

        # Spain ablations
        "spain_ltgcn_temporal_only",
        "spain_ltgcn_spatial_only",
        "spain_ltgcn_no_fusion",
        "spain_linear_temporal_gcn",

        # Brazil top40 main
        "brazil_gcrn_top40",
        "brazil_local_transformer_top40",
        "brazil_graph_wavenet_top40",
        "brazil_global_transformer_top40",
        "brazil_persistence_top40",

        # Brazil top40 ablations
        "brazil_ltgcn_temporal_only_top40",
        "brazil_ltgcn_spatial_only_top40",
        "brazil_ltgcn_no_fusion_top40",
        "brazil_linear_temporal_gcn_top40",

        # Brazil cleaned main (global_transformer excluded — OOM at 1305 nodes)
        "brazil_gcrn_cleaned",
        "brazil_local_transformer_cleaned",
        "brazil_graph_wavenet_cleaned",
        "brazil_persistence_cleaned",

        # Brazil cleaned ablations
        "brazil_ltgcn_temporal_only_cleaned",
        "brazil_ltgcn_spatial_only_cleaned",
        "brazil_ltgcn_no_fusion_cleaned",
        "brazil_linear_temporal_gcn_cleaned",

        # Brazil full main (restricted)
        "brazil_gcrn_full",
        "brazil_local_transformer_full",

        # Brazil full ablations
        "brazil_ltgcn_temporal_only_full",
        "brazil_ltgcn_spatial_only_full",
        "brazil_ltgcn_no_fusion_full",
        "brazil_linear_temporal_gcn_full",
    ]
    return [name for name in ordered if name in EXPERIMENTS]


def ordered_spain_experiments() -> list[str]:
    ordered = [
        "spain_gcrn",
        "spain_local_transformer",
        "spain_graph_wavenet",
        "spain_global_transformer",
        "spain_persistence",
        "spain_ltgcn_temporal_only",
        "spain_ltgcn_spatial_only",
        "spain_ltgcn_no_fusion",
        "spain_linear_temporal_gcn",
    ]
    return [name for name in ordered if name in EXPERIMENTS]


def ordered_brazil_experiments() -> list[str]:
    ordered = [
        # Brazil top40 main
        "brazil_gcrn_top40",
        "brazil_local_transformer_top40",
        "brazil_graph_wavenet_top40",
        "brazil_global_transformer_top40",
        "brazil_persistence_top40",

        # Brazil top40 ablations
        "brazil_ltgcn_temporal_only_top40",
        "brazil_ltgcn_spatial_only_top40",
        "brazil_ltgcn_no_fusion_top40",
        "brazil_linear_temporal_gcn_top40",

        # Brazil cleaned main (global_transformer excluded — OOM at 1305 nodes)
        "brazil_gcrn_cleaned",
        "brazil_local_transformer_cleaned",
        "brazil_graph_wavenet_cleaned",
        "brazil_persistence_cleaned",

        # Brazil cleaned ablations
        "brazil_ltgcn_temporal_only_cleaned",
        "brazil_ltgcn_spatial_only_cleaned",
        "brazil_ltgcn_no_fusion_cleaned",
        "brazil_linear_temporal_gcn_cleaned",

        # Brazil full main (restricted)
        "brazil_gcrn_full",
        "brazil_local_transformer_full",

        # Brazil full ablations
        "brazil_ltgcn_temporal_only_full",
        "brazil_ltgcn_spatial_only_full",
        "brazil_ltgcn_no_fusion_full",
        "brazil_linear_temporal_gcn_full",
    ]
    return [name for name in ordered if name in EXPERIMENTS]

def _enrich_metrics(metrics: dict, trainer) -> dict:
    metrics = dict(metrics)
    if trainer is not None:
        metrics["train_time_sec"] = getattr(trainer, "total_train_time_sec", 0.0)
        metrics["avg_epoch_time_sec"] = getattr(trainer, "avg_epoch_time_sec", 0.0)
        metrics["epochs_ran"] = getattr(trainer, "epochs_ran", 0)
        metrics["peak_torch_allocated_gb"] = getattr(trainer, "peak_torch_allocated_gb", 0.0)
        metrics["peak_gpu_vram_gb"] = getattr(trainer, "peak_gpu_vram_gb", 0.0)
        metrics["peak_tracked_memory_kind"] = getattr(trainer, "peak_tracked_memory_kind", "unknown")
    else:
        metrics["train_time_sec"] = 0.0
        metrics["avg_epoch_time_sec"] = 0.0
        metrics["epochs_ran"] = 0
        metrics["peak_torch_allocated_gb"] = 0.0
        metrics["peak_gpu_vram_gb"] = 0.0
        metrics["peak_tracked_memory_kind"] = "none"
    return metrics


def _save_metrics_json(name: str, metrics: dict):
    Path("outputs").mkdir(exist_ok=True)
    out_path = f"outputs/{name}_metrics.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[done] metrics saved → {out_path}")


def _aggregate_cv_metrics(fold_results: list[dict]) -> dict:
    numeric_keys = [
        "rmse",
        "mae",
        "smape",
        "mda",
        "pop_weighted_rmse",
        "train_time_sec",
        "avg_epoch_time_sec",
        "epochs_ran",
        "peak_torch_allocated_gb",
        "peak_gpu_vram_gb",
    ]
    summary = {"cv_folds": len(fold_results), "fold_metrics": fold_results}
    for key in numeric_keys:
        values = [float(m[key]) for m in fold_results if key in m]
        if not values:
            continue
        mean = sum(values) / len(values)
        std = (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5
        summary[key] = mean
        summary[f"{key}_std"] = std

    kinds = sorted({m.get("peak_tracked_memory_kind", "unknown") for m in fold_results})
    summary["peak_tracked_memory_kind"] = kinds[0] if len(kinds) == 1 else ",".join(kinds)
    return summary


def run_experiment(name: str, rolling_cv_folds: int = 1) -> dict:
    import importlib
    module_path, cfg = EXPERIMENTS[name]
    mod = importlib.import_module(module_path)

    if rolling_cv_folds <= 1:
        print(f"\n{'='*60}")
        print(f"  EXPERIMENT: {name}")
        print(f"{'='*60}\n")
        metrics, preds, trues, trainer = mod.run(cfg)
        metrics = _enrich_metrics(metrics, trainer)
        _save_metrics_json(name, metrics)
        return metrics

    if cfg is None:
        raise ValueError(f"Rolling CV requires an explicit ExperimentConfig for {name}.")

    fold_results = []
    for fold_idx in range(1, rolling_cv_folds + 1):
        cfg_fold = copy.deepcopy(cfg)
        cfg_fold.name = f"{name}_fold{fold_idx}"
        cfg_fold.data.rolling_folds = rolling_cv_folds
        cfg_fold.data.rolling_fold_index = fold_idx

        print(f"\n{'='*60}")
        print(f"  EXPERIMENT: {name}  [fold {fold_idx}/{rolling_cv_folds}]")
        print(f"{'='*60}\n")

        metrics, preds, trues, trainer = mod.run(cfg_fold)
        metrics = _enrich_metrics(metrics, trainer)
        metrics["fold_index"] = fold_idx
        _save_metrics_json(cfg_fold.name, metrics)
        fold_results.append(metrics)

    summary = _aggregate_cv_metrics(fold_results)
    _save_metrics_json(name, summary)
    return summary


def save_comparison_csv(results: dict[str, dict], out_path: str = "outputs/comparison_metrics.csv"):
    Path(out_path).parent.mkdir(exist_ok=True)
    fieldnames = [
        "experiment",
        "rmse",
        "mae",
        "smape",
        "mda",
        "pop_weighted_rmse",
        "train_time_sec",
        "avg_epoch_time_sec",
        "epochs_ran",
        "peak_torch_allocated_gb",
        "peak_gpu_vram_gb",
        "peak_tracked_memory_kind",
        "cv_folds",
        "rmse_std",
        "mae_std",
        "smape_std",
        "mda_std",
        "pop_weighted_rmse_std",
        "train_time_sec_std",
        "avg_epoch_time_sec_std",
        "epochs_ran_std",
        "peak_torch_allocated_gb_std",
        "peak_gpu_vram_gb_std",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for experiment, metrics in results.items():
            row = {"experiment": experiment}
            for key in fieldnames[1:]:
                row[key] = metrics.get(key, "")
            writer.writerow(row)
    print(f"[done] comparison CSV saved → {out_path}")


def run_experiment_list(names: list[str], csv_path: str, rolling_cv_folds: int = 1):
    results = {}
    for name in names:
        results[name] = run_experiment(name, rolling_cv_folds=rolling_cv_folds)
    save_comparison_csv(results, out_path=csv_path)
    print("\n\n===== COMPARISON =====")
    print(
        f"{'Experiment':<45} {'RMSE':>10} {'MAE':>10} {'SMAPE':>10} "
        f"{'MDA':>8} {'PW-RMSE':>10} {'Train(s)':>10} {'TorchGB':>10} {'VRAMGB':>10}"
    )
    print("-" * 132)
    for name, m in results.items():
        print(
            f"{name:<45} {m['rmse']:>10.2f} {m['mae']:>10.2f} "
            f"{m['smape']:>10.2f}% {m.get('mda', float('nan')):>8.3f} "
            f"{m.get('pop_weighted_rmse', float('nan')):>10.2f} "
            f"{m.get('train_time_sec', 0.0):>10.1f} "
            f"{m.get('peak_torch_allocated_gb', 0.0):>10.2f} "
            f"{m.get('peak_gpu_vram_gb', 0.0):>10.2f}"
        )


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="COVID GNN experiment runner")
    parser.add_argument("--experiment", "-e", type=str, default=None,
                        help="Experiment name or 'all'")
    parser.add_argument("--list", "-l", action="store_true",
                        help="List available experiments")
    parser.add_argument("--all-spain", action="store_true",
                        help="Run the full Spain sweep only")
    parser.add_argument("--all-brazil", action="store_true",
                        help="Run the full Brazil sweep only")
    parser.add_argument("--rolling-cv-folds", type=int, default=1,
                        help="Use rolling-origin CV with this many folds (default: 1 = disabled)")
    args = parser.parse_args()

    if args.all_spain:
        run_experiment_list(
            ordered_spain_experiments(),
            "outputs/comparison_metrics_spain.csv",
            rolling_cv_folds=args.rolling_cv_folds,
        )
        return

    if args.all_brazil:
        run_experiment_list(
            ordered_brazil_experiments(),
            "outputs/comparison_metrics_brazil.csv",
            rolling_cv_folds=args.rolling_cv_folds,
        )
        return

    if args.list or (args.experiment is None and not args.all_spain and not args.all_brazil):
        print("Available experiments:\n")

        print("  Brazil (city-scale variants per experiment family):")
        for scale, (_, _log, label) in _BRAZIL_SCALES.items():
            print(f"    [{scale:8s}]  {label}")
        print()
        for model in _BRAZIL_CFG_FACTORIES:
            variants = "  |  ".join(f"{model}_{s}" for s in _BRAZIL_SCALES)
            print(f"    {variants}")
        print()

        print("  Spain:")
        for name in EXPERIMENTS:
            if name.startswith("spain_"):
                print(f"    {name}")
        return

    if args.experiment == "all":
        run_experiment_list(
            ordered_all_experiments(),
            "outputs/comparison_metrics.csv",
            rolling_cv_folds=args.rolling_cv_folds,
        )
        return

    if args.experiment not in EXPERIMENTS:
        print(f"Unknown experiment: '{args.experiment}'")
        print("Run with --list to see available experiments.")
        return

    run_experiment(args.experiment, rolling_cv_folds=args.rolling_cv_folds)


if __name__ == "__main__":
    main()

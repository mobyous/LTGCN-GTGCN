"""
Persistence baseline experiment.

Predicts the next-step case count as the last observed count in the input window:
    y_t = y_{t-1}
"""
import torch

from src.config import DataConfig, ModelConfig, TrainConfig, ExperimentConfig
from src.data.dataset import make_temporal_splits
from src.data.graph import (
    build_brazil_graph,
    build_spain_graph,
    extract_brazil_backbone,
    extract_spain_backbone,
    load_spain_mobility,
    top_k_brazil_cities,
)
from src.data.loader import load_brazil_covid, load_spain_covid
from src.data.preprocess import (
    drop_constant_nodes_brazil,
    drop_constant_nodes_spain,
    filter_brazil_covid,
    filter_spain_covid,
)
from src.experiments._common import (
    get_device,
    load_brazil_pop_weights,
    load_spain_pop_weights,
    set_seed,
)
from src.training.metrics import compute_all


def _prepare_brazil(cfg: ExperimentConfig, device: torch.device):
    covid_df = load_brazil_covid()
    filtered_df = filter_brazil_covid(covid_df)
    filtered_df = drop_constant_nodes_brazil(filtered_df)

    candidate_ids = set(filtered_df["ibgeID"].unique())
    if cfg.data.brazil_city_top_k > 0:
        candidate_ids &= top_k_brazil_cities(cfg.data.brazil_city_top_k, candidate_ids)
        print(
            f"[data] Brazil: restricted to top-{cfg.data.brazil_city_top_k} cities by population "
            f"({len(candidate_ids)} remain)"
        )

    backbone_df, _ = extract_brazil_backbone(
        city_whitelist=candidate_ids,
        alpha=cfg.data.backbone_alpha,
        top_k=cfg.data.backbone_top_k,
    )
    _, node_order = build_brazil_graph(backbone_df)
    filtered_df = filtered_df[filtered_df["ibgeID"].isin(set(node_order))]
    pop_weights = load_brazil_pop_weights(node_order)

    split_kwargs = dict(
        df=filtered_df,
        date_col="date",
        id_col="ibgeID",
        value_col="newCases",
        node_order=node_order,
        input_window=cfg.data.input_window,
        output_window=cfg.data.output_window,
        train_ratio=cfg.data.train_ratio,
        val_ratio=cfg.data.val_ratio,
        rolling_folds=cfg.data.rolling_folds,
        rolling_fold_index=cfg.data.rolling_fold_index,
        device=device,
        log_transform=cfg.data.log_transform,
    )
    return make_temporal_splits(**split_kwargs), pop_weights


def _prepare_spain(cfg: ExperimentConfig, device: torch.device):
    import pandas as pd

    covid_df = load_spain_covid()
    filtered_df = filter_spain_covid(covid_df)
    filtered_df = drop_constant_nodes_spain(filtered_df)

    mobility_df = load_spain_mobility()
    centrality_df = pd.read_csv("data/Spain/centrality_provinces.csv")
    cities = sorted(centrality_df["nomemun"].unique())

    backbone_df = extract_spain_backbone(
        mobility_df=mobility_df,
        cities=cities,
        alpha=cfg.data.backbone_alpha,
        top_k=cfg.data.backbone_top_k,
    )
    _, node_order = build_spain_graph(backbone_df)

    name_to_cod = dict(zip(centrality_df["nomemun"], centrality_df["Codmundv"].astype(int)))
    cod_order = [name_to_cod[name] for name in node_order if name in name_to_cod]
    filtered_df = filtered_df[filtered_df["cod_ine"].isin(set(cod_order))]
    pop_weights = load_spain_pop_weights(cod_order, name_to_cod)

    split_kwargs = dict(
        df=filtered_df,
        date_col="Fecha",
        id_col="cod_ine",
        value_col="Casos",
        node_order=cod_order,
        input_window=cfg.data.input_window,
        output_window=cfg.data.output_window,
        train_ratio=cfg.data.train_ratio,
        val_ratio=cfg.data.val_ratio,
        rolling_folds=cfg.data.rolling_folds,
        rolling_fold_index=cfg.data.rolling_fold_index,
        device=device,
        log_transform=cfg.data.log_transform,
    )
    return make_temporal_splits(**split_kwargs), pop_weights


def _denormalize(values: torch.Tensor, means: torch.Tensor, stds: torch.Tensor, log_transform: bool) -> torch.Tensor:
    values = values * stds + means
    if log_transform:
        values = torch.expm1(values)
    return values.clamp(min=0)


def run(cfg: ExperimentConfig = None):
    if cfg is None:
        cfg = ExperimentConfig(
            name="spain_persistence",
            data=DataConfig(country="spain", input_window=14, output_window=1),
            model=ModelConfig(model_type="persistence"),
            train=TrainConfig(epochs=0),
        )

    if cfg.data.output_window != 1:
        raise ValueError("Persistence baseline currently supports output_window=1 only.")

    set_seed(cfg.train.seed)
    device = get_device(cfg.train.device)
    print(f"[{cfg.name}] device={device}")

    if cfg.data.country == "brazil":
        splits, pop_weights = _prepare_brazil(cfg, device)
    elif cfg.data.country == "spain":
        splits, pop_weights = _prepare_spain(cfg, device)
    else:
        raise ValueError(f"Unsupported country: {cfg.data.country}")

    _, _, _, _, X_te, Y_te, means, stds = splits
    if len(X_te) == 0:
        raise ValueError("No test windows available for persistence evaluation.")

    last_norm = X_te[:, -1, :, 0].cpu()
    true_norm = Y_te[:, 0, :, 0].cpu()
    means = means.cpu()
    stds = stds.cpu()

    preds_real = _denormalize(last_norm, means, stds, cfg.data.log_transform)
    trues_real = _denormalize(true_norm, means, stds, cfg.data.log_transform)
    lasts_real = _denormalize(last_norm, means, stds, cfg.data.log_transform)

    metrics = compute_all(
        pred=preds_real,
        target=trues_real,
        last_known=lasts_real,
        pop_weights=pop_weights,
    )

    print(
        f"\n[TEST] RMSE={metrics['rmse']:.2f} | "
        f"MAE={metrics['mae']:.2f} | "
        f"SMAPE={metrics['smape']:.2f}% | "
        f"MDA={metrics.get('mda', float('nan')):.3f}",
        flush=True,
    )
    if "pop_weighted_rmse" in metrics:
        print(
            f"[TEST] Pop-weighted RMSE={metrics['pop_weighted_rmse']:.2f}",
            flush=True,
        )

    return metrics, preds_real, trues_real, None


if __name__ == "__main__":
    run()

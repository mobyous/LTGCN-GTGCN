"""
Spain — Graph WaveNet experiment.
"""
import pandas as pd

from src.config import DataConfig, ModelConfig, TrainConfig, ExperimentConfig
from src.data.loader import load_spain_covid
from src.data.preprocess import filter_spain_covid, drop_constant_nodes_spain
from src.data.graph import load_spain_mobility, extract_spain_backbone, build_spain_graph
from src.data.dataset import make_temporal_splits, make_loaders
from src.models.graph_wavenet import GraphWaveNet
from src.training.trainer import Trainer
from src.analytics.plots import plot_losses
from src.experiments._common import (
    set_seed,
    get_device,
    build_optimizer_and_scheduler,
    get_edge_tensors,
    load_spain_pop_weights,
    maybe_compile,
)


def run(cfg: ExperimentConfig = None):
    if cfg is None:
        cfg = ExperimentConfig(
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
        )

    set_seed(cfg.train.seed)
    device = get_device(cfg.train.device)
    print(f"[{cfg.name}] device={device}")

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
    pyg_data, node_order = build_spain_graph(backbone_df)
    edge_index, edge_weight = get_edge_tensors(pyg_data)

    name_to_cod = dict(zip(centrality_df["nomemun"], centrality_df["Codmundv"].astype(int)))
    cod_order = [name_to_cod[n] for n in node_order if n in name_to_cod]
    filtered_df = filtered_df[filtered_df["cod_ine"].isin(set(cod_order))]

    X_tr, Y_tr, X_vl, Y_vl, X_te, Y_te, means, stds = make_temporal_splits(
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
    )
    train_loader, val_loader, test_loader = make_loaders(
        X_tr, Y_tr, X_vl, Y_vl, X_te, Y_te, batch_size=cfg.train.batch_size
    )

    pop_weights = load_spain_pop_weights(cod_order, name_to_cod)

    model = GraphWaveNet(
        num_nodes=len(cod_order),
        in_dim=1,
        out_dim=1,
        residual_channels=cfg.model.residual_channels,
        dilation_channels=cfg.model.dilation_channels,
        skip_channels=cfg.model.skip_channels,
        end_channels=cfg.model.end_channels,
        blocks=cfg.model.blocks,
        layers=cfg.model.num_layers,
    ).to(device)

    optimizer, scheduler = build_optimizer_and_scheduler(model, cfg.train)
    model = maybe_compile(model, device)

    trainer = Trainer(
        model=model,
        edge_index=edge_index,
        edge_weight=edge_weight,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        means=means,
        stds=stds,
        pop_weights=pop_weights,
        grad_clip=cfg.train.grad_clip,
        output_window=cfg.data.output_window,
    )

    trainer.fit(train_loader, val_loader, cfg.train.epochs)
    plot_losses(trainer.train_losses, trainer.val_losses, title=cfg.name, save_path=f"outputs/{cfg.name}_loss.png")

    metrics, preds, trues = trainer.test(test_loader)
    return metrics, preds, trues, trainer


if __name__ == "__main__":
    run()

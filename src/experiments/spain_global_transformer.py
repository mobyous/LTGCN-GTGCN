"""
Spain — Global Transformer experiment.
"""
import torch
import pandas as pd
from src.config import DataConfig, ModelConfig, TrainConfig, ExperimentConfig
from src.data.loader import load_spain_covid
from src.data.preprocess import filter_spain_covid, drop_constant_nodes_spain
from src.data.graph import load_spain_mobility, extract_spain_backbone, build_spain_graph
from src.data.dataset import make_temporal_splits, make_loaders
from src.models.global_transformer import GlobalTransformer
from src.training.trainer import Trainer
from src.analytics.plots import plot_losses
from src.experiments._common import set_seed, get_device, build_optimizer_and_scheduler, get_edge_tensors, load_spain_pop_weights, maybe_compile


def run(cfg: ExperimentConfig = None):
    if cfg is None:
        cfg = ExperimentConfig(
            name="spain_global_transformer",
            data=DataConfig(country="spain", input_window=14, output_window=1),
            model=ModelConfig(
                model_type="global_transformer",
                hidden_dim=128, nhead=4, num_layers=1,
                ff_dropout=0.2, gat_heads=2,
            ),
            train=TrainConfig(epochs=50, batch_size=16, lr=3e-4, scheduler_step=5, scheduler_gamma=0.5),
        )

    set_seed(cfg.train.seed)
    device = get_device(cfg.train.device)
    print(f"[{cfg.name}] device={device}")

    # ── Data ─────────────────────────────────────────────────────────────────
    covid_df    = load_spain_covid()
    filtered_df = filter_spain_covid(covid_df)
    filtered_df = drop_constant_nodes_spain(filtered_df)

    mobility_df   = load_spain_mobility()
    centrality_df = pd.read_csv("data/Spain/centrality_provinces.csv")
    cities        = sorted(centrality_df["nomemun"].unique())

    backbone_df = extract_spain_backbone(
        mobility_df=mobility_df,
        cities=cities,
        alpha=cfg.data.backbone_alpha,
        top_k=cfg.data.backbone_top_k,
    )
    pyg_data, node_order = build_spain_graph(backbone_df)
    edge_index, edge_weight = get_edge_tensors(pyg_data)

    name_to_cod = dict(zip(centrality_df["nomemun"], centrality_df["Codmundv"].astype(int)))
    cod_order   = [name_to_cod[n] for n in node_order if n in name_to_cod]
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

    # ── Population weights ────────────────────────────────────────────────────
    pop_weights = load_spain_pop_weights(cod_order, name_to_cod)

    # ── Model ─────────────────────────────────────────────────────────────────
    N = len(cod_order)
    # Weighted degree: sum of edge weights incident to each node, normalized to [0,1].
    # Computed on CPU (edge_index lives there until Trainer.to(device)), then moved.
    _ei = edge_index.cpu()
    _ew = edge_weight.cpu() if edge_weight is not None else torch.ones(_ei.size(1))
    deg = torch.zeros(N)
    deg.scatter_add_(0, _ei[0], _ew)
    deg.scatter_add_(0, _ei[1], _ew)         # undirected: count both directions
    node_features = (deg / (deg.max() + 1e-8)).unsqueeze(-1).to(device)  # [N, 1]

    model = GlobalTransformer(
        input_dim=1,
        gcn_dim=1,
        hidden_dim=cfg.model.hidden_dim,
        nhead=cfg.model.nhead,
        num_layers=cfg.model.num_layers,
        num_nodes=N,
        forecast_dim=1,
        attn_dropout=cfg.model.attn_dropout,
        ff_dropout=cfg.model.ff_dropout,
        gat_heads=cfg.model.gat_heads,
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
        node_features=node_features,
        pop_weights=pop_weights,
        grad_clip=cfg.train.grad_clip,
        output_window=cfg.data.output_window,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer.fit(train_loader, val_loader, cfg.train.epochs)
    plot_losses(trainer.train_losses, trainer.val_losses,
                title=cfg.name, save_path=f"outputs/{cfg.name}_loss.png")

    # ── Test ──────────────────────────────────────────────────────────────────
    metrics, preds, trues = trainer.test(test_loader)
    return metrics, preds, trues, trainer


if __name__ == "__main__":
    run()

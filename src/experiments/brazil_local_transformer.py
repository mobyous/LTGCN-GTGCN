"""
Brazil — Local Transformer experiment.
"""
from src.config import DataConfig, ModelConfig, TrainConfig, ExperimentConfig
from src.data.loader import load_brazil_covid
from src.data.preprocess import filter_brazil_covid, drop_constant_nodes_brazil
from src.data.graph import extract_brazil_backbone, build_brazil_graph, top_k_brazil_cities
from src.data.dataset import make_temporal_splits, make_loaders
from src.models.local_transformer import LocalTransformer
from src.training.trainer import Trainer
from src.analytics.plots import plot_losses
from src.experiments._common import set_seed, get_device, build_optimizer_and_scheduler, get_edge_tensors, load_brazil_pop_weights, maybe_compile


def run(cfg: ExperimentConfig = None):
    if cfg is None:
        cfg = ExperimentConfig(
            name="brazil_local_transformer",
            data=DataConfig(country="brazil", input_window=14, output_window=1),
            model=ModelConfig(model_type="local_transformer", trans_hidden=64, nhead=4, graph_feat_dim=1),
            train=TrainConfig(epochs=50, batch_size=32, lr=1e-3, scheduler_step=10, scheduler_gamma=0.7),
        )

    set_seed(cfg.train.seed)
    device = get_device(cfg.train.device)
    print(f"[{cfg.name}] device={device}")

    # ── Data ─────────────────────────────────────────────────────────────────
    covid_df    = load_brazil_covid()
    filtered_df = filter_brazil_covid(covid_df)
    filtered_df = drop_constant_nodes_brazil(filtered_df)

    candidate_ids = set(filtered_df["ibgeID"].unique())
    if cfg.data.brazil_city_top_k > 0:
        candidate_ids &= top_k_brazil_cities(cfg.data.brazil_city_top_k, candidate_ids)
        print(f"[data] Brazil: restricted to top-{cfg.data.brazil_city_top_k} cities by population ({len(candidate_ids)} remain)")

    backbone_df, backbone_cities = extract_brazil_backbone(
        city_whitelist=candidate_ids,
        alpha=cfg.data.backbone_alpha,
        top_k=cfg.data.backbone_top_k,
    )
    pyg_data, node_order = build_brazil_graph(backbone_df)
    edge_index, edge_weight = get_edge_tensors(pyg_data)

    filtered_df = filtered_df[filtered_df["ibgeID"].isin(set(node_order))]

    X_tr, Y_tr, X_vl, Y_vl, X_te, Y_te, means, stds = make_temporal_splits(
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
    train_loader, val_loader, test_loader = make_loaders(
        X_tr, Y_tr, X_vl, Y_vl, X_te, Y_te, batch_size=cfg.train.batch_size
    )

    # ── Population weights ────────────────────────────────────────────────────
    pop_weights = load_brazil_pop_weights(node_order)

    # ── Model ─────────────────────────────────────────────────────────────────
    N = len(node_order)
    model = LocalTransformer(
        in_channels=1,
        graph_feat_dim=cfg.model.graph_feat_dim,
        trans_hidden=cfg.model.trans_hidden,
        out_channels=1,
        num_nodes=N,
        nhead=cfg.model.nhead,
        num_layers=cfg.model.num_layers,
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
        log_transform=cfg.data.log_transform,
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

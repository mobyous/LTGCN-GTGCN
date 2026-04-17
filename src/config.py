from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent  # project root


@dataclass
class DataConfig:
    country: str = "brazil"          # "brazil" | "spain"
    backbone_alpha: float = 0.01
    backbone_top_k: int = 5
    input_window: int = 14
    output_window: int = 1
    train_ratio: float = 0.70
    val_ratio: float = 0.15          # test = 1 - train - val = 0.15
    brazil_city_top_k: int = 0       # 0 = all cities; >0 = top-K by population (e.g. 40 matches old notebooks)
    log_transform: bool = False      # apply log1p before z-scoring; recommended for Brazil (multi-wave non-stationarity)
    rolling_folds: int = 1           # 1 = standard single split; >1 = rolling-origin CV
    rolling_fold_index: int = 1      # 1-based fold index used when rolling_folds > 1


@dataclass
class ModelConfig:
    model_type: str = "gcrn"         # "gcrn" | "graph_wavenet" | "gcrn_transformer" | "local_transformer" | "global_transformer" | "lstm_gcn" | "persistence" | LTGCN ablation variants
    hidden_channels: int = 64        # GCRN / LSTM hidden size
    # Shared transformer params
    trans_hidden: int = 64
    hidden_dim: int = 64
    embed_dim: int = 10
    cheb_k: int = 2
    residual_channels: int = 32
    dilation_channels: int = 32
    skip_channels: int = 256
    end_channels: int = 512
    blocks: int = 4
    nhead: int = 4
    num_layers: int = 1
    attn_dropout: float = 0.0
    ff_dropout: float = 0.2
    gat_heads: int = 2
    graph_feat_dim: int = 1


@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    scheduler_step: int = 10
    scheduler_gamma: float = 0.7
    grad_clip: float = 1.0
    seed: int = 42
    device: str = "auto"             # "auto" | "cuda" | "mps" | "cpu"


@dataclass
class ExperimentConfig:
    name: str = "experiment"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    save_dir: str = "outputs"

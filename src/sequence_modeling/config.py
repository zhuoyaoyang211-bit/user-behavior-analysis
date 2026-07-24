"""Configuration objects for sequence model training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal


ModelName = Literal["lstm", "gru", "din"]


@dataclass(frozen=True)
class SequenceDataConfig:
    """File paths and dataset construction parameters.

    Attributes:
        behavior_path: Cleaned behavior log path.
        item_dim_path: Item dimension table path.
        train_path: Training sample table path.
        val_path: Validation sample table path.
        output_dir: Directory for metrics and model checkpoints.
        max_seq_len: Maximum number of historical events per sample.
        item_hash_size: Number of hash buckets for item embeddings.
        max_train_rows: Optional row cap for quick experiments.
        max_val_rows: Optional validation row cap for quick experiments.
    """

    behavior_path: Path
    item_dim_path: Path
    train_path: Path
    val_path: Path
    output_dir: Path
    max_seq_len: int = 50
    item_hash_size: int = 500_000
    max_train_rows: int | None = None
    max_val_rows: int | None = None

    def to_dict(self) -> dict[str, object]:
        """Convert config to a JSON-serializable dictionary."""
        values = asdict(self)
        for key, value in values.items():
            if isinstance(value, Path):
                values[key] = str(value)
        return values


@dataclass(frozen=True)
class SequenceTrainConfig:
    """Training hyperparameters shared by LSTM, GRU, and DIN.

    Attributes:
        embedding_dim: Item and category embedding dimension.
        behavior_embedding_dim: Behavior type embedding dimension.
        hidden_size: RNN hidden size or DIN attention projection size.
        num_layers: Number of recurrent layers for LSTM/GRU.
        dropout: Dropout rate used in model heads.
        batch_size: Mini-batch size.
        learning_rate: Adam learning rate.
        epochs: Maximum training epochs.
        early_stopping_patience: Epochs to wait for PR-AUC improvement.
        num_workers: DataLoader worker count.
        device: Torch device. Use "auto" to prefer MPS when available.
        use_pos_weight: Whether BCE loss uses negative/positive class ratio.
        random_state: Random seed.
    """

    embedding_dim: int = 32
    behavior_embedding_dim: int = 8
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.2
    batch_size: int = 512
    learning_rate: float = 1e-3
    epochs: int = 5
    early_stopping_patience: int = 2
    num_workers: int = 0
    device: str = "auto"
    use_pos_weight: bool = True
    random_state: int = 42

    def to_dict(self) -> dict[str, object]:
        """Convert config to a JSON-serializable dictionary."""
        return asdict(self)

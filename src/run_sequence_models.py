"""Train LSTM, GRU, and DIN sequence baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from config import get_config
from common.logger import get_logger
from sequence_modeling.config import (
    ModelName,
    SequenceDataConfig,
    SequenceTrainConfig,
)
from sequence_modeling.dataset import (
    SequenceDataset,
    load_behavior_history,
    load_sequence_samples,
)
from sequence_modeling.trainer import train_sequence_model


logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    project_root = get_config().project_root
    output_dir = project_root / "output"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["lstm", "gru", "din"],
        default=["lstm", "gru", "din"],
    )
    parser.add_argument(
        "--behavior-path",
        type=Path,
        default=output_dir / "cleaned_data.parquet",
    )
    parser.add_argument(
        "--item-dim-path",
        type=Path,
        default=output_dir / "dim_item.parquet",
    )
    parser.add_argument("--train-path", type=Path, default=output_dir / "train.parquet")
    parser.add_argument("--val-path", type=Path, default=output_dir / "val.parquet")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=output_dir / "sequence_models",
    )
    parser.add_argument("--max-seq-len", type=int, default=50)
    parser.add_argument("--item-hash-size", type=int, default=500_000)
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-val-rows", type=int, default=None)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--behavior-embedding-dim", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--disable-pos-weight", action="store_true")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def build_configs(
    args: argparse.Namespace,
) -> tuple[SequenceDataConfig, SequenceTrainConfig]:
    """Build typed config objects from parsed arguments."""
    data_config = SequenceDataConfig(
        behavior_path=args.behavior_path,
        item_dim_path=args.item_dim_path,
        train_path=args.train_path,
        val_path=args.val_path,
        output_dir=args.output_dir,
        max_seq_len=args.max_seq_len,
        item_hash_size=args.item_hash_size,
        max_train_rows=args.max_train_rows,
        max_val_rows=args.max_val_rows,
    )
    train_config = SequenceTrainConfig(
        embedding_dim=args.embedding_dim,
        behavior_embedding_dim=args.behavior_embedding_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        early_stopping_patience=args.early_stopping_patience,
        num_workers=args.num_workers,
        device=args.device,
        use_pos_weight=not args.disable_pos_weight,
        random_state=args.random_state,
    )
    return data_config, train_config


def write_run_config(
    data_config: SequenceDataConfig,
    train_config: SequenceTrainConfig,
    model_names: list[str],
) -> None:
    """Persist run configuration."""
    data_config.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "models": model_names,
        "data_config": data_config.to_dict(),
        "train_config": train_config.to_dict(),
    }
    (data_config.output_dir / "sequence_run_config.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    """Train requested sequence baselines and write comparison artifacts."""
    args = parse_args()
    data_config, train_config = build_configs(args)
    model_names: list[ModelName] = args.models
    write_run_config(data_config, train_config, model_names)

    logger.info("Loading behavior history from %s", data_config.behavior_path)
    behavior_history = load_behavior_history(
        behavior_path=data_config.behavior_path,
        item_hash_size=data_config.item_hash_size,
    )
    logger.info(
        "Loaded %d users | item vocab %d | category vocab %d | behavior vocab %d",
        len(behavior_history.histories),
        behavior_history.item_vocab_size,
        behavior_history.category_vocab_size,
        behavior_history.behavior_vocab_size,
    )

    logger.info("Loading train samples from %s", data_config.train_path)
    train_samples = load_sequence_samples(
        sample_path=data_config.train_path,
        item_dim_path=data_config.item_dim_path,
        item_hash_size=data_config.item_hash_size,
        max_rows=data_config.max_train_rows,
    )
    logger.info("Loading validation samples from %s", data_config.val_path)
    val_samples = load_sequence_samples(
        sample_path=data_config.val_path,
        item_dim_path=data_config.item_dim_path,
        item_hash_size=data_config.item_hash_size,
        max_rows=data_config.max_val_rows,
    )
    train_dataset = SequenceDataset(
        samples=train_samples,
        behavior_history=behavior_history,
        max_seq_len=data_config.max_seq_len,
    )
    val_dataset = SequenceDataset(
        samples=val_samples,
        behavior_history=behavior_history,
        max_seq_len=data_config.max_seq_len,
    )
    logger.info(
        "Dataset ready | train %d rows | val %d rows | max_seq_len %d",
        len(train_dataset),
        len(val_dataset),
        data_config.max_seq_len,
    )

    results = []
    for model_name in model_names:
        result = train_sequence_model(
            model_name=model_name,
            behavior_history=behavior_history,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            train_config=train_config,
            output_dir=data_config.output_dir,
        )
        results.append(result.to_dict())

    rows = []
    for result in results:
        row = {
            "model": result["model"],
            "best_epoch": result["best_epoch"],
            "train_seconds": result["train_seconds"],
            "checkpoint_path": result["checkpoint_path"],
        }
        row.update(result["metrics"])
        rows.append(row)

    metrics_df = pd.DataFrame(rows).sort_values("pr_auc_ap", ascending=False)
    metrics_path = data_config.output_dir / "sequence_baseline_metrics.csv"
    results_path = data_config.output_dir / "sequence_results.json"
    metrics_df.to_csv(metrics_path, index=False)
    results_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Sequence baseline metrics saved to %s", metrics_path)
    logger.info("\n%s", metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()

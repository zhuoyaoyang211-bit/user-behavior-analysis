"""Evaluate sequence model baselines under multiple thresholds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from common.logger import get_logger
from config import get_config
from sequence_modeling.config import SequenceDataConfig, SequenceTrainConfig
from sequence_modeling.dataset import (
    SequenceDataset,
    load_behavior_history,
    load_sequence_samples,
)
from sequence_modeling.metrics import compute_threshold_metrics
from sequence_modeling.models import build_sequence_model
from sequence_modeling.trainer import predict_probabilities, resolve_device


logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    project_root = get_config().project_root
    output_dir = project_root / "output"

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--run-config-path", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=output_dir / "sequence_threshold_analysis",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[
            0.05,
            0.10,
            0.15,
            0.20,
            0.25,
            0.30,
            0.35,
            0.40,
            0.45,
            0.50,
            0.60,
            0.70,
            0.80,
            0.90,
        ],
    )
    return parser.parse_args()


def load_run_configs(
    run_config_path: Path,
) -> tuple[SequenceDataConfig, SequenceTrainConfig]:
    """Load data and train configs saved by run_sequence_models.py."""
    payload = json.loads(run_config_path.read_text(encoding="utf-8"))
    data_payload = payload["data_config"]
    train_payload = payload["train_config"]
    data_config = SequenceDataConfig(
        behavior_path=Path(data_payload["behavior_path"]),
        item_dim_path=Path(data_payload["item_dim_path"]),
        train_path=Path(data_payload["train_path"]),
        val_path=Path(data_payload["val_path"]),
        output_dir=Path(data_payload["output_dir"]),
        max_seq_len=int(data_payload["max_seq_len"]),
        item_hash_size=int(data_payload["item_hash_size"]),
        max_train_rows=data_payload["max_train_rows"],
        max_val_rows=data_payload["max_val_rows"],
    )
    train_config = SequenceTrainConfig(
        embedding_dim=int(train_payload["embedding_dim"]),
        behavior_embedding_dim=int(train_payload["behavior_embedding_dim"]),
        hidden_size=int(train_payload["hidden_size"]),
        num_layers=int(train_payload["num_layers"]),
        dropout=float(train_payload["dropout"]),
        batch_size=int(train_payload["batch_size"]),
        learning_rate=float(train_payload["learning_rate"]),
        epochs=int(train_payload["epochs"]),
        early_stopping_patience=int(train_payload["early_stopping_patience"]),
        num_workers=int(train_payload["num_workers"]),
        device=str(train_payload["device"]),
        use_pos_weight=bool(train_payload["use_pos_weight"]),
        random_state=int(train_payload["random_state"]),
    )
    return data_config, train_config


def load_checkpoint_model(
    checkpoint_path: Path,
    train_config: SequenceTrainConfig,
    device: torch.device,
) -> torch.nn.Module:
    """Load a trained sequence model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_sequence_model(
        model_name=checkpoint["model_name"],
        item_vocab_size=checkpoint["vocab_sizes"]["item_vocab_size"],
        category_vocab_size=checkpoint["vocab_sizes"]["category_vocab_size"],
        behavior_vocab_size=checkpoint["vocab_sizes"]["behavior_vocab_size"],
        embedding_dim=train_config.embedding_dim,
        behavior_embedding_dim=train_config.behavior_embedding_dim,
        hidden_size=train_config.hidden_size,
        num_layers=train_config.num_layers,
        dropout=train_config.dropout,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def write_threshold_report(
    output_dir: Path,
    model_name: str,
    metrics_df: pd.DataFrame,
) -> None:
    """Write markdown threshold report."""
    report_df = metrics_df.copy()
    for col in report_df.select_dtypes(include=["float"]).columns:
        report_df[col] = report_df[col].map(lambda value: f"{value:.6f}")

    columns = [
        "threshold",
        "precision",
        "recall",
        "f1",
        "accuracy",
        "tn",
        "fp",
        "fn",
        "tp",
    ]
    view = report_df[columns]
    header = "| " + " | ".join(view.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(view.columns)) + " |"
    body = ["| " + " | ".join(map(str, row)) + " |" for row in view.to_numpy()]
    lines = [
        f"# {model_name.upper()} Threshold Analysis",
        "",
        "This report evaluates fixed probability thresholds only. It does not "
        "change model parameters or retrain the baseline model.",
        "",
        header,
        separator,
        *body,
        "",
    ]
    (output_dir / f"{model_name}_threshold_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    """Evaluate a trained sequence checkpoint under multiple thresholds."""
    args = parse_args()
    data_config, train_config = load_run_configs(args.run_config_path)
    device_name = args.device or train_config.device
    device = resolve_device(device_name)
    batch_size = args.batch_size or train_config.batch_size

    logger.info("Loading behavior history from %s", data_config.behavior_path)
    behavior_history = load_behavior_history(
        behavior_path=data_config.behavior_path,
        item_hash_size=data_config.item_hash_size,
    )
    logger.info("Loading validation samples from %s", data_config.val_path)
    val_samples = load_sequence_samples(
        sample_path=data_config.val_path,
        item_dim_path=data_config.item_dim_path,
        item_hash_size=data_config.item_hash_size,
        max_rows=data_config.max_val_rows,
    )
    val_dataset = SequenceDataset(
        samples=val_samples,
        behavior_history=behavior_history,
        max_seq_len=data_config.max_seq_len,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=train_config.num_workers,
    )

    model = load_checkpoint_model(args.checkpoint_path, train_config, device)
    checkpoint = torch.load(
        args.checkpoint_path, map_location=device, weights_only=False
    )
    model_name = checkpoint["model_name"]
    logger.info("Scoring %s checkpoint on %s", model_name, device)
    y_true, y_prob = predict_probabilities(model, val_loader, device)

    threshold_rows = compute_threshold_metrics(
        y_true=y_true,
        y_prob=y_prob,
        thresholds=args.thresholds,
    )
    metrics_df = pd.DataFrame(threshold_rows).sort_values(
        ["f1", "precision"],
        ascending=False,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / f"{model_name}_threshold_metrics.csv"
    prediction_path = args.output_dir / f"{model_name}_validation_predictions.csv"
    metrics_df.to_csv(metrics_path, index=False)
    pd.DataFrame({"label": y_true, "probability": y_prob}).to_csv(
        prediction_path,
        index=False,
    )
    write_threshold_report(args.output_dir, model_name, metrics_df)
    logger.info("Threshold metrics saved to %s", metrics_path)
    logger.info("\n%s", metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()

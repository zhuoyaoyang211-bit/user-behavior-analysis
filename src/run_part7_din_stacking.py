"""Tune DIN on a distribution-matched split and test tree-model fusion.

This script is the Part7 deep-learning workflow:

1. Stratified-sample 500,000 rows from the original sequence training data,
   preserving both label and time-stratum distributions.
2. Split each time stratum chronologically into tune-train and tune-validation.
3. Compare a small, manual set of DIN configurations on tune-validation.
4. Refit the best DIN configuration on all 500,000 sampled rows.
5. Evaluate the final DIN and fixed-weight blends with Optuna-tuned
   LightGBM/XGBoost on the untouched full validation set.

The ``--final-full-train`` mode skips tuning and refits the selected
configuration on the complete ``train.parquet``.

The fusion section intentionally uses fixed, transparent weighted blends. It
is a pre-test for stacking, not a replacement for out-of-fold meta-learning.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from common.logger import get_logger
from sequence_modeling.config import (
    LrSchedulerName,
    OptimizerName,
    SequenceTrainConfig,
)
from sequence_modeling.dataset import (
    SequenceDataset,
    SequenceSampleFrame,
    category_to_index,
    datetime_to_seconds,
    hash_item_ids,
    load_behavior_history,
    load_sequence_samples,
)
from sequence_modeling.metrics import compute_threshold_metrics
from sequence_modeling.models import build_sequence_model
from sequence_modeling.trainer import (
    build_loss,
    build_optimizer,
    move_batch_to_device,
    predict_probabilities,
    resolve_device,
    set_random_seed,
    train_one_epoch,
    train_sequence_model,
)


logger = get_logger(__name__)
PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_DIR / "output"
DEFAULT_TRAIN_PATH = OUTPUT_DIR / "train.parquet"
DEFAULT_VAL_PATH = OUTPUT_DIR / "val.parquet"
DEFAULT_BEHAVIOR_PATH = OUTPUT_DIR / "cleaned_data.parquet"
DEFAULT_ITEM_DIM_PATH = OUTPUT_DIR / "dim_item.parquet"
DEFAULT_BASELINE_DIR = OUTPUT_DIR / "baseline_models"
DEFAULT_TREE_DIR = OUTPUT_DIR / "optuna_tuned_models"
DEFAULT_TREE_STACKING_DIR = OUTPUT_DIR / "tree_stacking"
DEFAULT_OUTPUT_DIR = OUTPUT_DIR / "part7_din_stacking"
RANDOM_STATE = 42
TARGET_COL = "label"
SEQUENCE_COLUMNS = ["user_id", "item_id", "last_time", "label"]
DEFAULT_THRESHOLDS = [
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
]

SPECIAL_DATES = {pd.Timestamp("2025-12-12").date()}


def build_time_strata(frame: pd.DataFrame) -> pd.Series:
    """Build the same time strata used by the project sample split.

    The date is retained so that every calendar date represented in the
    sampled training data can appear in both tuning partitions.  The
    day-type and hour-bin fields ensure the teacher-requested weekday,
    weekend/special-day, and time-of-day coverage is preserved.
    """
    dt = pd.to_datetime(frame["last_time"])
    day_type = np.select(
        [
            dt.dt.date.isin(SPECIAL_DATES),
            dt.dt.dayofweek >= 5,
        ],
        [
            "special",
            "weekend",
        ],
        default="weekday",
    )
    hour_bin = pd.cut(
        dt.dt.hour,
        bins=[-1, 5, 11, 17, 23],
        labels=["night", "morning", "afternoon", "evening"],
    )
    return (
        dt.dt.date.astype(str)
        + "|"
        + pd.Series(day_type, index=frame.index).astype(str)
        + "|"
        + hour_bin.astype(str)
    )


def proportional_allocations(
    counts: pd.Series,
    target: int,
) -> pd.Series:
    """Allocate an exact target count in proportion to group sizes."""
    if target < 0 or target > int(counts.sum()):
        raise ValueError("Allocation target is outside the available row count.")
    if target == 0:
        return pd.Series(0, index=counts.index, dtype="int64")

    raw = counts.astype(float) * (target / float(counts.sum()))
    allocations = np.floor(raw).astype("int64")
    allocations = allocations.clip(upper=counts.astype("int64"))
    remaining = target - int(allocations.sum())

    if remaining > 0:
        order = (raw - allocations).sort_values(ascending=False).index
        for key in order:
            if remaining == 0:
                break
            if allocations.loc[key] < counts.loc[key]:
                allocations.loc[key] += 1
                remaining -= 1

    if remaining != 0:
        raise RuntimeError("Could not allocate the requested sample size.")
    return allocations


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--val-path", type=Path, default=DEFAULT_VAL_PATH)
    parser.add_argument(
        "--behavior-path",
        type=Path,
        default=DEFAULT_BEHAVIOR_PATH,
    )
    parser.add_argument(
        "--item-dim-path",
        type=Path,
        default=DEFAULT_ITEM_DIM_PATH,
    )
    parser.add_argument("--tree-dir", type=Path, default=DEFAULT_TREE_DIR)
    parser.add_argument(
        "--lightgbm-path",
        type=Path,
        default=DEFAULT_BASELINE_DIR / "lightgbm_baseline.joblib",
        help="Selected LightGBM artifact used in DIN + tree fusion.",
    )
    parser.add_argument(
        "--xgboost-path",
        type=Path,
        default=DEFAULT_TREE_DIR / "xgboost_optuna.joblib",
        help="Selected XGBoost artifact used in DIN + tree fusion.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-size", type=int, default=500_000)
    parser.add_argument("--tune-val-frac", type=float, default=0.2)
    parser.add_argument("--max-configs", type=int, default=4)
    parser.add_argument(
        "--final-full-train",
        action="store_true",
        help="Skip tuning and train the selected DIN configuration on all train rows.",
    )
    parser.add_argument(
        "--fusion-only",
        action="store_true",
        help="Recompute tree/DIN fixed blends from an existing DIN validation prediction file.",
    )
    parser.add_argument(
        "--din-pred-path",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "din_full_train_validation_predictions.csv",
        help="Existing DIN validation prediction file used by --fusion-only.",
    )
    parser.add_argument(
        "--tree-pred-path",
        type=Path,
        default=DEFAULT_TREE_STACKING_DIR / "stacking_validation_predictions.csv",
        help=(
            "Existing LightGBM/XGBoost validation prediction file used by "
            "--fusion-only when available."
        ),
    )
    parser.add_argument(
        "--best-config-path",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "din_best_config.json",
        help="Previously saved DIN configuration used by --final-full-train.",
    )
    parser.add_argument(
        "--full-train-output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "din_final_full_train",
        help="Output directory for the full-training DIN checkpoint and metrics.",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--item-hash-size", type=int, default=200_000)
    parser.add_argument("--max-seq-len", type=int, default=50)
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=DEFAULT_THRESHOLDS,
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate arguments before loading the large behavior history."""
    if args.sample_size <= 0:
        raise ValueError("--sample-size must be positive.")
    if not 0 < args.tune_val_frac < 1:
        raise ValueError("--tune-val-frac must be in (0, 1).")
    if args.max_configs <= 0:
        raise ValueError("--max-configs must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    if args.early_stopping_patience <= 0:
        raise ValueError("--early-stopping-patience must be positive.")
    if args.item_hash_size <= 0:
        raise ValueError("--item-hash-size must be positive.")
    if args.max_seq_len <= 0:
        raise ValueError("--max-seq-len must be positive.")
    if args.final_full_train:
        if not args.best_config_path.exists():
            raise FileNotFoundError(
                f"Best DIN config not found: {args.best_config_path}"
            )
        return
    if args.fusion_only and not args.din_pred_path.exists():
        raise FileNotFoundError(f"DIN prediction file not found: {args.din_pred_path}")
    if not args.tree_dir.exists():
        raise FileNotFoundError(f"Tree model directory not found: {args.tree_dir}")
    for model_name, model_path in {
        "lightgbm": args.lightgbm_path,
        "xgboost": args.xgboost_path,
    }.items():
        if not model_path.exists():
            raise FileNotFoundError(f"Tree model not found for {model_name}: {model_path}")


def stratified_sample(
    frame: pd.DataFrame,
    sample_size: int,
    random_state: int,
) -> pd.DataFrame:
    """Sample rows while preserving label and time-stratum ratios."""
    if sample_size >= len(frame):
        result = frame.copy()
        result["_time_stratum"] = build_time_strata(result)
        return result.reset_index(drop=True)
    if sample_size <= 0:
        raise ValueError("sample_size must be positive.")

    working = frame.copy()
    working["_time_stratum"] = build_time_strata(working)
    parts = []

    # First preserve the original label ratio, then preserve each label's
    # time-stratum ratio.  This avoids over-representing rare positive rows
    # from only a few dates or time periods.
    label_allocations = proportional_allocations(
        working[TARGET_COL].value_counts().sort_index(),
        sample_size,
    )
    for label, label_target in label_allocations.items():
        label_rows = working[working[TARGET_COL] == label]
        stratum_counts = label_rows["_time_stratum"].value_counts().sort_index()
        stratum_allocations = proportional_allocations(
            stratum_counts,
            int(label_target),
        )
        for stratum, n_rows in stratum_allocations.items():
            if n_rows <= 0:
                continue
            stratum_rows = label_rows[label_rows["_time_stratum"] == stratum]
            parts.append(
                stratum_rows.sample(
                    n=int(n_rows),
                    random_state=random_state,
                )
            )

    return (
        pd.concat(parts, axis=0)
        .sample(frac=1.0, random_state=random_state)
        .reset_index(drop=True)
    )


def temporal_split(
    frame: pd.DataFrame,
    val_frac: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Split every time stratum chronologically into tune-train/tune-val.

    This follows the project's time-stratified holdout design: each partition
    keeps weekday, weekend/special-day, hour-bin, and calendar-date coverage.
    It is intentionally not a global future-only split.
    """
    del random_state
    ordered = frame.copy()
    ordered["last_time"] = pd.to_datetime(ordered["last_time"])
    if "_time_stratum" not in ordered.columns:
        ordered["_time_stratum"] = build_time_strata(ordered)

    train_parts = []
    val_parts = []
    for _, group in ordered.groupby("_time_stratum", sort=True):
        group = group.sort_values("last_time", kind="mergesort")
        n = len(group)
        if n < 2:
            train_parts.append(group)
            continue
        split_index = int(round(n * (1.0 - val_frac)))
        split_index = max(1, min(split_index, n - 1))
        train_parts.append(group.iloc[:split_index])
        val_parts.append(group.iloc[split_index:])

    train_frame = pd.concat(train_parts, ignore_index=True)
    val_frame = pd.concat(val_parts, ignore_index=True)
    if train_frame.empty or val_frame.empty:
        raise ValueError(
            "Time-stratified split produced an empty train or validation set."
        )
    cutoff = pd.to_datetime(ordered["last_time"]).min()
    if val_parts:
        cutoff = min(part["last_time"].min() for part in val_parts)
    return (
        train_frame.drop(columns=["_time_stratum"]).reset_index(drop=True),
        val_frame.drop(columns=["_time_stratum"]).reset_index(drop=True),
        cutoff,
    )


def frame_to_sequence_samples(
    frame: pd.DataFrame,
    item_dim_path: Path,
    item_hash_size: int,
) -> Any:
    """Convert raw sequence rows into the dense dataset representation."""
    samples = frame[SEQUENCE_COLUMNS].copy()
    item_dim = pd.read_parquet(item_dim_path, columns=["item_category"])
    samples = samples.join(item_dim, on="item_id")
    return SequenceSampleFrame(
        user_id=samples["user_id"].to_numpy(dtype="int64", copy=True),
        target_item_index=hash_item_ids(samples["item_id"], item_hash_size),
        target_category_index=category_to_index(samples["item_category"]),
        sample_time=datetime_to_seconds(samples["last_time"]),
        label=samples[TARGET_COL].to_numpy(dtype="float32", copy=True),
    )


def build_sequence_dataset(
    frame: pd.DataFrame,
    behavior_history: Any,
    item_dim_path: Path,
    item_hash_size: int,
    max_seq_len: int,
) -> SequenceDataset:
    """Build a SequenceDataset from raw sequence rows."""
    return SequenceDataset(
        samples=frame_to_sequence_samples(
            frame=frame,
            item_dim_path=item_dim_path,
            item_hash_size=item_hash_size,
        ),
        behavior_history=behavior_history,
        max_seq_len=max_seq_len,
    )


def din_configs(
    batch_size: int,
    epochs: int,
    early_stopping_patience: int,
    num_workers: int,
    device: str,
) -> list[tuple[str, SequenceTrainConfig]]:
    """Return a small manual DIN search space."""
    common = {
        "batch_size": batch_size,
        "epochs": epochs,
        "early_stopping_patience": early_stopping_patience,
        "num_workers": num_workers,
        "device": device,
        "use_pos_weight": True,
        "random_state": RANDOM_STATE,
    }
    configs = [
        (
            "din_standard_adamw",
            SequenceTrainConfig(
                embedding_dim=32,
                behavior_embedding_dim=8,
                hidden_size=64,
                num_layers=1,
                dropout=0.2,
                attention_heads=0,
                optimizer="adamw",
                learning_rate=1e-3,
                weight_decay=1e-5,
                lr_scheduler="reduce_on_plateau",
                lr_scheduler_factor=0.5,
                lr_scheduler_patience=1,
                min_learning_rate=1e-6,
                **common,
            ),
        ),
        (
            "din_multihead_regularized",
            SequenceTrainConfig(
                embedding_dim=32,
                behavior_embedding_dim=8,
                hidden_size=64,
                num_layers=1,
                dropout=0.3,
                attention_heads=4,
                optimizer="adamw",
                learning_rate=1e-3,
                weight_decay=1e-5,
                lr_scheduler="reduce_on_plateau",
                lr_scheduler_factor=0.5,
                lr_scheduler_patience=1,
                min_learning_rate=1e-6,
                **common,
            ),
        ),
        (
            "din_wide_multihead",
            SequenceTrainConfig(
                embedding_dim=32,
                behavior_embedding_dim=8,
                hidden_size=128,
                num_layers=1,
                dropout=0.3,
                attention_heads=4,
                optimizer="adamw",
                learning_rate=1e-3,
                weight_decay=1e-5,
                lr_scheduler="reduce_on_plateau",
                lr_scheduler_factor=1.0 / 2.0,
                lr_scheduler_patience=1,
                min_learning_rate=1e-6,
                **common,
            ),
        ),
        (
            "din_low_dropout_multihead",
            SequenceTrainConfig(
                embedding_dim=48,
                behavior_embedding_dim=8,
                hidden_size=64,
                num_layers=1,
                dropout=0.2,
                attention_heads=4,
                optimizer="adamw",
                learning_rate=1e-3,
                weight_decay=1e-5,
                lr_scheduler="reduce_on_plateau",
                lr_scheduler_factor=0.5,
                lr_scheduler_patience=1,
                min_learning_rate=1e-6,
                **common,
            ),
        ),
    ]
    return configs


def config_to_dict(config: SequenceTrainConfig) -> dict[str, object]:
    """Serialize a training config."""
    return asdict(config)


def load_checkpoint_model(
    checkpoint_path: Path,
    device: torch.device,
) -> torch.nn.Module:
    """Load a DIN checkpoint using the exact config stored in the checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    train_payload = checkpoint["train_config"]
    model = build_sequence_model(
        model_name="din",
        item_vocab_size=checkpoint["vocab_sizes"]["item_vocab_size"],
        category_vocab_size=checkpoint["vocab_sizes"]["category_vocab_size"],
        behavior_vocab_size=checkpoint["vocab_sizes"]["behavior_vocab_size"],
        embedding_dim=int(train_payload["embedding_dim"]),
        behavior_embedding_dim=int(train_payload["behavior_embedding_dim"]),
        hidden_size=int(train_payload["hidden_size"]),
        num_layers=int(train_payload["num_layers"]),
        dropout=float(train_payload["dropout"]),
        attention_heads=int(train_payload.get("attention_heads", 0)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def fit_fixed_din(
    behavior_history: Any,
    train_dataset: SequenceDataset,
    train_config: SequenceTrainConfig,
    output_dir: Path,
    epochs: int,
) -> Path:
    """Refit DIN on all sampled rows for a fixed number of selected epochs.

    The full validation set is deliberately not used for early stopping here.
    The epoch count comes from the distribution-matched tuning split.
    """
    set_random_seed(train_config.random_state)
    device = resolve_device(train_config.device)
    model = build_sequence_model(
        model_name="din",
        item_vocab_size=behavior_history.item_vocab_size,
        category_vocab_size=behavior_history.category_vocab_size,
        behavior_vocab_size=behavior_history.behavior_vocab_size,
        embedding_dim=train_config.embedding_dim,
        behavior_embedding_dim=train_config.behavior_embedding_dim,
        hidden_size=train_config.hidden_size,
        num_layers=train_config.num_layers,
        dropout=train_config.dropout,
        attention_heads=train_config.attention_heads,
    ).to(device)
    criterion = build_loss(train_dataset, train_config, device)
    optimizer = build_optimizer(model, train_config)
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=train_config.num_workers,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    epoch_losses = []
    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        epoch_losses.append({"epoch": epoch, "train_loss": loss})
        logger.info("DIN final refit epoch %d/%d | loss %.6f", epoch, epochs, loss)

    checkpoint_path = output_dir / "din_baseline.pt"
    torch.save(
        {
            "model_name": "din",
            "model_state_dict": model.state_dict(),
            "train_config": train_config.to_dict(),
            "metrics": {
                "refit_epochs": epochs,
                "train_loss_history": epoch_losses,
            },
            "vocab_sizes": {
                "item_vocab_size": behavior_history.item_vocab_size,
                "category_vocab_size": behavior_history.category_vocab_size,
                "behavior_vocab_size": behavior_history.behavior_vocab_size,
            },
        },
        checkpoint_path,
    )
    return checkpoint_path


def predict_dataset(
    checkpoint_path: Path,
    dataset: SequenceDataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict a sequence dataset from a saved checkpoint."""
    model = load_checkpoint_model(checkpoint_path, device)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return predict_probabilities(model, loader, device)


def load_selected_config(
    config_path: Path,
    batch_size: int,
    num_workers: int,
    device: str,
) -> tuple[str, SequenceTrainConfig, int]:
    """Load the selected DIN configuration and fixed epoch count."""
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    config_name = str(payload["config_name"])
    config_values = dict(payload["train_config"])
    tune_metrics = payload.get("tune_metrics", {})
    selected_epochs = max(1, int(tune_metrics.get("best_epoch", config_values["epochs"])))

    # There is no validation set in the final full-data fit, so the scheduler
    # is disabled and the epoch count is fixed from the tuning experiment.
    config_values.update(
        {
            "batch_size": batch_size,
            "epochs": selected_epochs,
            "early_stopping_patience": max(
                1,
                int(config_values.get("early_stopping_patience", 2)),
            ),
            "num_workers": num_workers,
            "device": device,
            "lr_scheduler": "none",
        }
    )
    return config_name, SequenceTrainConfig(**config_values), selected_epochs


def run_final_full_train(args: argparse.Namespace) -> None:
    """Train the selected DIN configuration on the complete train.parquet."""
    start = time.time()
    args.full_train_output_dir.mkdir(parents=True, exist_ok=True)

    train_frame = pd.read_parquet(args.train_path, columns=SEQUENCE_COLUMNS)
    val_frame = pd.read_parquet(args.val_path, columns=SEQUENCE_COLUMNS)
    logger.info(
        "Final full DIN training | train rows %d (positive %d) | val rows %d (positive %d)",
        len(train_frame),
        int(train_frame[TARGET_COL].sum()),
        len(val_frame),
        int(val_frame[TARGET_COL].sum()),
    )
    if len(train_frame) != 3_278_239:
        logger.warning(
            "Expected the project's full train row count 3,278,239, got %d.",
            len(train_frame),
        )

    config_name, train_config, selected_epochs = load_selected_config(
        config_path=args.best_config_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
    )
    logger.info(
        "Using selected DIN config %s | epochs=%d | scheduler=%s",
        config_name,
        selected_epochs,
        train_config.lr_scheduler,
    )

    behavior_history = load_behavior_history(
        behavior_path=args.behavior_path,
        item_hash_size=args.item_hash_size,
    )
    full_train_dataset = build_sequence_dataset(
        train_frame,
        behavior_history,
        args.item_dim_path,
        args.item_hash_size,
        args.max_seq_len,
    )
    full_val_dataset = build_sequence_dataset(
        val_frame,
        behavior_history,
        args.item_dim_path,
        args.item_hash_size,
        args.max_seq_len,
    )
    checkpoint_path = fit_fixed_din(
        behavior_history=behavior_history,
        train_dataset=full_train_dataset,
        train_config=train_config,
        output_dir=args.full_train_output_dir,
        epochs=selected_epochs,
    )
    _, val_probability = predict_dataset(
        checkpoint_path,
        full_val_dataset,
        args.batch_size,
        args.num_workers,
        resolve_device(args.device),
    )
    y_val = val_frame[TARGET_COL].to_numpy(dtype=np.int8, copy=True)
    metrics = binary_summary(y_val, val_probability, args.thresholds)
    metrics_payload = {
        "checkpoint_path": str(checkpoint_path),
        "config_name": config_name,
        "train_config": config_to_dict(train_config),
        "train_rows": len(train_frame),
        "train_positive_count": int(train_frame[TARGET_COL].sum()),
        "validation_rows": len(val_frame),
        "validation_positive_count": int(val_frame[TARGET_COL].sum()),
        "metrics": metrics,
        "training_policy": (
            "fixed epochs selected during 500k tuning; no validation early stopping "
            "during final full-train fit"
        ),
        "elapsed_seconds": time.time() - start,
    }
    (args.output_dir / "din_full_train_metrics.json").write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "label": y_val,
            "din_probability": val_probability,
        }
    ).to_csv(
        args.output_dir / "din_full_train_validation_predictions.csv",
        index=False,
    )
    metric_rows_for_thresholds(
        "din_full_train",
        y_val,
        val_probability,
        args.thresholds,
    ).to_csv(
        args.output_dir / "din_full_train_threshold_metrics.csv",
        index=False,
    )
    (args.output_dir / "din_full_train_setup.json").write_text(
        json.dumps(
            {
                "train_path": str(args.train_path),
                "val_path": str(args.val_path),
                "behavior_path": str(args.behavior_path),
                "item_dim_path": str(args.item_dim_path),
                "best_config_path": str(args.best_config_path),
                "full_train_output_dir": str(args.full_train_output_dir),
                **metrics_payload,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    logger.info("Final full DIN outputs written to %s", args.output_dir)
    logger.info("Final full DIN metrics: %s", json.dumps(metrics, ensure_ascii=False))


def binary_summary(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: list[float],
) -> dict[str, float | int]:
    """Return ranking and default-threshold metrics."""
    y_pred = (y_prob >= 0.5).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc_ap": float(average_precision_score(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, np.clip(y_prob, 1e-15, 1 - 1e-15))),
        "precision_at_0_5": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_at_0_5": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_at_0_5": float(f1_score(y_true, y_pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "best_threshold": best_threshold(thresholds, y_true, y_prob),
    }


def best_threshold(
    thresholds: list[float],
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> float:
    """Return the fixed candidate threshold with the highest F1."""
    threshold_df = pd.DataFrame(
        compute_threshold_metrics(y_true, y_prob, thresholds)
    )
    row = threshold_df.sort_values(["f1", "precision"], ascending=False).iloc[0]
    return float(row["threshold"])


def metric_rows_for_thresholds(
    name: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: list[float],
) -> pd.DataFrame:
    """Create a detailed threshold table for one prediction vector."""
    rows = pd.DataFrame(compute_threshold_metrics(y_true, y_prob, thresholds))
    rows.insert(0, "model", name)
    return rows


def load_tree_models(
    lightgbm_path: Path,
    xgboost_path: Path,
) -> dict[str, dict[str, Any]]:
    """Load the selected LightGBM and XGBoost artifacts."""
    models = {}
    for model_name, path in {
        "lightgbm": lightgbm_path,
        "xgboost": xgboost_path,
    }.items():
        payload = joblib.load(path)
        payload["artifact_path"] = str(path)
        models[model_name] = payload
        logger.info(
            "Loaded %s | features %d | artifact %s",
            model_name,
            len(payload["feature_cols"]),
            path,
        )
    if models["lightgbm"]["feature_cols"] != models["xgboost"]["feature_cols"]:
        raise ValueError("LightGBM and XGBoost feature columns do not match.")
    return models


def tree_predictions(
    tree_models: dict[str, dict[str, Any]],
    frame: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """Score a raw feature frame with the tuned tree models."""
    predictions = {}
    for model_name, payload in tree_models.items():
        feature_cols = payload["feature_cols"]
        features = frame[feature_cols].to_numpy(dtype=np.float32, copy=True)
        if np.isnan(features).any():
            raise ValueError(f"Missing feature values found for {model_name}.")
        predictions[model_name] = payload["model"].predict_proba(features)[:, 1]
    return predictions


def fixed_blends(
    predictions: dict[str, np.ndarray],
) -> dict[str, tuple[dict[str, float], np.ndarray]]:
    """Return transparent fixed-weight fusion candidates."""
    return {
        "tree_equal": (
            {"lightgbm": 0.5, "xgboost": 0.5},
            0.5 * predictions["lightgbm"] + 0.5 * predictions["xgboost"],
        ),
        "tree_plus_din_equal": (
            {"lightgbm": 1 / 3, "xgboost": 1 / 3, "din": 1 / 3},
            (
                predictions["lightgbm"]
                + predictions["xgboost"]
                + predictions["din"]
            )
            / 3,
        ),
        "tree_plus_din_20pct": (
            {"lightgbm": 0.4, "xgboost": 0.4, "din": 0.2},
            0.4 * predictions["lightgbm"]
            + 0.4 * predictions["xgboost"]
            + 0.2 * predictions["din"],
        ),
        "tree_plus_din_40pct": (
            {"lightgbm": 0.3, "xgboost": 0.3, "din": 0.4},
            0.3 * predictions["lightgbm"]
            + 0.3 * predictions["xgboost"]
            + 0.4 * predictions["din"],
        ),
    }


def evaluate_blends(
    predictions: dict[str, np.ndarray],
    y_true: np.ndarray,
    thresholds: list[float],
    evaluation_set: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate fixed blends and return summary plus threshold rows."""
    summary_rows = []
    threshold_frames = []
    for name, (weights, probability) in fixed_blends(predictions).items():
        summary = binary_summary(y_true, probability, thresholds)
        summary_rows.append(
            {
                "evaluation_set": evaluation_set,
                "blend": name,
                "weights": json.dumps(weights, ensure_ascii=False),
                **summary,
            }
        )
        threshold_frames.append(
            metric_rows_for_thresholds(
                name=f"{evaluation_set}_{name}",
                y_true=y_true,
                y_prob=probability,
                thresholds=thresholds,
            )
        )
    return pd.DataFrame(summary_rows), pd.concat(threshold_frames, ignore_index=True)


def write_tree_din_fusion_outputs(
    args: argparse.Namespace,
    din_probability: np.ndarray,
    output_prefix: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate selected tree artifacts plus DIN predictions on full validation."""
    use_cached_tree_predictions = (
        bool(getattr(args, "fusion_only", False))
        and getattr(args, "tree_pred_path", Path()).exists()
    )
    if use_cached_tree_predictions:
        tree_pred = pd.read_csv(args.tree_pred_path)
        required_cols = {"lightgbm", "xgboost", TARGET_COL}
        missing_cols = sorted(required_cols - set(tree_pred.columns))
        if missing_cols:
            raise ValueError(
                f"Tree prediction file missing required columns: {missing_cols}"
            )
        y_true = tree_pred[TARGET_COL].to_numpy(dtype=np.int8, copy=True)
        tree_predictions_full = {
            "lightgbm": tree_pred["lightgbm"].to_numpy(dtype=np.float64, copy=True),
            "xgboost": tree_pred["xgboost"].to_numpy(dtype=np.float64, copy=True),
        }
        logger.info("Loaded cached tree predictions from %s", args.tree_pred_path)
    else:
        tree_models = load_tree_models(
            lightgbm_path=args.lightgbm_path,
            xgboost_path=args.xgboost_path,
        )
        tree_val_frame = pd.read_parquet(
            args.val_path,
            columns=tree_models["lightgbm"]["feature_cols"] + [TARGET_COL],
        )
        y_true = tree_val_frame[TARGET_COL].to_numpy(dtype=np.int8, copy=True)
        tree_predictions_full = tree_predictions(tree_models, tree_val_frame)

    if len(din_probability) != len(y_true):
        raise ValueError(
            f"DIN prediction rows {len(din_probability)} != validation rows {len(y_true)}."
        )

    full_predictions = {
        **tree_predictions_full,
        "din": din_probability,
    }
    full_blend_summary, full_blend_thresholds = evaluate_blends(
        full_predictions,
        y_true,
        args.thresholds,
        evaluation_set="full_validation",
    )
    summary_path = args.output_dir / f"{output_prefix}stacking_comparison_full_validation.csv"
    threshold_path = args.output_dir / f"{output_prefix}stacking_threshold_metrics_full_validation.csv"
    prediction_path = args.output_dir / f"{output_prefix}stacking_full_validation_predictions.csv"
    full_blend_summary.to_csv(summary_path, index=False)
    full_blend_thresholds.to_csv(threshold_path, index=False)
    pd.DataFrame(full_predictions).assign(label=y_true).to_csv(
        prediction_path,
        index=False,
    )
    setup_path = args.output_dir / f"{output_prefix}tree_din_fusion_setup.json"
    setup_path.write_text(
        json.dumps(
            {
                "val_path": str(args.val_path),
                "tree_dir": str(args.tree_dir),
                "lightgbm_path": str(args.lightgbm_path),
                "xgboost_path": str(args.xgboost_path),
                "din_pred_path": str(getattr(args, "din_pred_path", "")),
                "tree_pred_path": str(getattr(args, "tree_pred_path", "")),
                "used_cached_tree_predictions": use_cached_tree_predictions,
                "output_prefix": output_prefix,
                "fusion_policy": (
                    "Fixed weights over LightGBM baseline, XGBoost Optuna, and DIN; "
                    "validation is used only for reporting."
                ),
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    logger.info("Tree + DIN fusion summary saved to %s", summary_path)
    return full_blend_summary, full_blend_thresholds


def run_fusion_only(args: argparse.Namespace) -> None:
    """Recompute selected tree + DIN blends from an existing DIN prediction file."""
    args.output_dir.mkdir(parents=True, exist_ok=True)
    din_pred = pd.read_csv(args.din_pred_path)
    if "din_probability" not in din_pred.columns:
        raise ValueError("DIN prediction file must contain `din_probability`.")
    if TARGET_COL in din_pred.columns:
        val_labels = pd.read_parquet(args.val_path, columns=[TARGET_COL])[TARGET_COL]
        if not np.array_equal(
            din_pred[TARGET_COL].to_numpy(dtype=np.int8, copy=True),
            val_labels.to_numpy(dtype=np.int8, copy=True),
        ):
            raise ValueError("DIN prediction labels do not match val.parquet labels.")
    din_probability = din_pred["din_probability"].to_numpy(dtype=np.float64, copy=True)
    summary, _ = write_tree_din_fusion_outputs(
        args,
        din_probability=din_probability,
    )
    print("\nTree + DIN fusion metrics:")
    print(summary.to_string(index=False))


def main() -> None:
    """Run Part7 DIN tuning and tree-model fusion evaluation."""
    args = parse_args()
    validate_args(args)
    if args.final_full_train:
        run_final_full_train(args)
        return
    if args.fusion_only:
        run_fusion_only(args)
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()

    logger.info("Loading sequence training rows from %s", args.train_path)
    train_frame = pd.read_parquet(args.train_path, columns=SEQUENCE_COLUMNS)
    sampled_frame = stratified_sample(
        train_frame,
        sample_size=min(args.sample_size, len(train_frame)),
        random_state=RANDOM_STATE,
    )
    tune_train_frame, tune_val_frame, tune_cutoff = temporal_split(
        sampled_frame,
        val_frac=args.tune_val_frac,
        random_state=RANDOM_STATE,
    )
    val_frame = pd.read_parquet(args.val_path, columns=SEQUENCE_COLUMNS)
    train_max_time = pd.to_datetime(train_frame["last_time"]).max()
    full_val_min_time = pd.to_datetime(val_frame["last_time"]).min()
    logger.info(
        "DIN split | sample %d (positive %d) | tune train %d (positive %d) | "
        "tune val %d (positive %d) | full val %d (positive %d)",
        len(sampled_frame),
        int(sampled_frame[TARGET_COL].sum()),
        len(tune_train_frame),
        int(tune_train_frame[TARGET_COL].sum()),
        len(tune_val_frame),
        int(tune_val_frame[TARGET_COL].sum()),
        len(val_frame),
        int(val_frame[TARGET_COL].sum()),
    )
    logger.info(
        "Time-stratified tune split | train time %s ~ %s | val time %s ~ %s | "
        "earliest val time %s",
        tune_train_frame["last_time"].min(),
        tune_train_frame["last_time"].max(),
        tune_val_frame["last_time"].min(),
        tune_val_frame["last_time"].max(),
        tune_cutoff,
    )
    if full_val_min_time <= train_max_time:
        logger.warning(
            "Existing val.parquet overlaps train.parquet in time: "
            "train_max=%s, val_min=%s. Treat it as the project holdout, "
            "not a strict future-only evaluation set.",
            train_max_time,
            full_val_min_time,
        )

    behavior_history = load_behavior_history(
        behavior_path=args.behavior_path,
        item_hash_size=args.item_hash_size,
    )
    tune_train_dataset = build_sequence_dataset(
        tune_train_frame,
        behavior_history,
        args.item_dim_path,
        args.item_hash_size,
        args.max_seq_len,
    )
    tune_val_dataset = build_sequence_dataset(
        tune_val_frame,
        behavior_history,
        args.item_dim_path,
        args.item_hash_size,
        args.max_seq_len,
    )
    full_sample_dataset = build_sequence_dataset(
        sampled_frame,
        behavior_history,
        args.item_dim_path,
        args.item_hash_size,
        args.max_seq_len,
    )
    full_val_dataset = build_sequence_dataset(
        val_frame,
        behavior_history,
        args.item_dim_path,
        args.item_hash_size,
        args.max_seq_len,
    )

    all_configs = din_configs(
        batch_size=args.batch_size,
        epochs=args.epochs,
        early_stopping_patience=args.early_stopping_patience,
        num_workers=args.num_workers,
        device=args.device,
    )[: args.max_configs]
    trial_rows = []
    trial_root = args.output_dir / "din_trials"
    trial_root.mkdir(parents=True, exist_ok=True)

    for config_name, train_config in all_configs:
        logger.info("Training DIN config %s", config_name)
        trial_dir = trial_root / config_name
        result = train_sequence_model(
            model_name="din",
            behavior_history=behavior_history,
            train_dataset=tune_train_dataset,
            val_dataset=tune_val_dataset,
            train_config=train_config,
            output_dir=trial_dir,
        )
        checkpoint_path = Path(result.checkpoint_path)
        tune_train_true, tune_train_prob = predict_dataset(
            checkpoint_path,
            tune_train_dataset,
            args.batch_size,
            args.num_workers,
            resolve_device(args.device),
        )
        tune_val_true, tune_val_prob = predict_dataset(
            checkpoint_path,
            tune_val_dataset,
            args.batch_size,
            args.num_workers,
            resolve_device(args.device),
        )
        train_summary = binary_summary(
            tune_train_true,
            tune_train_prob,
            args.thresholds,
        )
        val_summary = binary_summary(
            tune_val_true,
            tune_val_prob,
            args.thresholds,
        )
        trial_rows.append(
            {
                "config_name": config_name,
                "checkpoint_path": str(checkpoint_path),
                "best_epoch": result.best_epoch,
                "train_seconds": result.train_seconds,
                "train_pr_auc_ap": train_summary["pr_auc_ap"],
                "tune_val_pr_auc_ap": val_summary["pr_auc_ap"],
                "pr_auc_gap": (
                    train_summary["pr_auc_ap"] - val_summary["pr_auc_ap"]
                ),
                "train_log_loss": train_summary["log_loss"],
                "tune_val_log_loss": val_summary["log_loss"],
                "tune_val_roc_auc": val_summary["roc_auc"],
                "tune_val_f1_at_0_5": val_summary["f1_at_0_5"],
                "tune_val_best_threshold": val_summary["best_threshold"],
                "train_config": json.dumps(
                    config_to_dict(train_config),
                    ensure_ascii=False,
                ),
            }
        )

    trial_df = pd.DataFrame(trial_rows).sort_values(
        ["tune_val_pr_auc_ap", "tune_val_roc_auc"],
        ascending=False,
    )
    trial_df.to_csv(args.output_dir / "din_tuning_trials.csv", index=False)
    best_trial = trial_df.iloc[0]
    best_config_name = str(best_trial["config_name"])
    best_config = dict(all_configs)[best_config_name]
    (args.output_dir / "din_best_config.json").write_text(
        json.dumps(
            {
                "config_name": best_config_name,
                "selection_metric": "tune_val_pr_auc_ap",
                "sample_size": len(sampled_frame),
                "tune_train_rows": len(tune_train_frame),
                "tune_val_rows": len(tune_val_frame),
                "train_config": config_to_dict(best_config),
                "tune_metrics": best_trial.to_dict(),
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    logger.info("Refitting best DIN config %s on all sampled rows", best_config_name)
    final_dir = args.output_dir / "din_final_500k_fullval"
    selected_epochs = max(1, int(best_trial["best_epoch"]))
    final_config = SequenceTrainConfig(
        **{
            **config_to_dict(best_config),
            "epochs": selected_epochs,
            "lr_scheduler": "none",
        }
    )
    final_checkpoint = fit_fixed_din(
        behavior_history=behavior_history,
        train_dataset=full_sample_dataset,
        train_config=final_config,
        output_dir=final_dir,
        epochs=selected_epochs,
    )
    _, din_full_probability = predict_dataset(
        final_checkpoint,
        full_val_dataset,
        args.batch_size,
        args.num_workers,
        resolve_device(args.device),
    )
    din_full_metrics = binary_summary(
        val_frame[TARGET_COL].to_numpy(dtype=np.int8, copy=True),
        din_full_probability,
        args.thresholds,
    )
    (args.output_dir / "din_final_metrics.json").write_text(
        json.dumps(
            {
                "checkpoint_path": str(final_checkpoint),
                "config_name": best_config_name,
                "metrics": din_full_metrics,
                "train_rows": len(sampled_frame),
                "validation_rows": len(val_frame),
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "label": val_frame[TARGET_COL].to_numpy(dtype=np.int8, copy=True),
            "din_probability": din_full_probability,
        }
    ).to_csv(args.output_dir / "din_full_validation_predictions.csv", index=False)
    metric_rows_for_thresholds(
        "din_final",
        val_frame[TARGET_COL].to_numpy(dtype=np.int8, copy=True),
        din_full_probability,
        args.thresholds,
    ).to_csv(args.output_dir / "din_final_threshold_metrics.csv", index=False)

    full_blend_summary, _ = write_tree_din_fusion_outputs(
        args,
        din_probability=din_full_probability,
    )

    setup = {
        "train_path": str(args.train_path),
        "val_path": str(args.val_path),
        "behavior_path": str(args.behavior_path),
        "item_dim_path": str(args.item_dim_path),
        "tree_dir": str(args.tree_dir),
        "lightgbm_path": str(args.lightgbm_path),
        "xgboost_path": str(args.xgboost_path),
        "sample_size": len(sampled_frame),
        "sample_positive_count": int(sampled_frame[TARGET_COL].sum()),
        "tune_train_rows": len(tune_train_frame),
        "tune_train_positive_count": int(tune_train_frame[TARGET_COL].sum()),
        "tune_val_rows": len(tune_val_frame),
        "tune_val_positive_count": int(tune_val_frame[TARGET_COL].sum()),
        "full_val_rows": len(val_frame),
        "full_val_positive_count": int(val_frame[TARGET_COL].sum()),
        "sample_strategy": (
            "stratified_by_label_and_time_stratum: "
            "date|day_type|hour_bin"
        ),
        "tune_split_strategy": (
            "chronological_within_each_time_stratum: "
            "date|day_type|hour_bin"
        ),
        "tune_cutoff_semantics": (
            "earliest tune-validation timestamp; not a global time boundary"
        ),
        "tune_cutoff": str(tune_cutoff),
        "tune_train_time_min": str(pd.to_datetime(tune_train_frame["last_time"]).min()),
        "tune_train_time_max": str(pd.to_datetime(tune_train_frame["last_time"]).max()),
        "tune_val_time_min": str(pd.to_datetime(tune_val_frame["last_time"]).min()),
        "tune_val_time_max": str(pd.to_datetime(tune_val_frame["last_time"]).max()),
        "train_time_min": str(pd.to_datetime(train_frame["last_time"]).min()),
        "train_time_max": str(train_max_time),
        "full_val_time_min": str(full_val_min_time),
        "full_val_time_max": str(pd.to_datetime(val_frame["last_time"]).max()),
        "full_val_is_strict_future": bool(full_val_min_time > train_max_time),
        "random_state": RANDOM_STATE,
        "selection_metric": "tune_val_pr_auc_ap",
        "din_configs": [name for name, _ in all_configs],
        "final_din_checkpoint": str(final_checkpoint),
        "elapsed_seconds": time.time() - start,
    }
    (args.output_dir / "part7_setup.json").write_text(
        json.dumps(setup, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    logger.info("Part7 outputs written to %s", args.output_dir)
    logger.info("\n%s", trial_df.to_string(index=False))
    logger.info("\n%s", full_blend_summary.to_string(index=False))


if __name__ == "__main__":
    main()

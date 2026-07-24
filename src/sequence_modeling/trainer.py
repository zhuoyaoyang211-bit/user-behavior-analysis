"""Training loop for sequence models."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from common.logger import get_logger
from sequence_modeling.config import ModelName, SequenceTrainConfig
from sequence_modeling.dataset import BehaviorHistory, SequenceDataset
from sequence_modeling.metrics import compute_binary_metrics
from sequence_modeling.models import build_sequence_model


logger = get_logger(__name__)


@dataclass(frozen=True)
class TrainingResult:
    """Training result summary."""

    model: str
    best_epoch: int
    train_seconds: float
    best_pr_auc_ap: float
    metrics: dict[str, float | int]
    checkpoint_path: str

    def to_dict(self) -> dict[str, object]:
        """Convert result to a JSON-serializable dictionary."""
        return asdict(self)


def set_random_seed(seed: int) -> None:
    """Set numpy and torch random seeds."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    """Resolve configured device name to a torch.device."""
    if device_name != "auto":
        return torch.device(device_name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def move_batch_to_device(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Move a DataLoader batch to target device."""
    return {key: value.to(device) for key, value in batch.items()}


def build_loss(
    train_dataset: SequenceDataset,
    train_config: SequenceTrainConfig,
    device: torch.device,
) -> nn.Module:
    """Build BCEWithLogitsLoss with optional positive-class weighting."""
    if not train_config.use_pos_weight:
        return nn.BCEWithLogitsLoss()

    labels = train_dataset.samples.label
    positive_count = float(labels.sum())
    negative_count = float(len(labels) - positive_count)
    if positive_count <= 0:
        logger.warning("No positive samples found; disabling pos_weight.")
        return nn.BCEWithLogitsLoss()

    pos_weight = torch.tensor([negative_count / positive_count], device=device)
    logger.info("Using BCE pos_weight=%.4f", float(pos_weight.item()))
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> dict[str, float | int]:
    """Evaluate a model on a validation DataLoader."""
    y_true, y_prob = predict_probabilities(model, data_loader, device)
    return compute_binary_metrics(y_true, y_prob)


def predict_probabilities(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict positive probabilities and labels for a DataLoader.

    Args:
        model: Trained sequence model.
        data_loader: DataLoader to score.
        device: Device used for inference.

    Returns:
        Tuple of ground-truth labels and predicted positive probabilities.
    """
    model.eval()
    probabilities: list[np.ndarray] = []
    labels: list[np.ndarray] = []

    with torch.no_grad():
        for batch in data_loader:
            batch = move_batch_to_device(batch, device)
            logits = model(batch)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            probabilities.append(probs)
            labels.append(batch["label"].detach().cpu().numpy())

    y_prob = np.concatenate(probabilities)
    y_true = np.concatenate(labels)
    return y_true, y_prob


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train one epoch and return average loss."""
    model.train()
    total_loss = 0.0
    total_count = 0

    for batch in data_loader:
        batch = move_batch_to_device(batch, device)
        labels = batch["label"]

        optimizer.zero_grad(set_to_none=True)
        logits = model(batch)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size

    return total_loss / max(total_count, 1)


def train_sequence_model(
    model_name: ModelName,
    behavior_history: BehaviorHistory,
    train_dataset: SequenceDataset,
    val_dataset: SequenceDataset,
    train_config: SequenceTrainConfig,
    output_dir: Path,
) -> TrainingResult:
    """Train one sequence model and save the best checkpoint.

    Args:
        model_name: Model to train: lstm, gru, or din.
        behavior_history: Vocab-size metadata and user history store.
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        train_config: Shared training hyperparameters.
        output_dir: Directory for checkpoints and metric logs.

    Returns:
        TrainingResult containing best validation metrics.
    """
    set_random_seed(train_config.random_state)
    device = resolve_device(train_config.device)
    logger.info("Training %s on %s", model_name, device)

    model = build_sequence_model(
        model_name=model_name,
        item_vocab_size=behavior_history.item_vocab_size,
        category_vocab_size=behavior_history.category_vocab_size,
        behavior_vocab_size=behavior_history.behavior_vocab_size,
        embedding_dim=train_config.embedding_dim,
        behavior_embedding_dim=train_config.behavior_embedding_dim,
        hidden_size=train_config.hidden_size,
        num_layers=train_config.num_layers,
        dropout=train_config.dropout,
    ).to(device)
    criterion = build_loss(train_dataset, train_config, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.learning_rate)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=train_config.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=train_config.num_workers,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"{model_name}_baseline.pt"
    history_path = output_dir / f"{model_name}_history.json"

    best_metrics: dict[str, float | int] | None = None
    best_epoch = 0
    best_pr_auc = -1.0
    patience_used = 0
    epoch_history: list[dict[str, float | int]] = []
    start = time.time()

    for epoch in range(1, train_config.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        metrics = evaluate_model(model, val_loader, device)
        metrics["epoch"] = epoch
        metrics["train_loss"] = train_loss
        epoch_history.append(metrics)

        pr_auc = float(metrics["pr_auc_ap"])
        logger.info(
            "%s epoch %d | loss %.6f | val PR-AUC %.6f | ROC-AUC %.6f",
            model_name,
            epoch,
            train_loss,
            pr_auc,
            float(metrics["roc_auc"]),
        )

        if pr_auc > best_pr_auc:
            best_pr_auc = pr_auc
            best_epoch = epoch
            best_metrics = metrics.copy()
            patience_used = 0
            torch.save(
                {
                    "model_name": model_name,
                    "model_state_dict": model.state_dict(),
                    "train_config": train_config.to_dict(),
                    "metrics": best_metrics,
                    "vocab_sizes": {
                        "item_vocab_size": behavior_history.item_vocab_size,
                        "category_vocab_size": behavior_history.category_vocab_size,
                        "behavior_vocab_size": behavior_history.behavior_vocab_size,
                    },
                },
                checkpoint_path,
            )
        else:
            patience_used += 1
            if patience_used >= train_config.early_stopping_patience:
                logger.info("%s early stopped at epoch %d", model_name, epoch)
                break

    history_path.write_text(
        json.dumps(epoch_history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if best_metrics is None:
        raise RuntimeError(f"{model_name} did not finish a validation epoch.")

    return TrainingResult(
        model=model_name,
        best_epoch=best_epoch,
        train_seconds=time.time() - start,
        best_pr_auc_ap=best_pr_auc,
        metrics=best_metrics,
        checkpoint_path=str(checkpoint_path),
    )

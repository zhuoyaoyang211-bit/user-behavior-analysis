"""Dataset utilities for user behavior sequence models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


SECONDS_PER_DAY = 86_400.0
MAX_TIME_GAP_DAYS = 30.0
PADDING_INDEX = 0


@dataclass(frozen=True)
class UserHistory:
    """Chronologically sorted behavior arrays for one user."""

    event_time: np.ndarray
    item_index: np.ndarray
    category_index: np.ndarray
    behavior_index: np.ndarray


@dataclass(frozen=True)
class BehaviorHistory:
    """In-memory user behavior history store.

    Attributes:
        histories: Mapping from user_id to sorted behavior arrays.
        item_vocab_size: Number of item embedding rows including padding.
        category_vocab_size: Number of category embedding rows including padding.
        behavior_vocab_size: Number of behavior embedding rows including padding.
    """

    histories: dict[int, UserHistory]
    item_vocab_size: int
    category_vocab_size: int
    behavior_vocab_size: int


@dataclass(frozen=True)
class SequenceSampleFrame:
    """Dense arrays used by SequenceDataset."""

    user_id: np.ndarray
    target_item_index: np.ndarray
    target_category_index: np.ndarray
    sample_time: np.ndarray
    label: np.ndarray


def datetime_to_seconds(series: pd.Series) -> np.ndarray:
    """Convert pandas datetime values to Unix seconds.

    Args:
        series: Datetime Series.

    Returns:
        int64 numpy array containing Unix seconds.
    """
    datetime_values = pd.to_datetime(series).to_numpy(dtype="datetime64[s]")
    return datetime_values.astype("int64")


def hash_item_ids(item_ids: pd.Series | np.ndarray, item_hash_size: int) -> np.ndarray:
    """Map raw item IDs into compact embedding buckets.

    Args:
        item_ids: Raw item ID values.
        item_hash_size: Number of hash buckets excluding padding.

    Returns:
        int64 item bucket indices in [1, item_hash_size].
    """
    values = np.asarray(item_ids, dtype=np.int64)
    return (np.mod(values, item_hash_size) + 1).astype(np.int64)


def category_to_index(categories: pd.Series | np.ndarray) -> np.ndarray:
    """Map raw category IDs to embedding indices with zero as padding."""
    values = np.asarray(categories, dtype=np.float64)
    values = np.nan_to_num(values, nan=-1.0)
    return np.where(values >= 0, values.astype(np.int64) + 1, PADDING_INDEX)


def load_behavior_history(
    behavior_path: Path,
    item_hash_size: int,
) -> BehaviorHistory:
    """Load cleaned behavior logs and build per-user history arrays.

    Args:
        behavior_path: Path to cleaned_data.parquet.
        item_hash_size: Number of item hash buckets excluding padding.

    Returns:
        BehaviorHistory used by SequenceDataset.
    """
    columns = [
        "time",
        "user_id",
        "item_id",
        "item_category",
        "behavior_type",
    ]
    events = pd.read_parquet(behavior_path, columns=columns)
    events = events.sort_values(["user_id", "time"], kind="mergesort")
    events["event_time"] = datetime_to_seconds(events["time"])
    events["item_index"] = hash_item_ids(events["item_id"], item_hash_size)
    events["category_index"] = category_to_index(events["item_category"])
    events["behavior_index"] = events["behavior_type"].astype(np.int64)

    histories: dict[int, UserHistory] = {}
    for user_id, group in events.groupby("user_id", sort=False):
        histories[int(user_id)] = UserHistory(
            event_time=group["event_time"].to_numpy(dtype=np.int64, copy=True),
            item_index=group["item_index"].to_numpy(dtype=np.int64, copy=True),
            category_index=group["category_index"].to_numpy(dtype=np.int64, copy=True),
            behavior_index=group["behavior_index"].to_numpy(dtype=np.int64, copy=True),
        )

    max_category = int(events["category_index"].max())
    max_behavior = int(events["behavior_index"].max())
    return BehaviorHistory(
        histories=histories,
        item_vocab_size=item_hash_size + 1,
        category_vocab_size=max_category + 1,
        behavior_vocab_size=max_behavior + 1,
    )


def load_sequence_samples(
    sample_path: Path,
    item_dim_path: Path,
    item_hash_size: int,
    max_rows: int | None = None,
) -> SequenceSampleFrame:
    """Load sample table and attach target item/category indices.

    Args:
        sample_path: train.parquet, val.parquet, or test.parquet path.
        item_dim_path: Item dimension table containing item_category.
        item_hash_size: Number of item hash buckets excluding padding.
        max_rows: Optional row cap for quick experiments.

    Returns:
        SequenceSampleFrame with dense numpy arrays.
    """
    sample_columns = ["user_id", "item_id", "last_time", "label"]
    samples = pd.read_parquet(sample_path, columns=sample_columns)
    if max_rows is not None:
        samples = samples.head(max_rows).copy()

    item_dim = pd.read_parquet(item_dim_path, columns=["item_category"])
    samples = samples.join(item_dim, on="item_id")

    return SequenceSampleFrame(
        user_id=samples["user_id"].to_numpy(dtype=np.int64, copy=True),
        target_item_index=hash_item_ids(samples["item_id"], item_hash_size),
        target_category_index=category_to_index(samples["item_category"]),
        sample_time=datetime_to_seconds(samples["last_time"]),
        label=samples["label"].to_numpy(dtype=np.float32, copy=True),
    )


class SequenceDataset(Dataset):
    """PyTorch Dataset that builds time-safe user histories per sample."""

    def __init__(
        self,
        samples: SequenceSampleFrame,
        behavior_history: BehaviorHistory,
        max_seq_len: int,
    ) -> None:
        """Initialize dataset.

        Args:
            samples: Dense sample arrays.
            behavior_history: Per-user historical behavior arrays.
            max_seq_len: Maximum historical events retained per sample.
        """
        self.samples = samples
        self.behavior_history = behavior_history
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.samples.label)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Build one padded sequence sample."""
        sample_time = int(self.samples.sample_time[index])
        user_id = int(self.samples.user_id[index])
        history = self.behavior_history.histories.get(user_id)

        item_sequence = np.zeros(self.max_seq_len, dtype=np.int64)
        category_sequence = np.zeros(self.max_seq_len, dtype=np.int64)
        behavior_sequence = np.zeros(self.max_seq_len, dtype=np.int64)
        time_gap_sequence = np.zeros(self.max_seq_len, dtype=np.float32)
        sequence_length = 0

        if history is not None:
            end = np.searchsorted(history.event_time, sample_time, side="left")
            start = max(0, end - self.max_seq_len)
            sequence_length = end - start
            if sequence_length > 0:
                item_sequence[:sequence_length] = history.item_index[start:end]
                category_sequence[:sequence_length] = history.category_index[start:end]
                behavior_sequence[:sequence_length] = history.behavior_index[start:end]
                gaps = (sample_time - history.event_time[start:end]) / SECONDS_PER_DAY
                gaps = np.clip(gaps, 0.0, MAX_TIME_GAP_DAYS) / MAX_TIME_GAP_DAYS
                time_gap_sequence[:sequence_length] = gaps.astype(np.float32)

        return {
            "hist_item": torch.from_numpy(item_sequence),
            "hist_category": torch.from_numpy(category_sequence),
            "hist_behavior": torch.from_numpy(behavior_sequence),
            "hist_time_gap": torch.from_numpy(time_gap_sequence),
            "seq_len": torch.tensor(sequence_length, dtype=torch.long),
            "target_item": torch.tensor(
                self.samples.target_item_index[index],
                dtype=torch.long,
            ),
            "target_category": torch.tensor(
                self.samples.target_category_index[index],
                dtype=torch.long,
            ),
            "label": torch.tensor(self.samples.label[index], dtype=torch.float32),
        }

"""Shared data-splitting helpers for traditional ML experiments."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE


TARGET_COL = "label"
RANDOM_STATE = 42
SPECIAL_DATES = {pd.Timestamp("2025-12-12").date()}
EXCLUDE_COLS = {
    TARGET_COL,
    "user_id",
    "item_id",
    "last_time",
    "buy_path_type",
    "behavior_type",
    "item_category",
}


def build_time_strata(frame: pd.DataFrame) -> pd.Series:
    """Build date/day-type/hour-bin strata matching the project split design."""
    dt = pd.to_datetime(frame["last_time"])
    day_type = np.select(
        [
            dt.dt.date.isin(SPECIAL_DATES),
            dt.dt.dayofweek >= 5,
        ],
        ["special", "weekend"],
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


def proportional_allocations(counts: pd.Series, target: int) -> pd.Series:
    """Allocate an exact row count in proportion to group sizes."""
    if target < 0 or target > int(counts.sum()):
        raise ValueError("Allocation target is outside available rows.")
    raw = counts.astype(float) * (target / float(counts.sum()))
    allocations = np.floor(raw).astype("int64")
    remaining = target - int(allocations.sum())
    order = (raw - allocations).sort_values(ascending=False).index
    for key in order:
        if remaining == 0:
            break
        if allocations.loc[key] < counts.loc[key]:
            allocations.loc[key] += 1
            remaining -= 1
    if remaining:
        raise RuntimeError("Could not allocate requested sample size.")
    return allocations


def sample_by_label_time_strata(
    frame: pd.DataFrame,
    sample_size: int,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Sample while preserving label and time-stratum ratios."""
    sample_size = min(sample_size, len(frame))
    if sample_size <= 0:
        raise ValueError("sample_size must be positive.")

    work = frame.copy()
    work["_time_stratum"] = build_time_strata(work)
    if sample_size == len(work):
        return work.reset_index(drop=True)

    label_alloc = proportional_allocations(
        work[TARGET_COL].value_counts().sort_index(),
        sample_size,
    )
    parts = []
    for label, label_target in label_alloc.items():
        label_rows = work[work[TARGET_COL] == label]
        stratum_counts = label_rows["_time_stratum"].value_counts().sort_index()
        stratum_alloc = proportional_allocations(stratum_counts, int(label_target))
        for stratum, n_rows in stratum_alloc.items():
            if n_rows <= 0:
                continue
            rows = label_rows[label_rows["_time_stratum"] == stratum]
            parts.append(rows.sample(int(n_rows), random_state=random_state))
    return (
        pd.concat(parts)
        .sample(frac=1.0, random_state=random_state)
        .reset_index(drop=True)
    )


def split_within_time_strata(
    frame: pd.DataFrame,
    val_frac: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronologically split each time stratum into tune-train/tune-validation."""
    if not 0 < val_frac < 1:
        raise ValueError("val_frac must be in (0, 1).")

    work = frame.copy()
    work["last_time"] = pd.to_datetime(work["last_time"])
    if "_time_stratum" not in work.columns:
        work["_time_stratum"] = build_time_strata(work)

    train_parts = []
    val_parts = []
    for _, group in work.groupby("_time_stratum", sort=True):
        group = group.sort_values("last_time", kind="mergesort")
        if len(group) < 2:
            train_parts.append(group)
            continue
        split_index = int(round(len(group) * (1.0 - val_frac)))
        split_index = max(1, min(split_index, len(group) - 1))
        train_parts.append(group.iloc[:split_index])
        val_parts.append(group.iloc[split_index:])

    train = pd.concat(train_parts, ignore_index=True)
    val = pd.concat(val_parts, ignore_index=True)
    return (
        train.drop(columns=["_time_stratum"]).reset_index(drop=True),
        val.drop(columns=["_time_stratum"]).reset_index(drop=True),
    )


def numeric_feature_columns(*frames: pd.DataFrame) -> list[str]:
    """Return common numeric feature columns shared by all frames."""
    common = set(frames[0].columns)
    for frame in frames[1:]:
        common &= set(frame.columns)
    feature_cols = []
    for col in frames[0].columns:
        if col not in common or col in EXCLUDE_COLS:
            continue
        if all(pd.api.types.is_numeric_dtype(frame[col]) for frame in frames):
            feature_cols.append(col)
    if not feature_cols:
        raise ValueError("No common numeric feature columns found.")
    return feature_cols


def to_xy(
    frame: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a frame into float32 feature matrix and int8 labels."""
    X = frame[feature_cols].to_numpy(dtype=np.float32, copy=True)
    y = frame[TARGET_COL].to_numpy(dtype=np.int8, copy=True)
    if np.isnan(X).any():
        raise ValueError("Missing values found in feature matrix.")
    return X, y


def smote_to_ratio(
    frame: pd.DataFrame,
    feature_cols: list[str],
    target_pos_ratio: float,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    """Apply SMOTE to one training frame and return arrays plus summary info."""
    if not 0 < target_pos_ratio < 1:
        raise ValueError("target_pos_ratio must be in (0, 1).")
    X, y = to_xy(frame, feature_cols)
    pos_count = int(y.sum())
    neg_count = len(y) - pos_count
    before_ratio = pos_count / len(y)
    if pos_count < 2 or target_pos_ratio <= before_ratio:
        return X, y, {
            "before_rows": len(y),
            "after_rows": len(y),
            "before_positive_count": pos_count,
            "after_positive_count": pos_count,
            "before_positive_ratio": before_ratio,
            "after_positive_ratio": before_ratio,
        }

    smote = SMOTE(
        random_state=random_state,
        sampling_strategy=target_pos_ratio / (1.0 - target_pos_ratio),
        k_neighbors=min(5, pos_count - 1),
    )
    X_res, y_res = smote.fit_resample(X, y)
    after_pos = int(y_res.sum())
    return X_res, y_res.astype(np.int8), {
        "before_rows": len(y),
        "after_rows": len(y_res),
        "before_positive_count": pos_count,
        "after_positive_count": after_pos,
        "before_positive_ratio": before_ratio,
        "after_positive_ratio": after_pos / len(y_res),
        "negative_count": neg_count,
    }


def load_parquet(path: Path) -> pd.DataFrame:
    """Small wrapper to keep call sites concise."""
    return pd.read_parquet(path)

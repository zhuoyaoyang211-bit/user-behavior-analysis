"""样本构建主入口 — 时间特征分层窗口版。

流程：
    1. 合并已筛选核心特征 + 标签表 → 全量样本
    2. 按 last_time 时间特征分层后，层内按时间窗口划分 train/val/test

划分方案（只使用时间维度，不按 label 随机保分布）：
    1. 按 日期 × 工作日/周末/特殊日 × 小时段 构造时间小层
    2. 每个时间小层内按 last_time 排序
    3. 层内按 7:2:1 连续切分 train/val/test

说明：
    selected_features.parquet 已是核心特征数据集，本脚本不再重复填缺失、
    目标编码、标准化或特征筛选，只负责合并标签、按时间维度划分和保存样本。

输入：
    output/selected_features.parquet（4,686,904 行 × 28 列）
    output/label_table.parquet（4,683,196 行 × 4 列）

输出：
    output/train.parquet
    output/val.parquet
    output/test.parquet
"""

from __future__ import annotations

import gc
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common.logger import get_logger
from config import get_config

logger = get_logger(__name__)

_cfg = get_config()
OUTPUT_DIR = str(_cfg.project_root / "output")
SELECTED_FEATURES_PATH = os.path.join(OUTPUT_DIR, "selected_features.parquet")
LABEL_TABLE_PATH = os.path.join(OUTPUT_DIR, "label_table.parquet")

# ── 时间特征分层窗口划分参数 ───────────────────────────────────
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1
assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-9

# 可按业务补充，例如电商大促日。这里把 12.12 单独作为特殊日分层。
SPECIAL_DATES = {pd.Timestamp("2025-12-12").date()}

# ===========================================================================
# 步骤 1：合并已筛选核心特征 + 标签表
# ===========================================================================
def merge_features_and_labels() -> pd.DataFrame:
    """加载已筛选核心特征和标签表，inner join 合并。

    Returns:
        合并后的 DataFrame，含特征 + label + last_time。
    """
    logger.info("加载已筛选核心特征 ...")
    features = pd.read_parquet(SELECTED_FEATURES_PATH)
    logger.info("  核心特征: %s 行 × %s 列", f"{len(features):,}", len(features.columns))

    if "buy_path_type" in features.columns:
        features = features.drop(columns=["buy_path_type"])
        logger.info("  删除旧目标列: buy_path_type")

    logger.info("加载标签表 ...")
    labels = pd.read_parquet(LABEL_TABLE_PATH)
    logger.info("  标签表: %s 行 × %s 列", f"{len(labels):,}", len(labels.columns))
    logger.info("  正样本: %s (%.2f%%)",
                f"{labels['label'].sum():,}",
                labels["label"].mean() * 100)

    logger.info("合并核心特征 + 标签表 (inner join) ...")
    df = features.merge(labels, on=["user_id", "item_id"], how="inner")
    logger.info("  合并后: %s 行 × %s 列", f"{len(df):,}", len(df.columns))

    # 释放原始大表
    del features, labels
    gc.collect()

    return df


# ===========================================================================
# 步骤 2：时间特征分层窗口划分
# ===========================================================================
def ensure_last_time_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """确保 last_time 是 datetime 类型。"""
    if not pd.api.types.is_datetime64_any_dtype(df["last_time"]):
        df = df.copy()
        df["last_time"] = pd.to_datetime(df["last_time"])
    return df


def build_time_strata(df: pd.DataFrame) -> pd.Series:
    """构造时间小层：日期 × 工作日/周末/特殊日 × 小时段。"""
    dt = df["last_time"]
    dates = dt.dt.date

    day_type = np.select(
        [
            dates.isin(SPECIAL_DATES),
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
        dates.astype(str)
        + "|"
        + pd.Series(day_type, index=df.index).astype(str)
        + "|"
        + hour_bin.astype(str)
    )


def split_ordered_group(
    group: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """单个时间小层内按 last_time 顺序连续切分。"""
    group = group.sort_values("last_time")
    n = len(group)

    if n == 1:
        return group, group.iloc[0:0], group.iloc[0:0]
    if n == 2:
        return group.iloc[:1], group.iloc[1:], group.iloc[0:0]

    test_n = max(1, int(round(n * TEST_RATIO)))
    val_n = max(1, int(round(n * VAL_RATIO)))
    if val_n + test_n >= n:
        val_n = 1
        test_n = 1

    train_n = n - val_n - test_n
    val_end = train_n + val_n

    return group.iloc[:train_n], group.iloc[train_n:val_end], group.iloc[val_end:]


def split_by_time(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """按时间特征分层后，在每个小层内按时间顺序切分 train/val/test。

    Returns:
        (train_df, val_df, test_df)
    """
    logger.info("开始时间特征分层窗口划分 ...")

    df = ensure_last_time_datetime(df).copy()
    df["_time_stratum"] = build_time_strata(df)

    train_parts: list[pd.DataFrame] = []
    val_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []

    for _, group in df.groupby("_time_stratum", sort=False):
        train_part, val_part, test_part = split_ordered_group(group)
        train_parts.append(train_part)
        val_parts.append(val_part)
        test_parts.append(test_part)

    train = pd.concat(train_parts, ignore_index=True)
    val = pd.concat(val_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True)

    for partial in (train, val, test):
        partial.drop(columns=["_time_stratum"], inplace=True)

    total = len(df)
    logger.info(
        "  train: %s 行 (%5.1f%%)",
        f"{len(train):,}",
        len(train) / total * 100,
    )
    logger.info(
        "  val:   %s 行 (%5.1f%%)",
        f"{len(val):,}",
        len(val) / total * 100,
    )
    logger.info(
        "  test:  %s 行 (%5.1f%%)",
        f"{len(test):,}",
        len(test) / total * 100,
    )

    # 验证无重叠、无遗漏
    assert len(train) + len(val) + len(test) == total, "划分有遗漏！"
    log_split_quality(train, val, test)

    return train, val, test


def log_split_quality(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """打印各集合的时间范围、标签分布和时间类型覆盖。"""
    logger.info("划分质量检查")
    total = len(train) + len(val) + len(test)

    for name, partial in [("train", train), ("val", val), ("test", test)]:
        dt = partial["last_time"]
        dates = dt.dt.date
        day_type = np.select(
            [
                dates.isin(SPECIAL_DATES),
                dt.dt.dayofweek >= 5,
            ],
            [
                "special",
                "weekend",
            ],
            default="weekday",
        )
        day_type_counts = pd.Series(day_type).value_counts().to_dict()
        logger.info(
            "  %-5s rows=%s (%5.1f%%), pos=%.4f%%, weekday_rows=%s, weekend_rows=%s, special_rows=%s, range=%s ~ %s",
            name,
            f"{len(partial):,}",
            len(partial) / total * 100,
            partial["label"].mean() * 100,
            f"{day_type_counts.get('weekday', 0):,}",
            f"{day_type_counts.get('weekend', 0):,}",
            f"{day_type_counts.get('special', 0):,}",
            str(dt.min())[:16],
            str(dt.max())[:16],
        )


# ===========================================================================
# 主入口
# ===========================================================================
def main() -> None:
    """样本构建主流程。"""
    # 1. 合并
    df = merge_features_and_labels()

    # 2. 时间特征分层窗口划分
    train, val, test = split_by_time(df)
    del df
    gc.collect()

    # 3. 保存
    train_path = os.path.join(OUTPUT_DIR, "train.parquet")
    val_path = os.path.join(OUTPUT_DIR, "val.parquet")
    test_path = os.path.join(OUTPUT_DIR, "test.parquet")

    train.to_parquet(train_path, index=False)
    val.to_parquet(val_path, index=False)
    test.to_parquet(test_path, index=False)

    train_mb = os.path.getsize(train_path) / 1024 / 1024
    val_mb = os.path.getsize(val_path) / 1024 / 1024
    test_mb = os.path.getsize(test_path) / 1024 / 1024

    # 4. 打印摘要
    print("\n" + "=" * 65)
    print("样本构建完成（时间特征分层窗口版）")
    print("=" * 65)
    print("\n  划分方案: 日期 × 工作日/周末/特殊日 × 小时段分层，层内按 last_time 顺序 7:2:1 切分")
    print(f"\n  {'':>8} {'行数':>12} {'占比':>8} {'正样本':>8} {'正样本率':>10} {'文件大小':>10}")
    print(f"  {'-' * 56}")
    total = len(train) + len(val) + len(test)
    for name, d, path, mb in [
        ("train", train, train_path, train_mb),
        ("val", val, val_path, val_mb),
        ("test", test, test_path, test_mb),
    ]:
        n_pos = d["label"].sum()
        print(
            f"  {name:>8} {len(d):>12,} {len(d)/total*100:>7.1f}% "
            f"{n_pos:>8,} {n_pos/len(d)*100:>9.2f}% {mb:>9.1f} MB"
        )
    print(f"\n  总特征列数: {len([c for c in train.columns if c not in ('user_id','item_id','label','last_time')])}")
    print(f"  输出目录: {OUTPUT_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    main()

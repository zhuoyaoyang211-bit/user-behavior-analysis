"""样本构建主入口 — 时间窗口版。

流程：
    1. 合并特征宽表 + 标签表 → 全量样本
    2. 预处理：删冗余列 → 填缺失 → 目标编码 → 标准化
    3. 特征筛选：方差阈值 → 互信息 → 相关性分析
    4. 按 last_time 时间窗口划分 train/val/test

划分方案（按 last_time 日期切分）：
    train: ≤ 2025-12-10 (~70%)
    val:   2025-12-11 ~ 2025-12-15 (~20%)
    test:  ≥ 2025-12-16 (~10%)

输入：
    output/feature_wide_table.parquet（4,686,904 行 × 47 列）
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
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common.logger import get_logger
from config import get_config
from feature_selection.selector import (
    _correlation_filter,
    _get_feature_cols,
    _mutual_info_filter,
    _variance_filter,
)

logger = get_logger(__name__)

_cfg = get_config()
OUTPUT_DIR = str(_cfg.project_root / "output")
WIDE_TABLE_PATH = os.path.join(OUTPUT_DIR, "feature_wide_table.parquet")
LABEL_TABLE_PATH = os.path.join(OUTPUT_DIR, "label_table.parquet")

# ── 时间窗口切分点 ─────────────────────────────────────────────
TRAIN_CUTOFF = pd.Timestamp("2025-12-11")   # train: last_time < 12.11 (即 ≤ 12.10)
VAL_CUTOFF = pd.Timestamp("2025-12-16")     # val:   12.11 ≤ last_time < 12.16 (即 12.11-12.15)
                                             # test:  last_time ≥ 12.16

# ── 不参与标准化的列 ───────────────────────────────────────────
COLUMNS_NO_SCALE = {
    "user_id", "item_id", "item_category",
    "is_power_user", "label", "last_time",
}

# ── 缺失值填充策略 ─────────────────────────────────────────────
FILL_STRATEGY: dict[str, str | float] = {
    "item_decay_slope": 0,
    "user_category_pref_score": 0,
    "user_avg_interval_hours": "median",
}

# ── 特征筛选：不参与筛选的列 ──────────────────────────────────
EXCLUDE_COLS = {"user_id", "item_id", "item_category", "label", "last_time"}


# ===========================================================================
# 步骤 1：合并特征宽表 + 标签表
# ===========================================================================
def merge_features_and_labels() -> pd.DataFrame:
    """加载特征宽表和标签表，inner join 合并。

    Returns:
        合并后的 DataFrame，含特征 + label + last_time。
    """
    logger.info("加载特征宽表 ...")
    wide = pd.read_parquet(WIDE_TABLE_PATH)
    logger.info("  特征宽表: %s 行 × %s 列", f"{len(wide):,}", len(wide.columns))

    logger.info("加载标签表 ...")
    labels = pd.read_parquet(LABEL_TABLE_PATH)
    logger.info("  标签表: %s 行 × %s 列", f"{len(labels):,}", len(labels.columns))
    logger.info("  正样本: %s (%.2f%%)",
                f"{labels['label'].sum():,}",
                labels["label"].mean() * 100)

    logger.info("合并特征宽表 + 标签表 (inner join) ...")
    df = wide.merge(labels, on=["user_id", "item_id"], how="inner")
    logger.info("  合并后: %s 行 × %s 列", f"{len(df):,}", len(df.columns))

    # 释放原始大表
    del wide, labels
    gc.collect()

    return df


# ===========================================================================
# 步骤 2：预处理
# ===========================================================================
def drop_redundant(df: pd.DataFrame) -> pd.DataFrame:
    """删除冗余列。

    - buy_path_type：已被新 label 替代
    - first_active_time / last_active_time：datetime 列，信息已被其他特征覆盖
    """
    drop_cols = ["buy_path_type", "first_active_time", "last_active_time"]
    existing = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=existing)
    logger.info("  删除冗余列: %s", existing)
    return df


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """填充缺失值。"""
    for col, strategy in FILL_STRATEGY.items():
        if col not in df.columns:
            continue
        n_miss = df[col].isnull().sum()
        if n_miss == 0:
            continue
        if strategy == "median":
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(strategy)
        logger.info("  填充 %s: %s 个缺失值 → %s", col, f"{n_miss:,}", strategy)
    return df


def target_encode(df: pd.DataFrame) -> pd.DataFrame:
    """目标编码：用新 label 算类目/用户的购买率。

    生成 2 列：
    - item_category_te：该类目下 label=1 的比例
    - user_id_te：该用户 label=1 的比例
    """
    target = df["label"]

    # 类目级别购买率
    cat_rate = (
        pd.DataFrame({"cat": df["item_category"], "y": target})
        .groupby("cat")["y"]
        .mean()
    )
    df["item_category_te"] = df["item_category"].map(cat_rate).astype(np.float32)
    logger.info("  目标编码 item_category → item_category_te (%d 个类目)", len(cat_rate))

    # 用户级别购买率
    user_rate = (
        pd.DataFrame({"uid": df["user_id"], "y": target})
        .groupby("uid")["y"]
        .mean()
    )
    df["user_id_te"] = df["user_id"].map(user_rate).astype(np.float32)
    logger.info("  目标编码 user_id → user_id_te (%d 个用户)", len(user_rate))

    return df


def standardize(df: pd.DataFrame) -> pd.DataFrame:
    """StandardScaler 标准化数值列。

    排除 COLUMNS_NO_SCALE 中的列，其余数值列全部标准化。
    """
    scale_cols = [
        c for c in df.columns
        if c not in COLUMNS_NO_SCALE
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    logger.info("  待标准化: %d 列", len(scale_cols))

    scaler = StandardScaler()
    df[scale_cols] = scaler.fit_transform(df[scale_cols].astype(np.float64))
    # 降回 float32 省内存
    for c in scale_cols:
        df[c] = df[c].astype(np.float32)

    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """预处理主流程：删冗余 → 填缺失 → 目标编码 → 标准化。"""
    logger.info("开始预处理 ...")

    df = drop_redundant(df)
    df = fill_missing(df)
    df = target_encode(df)
    df = standardize(df)

    logger.info("预处理完成: %s 行 × %s 列", f"{len(df):,}", len(df.columns))
    return df


# ===========================================================================
# 步骤 3：特征筛选
# ===========================================================================
def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """三轮特征筛选（方差 → 互信息 → 相关性）。

    Args:
        df: 预处理后的 DataFrame。

    Returns:
        筛选后的 DataFrame。
    """
    logger.info("开始特征筛选 ...")

    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    logger.info("  参与筛选的特征列: %d 列", len(feature_cols))

    # 第 1 轮：方差阈值
    cols_vt, dropped_vt = _variance_filter(df, feature_cols)
    logger.info("  方差筛选: %d → %d 列", len(feature_cols), len(cols_vt))

    # 第 2 轮：互信息（用 label 作为目标）
    cols_mi, dropped_mi, mi_series = _mutual_info_filter(
        df, cols_vt, target_col="label"
    )
    logger.info("  互信息筛选: %d → %d 列", len(cols_vt), len(cols_mi))

    # 第 3 轮：相关性
    cols_final, dropped_corr = _correlation_filter(df, cols_mi, mi_series)
    logger.info("  相关性筛选: %d → %d 列", len(cols_mi), len(cols_final))

    # 构建输出
    output_cols = ["user_id", "item_id"] + cols_final + ["label", "last_time"]
    selected_df = df[output_cols].copy()

    logger.info(
        "特征筛选完成: %d → %d 列",
        len(feature_cols),
        len(cols_final),
    )

    # 打印 Top 10 互信息
    print("\n  [互信息 Top 10（新 label）]")
    for i, (col, score) in enumerate(mi_series.head(10).items(), 1):
        marker = "*" if col in cols_final else "x"
        print(f"    {i:2d}. [{marker}] {col:<35s} MI={score:.6f}")

    return selected_df


# ===========================================================================
# 步骤 4：时间窗口划分
# ===========================================================================
def split_by_time(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """按 last_time 时间窗口划分 train/val/test。

    Returns:
        (train_df, val_df, test_df)
    """
    logger.info("开始时间窗口划分 ...")

    # 确保 last_time 是 datetime
    if not pd.api.types.is_datetime64_any_dtype(df["last_time"]):
        df["last_time"] = pd.to_datetime(df["last_time"])

    train = df[df["last_time"] < TRAIN_CUTOFF].copy()
    val = df[
        (df["last_time"] >= TRAIN_CUTOFF) & (df["last_time"] < VAL_CUTOFF)
    ].copy()
    test = df[df["last_time"] >= VAL_CUTOFF].copy()

    total = len(df)
    logger.info(
        "  train: %s 行 (%5.1f%%)  last_time ≤ 12.10",
        f"{len(train):,}",
        len(train) / total * 100,
    )
    logger.info(
        "  val:   %s 行 (%5.1f%%)  last_time 12.11-12.15",
        f"{len(val):,}",
        len(val) / total * 100,
    )
    logger.info(
        "  test:  %s 行 (%5.1f%%)  last_time ≥ 12.16",
        f"{len(test):,}",
        len(test) / total * 100,
    )

    # 验证无重叠、无遗漏
    assert len(train) + len(val) + len(test) == total, "划分有遗漏！"

    return train, val, test


# ===========================================================================
# 主入口
# ===========================================================================
def main() -> None:
    """样本构建主流程。"""
    # 1. 合并
    df = merge_features_and_labels()

    # 2. 预处理
    df = preprocess(df)
    gc.collect()

    # 3. 特征筛选
    df = select_features(df)
    gc.collect()

    # 4. 时间窗口划分
    train, val, test = split_by_time(df)

    # 5. 保存
    train_path = os.path.join(OUTPUT_DIR, "train.parquet")
    val_path = os.path.join(OUTPUT_DIR, "val.parquet")
    test_path = os.path.join(OUTPUT_DIR, "test.parquet")

    train.to_parquet(train_path, index=False)
    val.to_parquet(val_path, index=False)
    test.to_parquet(test_path, index=False)

    train_mb = os.path.getsize(train_path) / 1024 / 1024
    val_mb = os.path.getsize(val_path) / 1024 / 1024
    test_mb = os.path.getsize(test_path) / 1024 / 1024

    # 6. 打印摘要
    print("\n" + "=" * 65)
    print("样本构建完成（时间窗口版）")
    print("=" * 65)
    print(f"\n  划分方案: last_time ≤ 12.10 | 12.11-12.15 | ≥ 12.16")
    print(f"\n  {'':>8} {'行数':>12} {'占比':>8} {'正样本':>8} {'正样本率':>10} {'文件大小':>10}")
    print(f"  {'-' * 56}")
    for name, d, path, mb in [
        ("train", train, train_path, train_mb),
        ("val", val, val_path, val_mb),
        ("test", test, test_path, test_mb),
    ]:
        n_pos = d["label"].sum()
        print(
            f"  {name:>8} {len(d):>12,} {len(d)/len(df)*100:>7.1f}% "
            f"{n_pos:>8,} {n_pos/len(d)*100:>9.2f}% {mb:>9.1f} MB"
        )
    print(f"\n  总特征列数: {len([c for c in train.columns if c not in ('user_id','item_id','label','last_time')])}")
    print(f"  输出目录: {OUTPUT_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    main()

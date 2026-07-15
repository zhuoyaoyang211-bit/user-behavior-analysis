"""序列构建模块。

从 cleaned_data.parquet 重建每个 (user_id, item_id) 对的 31 天行为序列，
输出 (N, 31, 7) 的 numpy 数组供 LSTM/GRU 训练使用。

数据流：
    cleaned_data.parquet (12M行, 有时间维度)
    → 按 (user_id, item_id, date) 日聚合
    → 按 train/val/test 中的 (user_id, item_id) 对齐
    → 构建 31 天序列，缺失天补零
    → 附加时间标记特征（weekday, is_weekend, day_index）
    → 输出 .npy 文件
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from common.logger import get_logger

logger = get_logger(__name__)

# ---- 常量 ----
# 数据集时间范围：2025-11-18 ~ 2025-12-18，共 31 天
START_DATE = pd.Timestamp("2025-11-18")
END_DATE = pd.Timestamp("2025-12-18")
ALL_DATES = pd.date_range(START_DATE, END_DATE, freq="D")  # 31 天
N_DAYS = len(ALL_DATES)  # 31

# 行为类型映射：cleaned_data 中 behavior_type 的取值范围
BEHAVIOR_MAP = {1: "pv", 2: "fav", 3: "cart", 4: "buy"}
# 这 4 个行为特征在序列中的位置（第 0-3 位）
BEHAVIOR_FEATURES = ["pv", "fav", "cart", "buy"]
# 时间标记特征在序列中的位置（第 4-6 位）
TIME_FEATURES = ["weekday", "is_weekend", "day_index"]
# 总共 7 个特征
N_FEATURES = len(BEHAVIOR_FEATURES) + len(TIME_FEATURES)  # 7


def _build_daily_aggregation(
    cleaned_data_path: str | Path,
) -> pd.DataFrame:
    """对 cleaned_data 按 (user_id, item_id, date) 日聚合。

    将原始 12M 行数据转换为每日行为计数的宽表，大幅减少数据量。

    Args:
        cleaned_data_path: cleaned_data.parquet 路径

    Returns:
        DataFrame，每行代表一个 (user_id, item_id, date) 组合，
        包含 pv/fav/cart/buy 四列行为计数。
        按 (user_id, item_id) 排序以加速后续 merge。
    """
    logger.info("加载 cleaned_data.parquet ...")
    df = pd.read_parquet(
        cleaned_data_path,
        columns=["time", "user_id", "item_id", "behavior_type"],
    )
    logger.info("原始数据: %s 行", f"{len(df):,}")

    # 提取日期列
    df["date"] = df["time"].dt.date
    df["date"] = pd.to_datetime(df["date"])

    # One-hot 编码行为类型，然后聚合
    for bt_code, bt_name in BEHAVIOR_MAP.items():
        df[bt_name] = (df["behavior_type"] == bt_code).astype(np.uint8)

    logger.info("按 (user_id, item_id, date) 聚合 ...")
    daily = (
        df.groupby(["user_id", "item_id", "date"], observed=True)[
            BEHAVIOR_FEATURES
        ]
        .sum()
        .reset_index()
    )

    # 按 (user_id, item_id) 排序以加速后续操作
    daily = daily.sort_values(["user_id", "item_id", "date"]).reset_index(
        drop=True
    )

    del df  # 释放内存
    logger.info("日聚合完成: %s 条 (user,item,date) 记录", f"{len(daily):,}")
    return daily


def _build_time_features(dates: pd.DatetimeIndex) -> np.ndarray:
    """为给定日期序列生成时间标记特征。

    Args:
        dates: 长度为 N_DAYS 的日期序列

    Returns:
        shape (N_DAYS, 3) 的 numpy 数组：
        [:, 0] = weekday（0=周一 ~ 6=周日）
        [:, 1] = is_weekend（0/1）
        [:, 2] = day_index（0 ~ N_DAYS-1，归一化到 [0, 1]）
    """
    weekday = dates.weekday.values.astype(np.float32)  # 0=周一
    is_weekend = (weekday >= 5).astype(np.float32)  # 周六日=1
    day_index = np.arange(N_DAYS, dtype=np.float32) / (N_DAYS - 1)  # 归一化
    return np.stack([weekday, is_weekend, day_index], axis=1)  # (31, 3)


def _merge_sequences(
    daily: pd.DataFrame,
    pairs_df: pd.DataFrame,
    label_col: str,
    set_name: str,
    output_path: str,
) -> Tuple[str, np.ndarray]:
    """将日聚合数据与目标 (user_id, item_id) 对合并，构建序列。

    策略（避免 O(N) dict 内存开销）：
    1. 给 pairs_df 加 _idx（行号 0~N-1）
    2. 用 merge 把 daily 的日记录匹配到对应行号
    3. 用 numpy memmap 直接在硬盘上创建序列文件，避免内存爆炸

    对每个 (user_id, item_id) 对：
    缺失的天自动为零（初始化的 np.memmap 已处理），
    没有历史记录的 pair 得到全零序列，不影响训练。

    Args:
        daily: 日聚合 DataFrame，列: user_id, item_id, date, pv, fav, cart, buy
        pairs_df: 目标 (user_id, item_id, label) 对
        label_col: 标签列名
        set_name: 数据集名称（用于日志）
        output_path: memmap 文件输出路径

    Returns:
        (output_path, labels):
        - output_path: memmap 文件路径（序列已写入硬盘）
        - labels: (N,) numpy int64
    """
    logger.info("构建 %s 序列 ...", set_name)
    n_pairs = len(pairs_df)

    # 预计算时间特征（31天固定，所有序列共享）
    time_feat = _build_time_features(ALL_DATES)  # (31, 3)

    # ---- 1. 给 pairs_df 加行号索引 ----
    pairs_with_idx = pairs_df.copy()
    pairs_with_idx["_idx"] = np.arange(n_pairs, dtype=np.int32)

    # ---- 2. 将行号索引 + 日记录合并 ----
    # 只取 daily 中在目标 pair 集合里的记录（inner join 过滤无关数据）
    merge_cols = ["user_id", "item_id"]
    merged = pairs_with_idx[merge_cols + ["_idx"]].merge(
        daily,
        on=merge_cols,
        how="inner",  # inner: 只保留有历史记录的 pair
    )

    # 过滤日期范围（理论上 daily 已在范围内，但加上安全性更高）
    merged = merged[
        (merged["date"] >= START_DATE) & (merged["date"] <= END_DATE)
    ]

    if len(merged) == 0:
        logger.warning("%s: 无任何历史记录，返回全零序列", set_name)
        sequences = np.memmap(
            output_path,
            dtype=np.float32,
            mode="w+",
            shape=(n_pairs, N_DAYS, N_FEATURES),
        )
        sequences[:, :, 4:7] = time_feat
        sequences.flush()
        labels = pairs_df[label_col].values.astype(np.int64)
        return output_path, labels

    # 计算每行对应的 day_index（0~30）
    merged["day_index"] = (merged["date"] - START_DATE).dt.days

    # 行为列填充 NaN 为 0
    for feat in BEHAVIOR_FEATURES:
        merged[feat] = merged[feat].fillna(0).astype(np.float32)

    # ---- 3. 用 memmap 直接在硬盘上创建序列文件 ----
    sequences = np.memmap(
        output_path,
        dtype=np.float32,
        mode="w+",
        shape=(n_pairs, N_DAYS, N_FEATURES),
    )

    # 填充时间特征（所有序列相同）
    sequences[:, :, 4:7] = time_feat  # (N, 31, 3)

    # 取出索引数组
    row_indices = merged["_idx"].values.astype(int)  # 每个记录对应的序列行
    day_indices = merged["day_index"].values.astype(int)  # 每个记录对应的天

    # 逐行为特征赋值
    for feat_i, feat in enumerate(BEHAVIOR_FEATURES):
        sequences[row_indices, day_indices, feat_i] = merged[feat].values

    # 确保数据完全写入硬盘
    sequences.flush()

    # 统计有历史记录的 pair 数
    filled_count = len(np.unique(row_indices))

    labels = pairs_df[label_col].values.astype(np.int64)

    logger.info(
        "%s 序列构建完成: %s 对, 有历史=%s, 无历史=%s",
        set_name,
        f"{n_pairs:,}",
        f"{filled_count:,}",
        f"{n_pairs - filled_count:,}",
    )

    return output_path, labels


def build_all_sequences(
    cleaned_data_path: str | Path,
    train_path: str | Path,
    val_path: str | Path,
    test_path: str | Path,
    output_dir: str | Path,
) -> Dict[str, str]:
    """构建训练/验证/测试三部曲序列数据。

    完整流程：
    1. 从 cleaned_data 构建日聚合表
    2. 对 train/val/test 分别构建序列
    3. 保存为 .npy 文件

    Args:
        cleaned_data_path: cleaned_data.parquet 路径
        train_path: train.parquet 路径（需含 user_id, item_id, label）
        val_path: val.parquet 路径
        test_path: test.parquet 路径
        output_dir: 输出目录，保存 .npy 文件

    Returns:
        {
            "train_seq": train_sequences.npy 路径,
            "train_label": train_labels.npy 路径,
            "val_seq": val_sequences.npy 路径,
            "val_label": val_labels.npy 路径,
            "test_seq": test_sequences.npy 路径,
            "test_label": test_labels.npy 路径,
        }
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. 日聚合 ----
    daily = _build_daily_aggregation(cleaned_data_path)

    # ---- 2. 加载 train/val/test ----
    results = {}
    for set_name, parquet_path in [
        ("train", train_path),
        ("val", val_path),
        ("test", test_path),
    ]:
        logger.info("加载 %s: %s", set_name, parquet_path)
        df = pd.read_parquet(parquet_path)

        # 校验必要列存在
        required_cols = ["user_id", "item_id", "label"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"{set_name}.parquet 缺少必要列: {missing}。"
                f"请先运行 build_samples 生成带 item_id 的样本集。"
            )

        id_label_cols = ["user_id", "item_id", "label"]
        pairs_df = df[id_label_cols].copy()
        logger.info("%s 样本数: %s", set_name, f"{len(pairs_df):,}")

        seq_path = str(output_dir / f"{set_name}_sequences.npy")
        lbl_path = str(output_dir / f"{set_name}_labels.npy")

        seq_path, lbl = _merge_sequences(
            daily, pairs_df, "label", set_name, seq_path
        )

        np.save(lbl_path, lbl)
        logger.info(
            "已保存序列: %s (%.1f MB)",
            seq_path,
            os.path.getsize(seq_path) / 1024**2,
        )
        logger.info("已保存标签: %s", lbl_path)

        results[f"{set_name}_seq"] = seq_path
        results[f"{set_name}_label"] = lbl_path

    # ---- 3. 汇总（用 memmap 读取 shape，不加载到内存） ----
    train_seq = np.load(results["train_seq"], mmap_mode="r")
    val_seq = np.load(results["val_seq"], mmap_mode="r")
    test_seq = np.load(results["test_seq"], mmap_mode="r")

    logger.info("=" * 50)
    logger.info("序列构建汇总:")
    logger.info("  训练集: %s 条, shape=%s", f"{train_seq.shape[0]:,}", train_seq.shape)
    logger.info("  验证集: %s 条, shape=%s", f"{val_seq.shape[0]:,}", val_seq.shape)
    logger.info("  测试集: %s 条, shape=%s", f"{test_seq.shape[0]:,}", test_seq.shape)
    logger.info("  序列长度: %d 天, %d 特征", N_DAYS, N_FEATURES)
    logger.info(
        "  特征含义: [pv, fav, cart, buy, weekday, is_weekend, day_index]"
    )
    logger.info("=" * 50)

    return results

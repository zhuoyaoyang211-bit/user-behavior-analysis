"""样本构建模块。

加载 selected_features.parquet，将 buy_path_type 二值化为 0/1 标签，
按 7:2:1 时间窗口划分训练集、验证集、测试集。

划分方式：
    从 cleaned_data.parquet 取每个 (user_id, item_id) 对的最后交互时间，
    按时间升序排序后，前 70% 为训练集、中间 20% 为验证集、最后 10% 为测试集。
    保证训练集的时间段严格早于验证集，验证集严格早于测试集，模拟真实场景。
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from common.logger import get_logger

logger = get_logger(__name__)

# ---- 常量 ----
TARGET_COL = "buy_path_type"
LABEL_COL = "label"
ID_COLS = ["user_id", "item_id"]

# 时间窗口切分比例
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2   # 实际从 70% 位置开始取 20%
TEST_RATIO = 0.1  # 实际从 90% 位置开始取 10%


def build_samples(
    input_path: str,
    output_dir: str,
    cleaned_data_path: str | None = None,
) -> dict:
    """加载特征数据集，构建二分类标签，按 7:2:1 时间窗口划分。

    划分策略：
        1. 从 cleaned_data.parquet 取每个 (user_id, item_id) 对的最后交互时间
        2. 将特征宽表按 last_time 升序排序
        3. 前 70% → 训练集，中间 20% → 验证集，最后 10% → 测试集

    Args:
        input_path: selected_features.parquet 路径
        output_dir: 输出目录，train/val/test 三个 parquet 文件保存于此
        cleaned_data_path: cleaned_data.parquet 路径（用于取 last_time）。
            若不传，默认从 input_path 上一级的 output/ 目录找。

    Returns:
        summary: {
            "total": 全量样本数,
            "train_shape": (行, 列),
            "val_shape": (行, 列),
            "test_shape": (行, 列),
            "train_pos_ratio": 训练集正样本占比,
            "val_pos_ratio": 验证集正样本占比,
            "test_pos_ratio": 测试集正样本占比,
            "feature_cols": 特征列名列表,
            "train_path": 训练集路径,
            "val_path": 验证集路径,
            "test_path": 测试集路径,
            "df_train": 训练集 DataFrame,
            "df_val": 验证集 DataFrame,
            "df_test": 测试集 DataFrame,
        }
    """
    # ---- 1. 加载特征表 ----
    logger.info("加载特征数据集: %s", input_path)
    df = pd.read_parquet(input_path)
    logger.info("输入: %s 行 × %s 列", f"{len(df):,}", len(df.columns))

    # 1.5 从 feature_wide_table 获取 item_id
    fw_path = os.path.join(os.path.dirname(input_path), "feature_wide_table.parquet")
    if os.path.exists(fw_path):
        logger.info("从 %s 加载 item_id ...", fw_path)
        fw = pd.read_parquet(fw_path, columns=["item_id"])
        if len(fw) == len(df):
            df["item_id"] = fw["item_id"].values
            logger.info("已追加 item_id 列 (%s 行)", f"{len(df):,}")
        else:
            logger.warning(
                "feature_wide_table 行数 (%s) 与 selected_features (%s) 不一致，跳过匹配",
                f"{len(fw):,}", f"{len(df):,}",
            )
    else:
        logger.warning("未找到 feature_wide_table.parquet，跳过 item_id 匹配")

    # ---- 1.6 从 cleaned_data 取每个 (user,item) 对的最后交互时间 ----
    if cleaned_data_path is None:
        cleaned_data_path = os.path.join(
            os.path.dirname(os.path.dirname(input_path)), "output", "cleaned_data.parquet"
        )

    logger.info("从 %s 计算 last_time ...", cleaned_data_path)
    raw = pd.read_parquet(
        cleaned_data_path,
        columns=["user_id", "item_id", "time"],
    )
    # 每个 (用户, 商品) 对取 last_time
    last_time = (
        raw.groupby(["user_id", "item_id"], as_index=False)["time"]
        .max()
        .rename(columns={"time": "last_time"})
    )
    del raw
    logger.info(
        "计算出 %s 个 (user_id, item_id) 对的 last_time",
        f"{len(last_time):,}",
    )

    # merge 进来——用 left join，保证特征表行数不变
    df = df.merge(last_time, on=["user_id", "item_id"], how="left")
    missing = df["last_time"].isna().sum()
    if missing > 0:
        logger.warning(
            "%s 条样本在 cleaned_data 中找不到时间信息，填充最早时间",
            f"{missing:,}",
        )
        df["last_time"] = df["last_time"].fillna(pd.Timestamp("2025-11-18"))

    # ---- 2. 构建二分类标签 ----
    df[LABEL_COL] = (df[TARGET_COL] != 0).astype(int)

    pos_cnt = df[LABEL_COL].sum()
    neg_cnt = len(df) - pos_cnt
    logger.info(
        "标签分布: 正样本 %s (%.2f%%), 负样本 %s (%.2f%%)",
        f"{pos_cnt:,}",
        pos_cnt / len(df) * 100,
        f"{neg_cnt:,}",
        neg_cnt / len(df) * 100,
    )

    # 确定特征列（排除 ID 列、buy_path_type、label、last_time）
    feature_cols = [
        c for c in df.columns
        if c not in {*ID_COLS, TARGET_COL, LABEL_COL, "last_time"}
    ]
    logger.info("特征列: %d 列", len(feature_cols))

    # ---- 3. 按 last_time 排序后切 7:2:1 ----
    logger.info("按 last_time 升序排序，做时间窗口划分 ...")
    df = df.sort_values("last_time").reset_index(drop=True)
    n = len(df)

    # 时间范围日志
    t_min = df["last_time"].min()
    t_max = df["last_time"].max()
    logger.info("last_time 范围: %s ~ %s", str(t_min)[:16], str(t_max)[:16])

    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    df_train = df.iloc[:train_end].copy()
    df_val = df.iloc[train_end:val_end].copy()
    df_test = df.iloc[val_end:].copy()

    logger.info(
        "训练集: %s 行 (last_time: %s ~ %s)",
        f"{len(df_train):,}",
        str(df_train["last_time"].min())[:16],
        str(df_train["last_time"].max())[:16],
    )
    logger.info(
        "验证集: %s 行 (last_time: %s ~ %s)",
        f"{len(df_val):,}",
        str(df_val["last_time"].min())[:16],
        str(df_val["last_time"].max())[:16],
    )
    logger.info(
        "测试集: %s 行 (last_time: %s ~ %s)",
        f"{len(df_test):,}",
        str(df_test["last_time"].min())[:16],
        str(df_test["last_time"].max())[:16],
    )

    # 切完后扔掉 last_time，保存时不带时间列
    for partial in (df_train, df_val, df_test):
        partial.drop(columns=["last_time"], inplace=True)

    # ---- 4. 保存 ----
    train_path = os.path.join(output_dir, "train.parquet")
    val_path = os.path.join(output_dir, "val.parquet")
    test_path = os.path.join(output_dir, "test.parquet")

    df_train.to_parquet(train_path, index=False)
    logger.info("训练集已保存: %s (%s 行)", train_path, f"{len(df_train):,}")

    df_val.to_parquet(val_path, index=False)
    logger.info("验证集已保存: %s (%s 行)", val_path, f"{len(df_val):,}")

    df_test.to_parquet(test_path, index=False)
    logger.info("测试集已保存: %s (%s 行)", test_path, f"{len(df_test):,}")

    # ---- 5. 汇总 ----
    train_pos_ratio = df_train[LABEL_COL].mean()
    val_pos_ratio = df_val[LABEL_COL].mean()
    test_pos_ratio = df_test[LABEL_COL].mean()

    logger.info(
        "划分完成: train=%.1f%%  val=%.1f%%  test=%.1f%%",
        len(df_train) / n * 100,
        len(df_val) / n * 100,
        len(df_test) / n * 100,
    )
    logger.info(
        "正样本占比: train=%.4f  val=%.4f  test=%.4f",
        train_pos_ratio, val_pos_ratio, test_pos_ratio,
    )

    return {
        "total": n,
        "train_shape": df_train.shape,
        "val_shape": df_val.shape,
        "test_shape": df_test.shape,
        "train_pos_ratio": train_pos_ratio,
        "val_pos_ratio": val_pos_ratio,
        "test_pos_ratio": test_pos_ratio,
        "feature_cols": feature_cols,
        "train_path": train_path,
        "val_path": val_path,
        "test_path": test_path,
        "df_train": df_train,
        "df_val": df_val,
        "df_test": df_test,
    }


def prepare_xy(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """从 DataFrame 中提取特征矩阵 X 和标签向量 y。

    Args:
        df: 包含特征列和 label 列的 DataFrame
        feature_cols: 特征列名列表

    Returns:
        (X, y): 特征矩阵和标签向量
    """
    X = df[feature_cols].values.astype(np.float64)
    y = df[LABEL_COL].values.astype(int)
    return X, y

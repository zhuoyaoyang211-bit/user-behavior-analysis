"""样本构建模块。

加载 selected_features.parquet，将 buy_path_type 二值化为 0/1 标签，
按 7:2:1 分层抽样划分训练集、验证集、测试集。
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from common.logger import get_logger

logger = get_logger(__name__)

# ---- 常量 ----
TARGET_COL = "buy_path_type"
LABEL_COL = "label"
ID_COLS = ["user_id"]

# 随机种子，保证可复现
RANDOM_STATE = 42


def build_samples(
    input_path: str,
    output_dir: str,
) -> dict:
    """加载特征数据集，构建二分类标签，按 7:2:1 分层划分。

    划分策略：
        - 第一刀：全量 → train (70%) + temp (30%)，用 label 分层
        - 第二刀：temp → val (2/3，即全量 20%) + test (1/3，即全量 10%)

    Args:
        input_path: selected_features.parquet 路径
        output_dir: 输出目录，train/val/test 三个 parquet 文件保存于此

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
    # ---- 1. 加载数据 ----
    logger.info("加载特征数据集: %s", input_path)
    df = pd.read_parquet(input_path)
    logger.info("输入: %s 行 × %s 列", f"{len(df):,}", len(df.columns))

    # 1.5 从 feature_wide_table 获取 item_id
    # selected_features 和 feature_wide_table 行数一致、行序一致，直接拼接
    fw_path = os.path.join(os.path.dirname(input_path), "feature_wide_table.parquet")
    if os.path.exists(fw_path):
        logger.info("从 %s 加载 item_id ...", fw_path)
        fw = pd.read_parquet(fw_path, columns=["item_id"])
        if len(fw) == len(df):
            df["item_id"] = fw["item_id"].values
            ID_COLS = ["user_id", "item_id"]
            logger.info("已追加 item_id 列，ID 列: %s", ID_COLS)
        else:
            logger.warning(
                "feature_wide_table 行数 (%s) 与 selected_features (%s) 不一致，跳过匹配",
                f"{len(fw):,}", f"{len(df):,}",
            )
    else:
        logger.warning("未找到 feature_wide_table.parquet，跳过 item_id 匹配")

    # ---- 2. 构建二分类标签 ----
    # buy_path_type: 0=没买, 1/2/3/4=不同购买路径
    # label: 0=没买(负样本), 1=买了(正样本)
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

    # 确定特征列（排除 ID 列、buy_path_type、label）
    feature_cols = [
        c for c in df.columns if c not in {*ID_COLS, TARGET_COL, LABEL_COL}
    ]
    logger.info("特征列: %d 列", len(feature_cols))

    # ---- 3. 7:2:1 分层抽样划分 ----
    logger.info("开始 7:2:1 分层抽样划分 ...")

    # 第一刀：train (70%) vs temp (30%)
    df_train, df_temp = train_test_split(
        df,
        test_size=0.3,
        stratify=df[LABEL_COL],
        random_state=RANDOM_STATE,
    )

    # 第二刀：temp → val (2/3 of temp = 20%) + test (1/3 of temp = 10%)
    df_val, df_test = train_test_split(
        df_temp,
        test_size=1 / 3,
        stratify=df_temp[LABEL_COL],
        random_state=RANDOM_STATE,
    )

    # 丢弃临时变量
    del df_temp

    # ---- 4. 保存 ----
    train_path = os.path.join(output_dir, "train.parquet")
    val_path = os.path.join(output_dir, "val.parquet")
    test_path = os.path.join(output_dir, "test.parquet")

    df_train.to_parquet(train_path, index=False)
    logger.info(
        "训练集已保存: %s (%s 行)", train_path, f"{len(df_train):,}"
    )

    df_val.to_parquet(val_path, index=False)
    logger.info(
        "验证集已保存: %s (%s 行)", val_path, f"{len(df_val):,}"
    )

    df_test.to_parquet(test_path, index=False)
    logger.info(
        "测试集已保存: %s (%s 行)", test_path, f"{len(df_test):,}"
    )

    # ---- 5. 汇总 ----
    train_pos_ratio = df_train[LABEL_COL].mean()
    val_pos_ratio = df_val[LABEL_COL].mean()
    test_pos_ratio = df_test[LABEL_COL].mean()

    logger.info("划分完成: train=%.1f%%  val=%.1f%%  test=%.1f%%",
                len(df_train) / len(df) * 100,
                len(df_val) / len(df) * 100,
                len(df_test) / len(df) * 100)
    logger.info("正样本占比: train=%.4f  val=%.4f  test=%.4f",
                train_pos_ratio, val_pos_ratio, test_pos_ratio)

    return {
        "total": len(df),
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

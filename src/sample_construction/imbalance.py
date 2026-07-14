"""类别不平衡处理模块。

提供三种正负样本不平衡处理方案：
    1. SMOTE 过采样 — 生成合成正样本，使正负样本数量接近
    2. 欠采样 — 随机减少负样本，使正负样本数量接近
    3. 类别权重 — 训练时给正样本更高权重，不改动数据

关键原则：只处理训练集，验证集和测试集保持原始分布。
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

from common.logger import get_logger

logger = get_logger(__name__)

# 随机种子
RANDOM_STATE = 42


def apply_smote(
    df_train: pd.DataFrame,
    feature_cols: list[str],
    output_dir: str,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, dict]:
    """SMOTE 过采样：生成合成正样本，使正负样本 1:1 平衡。

    原理：
        SMOTE 在正样本之间插值，生成新的"假"正样本。
        不会简单复制，而是取两个正样本之间的随机点，让新样本具有多样性。

    Args:
        df_train: 原始训练集 DataFrame，含 label 列
        feature_cols: 特征列名列表
        output_dir: 输出目录
        random_state: 随机种子

    Returns:
        (df_smote, info):
            df_smote: SMOTE 处理后的训练集
            info: {"before_shape", "after_shape", "before_pos_ratio",
                   "after_pos_ratio", "path"}
    """
    from imblearn.over_sampling import SMOTE

    label_col = "label"

    X = df_train[feature_cols].values.astype(np.float64)
    y = df_train[label_col].values.astype(int)

    before_pos = y.sum()
    before_neg = len(y) - before_pos
    logger.info(
        "SMOTE 处理前: 正样本 %s, 负样本 %s (正样本 %.2f%%)",
        f"{before_pos:,}",
        f"{before_neg:,}",
        before_pos / len(y) * 100,
    )

    smote = SMOTE(random_state=random_state)
    X_res, y_res = smote.fit_resample(X, y)

    after_pos = y_res.sum()
    after_neg = len(y_res) - after_pos
    logger.info(
        "SMOTE 处理后: 正样本 %s, 负样本 %s (正样本 %.2f%%)",
        f"{after_pos:,}",
        f"{after_neg:,}",
        after_pos / len(y_res) * 100,
    )

    # 构建新的 DataFrame：只保留特征列 + label，丢掉 user_id 等（SMOTE 生成的行没有真实 user_id）
    df_smote = pd.DataFrame(X_res, columns=feature_cols)
    df_smote[label_col] = y_res.astype(int)

    path = os.path.join(output_dir, "train_smote.parquet")
    df_smote.to_parquet(path, index=False)
    logger.info("SMOTE 训练集已保存: %s (%s 行)", path, f"{len(df_smote):,}")

    info = {
        "before_shape": (len(y), len(feature_cols)),
        "after_shape": df_smote.shape,
        "before_pos_ratio": before_pos / len(y),
        "after_pos_ratio": after_pos / len(y_res),
        "path": path,
    }
    return df_smote, info


def apply_undersample(
    df_train: pd.DataFrame,
    feature_cols: list[str],
    output_dir: str,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, dict]:
    """欠采样：随机减少负样本数量，使正负样本 1:1 平衡。

    原理：
        从大量负样本中随机抽样，保留数量与正样本相同。
        简单直接，但会丢弃大量负样本信息。

    Args:
        df_train: 原始训练集 DataFrame，含 label 列
        feature_cols: 特征列名列表
        output_dir: 输出目录
        random_state: 随机种子

    Returns:
        (df_under, info):
            df_under: 欠采样后的训练集
            info: {"before_shape", "after_shape", "before_pos_ratio",
                   "after_pos_ratio", "path"}
    """
    from imblearn.under_sampling import RandomUnderSampler

    label_col = "label"

    X = df_train[feature_cols].values.astype(np.float64)
    y = df_train[label_col].values.astype(int)

    before_pos = y.sum()
    before_neg = len(y) - before_pos
    logger.info(
        "欠采样处理前: 正样本 %s, 负样本 %s (正样本 %.2f%%)",
        f"{before_pos:,}",
        f"{before_neg:,}",
        before_pos / len(y) * 100,
    )

    undersampler = RandomUnderSampler(random_state=random_state)
    X_res, y_res = undersampler.fit_resample(X, y)

    after_pos = y_res.sum()
    after_neg = len(y_res) - after_pos
    logger.info(
        "欠采样处理后: 正样本 %s, 负样本 %s (正样本 %.2f%%)",
        f"{after_pos:,}",
        f"{after_neg:,}",
        after_pos / len(y_res) * 100,
    )

    # 构建新的 DataFrame
    df_under = pd.DataFrame(X_res, columns=feature_cols)
    df_under[label_col] = y_res.astype(int)

    path = os.path.join(output_dir, "train_undersample.parquet")
    df_under.to_parquet(path, index=False)
    logger.info(
        "欠采样训练集已保存: %s (%s 行)", path, f"{len(df_under):,}"
    )

    info = {
        "before_shape": (len(y), len(feature_cols)),
        "after_shape": df_under.shape,
        "before_pos_ratio": before_pos / len(y),
        "after_pos_ratio": after_pos / len(y_res),
        "path": path,
    }
    return df_under, info


def apply_class_weight(
    df_train: pd.DataFrame,
) -> dict[str, Any]:
    """计算平衡类别权重，不改动训练集数据。

    原理：
        训练模型时，给少数类（正样本）更高的权重，给多数类（负样本）
        更低的权重。模型在计算损失函数时会更重视正样本的错误。

    权重公式（sklearn balanced）：
        weight_class = n_samples / (n_classes × n_samples_per_class)

    Args:
        df_train: 训练集 DataFrame，含 label 列

    Returns:
        info: {
            "class_weights": {0: 负样本权重, 1: 正样本权重},
            "n_pos": 正样本数,
            "n_neg": 负样本数,
        }
    """
    label_col = "label"
    y = df_train[label_col].values.astype(int)

    classes = np.array([0, 1])
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y,
    )

    class_weights = {int(c): float(w) for c, w in zip(classes, weights)}

    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)

    logger.info(
        "类别权重: 负样本(class=0)=%.4f, 正样本(class=1)=%.4f",
        class_weights[0],
        class_weights[1],
    )

    info = {
        "class_weights": class_weights,
        "n_pos": n_pos,
        "n_neg": n_neg,
    }
    return info

"""L1 逻辑回归特征重要性评估模块。

对三轮筛选后的特征集，跑 L1 正则化逻辑回归获取特征系数绝对值排名，
验证前三轮筛选的合理性，输出特征重要性排行榜。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from common.logger import get_logger

logger = get_logger(__name__)

# L1 正则化强度（C 越小，正则化越强，越多系数被压成 0）
L1_C = 0.1


def evaluate_features(df: pd.DataFrame) -> dict:
    """用 L1 逻辑回归评估特征重要性。

    Args:
        df: selected_features.parquet，包含 user_id + 入模特征 + buy_path_type

    Returns:
        summary: {
            "coefs": Series，各特征 L1 系数绝对值（降序）
            "zero_coef_cols": 系数为 0 的特征列表
            "l1_c": 使用的 C 值
            "train_score": 训练集准确率
        }
    """
    logger.info("开始 L1 逻辑回归特征重要性评估")

    # 分离特征和目标
    feature_cols = [c for c in df.columns if c not in {"user_id", "buy_path_type"}]
    y = (df["buy_path_type"] != 0).astype(int)  # 二分类：0=没买，1=买（含所有购买路径）
    X = df[feature_cols].values.astype(np.float64)

    logger.info("特征: %d 列, 样本: %d 行", len(feature_cols), len(X))
    logger.info("正样本占比: %.2f%%", y.mean() * 100)

    # 确保标准化（processed_features 已标准化过，这里做防御性处理）
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # L1 逻辑回归（class_weight='balanced' 处理正负样本不均衡）
    model = LogisticRegression(
        penalty="l1",
        C=L1_C,
        solver="saga",          # saga 支持 L1 + 大数据集
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    logger.info("训练 L1 逻辑回归 (C=%.2f, solver=saga)...", L1_C)
    model.fit(X, y)

    # 提取系数
    coefs = model.coef_.flatten()
    coef_series = pd.Series(np.abs(coefs), index=feature_cols).sort_values(ascending=False)
    zero_coef_cols = [c for c in feature_cols if c not in coef_series[coef_series > 0].index]

    train_score = model.score(X, y)

    logger.info("训练完成: 准确率=%.4f", train_score)
    logger.info(
        "系数非零: %d 列, 被压成 0: %d 列",
        len(feature_cols) - len(zero_coef_cols),
        len(zero_coef_cols),
    )
    if zero_coef_cols:
        logger.info("L1 系数为 0 的特征: %s", ", ".join(zero_coef_cols))

    return {
        "coefs": coef_series,
        "zero_coef_cols": zero_coef_cols,
        "l1_c": L1_C,
        "train_score": train_score,
    }

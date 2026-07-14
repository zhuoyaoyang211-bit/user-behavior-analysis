"""三种不平衡处理方案对比评估模块。

在验证集上对比三种不平衡处理方案的效果：
    1. SMOTE 过采样
    2. 欠采样
    3. 类别权重

评估指标：Precision、Recall、F1-score、AUC（ROC 曲线下面积）。

注：验证集保持原始不平衡分布，不参与任何采样或权重调整。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler

from common.logger import get_logger

logger = get_logger(__name__)

# L1 正则化强度
L1_C = 0.1
RANDOM_STATE = 42


def _train_and_evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    method_name: str,
    class_weight: dict[int, float] | None = None,
) -> dict[str, Any]:
    """训练 L1 逻辑回归并在验证集上评估。

    Args:
        X_train: 训练集特征矩阵
        y_train: 训练集标签
        X_val: 验证集特征矩阵
        y_val: 验证集标签
        method_name: 方案名称（SMOTE / 欠采样 / 类别权重）
        class_weight: 类别权重字典，为 None 表示不设权重

    Returns:
        metrics: {
            "method": 方案名称,
            "precision": 精确率,
            "recall": 召回率,
            "f1": F1 分数,
            "auc": ROC AUC,
            "train_score": 训练集准确率,
            "n_train": 训练集行数,
        }
    """
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # 训练
    model = LogisticRegression(
        penalty="l1",
        C=L1_C,
        solver="saga",
        max_iter=2000,
        class_weight=class_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    logger.info("  [%s] 训练中 (n_train=%s)...", method_name, f"{len(X_train):,}")
    model.fit(X_train_scaled, y_train)

    train_score = model.score(X_train_scaled, y_train)

    # 预测
    y_pred = model.predict(X_val_scaled)
    y_proba = model.predict_proba(X_val_scaled)[:, 1]

    precision = precision_score(y_val, y_pred, zero_division=0)
    recall = recall_score(y_val, y_pred, zero_division=0)
    f1 = f1_score(y_val, y_pred, zero_division=0)
    auc_score = roc_auc_score(y_val, y_proba)

    logger.info(
        "  [%s] Precision=%.4f  Recall=%.4f  F1=%.4f  AUC=%.4f  TrainAcc=%.4f",
        method_name,
        precision,
        recall,
        f1,
        auc_score,
        train_score,
    )

    return {
        "method": method_name,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc_score,
        "train_score": train_score,
        "n_train": len(X_train),
    }


def compare_methods(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    feature_cols: list[str],
    df_train_smote: pd.DataFrame | None = None,
    df_train_under: pd.DataFrame | None = None,
    class_weight_info: dict | None = None,
) -> list[dict[str, Any]]:
    """对比三种不平衡处理方案，在验证集上评估各方案效果。

    Args:
        df_train: 原始训练集
        df_val: 验证集（保持原始分布，不做任何处理）
        feature_cols: 特征列名列表
        df_train_smote: SMOTE 处理后的训练集，None 则跳过
        df_train_under: 欠采样后的训练集，None 则跳过
        class_weight_info: 类别权重信息，None 则跳过

    Returns:
        results: 各方案评估结果列表，按 F1 降序排列
    """
    label_col = "label"

    # 准备验证集（只做一次）
    X_val = df_val[feature_cols].values.astype(np.float64)
    y_val = df_val[label_col].values.astype(int)

    logger.info("=" * 60)
    logger.info("开始对比三种不平衡处理方案 (验证集 %s 行)", f"{len(df_val):,}")
    logger.info("验证集正样本占比: %.2f%%", y_val.mean() * 100)
    logger.info("=" * 60)

    all_results: list[dict[str, Any]] = []

    # 方案1：SMOTE 过采样
    if df_train_smote is not None:
        X_smote = df_train_smote[feature_cols].values.astype(np.float64)
        y_smote = df_train_smote[label_col].values.astype(int)
        r = _train_and_evaluate(
            X_smote, y_smote, X_val, y_val,
            method_name="SMOTE 过采样",
        )
        all_results.append(r)
    else:
        logger.warning("跳过 SMOTE（未提供 SMOTE 训练集）")

    # 方案2：欠采样
    if df_train_under is not None:
        X_under = df_train_under[feature_cols].values.astype(np.float64)
        y_under = df_train_under[label_col].values.astype(int)
        r = _train_and_evaluate(
            X_under, y_under, X_val, y_val,
            method_name="欠采样",
        )
        all_results.append(r)
    else:
        logger.warning("跳过欠采样（未提供欠采样训练集）")

    # 方案3：类别权重 — 用原始训练集 + class_weight 参数
    if class_weight_info is not None:
        X_raw = df_train[feature_cols].values.astype(np.float64)
        y_raw = df_train[label_col].values.astype(int)
        cw = class_weight_info.get("class_weights")
        r = _train_and_evaluate(
            X_raw, y_raw, X_val, y_val,
            method_name="类别权重",
            class_weight=cw,
        )
        all_results.append(r)
    else:
        logger.warning("跳过类别权重（未提供权重信息）")

    # 按 F1 降序排列
    all_results.sort(key=lambda x: x["f1"], reverse=True)

    return all_results


def print_comparison_table(results: list[dict[str, Any]]) -> None:
    """打印对比结果表格到控制台。

    Args:
        results: compare_methods 返回的结果列表
    """
    print("\n" + "=" * 75)
    print("三种不平衡处理方案对比（验证集）")
    print("=" * 75)

    header = f"  {'方案':<14s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s} {'AUC':>10s} {'TrainAcc':>10s}"
    print(header)
    print("  " + "-" * 68)

    best_f1 = results[0]["f1"] if results else 0

    for r in results:
        marker = " ★" if r["f1"] == best_f1 else "  "
        line = (
            f"  {r['method']:<12s}{marker}"
            f" {r['precision']:>10.4f}"
            f" {r['recall']:>10.4f}"
            f" {r['f1']:>10.4f}"
            f" {r['auc']:>10.4f}"
            f" {r['train_score']:>10.4f}"
        )
        print(line)

    print("  " + "-" * 68)
    print(f"  ★ = F1 最高方案")
    print("  (验证集保持原始分布，未参与任何采样或权重调整)")
    print("=" * 75 + "\n")

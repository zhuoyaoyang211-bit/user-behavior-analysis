"""基线模型训练与评估模块。

对逻辑回归/XGBoost/LightGBM 等模型提供统一的训练、评估、保存接口。
每个模型返回标准化的评估结果（Precision/Recall/F1/AUC + 训练时间）。
"""

from __future__ import annotations

import logging
import os
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler

from common.logger import get_logger

logger = get_logger(__name__)

# ---------- 全局配置 ----------

# 模型输出目录
MODEL_DIR = Path("output/models")

# 特征列（排除主键、原始目标、标签）
EXCLUDE_COLS = {"user_id", "buy_path_type", "label"}

# 随机种子
SEED = 42

# L1 正则化强度（与 Part4 保持一致）
L1_C = 0.1


def _load_data(train_path: str, val_path: str) -> dict[str, Any]:
    """加载 Part5 输出的训练集和验证集，分离特征和标签。

    Args:
        train_path: train.parquet 路径
        val_path: val.parquet 路径

    Returns:
        {"X_train": ..., "y_train": ..., "X_val": ..., "y_val": ...,
         "feature_cols": ..., "n_features": ..., "n_train": ..., "n_val": ...}
    """
    logger.info("加载训练集: %s", train_path)
    train = pd.read_parquet(train_path)
    logger.info("加载验证集: %s", val_path)
    val = pd.read_parquet(val_path)

    feature_cols = [c for c in train.columns if c not in EXCLUDE_COLS]

    X_train = train[feature_cols].values.astype(np.float64)
    y_train = train["label"].values.astype(np.int64)
    X_val = val[feature_cols].values.astype(np.float64)
    y_val = val["label"].values.astype(np.int64)

    logger.info(
        "特征: %d 列 | 训练集: %d 行 (正样本 %.2f%%) | 验证集: %d 行 (正样本 %.2f%%)",
        len(feature_cols),
        len(X_train),
        y_train.mean() * 100,
        len(X_val),
        y_val.mean() * 100,
    )

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "feature_cols": feature_cols,
        "n_features": len(feature_cols),
        "n_train": len(X_train),
        "n_val": len(X_val),
    }


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, float]:
    """计算标准评估指标。

    Args:
        y_true: 真实标签 (0/1)
        y_pred: 预测标签 (0/1)，阈值 0.5
        y_prob: 预测概率

    Returns:
        {"accuracy", "precision", "recall", "f1", "auc", "tn", "fp", "fn", "tp"}
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "auc": round(roc_auc_score(y_true, y_prob), 4),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def train_lr_baseline(
    train_path: str = "output/train.parquet",
    val_path: str = "output/val.parquet",
    save_model: bool = True,
) -> dict[str, Any]:
    """训练逻辑回归基线模型（L1 正则化 + class_weight='balanced'）。

    设计决策：
    - C=0.1（与 Part4 保持一致，增强正则化以获得稀疏特征权重）
    - class_weight='balanced'（Part5 对比验证后确认效果与 SMOTE 持平，且更简洁）
    - 标准化：所有特征先做 StandardScaler（量纲统一 + L1 对尺度敏感）

    Args:
        train_path: 训练集路径，默认 output/train.parquet
        val_path: 验证集路径，默认 output/val.parquet
        save_model: 是否保存模型到 output/models/

    Returns:
        {"model": 训练好的模型, "metrics": 评估指标 dict, "train_time": 训练秒数,
         "feature_cols": 特征列名列表, "coefs": 特征系数 Series}
    """
    logger.info("=" * 60)
    logger.info("逻辑回归基线训练（L1 + class_weight='balanced'）")
    logger.info("=" * 60)

    # 1. 加载数据
    data = _load_data(train_path, val_path)
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    feature_cols = data["feature_cols"]

    # 2. 标准化
    logger.info("标准化 ...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # 3. 训练 L1 逻辑回归
    model = LogisticRegression(
        penalty="l1",
        C=L1_C,
        solver="saga",
        max_iter=2000,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )

    logger.info("训练中 (C=%.2f, %d 样本)...", L1_C, len(X_train))
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = round(time.time() - t0, 2)
    logger.info("训练完成: %.2f 秒", train_time)

    # 4. 训练集评估
    y_train_pred = model.predict(X_train)
    y_train_prob = model.predict_proba(X_train)[:, 1]
    train_metrics = _compute_metrics(y_train, y_train_pred, y_train_prob)
    logger.info(
        "训练集 - Acc: %.4f | Prec: %.4f | Recall: %.4f | F1: %.4f | AUC: %.4f",
        train_metrics["accuracy"],
        train_metrics["precision"],
        train_metrics["recall"],
        train_metrics["f1"],
        train_metrics["auc"],
    )

    # 5. 验证集评估（核心产出）
    y_val_pred = model.predict(X_val)
    y_val_prob = model.predict_proba(X_val)[:, 1]
    val_metrics = _compute_metrics(y_val, y_val_pred, y_val_prob)
    logger.info(
        "验证集 - Acc: %.4f | Prec: %.4f | Recall: %.4f | F1: %.4f | AUC: %.4f",
        val_metrics["accuracy"],
        val_metrics["precision"],
        val_metrics["recall"],
        val_metrics["f1"],
        val_metrics["auc"],
    )
    logger.info(
        "混淆矩阵: TN=%d, FP=%d, FN=%d, TP=%d",
        val_metrics["tn"],
        val_metrics["fp"],
        val_metrics["fn"],
        val_metrics["tp"],
    )

    # 6. 特征系数排名
    coefs = model.coef_.flatten()
    coef_series = pd.Series(
        np.abs(coefs), index=feature_cols
    ).sort_values(ascending=False)
    zero_coefs = [c for c in feature_cols if coef_series.get(c, 0) == 0]

    logger.info("特征系数非零: %d / %d", len(feature_cols) - len(zero_coefs), len(feature_cols))
    if zero_coefs:
        logger.info("系数为 0 的特征: %s", ", ".join(zero_coefs))

    # 7. 保存模型
    if save_model:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODEL_DIR / "lr_baseline.pkl"
        with open(model_path, "wb") as f:
            pickle.dump({"model": model, "scaler": scaler, "feature_cols": feature_cols}, f)
        logger.info("模型已保存: %s", model_path)

    return {
        "model": model,
        "scaler": scaler,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "train_time": train_time,
        "feature_cols": feature_cols,
        "coefs": coef_series,
        "zero_coefs": zero_coefs,
    }


def train_lgb_baseline(
    train_path: str = "output/train.parquet",
    val_path: str = "output/val.parquet",
    save_model: bool = True,
) -> dict[str, Any]:
    """训练 LightGBM 基线模型。

    设计决策：
    - 树模型不需要标准化（对特征尺度不敏感）
    - scale_pos_weight 从数据中计算（≈负样本/正样本），等价于 LR 的 class_weight='balanced'
    - 所有参数为行业经验起步值，步骤 7 用 Optuna 调参

    Args:
        train_path: 训练集路径，默认 output/train.parquet
        val_path: 验证集路径，默认 output/val.parquet
        save_model: 是否保存模型到 output/models/

    Returns:
        {"model": 训练好的模型, "metrics": 评估指标 dict, "train_time": 训练秒数,
         "feature_cols": 特征列名列表, "importances": 特征重要性 Series}
    """
    logger.info("=" * 60)
    logger.info("LightGBM 基线训练")
    logger.info("=" * 60)

    # 1. 加载数据（树模型不需要标准化）
    data = _load_data(train_path, val_path)
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    feature_cols = data["feature_cols"]

    # 2. 计算 scale_pos_weight（负样本数 ÷ 正样本数）
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    scale_pos_weight = round(n_neg / n_pos, 1) if n_pos > 0 else 1.0
    logger.info("正样本: %d, 负样本: %d, scale_pos_weight: %.1f", n_pos, n_neg, scale_pos_weight)

    # 3. 初始化 LightGBM（起步参数，步骤 7 再调）
    model = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        min_child_samples=50,
        scale_pos_weight=scale_pos_weight,
        subsample=0.8,
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
    )

    # 4. 训练
    logger.info("训练中 (%d 样本, %d 棵树)...", len(X_train), 500)
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = round(time.time() - t0, 2)
    logger.info("训练完成: %.2f 秒", train_time)

    # 5. 训练集评估
    y_train_pred = model.predict(X_train)
    y_train_prob = model.predict_proba(X_train)[:, 1]
    train_metrics = _compute_metrics(y_train, y_train_pred, y_train_prob)
    logger.info(
        "训练集 - Acc: %.4f | Prec: %.4f | Recall: %.4f | F1: %.4f | AUC: %.4f",
        train_metrics["accuracy"],
        train_metrics["precision"],
        train_metrics["recall"],
        train_metrics["f1"],
        train_metrics["auc"],
    )

    # 6. 验证集评估（核心产出）
    y_val_pred = model.predict(X_val)
    y_val_prob = model.predict_proba(X_val)[:, 1]
    val_metrics = _compute_metrics(y_val, y_val_pred, y_val_prob)
    logger.info(
        "验证集 - Acc: %.4f | Prec: %.4f | Recall: %.4f | F1: %.4f | AUC: %.4f",
        val_metrics["accuracy"],
        val_metrics["precision"],
        val_metrics["recall"],
        val_metrics["f1"],
        val_metrics["auc"],
    )
    logger.info(
        "混淆矩阵: TN=%d, FP=%d, FN=%d, TP=%d",
        val_metrics["tn"],
        val_metrics["fp"],
        val_metrics["fn"],
        val_metrics["tp"],
    )

    # 7. 特征重要性（gain 型：该特征在分裂点"降低多少误差"）
    importances = model.feature_importances_
    imp_series = pd.Series(importances, index=feature_cols).sort_values(ascending=False)
    zero_imp = [c for c in feature_cols if imp_series.get(c, 0) == 0]

    logger.info("特征重要性非零: %d / %d", len(feature_cols) - len(zero_imp), len(feature_cols))
    if zero_imp:
        logger.info("重要性为 0 的特征: %s", ", ".join(zero_imp))

    # 8. 保存模型
    if save_model:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODEL_DIR / "lgb_baseline.pkl"
        with open(model_path, "wb") as f:
            pickle.dump({"model": model, "feature_cols": feature_cols}, f)
        logger.info("模型已保存: %s", model_path)

    return {
        "model": model,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "train_time": train_time,
        "feature_cols": feature_cols,
        "importances": imp_series,
        "zero_imp": zero_imp,
    }


def train_xgb_baseline(
    train_path: str = "output/train.parquet",
    val_path: str = "output/val.parquet",
    save_model: bool = True,
) -> dict[str, Any]:
    """训练 XGBoost 基线模型。

    设计决策：
    - 树模型不需要标准化（对特征尺度不敏感）
    - scale_pos_weight 从数据中计算（≈负样本/正样本），等价于 LR 的 class_weight='balanced'
    - tree_method='hist'：M2 Mac 没有 NVIDIA GPU，用 CPU 上的直方图加速
    - 所有参数为行业经验起步值，步骤 7 用 Optuna 调参

    Args:
        train_path: 训练集路径，默认 output/train.parquet
        val_path: 验证集路径，默认 output/val.parquet
        save_model: 是否保存模型到 output/models/

    Returns:
        {"model": 训练好的模型, "metrics": 评估指标 dict, "train_time": 训练秒数,
         "feature_cols": 特征列名列表, "importances": 特征重要性 Series}
    """
    logger.info("=" * 60)
    logger.info("XGBoost 基线训练")
    logger.info("=" * 60)

    # 1. 加载数据（树模型不需要标准化）
    data = _load_data(train_path, val_path)
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    feature_cols = data["feature_cols"]

    # 2. 计算 scale_pos_weight（负样本数 ÷ 正样本数）
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    scale_pos_weight = round(n_neg / n_pos, 1) if n_pos > 0 else 1.0
    logger.info("正样本: %d, 负样本: %d, scale_pos_weight: %.1f", n_pos, n_neg, scale_pos_weight)

    # 3. 初始化 XGBoost（起步参数，步骤 7 再调）
    model = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        scale_pos_weight=scale_pos_weight,
        subsample=0.8,
        tree_method="hist",      # M2 Mac CPU 最优方法
        random_state=SEED,
        n_jobs=-1,
        verbosity=0,
    )

    # 4. 训练
    logger.info("训练中 (%d 样本, %d 棵树)...", len(X_train), 500)
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = round(time.time() - t0, 2)
    logger.info("训练完成: %.2f 秒", train_time)

    # 5. 训练集评估
    y_train_pred = model.predict(X_train)
    y_train_prob = model.predict_proba(X_train)[:, 1]
    train_metrics = _compute_metrics(y_train, y_train_pred, y_train_prob)
    logger.info(
        "训练集 - Acc: %.4f | Prec: %.4f | Recall: %.4f | F1: %.4f | AUC: %.4f",
        train_metrics["accuracy"],
        train_metrics["precision"],
        train_metrics["recall"],
        train_metrics["f1"],
        train_metrics["auc"],
    )

    # 6. 验证集评估（核心产出）
    y_val_pred = model.predict(X_val)
    y_val_prob = model.predict_proba(X_val)[:, 1]
    val_metrics = _compute_metrics(y_val, y_val_pred, y_val_prob)
    logger.info(
        "验证集 - Acc: %.4f | Prec: %.4f | Recall: %.4f | F1: %.4f | AUC: %.4f",
        val_metrics["accuracy"],
        val_metrics["precision"],
        val_metrics["recall"],
        val_metrics["f1"],
        val_metrics["auc"],
    )
    logger.info(
        "混淆矩阵: TN=%d, FP=%d, FN=%d, TP=%d",
        val_metrics["tn"],
        val_metrics["fp"],
        val_metrics["fn"],
        val_metrics["tp"],
    )

    # 7. 特征重要性（gain 型：该特征在分裂点"降低多少误差"）
    importances = model.feature_importances_
    imp_series = pd.Series(importances, index=feature_cols).sort_values(ascending=False)
    zero_imp = [c for c in feature_cols if imp_series.get(c, 0) == 0]

    logger.info("特征重要性非零: %d / %d", len(feature_cols) - len(zero_imp), len(feature_cols))
    if zero_imp:
        logger.info("重要性为 0 的特征: %s", ", ".join(zero_imp))

    # 8. 保存模型
    if save_model:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODEL_DIR / "xgb_baseline.pkl"
        with open(model_path, "wb") as f:
            pickle.dump({"model": model, "feature_cols": feature_cols}, f)
        logger.info("模型已保存: %s", model_path)

    return {
        "model": model,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "train_time": train_time,
        "feature_cols": feature_cols,
        "importances": imp_series,
        "zero_imp": zero_imp,
    }

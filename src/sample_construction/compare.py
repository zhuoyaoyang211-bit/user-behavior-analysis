"""三种不平衡处理方案对比评估模块。

在验证集上对比三种不平衡处理方案的效果：
    1. SMOTE 过采样
    2. 欠采样（3 次随机取平均）
    3. 类别权重

评估指标：Precision、Recall、F1-score、AUC（ROC 曲线下面积）。

改进点：
- 统一的 L1 LR 模型，C 值由 GridSearchCV 在原始训练集上选最优（3 折交叉验证）
- 欠采样用 3 个不同 random_state 跑 3 次取平均，避免单次随机抽样的偶然性
- 验证集保持原始不平衡分布，不参与任何采样或权重调整
"""

from __future__ import annotations

import os
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from common.logger import get_logger

logger = get_logger(__name__)

# 待搜索的 C 值候选集
C_GRID = [0.001, 0.01, 0.1, 1, 10]

# 交叉验证折数
CV_FOLDS = 3

# 欠采样重复次数（不同 random_state）
UNDERSAMPLE_SEEDS = [42, 100, 2024]

# 其他 L1 LR 固定参数
SOLVER = "saga"
MAX_ITER = 2000
RANDOM_STATE = 42


def find_best_C(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv_folds: int = CV_FOLDS,
) -> dict[str, Any]:
    """用 GridSearchCV 在原始训练集上找最优 C。

    原理：
        在原始（不平衡）训练集上，对每个候选 C 做 K 折分层交叉验证，
        评分指标为 F1。验证集保持完全独立，CV 折分只在训练集内部。

    Args:
        X_train: 原始训练集特征矩阵（未采样）
        y_train: 原始训练集标签
        cv_folds: 交叉验证折数

    Returns:
        {"best_C": 最优 C, "best_cv_f1": 最佳 CV F1, "cv_results": 完整 CV 详情}
    """
    logger.info("=" * 60)
    logger.info("Step 0: GridSearchCV 找最优 C（候选 %s，%d 折 CV）", C_GRID, cv_folds)
    logger.info("=" * 60)

    # 标准化（CV 内部每折会自己 fit，但这里先做一次以统一入口）
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    base_model = LogisticRegression(
        penalty="l1",
        solver=SOLVER,
        max_iter=MAX_ITER,
        class_weight="balanced",  # CV 阶段就用 balanced 权重，贴近最终方案3
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(
        estimator=base_model,
        param_grid={"C": C_GRID},
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        return_train_score=False,
        verbose=0,
    )

    t0 = time.time()
    grid.fit(X_scaled, y_train)
    elapsed = round(time.time() - t0, 2)

    best_C = grid.best_params_["C"]
    best_f1 = float(grid.best_score_)

    # 整理所有候选的 CV 结果
    cv_table = []
    for mean, std, params in zip(
        grid.cv_results_["mean_test_score"],
        grid.cv_results_["std_test_score"],
        grid.cv_results_["params"],
    ):
        cv_table.append({
            "C": params["C"],
            "cv_f1_mean": round(float(mean), 4),
            "cv_f1_std": round(float(std), 4),
        })

    logger.info("GridSearchCV 完成（%.1fs）", elapsed)
    logger.info("最优 C = %s（CV F1 = %.4f）", best_C, best_f1)
    logger.info("所有候选 C 的 CV 表现：")
    for row in cv_table:
        marker = "  ← 最优" if row["C"] == best_C else ""
        logger.info(
            "  C=%-8g  CV F1 = %.4f ± %.4f%s",
            row["C"], row["cv_f1_mean"], row["cv_f1_std"], marker,
        )

    return {
        "best_C": best_C,
        "best_cv_f1": best_f1,
        "cv_table": cv_table,
        "scaler": scaler,
    }


def _train_and_evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    method_name: str,
    C: float,
    class_weight: dict[int, float] | None = None,
) -> dict[str, Any]:
    """训练 L1 逻辑回归（C 由 GridSearchCV 给出）并在验证集上评估。

    Args:
        X_train: 训练集特征矩阵（可能已 SMOTE / 欠采样）
        y_train: 训练集标签
        X_val: 验证集特征矩阵
        y_val: 验证集标签
        method_name: 方案名称
        C: L1 正则化强度倒数（由 find_best_C 给出）
        class_weight: 类别权重字典，None 表示不设权重

    Returns:
        {"method", "precision", "recall", "f1", "auc", "train_score", "n_train", "C"}
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    model = LogisticRegression(
        penalty="l1",
        C=C,
        solver=SOLVER,
        max_iter=MAX_ITER,
        class_weight=class_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    logger.info("  [%s] 训练中 (n_train=%s, C=%s)...", method_name, f"{len(X_train):,}", C)
    t0 = time.time()
    model.fit(X_train_scaled, y_train)
    elapsed = round(time.time() - t0, 2)
    logger.info("  [%s] 训练耗时 %.1fs", method_name, elapsed)

    train_score = model.score(X_train_scaled, y_train)
    y_pred = model.predict(X_val_scaled)
    y_proba = model.predict_proba(X_val_scaled)[:, 1]

    precision = precision_score(y_val, y_pred, zero_division=0)
    recall = recall_score(y_val, y_pred, zero_division=0)
    f1 = f1_score(y_val, y_pred, zero_division=0)
    auc_score = roc_auc_score(y_val, y_proba)

    logger.info(
        "  [%s] Prec=%.4f  Recall=%.4f  F1=%.4f  AUC=%.4f  TrainAcc=%.4f",
        method_name, precision, recall, f1, auc_score, train_score,
    )

    return {
        "method": method_name,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc_score,
        "train_score": train_score,
        "n_train": len(X_train),
        "C": C,
        "n_zero_coef": int(np.sum(model.coef_ == 0)),
    }


def _average_metrics(run_results: list[dict[str, Any]]) -> dict[str, Any]:
    """对多次运行的指标取平均。

    Args:
        run_results: 多次 _train_and_evaluate 的结果列表

    Returns:
        平均后的 dict，带 std 字段
    """
    n = len(run_results)
    keys = ["precision", "recall", "f1", "auc", "train_score"]
    averaged: dict[str, Any] = {
        "method": f"欠采样 (n_runs={n})",
        "n_train": run_results[0]["n_train"],
        "C": run_results[0]["C"],
    }
    for k in keys:
        values = np.array([r[k] for r in run_results], dtype=float)
        averaged[k] = float(values.mean())
        averaged[f"{k}_std"] = float(values.std())
    averaged["raw_runs"] = run_results
    return averaged


def compare_methods(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    feature_cols: list[str],
    df_train_smote: pd.DataFrame | None = None,
    df_train_under_list: list[pd.DataFrame] | None = None,
    class_weight_info: dict | None = None,
    best_C: float | None = None,
) -> list[dict[str, Any]]:
    """对比三种不平衡处理方案。

    流程：
        1) 在原始训练集上 GridSearchCV 找最优 C
        2) 用最优 C 分别训练三种方案，在验证集上评估
        3) 欠采样方案对传入的多个数据集（不同 random_state）跑多次取平均

    Args:
        df_train: 原始训练集（未采样）
        df_val: 验证集
        feature_cols: 特征列名
        df_train_smote: SMOTE 处理后的训练集，None 跳过
        df_train_under_list: 多次欠采样后的训练集列表（每个对应一个 random_state）
        class_weight_info: 类别权重信息，None 跳过

    Returns:
        各方案评估结果列表，按 F1 降序排列
    """
    label_col = "label"

    X_train_raw = df_train[feature_cols].values.astype(np.float64)
    y_train_raw = df_train[label_col].values.astype(int)
    X_val = df_val[feature_cols].values.astype(np.float64)
    y_val = df_val[label_col].values.astype(int)

    logger.info("=" * 60)
    logger.info("开始对比三种不平衡处理方案（验证集 %s 行）", f"{len(df_val):,}")
    logger.info(
        "训练集 %s 行（正样本 %.2f%%）→ 验证集 %s 行（正样本 %.2f%%）",
        f"{len(df_train):,}", y_train_raw.mean() * 100,
        f"{len(df_val):,}", y_val.mean() * 100,
    )
    logger.info("=" * 60)

    # Step 0: 确定 C 值
    if best_C is not None:
        logger.info("使用指定的 C=%s（跳过 GridSearchCV）", best_C)
    else:
        grid_result = find_best_C(X_train_raw, y_train_raw)
        best_C = grid_result["best_C"]

    logger.info("=" * 60)
    logger.info("Step 1~3: 用 C=%s 跑三种方案", best_C)
    logger.info("=" * 60)

    all_results: list[dict[str, Any]] = []

    # 方案1：SMOTE 过采样（1 次）
    if df_train_smote is not None:
        X = df_train_smote[feature_cols].values.astype(np.float64)
        y = df_train_smote[label_col].values.astype(int)
        r = _train_and_evaluate(X, y, X_val, y_val, "SMOTE 过采样", C=best_C)
        all_results.append(r)
    else:
        logger.warning("跳过 SMOTE（未提供 SMOTE 训练集）")

    # 方案2：欠采样（多次取平均）
    if df_train_under_list and len(df_train_under_list) > 0:
        run_results = []
        for i, df_u in enumerate(df_train_under_list):
            X = df_u[feature_cols].values.astype(np.float64)
            y = df_u[label_col].values.astype(int)
            r = _train_and_evaluate(
                X, y, X_val, y_val,
                method_name=f"欠采样 run-{i + 1}",
                C=best_C,
            )
            run_results.append(r)
        averaged = _average_metrics(run_results)
        all_results.append(averaged)
        logger.info(
            "  [欠采样 汇总] F1 = %.4f ± %.4f（%d 次平均）",
            averaged["f1"], averaged["f1_std"], len(run_results),
        )
    else:
        logger.warning("跳过欠采样（未提供欠采样训练集列表）")

    # 方案3：类别权重
    if class_weight_info is not None:
        cw = class_weight_info.get("class_weights")
        r = _train_and_evaluate(
            X_train_raw, y_train_raw, X_val, y_val,
            method_name="类别权重",
            C=best_C,
            class_weight=cw,
        )
        all_results.append(r)
    else:
        logger.warning("跳过类别权重（未提供权重信息）")

    # 按 F1 降序
    all_results.sort(key=lambda x: x["f1"], reverse=True)
    return all_results


def print_comparison_table(results: list[dict[str, Any]]) -> None:
    """打印对比结果表格到控制台。

    Args:
        results: compare_methods 返回的结果列表
    """
    print("\n" + "=" * 90)
    print("三种不平衡处理方案对比（验证集）")
    print("=" * 90)

    header = (
        f"  {'方案':<22s} {'Precision':>10s} {'Recall':>10s} "
        f"{'F1':>10s} {'AUC':>10s} {'TrainAcc':>10s} {'n_train':>12s}"
    )
    print(header)
    print("  " + "-" * 84)

    best_f1 = results[0]["f1"] if results else 0

    for r in results:
        marker = " ★" if r["f1"] == best_f1 else "  "
        line = (
            f"  {r['method']:<20s}{marker}"
            f" {r['precision']:>10.4f}"
            f" {r['recall']:>10.4f}"
            f" {r['f1']:>10.4f}"
            f" {r['auc']:>10.4f}"
            f" {r['train_score']:>10.4f}"
            f" {r['n_train']:>12,d}"
        )
        print(line)
        # 欠采样方案额外打印 std
        if "f1_std" in r:
            print(
                f"  {'    (F1 std)':<22s}    "
                f"{'':>10s}{'':>10s}"
                f" {r['f1_std']:>10.4f}"
                f" {r['auc_std']:>10.4f}"
                f" {r['train_score_std']:>10.4f}"
            )
    print("  " + "-" * 84)
    print(f"  ★ = F1 最高方案")
    print(f"  （验证集保持原始分布，C={results[0]['C'] if results else '0.1'}，三种方案使用相同 C 值）")
    print("=" * 90 + "\n")

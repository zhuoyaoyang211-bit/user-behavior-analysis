"""Part6 模型选型与基线训练 —— 主入口。

当前完成：逻辑回归基线训练（可解释性基准）。
后续扩展：XGBoost → LightGBM → LSTM/GRU → DIN。

运行方式:
    cd "/Users/yangzhuoyao/Desktop/阿里/项目1"
    python src/train_baselines.py
"""

from __future__ import annotations

import logging
import os
import sys

# 切换工作目录到项目根
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from model_training.trainer import train_lgb_baseline, train_lr_baseline, train_xgb_baseline

from common.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    """主流程：训练逻辑回归基线模型，输出验证集评估结果。"""
    logger.info("=" * 60)
    logger.info("Part6 模型选型与基线训练")
    logger.info("=" * 60)

    # ─── 阶段 1：逻辑回归（可解释性基准）───
    logger.info("\n>>> 阶段 1：逻辑回归基线训练")
    result_lr = train_lr_baseline(
        train_path="output/train.parquet",
        val_path="output/val.parquet",
        save_model=True,
    )

    # ─── 输出汇总 ───
    logger.info("\n" + "=" * 60)
    logger.info("汇总：基线模型验证集性能")
    logger.info("=" * 60)
    val_metrics = result_lr["val_metrics"]
    logger.info("  LogisticRegression (L1, C=0.1, balanced)")
    logger.info(
        "    Acc=%.4f  Prec=%.4f  Recall=%.4f  F1=%.4f  AUC=%.4f  Time=%.1fs",
        val_metrics["accuracy"],
        val_metrics["precision"],
        val_metrics["recall"],
        val_metrics["f1"],
        val_metrics["auc"],
        result_lr["train_time"],
    )

    # ─── 特征重要性 Top 10（LR）───
    coefs = result_lr["coefs"]
    logger.info("\n逻辑回归特征重要性 Top 10 (L1 系数绝对值):")
    for i, (feat, val) in enumerate(coefs.head(10).items(), 1):
        logger.info("  %2d. %-35s = %.6f", i, feat, val)

    # ─── 阶段 2：LightGBM（树模型性能基准）───
    logger.info("\n>>> 阶段 2：LightGBM 基线训练")
    result_lgb = train_lgb_baseline(
        train_path="output/train.parquet",
        val_path="output/val.parquet",
        save_model=True,
    )

    # ─── LightGBM 评估 ───
    lgb_metrics = result_lgb["val_metrics"]
    logger.info("\n  LightGBM")
    logger.info(
        "    Acc=%.4f  Prec=%.4f  Recall=%.4f  F1=%.4f  AUC=%.4f  Time=%.1fs",
        lgb_metrics["accuracy"],
        lgb_metrics["precision"],
        lgb_metrics["recall"],
        lgb_metrics["f1"],
        lgb_metrics["auc"],
        result_lgb["train_time"],
    )

    # ─── LightGBM 特征重要性 Top 10 ───
    importances = result_lgb["importances"]
    logger.info("\nLightGBM 特征重要性 Top 10 (gain):")
    for i, (feat, val) in enumerate(importances.head(10).items(), 1):
        logger.info("  %2d. %-35s = %.6f", i, feat, val)

    # ─── 阶段 3：XGBoost（树模型性能基准 2）───
    logger.info("\n>>> 阶段 3：XGBoost 基线训练")
    result_xgb = train_xgb_baseline(
        train_path="output/train.parquet",
        val_path="output/val.parquet",
        save_model=True,
    )

    # ─── XGBoost 评估 ───
    xgb_metrics = result_xgb["val_metrics"]
    logger.info("\n  XGBoost")
    logger.info(
        "    Acc=%.4f  Prec=%.4f  Recall=%.4f  F1=%.4f  AUC=%.4f  Time=%.1fs",
        xgb_metrics["accuracy"],
        xgb_metrics["precision"],
        xgb_metrics["recall"],
        xgb_metrics["f1"],
        xgb_metrics["auc"],
        result_xgb["train_time"],
    )

    # ─── XGBoost 特征重要性 Top 10 ───
    xgb_importances = result_xgb["importances"]
    logger.info("\nXGBoost 特征重要性 Top 10 (gain):")
    for i, (feat, val) in enumerate(xgb_importances.head(10).items(), 1):
        logger.info("  %2d. %-35s = %.6f", i, feat, val)

    # ─── 基线性能对比表（训练集 vs 验证集）───
    logger.info("\n" + "=" * 70)
    logger.info("基线性能对比表（训练集 vs 验证集）")
    logger.info("=" * 70)
    header = f"{'模型':<16} {'数据集':<8} {'Acc':>8} {'Prec':>8} {'Recall':>8} {'F1':>8} {'AUC':>8} {'Time':>8}"
    logger.info(header)
    logger.info("-" * 70)

    models = [
        ("逻辑回归(L1)", result_lr),
        ("LightGBM", result_lgb),
        ("XGBoost", result_xgb),
    ]
    rows = []
    for name, result in models:
        for tag, key in [("训练集", "train_metrics"), ("验证集", "val_metrics")]:
            m = result[key]
            row = (
                f"{name:<16} {tag:<8} "
                f"{m['accuracy']:>8.4f} {m['precision']:>8.4f} "
                f"{m['recall']:>8.4f} {m['f1']:>8.4f} "
                f"{m['auc']:>8.4f}"
            )
            logger.info(row)
            rows.append({
                "model": name, "dataset": tag,
                "acc": m["accuracy"], "prec": m["precision"],
                "recall": m["recall"], "f1": m["f1"],
                "auc": m["auc"],
            })
        logger.info(f"  {'':40} Time: {result['train_time']:.1f}s")
        logger.info("-" * 70)

    # ─── 写入 Markdown 报告 ───
    import os
    docs_dir = os.path.join(PROJECT_ROOT, "docs")
    os.makedirs(docs_dir, exist_ok=True)
    md_path = os.path.join(docs_dir, "Part6_基线性能对比表.md")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Part6 基线性能对比表\n\n")
        f.write("| 模型 | 数据集 | Acc | Prec | Recall | F1 | AUC |\n")
        f.write("|------|-------|-----|------|--------|-----|------|\n")
        for r in rows:
            f.write(
                f"| {r['model']} | {r['dataset']} | {r['acc']:.4f} | "
                f"{r['prec']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} | {r['auc']:.4f} |\n"
            )
    logger.info("\n对比表已保存: docs/Part6_基线性能对比表.md")

    logger.info("\nDone. 模型已保存至 output/models/")


if __name__ == "__main__":
    main()

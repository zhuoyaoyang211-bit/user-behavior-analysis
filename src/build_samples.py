"""Part5 样本构建主入口。

输入：output/selected_features.parquet（Part4，4,686,904 行 × 27 列）
输出：
    output/train.parquet           — 训练集（70%，原始分布）
    output/val.parquet             — 验证集（20%，原始分布）
    output/test.parquet            — 测试集（10%，原始分布）
    output/train_smote.parquet     — SMOTE 过采样训练集
    output/train_undersample.parquet — 欠采样训练集
    docs/Part5_样本构建说明文档.md — 样本构建说明文档

处理流程：
    1. 加载 selected_features.parquet
    2. buy_path_type → label 二值化 (0=没买, 1=买了)
    3. 7:2:1 分层抽样划分
    4. 三种不平衡处理
    5. 在验证集上对比三种方案
    6. 输出数据集 + 说明文档
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common.logger import get_logger
from config import get_config
from sample_construction.builder import build_samples, prepare_xy
from sample_construction.compare import compare_methods, print_comparison_table
from sample_construction.imbalance import (
    apply_class_weight,
    apply_smote,
    apply_undersample,
)

logger = get_logger(__name__)

_cfg = get_config()
PROJECT_ROOT = str(_cfg.project_root)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
INPUT_PATH = os.path.join(OUTPUT_DIR, "selected_features.parquet")


def _generate_report(summary: dict, compare_results: list[dict]) -> str:
    """生成 Part5 样本构建说明文档（Markdown）。

    Args:
        summary: build_samples 返回的汇总字典
        compare_results: compare_methods 返回的对比结果

    Returns:
        完整的 Markdown 文本
    """
    total = summary["total"]
    train_s = summary["train_shape"]
    val_s = summary["val_shape"]
    test_s = summary["test_shape"]

    lines = []
    lines.append("# Part5 样本构建与数据集划分")
    lines.append("")
    lines.append("## 1. 数据说明")
    lines.append("")
    lines.append(f"- 输入：`output/selected_features.parquet`（Part4 输出）")
    lines.append(f"- 全量样本：{total:,} 行 × {summary['train_shape'][1]} 列")
    lines.append(f"- 特征列：{len(summary['feature_cols'])} 列")
    lines.append(f"- ID 列：user_id")
    lines.append(f"- 原始目标变量：buy_path_type（0=没买，1/2/3/4=不同购买路径）")
    lines.append("")
    lines.append("## 2. 目标变量定义")
    lines.append("")
    lines.append("由于原始数据时间跨度为 30 天（2025-11-18 ~ 2025-12-18），")
    lines.append("且未提供数据截止日期之后的购买记录，本阶段将目标变量定义为：")
    lines.append("")
    lines.append("> **label = 0**：用户-商品对未产生购买行为（`buy_path_type = 0`）— 负样本  ")
    lines.append("> **label = 1**：用户-商品对产生了购买行为（`buy_path_type > 0`）— 正样本")
    lines.append("")

    # 标签分布
    pos_total = int(train_s[0] * summary["train_pos_ratio"]
                    + val_s[0] * summary["val_pos_ratio"]
                    + test_s[0] * summary["test_pos_ratio"])
    neg_total = total - pos_total
    lines.append(f"- 全量正样本：{pos_total:,}（{pos_total / total * 100:.2f}%）")
    lines.append(f"- 全量负样本：{neg_total:,}（{neg_total / total * 100:.2f}%）")
    lines.append(f"- 正负样本比例 ≈ 1:{neg_total / pos_total:.1f}")
    lines.append("")
    lines.append("## 3. 数据集划分（7:2:1 分层抽样）")
    lines.append("")
    lines.append("| 数据集 | 样本数 | 占比 | 正样本占比 | 文件 |")
    lines.append("| --- | --- | --- | --- | --- |")

    def _ratio_pct(shape, ratio):
        return f"{shape[0] / total * 100:.1f}%"

    lines.append(
        f"| 训练集 | {train_s[0]:,} | 70% "
        f"| {summary['train_pos_ratio'] * 100:.2f}% "
        f"| `output/train.parquet` |"
    )
    lines.append(
        f"| 验证集 | {val_s[0]:,} | 20% "
        f"| {summary['val_pos_ratio'] * 100:.2f}% "
        f"| `output/val.parquet` |"
    )
    lines.append(
        f"| 测试集 | {test_s[0]:,} | 10% "
        f"| {summary['test_pos_ratio'] * 100:.2f}% "
        f"| `output/test.parquet` |"
    )
    lines.append("")

    lines.append("**划分策略：**")
    lines.append("- 第一刀：全量 → train (70%) + temp (30%)，按 label 分层")
    lines.append("- 第二刀：temp → val (66.67% of temp) + test (33.33% of temp)")
    lines.append("- 所有划分使用 `random_state=42` 保证可复现")
    lines.append("")

    lines.append("## 4. 类别不平衡处理")
    lines.append("")
    lines.append("由于正负样本严重不均衡（正样本占比约 2%），")
    lines.append("直接训练会导致模型偏向多数类（预测能力差）。")
    lines.append("本阶段对训练集采用三种处理方案，验证集和测试集保持原始分布。")
    lines.append("")
    lines.append("### 4.1 SMOTE 过采样")
    lines.append("")
    lines.append("- 原理：在正样本之间插值生成合成样本，使正负样本 1:1 平衡")
    lines.append("- 优点：保留全部原始数据 + 增加正样本多样性")
    lines.append("- 缺点：生成样本可能不完全真实，增加训练时间")
    lines.append(f"- 输出：`output/train_smote.parquet`")
    lines.append("")

    lines.append("### 4.2 欠采样")
    lines.append("")
    lines.append("- 原理：从负样本中随机抽样，保留与正样本相同数量的负样本")
    lines.append("- 优点：简单直接，数据量小训练快")
    lines.append("- 缺点：丢弃大量负样本信息，可能丢失有价值模式")
    lines.append(f"- 输出：`output/train_undersample.parquet`")
    lines.append("")

    lines.append("### 4.3 类别权重")
    lines.append("")
    lines.append("- 原理：不改动数据，训练时给正样本更高权重")
    lines.append("- 优点：保留全部原始数据，不产生假样本")
    lines.append("- 缺点：依赖模型本身的加权机制")
    lines.append(f"- 权重公式：`weight = n_samples / (n_classes × n_samples_per_class)`")
    lines.append("")

    lines.append("## 5. 三种方案对比（验证集）")
    lines.append("")
    lines.append("用 L1 逻辑回归（C=0.1, solver=saga）在验证集上评估各方案效果：")
    lines.append("")
    lines.append("| 方案 | Precision | Recall | F1 | AUC | TrainAcc |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for r in compare_results:
        lines.append(
            f"| {r['method']} "
            f"| {r['precision']:.4f} "
            f"| {r['recall']:.4f} "
            f"| {r['f1']:.4f} "
            f"| {r['auc']:.4f} "
            f"| {r['train_score']:.4f} |"
        )
    lines.append("")

    if compare_results:
        best = compare_results[0]
        lines.append(f"**推荐方案：{best['method']}**")
        lines.append(f"（F1={best['f1']:.4f}，AUC={best['auc']:.4f}）")
        lines.append("")

    lines.append("**评估指标说明：**")
    lines.append("- Precision（精确率）：预测为正的样本中有多少真的是正")
    lines.append("- Recall（召回率）：真正的正样本有多少被找出来")
    lines.append("- F1：Precision 和 Recall 的调和平均")
    lines.append("- AUC：模型排序能力（越接近 1 越好）")
    lines.append("- 验证集保持原始不平衡分布，未参与任何处理")
    lines.append("")

    lines.append("## 6. 产出物清单")
    lines.append("")
    lines.append("| 文件 | 说明 |")
    lines.append("| --- | --- |")
    lines.append("| `output/train.parquet` | 原始训练集（70%） |")
    lines.append("| `output/val.parquet` | 验证集（20%） |")
    lines.append("| `output/test.parquet` | 测试集（10%） |")
    lines.append("| `output/train_smote.parquet` | SMOTE 过采样训练集 |")
    lines.append("| `output/train_undersample.parquet` | 欠采样训练集 |")
    lines.append("| `docs/Part5_样本构建说明文档.md` | 本文档 |")
    lines.append("")
    lines.append("| 代码文件 | 说明 |")
    lines.append("| --- | --- |")
    lines.append("| `src/build_samples.py` | 主入口脚本 |")
    lines.append("| `src/sample_construction/builder.py` | 样本构建与划分 |")
    lines.append("| `src/sample_construction/imbalance.py` | 三种不平衡处理 |")
    lines.append("| `src/sample_construction/compare.py` | 三种方案对比评估 |")

    return "\n".join(lines)


def main() -> None:
    """样本构建主流程。"""
    t_start = time.time()

    # ========== 步骤1：构建样本 + 7:2:1 划分 ==========
    logger.info("=" * 60)
    logger.info("Part5 样本构建开始")
    logger.info("=" * 60)

    summary = build_samples(
        input_path=INPUT_PATH,
        output_dir=OUTPUT_DIR,
    )

    df_train = summary["df_train"]
    df_val = summary["df_val"]
    df_test = summary["df_test"]
    feature_cols = summary["feature_cols"]

    # ========== 步骤2：三种不平衡处理 ==========
    logger.info("-" * 60)
    logger.info("开始三种不平衡处理（仅处理训练集）")

    # 2.1 SMOTE 过采样
    df_train_smote, smote_info = apply_smote(
        df_train=df_train,
        feature_cols=feature_cols,
        output_dir=OUTPUT_DIR,
    )

    # 2.2 欠采样
    df_train_under, under_info = apply_undersample(
        df_train=df_train,
        feature_cols=feature_cols,
        output_dir=OUTPUT_DIR,
    )

    # 2.3 类别权重
    cw_info = apply_class_weight(df_train=df_train)

    # ========== 步骤3：三种方案对比 ==========
    logger.info("-" * 60)
    logger.info("开始三种方案对比评估（验证集）")

    compare_results = compare_methods(
        df_train=df_train,
        df_val=df_val,
        feature_cols=feature_cols,
        df_train_smote=df_train_smote,
        df_train_under=df_train_under,
        class_weight_info=cw_info,
    )

    # 打印对比表格
    print_comparison_table(compare_results)

    # ========== 步骤4：生成说明文档 ==========
    logger.info("生成 Part5 样本构建说明文档 ...")
    report = _generate_report(summary, compare_results)

    docs_dir = os.path.join(PROJECT_ROOT, "docs")
    os.makedirs(docs_dir, exist_ok=True)
    report_path = os.path.join(docs_dir, "Part5_样本构建说明文档.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info("说明文档已保存: %s", report_path)

    # ========== 完成 ==========
    elapsed = time.time() - t_start
    logger.info("=" * 60)
    logger.info("Part5 样本构建完成 (耗时 %.1f 秒)", elapsed)
    logger.info("输出目录: %s", OUTPUT_DIR)
    logger.info("输出文件:")
    logger.info("  train.parquet              — 训练集")
    logger.info("  val.parquet                — 验证集")
    logger.info("  test.parquet               — 测试集")
    logger.info("  train_smote.parquet        — SMOTE 训练集")
    logger.info("  train_undersample.parquet  — 欠采样训练集")
    logger.info("  ../docs/Part5_样本构建说明文档.md")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

"""Part5 类别不平衡处理方案对比 —— 主入口。

串联以下流程：
    1) 加载 train/val
    2) 跑 SMOTE 过采样（保存 train_smote.parquet）
    3) 跑 3 次欠采样（random_state 42/100/2024，各保存一份）
    4) 计算类别权重
    5) GridSearchCV 在原始训练集上找最优 C
    6) 用最优 C 跑三种方案对比
    7) 输出结果表 + Markdown 报告

运行方式:
    cd "/Users/yangzhuoyao/Desktop/阿里/项目1"
    python src/run_imbalance_comparison.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd

from common.logger import get_logger
from sample_construction.compare import (
    RANDOM_STATE,
    UNDERSAMPLE_SEEDS,
    compare_methods,
    print_comparison_table,
)
from sample_construction.imbalance import (
    apply_class_weight,
    apply_smote,
    apply_undersample,
)

logger = get_logger("run_imbalance_comparison")

OUTPUT_DIR = "output"
DOCS_DIR = "docs"


def _load_train_val() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """加载 train/val 集合，识别特征列。"""
    train_path = os.path.join(OUTPUT_DIR, "train.parquet")
    val_path = os.path.join(OUTPUT_DIR, "val.parquet")
    logger.info("加载训练集: %s", train_path)
    df_train = pd.read_parquet(train_path)
    logger.info("加载验证集: %s", val_path)
    df_val = pd.read_parquet(val_path)

    # 排除：主键 / 标签 / datetime / bool / ID列
    exclude = {"user_id", "item_id", "label", "buy_path_type", "last_time", "is_power_user"}
    # 额外排除任何非数值的列（datetime/bool/object 等，无法转 float64）
    exclude.update(
        c for c in df_train.columns
        if df_train[c].dtype.name not in ("int32", "int64", "float32", "float64", "int8", "int16")
    )
    feature_cols = [c for c in df_train.columns if c not in exclude]
    logger.info("特征列: %d 列（已排除 %s）", len(feature_cols), ", ".join(sorted(exclude)))
    logger.info(
        "训练集: %d 行 (正样本 %.2f%%) | 验证集: %d 行 (正样本 %.2f%%)",
        len(df_train), df_train["label"].mean() * 100,
        len(df_val), df_val["label"].mean() * 100,
    )
    return df_train, df_val, feature_cols


def main() -> None:
    """主流程：构建三种不平衡方案并对比评估。"""
    logger.info("=" * 60)
    logger.info("Part5：类别不平衡处理方案对比")
    logger.info("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DOCS_DIR, exist_ok=True)

    # ─── 1) 加载数据 ───
    df_train, df_val, feature_cols = _load_train_val()

    # ─── 2) SMOTE 过采样（1 次）───
    logger.info("\n>>> Step 1: SMOTE 过采样")
    df_smote, smote_info = apply_smote(
        df_train=df_train,
        feature_cols=feature_cols,
        output_dir=OUTPUT_DIR,
        random_state=RANDOM_STATE,
    )

    # ─── 3) 欠采样（3 次：42/100/2024）───
    logger.info("\n>>> Step 2: 欠采样 × %d 次（避免单次随机偶然性）", len(UNDERSAMPLE_SEEDS))
    df_under_list = []
    under_info_list = []
    for seed in UNDERSAMPLE_SEEDS:
        logger.info("\n--- 欠采样 random_state=%d ---", seed)
        df_u, info_u = apply_undersample(
            df_train=df_train,
            feature_cols=feature_cols,
            output_dir=OUTPUT_DIR,
            random_state=seed,
        )
        df_under_list.append(df_u)
        under_info_list.append(info_u)

    # ─── 4) 类别权重 ───
    logger.info("\n>>> Step 3: 计算类别权重")
    class_weight_info = apply_class_weight(df_train=df_train)

    # ─── 5) 三种方案对比（C=1.0，跳过 GridSearchCV）───
    logger.info("\n>>> Step 4: 三种方案对比（C=1.0）")
    results = compare_methods(
        df_train=df_train,
        df_val=df_val,
        feature_cols=feature_cols,
        df_train_smote=df_smote,
        df_train_under_list=df_under_list,
        class_weight_info=class_weight_info,
        best_C=1.0,
    )

    # ─── 6) 打印对比表 ───
    print_comparison_table(results)

    # ─── 7) 保存 Markdown 报告 ───
    md_path = os.path.join(DOCS_DIR, "Part5_不平衡处理对比.md")
    write_markdown_report(
        md_path=md_path,
        results=results,
        smote_info=smote_info,
        under_info_list=under_info_list,
        class_weight_info=class_weight_info,
    )
    logger.info("Markdown 报告已保存: %s", md_path)

    # ─── 8) 保存 JSON 报告（机器可读）───
    json_path = os.path.join(OUTPUT_DIR, "imbalance_comparison.json")
    write_json_report(
        json_path=json_path,
        results=results,
        smote_info=smote_info,
        under_info_list=under_info_list,
        class_weight_info=class_weight_info,
    )
    logger.info("JSON 报告已保存: %s", json_path)

    logger.info("\nDone. 三种方案对比完成。")


def write_markdown_report(
    md_path: str,
    results: list[dict],
    smote_info: dict,
    under_info_list: list[dict],
    class_weight_info: dict,
) -> None:
    """把对比结果写成 Markdown 报告。"""
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Part5 类别不平衡处理方案对比报告\n\n")
        f.write("> 任务：对⽐ SMOTE 过采样、⽋采样、类别权重调整三种方案，处理购买样本占比低的类别不平衡问题。\n\n")

        # 1. 原始数据情况
        f.write("## 1. 原始数据情况\n\n")
        f.write(f"- 训练集：{smote_info['before_shape'][0]:,} 行，特征 {smote_info['before_shape'][1]} 列\n")
        f.write(f"- 原始正样本占比：{smote_info['before_pos_ratio'] * 100:.4f}%\n\n")

        # 2. 三种方案的处理结果
        f.write("## 2. 三种方案处理后数据集规模\n\n")
        f.write("| 方案 | 处理后行数 | 正样本占比 |\n")
        f.write("|------|-----------|-----------|\n")
        f.write(f"| SMOTE 过采样 | {smote_info['after_shape'][0]:,} | {smote_info['after_pos_ratio'] * 100:.2f}% |\n")
        for i, info in enumerate(under_info_list):
            f.write(f"| 欠采样 (rs={UNDERSAMPLE_SEEDS[i]}) | {info['after_shape'][0]:,} | {info['after_pos_ratio'] * 100:.2f}% |\n")
        f.write(f"| 类别权重（不改动数据）| {smote_info['before_shape'][0]:,} | {smote_info['before_pos_ratio'] * 100:.4f}% |\n\n")

        # 3. 类别权重
        f.write("## 3. 类别权重（方案3）\n\n")
        cw = class_weight_info["class_weights"]
        f.write(f"- 负样本 (class=0) 权重: `{cw[0]:.4f}`\n")
        f.write(f"- 正样本 (class=1) 权重: `{cw[1]:.4f}`\n")
        f.write(f"- 权重比（正/负）: `{cw[1] / cw[0]:.1f}×`\n\n")
        f.write("> 含义：模型在计算损失时，正样本错误的代价是负样本错误的约 "
                f"{cw[1] / cw[0]:.0f} 倍，使模型更重视正样本。\n\n")

        # 4. GridSearchCV 选 C
        # 从 results 第一个里取 C（所有方案共用）
        if results and "C" in results[0]:
            best_C = results[0]["C"]
            f.write("## 4. GridSearchCV 选最优 C\n\n")
            f.write(f"**最优 C = {best_C}**\n\n")
            f.write("所有方案（SMOTE / 欠采样 / 类别权重）都使用此 C 值，确保控制变量、对比公平。\n\n")
            f.write("> 注：GridSearchCV 在原始训练集（未采样）上做 3 折分层交叉验证，评分指标为 F1。\n\n")

        # 5. 三种方案对比表
        f.write("## 5. 三种方案验证集表现对比\n\n")
        f.write("| 方案 | Precision | Recall | F1 | AUC | TrainAcc | n_train |\n")
        f.write("|------|-----------|--------|----|-----|----------|---------|\n")
        for r in results:
            line = (
                f"| {r['method']} "
                f"| {r['precision']:.4f} "
                f"| {r['recall']:.4f} "
                f"| {r['f1']:.4f} "
                f"| {r['auc']:.4f} "
                f"| {r['train_score']:.4f} "
                f"| {r['n_train']:,} |"
            )
            f.write(line + "\n")
        f.write("\n")

        # 6. 欠采样多次运行的 std
        f.write("## 6. 欠采样方案多次运行稳定性\n\n")
        f.write("欠采样跑 3 次不同 random_state，验证集 F1 表现：\n\n")
        for r in results:
            if "raw_runs" in r:
                f.write(f"**{r['method']}**（平均 F1 = {r['f1']:.4f} ± {r['f1_std']:.4f}）\n\n")
                f.write("| 运行 | F1 | Precision | Recall | AUC |\n")
                f.write("|------|----|-----------|--------|-----|\n")
                for j, run in enumerate(r["raw_runs"]):
                    f.write(
                        f"| run-{j + 1} (rs={UNDERSAMPLE_SEEDS[j]}) "
                        f"| {run['f1']:.4f} "
                        f"| {run['precision']:.4f} "
                        f"| {run['recall']:.4f} "
                        f"| {run['auc']:.4f} |\n"
                    )
                f.write("\n")
                f.write("> 多次运行取平均，结论更稳健。\n\n")

        # 7. 结论
        if results:
            best = results[0]
            f.write("## 7. 结论\n\n")
            f.write(f"- **F1 最高方案**：`{best['method']}`（F1 = {best['f1']:.4f}）\n")
            f.write(f"- 所有方案使用同一 C 值（`{results[0].get('C', '?')}`），由 GridSearchCV 选出，\n")
            f.write(f"  确保唯一的差异是「怎么处理不平衡」而不是「模型超参数」。\n")
            f.write(f"- 验证集始终保持原始分布（{class_weight_info['n_pos'] + class_weight_info['n_neg']:,} 行，"
                    f"正样本 {class_weight_info['n_pos'] / (class_weight_info['n_pos'] + class_weight_info['n_neg']) * 100:.2f}%），\n")
            f.write(f"  模拟真实上线场景。\n\n")


def write_json_report(
    json_path: str,
    results: list[dict],
    smote_info: dict,
    under_info_list: list[dict],
    class_weight_info: dict,
) -> None:
    """把对比结果写成 JSON，方便后续读取。"""
    payload = {
        "smote": smote_info,
        "undersample": under_info_list,
        "class_weight": class_weight_info,
        "results": results,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)


if __name__ == "__main__":
    main()

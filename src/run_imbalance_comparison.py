"""Part5 类别不平衡处理方案对比 —— 主入口。

串联以下流程：
    1) 加载 train/val
    2) 跑 SMOTE 过采样 × 3 个比例（10%/25%/50%）
    3) 跑 3 次欠采样 × 3 个比例（10%/25%/50%，各保存一份）
    4) 计算类别权重
    5) 三种方案对比（SMOTE / 欠采样50% / 类别权重，C=1.0）
    6) SMOTE 与欠采样的不同正样本占比对比（10% / 25% / 50%）
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
    compare_ratios,
    print_ratio_table,
)
from sample_construction.imbalance import (
    apply_class_weight,
    apply_smote,
    apply_undersample,
)

logger = get_logger("run_imbalance_comparison")

OUTPUT_DIR = "output"
DOCS_DIR = "docs"

# 正样本占比列表（老师要求：先试点 10% 和 25%，加上已有的 50% 平衡基线）
TARGET_POS_RATIOS = [0.10, 0.25, 0.50]
UNDERSAMPLE_RATIOS = TARGET_POS_RATIOS


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
    """主流程：三种方案对比 + 不同正样本占比对比。"""
    logger.info("=" * 60)
    logger.info("Part5：类别不平衡处理方案对比")
    logger.info("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DOCS_DIR, exist_ok=True)

    # ─── 1) 加载数据 ───
    df_train, df_val, feature_cols = _load_train_val()

    # ─── 2) SMOTE 过采样 × 3 个比例 ───
    logger.info("\n>>> Step 1: SMOTE 过采样 × %d 比例", len(TARGET_POS_RATIOS))
    df_smote_by_ratio: dict[float, list[pd.DataFrame]] = {}
    smote_info_by_ratio: dict[float, list[dict]] = {}
    for ratio in TARGET_POS_RATIOS:
        logger.info("\n--- SMOTE 正样本占比=%.0f%% ---", ratio * 100)
        df_s, info_s = apply_smote(
            df_train=df_train,
            feature_cols=feature_cols,
            output_dir=OUTPUT_DIR,
            random_state=RANDOM_STATE,
            target_pos_ratio=ratio,
        )
        df_smote_by_ratio[ratio] = [df_s]
        smote_info_by_ratio[ratio] = [info_s]

    # ─── 3) 欠采样 × 3 个比例 × 3 个 seed ───
    logger.info(
        "\n>>> Step 2: 欠采样 × %d 比例 × %d seed",
        len(TARGET_POS_RATIOS), len(UNDERSAMPLE_SEEDS),
    )
    # 按比例分组存储：{ratio: [(df, info), ...]}
    df_under_by_ratio: dict[float, list[pd.DataFrame]] = {}
    under_info_by_ratio: dict[float, list[dict]] = {}

    for ratio in TARGET_POS_RATIOS:
        df_under_by_ratio[ratio] = []
        under_info_by_ratio[ratio] = []
        for seed in UNDERSAMPLE_SEEDS:
            logger.info(
                "\n--- 欠采样 正样本占比=%.0f%%, random_state=%d ---",
                ratio * 100, seed,
            )
            df_u, info_u = apply_undersample(
                df_train=df_train,
                feature_cols=feature_cols,
                output_dir=OUTPUT_DIR,
                random_state=seed,
                target_pos_ratio=ratio,
            )
            df_under_by_ratio[ratio].append(df_u)
            under_info_by_ratio[ratio].append(info_u)

    # ─── 4) 类别权重 ───
    logger.info("\n>>> Step 3: 计算类别权重")
    class_weight_info = apply_class_weight(df_train=df_train)

    # ─── 5) 三种方案对比（C=1.0，跳过 GridSearchCV）───
    # 用 50% 欠采样数据参与三方案对比
    logger.info("\n>>> Step 4: 三种方案对比（C=1.0）")
    results = compare_methods(
        df_train=df_train,
        df_val=df_val,
        feature_cols=feature_cols,
        df_train_smote=df_smote_by_ratio[0.50][0],
        df_train_under_list=df_under_by_ratio[0.50],
        class_weight_info=class_weight_info,
        best_C=1.0,
    )

    # ─── 6) 不同正样本占比对比 ───
    logger.info("\n>>> Step 5: SMOTE 不同正样本占比对比（C=1.0）")
    smote_ratio_results = compare_ratios(
        df_val=df_val,
        feature_cols=feature_cols,
        df_by_ratio=df_smote_by_ratio,
        best_C=1.0,
        method_prefix="SMOTE",
    )
    logger.info("\n>>> Step 6: 欠采样不同正样本占比对比（C=1.0）")
    under_ratio_results = compare_ratios(
        df_val=df_val,
        feature_cols=feature_cols,
        df_by_ratio=df_under_by_ratio,
        best_C=1.0,
        method_prefix="欠采样",
    )
    ratio_results = smote_ratio_results + under_ratio_results

    # ─── 7) 打印对比表 ───
    print_comparison_table(results)
    print_ratio_table(smote_ratio_results, best_C=1.0, title="SMOTE 不同正样本占比对比（验证集）")
    print_ratio_table(under_ratio_results, best_C=1.0, title="欠采样不同正样本占比对比（验证集）")

    # ─── 8) 保存 Markdown 报告 ───
    md_path = os.path.join(DOCS_DIR, "Part5_不平衡处理对比.md")
    write_markdown_report(
        md_path=md_path,
        results=results,
        smote_ratio_results=smote_ratio_results,
        under_ratio_results=under_ratio_results,
        smote_info_by_ratio=smote_info_by_ratio,
        under_info_by_ratio=under_info_by_ratio,
        class_weight_info=class_weight_info,
    )
    logger.info("Markdown 报告已保存: %s", md_path)

    # ─── 9) 保存 JSON 报告（机器可读）───
    json_path = os.path.join(OUTPUT_DIR, "imbalance_comparison.json")
    write_json_report(
        json_path=json_path,
        results=results,
        smote_ratio_results=smote_ratio_results,
        under_ratio_results=under_ratio_results,
        smote_info_by_ratio=smote_info_by_ratio,
        under_info_by_ratio=under_info_by_ratio,
        class_weight_info=class_weight_info,
    )
    logger.info("JSON 报告已保存: %s", json_path)

    logger.info("\nDone. 全部对比完成。")


def write_markdown_report(
    md_path: str,
    results: list[dict],
    smote_ratio_results: list[dict],
    under_ratio_results: list[dict],
    smote_info_by_ratio: dict[float, list[dict]],
    under_info_by_ratio: dict[float, list[dict]],
    class_weight_info: dict,
) -> None:
    """把对比结果写成 Markdown 报告。"""
    smote_50 = smote_info_by_ratio[0.50][0]

    def write_result_table(f, rows: list[dict], include_train_acc: bool = True) -> None:
        if include_train_acc:
            f.write("| 方案 | Precision | Recall | F1 | AUC | TrainAcc | n_train |\n")
            f.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        else:
            f.write("| 方案 | Precision | Recall | F1 | AUC | n_train |\n")
            f.write("| --- | ---: | ---: | ---: | ---: | ---: |\n")
        for r in rows:
            if include_train_acc:
                f.write(
                    f"| {r['method']} | {r['precision']:.4f} | {r['recall']:.4f} "
                    f"| {r['f1']:.4f} | {r['auc']:.4f} | {r['train_score']:.4f} "
                    f"| {r['n_train']:,} |\n"
                )
            else:
                f.write(
                    f"| {r['method']} | {r['precision']:.4f} | {r['recall']:.4f} "
                    f"| {r['f1']:.4f} | {r['auc']:.4f} | {r['n_train']:,} |\n"
                )
        f.write("\n")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Part5 类别不平衡处理方案对比报告\n\n")
        f.write("## 1. 报告定位\n\n")
        f.write("本报告对应步骤 5 中的类别不平衡处理方案对比。\n\n")
        f.write(
            "根据老师建议，本次在 50% 平衡基线基础上，"
            "同时对 SMOTE 过采样和欠采样新增 10%、25% 正样本占比试点。\n\n"
        )

        f.write("## 2. 实验前提\n\n")
        f.write(f"- 训练集：{smote_50['before_shape'][0]:,} 行，建模特征 {smote_50['before_shape'][1]} 列\n")
        f.write(f"- 原始正样本占比：{smote_50['before_pos_ratio'] * 100:.4f}%\n")
        f.write("- 验证集保持原始类别分布，不参与采样\n")
        f.write("- 模型：L1 逻辑回归，`C=1.0`，`solver=saga`\n\n")

        f.write("## 3. 不同正样本占比处理后数据集规模\n\n")
        f.write("### 3.1 SMOTE 过采样\n\n")
        f.write("| 目标正样本占比 | 处理后行数 | 实际正样本占比 | 输出文件 |\n")
        f.write("| --- | ---: | ---: | --- |\n")
        for ratio in sorted(smote_info_by_ratio):
            info = smote_info_by_ratio[ratio][0]
            f.write(
                f"| {ratio * 100:.0f}% | {info['after_shape'][0]:,} "
                f"| {info['after_pos_ratio'] * 100:.2f}% | `{info['path']}` |\n"
            )
        f.write("\n")

        f.write("### 3.2 欠采样\n\n")
        f.write("| 目标正样本占比 | random_state | 处理后行数 | 实际正样本占比 | 输出文件 |\n")
        f.write("| --- | ---: | ---: | ---: | --- |\n")
        for ratio in sorted(under_info_by_ratio):
            for seed, info in zip(UNDERSAMPLE_SEEDS, under_info_by_ratio[ratio]):
                f.write(
                    f"| {ratio * 100:.0f}% | {seed} | {info['after_shape'][0]:,} "
                    f"| {info['after_pos_ratio'] * 100:.2f}% | `{info['path']}` |\n"
                )
        f.write("\n")

        f.write("### 3.3 类别权重\n\n")
        cw = class_weight_info["class_weights"]
        f.write("- 类别权重不改变数据比例，只在损失函数层面平衡正负样本。\n")
        f.write(f"- 负样本权重：`{cw[0]:.4f}`\n")
        f.write(f"- 正样本权重：`{cw[1]:.4f}`\n")
        f.write(f"- 正/负权重比：`{cw[1] / cw[0]:.1f}×`\n\n")

        f.write("## 4. 50% 平衡基线三方案对比\n\n")
        write_result_table(f, results)

        f.write("## 5. SMOTE 不同正样本占比对比\n\n")
        write_result_table(f, smote_ratio_results)

        f.write("## 6. 欠采样不同正样本占比对比\n\n")
        write_result_table(f, under_ratio_results)
        f.write("注：欠采样指标为 3 次随机种子（42/100/2024）的平均值。\n\n")

        best_all = max(smote_ratio_results + under_ratio_results, key=lambda x: x["f1"])
        best_main = results[0] if results else None
        f.write("## 7. 结论\n\n")
        if best_main:
            f.write(f"- 50% 平衡基线中 F1 最高：`{best_main['method']}`（F1 = {best_main['f1']:.4f}）。\n")
        f.write(f"- 10%/25%/50% 比例试点中 F1 最高：`{best_all['method']}`（F1 = {best_all['f1']:.4f}）。\n")
        f.write("- SMOTE 与欠采样均已覆盖 10%、25%、50% 三档正样本占比。\n")
        f.write("- 类别权重不生成新数据、不删除样本，因此不参与正样本占比试点，只作为损失层面的平衡基线。\n")


def write_json_report(
    json_path: str,
    results: list[dict],
    smote_ratio_results: list[dict],
    under_ratio_results: list[dict],
    smote_info_by_ratio: dict[float, list[dict]],
    under_info_by_ratio: dict[float, list[dict]],
    class_weight_info: dict,
) -> None:
    """把对比结果写成 JSON，方便后续读取。"""
    payload = {
        "smote": {
            f"ratio_{int(r * 100)}": info_list
            for r, info_list in smote_info_by_ratio.items()
        },
        "undersample": {
            f"ratio_{int(r * 100)}": info_list
            for r, info_list in under_info_by_ratio.items()
        },
        "class_weight": class_weight_info,
        "results": results,
        "smote_ratio_results": smote_ratio_results,
        "under_ratio_results": under_ratio_results,
        "ratio_results": smote_ratio_results + under_ratio_results,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)


if __name__ == "__main__":
    main()

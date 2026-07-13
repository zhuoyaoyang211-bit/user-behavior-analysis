"""L1 逻辑回归特征重要性评估入口。

输入：output/selected_features.parquet（三轮筛选后特征集）
输出：控制台打印特征重要性排行榜 + 与三轮筛选交叉验证结果
"""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common.logger import get_logger
from config import get_config
from feature_selection.evaluator import evaluate_features

logger = get_logger(__name__)

_cfg = get_config()
OUTPUT_DIR = str(_cfg.project_root / "output")
INPUT_PATH = os.path.join(OUTPUT_DIR, "selected_features.parquet")


def _print_report(summary: dict, mi_scores: pd.Series | None) -> None:
    """打印特征重要性评估报告。"""
    coefs = summary["coefs"]
    zero_coef_cols = summary["zero_coef_cols"]

    print("\n" + "=" * 70)
    print("L1 逻辑回归特征重要性评估")
    print("=" * 70)

    print(f"\n  模型: LogisticRegression(penalty=L1, C={summary['l1_c']})")
    print(f"  目标: 二分类 (没买 vs 买了)")
    print(f"  训练集准确率: {summary['train_score']:.4f}")

    # --- 特征重要性排行榜 ---
    n_total = len(coefs)
    n_top = min(15, n_total)
    print(f"\n  ┌─ 特征重要性 Top {n_top} (L1 系数绝对值) ───────────────────────")
    for rank, (col, coef_val) in enumerate(coefs.head(n_top).items(), 1):
        mi_str = f"MI={mi_scores[col]:.4f}" if mi_scores is not None else ""
        print(f"  │ {rank:2d}. {col:<38s} |coef|={coef_val:.4f}  {mi_str}")
    print(f"  └{'─' * 52}")

    if len(coefs) > n_top:
        print(f"\n  （共 {n_total} 列，仅展示 Top {n_top}）")

    # --- L1 压成 0 的特征 ---
    if zero_coef_cols:
        print(f"\n  L1 系数被压成 0 ({len(zero_coef_cols)} 列):")
        for c in zero_coef_cols:
            mi_str = f"(MI={mi_scores[c]:.4f})" if mi_scores is not None else ""
            print(f"    - {c} {mi_str}")
    else:
        print(f"\n  L1 系数未压成 0: 所有 {n_total} 列系数均 > 0")

    # --- 交叉验证：三轮筛选 vs L1 ---
    if mi_scores is not None:
        mi_rank = mi_scores.sort_values(ascending=False)
        l1_rank = coefs

        # 计算两个排名的重合度（Top N 的交集）
        for k, label in [(5, "Top 5"), (10, "Top 10"), (15, "Top 15")]:
            mi_top = set(mi_rank.head(k).index)
            l1_top = set(l1_rank.head(k).index)
            overlap = mi_top & l1_top
            print(f"  {label} 重合: {len(overlap)}/{k} → {sorted(overlap)}")

    print("=" * 70)


def main() -> None:
    """L1 特征重要性评估主流程。"""
    logger.info("加载筛选后特征: %s", INPUT_PATH)
    df = pd.read_parquet(INPUT_PATH)
    logger.info("输入: %s 行 × %s 列", f"{len(df):,}", len(df.columns))

    # L1 评估
    summary = evaluate_features(df)

    # 读取三轮筛选的互信息分数，用于交叉对比
    mi_path = os.path.join(OUTPUT_DIR, "mi_scores.csv")
    mi_scores = None
    if os.path.exists(mi_path):
        mi_scores = pd.read_csv(mi_path, index_col=0, header=None).iloc[:, 0]
        mi_scores.name = "mi"
        logger.info("已加载互信息分数: %d 列", len(mi_scores))

    _print_report(summary, mi_scores)


if __name__ == "__main__":
    main()

"""特征筛选主入口。

输入：output/processed_features.parquet（Part4，4,686,904 行 × 46 列）
输出：output/selected_features.parquet（筛选后的特征集）

筛选流程：
    方差阈值法（删方差 < 0.01）→ 互信息法（删后 25%）→ 相关性分析（删 |r| > 0.95）
"""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common.logger import get_logger
from config import get_config
from feature_selection.selector import select_features

logger = get_logger(__name__)

_cfg = get_config()
OUTPUT_DIR = str(_cfg.project_root / "output")
INPUT_PATH = os.path.join(OUTPUT_DIR, "processed_features.parquet")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "selected_features.parquet")


def _print_summary(df: pd.DataFrame, summary: dict) -> None:
    """打印筛选摘要到控制台。"""
    mi_scores = summary["mi_scores"]

    print("\n" + "=" * 65)
    print("特征筛选完成")
    print("=" * 65)

    # 总体
    print(f"\n  输入: {INPUT_PATH}")
    print(f"  输出: {OUTPUT_PATH}")
    print(f"  行数: {len(df):,}")
    print(f"  特征列: {summary['initial']} → {summary['final_count']}")

    # 方差阈值
    dropped_vt = summary["variance_dropped"]
    print(f"\n  [第1轮] 方差阈值法 (threshold={0.01})")
    if dropped_vt:
        for c in dropped_vt:
            var = df[c].var() if c in df.columns else float("nan")
            print(f"    删除: {c:<35s} 方差={var:.6f}")
    else:
        print("    无列被删除")

    # 互信息
    dropped_mi = summary["mi_dropped"]
    print(f"\n  [第2轮] 互信息法 (删后25%)")
    print(f"    阈值: {mi_scores.quantile(0.25):.6f}")
    if dropped_mi:
        for c in dropped_mi:
            print(f"    删除: {c:<35s} MI={mi_scores[c]:.6f}")
    else:
        print("    无列被删除")

    # 相关性
    dropped_corr = summary["correlation_dropped_pairs"]
    print(f"\n  [第3轮] 相关性分析 (threshold=0.95)")
    if dropped_corr:
        for keep, drop, r in dropped_corr:
            print(f"    保留: {keep:<35s}  删除: {drop:<35s}  |r|={r:.4f}")
    else:
        print("    无列被删除")

    # 互信息排名 Top 10
    print(f"\n  [互信息 Top 10]")
    for i, (col, score) in enumerate(mi_scores.head(10).items(), 1):
        marker = "*" if col in summary["final_columns"] else "x"
        print(f"    {i:2d}. [{marker}] {col:<35s} MI={score:.6f}")

    # 最终保留
    print(f"\n  最终保留 {summary['final_count']} 列特征:")
    for col in summary["final_columns"]:
        print(f"    {col}")

    print("=" * 65)


def main() -> None:
    """特征筛选主流程。"""
    logger.info("加载预处理特征: %s", INPUT_PATH)
    df = pd.read_parquet(INPUT_PATH)
    logger.info("输入: %s 行 × %s 列", f"{len(df):,}", len(df.columns))

    logger.info("开始三轮特征筛选 ...")
    selected_df, summary = select_features(df)

    selected_df.to_parquet(OUTPUT_PATH, index=False)
    file_size = os.path.getsize(OUTPUT_PATH) / 1024 / 1024
    logger.info("筛选结果已保存: %s (%.1f MB)", OUTPUT_PATH, file_size)

    # 保存互信息分数，供 L1 评估交叉对比用
    mi_path = os.path.join(OUTPUT_DIR, "mi_scores.csv")
    summary["mi_scores"].to_csv(mi_path)
    logger.info("互信息分数已保存: %s", mi_path)

    _print_summary(selected_df, summary)


if __name__ == "__main__":
    main()

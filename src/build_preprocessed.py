"""特征预处理主入口。

输入：Part3 feature_wide_table.parquet（4,686,904 行 × 47 列）
输出：output/processed_features.parquet（4,686,904 行 × 47 列）

处理流程：删 2 列 → 填缺失 → 目标编码 2 列 → 标准化 42 列 → 保留 5 列不动
"""

from __future__ import annotations

import os
import sys

import pandas as pd

# 让脚本可以 import src.common 和 src.feature_preprocessing
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common.logger import get_logger
from config import get_config
from feature_preprocessing.preprocessor import preprocess


logger = get_logger(__name__)

_cfg = get_config()
OUTPUT_DIR = str(_cfg.project_root / "output")
WIDE_TABLE_PATH = os.path.join(OUTPUT_DIR, "feature_wide_table.parquet")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "processed_features.parquet")


def main() -> None:
    """特征预处理主流程。"""
    # 1. 加载 Part3 特征宽表
    logger.info("加载特征宽表: %s", WIDE_TABLE_PATH)
    df = pd.read_parquet(WIDE_TABLE_PATH)
    logger.info(
        "原始宽表: %s 行 × %s 列", f"{len(df):,}", len(df.columns)
    )

    # 2. 预处理
    logger.info("开始特征预处理 ...")
    df = preprocess(df)

    # 3. 保存
    df.to_parquet(OUTPUT_PATH, index=False)
    file_size = os.path.getsize(OUTPUT_PATH) / 1024 / 1024
    logger.info("预处理结果已保存: %s", OUTPUT_PATH)
    logger.info("输出: %s 行 × %s 列, %.1f MB", f"{len(df):,}", len(df.columns), file_size)

    # 4. 打印摘要
    print("\n" + "=" * 60)
    print("特征预处理完成")
    print("=" * 60)
    print(f"  输入: {WIDE_TABLE_PATH}")
    print(f"  输出: {OUTPUT_PATH}")
    print(f"  行数: {len(df):,}")
    print(f"  列数: {len(df.columns)}")
    print(f"  文件大小: {file_size:.1f} MB")
    print(f"  缺失值: {df.isnull().sum().sum()} 个")
    print("=" * 60)


if __name__ == "__main__":
    main()

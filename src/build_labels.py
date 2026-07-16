"""目标变量定义主入口（Part5 第一步）。

从 cleaned_data 计算每个 (用户, 商品) 对的目标变量：
- last_time: 该对最后一次 1/2/3 行为的时间（用于后续时间窗口划分）
- label: 未来 7 天内是否产生购买行为（1=是, 0=否）

输入：output/cleaned_data.parquet
输出：output/label_table.parquet

标签定义逻辑：
    对每个有过浏览(1)/收藏(2)/加购(3)行为的 (用户,商品) 对：
    - 取最后一次 1/2/3 行为的时间作为 last_time
    - 检查 (last_time, last_time + 7天] 窗口内是否有购买(4)行为
    - 有 → label=1（正样本），没有 → label=0（负样本）

注意：
    只有 1/2/3 行为、没有 4 行为的对：label=0（这部分是多数，~97%）
    仅有 4 行为、没有 1/2/3 行为的对：不纳入样本（无法定义 last_time）
"""

from __future__ import annotations

import os
import sys

import pandas as pd

# 让脚本可以 import src.common 和 config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common.logger import get_logger
from config import get_config

logger = get_logger(__name__)

_cfg = get_config()
OUTPUT_DIR = str(_cfg.project_root / "output")
CLEANED_DATA_PATH = os.path.join(OUTPUT_DIR, "cleaned_data.parquet")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "label_table.parquet")

# 常量
INTERACT_TYPES = frozenset({1, 2, 3})  # 浏览/收藏/加购
BUY_TYPE = 4  # 购买
WINDOW_DAYS = 7  # 未来 7 天窗口


def build_labels(df: pd.DataFrame) -> pd.DataFrame:
    """从原始行为数据构建(用户,商品)对的标签。

    两步走：
        1. 对有过 1/2/3 的 (user_id, item_id)，取 max(time) → last_time
        2. 检查是否有购买记录落在 (last_time, last_time + 7天] 窗口内

    Args:
        df: cleaned_data，含 time / user_id / item_id / behavior_type

    Returns:
        DataFrame，列：user_id, item_id, last_time, label
    """
    # ── 第 1 步：计算每对的 last_time ──────────────────────
    logger.info("计算 last_time（最后一次 1/2/3 行为时间）...")
    interact_mask = df["behavior_type"].isin(INTERACT_TYPES)
    last_time_df = (
        df.loc[interact_mask]
        .groupby(["user_id", "item_id"], as_index=False)["time"]
        .max()
        .rename(columns={"time": "last_time"})
    )
    logger.info("有 1/2/3 行为的用户-商品对: %s", f"{len(last_time_df):,}")

    # ── 第 2 步：取出所有购买记录 ─────────────────────────
    logger.info("提取购买记录...")
    buy_df = df.loc[df["behavior_type"] == BUY_TYPE, ["user_id", "item_id", "time"]].copy()
    buy_df = buy_df.rename(columns={"time": "buy_time"})
    logger.info("购买记录数: %s", f"{len(buy_df):,}")

    # ── 第 3 步：Left Join 关联购买记录 ──────────────────
    logger.info("关联购买记录，判断 7 天窗口...")
    merged = last_time_df.merge(buy_df, on=["user_id", "item_id"], how="left")

    # ── 第 4 步：窗口判断 ────────────────────────────────
    # 条件：last_time < buy_time <= last_time + 7天
    window_end = merged["last_time"] + pd.Timedelta(days=WINDOW_DAYS)
    in_window = (
        merged["buy_time"].notna()
        & (merged["buy_time"] > merged["last_time"])
        & (merged["buy_time"] <= window_end)
    )

    # ── 第 5 步：聚合成标签 ───────────────────────────────
    # 一个对只要有一条购买记录在窗口内 → label=1
    merged["in_window"] = in_window
    label_df = (
        merged.groupby(["user_id", "item_id"], as_index=False)
        .agg(
            last_time=("last_time", "first"),
            label=("in_window", "any"),
        )
    )
    label_df["label"] = label_df["label"].astype("int8")

    # 统计
    n_pos = int(label_df["label"].sum())
    n_total = len(label_df)
    logger.info(
        "标签分布: 正样本 %s (%.2f%%) / 负样本 %s (%.2f%%)",
        f"{n_pos:,}",
        n_pos / n_total * 100,
        f"{n_total - n_pos:,}",
        (n_total - n_pos) / n_total * 100,
    )

    return label_df


def main() -> None:
    """目标变量定义主流程。"""
    # 1. 加载清洗后数据
    logger.info("加载 cleaned_data: %s", CLEANED_DATA_PATH)
    df = pd.read_parquet(CLEANED_DATA_PATH)
    logger.info("原始数据: %s 行 × %s 列", f"{len(df):,}", len(df.columns))

    # 2. 构建标签
    label_df = build_labels(df)

    # 3. 保存
    label_df.to_parquet(OUTPUT_PATH, index=False)
    file_size = os.path.getsize(OUTPUT_PATH) / 1024 / 1024
    logger.info("标签表已保存: %s", OUTPUT_PATH)

    # 4. 打印摘要
    n_pos = int(label_df["label"].sum())
    n_total = len(label_df)
    print("\n" + "=" * 60)
    print("目标变量定义完成")
    print("=" * 60)
    print(f"  输入: {CLEANED_DATA_PATH}")
    print(f"  输出: {OUTPUT_PATH}")
    print(f"  行数: {n_total:,}")
    print(f"  列数: {len(label_df.columns)}")
    print(f"  文件大小: {file_size:.1f} MB")
    print(f"  正样本(label=1): {n_pos:,} ({n_pos / n_total * 100:.2f}%)")
    print(f"  负样本(label=0): {n_total - n_pos:,} ({(n_total - n_pos) / n_total * 100:.2f}%)")
    print("=" * 60)


if __name__ == "__main__":
    main()

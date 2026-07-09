"""特征宽表构建主入口。

输入：Part1 cleaned_data.parquet + Part2 6张中间表
输出：output/feature_wide_table.parquet

主键范围：全量(用户,商品)对——所有有过任意行为(浏览/收藏/加购/购买)的对。
"""

from __future__ import annotations

import gc
import os
import sys

import numpy as np
import pandas as pd

# 让脚本可以 import src.common 和 src.feature_engineering
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common.logger import get_logger
from config import get_config
from feature_engineering.business_feature import calc_rfm_scores, calc_user_category_pref
from feature_engineering.lifecycle_feature import (
    calc_buy_path_type,
    calc_item_decay_slope,
    calc_user_avg_interval_hours,
    calc_user_streak_days,
)
from feature_engineering.semantic_feature import calc_user_item_svd_score


logger = get_logger(__name__)

_cfg = get_config()
OUTPUT_DIR = str(_cfg.project_root / "output")
PROJECT_ROOT = str(_cfg.project_root)
WIDE_TABLE_PATH = os.path.join(OUTPUT_DIR, "feature_wide_table.parquet")


def _downcast_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """dtype 降级：int64→int32, float64→float32, 省一半内存。"""
    for col in df.columns:
        dt = df[col].dtype
        if pd.api.types.is_integer_dtype(dt):
            df[col] = pd.to_numeric(df[col], downcast="integer")
        elif pd.api.types.is_float_dtype(dt):
            df[col] = pd.to_numeric(df[col], downcast="float")
    return df


def load_inputs() -> dict:
    """加载所有输入数据。"""
    logger.info("加载 Part1 cleaned_data + Part2 中间表")
    return {
        "cleaned": pd.read_parquet(os.path.join(OUTPUT_DIR, "cleaned_data.parquet")),
        "dim_user": pd.read_parquet(os.path.join(OUTPUT_DIR, "dim_user.parquet")),
        "dim_item": pd.read_parquet(os.path.join(OUTPUT_DIR, "dim_item.parquet")),
        "dim_category": pd.read_parquet(
            os.path.join(OUTPUT_DIR, "dim_category.parquet")
        ),
    }


def build_target_pairs(cleaned: pd.DataFrame) -> pd.DataFrame:
    """构造主键：全量(用户,商品)对——所有有过任意行为的对。"""
    pairs = (
        cleaned[["user_id", "item_id", "item_category"]]
        .drop_duplicates(subset=["user_id", "item_id"])
        .reset_index(drop=True)
    )
    logger.info("全量(用户,商品)对数量: %s", f"{len(pairs):,}")
    return pairs


def build_lifecycle_features(cleaned: pd.DataFrame) -> dict:
    """算 4 个生命周期特征。

    返回 dict，value 可能是 Series（用户级/商品级）或 DataFrame（(用户,商品)对级）。
    """
    logger.info("计算生命周期特征 ...")
    return {
        "user_streak_days": calc_user_streak_days(cleaned),
        "item_decay_slope": calc_item_decay_slope(cleaned),
        "user_avg_interval_hours": calc_user_avg_interval_hours(cleaned),
        "buy_path_type": calc_buy_path_type(cleaned),
    }


def build_business_features(cleaned: pd.DataFrame, dim_user: pd.DataFrame) -> dict:
    """算 RFM 三档 + 类目偏好。"""
    logger.info("计算业务导向特征 ...")
    rfm = calc_rfm_scores(dim_user, cleaned)
    cat_pref = calc_user_category_pref(cleaned)
    return {"rfm": rfm, "cat_pref": cat_pref}


def assemble_wide_table(
    target_pairs: pd.DataFrame,
    inputs: dict,
    lifecycle: dict,
    business: dict,
    svd_scores: pd.DataFrame,
) -> pd.DataFrame:
    """拼接宽表（分步 merge + dtype 降级 + 及时 gc）。"""
    logger.info("开始拼接宽表 ...")
    wide = target_pairs.copy()

    # --- 1. 商品维度 (dim_item 预过滤：只取宽表需要的 item_id) ---
    needed_items = wide["item_id"].unique()
    dim_item = inputs["dim_item"].reset_index()
    dim_item = dim_item[dim_item["item_id"].isin(needed_items)][
        ["item_id", "pv_count", "view_user_count",
         "fav_count", "cart_count", "buy_count", "buy_user_count",
         "pv_to_buy_rate", "cart_to_buy_rate", "repurchase_user_count"]
    ].copy()
    dim_item.columns = [f"item_{c}" if c != "item_id" else c
                        for c in dim_item.columns]
    dim_item = _downcast_dtypes(dim_item)
    wide = wide.merge(dim_item, on="item_id", how="left")
    del dim_item
    gc.collect()
    logger.info("  商品维度拼接完成: %s 行", f"{len(wide):,}")

    # --- 2. 类目维度 (dim_category) ---
    dim_category = inputs["dim_category"].reset_index()[
        ["item_category", "item_count", "buy_item_count",
         "pv_count", "view_user_count", "fav_count", "cart_count",
         "buy_count", "buy_user_count", "pv_to_buy_rate", "buy_item_pct"]
    ].copy()
    dim_category.columns = [f"cat_{c}" if c != "item_category" else c
                            for c in dim_category.columns]
    dim_category = _downcast_dtypes(dim_category)
    wide = wide.merge(dim_category, on="item_category", how="left")
    del dim_category
    gc.collect()
    logger.info("  类目维度拼接完成")

    # --- 3. 用户维度 (dim_user) ---
    dim_user_cols = [
        "user_id", "pv_count", "fav_count", "cart_count", "buy_count",
        "active_days", "first_active_time", "last_active_time",
        "day_pct", "evening_pct", "night_pct",
        "buy_item_count", "buy_conversion_rate", "fav_to_buy_rate",
        "cart_to_buy_rate", "repurchase_item_count", "is_power_user",
    ]
    dim_user = inputs["dim_user"].reset_index()[dim_user_cols].copy()
    dim_user = _downcast_dtypes(dim_user)
    wide = wide.merge(dim_user, on="user_id", how="left")
    # 解决重名列：买/浏览在用户表和商品表都有
    for col in ["pv_count", "fav_count", "cart_count", "buy_count"]:
        if col in wide.columns and f"item_{col}" in wide.columns:
            wide = wide.rename(columns={col: f"user_{col}"})
    del dim_user
    gc.collect()
    logger.info("  用户维度拼接完成")

    # --- 4. 生命周期特征 ---
    # 用户级/商品级用 map，(用户,商品)对级用 merge
    if "user_streak_days" in lifecycle:
        wide["user_streak_days"] = wide["user_id"].map(
            lifecycle["user_streak_days"]
        )
    if "item_decay_slope" in lifecycle:
        wide["item_decay_slope"] = wide["item_id"].map(
            lifecycle["item_decay_slope"]
        )
    if "user_avg_interval_hours" in lifecycle:
        wide["user_avg_interval_hours"] = wide["user_id"].map(
            lifecycle["user_avg_interval_hours"]
        )
    if "buy_path_type" in lifecycle:
        # buy_path_type 是 (用户,商品) 对级，用 merge
        wide = wide.merge(
            lifecycle["buy_path_type"], on=["user_id", "item_id"], how="left"
        )
    logger.info("  生命周期特征拼接完成")

    # --- 5. RFM 得分（用户级） ---
    wide = wide.merge(business["rfm"], on="user_id", how="left")

    # --- 6. 用户-类目偏好（(用户,类目)级） ---
    wide = wide.merge(
        business["cat_pref"], on=["user_id", "item_category"], how="left"
    )
    gc.collect()
    logger.info("  业务特征拼接完成")

    # --- 7. SVD 匹配分（(用户,商品)对级） ---
    wide = wide.merge(svd_scores, on=["user_id", "item_id"], how="left")
    del svd_scores
    gc.collect()

    # 全表 dtype 降级
    wide = _downcast_dtypes(wide)

    logger.info(
        "宽表拼接完成: %s 行 × %s 列", f"{len(wide):,}", len(wide.columns)
    )
    return wide


def main() -> None:
    inputs = load_inputs()
    target_pairs = build_target_pairs(inputs["cleaned"])

    lifecycle = build_lifecycle_features(inputs["cleaned"])
    business = build_business_features(inputs["cleaned"], inputs["dim_user"])

    logger.info("计算 SVD 隐语义特征 ...")
    svd_scores = calc_user_item_svd_score(inputs["cleaned"], target_pairs)

    wide = assemble_wide_table(
        target_pairs, inputs, lifecycle, business, svd_scores
    )

    wide.to_parquet(WIDE_TABLE_PATH, index=False)
    logger.info("宽表已保存: %s", WIDE_TABLE_PATH)
    logger.info(
        "宽表大小: %s MB",
        round(os.path.getsize(WIDE_TABLE_PATH) / 1024 / 1024, 2),
    )


if __name__ == "__main__":
    main()

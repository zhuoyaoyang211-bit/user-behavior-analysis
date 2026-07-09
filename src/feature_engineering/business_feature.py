"""业务导向特征模块。

输出字段（4个）：
    rfm_r_score : RFM 的 R 值得分（1-5）
    rfm_f_score : RFM 的 F 值得分（1-5）
    rfm_m_score : RFM 的 M 值得分（1-5，用购买商品种类数代替金额）
    user_category_pref_score : 用户对所买类目的偏好强度（每个(用户,类目)对 1 个）
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# RFM 分档阈值（基于历史数据的分位数）
# R（最近购买天数）：距 12-18 越近分越高
RFM_R_BINS = [-1, 7, 14, 21, 28, np.inf]   # 5 档（-1 确保 r_days=0 落入第一档）
# F（购买次数）：越多分越高
RFM_F_BINS = [0, 2, 5, 10, 20, np.inf]     # 5 档
# M（购买商品种类数）：越多分越高
RFM_M_BINS = [0, 1, 3, 6, 10, np.inf]      # 5 档

RFM_OBSERVE_DATE = pd.Timestamp("2025-12-19")  # 观察日=数据最后一天(12-18)的结束时刻


def calc_rfm_scores(
    dim_user: pd.DataFrame, cleaned: pd.DataFrame
) -> pd.DataFrame:
    """从 dim_user + 原始明细计算 R/F/M 三值的 1-5 档得分。

    Args:
        dim_user: 步骤二 dim_user 中间表，需含 buy_count、buy_item_count 等列。
        cleaned: 原始明细数据，用于计算最后购买时间（R 值）。

    Returns:
        DataFrame, 列 = [user_id, rfm_r_score, rfm_f_score, rfm_m_score]
    """
    dim_user = dim_user.reset_index()
    user = dim_user[["user_id"]].copy()

    # R 值：最后一次购买距 12-18 的天数（用最后购买时间，不是最后行为时间）
    buy = cleaned[cleaned["behavior_type"] == 4]
    last_buy = buy.groupby("user_id")["time"].max()
    # 对齐到 dim_user 的 user_id
    last_buy = last_buy.reindex(dim_user["user_id"])
    r_days = (RFM_OBSERVE_DATE - last_buy).dt.days.values  # 没买过的为 NaN
    # 翻转方向：距离越近分越高（≤7天=5分，>28天=1分）
    # 没买过的 r_days=NaN → nan_to_num(nan=4) → 5-4=1（最差）
    r_score = pd.cut(r_days, bins=RFM_R_BINS, labels=False, right=True)
    user["rfm_r_score"] = 5 - np.nan_to_num(r_score, nan=4).astype(int)

    # F 值：购买次数
    f_score = pd.cut(
        dim_user["buy_count"].values, bins=RFM_F_BINS, labels=False, right=True
    )
    user["rfm_f_score"] = np.nan_to_num(f_score, nan=0).astype(int) + 1

    # M 值：购买商品种类数
    m_score = pd.cut(
        dim_user["buy_item_count"].values, bins=RFM_M_BINS,
        labels=False, right=True,
    )
    user["rfm_m_score"] = np.nan_to_num(m_score, nan=0).astype(int) + 1

    return user[["user_id", "rfm_r_score", "rfm_f_score", "rfm_m_score"]]


def calc_user_category_pref(df: pd.DataFrame) -> pd.DataFrame:
    """计算用户对所买类目的偏好强度。

    score = 用户在该类目的购买次数 / 用户总购买次数。
    输出每个(用户,类目)对一行，没买过任何东西的用户无记录。

    Args:
        df: 原始明细数据。

    Returns:
        DataFrame, 列 = [user_id, item_category, user_category_pref_score]
    """
    buy = df[df["behavior_type"] == 4]
    # 用户在各品类的购买次数
    user_cat_count = buy.groupby(["user_id", "item_category"]).size().reset_index(
        name="cat_buy_count"
    )
    # 用户总购买次数
    user_total = buy.groupby("user_id").size().reset_index(name="total_buy_count")
    result = user_cat_count.merge(user_total, on="user_id")
    result["user_category_pref_score"] = (
        result["cat_buy_count"] / result["total_buy_count"]
    ).astype("float32")
    return result[["user_id", "item_category", "user_category_pref_score"]]

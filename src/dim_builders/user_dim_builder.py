"""用户维度中间表构建器。

将清洗后的行为流水表聚合为用户维度宽表（dim_user），
每行对应一个用户，包含行为量、活跃度、转化能力、标签四类指标。

输入: cleaned_data.parquet 的 DataFrame（6列，含 is_power_user）
输出: dim_user DataFrame（17列，1万行）
"""

import pandas as pd

from common.agg_specs import (
    DAY_HOURS,
    EVENING_HOURS,
    NIGHT_HOURS,
)
from common.constants import (
    BEHAVIOR_BUY,
    BEHAVIOR_CART,
    BEHAVIOR_FAV,
    BEHAVIOR_PV,
)
from common.logger import get_logger

logger = get_logger("user_dim_builder")


def build_user_dim(df: pd.DataFrame) -> pd.DataFrame:
    """构建用户维度中间表。

    聚合逻辑分4组:
        1. 行为量(pv口径): 浏览/收藏/加购/购买 各自的总次数
        2. 活跃度: 活跃天数、首末次行为时间、时段偏好占比
        3. 转化能力: 购买商品数(去重)、各环节转化率、复购商品数
        4. 标签: is_power_user（直接复用Part1标记，不重算）

    Args:
        df: 清洗后的行为流水表，需包含列:
            user_id, item_id, behavior_type, time, is_power_user

    Returns:
        用户维度宽表，每行一个用户，17列。

    Note:
        - 转化率(漏斗口径): 用户做过上游行为的商品中，有多少最终被买了。
          分子是分母的子集，转化率一定在 [0, 1]，分母为0时填0。
        - 活跃天数定义: 有任意行为(浏览/收藏/加购/购买)的不同日期数
        - 时段占比: 白天6-17 / 晚间18-23 / 深夜0-5，三者之和=1
    """
    logger.info("开始构建 dim_user，输入行数: %d", len(df))

    # --- 第1组: 行为量(pv口径) ---
    # 按 user_id 分组，统计每种行为的总次数
    behavior_counts = (
        df.groupby("user_id")["behavior_type"]
        .value_counts()
        .unstack(fill_value=0)
    )
    # 确保四种行为列都存在（某些用户可能缺少某种行为）
    for bt in [BEHAVIOR_PV, BEHAVIOR_FAV, BEHAVIOR_CART, BEHAVIOR_BUY]:
        if bt not in behavior_counts.columns:
            behavior_counts[bt] = 0
    behavior_counts = behavior_counts.rename(
        columns={
            BEHAVIOR_PV: "pv_count",
            BEHAVIOR_FAV: "fav_count",
            BEHAVIOR_CART: "cart_count",
            BEHAVIOR_BUY: "buy_count",
        }
    )

    # --- 第2组: 活跃度 ---
    # 活跃天数: 有行为的不同日期数
    df_temp = df.copy()
    df_temp["_date"] = df_temp["time"].dt.date
    active_days = df_temp.groupby("user_id")["_date"].nunique()
    active_days.name = "active_days"

    # 首末次行为时间
    first_active = df.groupby("user_id")["time"].min()
    first_active.name = "first_active_time"
    last_active = df.groupby("user_id")["time"].max()
    last_active.name = "last_active_time"

    # 时段偏好占比
    df_temp["_hour"] = df_temp["time"].dt.hour
    total_actions = df_temp.groupby("user_id").size()
    total_actions.name = "_total"

    def _count_hours(df_in: pd.DataFrame, hours: range) -> pd.Series:
        """统计指定时段的行为次数占比。"""
        mask = df_in["_hour"].isin(hours)
        counts = df_in[mask].groupby("user_id").size()
        return counts

    day_counts = _count_hours(df_temp, DAY_HOURS)
    evening_counts = _count_hours(df_temp, EVENING_HOURS)
    night_counts = _count_hours(df_temp, NIGHT_HOURS)

    # --- 第3组: 转化能力 ---
    # 购买商品数(去重): 只看购买行为，按item_id去重计数
    buy_df = df[df["behavior_type"] == BEHAVIOR_BUY]
    buy_item_count = buy_df.groupby("user_id")["item_id"].nunique()
    buy_item_count.name = "buy_item_count"

    # 复购商品数: 购买次数>=2的不同商品数
    user_item_buy = buy_df.groupby(["user_id", "item_id"]).size()
    repurchase_items = user_item_buy[user_item_buy >= 2]
    repurchase_item_count = repurchase_items.groupby("user_id").size()
    repurchase_item_count.name = "repurchase_item_count"

    # --- 第4组: 标签 ---
    # is_power_user: 直接复用Part1标记（取每个用户任一行的值）
    power_user = df.groupby("user_id")["is_power_user"].any()
    power_user.name = "is_power_user"

    # --- 合并所有指标 ---
    result = behavior_counts.copy()
    result["active_days"] = active_days
    result["first_active_time"] = first_active
    result["last_active_time"] = last_active
    result["day_pct"] = (day_counts / total_actions).fillna(0.0)
    result["evening_pct"] = (evening_counts / total_actions).fillna(0.0)
    result["night_pct"] = (night_counts / total_actions).fillna(0.0)
    result["buy_item_count"] = buy_item_count
    result["repurchase_item_count"] = repurchase_item_count
    result["is_power_user"] = power_user

    # 对齐后fillna: 没有购买行为的用户这些字段为NaN → 填0
    result["buy_item_count"] = result["buy_item_count"].fillna(0).astype("int32")
    result["repurchase_item_count"] = (
        result["repurchase_item_count"].fillna(0).astype("int32")
    )

    # --- 转化率(漏斗口径: 基于商品集合) ---
    # buy_conversion_rate: 用户浏览过的商品中，有多少最终被买了
    #   = |{浏览过的商品} ∩ {买过的商品}| ÷ |{浏览过的商品}|
    # fav_to_buy_rate: 用户收藏过的商品中，有多少最终被买了
    # cart_to_buy_rate: 用户加购过的商品中，有多少最终被买了
    # 分子是分母的子集，转化率一定在 [0, 1]
    pv_pairs = df[df["behavior_type"] == BEHAVIOR_PV][
        ["user_id", "item_id"]
    ].drop_duplicates()
    fav_pairs = df[df["behavior_type"] == BEHAVIOR_FAV][
        ["user_id", "item_id"]
    ].drop_duplicates()
    cart_pairs = df[df["behavior_type"] == BEHAVIOR_CART][
        ["user_id", "item_id"]
    ].drop_duplicates()
    buy_pairs = buy_df[["user_id", "item_id"]].drop_duplicates()

    # 浏览过的商品数(去重) ÷ 浏览且购买的商品数
    pv_item_n = pv_pairs.groupby("user_id")["item_id"].nunique()
    pv_buy_n = pv_pairs.merge(buy_pairs, on=["user_id", "item_id"]).groupby(
        "user_id"
    )["item_id"].nunique()
    result["buy_conversion_rate"] = (pv_buy_n / pv_item_n).reindex(
        result.index
    ).fillna(0.0)

    # 收藏→购买
    fav_item_n = fav_pairs.groupby("user_id")["item_id"].nunique()
    fav_buy_n = fav_pairs.merge(buy_pairs, on=["user_id", "item_id"]).groupby(
        "user_id"
    )["item_id"].nunique()
    result["fav_to_buy_rate"] = (fav_buy_n / fav_item_n).reindex(
        result.index
    ).fillna(0.0)

    # 加购→购买
    cart_item_n = cart_pairs.groupby("user_id")["item_id"].nunique()
    cart_buy_n = cart_pairs.merge(buy_pairs, on=["user_id", "item_id"]).groupby(
        "user_id"
    )["item_id"].nunique()
    result["cart_to_buy_rate"] = (cart_buy_n / cart_item_n).reindex(
        result.index
    ).fillna(0.0)

    # 重置索引，user_id 变回普通列
    result = result.reset_index()

    # 按设计文档列顺序排列
    result = result[
        [
            "user_id",
            "pv_count",
            "fav_count",
            "cart_count",
            "buy_count",
            "active_days",
            "first_active_time",
            "last_active_time",
            "day_pct",
            "evening_pct",
            "night_pct",
            "buy_item_count",
            "buy_conversion_rate",
            "fav_to_buy_rate",
            "cart_to_buy_rate",
            "repurchase_item_count",
            "is_power_user",
        ]
    ]

    logger.info("dim_user 构建完成: %d 行, %d 列", len(result), len(result.columns))
    return result

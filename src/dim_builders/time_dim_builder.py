"""时间维度中间表构建器。

将清洗后的行为流水表聚合为三张不同粒度的时间维度表:
    - dim_time_hourly: 小时级（744行，31天×24小时）
    - dim_time_daily: 日级（31行）
    - dim_time_weekday_hour: 周×小时级（168行，7天×24小时）

三张表字段结构一致，区别只在聚合粒度。
"""

import pandas as pd

from common.constants import BEHAVIOR_BUY, BEHAVIOR_CART, BEHAVIOR_FAV, BEHAVIOR_PV
from common.logger import get_logger

logger = get_logger("time_dim_builder")


def _aggregate_time_metrics(
    df: pd.DataFrame, group_cols: list[str]
) -> pd.DataFrame:
    """按指定维度列聚合时间维度指标。

    统一计算6个指标: 浏览/收藏/加购/购买次数 + 独立活跃用户数 + 独立活跃商品数。

    Args:
        df: 清洗后的行为流水表，需含 _hour/_date/_weekday 辅助列
        group_cols: 分组列名列表

    Returns:
        聚合后的 DataFrame，包含 group_cols + 6个指标列。
    """
    # 行为次数统计
    behavior_counts = (
        df.groupby(group_cols + ["behavior_type"]).size().unstack(fill_value=0)
    )
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

    # 独立活跃用户数
    active_user_count = df.groupby(group_cols)["user_id"].nunique()
    active_user_count.name = "active_user_count"

    # 独立活跃商品数
    active_item_count = df.groupby(group_cols)["item_id"].nunique()
    active_item_count.name = "active_item_count"

    result = behavior_counts.copy()
    result["active_user_count"] = active_user_count.astype("int32")
    result["active_item_count"] = active_item_count.astype("int32")
    result = result.reset_index()

    return result[
        group_cols
        + [
            "pv_count",
            "fav_count",
            "cart_count",
            "buy_count",
            "active_user_count",
            "active_item_count",
        ]
    ]


def build_time_dims(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """构建三张时间维度中间表。

    Args:
        df: 清洗后的行为流水表，需包含 time 列（datetime64）

    Returns:
        字典: {"hourly": df, "daily": df, "weekday_hour": df}
    """
    logger.info("开始构建 dim_time 三张表，输入行数: %d", len(df))

    # 预计算辅助列，避免重复计算
    df_time = df.copy()
    df_time["_date"] = df_time["time"].dt.date
    df_time["_hour"] = df_time["time"].dt.hour
    # weekday: 0=周一, 6=周日
    df_time["_weekday"] = df_time["time"].dt.weekday

    # --- dim_time_hourly: 按(日期, 小时)聚合 ---
    hourly = _aggregate_time_metrics(df_time, ["_date", "_hour"])
    hourly = hourly.rename(columns={"_date": "date", "_hour": "hour"})
    logger.info(
        "dim_time_hourly 构建完成: %d 行, %d 列",
        len(hourly),
        len(hourly.columns),
    )

    # --- dim_time_daily: 按日期聚合 ---
    daily = _aggregate_time_metrics(df_time, ["_date"])
    daily = daily.rename(columns={"_date": "date"})
    logger.info(
        "dim_time_daily 构建完成: %d 行, %d 列",
        len(daily),
        len(daily.columns),
    )

    # --- dim_time_weekday_hour: 按(星期几, 小时)聚合 ---
    weekday_hour = _aggregate_time_metrics(df_time, ["_weekday", "_hour"])
    weekday_hour = weekday_hour.rename(
        columns={"_weekday": "weekday", "_hour": "hour"}
    )
    logger.info(
        "dim_time_weekday_hour 构建完成: %d 行, %d 列",
        len(weekday_hour),
        len(weekday_hour.columns),
    )

    return {"hourly": hourly, "daily": daily, "weekday_hour": weekday_hour}

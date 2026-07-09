"""生命周期特征模块。

输出字段（4个）：
    user_streak_days        : 用户最长连续活跃天数（用户级）
    item_decay_slope        : 商品热度趋势斜率（商品级，门槛：有互动天数>=3）
    user_avg_interval_hours : 用户平均行为间隔（用户级）
    buy_path_type           : (用户,商品)对的购买路径分类（0-4）
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


# 商品热度衰减斜率计算门槛
# 线性回归至少需要3个点才有趋势意义，2个点只是一条直线
DECAY_MIN_ACTIVE_DAYS = 3  # 有互动天数门槛

# 行为价值排序：购买>加购>收藏>浏览，用于同一小时窗口内行为的去重
BEHAVIOR_VALUE = {4: 4, 3: 3, 2: 2, 1: 1}

# 行为名映射（路径分类）
BEHAVIOR_NAMES = {1: "pv", 2: "fav", 3: "cart", 4: "buy"}


def calc_user_streak_days(df: pd.DataFrame) -> pd.Series:
    """计算用户最长连续活跃天数。

    Args:
        df: 原始明细数据，包含 user_id 和 time 两列。

    Returns:
        Series, index=user_id, value=最长连续活跃天数（1-31）。
    """
    # 提取每个用户活跃的日期集合
    user_dates = df[["user_id", "time"]].copy()
    user_dates["date"] = user_dates["time"].dt.date
    user_dates = user_dates.drop_duplicates(["user_id", "date"])

    def longest_streak(dates: pd.Series) -> int:
        if dates.empty:
            return 0
        sorted_dates = pd.to_datetime(dates).sort_values().reset_index(drop=True)
        if len(sorted_dates) == 1:
            return 1
        diffs = sorted_dates.diff().dt.days.iloc[1:]
        # diffs 中等于1的位置表示连续
        max_streak = cur_streak = 1
        for d in diffs:
            if d == 1:
                cur_streak += 1
                max_streak = max(max_streak, cur_streak)
            else:
                cur_streak = 1
        return max_streak

    return user_dates.groupby("user_id")["date"].apply(longest_streak)


def _daily_heat(df: pd.DataFrame) -> pd.DataFrame:
    """算每个商品每天的综合热度（pv×1 + fav×2 + cart×3 + buy×5）。"""
    heat = df.copy()
    heat["heat"] = (
        (heat["behavior_type"] == 1).astype(int) * 1
        + (heat["behavior_type"] == 2).astype(int) * 2
        + (heat["behavior_type"] == 3).astype(int) * 3
        + (heat["behavior_type"] == 4).astype(int) * 5
    )
    return heat.groupby(["item_id", heat["time"].dt.date])["heat"].sum().reset_index()


def calc_item_decay_slope(df: pd.DataFrame) -> pd.Series:
    """计算商品热度趋势斜率（线性回归）。

    门槛：有互动天数>=3（线性回归至少需要3个点）。不满足的设为 NaN。

    Args:
        df: 原始明细数据。

    Returns:
        Series, index=item_id, value=线性回归斜率。
    """
    daily = _daily_heat(df)
    item_active_days = daily.groupby("item_id")["heat"].count()
    qualified = item_active_days[item_active_days >= DECAY_MIN_ACTIVE_DAYS].index
    daily_q = daily[daily["item_id"].isin(qualified)]

    slopes: dict[int, float] = {}
    for item_id, grp in daily_q.groupby("item_id"):
        grp_sorted = grp.sort_values("time")
        x = np.arange(len(grp_sorted)).reshape(-1, 1)
        y = grp_sorted["heat"].values
        if len(x) < 2:
            continue
        model = LinearRegression().fit(x, y)
        slopes[item_id] = model.coef_[0]

    return pd.Series(slopes, name="item_decay_slope", dtype="float64")


def calc_user_avg_interval_hours(df: pd.DataFrame) -> pd.Series:
    """计算用户平均行为间隔（小时）。

    算法：同一小时窗口内的行为合并为 1 个时间点，再算相邻窗口的时间差平均。
    只有 1 个时间点的用户设为 NaN。

    Args:
        df: 原始明细数据。

    Returns:
        Series, index=user_id, value=平均间隔小时数。
    """
    # 每个用户去重后的时间窗口
    user_hours = df[["user_id", "time"]].drop_duplicates()

    def avg_interval(times: pd.Series) -> float:
        if len(times) < 2:
            return np.nan
        sorted_times = times.sort_values()
        diffs = sorted_times.diff().dropna().dt.total_seconds() / 3600
        return diffs.mean()

    return user_hours.groupby("user_id")["time"].apply(avg_interval)


def calc_buy_path_type(df: pd.DataFrame) -> pd.DataFrame:
    """计算(用户,商品)对的购买路径分类。

    分类规则（针对每个用户对每件商品的行为）：
        0 = 未购买该商品
        1 = 直接购买（对该商品无收藏无加购）
        2 = 有收藏无加购
        3 = 有加购无收藏
        4 = 收藏+加购都有

    Args:
        df: 原始明细数据。

    Returns:
        DataFrame, 列 = [user_id, item_id, buy_path_type]。
    """
    # 每个(用户,商品)对有哪些行为
    pair_behaviors = df.groupby(["user_id", "item_id"])["behavior_type"].apply(set)

    def classify(behaviors: set) -> int:
        if 4 not in behaviors:  # 没买过这件商品
            return 0
        has_fav = 2 in behaviors
        has_cart = 3 in behaviors
        if has_fav and has_cart:
            return 4
        if has_cart:
            return 3
        if has_fav:
            return 2
        return 1

    result = pair_behaviors.apply(classify).reset_index(name="buy_path_type")
    return result[["user_id", "item_id", "buy_path_type"]]

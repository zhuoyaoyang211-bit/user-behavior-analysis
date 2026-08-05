"""推理阶段的分块行为特征聚合模块。

该模块只保留构建 XGBoost 所需的聚合中间结果，不拼接完整原始行为表。
输入行为数据按块调用 :class:`RawBehaviorAggregator.add`，最终生成：

    - 用户-商品候选对
    - 商品、类目和用户统计特征
    - 商品热度趋势和用户平均行为间隔
    - RFM 特征和用户类目偏好特征

所有公开函数和方法均保持纯数据处理职责，不负责文件读取或模型预测。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from feature_engineering.business_feature import (
    RFM_F_BINS,
    RFM_M_BINS,
    RFM_OBSERVE_DATE,
    RFM_R_BINS,
)

BEHAVIOR_COLUMNS = {
    1: "pv_count",
    2: "fav_count",
    3: "cart_count",
    4: "buy_count",
}
PAIR_KEYS = ["user_id", "item_id", "item_category"]


def _ensure_behavior_columns(df: pd.DataFrame) -> pd.DataFrame:
    """补齐四种行为统计列并保持固定列顺序。"""
    for behavior_type, column in BEHAVIOR_COLUMNS.items():
        if behavior_type not in df.columns:
            df[behavior_type] = 0
    return df.rename(columns=BEHAVIOR_COLUMNS)[
        PAIR_KEYS + list(BEHAVIOR_COLUMNS.values())
    ]


def _calc_item_decay_slope(daily_heat: pd.DataFrame) -> pd.Series:
    """根据已聚合的商品日热度计算商品热度趋势斜率。"""
    if daily_heat.empty:
        return pd.Series(dtype="float64", name="item_decay_slope")

    active_days = daily_heat.groupby("item_id")["date"].nunique()
    qualified_items = active_days[active_days >= 3].index
    qualified_heat = daily_heat[daily_heat["item_id"].isin(qualified_items)]
    slopes: dict[int, float] = {}

    for item_id, item_group in qualified_heat.groupby("item_id"):
        item_group = item_group.sort_values("date")
        x_values = np.arange(len(item_group)).reshape(-1, 1)
        y_values = item_group["heat"].to_numpy(dtype=np.float64)
        model = LinearRegression().fit(x_values, y_values)
        slopes[int(item_id)] = float(model.coef_[0])

    return pd.Series(slopes, name="item_decay_slope", dtype="float64")


def _calc_user_avg_interval(user_hours: pd.DataFrame) -> pd.Series:
    """根据用户去重后的活跃小时计算平均行为间隔。"""
    if user_hours.empty:
        return pd.Series(dtype="float64", name="user_avg_interval_hours")

    user_hours = user_hours.sort_values(["user_id", "time"])
    intervals = user_hours.groupby("user_id")["time"].diff()
    interval_hours = intervals.dt.total_seconds() / 3600
    result = interval_hours.groupby(user_hours["user_id"]).mean()
    result.name = "user_avg_interval_hours"
    return result


def _calc_rfm_features(user_features: pd.DataFrame) -> pd.DataFrame:
    """按项目原有 RFM 分档规则计算用户 RFM 特征。"""
    result = user_features[["user_id"]].copy()
    last_buy = user_features["last_buy_time"]
    r_days = (RFM_OBSERVE_DATE - last_buy).dt.days.values
    r_score = pd.cut(r_days, bins=RFM_R_BINS, labels=False, right=True)
    result["rfm_r_score"] = 5 - np.nan_to_num(r_score, nan=4).astype(int)

    f_score = pd.cut(
        user_features["buy_count"].values,
        bins=RFM_F_BINS,
        labels=False,
        right=True,
    )
    result["rfm_f_score"] = np.nan_to_num(f_score, nan=0).astype(int) + 1

    m_score = pd.cut(
        user_features["buy_item_count"].values,
        bins=RFM_M_BINS,
        labels=False,
        right=True,
    )
    result["rfm_m_score"] = np.nan_to_num(m_score, nan=0).astype(int) + 1
    return result


@dataclass
class RawBehaviorAggregator:
    """分块累计推理特征所需的行为统计。

    该类不保存原始行为块，只保存按用户、商品、用户-商品对、
    商品-日期和用户-小时聚合后的中间结果。
    """

    pair_chunks: list[pd.DataFrame] = field(default_factory=list)
    user_chunks: list[pd.DataFrame] = field(default_factory=list)
    category_buy_chunks: list[pd.DataFrame] = field(default_factory=list)
    daily_heat_chunks: list[pd.DataFrame] = field(default_factory=list)
    user_hour_chunks: list[pd.DataFrame] = field(default_factory=list)
    _finalized: bool = False
    _pair_events: pd.DataFrame | None = None
    _item_category_map: pd.DataFrame | None = None
    _observed_pairs: pd.DataFrame | None = None
    _user_features: pd.DataFrame | None = None
    _item_features: pd.DataFrame | None = None
    _category_features: pd.DataFrame | None = None
    _category_pref: pd.DataFrame | None = None
    _item_decay_slope: pd.Series | None = None
    _user_avg_interval: pd.Series | None = None

    _HEAT_WEIGHTS: ClassVar[dict[int, int]] = {
        1: 1,
        2: 2,
        3: 3,
        4: 5,
    }

    def add(self, behavior_df: pd.DataFrame) -> None:
        """加入一个已清洗的行为数据块。

        Args:
            behavior_df: 已完成字段校验和时间标准化的行为数据块。

        Raises:
            ValueError: 输入缺少必要字段或该聚合器已经完成 finalize。
        """
        if self._finalized:
            raise ValueError("Cannot add data after aggregator has been finalized.")

        required_columns = {
            "user_id",
            "item_id",
            "item_category",
            "behavior_type",
            "time",
        }
        missing_columns = sorted(required_columns - set(behavior_df.columns))
        if missing_columns:
            raise ValueError(f"Behavior chunk missing columns: {missing_columns}")

        chunk = behavior_df[
            [
                "user_id",
                "item_id",
                "item_category",
                "behavior_type",
                "time",
            ]
        ].copy()

        pair_events = (
            chunk.groupby(PAIR_KEYS + ["behavior_type"])
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )
        self.pair_chunks.append(_ensure_behavior_columns(pair_events))

        behavior_counts = (
            chunk.groupby(["user_id", "behavior_type"]).size().unstack(fill_value=0)
        )
        behavior_counts = behavior_counts.rename(columns=BEHAVIOR_COLUMNS)
        for column in BEHAVIOR_COLUMNS.values():
            if column not in behavior_counts.columns:
                behavior_counts[column] = 0

        chunk["is_day"] = chunk["time"].dt.hour.between(6, 17)
        chunk["is_night"] = chunk["time"].dt.hour.isin(range(0, 6))
        user_time = chunk.groupby("user_id").agg(
            total_actions=("user_id", "size"),
            day_actions=("is_day", "sum"),
            night_actions=("is_night", "sum"),
            first_active_time=("time", "min"),
            last_active_time=("time", "max"),
        )
        user_time = user_time.join(behavior_counts, how="left")
        buy_rows = chunk[chunk["behavior_type"] == 4]
        user_time["last_buy_time"] = buy_rows.groupby("user_id")["time"].max()
        self.user_chunks.append(user_time.reset_index())

        category_buy = (
            buy_rows.groupby(["user_id", "item_category"])
            .size()
            .reset_index(name="category_buy_count")
        )
        self.category_buy_chunks.append(category_buy)

        chunk["heat"] = chunk["behavior_type"].map(self._HEAT_WEIGHTS).astype(np.int16)
        daily_heat = (
            chunk.assign(date=chunk["time"].dt.normalize())
            .groupby(["item_id", "date"], as_index=False)["heat"]
            .sum()
        )
        self.daily_heat_chunks.append(daily_heat)
        self.user_hour_chunks.append(
            chunk[["user_id", "time"]].drop_duplicates().reset_index(drop=True)
        )

    def finalize(self) -> None:
        """合并分块聚合结果并计算派生统计。"""
        if self._finalized:
            return
        if not self.pair_chunks:
            raise ValueError("Cannot finalize an empty behavior aggregator.")

        pair_events = pd.concat(self.pair_chunks, ignore_index=True)
        pair_events = pair_events.groupby(PAIR_KEYS, as_index=False, sort=False)[
            list(BEHAVIOR_COLUMNS.values())
        ].sum()
        self._pair_events = pair_events
        self._validate_item_category_consistency(pair_events)
        self._item_category_map = pair_events[
            ["item_id", "item_category"]
        ].drop_duplicates("item_id")
        self._observed_pairs = pair_events[PAIR_KEYS].copy()

        user_events = self._build_user_features(pair_events)
        self._user_features = user_events
        self._item_features = self._build_item_features(pair_events)
        self._category_features = self._build_category_features(pair_events)
        self._category_pref = self._build_category_preference(user_events)

        daily_heat = pd.concat(self.daily_heat_chunks, ignore_index=True)
        daily_heat = daily_heat.groupby(["item_id", "date"], as_index=False)[
            "heat"
        ].sum()
        self._item_decay_slope = _calc_item_decay_slope(daily_heat)

        user_hours = pd.concat(self.user_hour_chunks, ignore_index=True)
        user_hours = user_hours.drop_duplicates(["user_id", "time"])
        self._user_avg_interval = _calc_user_avg_interval(user_hours)

        self.pair_chunks.clear()
        self.user_chunks.clear()
        self.category_buy_chunks.clear()
        self.daily_heat_chunks.clear()
        self.user_hour_chunks.clear()
        self._finalized = True

    @property
    def item_category_map(self) -> pd.DataFrame:
        """返回商品到类目的唯一映射表。"""
        self.finalize()
        if self._item_category_map is None:
            raise RuntimeError("Item category map was not built.")
        return self._item_category_map.copy()

    @property
    def observed_pairs(self) -> pd.DataFrame:
        """返回原始行为中出现过的全部用户-商品对。"""
        self.finalize()
        if self._observed_pairs is None:
            raise RuntimeError("Observed pairs were not built.")
        return self._observed_pairs.copy()

    def build_wide_table(self, target_pairs: pd.DataFrame) -> pd.DataFrame:
        """将候选对与已聚合特征拼接为模型推理宽表。

        Args:
            target_pairs: 包含 user_id、item_id、item_category 的候选对。

        Returns:
            模型推理宽表。
        """
        self.finalize()
        if self._item_features is None or self._category_features is None:
            raise RuntimeError("Feature aggregation was not completed.")

        wide_df = target_pairs.copy()
        wide_df = wide_df.merge(self._item_features, on="item_id", how="left")
        wide_df = wide_df.merge(
            self._category_features,
            on="item_category",
            how="left",
        )
        wide_df = wide_df.merge(
            self._user_features[
                [
                    "user_id",
                    "user_pv_count",
                    "day_pct",
                    "night_pct",
                    "buy_conversion_rate",
                    "fav_to_buy_rate",
                    "cart_to_buy_rate",
                    "repurchase_item_count",
                ]
            ],
            on="user_id",
            how="left",
        )
        wide_df["item_decay_slope"] = wide_df["item_id"].map(self._item_decay_slope)
        wide_df["user_avg_interval_hours"] = wide_df["user_id"].map(
            self._user_avg_interval
        )
        wide_df = wide_df.merge(
            _calc_rfm_features(self._user_features),
            on="user_id",
            how="left",
        )
        wide_df = wide_df.merge(
            self._category_pref,
            on=["user_id", "item_category"],
            how="left",
        )
        return _downcast_dtypes(wide_df)

    @staticmethod
    def _validate_item_category_consistency(pair_events: pd.DataFrame) -> None:
        """校验同一商品没有对应多个类目。"""
        category_counts = pair_events.groupby("item_id")["item_category"].nunique()
        conflicts = category_counts[category_counts > 1]
        if not conflicts.empty:
            sample_ids = conflicts.head(10).index.tolist()
            raise ValueError(
                "item_id maps to multiple item_category values. "
                f"Count={len(conflicts):,}, examples={sample_ids}"
            )

    def _build_user_features(self, pair_events: pd.DataFrame) -> pd.DataFrame:
        """构建用户级行为、转化和复购特征。"""
        user_events = pd.concat(self.user_chunks, ignore_index=True)
        user_events = user_events.groupby("user_id", as_index=False).agg(
            {
                "total_actions": "sum",
                "day_actions": "sum",
                "night_actions": "sum",
                "first_active_time": "min",
                "last_active_time": "max",
                "last_buy_time": "max",
                "pv_count": "sum",
                "fav_count": "sum",
                "cart_count": "sum",
                "buy_count": "sum",
            }
        )
        user_events["day_pct"] = (
            user_events["day_actions"] / user_events["total_actions"]
        ).fillna(0.0)
        user_events["night_pct"] = (
            user_events["night_actions"] / user_events["total_actions"]
        ).fillna(0.0)

        pv_pairs = pair_events[pair_events["pv_count"] > 0]
        fav_pairs = pair_events[pair_events["fav_count"] > 0]
        cart_pairs = pair_events[pair_events["cart_count"] > 0]
        buy_pairs = pair_events[pair_events["buy_count"] > 0]
        buy_item_count = buy_pairs.groupby("user_id")["item_id"].nunique()
        buy_item_count.name = "buy_item_count"
        repurchase_pairs = pair_events[pair_events["buy_count"] >= 2]
        repurchase_item_count = repurchase_pairs.groupby("user_id")["item_id"].nunique()
        repurchase_item_count.name = "repurchase_item_count"

        user_events = user_events.merge(
            buy_item_count,
            on="user_id",
            how="left",
        )
        user_events = user_events.merge(
            repurchase_item_count,
            on="user_id",
            how="left",
        )
        conversion_rates = pd.concat(
            [
                _user_conversion_rate(pv_pairs, buy_pairs).rename(
                    "buy_conversion_rate"
                ),
                _user_conversion_rate(fav_pairs, buy_pairs).rename("fav_to_buy_rate"),
                _user_conversion_rate(cart_pairs, buy_pairs).rename("cart_to_buy_rate"),
            ],
            axis=1,
        )
        user_events = user_events.merge(
            conversion_rates,
            left_on="user_id",
            right_index=True,
            how="left",
        )
        user_events["user_pv_count"] = user_events["pv_count"]
        user_events["buy_item_count"] = user_events["buy_item_count"].fillna(0)
        user_events["repurchase_item_count"] = user_events[
            "repurchase_item_count"
        ].fillna(0)
        return user_events

    @staticmethod
    def _build_item_features(pair_events: pd.DataFrame) -> pd.DataFrame:
        """构建商品级统计、转化和复购特征。"""
        item_events = pair_events.groupby("item_id", as_index=False).agg(
            item_pv_count=("pv_count", "sum"),
            item_fav_count=("fav_count", "sum"),
            item_cart_count=("cart_count", "sum"),
            item_buy_count=("buy_count", "sum"),
        )
        item_buy_user_count = (
            pair_events[pair_events["buy_count"] > 0]
            .groupby("item_id")["user_id"]
            .nunique()
        )
        item_view_user_count = (
            pair_events[pair_events["pv_count"] > 0]
            .groupby("item_id")["user_id"]
            .nunique()
        )
        item_repurchase_user_count = (
            pair_events[pair_events["buy_count"] >= 2]
            .groupby("item_id")["user_id"]
            .nunique()
        )

        item_pv_to_buy_rate = _item_conversion_rate(
            pair_events[pair_events["pv_count"] > 0],
            pair_events[pair_events["pv_count"] > 0][
                pair_events[pair_events["pv_count"] > 0]["buy_count"] > 0
            ],
        )
        item_cart_to_buy_rate = _item_conversion_rate(
            pair_events[pair_events["cart_count"] > 0],
            pair_events[pair_events["cart_count"] > 0][
                pair_events[pair_events["cart_count"] > 0]["buy_count"] > 0
            ],
        )
        item_events = item_events.merge(
            item_buy_user_count.rename("item_buy_user_count"),
            on="item_id",
            how="left",
        )
        item_events = item_events.merge(
            item_view_user_count.rename("item_view_user_count"),
            on="item_id",
            how="left",
        )
        item_events = item_events.merge(
            item_repurchase_user_count.rename("item_repurchase_user_count"),
            on="item_id",
            how="left",
        )
        item_events = item_events.merge(
            item_pv_to_buy_rate.rename("item_pv_to_buy_rate"),
            on="item_id",
            how="left",
        )
        item_events = item_events.merge(
            item_cart_to_buy_rate.rename("item_cart_to_buy_rate"),
            on="item_id",
            how="left",
        )
        item_events = item_events[
            [
                "item_id",
                "item_pv_count",
                "item_fav_count",
                "item_cart_count",
                "item_buy_count",
                "item_buy_user_count",
                "item_pv_to_buy_rate",
                "item_cart_to_buy_rate",
                "item_repurchase_user_count",
            ]
        ]
        return item_events

    @staticmethod
    def _build_category_features(pair_events: pd.DataFrame) -> pd.DataFrame:
        """构建类目浏览次数和独立浏览用户数。"""
        category_events = pair_events.groupby("item_category", as_index=False).agg(
            cat_pv_count=("pv_count", "sum")
        )
        category_view_user_count = (
            pair_events[pair_events["pv_count"] > 0]
            .groupby("item_category")["user_id"]
            .nunique()
        )
        return category_events.merge(
            category_view_user_count.rename("cat_view_user_count"),
            on="item_category",
            how="left",
        )

    def _build_category_preference(
        self,
        user_features: pd.DataFrame,
    ) -> pd.DataFrame:
        """构建用户-类目购买偏好特征。"""
        category_buy = pd.concat(self.category_buy_chunks, ignore_index=True)
        category_buy = category_buy.groupby(
            ["user_id", "item_category"], as_index=False
        )["category_buy_count"].sum()
        total_buy = user_features[["user_id", "buy_count"]]
        category_buy = category_buy.merge(total_buy, on="user_id", how="left")
        category_buy["user_category_pref_score"] = (
            category_buy["category_buy_count"] / category_buy["buy_count"]
        ).fillna(0.0)
        return category_buy[["user_id", "item_category", "user_category_pref_score"]]


def _user_conversion_rate(
    upstream_pairs: pd.DataFrame,
    buy_pairs: pd.DataFrame,
) -> pd.Series:
    """计算用户级上游行为到购买的转化率。"""
    upstream_count = upstream_pairs.groupby("user_id")["item_id"].nunique()
    bought_pairs = upstream_pairs.merge(
        buy_pairs[["user_id", "item_id"]].drop_duplicates(),
        on=["user_id", "item_id"],
        how="inner",
    )
    bought_count = bought_pairs.groupby("user_id")["item_id"].nunique()
    return bought_count / upstream_count


def _item_conversion_rate(
    upstream_pairs: pd.DataFrame,
    bought_pairs: pd.DataFrame,
) -> pd.Series:
    """计算商品级上游行为到购买的转化率。"""
    upstream_count = upstream_pairs.groupby("item_id")["user_id"].nunique()
    bought_count = bought_pairs.groupby("item_id")["user_id"].nunique()
    return bought_count / upstream_count


def _downcast_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """对数值列做降级，减少聚合结果内存。"""
    for column in df.columns:
        dtype = df[column].dtype
        if pd.api.types.is_integer_dtype(dtype):
            df[column] = pd.to_numeric(df[column], downcast="integer")
        elif pd.api.types.is_float_dtype(dtype):
            df[column] = pd.to_numeric(df[column], downcast="float")
    return df

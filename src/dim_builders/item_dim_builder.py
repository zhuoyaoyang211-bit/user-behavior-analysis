"""商品维度中间表构建器。

将清洗后的行为流水表聚合为两张商品维度宽表:
    - dim_item: 单品维度（287万行，全量不过滤）
    - dim_category: 类目维度（8916行，全量）

输入: cleaned_data.parquet 的 DataFrame（6列）
输出: (dim_item DataFrame, dim_category DataFrame)
"""

import pandas as pd

from common.constants import (
    BEHAVIOR_BUY,
    BEHAVIOR_CART,
    BEHAVIOR_FAV,
    BEHAVIOR_PV,
)
from common.logger import get_logger

logger = get_logger("item_dim_builder")


def build_item_dim(df: pd.DataFrame) -> pd.DataFrame:
    """构建商品单品维度中间表。

    聚合逻辑:
        - 流量: pv_count(pv口径), view_user_count(uv口径)
        - 互动: fav_count, cart_count
        - 购买: buy_count(pv口径), buy_user_count(uv口径)
        - 转化(漏斗口径): pv_to_buy_rate, cart_to_buy_rate
          浏览/加购过该商品的用户中，有多少最终买了，值域 [0, 1]
        - 复购: repurchase_user_count(买≥2次的用户数)

    Args:
        df: 清洗后的行为流水表

    Returns:
        商品单品维度宽表，每行一个商品，11列。
    """
    logger.info("开始构建 dim_item，输入行数: %d", len(df))

    # --- 基础行为计数 ---
    # 按 item_id × behavior_type 交叉统计次数
    item_behavior = (
        df.groupby(["item_id", "behavior_type"]).size().unstack(fill_value=0)
    )
    for bt in [BEHAVIOR_PV, BEHAVIOR_FAV, BEHAVIOR_CART, BEHAVIOR_BUY]:
        if bt not in item_behavior.columns:
            item_behavior[bt] = 0
    item_behavior = item_behavior.rename(
        columns={
            BEHAVIOR_PV: "pv_count",
            BEHAVIOR_FAV: "fav_count",
            BEHAVIOR_CART: "cart_count",
            BEHAVIOR_BUY: "buy_count",
        }
    )

    # --- uv口径: 独立用户数 ---
    # 浏览独立用户数
    pv_df = df[df["behavior_type"] == BEHAVIOR_PV]
    view_user_count = pv_df.groupby("item_id")["user_id"].nunique()
    view_user_count.name = "view_user_count"

    # 购买独立用户数
    buy_df = df[df["behavior_type"] == BEHAVIOR_BUY]
    buy_user_count = buy_df.groupby("item_id")["user_id"].nunique()
    buy_user_count.name = "buy_user_count"

    # --- 所属类目（外键） ---
    # Part1已校验每个item_id对应唯一item_category，取first即可
    item_category = df.groupby("item_id")["item_category"].first()
    item_category.name = "item_category"

    # --- 复购用户数: 买≥2次的用户数 ---
    user_item_buy = buy_df.groupby(["item_id", "user_id"]).size()
    repurchase_users = user_item_buy[user_item_buy >= 2]
    repurchase_user_count = repurchase_users.groupby("item_id").size()
    repurchase_user_count.name = "repurchase_user_count"

    # --- 合并 ---
    result = item_behavior.copy()
    result["item_category"] = item_category
    result["view_user_count"] = view_user_count
    result["buy_user_count"] = buy_user_count
    result["repurchase_user_count"] = repurchase_user_count

    # 对齐后fillna: 没有浏览/购买/复购的商品这些字段为NaN → 填0
    result["view_user_count"] = result["view_user_count"].fillna(0).astype("int32")
    result["buy_user_count"] = result["buy_user_count"].fillna(0).astype("int32")
    result["repurchase_user_count"] = (
        result["repurchase_user_count"].fillna(0).astype("int32")
    )

    # --- 转化率(漏斗口径: 基于用户集合) ---
    # pv_to_buy_rate: 浏览过该商品的用户中，有多少最终买了
    #   = |{浏览过的用户} ∩ {买过的用户}| ÷ |{浏览过的用户}|
    # cart_to_buy_rate: 加购过该商品的用户中，有多少最终买了
    # 分子是分母的子集，转化率一定在 [0, 1]
    pv_pairs = df[df["behavior_type"] == BEHAVIOR_PV][
        ["user_id", "item_id"]
    ].drop_duplicates()
    cart_pairs = df[df["behavior_type"] == BEHAVIOR_CART][
        ["user_id", "item_id"]
    ].drop_duplicates()
    buy_pairs = buy_df[["user_id", "item_id"]].drop_duplicates()

    pv_user_n = pv_pairs.groupby("item_id")["user_id"].nunique()
    pv_buy_user_n = pv_pairs.merge(buy_pairs, on=["user_id", "item_id"]).groupby(
        "item_id"
    )["user_id"].nunique()
    result["pv_to_buy_rate"] = (pv_buy_user_n / pv_user_n).reindex(
        result.index
    ).fillna(0.0)

    cart_user_n = cart_pairs.groupby("item_id")["user_id"].nunique()
    cart_buy_user_n = cart_pairs.merge(
        buy_pairs, on=["user_id", "item_id"]
    ).groupby("item_id")["user_id"].nunique()
    result["cart_to_buy_rate"] = (cart_buy_user_n / cart_user_n).reindex(
        result.index
    ).fillna(0.0)

    result = result.reset_index()

    result = result[
        [
            "item_id",
            "item_category",
            "pv_count",
            "view_user_count",
            "fav_count",
            "cart_count",
            "buy_count",
            "buy_user_count",
            "pv_to_buy_rate",
            "cart_to_buy_rate",
            "repurchase_user_count",
        ]
    ]

    logger.info(
        "dim_item 构建完成: %d 行, %d 列", len(result), len(result.columns)
    )
    return result


def build_category_dim(df: pd.DataFrame) -> pd.DataFrame:
    """构建商品类目维度中间表。

    聚合逻辑:
        - 规模: item_count(商品总数), buy_item_count(有购买的商品数)
        - 流量+互动+购买: 各行为次数和独立用户数
        - 转化(漏斗口径): pv_to_buy_rate
          该类目下被浏览过的商品中，有多少最终被买了，值域 [0, 1]
        - 覆盖: buy_item_pct(有购买商品占比)

    Args:
        df: 清洗后的行为流水表

    Returns:
        类目维度宽表，每行一个类目，11列。
    """
    logger.info("开始构建 dim_category，输入行数: %d", len(df))

    # --- 类目下商品总数 ---
    item_count = df.groupby("item_category")["item_id"].nunique()
    item_count.name = "item_count"

    # --- 有购买的商品数 ---
    buy_df = df[df["behavior_type"] == BEHAVIOR_BUY]
    buy_item_count = buy_df.groupby("item_category")["item_id"].nunique()
    buy_item_count.name = "buy_item_count"

    # --- 基础行为计数 ---
    cat_behavior = (
        df.groupby(["item_category", "behavior_type"])
        .size()
        .unstack(fill_value=0)
    )
    for bt in [BEHAVIOR_PV, BEHAVIOR_FAV, BEHAVIOR_CART, BEHAVIOR_BUY]:
        if bt not in cat_behavior.columns:
            cat_behavior[bt] = 0
    cat_behavior = cat_behavior.rename(
        columns={
            BEHAVIOR_PV: "pv_count",
            BEHAVIOR_FAV: "fav_count",
            BEHAVIOR_CART: "cart_count",
            BEHAVIOR_BUY: "buy_count",
        }
    )

    # --- uv口径 ---
    view_user_count = (
        df[df["behavior_type"] == BEHAVIOR_PV]
        .groupby("item_category")["user_id"]
        .nunique()
    )
    view_user_count.name = "view_user_count"

    buy_user_count = buy_df.groupby("item_category")["user_id"].nunique()
    buy_user_count.name = "buy_user_count"

    # --- 合并 ---
    result = cat_behavior.copy()
    result["item_count"] = item_count
    result["buy_item_count"] = buy_item_count
    result["view_user_count"] = view_user_count
    result["buy_user_count"] = buy_user_count

    # 对齐后fillna: 没有购买/浏览的类目这些字段为NaN → 填0
    result["buy_item_count"] = result["buy_item_count"].fillna(0).astype("int32")
    result["view_user_count"] = (
        result["view_user_count"].fillna(0).astype("int32")
    )
    result["buy_user_count"] = result["buy_user_count"].fillna(0).astype("int32")

    # 转化率(漏斗口径) & 覆盖率
    # pv_to_buy_rate: 该类目下被浏览过的商品中，有多少最终被买了
    #   = |{被浏览的商品} ∩ {被买的商品}| ÷ |{被浏览的商品}|，值域 [0, 1]
    pv_items_cat = df[df["behavior_type"] == BEHAVIOR_PV][
        ["item_id", "item_category"]
    ].drop_duplicates()
    buy_item_set = set(buy_df["item_id"].unique())
    pv_items_cat = pv_items_cat.copy()
    pv_items_cat["_bought"] = pv_items_cat["item_id"].isin(buy_item_set)

    pv_item_n = pv_items_cat.groupby("item_category")["item_id"].nunique()
    pv_buy_item_n = pv_items_cat[pv_items_cat["_bought"]].groupby(
        "item_category"
    )["item_id"].nunique()
    result["pv_to_buy_rate"] = (pv_buy_item_n / pv_item_n).reindex(
        result.index
    ).fillna(0.0)

    result["buy_item_pct"] = (
        result["buy_item_count"] / result["item_count"].replace(0, pd.NA)
    ).fillna(0.0)

    result = result.reset_index()

    result = result[
        [
            "item_category",
            "item_count",
            "buy_item_count",
            "pv_count",
            "view_user_count",
            "fav_count",
            "cart_count",
            "buy_count",
            "buy_user_count",
            "pv_to_buy_rate",
            "buy_item_pct",
        ]
    ]

    logger.info(
        "dim_category 构建完成: %d 行, %d 列",
        len(result),
        len(result.columns),
    )
    return result

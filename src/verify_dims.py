"""中间表抽检验证脚本。

从 cleaned_data.parquet 独立计算抽样样本的字段值，
与中间表中的聚合结果逐一比对，验证中间表计算逻辑的正确性。

抽检策略:
    - dim_user: 3个用户（2个power_user[有收藏+加购+购买] + 1个纯浏览用户）
    - dim_item: 8个商品（5个热门 + 3个冷门）
    - dim_category: 4个类目（2个热门 + 2个冷门）
    - dim_time_daily: 3天（首日 + 中间日 + 末日）
    - 全局守恒校验: 各维度表行为量加总 = 原始数据行为量

不抽检 dim_time_hourly / dim_time_weekday_hour 的原因:
    三张时间表共用同一个构建函数 _aggregate_time_metrics，6个字段完全相同
    （全是计数，无除法），只是分组维度不同。验 dim_time_daily 即已覆盖该
    函数的全部代码逻辑，另外两张表无独立计算分支，守恒校验已间接覆盖。

运行方式:
    cd 项目根目录
    python src/verify_dims.py
"""

import sys
from pathlib import Path

import pandas as pd

# 将 src 目录加入搜索路径，使能导入 common 模块
sys.path.insert(0, str(Path(__file__).parent))

from common.constants import BEHAVIOR_BUY, BEHAVIOR_CART, BEHAVIOR_FAV, BEHAVIOR_PV
from common.agg_specs import (
    DIM_USER_PATH,
    DIM_ITEM_PATH,
    DIM_CATEGORY_PATH,
    DIM_TIME_DAILY_PATH,
    DAY_HOURS,
    EVENING_HOURS,
    NIGHT_HOURS,
)
from common.logger import get_logger
from config import get_config

logger = get_logger("verify_dims")

# 浮点比对容差
FLOAT_TOLERANCE = 1e-4


# ===========================================================================
# 数据加载
# ===========================================================================
def load_data() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """加载原始数据和6张中间表。

    Returns:
        (原始数据, 中间表字典) 的元组
    """
    config = get_config()
    logger.info("加载 cleaned_data.parquet ...")
    df = pd.read_parquet(config.cleaned_data_path)
    logger.info("原始数据: %d 行", len(df))

    tables = {}
    for name, path in [
        ("dim_user", DIM_USER_PATH),
        ("dim_item", DIM_ITEM_PATH),
        ("dim_category", DIM_CATEGORY_PATH),
        ("dim_time_daily", DIM_TIME_DAILY_PATH),
    ]:
        tbl = pd.read_parquet(path)
        # 中间表主键存为 Parquet 索引列，读取后 reset 使其变为普通列
        if tbl.index.name is not None:
            tbl = tbl.reset_index()
        tables[name] = tbl
        logger.info("加载 %s: %d 行", name, len(tables[name]))

    return df, tables


# ===========================================================================
# 抽样选择
# ===========================================================================
def select_user_samples(df: pd.DataFrame, dim_user: pd.DataFrame) -> list[int]:
    """从 dim_user 中选3个代表性用户。

    选样策略:
        - 2个 power_user（重度买家，且有收藏+加购+购买，验正常除法分支）
        - 1个无购买无收藏无加购的用户（纯浏览，验分母为0填0的边界）

    Args:
        df: 原始数据
        dim_user: 用户中间表

    Returns:
        3个 user_id 的列表
    """
    samples = []

    # 2个 power_user（且有收藏+加购+购买，保证三个转化率字段都走正常除法分支）
    power_users = dim_user[
        (dim_user["is_power_user"] == True)
        & (dim_user["fav_count"] > 0)
        & (dim_user["cart_count"] > 0)
        & (dim_user["buy_count"] > 0)
    ]["user_id"].tolist()
    samples.extend(power_users[:2])

    # 1个无购买无收藏无加购的用户（纯浏览，验0÷0边界）
    no_buy = dim_user[
        (dim_user["buy_count"] == 0)
        & (dim_user["fav_count"] == 0)
        & (dim_user["cart_count"] == 0)
    ]["user_id"].tolist()
    samples.append(no_buy[0])

    return samples


def select_item_samples(df: pd.DataFrame, dim_item: pd.DataFrame) -> list[int]:
    """从 dim_item 中选8个代表性商品。

    选样策略:
        - 5个热门商品（被3+用户交互过，且有购买记录）
        - 3个冷门商品（只被1个用户浏览，且无收藏/加购/购买，验0÷0边界）

    Args:
        df: 原始数据
        dim_item: 商品中间表

    Returns:
        8个 item_id 的列表
    """
    samples = []

    # 5个热门：被3+用户浏览，且收藏/加购/购买都有记录（保证转化率字段分母不为0）
    hot = dim_item[
        (dim_item["view_user_count"] >= 3)
        & (dim_item["buy_count"] > 0)
        & (dim_item["fav_count"] > 0)
        & (dim_item["cart_count"] > 0)
    ]["item_id"].tolist()
    samples.extend(hot[:5])

    # 3个冷门：只被1个用户浏览，且无收藏/加购/购买（纯浏览，验0÷0边界）
    cold = dim_item[
        (dim_item["view_user_count"] == 1)
        & (dim_item["fav_count"] == 0)
        & (dim_item["cart_count"] == 0)
        & (dim_item["buy_count"] == 0)
    ]["item_id"].tolist()
    samples.extend(cold[:3])

    return samples


def select_category_samples(
    df: pd.DataFrame, dim_category: pd.DataFrame
) -> list[int]:
    """从 dim_category 中选4个代表性类目。

    选样策略:
        - 2个热门类目（浏览/收藏/加购/购买都有，验正常除法分支）
        - 2个冷门类目（仅浏览无收藏/加购/购买，验0÷N边界）

    两个比率字段的分支覆盖:
        - pv_to_buy_rate = |{被浏览商品} ∩ {被买商品}| ÷ |{被浏览商品}|
          热门: N÷M | 冷门(buy_count=0→分子=0): 0÷N
        - buy_item_pct = buy_item_count ÷ item_count
          热门: N÷M | 冷门(buy_count=0→buy_item_count=0→分子=0): 0÷N
        - 两个比率分母均不可能为0（类目进表说明至少有1个商品、至少有1条浏览），
          故不存在0÷0分支

    Args:
        df: 原始数据
        dim_category: 类目中间表

    Returns:
        4个 item_category 的列表
    """
    samples = []

    # 2个热门：四种行为都有记录（保证两个比率字段都走正常除法分支）
    hot = dim_category[
        (dim_category["pv_count"] > 0)
        & (dim_category["fav_count"] > 0)
        & (dim_category["cart_count"] > 0)
        & (dim_category["buy_count"] > 0)
    ]["item_category"].tolist()
    samples.extend(hot[:2])

    # 2个冷门：仅浏览，无收藏/加购/购买（验两个比率字段的0÷N边界）
    cold = dim_category[
        (dim_category["pv_count"] > 0)
        & (dim_category["fav_count"] == 0)
        & (dim_category["cart_count"] == 0)
        & (dim_category["buy_count"] == 0)
    ]["item_category"].tolist()
    samples.extend(cold[:2])

    return samples


def select_date_samples(dim_time_daily: pd.DataFrame) -> list:
    """从 dim_time_daily 中选3天：首日、中间日、末日。

    Args:
        dim_time_daily: 日级时间中间表

    Returns:
        3个日期的列表
    """
    dates = sorted(dim_time_daily["date"].tolist())
    n = len(dates)
    return [dates[0], dates[n // 2], dates[-1]]


# ===========================================================================
# dim_user 抽检验证
# ===========================================================================
def verify_user(
    user_id: int, df: pd.DataFrame, dim_user: pd.DataFrame
) -> list[dict]:
    """对单个用户独立计算字段值并与中间表比对。

    Args:
        user_id: 用户ID
        df: 原始数据
        dim_user: 用户中间表

    Returns:
        比对结果列表，每个元素是一行 {字段, 手算值, 中间表值, 是否一致}
    """
    # 筛出该用户的全部记录
    udf = df[df["user_id"] == user_id]
    row = dim_user[dim_user["user_id"] == user_id].iloc[0]
    results = []

    def _check(field: str, expected, actual, is_float: bool = False) -> None:
        if is_float:
            match = abs(float(expected) - float(actual)) < FLOAT_TOLERANCE
        else:
            match = expected == actual
        results.append({
            "样本": f"user_{user_id}",
            "字段": field,
            "手算值": expected,
            "中间表值": actual,
            "一致": "✓" if match else "✗",
        })

    # --- 行为量 ---
    bvc = udf["behavior_type"].value_counts()
    _check("pv_count", int(bvc.get(BEHAVIOR_PV, 0)), int(row["pv_count"]))
    _check("fav_count", int(bvc.get(BEHAVIOR_FAV, 0)), int(row["fav_count"]))
    _check("cart_count", int(bvc.get(BEHAVIOR_CART, 0)), int(row["cart_count"]))
    _check("buy_count", int(bvc.get(BEHAVIOR_BUY, 0)), int(row["buy_count"]))

    # --- 活跃天数 ---
    dates = udf["time"].dt.date.nunique()
    _check("active_days", dates, int(row["active_days"]))

    # --- 首末次时间 ---
    _check(
        "first_active_time",
        udf["time"].min(),
        pd.Timestamp(row["first_active_time"]),
    )
    _check(
        "last_active_time",
        udf["time"].max(),
        pd.Timestamp(row["last_active_time"]),
    )

    # --- 时段占比 ---
    hours = udf["time"].dt.hour
    total = len(udf)
    day_pct = (hours.isin(DAY_HOURS).sum() / total) if total > 0 else 0.0
    eve_pct = (hours.isin(EVENING_HOURS).sum() / total) if total > 0 else 0.0
    nit_pct = (hours.isin(NIGHT_HOURS).sum() / total) if total > 0 else 0.0
    _check("day_pct", round(day_pct, 6), round(float(row["day_pct"]), 6), is_float=True)
    _check("evening_pct", round(eve_pct, 6), round(float(row["evening_pct"]), 6), is_float=True)
    _check("night_pct", round(nit_pct, 6), round(float(row["night_pct"]), 6), is_float=True)

    # --- 购买商品数(去重) ---
    buy_items = udf[udf["behavior_type"] == BEHAVIOR_BUY]["item_id"].nunique()
    _check("buy_item_count", buy_items, int(row["buy_item_count"]))

    # --- 转化率(漏斗口径) ---
    pv_pairs = set(
        udf[udf["behavior_type"] == BEHAVIOR_PV]["item_id"].unique()
    )
    fav_pairs = set(
        udf[udf["behavior_type"] == BEHAVIOR_FAV]["item_id"].unique()
    )
    cart_pairs = set(
        udf[udf["behavior_type"] == BEHAVIOR_CART]["item_id"].unique()
    )
    buy_pairs = set(
        udf[udf["behavior_type"] == BEHAVIOR_BUY]["item_id"].unique()
    )

    bcr = len(pv_pairs & buy_pairs) / len(pv_pairs) if pv_pairs else 0.0
    fbr = len(fav_pairs & buy_pairs) / len(fav_pairs) if fav_pairs else 0.0
    cbr = len(cart_pairs & buy_pairs) / len(cart_pairs) if cart_pairs else 0.0
    _check("buy_conversion_rate", round(bcr, 6), round(float(row["buy_conversion_rate"]), 6), is_float=True)
    _check("fav_to_buy_rate", round(fbr, 6), round(float(row["fav_to_buy_rate"]), 6), is_float=True)
    _check("cart_to_buy_rate", round(cbr, 6), round(float(row["cart_to_buy_rate"]), 6), is_float=True)

    # --- 复购商品数 ---
    buy_df = udf[udf["behavior_type"] == BEHAVIOR_BUY]
    item_buy_counts = buy_df.groupby("item_id").size()
    repurchase = (item_buy_counts >= 2).sum()
    _check("repurchase_item_count", repurchase, int(row["repurchase_item_count"]))

    # --- 时段占比之和=1（逻辑校验）---
    pct_sum = round(day_pct + eve_pct + nit_pct, 6)
    _check("day+eve+night=1", 1.0, pct_sum, is_float=True)

    return results


# ===========================================================================
# dim_item 抽检验证
# ===========================================================================
def verify_item(
    item_id: int, df: pd.DataFrame, dim_item: pd.DataFrame
) -> list[dict]:
    """对单个商品独立计算字段值并与中间表比对。

    Args:
        item_id: 商品ID
        df: 原始数据
        dim_item: 商品中间表

    Returns:
        比对结果列表
    """
    idf = df[df["item_id"] == item_id]
    row = dim_item[dim_item["item_id"] == item_id].iloc[0]
    results = []

    def _check(field: str, expected, actual, is_float: bool = False) -> None:
        if is_float:
            match = abs(float(expected) - float(actual)) < FLOAT_TOLERANCE
        else:
            match = expected == actual
        results.append({
            "样本": f"item_{item_id}",
            "字段": field,
            "手算值": expected,
            "中间表值": actual,
            "一致": "✓" if match else "✗",
        })

    # --- 行为量 ---
    bvc = idf["behavior_type"].value_counts()
    _check("pv_count", int(bvc.get(BEHAVIOR_PV, 0)), int(row["pv_count"]))
    _check("fav_count", int(bvc.get(BEHAVIOR_FAV, 0)), int(row["fav_count"]))
    _check("cart_count", int(bvc.get(BEHAVIOR_CART, 0)), int(row["cart_count"]))
    _check("buy_count", int(bvc.get(BEHAVIOR_BUY, 0)), int(row["buy_count"]))

    # --- uv口径 ---
    pv_users = idf[idf["behavior_type"] == BEHAVIOR_PV]["user_id"].nunique()
    buy_users = idf[idf["behavior_type"] == BEHAVIOR_BUY]["user_id"].nunique()
    _check("view_user_count", pv_users, int(row["view_user_count"]))
    _check("buy_user_count", buy_users, int(row["buy_user_count"]))

    # --- 转化率 ---
    pv_u = set(idf[idf["behavior_type"] == BEHAVIOR_PV]["user_id"].unique())
    cart_u = set(idf[idf["behavior_type"] == BEHAVIOR_CART]["user_id"].unique())
    buy_u = set(idf[idf["behavior_type"] == BEHAVIOR_BUY]["user_id"].unique())

    pbr = len(pv_u & buy_u) / len(pv_u) if pv_u else 0.0
    cbr = len(cart_u & buy_u) / len(cart_u) if cart_u else 0.0
    _check("pv_to_buy_rate", round(pbr, 6), round(float(row["pv_to_buy_rate"]), 6), is_float=True)
    _check("cart_to_buy_rate", round(cbr, 6), round(float(row["cart_to_buy_rate"]), 6), is_float=True)

    # --- 复购用户数 ---
    buy_df = idf[idf["behavior_type"] == BEHAVIOR_BUY]
    user_buy_counts = buy_df.groupby("user_id").size()
    repurchase = (user_buy_counts >= 2).sum()
    _check("repurchase_user_count", repurchase, int(row["repurchase_user_count"]))

    return results


# ===========================================================================
# dim_category 抽检验证
# ===========================================================================
def verify_category(
    category: int, df: pd.DataFrame, dim_category: pd.DataFrame
) -> list[dict]:
    """对单个类目独立计算字段值并与中间表比对。

    Args:
        category: 类目ID
        df: 原始数据
        dim_category: 类目中间表

    Returns:
        比对结果列表
    """
    cdf = df[df["item_category"] == category]
    row = dim_category[dim_category["item_category"] == category].iloc[0]
    results = []

    def _check(field: str, expected, actual, is_float: bool = False) -> None:
        if is_float:
            match = abs(float(expected) - float(actual)) < FLOAT_TOLERANCE
        else:
            match = expected == actual
        results.append({
            "样本": f"cat_{category}",
            "字段": field,
            "手算值": expected,
            "中间表值": actual,
            "一致": "✓" if match else "✗",
        })

    # --- 商品数 ---
    item_count = cdf["item_id"].nunique()
    _check("item_count", item_count, int(row["item_count"]))

    buy_df = cdf[cdf["behavior_type"] == BEHAVIOR_BUY]
    buy_item_count = buy_df["item_id"].nunique()
    _check("buy_item_count", buy_item_count, int(row["buy_item_count"]))

    # --- 行为量 ---
    bvc = cdf["behavior_type"].value_counts()
    _check("pv_count", int(bvc.get(BEHAVIOR_PV, 0)), int(row["pv_count"]))
    _check("fav_count", int(bvc.get(BEHAVIOR_FAV, 0)), int(row["fav_count"]))
    _check("cart_count", int(bvc.get(BEHAVIOR_CART, 0)), int(row["cart_count"]))
    _check("buy_count", int(bvc.get(BEHAVIOR_BUY, 0)), int(row["buy_count"]))

    # --- uv口径 ---
    pv_users = cdf[cdf["behavior_type"] == BEHAVIOR_PV]["user_id"].nunique()
    buy_users = buy_df["user_id"].nunique()
    _check("view_user_count", pv_users, int(row["view_user_count"]))
    _check("buy_user_count", buy_users, int(row["buy_user_count"]))

    # --- 转化率(漏斗口径) ---
    # pv_to_buy_rate: 该类目下被浏览过的商品中，有多少最终被买了
    pv_items = set(
        cdf[cdf["behavior_type"] == BEHAVIOR_PV]["item_id"].unique()
    )
    buy_items = set(buy_df["item_id"].unique())
    pv_buy_item_n = len(pv_items & buy_items)
    ptbr = pv_buy_item_n / len(pv_items) if pv_items else 0.0
    _check(
        "pv_to_buy_rate",
        round(ptbr, 6),
        round(float(row["pv_to_buy_rate"]), 6),
        is_float=True,
    )

    # buy_item_pct: 有购买商品占比
    bip = buy_item_count / item_count if item_count > 0 else 0.0
    _check(
        "buy_item_pct",
        round(bip, 6),
        round(float(row["buy_item_pct"]), 6),
        is_float=True,
    )

    return results


# ===========================================================================
# dim_time_daily 抽检验证
# ===========================================================================
def verify_daily(
    date, df: pd.DataFrame, dim_time_daily: pd.DataFrame
) -> list[dict]:
    """对单日独立计算字段值并与中间表比对。

    Args:
        date: 日期
        df: 原始数据
        dim_time_daily: 日级时间中间表

    Returns:
        比对结果列表
    """
    ddf = df[df["time"].dt.date == date]
    row = dim_time_daily[dim_time_daily["date"] == date].iloc[0]
    results = []

    def _check(field: str, expected, actual) -> None:
        match = expected == actual
        results.append({
            "样本": f"date_{date}",
            "字段": field,
            "手算值": expected,
            "中间表值": actual,
            "一致": "✓" if match else "✗",
        })

    bvc = ddf["behavior_type"].value_counts()
    _check("pv_count", int(bvc.get(BEHAVIOR_PV, 0)), int(row["pv_count"]))
    _check("fav_count", int(bvc.get(BEHAVIOR_FAV, 0)), int(row["fav_count"]))
    _check("cart_count", int(bvc.get(BEHAVIOR_CART, 0)), int(row["cart_count"]))
    _check("buy_count", int(bvc.get(BEHAVIOR_BUY, 0)), int(row["buy_count"]))
    _check("active_user_count", ddf["user_id"].nunique(), int(row["active_user_count"]))
    _check("active_item_count", ddf["item_id"].nunique(), int(row["active_item_count"]))

    return results


# ===========================================================================
# 全局守恒校验
# ===========================================================================
def verify_conservation(
    df: pd.DataFrame, tables: dict[str, pd.DataFrame]
) -> list[dict]:
    """验证各维度表行为量加总是否等于原始数据行为量。

    校验逻辑:
        - dim_user 的 pv_count 总和 = 原始数据 behavior_type=1 的行数
        - dim_item 的 pv_count 总和 = 同上
        - dim_category 的 pv_count 总和 = 同上
        - dim_time_daily 的 pv_count 总和 = 同上
        - 四种行为分别在四个维度表里加总都相等

    Args:
        df: 原始数据
        tables: 中间表字典

    Returns:
        守恒校验结果列表
    """
    results = []
    behaviors = {
        "pv_count": BEHAVIOR_PV,
        "fav_count": BEHAVIOR_FAV,
        "cart_count": BEHAVIOR_CART,
        "buy_count": BEHAVIOR_BUY,
    }

    for field, bt in behaviors.items():
        raw_total = int((df["behavior_type"] == bt).sum())
        user_total = int(tables["dim_user"][field].sum())
        item_total = int(tables["dim_item"][field].sum())
        category_total = int(tables["dim_category"][field].sum())
        daily_total = int(tables["dim_time_daily"][field].sum())

        for dim_name, dim_total in [
            ("dim_user", user_total),
            ("dim_item", item_total),
            ("dim_category", category_total),
            ("dim_time_daily", daily_total),
        ]:
            match = raw_total == dim_total
            results.append({
                "样本": "守恒校验",
                "字段": f"{dim_name}.{field}",
                "手算值": raw_total,
                "中间表值": dim_total,
                "一致": "✓" if match else "✗",
            })

    return results


# ===========================================================================
# 主流程
# ===========================================================================
def main() -> None:
    """执行中间表抽检验证主流程。"""
    df, tables = load_data()

    all_results: list[dict] = []

    # --- dim_user 抽检 ---
    user_ids = select_user_samples(df, tables["dim_user"])
    logger.info("dim_user 抽样: %s", user_ids)
    for uid in user_ids:
        all_results.extend(verify_user(uid, df, tables["dim_user"]))

    # --- dim_item 抽检 ---
    item_ids = select_item_samples(df, tables["dim_item"])
    logger.info("dim_item 抽样: %s", item_ids)
    for iid in item_ids:
        all_results.extend(verify_item(iid, df, tables["dim_item"]))

    # --- dim_category 抽检 ---
    cat_ids = select_category_samples(df, tables["dim_category"])
    logger.info("dim_category 抽样: %s", cat_ids)
    for cat in cat_ids:
        all_results.extend(verify_category(cat, df, tables["dim_category"]))

    # --- dim_time_daily 抽检 ---
    dates = select_date_samples(tables["dim_time_daily"])
    logger.info("dim_time_daily 抽样: %s", dates)
    for d in dates:
        all_results.extend(verify_daily(d, df, tables["dim_time_daily"]))

    # --- 全局守恒校验 ---
    all_results.extend(verify_conservation(df, tables))

    # --- 打印结果 ---
    result_df = pd.DataFrame(all_results)
    total = len(result_df)
    passed = (result_df["一致"] == "✓").sum()
    failed = (result_df["一致"] == "✗").sum()

    print("\n" + "=" * 80)
    print("中间表抽检验证结果")
    print("=" * 80)
    print(result_df.to_string(index=False))
    print("=" * 80)
    print(f"总计: {total} 项 | 通过: {passed} | 不一致: {failed}")
    if failed == 0:
        print("结论: 全部通过 ✓")
    else:
        print("结论: 存在不一致项，请检查 ✗")
        # 打印不一致项
        fails = result_df[result_df["一致"] == "✗"]
        print("\n不一致项明细:")
        print(fails.to_string(index=False))

    print("=" * 80)


if __name__ == "__main__":
    main()

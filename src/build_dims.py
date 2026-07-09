"""中间表构建主入口（第二阶段）。

读取 Part1 清洗后的 Parquet 数据，调用三个维度构建器，
生成6张中间表 Parquet 文件，并执行验收检查。

使用方式:
    cd src/
    python build_dims.py
"""

import sys
from pathlib import Path

# 将 src 目录加入 Python 路径，支持直接运行
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from common.agg_specs import (
    DIM_CATEGORY_PATH,
    DIM_ITEM_PATH,
    DIM_OUTPUT_DIR,
    DIM_TIME_DAILY_PATH,
    DIM_TIME_HOURLY_PATH,
    DIM_TIME_WEEKDAY_HOUR_PATH,
    DIM_USER_PATH,
)
from common.constants import (
    BEHAVIOR_BUY,
    BEHAVIOR_CART,
    BEHAVIOR_FAV,
    BEHAVIOR_PV,
)
from common.logger import get_logger
from config import get_config
from dim_builders.item_dim_builder import build_category_dim, build_item_dim
from dim_builders.time_dim_builder import build_time_dims
from dim_builders.user_dim_builder import build_user_dim

logger = get_logger("build_dims")

# 行为类型 → 名称映射，用于验收日志
_BEHAVIOR_NAMES = {
    BEHAVIOR_PV: "浏览",
    BEHAVIOR_FAV: "收藏",
    BEHAVIOR_CART: "加购",
    BEHAVIOR_BUY: "购买",
}


def _save_parquet(df: pd.DataFrame, path: Path, index_col: str | list[str]) -> None:
    """设置索引并保存为 Parquet。

    sort + set_index 实现索引建立，Parquet 列式存储自带 min/max 索引，
    叠加行组元数据加速后续按主键查询。

    Args:
        df: 要保存的 DataFrame
        path: 输出路径
        index_col: 主键列名（单列str 或 多列list）
    """
    df = df.sort_values(index_col).set_index(index_col)
    df.to_parquet(path)
    size_mb = path.stat().st_size / 1024 / 1024
    logger.info("已保存 %s: %d 行, %.1f MB", path.name, len(df), size_mb)


def _verify_behavior_consistency(
    source_df: pd.DataFrame, tables: dict[str, pd.DataFrame]
) -> None:
    """验收检查: 中间表行为量总和与原始数据一致。

    每种行为的总次数在原始数据和中间表中应该守恒。

    Args:
        source_df: 原始清洗数据
        tables: {表名: DataFrame} 字典
    """
    source_counts = source_df["behavior_type"].value_counts()

    # dim_user 的行为量应与原始数据一致
    user_df = tables["dim_user"]
    for bt, name in _BEHAVIOR_NAMES.items():
        col = f"{name}_count" if False else None
        # 字段名映射
        col_map = {
            BEHAVIOR_PV: "pv_count",
            BEHAVIOR_FAV: "fav_count",
            BEHAVIOR_CART: "cart_count",
            BEHAVIOR_BUY: "buy_count",
        }
        col = col_map[bt]
        source_total = source_counts.get(bt, 0)
        dim_total = user_df[col].sum()
        match = "✓" if source_total == dim_total else "✗"
        logger.info(
            "  行为量守恒[%s]: %s 源=%d, dim_user=%d",
            match,
            name,
            source_total,
            dim_total,
        )
        if source_total != dim_total:
            logger.warning(
                "  ⚠ %s 行为量不一致! 源=%d, dim_user=%d",
                name,
                source_total,
                dim_total,
            )


def _verify_primary_keys(tables: dict[str, pd.DataFrame]) -> None:
    """验收检查: 每张表主键无重复。

    Args:
        tables: {表名: DataFrame} 字典
    """
    pk_map = {
        "dim_user": "user_id",
        "dim_item": "item_id",
        "dim_category": "item_category",
        "dim_time_hourly": ["date", "hour"],
        "dim_time_daily": "date",
        "dim_time_weekday_hour": ["weekday", "hour"],
    }
    for table_name, pk in pk_map.items():
        df = tables[table_name]
        if isinstance(pk, str):
            n_unique = df[pk].nunique()
        else:
            n_unique = df.groupby(pk).ngroups
        match = "✓" if n_unique == len(df) else "✗"
        logger.info(
            "  主键唯一[%s]: %s %s nunique=%d, rows=%d",
            match,
            table_name,
            pk,
            n_unique,
            len(df),
        )


def _verify_no_nulls(tables: dict[str, pd.DataFrame]) -> None:
    """验收检查: 关键字段无缺失值。

    Args:
        tables: {表名: DataFrame} 字典
    """
    for table_name, df in tables.items():
        null_counts = df.isnull().sum()
        has_null = null_counts[null_counts > 0]
        if has_null.empty:
            logger.info("  无缺失[%s]: ✓ %s", table_name, table_name)
        else:
            logger.warning(
                "  ⚠ %s 存在缺失值: %s", table_name, has_null.to_dict()
            )


def run_build_dims() -> None:
    """执行第二阶段中间表构建完整流程。

    流程步骤:
        1. 读取 Part1 清洗后的 Parquet 数据
        2. 调用3个维度构建器生成6张中间表
        3. 设置索引并保存为 Parquet
        4. 执行验收检查（主键唯一、行为量守恒、无缺失值）
    """
    config = get_config()
    logger.info("=" * 60)
    logger.info("第二阶段：多维度中间表构建")
    logger.info("=" * 60)

    # 步骤1: 读取 Part1 产物
    logger.info("读取清洗后数据: %s", config.cleaned_data_path)
    df = pd.read_parquet(config.cleaned_data_path)
    logger.info("输入数据: %d 行, %d 列", len(df), len(df.columns))

    # 确保 is_power_user 为 bool 类型
    if "is_power_user" in df.columns:
        df["is_power_user"] = df["is_power_user"].astype(bool)

    # 步骤2: 构建三张维度表
    logger.info("-" * 40)
    logger.info("构建用户维度表 dim_user...")
    dim_user = build_user_dim(df)

    logger.info("-" * 40)
    logger.info("构建商品维度表 dim_item + dim_category...")
    dim_item = build_item_dim(df)
    dim_category = build_category_dim(df)

    logger.info("-" * 40)
    logger.info("构建时间维度表 dim_time × 3...")
    time_dims = build_time_dims(df)
    dim_time_hourly = time_dims["hourly"]
    dim_time_daily = time_dims["daily"]
    dim_time_weekday_hour = time_dims["weekday_hour"]

    # 步骤3: 保存6张 Parquet（sort + set_index 建索引）
    logger.info("-" * 40)
    logger.info("保存中间表 Parquet 文件...")
    _save_parquet(dim_user, DIM_USER_PATH, "user_id")
    _save_parquet(dim_item, DIM_ITEM_PATH, "item_id")
    _save_parquet(dim_category, DIM_CATEGORY_PATH, "item_category")
    _save_parquet(dim_time_hourly, DIM_TIME_HOURLY_PATH, ["date", "hour"])
    _save_parquet(dim_time_daily, DIM_TIME_DAILY_PATH, "date")
    _save_parquet(
        dim_time_weekday_hour, DIM_TIME_WEEKDAY_HOUR_PATH, ["weekday", "hour"]
    )

    # 步骤4: 验收检查
    logger.info("-" * 40)
    logger.info("验收检查:")

    all_tables = {
        "dim_user": dim_user,
        "dim_item": dim_item,
        "dim_category": dim_category,
        "dim_time_hourly": dim_time_hourly,
        "dim_time_daily": dim_time_daily,
        "dim_time_weekday_hour": dim_time_weekday_hour,
    }

    logger.info("  [1/3] 主键唯一性检查:")
    _verify_primary_keys(all_tables)

    logger.info("  [2/3] 行为量守恒检查:")
    _verify_behavior_consistency(df, all_tables)

    logger.info("  [3/3] 缺失值检查:")
    _verify_no_nulls(all_tables)

    logger.info("=" * 60)
    logger.info("第二阶段中间表构建完成")
    logger.info("  dim_user:           %d 行", len(dim_user))
    logger.info("  dim_item:           %d 行", len(dim_item))
    logger.info("  dim_category:       %d 行", len(dim_category))
    logger.info("  dim_time_hourly:    %d 行", len(dim_time_hourly))
    logger.info("  dim_time_daily:     %d 行", len(dim_time_daily))
    logger.info("  dim_time_weekday_hour: %d 行", len(dim_time_weekday_hour))
    logger.info("=" * 60)


if __name__ == "__main__":
    run_build_dims()

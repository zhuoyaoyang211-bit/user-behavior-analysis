"""中间表聚合指标常量定义模块。

集中管理步骤二6张中间表的字段名、输出路径、时段划分阈值，
供3个 dim_builder 引用，避免字段名硬编码散落各处。

中间表清单:
    - dim_user: 用户维度（1万行，一行=一个用户）
    - dim_item: 商品单品维度（287万行，全量不过滤）
    - dim_category: 商品类目维度（8916行，全量）
    - dim_time_hourly: 小时级时间维度（744行，31天×24h）
    - dim_time_daily: 日级时间维度（31行）
    - dim_time_weekday_hour: 周×小时时间维度（168行，7天×24h）
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# 时段划分阈值（用于 dim_user 的时段偏好占比计算）
# 白天 6-17点 | 晚间 18-23点 | 深夜 0-5点
# ---------------------------------------------------------------------------
DAY_HOURS: range = range(6, 18)
EVENING_HOURS: range = range(18, 24)
NIGHT_HOURS: range = range(0, 6)

# ---------------------------------------------------------------------------
# 输出路径：6张中间表 Parquet 文件
# ---------------------------------------------------------------------------
DIM_OUTPUT_DIR: Path = Path("output")
DIM_USER_PATH: Path = DIM_OUTPUT_DIR / "dim_user.parquet"
DIM_ITEM_PATH: Path = DIM_OUTPUT_DIR / "dim_item.parquet"
DIM_CATEGORY_PATH: Path = DIM_OUTPUT_DIR / "dim_category.parquet"
DIM_TIME_HOURLY_PATH: Path = DIM_OUTPUT_DIR / "dim_time_hourly.parquet"
DIM_TIME_DAILY_PATH: Path = DIM_OUTPUT_DIR / "dim_time_daily.parquet"
DIM_TIME_WEEKDAY_HOUR_PATH: Path = (
    DIM_OUTPUT_DIR / "dim_time_weekday_hour.parquet"
)

# ---------------------------------------------------------------------------
# dim_user 字段定义（17列）
# ---------------------------------------------------------------------------
DIM_USER_COLUMNS: list[str] = [
    "user_id",                 # 主键
    "pv_count",                # 浏览次数
    "fav_count",               # 收藏次数
    "cart_count",              # 加购次数
    "buy_count",               # 购买次数
    "active_days",             # 活跃天数
    "first_active_time",       # 首次行为时间
    "last_active_time",        # 末次行为时间
    "day_pct",                 # 白天行为占比
    "evening_pct",             # 晚间行为占比
    "night_pct",               # 深夜行为占比
    "buy_item_count",          # 购买商品数(去重)
    "buy_conversion_rate",     # 购买转化率
    "fav_to_buy_rate",         # 收藏→购买转化率
    "cart_to_buy_rate",        # 加购→购买转化率
    "repurchase_item_count",   # 复购商品数
    "is_power_user",           # 是否重度买家(复用Part1标记)
]

# ---------------------------------------------------------------------------
# dim_item 字段定义（11列）
# ---------------------------------------------------------------------------
DIM_ITEM_COLUMNS: list[str] = [
    "item_id",                 # 主键
    "item_category",           # 外键→dim_category
    "pv_count",                # 浏览次数
    "view_user_count",         # 浏览独立用户数
    "fav_count",               # 收藏次数
    "cart_count",              # 加购次数
    "buy_count",               # 购买次数
    "buy_user_count",          # 购买独立用户数
    "pv_to_buy_rate",          # 浏览→购买转化率
    "cart_to_buy_rate",        # 加购→购买转化率
    "repurchase_user_count",   # 复购用户数
]

# ---------------------------------------------------------------------------
# dim_category 字段定义（11列）
# ---------------------------------------------------------------------------
DIM_CATEGORY_COLUMNS: list[str] = [
    "item_category",           # 主键
    "item_count",              # 该类目下商品总数
    "buy_item_count",          # 有购买的商品数
    "pv_count",                # 类目总浏览次数
    "view_user_count",         # 类目总浏览独立用户数
    "fav_count",               # 类目总收藏次数
    "cart_count",              # 类目总加购次数
    "buy_count",               # 类目总购买次数
    "buy_user_count",          # 类目总购买独立用户数
    "pv_to_buy_rate",          # 浏览→购买转化率
    "buy_item_pct",            # 有购买商品占比
]

# ---------------------------------------------------------------------------
# dim_time 三张表统一字段定义（6列，不含主键列）
# 主键列由各 builder 自行添加（date/hour/weekday）
# ---------------------------------------------------------------------------
DIM_TIME_METRIC_COLUMNS: list[str] = [
    "pv_count",                # 浏览次数
    "fav_count",               # 收藏次数
    "cart_count",              # 加购次数
    "buy_count",               # 购买次数
    "active_user_count",       # 独立活跃用户数
    "active_item_count",       # 独立活跃商品数
]

# ---------------------------------------------------------------------------
# 中间表元信息：表名 → (输出路径, 行数预估)
# 供 build_dims.py 验收时检查
# ---------------------------------------------------------------------------
DIM_TABLE_SPEC: dict[str, tuple[Path, int]] = {
    "dim_user": (DIM_USER_PATH, 10_000),
    "dim_item": (DIM_ITEM_PATH, 2_876_947),
    "dim_category": (DIM_CATEGORY_PATH, 8_916),
    "dim_time_hourly": (DIM_TIME_HOURLY_PATH, 744),
    "dim_time_daily": (DIM_TIME_DAILY_PATH, 31),
    "dim_time_weekday_hour": (DIM_TIME_WEEKDAY_HOUR_PATH, 168),
}

"""项目全局常量定义模块。

集中管理行为类型映射、数据列名、时间格式等全项目共享常量，
避免魔法数字和硬编码字符串散落在各业务模块中。
"""

# 行为类型映射：1=浏览, 2=收藏, 3=加购物车, 4=购买
BEHAVIOR_PV = 1
BEHAVIOR_FAV = 2
BEHAVIOR_CART = 3
BEHAVIOR_BUY = 4

# 行为类型到中文名称的映射字典，用于可视化与报告
BEHAVIOR_NAME_MAP: dict[int, str] = {
    BEHAVIOR_PV: "浏览",
    BEHAVIOR_FAV: "收藏",
    BEHAVIOR_CART: "加购物车",
    BEHAVIOR_BUY: "购买",
}

# 合法行为类型集合，用于数据校验时快速判断
VALID_BEHAVIOR_TYPES: set[int] = {BEHAVIOR_PV, BEHAVIOR_FAV, BEHAVIOR_CART, BEHAVIOR_BUY}

# 原始数据列名定义
COLUMN_TIME = "time"
COLUMN_USER_ID = "user_id"
COLUMN_ITEM_ID = "item_id"
COLUMN_ITEM_CATEGORY = "item_category"
COLUMN_BEHAVIOR_TYPE = "behavior_type"

# 时间格式：原始数据格式为 "YYYY-MM-DD HH"
RAW_TIME_FORMAT = "%Y-%m-%d %H"
# 标准化后的时间格式
STD_TIME_FORMAT = "%Y-%m-%d %H:00:00"

# 必要字段表：字段名 → 下游用途说明
# 非必要字段筛选的依据：不在此表中的列将被移除
# 每个字段都标注在后续哪个阶段用到，确保筛选有据可查
REQUIRED_FIELDS: dict[str, str] = {
    COLUMN_TIME: "时间维度特征提取、行为序列建模、标签定义的时间窗口",
    COLUMN_USER_ID: "用户维度聚合、RFM价值分、用户-商品对样本构建",
    COLUMN_ITEM_ID: "商品维度聚合、SVD隐向量分解、用户-商品对样本构建",
    COLUMN_ITEM_CATEGORY: "商品类目特征、类目偏好分析、跨商品关联",
    COLUMN_BEHAVIOR_TYPE: "行为漏斗统计、标签定义(购买=正样本)、行为序列特征",
}

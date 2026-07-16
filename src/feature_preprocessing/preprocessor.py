"""特征预处理核心逻辑。

5 步处理流程：
    1. 删除 2 列冗余 datetime（first_active_time / last_active_time）
    2. 填充 3 列缺失值（填 0 或中位数）
    3. 目标编码 2 列高维类别（item_category / user_id → 购买率）
    4. 标准化 42 列数值（StandardScaler 均值 0 方差 1）
    5. 保留 5 列不动（is_power_user / buy_path_type / 原始主键 user_id+item_id）

输入：output/feature_wide_table.parquet（4,686,904 行 × 47 列）
输出：output/processed_features.parquet（4,686,904 行 × 47 列）
"""

from __future__ import annotations

import gc

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ── 第①步：要删除的列 ──────────────────────────────────────────
DROP_COLUMNS: list[str] = [
    # "item_id" 已移除：用户-商品对是主键，删除会丢失样本标识
    "first_active_time",   # datetime，信息已被 active_days 覆盖
    "last_active_time",    # datetime，信息已被 rfm_r_score 覆盖
]

# ── 第②步：缺失值填充策略 ─────────────────────────────────────
# 值为 "median" 表示填该列中位数，否则填指定常量
FILL_STRATEGY: dict[str, str | float] = {
    "item_decay_slope": 0,             # 61.5% 缺失，0 = 无趋势
    "user_category_pref_score": 0,     # 74.8% 缺失，0 = 无偏好记录
    "user_avg_interval_hours": "median",  # 0.005% 缺失，填中位数
}

# ── 第④步：不参与标准化的列 ───────────────────────────────────
# 这些列保持原值，不做 StandardScaler
COLUMNS_NO_SCALE: list[str] = [
    "user_id",           # 原始主键，保留做关联，不参与建模
    "item_id",           # 原始主键，保留做关联，不参与建模
    "item_category",     # 原始类别列，保留做关联，不参与建模
    "is_power_user",     # 0/1 二值，标准化破坏语义
    "buy_path_type",     # 目标变量，预处理阶段绝对不能动
]


def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    """删除 2 列冗余特征。

    Args:
        df: 原始特征宽表。

    Returns:
        删除指定列后的 DataFrame。
    """
    cols_exist = [c for c in DROP_COLUMNS if c in df.columns]
    df = df.drop(columns=cols_exist)
    return df


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """填充 3 列缺失值。

    - item_decay_slope / user_category_pref_score → 填 0
    - user_avg_interval_hours → 填中位数

    Args:
        df: 删除冗余列后的 DataFrame。

    Returns:
        缺失值已填充的 DataFrame。
    """
    for col, strategy in FILL_STRATEGY.items():
        if col not in df.columns:
            continue
        if strategy == "median":
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
        else:
            df[col] = df[col].fillna(strategy)
    return df


def target_encode(df: pd.DataFrame) -> pd.DataFrame:
    """对 item_category 和 user_id 做目标编码。

    编码逻辑：用 buy_path_type > 0 作为临时二分类 target，
    算每个类别（类目 / 用户）的历史购买率，生成 2 列新数值。

    - item_category → item_category_te（该类目下被购买的比例）
    - user_id → user_id_te（该用户购买过的比例）

    原始列保留做关联，新列参与后续标准化。

    Args:
        df: 缺失值已填充的 DataFrame，需含 buy_path_type 列。

    Returns:
        新增 2 列目标编码值的 DataFrame。
    """
    # 临时 target：buy_path_type 0=没买, 1/2/3/4=买了 → 统一变 0/1
    target = (df["buy_path_type"] > 0).astype(np.int8)

    # item_category 目标编码：每个类目的历史购买率
    cat_rate = (
        pd.DataFrame({"cat": df["item_category"], "y": target})
        .groupby("cat")["y"]
        .mean()
    )
    df["item_category_te"] = (
        df["item_category"].map(cat_rate).astype(np.float32)
    )

    # user_id 目标编码：每个用户的历史购买率
    user_rate = (
        pd.DataFrame({"uid": df["user_id"], "y": target})
        .groupby("uid")["y"]
        .mean()
    )
    df["user_id_te"] = df["user_id"].map(user_rate).astype(np.float32)

    return df


def standardize(df: pd.DataFrame) -> pd.DataFrame:
    """对数值列做 StandardScaler 标准化（均值 0 方差 1）。

    排除 COLUMNS_NO_SCALE 中的 5 列，其余数值列全部标准化。

    Args:
        df: 目标编码后的 DataFrame。

    Returns:
        数值列已标准化的 DataFrame。
    """
    # 确定要标准化的列：数值类型 + 不在排除名单中
    no_scale_set = set(COLUMNS_NO_SCALE)
    scale_cols = [
        c for c in df.columns
        if c not in no_scale_set
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    scaler = StandardScaler()
    df[scale_cols] = scaler.fit_transform(df[scale_cols])
    # StandardScaler 输出 float64，降回 float32 省内存
    for c in scale_cols:
        df[c] = df[c].astype(np.float32)

    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """特征预处理主流程：5 步串联。

    Args:
        df: Part3 产出的特征宽表（4,686,904 行 × 47 列）。

    Returns:
        预处理后的 DataFrame（4,686,904 行 × 47 列），
        全数值、无缺失、无 datetime，可直接喂给特征筛选。
    """
    # ① 删除 2 列冗余特征
    df = drop_columns(df)
    gc.collect()

    # ② 填充 3 列缺失值
    df = fill_missing(df)

    # ③ 目标编码 2 列高维类别
    df = target_encode(df)

    # ④ 标准化 42 列数值
    df = standardize(df)

    # ⑤ 5 列不动（is_power_user / buy_path_type / 原始主键 user_id+item_id）
    #    无需额外操作，已在 COLUMNS_NO_SCALE 中排除

    return df

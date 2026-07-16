"""三轮特征筛选模块。

对预处理后的特征数据执行：
    1. 方差阈值法 — 剔除几乎不变的列（方差 < 0.01）
    2. 互信息法 — 剔除与目标变量关联极弱的列（后 25%）
    3. 相关性分析 — 剔除高度共线的冗余列（|r| > 0.95）

筛选对象：43 列数值特征（排除 user_id / item_category / buy_path_type）
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

from common.logger import get_logger

logger = get_logger(__name__)

# 不参与特征筛选的列
EXCLUDE_COLS = {"user_id", "item_id", "item_category", "buy_path_type"}

# 方差阈值：标准化后方差 < 此值视为伪常数
VARIANCE_THRESHOLD = 0.01

# 相关性阈值：|r| > 此值视为高度共线
CORRELATION_THRESHOLD = 0.95

# 互信息分位阈值：删掉 MI 排在后此比例的特征
MI_QUANTILE_THRESHOLD = 0.25

# 互信息计算采样数：全量 468 万行太慢，采样 20 万行足够估算
MI_SAMPLE_SIZE = 200_000


def _get_feature_cols(df: pd.DataFrame) -> list[str]:
    """从 DataFrame 中提取参与筛选的特征列名。"""
    return [c for c in df.columns if c not in EXCLUDE_COLS]


# ===========================================================================
# 第 1 轮：方差阈值法
# ===========================================================================
def _variance_filter(
    df: pd.DataFrame, feature_cols: list[str]
) -> tuple[list[str], list[str]]:
    """方差阈值筛选。

    Args:
        df: 全量 DataFrame（含所有列）
        feature_cols: 当前待筛选的特征列名列表

    Returns:
        (passing, dropped): 通过的和被筛掉的列名列表
    """
    variances = df[feature_cols].var()
    passing = [c for c in feature_cols if variances[c] >= VARIANCE_THRESHOLD]
    dropped = [c for c in feature_cols if c not in passing]

    logger.info(
        "方差阈值法: %d → %d 列 (阈值=%s)",
        len(feature_cols),
        len(passing),
        VARIANCE_THRESHOLD,
    )
    if dropped:
        for c in dropped:
            logger.info("  删除: %-35s 方差=%.6f", c, variances[c])

    return passing, dropped


# ===========================================================================
# 第 2 轮：互信息法
# ===========================================================================
def _mutual_info_filter(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "buy_path_type",
) -> tuple[list[str], list[str], pd.Series]:
    """互信息筛选。

    Args:
        df: 全量 DataFrame（含目标列）
        feature_cols: 当前待筛选的特征列名列表
        target_col: 目标变量列名

    Returns:
        (passing, dropped, mi_scores): 通过的列、被筛掉的列、全部 MI 分数
    """
    # 采样：全量 468 万行算 k-NN 互信息太慢，分层采样 20 万行足够估算
    n_total = len(df)
    if n_total > MI_SAMPLE_SIZE:
        sample = pd.concat(
            [
                g.sample(
                    n=max(1, int(MI_SAMPLE_SIZE * len(g) / n_total)),
                    random_state=42,
                )
                for _, g in df.groupby(target_col)
            ],
            ignore_index=True,
        )
        logger.info("  互信息采样: %d → %d 行", n_total, len(sample))
    else:
        sample = df

    X = sample[feature_cols].values.astype(np.float64)
    y = sample[target_col].values

    mi_scores = mutual_info_classif(X, y, random_state=42)
    mi_series = pd.Series(mi_scores, index=feature_cols).sort_values(ascending=False)

    threshold = mi_series.quantile(MI_QUANTILE_THRESHOLD)
    passing = [c for c in feature_cols if mi_series[c] >= threshold]
    dropped = [c for c in feature_cols if c not in passing]

    logger.info(
        "互信息法: %d → %d 列 (阈值=%.6f, 即删掉后%d%%)",
        len(feature_cols),
        len(passing),
        threshold,
        int(MI_QUANTILE_THRESHOLD * 100),
    )
    if dropped:
        for c in dropped:
            logger.info("  删除: %-35s MI=%.6f", c, mi_series[c])

    return passing, dropped, mi_series


# ===========================================================================
# 第 3 轮：相关性分析
# ===========================================================================
def _correlation_filter(
    df: pd.DataFrame,
    feature_cols: list[str],
    mi_series: pd.Series,
) -> tuple[list[str], list[tuple[str, str, float]]]:
    """相关性筛选。

    找出 |r| > 阈值的特征对，每对中删掉互信息更低的那列。

    Args:
        df: 全量 DataFrame
        feature_cols: 当前待筛选的特征列名列表
        mi_series: 互信息分数 Series（用于决定每对中删哪个）

    Returns:
        (passing, dropped_pairs): 通过的列、被筛掉的对 (col_a, col_b, r)
    """
    corr_matrix = df[feature_cols].corr().abs()
    to_drop: set[str] = set()
    dropped_pairs: list[tuple[str, str, float]] = []

    # 只遍历上三角，避免重复处理
    for i in range(len(feature_cols)):
        for j in range(i + 1, len(feature_cols)):
            col_a, col_b = feature_cols[i], feature_cols[j]
            r = corr_matrix.loc[col_a, col_b]

            if r <= CORRELATION_THRESHOLD:
                continue
            if col_a in to_drop or col_b in to_drop:
                continue

            # 删互信息更低的那个
            if mi_series.get(col_a, 0) >= mi_series.get(col_b, 0):
                to_drop.add(col_b)
                dropped_pairs.append((col_a, col_b, r))
            else:
                to_drop.add(col_a)
                dropped_pairs.append((col_b, col_a, r))

    passing = [c for c in feature_cols if c not in to_drop]

    logger.info(
        "相关性分析: %d → %d 列 (阈值=%.2f)",
        len(feature_cols),
        len(passing),
        CORRELATION_THRESHOLD,
    )
    for keep, drop, r in dropped_pairs:
        logger.info("  保留: %-35s  删除: %-35s  |r|=%.4f", keep, drop, r)

    return passing, dropped_pairs


# ===========================================================================
# 主函数
# ===========================================================================
def select_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """执行三轮特征筛选。

    Args:
        df: 预处理后的特征 DataFrame（需含 user_id / buy_path_type）

    Returns:
        (selected_df, summary):
            selected_df — 筛选后的 DataFrame（user_id + 保留特征 + buy_path_type）
            summary    — 每轮筛选详情字典
    """
    feature_cols = _get_feature_cols(df)
    logger.info("参与筛选的特征列: %d 列", len(feature_cols))

    # 第 1 轮：方差阈值
    cols_vt, dropped_vt = _variance_filter(df, feature_cols)

    # 第 2 轮：互信息
    cols_mi, dropped_mi, mi_series = _mutual_info_filter(df, cols_vt)

    # 第 3 轮：相关性
    cols_final, dropped_corr_pairs = _correlation_filter(df, cols_mi, mi_series)

    # 构建输出
    output_cols = ["user_id", "item_id"] + cols_final + ["buy_path_type"]
    selected_df = df[output_cols].copy()

    summary = {
        "initial": len(feature_cols),
        "after_variance": len(cols_vt),
        "variance_dropped": dropped_vt,
        "after_mi": len(cols_mi),
        "mi_dropped": dropped_mi,
        "mi_scores": mi_series,
        "after_correlation": len(cols_final),
        "correlation_dropped_pairs": dropped_corr_pairs,
        "final_count": len(cols_final),
        "final_columns": cols_final,
    }

    logger.info(
        "特征筛选完成: %d → %d 列 (方差%d + 互信息%d + 相关性%d)",
        len(feature_cols),
        len(cols_final),
        len(dropped_vt),
        len(dropped_mi),
        len(dropped_corr_pairs),
    )

    return selected_df, summary

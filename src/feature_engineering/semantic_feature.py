"""隐式语义特征模块。

输出字段（1个）：
    user_item_svd_score : 用户对该商品的隐向量匹配分
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD


# SVD 隐向量维度
SVD_N_COMPONENTS = 10


def calc_user_item_svd_score(
    df: pd.DataFrame,
    target_pairs: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """计算用户对商品的 SVD 隐向量匹配分。

    算法：
        1. 用 cleaned_data 构建 用户×商品 行为矩阵，值为综合热度 (pv×1+...+buy×5)
        2. TruncatedSVD 分解得到用户隐向量 (n_users × n_components) 和
           商品隐向量 (n_items × n_components)
        3. 对目标 (user_id, item_id) 对算点积

    Args:
        df: 原始明细数据。
        target_pairs: 目标(用户,商品)对 DataFrame，含 user_id 和 item_id 列。
            若为 None 则算所有(用户,商品)对（内存可能爆）。

    Returns:
        DataFrame, 列 = [user_id, item_id, user_item_svd_score]
    """
    # 1. 算综合热度
    df = df.copy()
    df["heat"] = (
        (df["behavior_type"] == 1).astype(int) * 1
        + (df["behavior_type"] == 2).astype(int) * 2
        + (df["behavior_type"] == 3).astype(int) * 3
        + (df["behavior_type"] == 4).astype(int) * 5
    )
    user_item_heat = df.groupby(["user_id", "item_id"])["heat"].sum().reset_index()

    # 2. 建稀疏矩阵
    user_ids = user_item_heat["user_id"].unique()
    item_ids = user_item_heat["item_id"].unique()
    user_id_map = {u: i for i, u in enumerate(user_ids)}
    item_id_map = {it: i for i, it in enumerate(item_ids)}

    rows = user_item_heat["user_id"].map(user_id_map)
    cols = user_item_heat["item_id"].map(item_id_map)
    data = user_item_heat["heat"].values.astype(np.float32)

    matrix = csr_matrix(
        (data, (rows, cols)),
        shape=(len(user_ids), len(item_ids)),
    )

    # 3. SVD 分解
    n_comp = min(SVD_N_COMPONENTS, min(matrix.shape) - 1)
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    user_vec = svd.fit_transform(matrix)        # (n_users, n_comp)
    item_vec = svd.components_.T                 # (n_items, n_comp)

    # 4. 算目标对的点积
    if target_pairs is None:
        target_pairs = user_item_heat[["user_id", "item_id"]].copy()
    else:
        target_pairs = target_pairs[["user_id", "item_id"]].copy()

    # 向量化算点积
    target_user_idx = target_pairs["user_id"].map(user_id_map)
    target_item_idx = target_pairs["item_id"].map(item_id_map)
    valid_mask = target_user_idx.notna() & target_item_idx.notna()

    scores = np.full(len(target_pairs), np.nan, dtype=np.float32)
    scores[valid_mask.values] = (
        user_vec[target_user_idx[valid_mask].astype(int)]
        * item_vec[target_item_idx[valid_mask].astype(int)]
    ).sum(axis=1)

    result = target_pairs.copy()
    result["user_item_svd_score"] = scores
    return result

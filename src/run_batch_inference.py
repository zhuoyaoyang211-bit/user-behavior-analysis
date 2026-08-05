"""XGBoost 本地批量推理脚本。

该脚本将原始用户行为数据到购买概率预测封装为一条可调用链路：
    1. 分块读取 CSV/Parquet 原始行为数据
    2. 执行字段校验、行为类型过滤、时间标准化等基础清洗
    3. 增量计算用户、商品、类目、生命周期和业务特征
    4. 按训练阶段的预处理参数对齐 XGBoost 模型特征
    5. 分批输出用户-商品购买概率、预测标签和推理元信息

候选文件中的商品必须能在原始行为数据中匹配到类目。默认匹配失败直接报错，
只有显式指定 ``--candidate-failure-policy drop`` 才会丢弃失败候选。

示例:
    python src/run_batch_inference.py \
        --raw-path user_behavior_processed.csv \
        --output-dir output/inference/xgboost_optuna_batch
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as parquet
from sklearn.preprocessing import StandardScaler

# 支持从项目根目录直接执行: python src/run_batch_inference.py
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.logger import get_logger
from data_cleaner import DataCleaner
from inference_feature_builder import RawBehaviorAggregator


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = (
    PROJECT_DIR / "output" / "optuna_tuned_models" / "xgboost_optuna.joblib"
)
DEFAULT_RAW_PATH = PROJECT_DIR / "user_behavior_processed.csv"
DEFAULT_REFERENCE_WIDE_PATH = PROJECT_DIR / "output" / "feature_wide_table.parquet"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "output" / "inference" / "xgboost_optuna_batch"
DEFAULT_PREPROCESSOR_PATH = (
    PROJECT_DIR / "output" / "inference" / "xgboost_inference_preprocessor.joblib"
)
ID_COLS = ["user_id", "item_id"]
TIME_COL = "time"
TARGET_COL = "buy_path_type"
LABEL_COL = "prediction_label"
PROB_COL = "purchase_probability"
RANDOM_STATE = 42
DEFAULT_CHUNK_SIZE = 1_000_000
RAW_DTYPE = {
    "user_id": "int32",
    "item_id": "int32",
    "item_category": "int16",
    "behavior_type": "int8",
}

logger = get_logger(__name__)


@dataclass(frozen=True)
class InferencePreprocessor:
    """推理阶段复用的预处理参数。

    Attributes:
        feature_cols: 模型训练时使用的特征列名。
        fill_values: 缺失值填充规则，来自训练宽表口径。
        item_category_te: item_category 的目标编码映射。
        item_category_te_default: 未见类目的目标编码兜底值。
        means: StandardScaler 的均值参数。
        scales: StandardScaler 的缩放参数。
    """

    feature_cols: list[str]
    fill_values: dict[str, float]
    item_category_te: dict[int, float]
    item_category_te_default: float
    means: dict[str, float]
    scales: dict[str, float]


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    Returns:
        argparse.Namespace: 命令行参数集合。
    """
    parser = argparse.ArgumentParser(
        description="Run local batch inference with optimized XGBoost model."
    )
    parser.add_argument("--raw-path", type=Path, default=DEFAULT_RAW_PATH)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--output-name",
        default="purchase_probability_predictions.csv",
        help="Prediction CSV file name under output-dir.",
    )
    parser.add_argument(
        "--candidate-path",
        type=Path,
        default=None,
        help=(
            "Optional CSV/Parquet containing user_id,item_id pairs. "
            "When omitted, all observed pairs in raw behavior data are predicted."
        ),
    )
    parser.add_argument(
        "--reference-wide-path",
        type=Path,
        default=DEFAULT_REFERENCE_WIDE_PATH,
        help="Training feature_wide_table used to fit inference preprocessing stats.",
    )
    parser.add_argument(
        "--preprocessor-path",
        type=Path,
        default=DEFAULT_PREPROCESSOR_PATH,
        help="Cached inference preprocessor path.",
    )
    parser.add_argument(
        "--rebuild-preprocessor",
        action="store_true",
        help="Rebuild preprocessing stats from reference-wide-path.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Classification threshold. Defaults to selected_threshold in model artifact.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500_000,
        help="Rows per probability prediction batch.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Rows per raw CSV/Parquet ingestion chunk.",
    )
    parser.add_argument(
        "--candidate-failure-policy",
        choices=("error", "drop"),
        default="error",
        help=(
            "How to handle candidate pairs without a matching item category. "
            "The default error policy prevents silent data loss."
        ),
    )
    parser.add_argument(
        "--limit-rows",
        type=int,
        default=None,
        help="Optional row limit for smoke testing raw data ingestion.",
    )
    return parser.parse_args()


def read_table(path: Path, limit_rows: int | None = None) -> pd.DataFrame:
    """读取 CSV 或 Parquet 数据表。

    Args:
        path: 输入文件路径，支持 .csv、.parquet、.pq。
        limit_rows: 可选行数上限，仅 CSV 使用 nrows，Parquet 读入后截断。

    Returns:
        读取后的 DataFrame。

    Raises:
        FileNotFoundError: 输入文件不存在。
        ValueError: 文件后缀不受支持。
    """
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, nrows=limit_rows)
    if suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
        if limit_rows is not None:
            df = df.head(limit_rows).copy()
        return df
    raise ValueError(f"Unsupported file suffix: {suffix}")


def iter_table_chunks(
    path: Path,
    chunk_size: int,
    limit_rows: int | None = None,
    columns: list[str] | None = None,
) -> Iterator[pd.DataFrame]:
    """以分块方式读取 CSV 或 Parquet。

    Args:
        path: 输入文件路径。
        chunk_size: 每个分块的最大行数。
        limit_rows: 可选的读取行数上限。
        columns: 可选列投影，仅 Parquet 和 CSV 均支持。

    Yields:
        一个个数据分块。

    Raises:
        FileNotFoundError: 输入文件不存在。
        ValueError: chunk_size 或文件后缀不合法。
    """
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    suffix = path.suffix.lower()
    rows_read = 0
    if suffix == ".csv":
        reader = pd.read_csv(
            path,
            chunksize=chunk_size,
            dtype=RAW_DTYPE,
            usecols=columns,
        )
        for chunk in reader:
            if limit_rows is not None:
                remaining = limit_rows - rows_read
                if remaining <= 0:
                    break
                chunk = chunk.head(remaining).copy()
            if chunk.empty:
                break
            rows_read += len(chunk)
            yield chunk
            if limit_rows is not None and rows_read >= limit_rows:
                break
        return

    if suffix in {".parquet", ".pq"}:
        parquet_file = parquet.ParquetFile(path)
        for record_batch in parquet_file.iter_batches(
            batch_size=chunk_size,
            columns=columns,
        ):
            chunk = record_batch.to_pandas()
            if limit_rows is not None:
                remaining = limit_rows - rows_read
                if remaining <= 0:
                    break
                chunk = chunk.head(remaining).copy()
            if chunk.empty:
                break
            rows_read += len(chunk)
            yield chunk
            if limit_rows is not None and rows_read >= limit_rows:
                break
        return

    raise ValueError(f"Unsupported file suffix: {suffix}")


def aggregate_behavior_chunks(
    path: Path,
    chunk_size: int,
    limit_rows: int | None = None,
) -> tuple[RawBehaviorAggregator, dict[str, int]]:
    """分块清洗并聚合原始行为数据。

    该函数不会拼接完整清洗明细，只保留模型特征所需的聚合中间结果。

    Args:
        path: 原始行为数据路径。
        chunk_size: 单个读取分块行数。
        limit_rows: 可选的读取行数上限。

    Returns:
        特征聚合器和读取统计信息。
    """
    cleaner = DataCleaner()
    aggregator = RawBehaviorAggregator()
    raw_rows = 0
    cleaned_rows = 0
    chunk_count = 0

    for chunk_count, raw_chunk in enumerate(
        iter_table_chunks(path, chunk_size, limit_rows),
        start=1,
    ):
        raw_rows += len(raw_chunk)
        cleaned_chunk, _clean_stats = cleaner.clean(raw_chunk)
        aggregator.add(cleaned_chunk)
        cleaned_rows += len(cleaned_chunk)
        logger.info(
            "聚合分块 %d: %s -> %s 行",
            chunk_count,
            f"{len(raw_chunk):,}",
            f"{len(cleaned_chunk):,}",
        )
        del raw_chunk
        del cleaned_chunk
        gc.collect()

    if chunk_count == 0:
        raise ValueError(f"No rows were read from raw behavior data: {path}")

    aggregator.finalize()
    return aggregator, {
        "raw_rows": raw_rows,
        "cleaned_rows": cleaned_rows,
        "chunk_count": chunk_count,
    }


def build_target_pairs(
    item_category_map: pd.DataFrame,
    observed_pairs: pd.DataFrame,
    candidate_path: Path | None,
    failure_policy: str = "error",
    reject_path: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """构造待预测的用户-商品候选对。

    Args:
        item_category_map: 商品到类目的唯一映射表。
        observed_pairs: 原始行为中出现过的用户-商品对。
        candidate_path: 可选候选对文件路径。
        failure_policy: 候选类目匹配失败时的处理策略，取 error 或 drop。
        reject_path: 候选失败明细输出路径。

    Returns:
        候选对 DataFrame 和候选处理统计信息。
    """
    if failure_policy not in {"error", "drop"}:
        raise ValueError("failure_policy must be 'error' or 'drop'.")

    if candidate_path is None:
        pairs = observed_pairs.drop_duplicates(["user_id", "item_id"]).reset_index(
            drop=True
        )
        logger.info("未提供候选对文件，使用原始行为中的全量用户-商品对")
        logger.info("候选对数量: %s", f"{len(pairs):,}")
        return pairs, {
            "input_candidates": len(pairs),
            "deduplicated_candidates": len(pairs),
            "rejected_candidates": 0,
        }

    candidates = read_table(candidate_path)
    missing_id_cols = sorted(set(ID_COLS) - set(candidates.columns))
    if missing_id_cols:
        raise ValueError(f"candidate file missing columns: {missing_id_cols}")

    candidate_columns = ID_COLS + [
        col for col in ["item_category"] if col in candidates
    ]
    raw_candidate_count = len(candidates)
    pairs = (
        candidates[candidate_columns].drop_duplicates(ID_COLS).reset_index(drop=True)
    )
    deduplicated_candidates = len(pairs)
    if "item_category" not in pairs.columns:
        pairs = pairs.merge(item_category_map, on="item_id", how="left")
    else:
        pairs = pairs.merge(
            item_category_map.rename(columns={"item_category": "_raw_item_category"}),
            on="item_id",
            how="left",
        )
        category_conflict = (
            pairs["_raw_item_category"].notna()
            & pairs["item_category"].notna()
            & (pairs["item_category"] != pairs["_raw_item_category"])
        )
        if category_conflict.any():
            conflict_count = int(category_conflict.sum())
            raise ValueError(
                f"{conflict_count:,} candidate rows have an item_category "
                "different from raw behavior data."
            )
        pairs["item_category"] = pairs["item_category"].fillna(
            pairs["_raw_item_category"]
        )
        pairs = pairs.drop(columns=["_raw_item_category"])

    missing_category = pairs["item_category"].isna().sum()
    rejected_pairs = pairs[pairs["item_category"].isna()].copy()
    if missing_category > 0:
        if reject_path is not None:
            reject_path.parent.mkdir(parents=True, exist_ok=True)
            rejected_pairs.to_csv(reject_path, index=False)
            logger.warning("候选失败明细已保存: %s", reject_path)
        message = f"{missing_category:,}"
        if failure_policy == "error":
            raise ValueError(
                f"{message} candidate rows cannot match item_category. "
                "Use --candidate-failure-policy drop only when dropping is intended."
            )
        logger.warning("%s 个候选对无法匹配 item_category，将按配置丢弃", message)
        pairs = pairs[pairs["item_category"].notna()].copy()

    pairs["item_category"] = pairs["item_category"].astype(
        item_category_map["item_category"].dtype
    )
    logger.info("候选对数量: %s", f"{len(pairs):,}")
    return pairs, {
        "input_candidates": raw_candidate_count,
        "deduplicated_candidates": deduplicated_candidates,
        "rejected_candidates": int(missing_category),
    }


def fit_inference_preprocessor(
    reference_wide_path: Path,
    feature_cols: list[str],
    chunk_size: int,
) -> InferencePreprocessor:
    """从训练宽表拟合推理阶段复用的预处理参数。

    Args:
        reference_wide_path: 训练阶段输出的 feature_wide_table.parquet。
        feature_cols: 模型训练时使用的特征列。
        chunk_size: 参考宽表分块读取行数。

    Returns:
        InferencePreprocessor 实例。

    Raises:
        FileNotFoundError: 参考宽表不存在。
        ValueError: 参考宽表缺少必要列。
    """
    if not reference_wide_path.exists():
        raise FileNotFoundError(
            f"Reference wide table does not exist: {reference_wide_path}"
        )

    needed_cols = sorted(
        {
            "item_category",
            TARGET_COL,
            *[c for c in feature_cols if c != "item_category_te"],
        }
    )
    fill_values: dict[str, float] = {
        "item_decay_slope": 0.0,
        "user_category_pref_score": 0.0,
    }

    logger.info("分块读取训练宽表，拟合推理预处理参数: %s", reference_wide_path)

    median_values: list[np.ndarray] = []
    if "user_avg_interval_hours" in needed_cols:
        for reference_chunk in iter_table_chunks(
            reference_wide_path,
            chunk_size,
            columns=["user_avg_interval_hours"],
        ):
            values = pd.to_numeric(
                reference_chunk["user_avg_interval_hours"],
                errors="coerce",
            ).dropna()
            median_values.append(values.to_numpy(dtype=np.float64))
        if median_values:
            fill_values["user_avg_interval_hours"] = float(
                np.median(np.concatenate(median_values))
            )

    category_sum = pd.Series(dtype="float64")
    category_count = pd.Series(dtype="float64")
    total_positive = 0.0
    total_rows = 0
    for reference_chunk in iter_table_chunks(
        reference_wide_path,
        chunk_size,
        columns=["item_category", TARGET_COL],
    ):
        target = (reference_chunk[TARGET_COL] > 0).astype(np.int8)
        grouped_sum = target.groupby(reference_chunk["item_category"]).sum()
        grouped_count = target.groupby(reference_chunk["item_category"]).count()
        category_sum = category_sum.add(grouped_sum, fill_value=0.0)
        category_count = category_count.add(grouped_count, fill_value=0.0)
        total_positive += float(target.sum())
        total_rows += len(reference_chunk)

    if total_rows == 0:
        raise ValueError(f"Reference wide table is empty: {reference_wide_path}")

    item_category_te_series = category_sum / category_count
    item_category_te_default = total_positive / total_rows

    scaler = StandardScaler()
    all_reference_columns = [col for col in needed_cols if col != TARGET_COL]
    for reference_chunk in iter_table_chunks(
        reference_wide_path,
        chunk_size,
        columns=all_reference_columns,
    ):
        reference_chunk["item_category_te"] = (
            reference_chunk["item_category"]
            .map(item_category_te_series)
            .fillna(item_category_te_default)
            .astype(np.float32)
        )
        for col in feature_cols:
            if col not in reference_chunk.columns:
                raise ValueError(f"reference wide table cannot build feature: {col}")
            values = pd.to_numeric(reference_chunk[col], errors="coerce")
            if col in fill_values:
                values = values.fillna(fill_values[col])
            reference_chunk[col] = values.fillna(0.0).astype(np.float64)
        scaler.partial_fit(reference_chunk[feature_cols].to_numpy())

    means = {col: float(value) for col, value in zip(feature_cols, scaler.mean_)}
    scales = {
        col: float(value) if value > 0 else 1.0
        for col, value in zip(feature_cols, scaler.scale_)
    }

    return InferencePreprocessor(
        feature_cols=feature_cols,
        fill_values=fill_values,
        item_category_te={
            int(key): float(value) for key, value in item_category_te_series.items()
        },
        item_category_te_default=item_category_te_default,
        means=means,
        scales=scales,
    )


def load_or_build_preprocessor(
    preprocessor_path: Path,
    reference_wide_path: Path,
    feature_cols: list[str],
    rebuild: bool,
    chunk_size: int,
) -> InferencePreprocessor:
    """加载或构建推理预处理参数。

    Args:
        preprocessor_path: 预处理参数缓存路径。
        reference_wide_path: 训练宽表路径。
        feature_cols: 模型训练特征列。
        rebuild: 是否强制重建缓存。
        chunk_size: 参考宽表分块读取行数。

    Returns:
        InferencePreprocessor 实例。
    """
    if preprocessor_path.exists() and not rebuild:
        logger.info("加载推理预处理参数缓存: %s", preprocessor_path)
        try:
            payload = joblib.load(preprocessor_path)
            if isinstance(payload, InferencePreprocessor):
                preprocessor = payload
            elif isinstance(payload, dict):
                preprocessor = InferencePreprocessor(**payload)
            else:
                raise TypeError("Unsupported preprocessor cache format.")
            if preprocessor.feature_cols != feature_cols:
                raise ValueError(
                    "Cached preprocessor feature_cols differs from model artifact."
                )
            return preprocessor
        except (AttributeError, ImportError, ModuleNotFoundError) as exc:
            logger.warning(
                "旧版预处理缓存无法加载，将自动重建: %s",
                exc,
            )

    preprocessor = fit_inference_preprocessor(
        reference_wide_path,
        feature_cols,
        chunk_size,
    )
    preprocessor_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(asdict(preprocessor), preprocessor_path)
    logger.info("推理预处理参数已保存: %s", preprocessor_path)
    return preprocessor


def transform_features(
    wide_df: pd.DataFrame,
    preprocessor: InferencePreprocessor,
) -> pd.DataFrame:
    """按训练阶段预处理参数转换推理特征。

    Args:
        wide_df: 推理宽表。
        preprocessor: 推理预处理参数。

    Returns:
        与模型 feature_cols 顺序一致的特征 DataFrame。

    Raises:
        ValueError: 无法生成模型必需特征时抛出。
    """
    df = wide_df.copy()
    for col, fill_value in preprocessor.fill_values.items():
        if col in df.columns:
            df[col] = df[col].fillna(fill_value)

    df["item_category_te"] = (
        df["item_category"]
        .map(preprocessor.item_category_te)
        .fillna(preprocessor.item_category_te_default)
        .astype(np.float32)
    )

    missing_cols = sorted(set(preprocessor.feature_cols) - set(df.columns))
    if missing_cols:
        raise ValueError(f"inference features missing required columns: {missing_cols}")

    feature_df = df[preprocessor.feature_cols].copy()
    for col in preprocessor.feature_cols:
        values = pd.to_numeric(feature_df[col], errors="coerce").fillna(0.0)
        feature_df[col] = (
            (values - preprocessor.means[col]) / preprocessor.scales[col]
        ).astype(np.float32)

    return feature_df


def predict_probabilities(
    model: Any,
    wide_df: pd.DataFrame,
    preprocessor: InferencePreprocessor,
    batch_size: int,
) -> np.ndarray:
    """分批输出正类购买概率。

    Args:
        model: 支持 predict_proba 的模型对象。
        wide_df: 推理特征宽表。
        preprocessor: 推理预处理参数。
        batch_size: 每批预测行数。

    Returns:
        正类概率数组。

    Raises:
        TypeError: 模型不支持 predict_proba 时抛出。
    """
    if not hasattr(model, "predict_proba"):
        raise TypeError(f"{type(model).__name__} does not support predict_proba().")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    probabilities: list[np.ndarray] = []
    for start in range(0, len(wide_df), batch_size):
        end = min(start + batch_size, len(wide_df))
        feature_batch = transform_features(
            wide_df.iloc[start:end],
            preprocessor,
        )
        batch_prob = model.predict_proba(feature_batch)[:, 1]
        probabilities.append(batch_prob.astype(np.float32))
        logger.info("预测批次完成: %s - %s", f"{start:,}", f"{end:,}")
        del feature_batch
        gc.collect()
    return (
        np.concatenate(probabilities)
        if probabilities
        else np.array([], dtype=np.float32)
    )


def build_prediction_output(
    wide_df: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    """组装最终预测结果表。

    Args:
        wide_df: 推理宽表。
        probabilities: 正类购买概率。
        threshold: 分类阈值。

    Returns:
        预测结果 DataFrame。
    """
    output_df = wide_df[["user_id", "item_id", "item_category"]].copy()
    output_df[PROB_COL] = probabilities
    output_df[LABEL_COL] = (output_df[PROB_COL] >= threshold).astype(np.int8)
    output_df["threshold"] = threshold
    output_df["rank_in_user"] = (
        output_df.groupby("user_id")[PROB_COL]
        .rank(method="first", ascending=False)
        .astype("int32")
    )
    return output_df.sort_values(["user_id", "rank_in_user"]).reset_index(drop=True)


def write_metadata(
    output_dir: Path,
    metadata: dict[str, Any],
) -> None:
    """保存推理运行元信息。

    Args:
        output_dir: 输出目录。
        metadata: 元信息字典。
    """
    metadata_path = output_dir / "inference_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    logger.info("推理元信息已保存: %s", metadata_path)


def main() -> None:
    """执行 XGBoost 本地批量推理主流程。"""
    args = parse_args()
    start_time = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("加载模型: %s", args.model_path)
    model_artifact = joblib.load(args.model_path)
    model = model_artifact["model"]
    feature_cols = list(model_artifact["feature_cols"])
    threshold = (
        float(args.threshold)
        if args.threshold is not None
        else float(model_artifact.get("selected_threshold", 0.5))
    )

    preprocessor = load_or_build_preprocessor(
        preprocessor_path=args.preprocessor_path,
        reference_wide_path=args.reference_wide_path,
        feature_cols=feature_cols,
        rebuild=args.rebuild_preprocessor,
        chunk_size=args.chunk_size,
    )

    logger.info("读取原始行为数据: %s", args.raw_path)
    behavior_aggregator, input_stats = aggregate_behavior_chunks(
        args.raw_path,
        chunk_size=args.chunk_size,
        limit_rows=args.limit_rows,
    )
    reject_path = args.output_dir / "candidate_rejections.csv"
    target_pairs, candidate_stats = build_target_pairs(
        behavior_aggregator.item_category_map,
        behavior_aggregator.observed_pairs,
        args.candidate_path,
        failure_policy=args.candidate_failure_policy,
        reject_path=reject_path if args.candidate_path else None,
    )
    wide_df = behavior_aggregator.build_wide_table(target_pairs)
    del behavior_aggregator
    gc.collect()

    logger.info(
        "开始模型预测，特征矩阵: %s 行 × %s 列",
        f"{len(wide_df):,}",
        len(feature_cols),
    )
    probabilities = predict_probabilities(
        model,
        wide_df,
        preprocessor,
        args.batch_size,
    )
    prediction_df = build_prediction_output(wide_df, probabilities, threshold)

    output_path = args.output_dir / args.output_name
    prediction_df.to_csv(output_path, index=False)
    elapsed_seconds = time.time() - start_time

    metadata = {
        "raw_path": str(args.raw_path),
        "candidate_path": str(args.candidate_path) if args.candidate_path else None,
        "model_path": str(args.model_path),
        "reference_wide_path": str(args.reference_wide_path),
        "preprocessor_path": str(args.preprocessor_path),
        "output_path": str(output_path),
        "memory_mode": "chunked_ingestion_and_incremental_feature_aggregation",
        "chunk_size": args.chunk_size,
        "batch_size": args.batch_size,
        "candidate_failure_policy": args.candidate_failure_policy,
        "raw_rows": input_stats["raw_rows"],
        "cleaned_rows": input_stats["cleaned_rows"],
        "input_chunk_count": input_stats["chunk_count"],
        "candidate_stats": candidate_stats,
        "prediction_rows": len(prediction_df),
        "feature_cols": feature_cols,
        "threshold": threshold,
        "positive_predictions": int(prediction_df[LABEL_COL].sum()),
        "elapsed_seconds": elapsed_seconds,
        "random_state": RANDOM_STATE,
    }
    write_metadata(args.output_dir, metadata)

    logger.info("预测结果已保存: %s", output_path)
    logger.info("推理完成，用时 %.2f 秒", elapsed_seconds)
    print(f"\nBatch inference finished: {output_path}")


if __name__ == "__main__":
    main()

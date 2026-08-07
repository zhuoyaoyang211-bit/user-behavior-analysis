"""Run Step 9 interpretability and error-sample analysis.

This script covers two deliverables after model metric evaluation:
    1. Model interpretability analysis:
       - XGBoost feature importance.
       - XGBoost native SHAP contribution values via pred_contribs=True.
    2. Error-sample analysis:
       - False positive / false negative feature profiles.
       - Segment metrics for cold-start users and long-tail items.
       - Markdown report for presentation and documentation.

The script does not train or tune models.

Example:
    python src/run_step9_interpretability_error_analysis.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TEST_PATH = PROJECT_DIR / "output" / "test.parquet"
DEFAULT_MODEL_PATH = (
    PROJECT_DIR / "output" / "optuna_tuned_models" / "xgboost_optuna.joblib"
)
DEFAULT_PREDICTION_PATH = (
    PROJECT_DIR
    / "output"
    / "step9_test_model_evaluation"
    / "step9_test_predictions.csv"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_DIR / "output" / "step9_test_interpretability_error_analysis"
)
DEFAULT_TARGET_COL = "label"
DEFAULT_THRESHOLD = 0.5
ID_COLS = ["user_id", "item_id"]
KEY_FEATURES = [
    "user_pv_count",
    "item_pv_count",
    "item_buy_count",
    "item_buy_user_count",
    "cat_pv_count",
    "cat_view_user_count",
    "buy_conversion_rate",
    "cart_to_buy_rate",
    "fav_to_buy_rate",
    "user_category_pref_score",
    "user_avg_interval_hours",
    "item_decay_slope",
    "rfm_r_score",
    "rfm_f_score",
    "rfm_m_score",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Step9 interpretability and error analysis for final model."
    )
    parser.add_argument(
        "--test-path",
        "--val-path",
        dest="test_path",
        type=Path,
        default=DEFAULT_TEST_PATH,
        help=(
            "Held-out test parquet file. --val-path is retained only for "
            "backward compatibility."
        ),
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--prediction-path",
        type=Path,
        default=DEFAULT_PREDICTION_PATH,
        help="Step9 prediction CSV. If missing, predictions are generated again.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--target-col",
        default=None,
        help="Target column. Defaults to model artifact target_col or label.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Classification threshold. Defaults to model selected_threshold.",
    )
    parser.add_argument(
        "--shap-sample-size",
        type=int,
        default=10_000,
        help="Maximum test rows used for native SHAP contribution analysis.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500_000,
        help="Prediction batch size when prediction file is unavailable.",
    )
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def load_artifact(path: Path) -> dict[str, Any]:
    """Load a joblib model artifact and normalize model-only objects."""
    if not path.exists():
        raise FileNotFoundError(f"Model artifact does not exist: {path}")
    artifact = joblib.load(path)
    if isinstance(artifact, dict):
        return artifact
    return {"model": artifact}


def get_threshold(
    artifact: dict[str, Any], override: float | None
) -> tuple[float, str]:
    """Resolve classification threshold."""
    if override is not None:
        if override < 0 or override > 1:
            raise ValueError(f"threshold must be within [0, 1], got: {override}")
        return float(override), "command_line"
    if "selected_threshold" in artifact:
        return float(artifact["selected_threshold"]), str(
            artifact.get("selected_threshold_source", "artifact_selected_threshold")
        )
    if "threshold" in artifact:
        return float(artifact["threshold"]), "artifact_threshold"
    return DEFAULT_THRESHOLD, "default_0_5"


def load_validation(path: Path, target_col: str) -> pd.DataFrame:
    """Load validation/test data."""
    if not path.exists():
        raise FileNotFoundError(f"Validation file does not exist: {path}")
    val_df = pd.read_parquet(path)
    if target_col not in val_df.columns:
        raise ValueError(f"Target column not found in validation data: {target_col}")
    return val_df


def build_feature_matrix(val_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Build numeric feature matrix in model training order."""
    missing = sorted(set(feature_cols) - set(val_df.columns))
    if missing:
        raise ValueError(f"Validation data is missing feature columns: {missing}")
    X = val_df[feature_cols].copy()
    if X.isna().any().any():
        missing_summary = X.isna().sum()
        missing_summary = missing_summary[missing_summary > 0].to_dict()
        raise ValueError(
            f"Validation features contain missing values: {missing_summary}"
        )
    return X.astype(np.float32)


def predict_probabilities(model: Any, X: pd.DataFrame, batch_size: int) -> np.ndarray:
    """Predict positive probabilities in batches."""
    if not hasattr(model, "predict_proba"):
        raise TypeError(f"{type(model).__name__} does not support predict_proba().")
    probabilities: list[np.ndarray] = []
    for start in range(0, len(X), batch_size):
        stop = min(start + batch_size, len(X))
        prob = model.predict_proba(X.iloc[start:stop])[:, 1]
        probabilities.append(np.asarray(prob, dtype=np.float32))
        print(f"Predicted rows {start:,} - {stop:,}", flush=True)
    return np.concatenate(probabilities)


def load_or_predict(
    prediction_path: Path,
    val_df: pd.DataFrame,
    model: Any,
    feature_cols: list[str],
    target_col: str,
    threshold: float,
    batch_size: int,
) -> pd.DataFrame:
    """Load Step9 predictions or regenerate them if unavailable."""
    if prediction_path.exists():
        prediction_df = pd.read_csv(prediction_path)
        required = {"y_true", "purchase_probability"}
        missing = sorted(required - set(prediction_df.columns))
        if missing:
            raise ValueError(f"Prediction file missing columns: {missing}")
        if len(prediction_df) == len(val_df) and all(
            col in prediction_df.columns for col in ID_COLS
        ):
            return prediction_df
        if len(prediction_df) != len(val_df):
            raise ValueError(
                f"Prediction rows do not match validation rows: "
                f"{len(prediction_df)} vs {len(val_df)}"
            )
        return prediction_df.reset_index(drop=True)

    X = build_feature_matrix(val_df, feature_cols)
    y_prob = predict_probabilities(model, X, batch_size)
    output_cols = [col for col in ID_COLS if col in val_df.columns]
    prediction_df = val_df[output_cols].copy()
    prediction_df["y_true"] = val_df[target_col].to_numpy(dtype=np.int8)
    prediction_df["purchase_probability"] = y_prob
    prediction_df["prediction_label"] = (y_prob >= threshold).astype(np.int8)
    prediction_df["threshold"] = threshold
    return prediction_df


def align_predictions_with_validation(
    val_df: pd.DataFrame,
    prediction_df: pd.DataFrame,
    target_col: str,
) -> pd.DataFrame:
    """Merge predictions back to validation features."""
    if all(col in prediction_df.columns for col in ID_COLS) and all(
        col in val_df.columns for col in ID_COLS
    ):
        pred_cols = ID_COLS + ["y_true", "purchase_probability", "prediction_label"]
        pred = prediction_df[pred_cols].copy()
        pred = pred.drop_duplicates(ID_COLS)
        merged = val_df.merge(pred, on=ID_COLS, how="left", validate="one_to_one")
        if merged["purchase_probability"].isna().any():
            missing_count = int(merged["purchase_probability"].isna().sum())
            raise ValueError(f"{missing_count:,} validation rows lack predictions.")
    else:
        merged = val_df.reset_index(drop=True).copy()
        merged["y_true"] = prediction_df["y_true"].to_numpy(dtype=np.int8)
        merged["purchase_probability"] = prediction_df["purchase_probability"].to_numpy(
            dtype=np.float32
        )
        merged["prediction_label"] = prediction_df["prediction_label"].to_numpy(
            dtype=np.int8
        )

    merged["y_true"] = merged["y_true"].astype(np.int8)
    if target_col in merged.columns and not np.array_equal(
        merged[target_col].to_numpy(dtype=np.int8),
        merged["y_true"].to_numpy(dtype=np.int8),
    ):
        raise ValueError("Prediction labels do not match validation target labels.")
    return merged


def xgboost_feature_importance(
    model: Any,
    feature_cols: list[str],
    output_dir: Path,
) -> pd.DataFrame:
    """Export XGBoost built-in feature importance."""
    if not hasattr(model, "feature_importances_"):
        return pd.DataFrame()
    importance_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "feature_importance": np.asarray(model.feature_importances_, dtype=float),
        }
    ).sort_values("feature_importance", ascending=False)
    importance_df.to_csv(output_dir / "xgboost_feature_importance.csv", index=False)
    return importance_df


def sample_for_shap(
    val_df: pd.DataFrame,
    target_col: str,
    sample_size: int,
    random_state: int,
) -> pd.DataFrame:
    """Sample validation rows for SHAP analysis with enough positive examples."""
    if sample_size <= 0:
        raise ValueError("shap-sample-size must be positive.")
    if len(val_df) <= sample_size:
        return val_df.copy()

    positives = val_df[val_df[target_col] == 1]
    negatives = val_df[val_df[target_col] == 0]
    pos_keep = min(len(positives), max(1, sample_size // 5))
    neg_keep = sample_size - pos_keep
    pos_sample = positives.sample(
        n=pos_keep,
        random_state=random_state,
        replace=False,
    )
    neg_sample = negatives.sample(
        n=neg_keep,
        random_state=random_state,
        replace=False,
    )
    return (
        pd.concat([pos_sample, neg_sample], ignore_index=True)
        .sample(frac=1.0, random_state=random_state)
        .reset_index(drop=True)
    )


def xgboost_native_shap_importance(
    model: Any,
    shap_sample_df: pd.DataFrame,
    feature_cols: list[str],
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute XGBoost native SHAP contribution values.

    XGBoost pred_contribs=True returns per-feature SHAP contribution values
    plus the final bias term. This avoids requiring the external shap package.
    """
    X_sample = build_feature_matrix(shap_sample_df, feature_cols)
    booster = model.get_booster() if hasattr(model, "get_booster") else model
    dmatrix = xgb.DMatrix(X_sample.to_numpy(dtype=np.float32))
    shap_values = booster.predict(dmatrix, pred_contribs=True)
    feature_shap = shap_values[:, : len(feature_cols)]

    shap_df = pd.DataFrame(feature_shap, columns=feature_cols)
    shap_df.insert(0, "label", shap_sample_df[DEFAULT_TARGET_COL].to_numpy(np.int8))
    shap_df.to_csv(output_dir / "xgboost_native_shap_values_sample.csv", index=False)

    shap_importance_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "mean_abs_shap": np.abs(feature_shap).mean(axis=0),
            "mean_shap": feature_shap.mean(axis=0),
            "positive_label_mean_abs_shap": np.nan,
            "negative_label_mean_abs_shap": np.nan,
            "positive_label_mean_shap": np.nan,
            "negative_label_mean_shap": np.nan,
        }
    )
    labels = shap_sample_df[DEFAULT_TARGET_COL].to_numpy(dtype=np.int8)
    if (labels == 1).any():
        shap_importance_df["positive_label_mean_abs_shap"] = np.abs(
            feature_shap[labels == 1]
        ).mean(axis=0)
        shap_importance_df["positive_label_mean_shap"] = feature_shap[labels == 1].mean(
            axis=0
        )
    if (labels == 0).any():
        shap_importance_df["negative_label_mean_abs_shap"] = np.abs(
            feature_shap[labels == 0]
        ).mean(axis=0)
        shap_importance_df["negative_label_mean_shap"] = feature_shap[labels == 0].mean(
            axis=0
        )
    shap_importance_df["positive_minus_negative_mean_shap"] = (
        shap_importance_df["positive_label_mean_shap"]
        - shap_importance_df["negative_label_mean_shap"]
    )
    shap_importance_df["contrast_direction"] = np.select(
        [
            shap_importance_df["positive_minus_negative_mean_shap"] > 0,
            shap_importance_df["positive_minus_negative_mean_shap"] < 0,
        ],
        [
            "more_positive_for_positive_labels",
            "more_negative_for_positive_labels",
        ],
        default="no_difference",
    )
    shap_importance_df = shap_importance_df.sort_values(
        "mean_abs_shap",
        ascending=False,
    )
    shap_importance_df.to_csv(
        output_dir / "xgboost_native_shap_importance.csv",
        index=False,
    )
    return shap_importance_df, shap_df


def metric_row(
    name: str,
    frame: pd.DataFrame,
    threshold: float,
) -> dict[str, Any]:
    """Compute one segment metric row."""
    y_true = frame["y_true"].to_numpy(dtype=np.int8)
    y_prob = frame["purchase_probability"].to_numpy(dtype=np.float32)
    y_pred = (y_prob >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    row = {
        "segment": name,
        "rows": int(len(frame)),
        "positive_count": int(y_true.sum()),
        "positive_rate": float(y_true.mean()) if len(y_true) else np.nan,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    if len(np.unique(y_true)) == 2:
        row["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        row["pr_auc_ap"] = float(average_precision_score(y_true, y_prob))
    else:
        row["roc_auc"] = np.nan
        row["pr_auc_ap"] = np.nan
    return row


def select_bottom_entity_segment(
    analysis_df: pd.DataFrame,
    entity_col: str,
    feature_col: str,
    fraction: float = 0.20,
) -> tuple[pd.Series, dict[str, Any]]:
    """Select the lowest-ranked entities with deterministic tie breaking.

    Entity-level selection avoids expanding a row-level quantile into a much
    larger segment when many users or items share the same standardized value.
    The resulting row count can differ from 20% because each entity contributes
    a different number of user-item rows.
    """
    required = {entity_col, feature_col}
    missing = sorted(required - set(analysis_df.columns))
    if missing:
        raise ValueError(f"Segment columns are missing: {missing}")
    if not 0 < fraction < 1:
        raise ValueError(f"fraction must be within (0, 1), got: {fraction}")

    entity_values = (
        analysis_df[[entity_col, feature_col]]
        .drop_duplicates(entity_col)
        .copy()
    )
    entity_values[feature_col] = pd.to_numeric(
        entity_values[feature_col], errors="coerce"
    )
    entity_values = entity_values.dropna(subset=[feature_col])
    entity_values = entity_values.sort_values(
        [feature_col, entity_col],
        kind="mergesort",
    ).reset_index(drop=True)
    entity_count = len(entity_values)
    if entity_count == 0:
        return (
            pd.Series(False, index=analysis_df.index),
            {
                "entity_col": entity_col,
                "feature_col": feature_col,
                "entity_count": 0,
                "selected_entity_count": 0,
                "feature_threshold": np.nan,
            },
        )

    selected_count = max(1, int(np.ceil(entity_count * fraction)))
    selected_entities = set(entity_values.iloc[:selected_count][entity_col])
    mask = analysis_df[entity_col].isin(selected_entities)
    return (
        mask,
        {
            "entity_col": entity_col,
            "feature_col": feature_col,
            "entity_count": int(entity_count),
            "selected_entity_count": int(selected_count),
            "feature_threshold": float(
                entity_values.iloc[selected_count - 1][feature_col]
            ),
        },
    )


def segment_metrics(
    analysis_df: pd.DataFrame,
    threshold: float,
    output_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Evaluate cold-start and long-tail segments."""
    user_mask, user_meta = select_bottom_entity_segment(
        analysis_df,
        entity_col="user_id",
        feature_col="user_pv_count",
    )
    item_mask, item_meta = select_bottom_entity_segment(
        analysis_df,
        entity_col="item_id",
        feature_col="item_pv_count",
    )

    segment_masks = {
        "overall": pd.Series(True, index=analysis_df.index),
        "cold_start_user_bottom20_entities": user_mask,
        "long_tail_item_bottom20_entities": item_mask,
    }

    rows = []
    for name, mask in segment_masks.items():
        segment_df = analysis_df[mask].copy()
        if len(segment_df) == 0:
            continue
        rows.append(metric_row(name, segment_df, threshold))

    segment_df = pd.DataFrame(rows)
    segment_df.to_csv(output_dir / "error_segment_metrics.csv", index=False)
    thresholds = {
        "cold_start_user": user_meta,
        "long_tail_item": item_meta,
    }
    return segment_df, thresholds


def error_sample_analysis(
    analysis_df: pd.DataFrame,
    threshold: float,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Analyze false positive and false negative samples."""
    df = analysis_df.copy()
    df["error_type"] = "correct"
    df.loc[
        (df["y_true"] == 0) & (df["purchase_probability"] >= threshold), "error_type"
    ] = "false_positive"
    df.loc[
        (df["y_true"] == 1) & (df["purchase_probability"] < threshold), "error_type"
    ] = "false_negative"
    df.loc[
        (df["y_true"] == 1) & (df["purchase_probability"] >= threshold), "error_type"
    ] = "true_positive"
    df.loc[
        (df["y_true"] == 0) & (df["purchase_probability"] < threshold), "error_type"
    ] = "true_negative"

    feature_cols = [col for col in KEY_FEATURES if col in df.columns]
    profile_df = (
        df.groupby("error_type")[feature_cols + ["purchase_probability"]]
        .agg(["count", "mean", "median"])
        .reset_index()
    )
    profile_df.columns = [
        (
            "_".join([str(part) for part in col if part != ""]).strip("_")
            if isinstance(col, tuple)
            else col
        )
        for col in profile_df.columns
    ]
    profile_df.to_csv(output_dir / "error_feature_profile.csv", index=False)

    error_cols = [
        col
        for col in [*ID_COLS, "y_true", "purchase_probability", "prediction_label"]
        if col in df.columns
    ] + feature_cols
    false_positive = (
        df[df["error_type"] == "false_positive"]
        .sort_values("purchase_probability", ascending=False)
        .head(200)
    )
    false_negative = (
        df[df["error_type"] == "false_negative"]
        .sort_values("purchase_probability", ascending=True)
        .head(200)
    )
    error_samples = pd.concat([false_positive, false_negative], ignore_index=True)
    error_samples[["error_type", *error_cols]].to_csv(
        output_dir / "top_error_samples.csv",
        index=False,
    )
    df[["error_type", *error_cols]].to_csv(
        output_dir / "test_error_labels.csv",
        index=False,
    )
    return profile_df, error_samples


def plot_top_bar(
    df: pd.DataFrame,
    value_col: str,
    title: str,
    output_path: Path,
    top_n: int,
    label_col: str = "feature",
) -> None:
    """Plot a horizontal top-N bar chart."""
    if df.empty or value_col not in df.columns or label_col not in df.columns:
        return
    plot_df = df.head(top_n).sort_values(value_col, ascending=True)
    plt.figure(figsize=(10, max(5, 0.35 * len(plot_df))))
    plt.barh(plot_df[label_col], plot_df[value_col])
    plt.title(title)
    plt.xlabel(value_col)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def to_markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    """Convert DataFrame to markdown."""
    table_df = df.copy()
    if max_rows is not None:
        table_df = table_df.head(max_rows)
    for col in table_df.select_dtypes(include=["float"]).columns:
        table_df[col] = table_df[col].map(lambda value: f"{value:.6f}")
    markdown = table_df.to_csv(index=False, sep="|").replace("|", " | ")
    lines = markdown.strip().splitlines()
    header = f"| {lines[0]} |"
    separator = "| " + " | ".join(["---"] * len(table_df.columns)) + " |"
    body = [f"| {line} |" for line in lines[1:]]
    return "\n".join([header, separator, *body])


def write_report(
    output_dir: Path,
    setup: dict[str, Any],
    tree_importance_df: pd.DataFrame,
    shap_importance_df: pd.DataFrame,
    segment_df: pd.DataFrame,
    profile_df: pd.DataFrame,
) -> None:
    """Write the interpretability and error-analysis report."""
    top_shap = shap_importance_df.head(10)[
        [
            "feature",
            "mean_abs_shap",
            "positive_label_mean_shap",
            "negative_label_mean_shap",
            "positive_minus_negative_mean_shap",
            "contrast_direction",
        ]
    ]
    top_tree = tree_importance_df.head(10)
    report_lines = [
        "# Step9 模型可解释性与错误样本分析报告",
        "",
        "## 1. 分析配置",
        "",
        f"- 测试集：`{setup['test_path']}`",
        f"- 最终模型：优化后的 XGBoost，`{setup['model_path']}`",
        f"- 预测结果：`{setup['prediction_path']}`",
        f"- 输出目录：`{setup['output_dir']}`",
        f"- 样本数：{setup['n_rows']:,}",
        f"- 正样本占比：{setup['positive_rate']:.6%}",
        f"- 固定分类阈值：{setup['threshold']}",
        f"- 阈值来源：`{setup['threshold_source']}`",
        f"- XGBoost SHAP 分析样本数：{setup['shap_sample_rows']:,}",
        (
            f"- SHAP 样本正样本数：{setup['shap_positive_count']:,}，"
            f"负样本数：{setup['shap_negative_count']:,}，"
            f"正样本占比：{setup['shap_positive_rate']:.6%}"
        ),
        "",
    ]

    report_lines.extend(
        [
            "## 2. XGBoost 特征重要性",
            "",
            "树模型内置特征重要性反映特征在树分裂中的贡献频率或增益综合表现，可用于观察模型主要依赖哪些业务特征。",
            "",
            to_markdown_table(top_tree),
            "",
            "## 3. XGBoost 原生 SHAP 解释",
            "",
            (
                "本部分使用 XGBoost 原生 `pred_contribs=True` 计算 SHAP contribution values。"
                "`mean_abs_shap` 越高，说明该特征对模型输出的影响越大。由于购买正样本极少，"
                "全体抽样的 `mean_shap` 会被负样本主导，因此不直接把它解释为特征的业务方向。"
                "本报告改用正负标签组的平均 SHAP 贡献及其差值，描述该特征在两类样本中的输出贡献差异。"
                "SHAP 抽样保留测试集全部正样本并抽取负样本，目的是保证稀有正样本可解释，"
                "因此该表用于机制分析，不作为测试集总体分布的重新估计。"
            ),
            "",
            to_markdown_table(top_shap),
            "",
            "## 4. 冷启动与长尾场景分段表现",
            "",
            (
                "冷启动用户和长尾商品分别按照测试集中用户实体、商品实体的行为特征排序，"
                "选取底部 20% 的实体进行诊断。由于每个实体包含的用户-商品样本数不同，"
                "对应的行占比不一定是 20%。部分特征已经标准化，因此阈值只用于相对分组，"
                "不直接解释为原始浏览次数。这些分段不是重新训练样本，只是测试集上的诊断切片。"
            ),
            "",
            to_markdown_table(segment_df),
            "",
            "## 5. 错误样本特征画像",
            "",
            "下表按 false positive、false negative、true positive、true negative 聚合关键特征均值和中位数，用于定位误判样本的共性。特征数值为标准化后的相对值，不代表原始业务计数。",
            "",
            to_markdown_table(profile_df, max_rows=20),
            "",
            "## 6. 结论口径",
            "",
            (
                "最终模型的主要解释对象是优化后的 XGBoost。若高影响特征集中在商品热度、用户行为强度、类目偏好和转化率特征，"
                "说明模型主要通过历史行为强度与商品受欢迎程度判断购买概率。"
            ),
            "",
            (
                "错误样本分析重点关注 false negative 和 false positive：false negative 代表真实购买但模型没有召回的样本，"
                "通常会暴露冷启动、低历史行为、长尾商品或类目偏好不足的问题；false positive 代表模型高估购买概率的样本，"
                "通常和高浏览、高加购或热门商品但最终未购买有关。"
            ),
            "",
            "## 7. 产出物",
            "",
            "| 文件 | 说明 |",
            "| --- | --- |",
            "| `xgboost_feature_importance.csv` | XGBoost 内置特征重要性 |",
            "| `xgboost_native_shap_importance.csv` | XGBoost 原生 SHAP 全局重要性 |",
            "| `xgboost_native_shap_values_sample.csv` | SHAP 样本明细 |",
            "| `error_segment_metrics.csv` | 冷启动、长尾等分段指标 |",
            "| `error_feature_profile.csv` | 错误类型特征画像 |",
            "| `top_error_samples.csv` | 高置信误判样本 Top 明细 |",
            "| `test_error_labels.csv` | 测试集逐样本错误类型 |",
            "| `step9_interpretability_error_report.md` | 本报告 |",
            "",
        ]
    )
    (output_dir / "step9_interpretability_error_report.md").write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )


def main() -> None:
    """Run interpretability and error analysis."""
    args = parse_args()
    start_time = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    artifact = load_artifact(args.model_path)
    model = artifact["model"]
    feature_cols = list(artifact["feature_cols"])
    target_col = args.target_col or artifact.get("target_col", DEFAULT_TARGET_COL)
    threshold, threshold_source = get_threshold(artifact, args.threshold)

    val_df = load_validation(args.test_path, target_col)
    prediction_df = load_or_predict(
        args.prediction_path,
        val_df,
        model,
        feature_cols,
        target_col,
        threshold,
        args.batch_size,
    )
    analysis_df = align_predictions_with_validation(val_df, prediction_df, target_col)
    analysis_df["prediction_label"] = (
        analysis_df["purchase_probability"] >= threshold
    ).astype(np.int8)

    tree_importance_df = xgboost_feature_importance(
        model,
        feature_cols,
        args.output_dir,
    )

    shap_sample = sample_for_shap(
        val_df,
        target_col,
        args.shap_sample_size,
        args.random_state,
    )
    shap_sample = shap_sample.rename(columns={target_col: DEFAULT_TARGET_COL})
    shap_importance_df, _shap_values_df = xgboost_native_shap_importance(
        model,
        shap_sample,
        feature_cols,
        args.output_dir,
    )

    segment_df, segment_thresholds = segment_metrics(
        analysis_df,
        threshold,
        args.output_dir,
    )
    profile_df, _error_samples = error_sample_analysis(
        analysis_df,
        threshold,
        args.output_dir,
    )

    plot_top_bar(
        shap_importance_df,
        "mean_abs_shap",
        "Top XGBoost Native SHAP Importance",
        args.output_dir / "xgboost_native_shap_importance_top.png",
        args.top_n,
    )
    plot_top_bar(
        tree_importance_df,
        "feature_importance",
        "Top XGBoost Feature Importance",
        args.output_dir / "xgboost_feature_importance_top.png",
        args.top_n,
    )
    y_true = analysis_df["y_true"].to_numpy(dtype=np.int8)
    shap_labels = shap_sample[DEFAULT_TARGET_COL].to_numpy(dtype=np.int8)
    setup = {
        "test_path": str(args.test_path),
        "model_path": str(args.model_path),
        "prediction_path": str(args.prediction_path),
        "output_dir": str(args.output_dir),
        "target_col": target_col,
        "feature_cols": feature_cols,
        "threshold": threshold,
        "threshold_source": threshold_source,
        "shap_sample_size": args.shap_sample_size,
        "shap_sample_rows": int(len(shap_sample)),
        "shap_positive_count": int((shap_labels == 1).sum()),
        "shap_negative_count": int((shap_labels == 0).sum()),
        "shap_positive_rate": float(shap_labels.mean()),
        "segment_thresholds": segment_thresholds,
        "n_rows": int(len(analysis_df)),
        "positive_count": int(y_true.sum()),
        "negative_count": int(len(y_true) - y_true.sum()),
        "positive_rate": float(y_true.mean()),
        "elapsed_seconds": time.time() - start_time,
    }
    (args.output_dir / "step9_interpretability_error_setup.json").write_text(
        json.dumps(setup, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    write_report(
        args.output_dir,
        setup,
        tree_importance_df,
        shap_importance_df,
        segment_df,
        profile_df,
    )

    print("\nStep9 interpretability and error analysis finished.", flush=True)
    print(f"Outputs saved to: {args.output_dir}", flush=True)
    print("\nTop native SHAP features:", flush=True)
    print(
        shap_importance_df[
            [
                "feature",
                "mean_abs_shap",
                "positive_minus_negative_mean_shap",
                "contrast_direction",
            ]
        ]
        .head(args.top_n)
        .to_string(index=False),
        flush=True,
    )
    print("\nSegment metrics:", flush=True)
    print(segment_df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

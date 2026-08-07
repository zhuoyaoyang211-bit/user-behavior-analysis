"""Evaluate the final model on the held-out test set.

This script is for Step 9: model effect comprehensive evaluation.
It does not train or tune models. It loads an existing model artifact,
generates probabilities on output/test.parquet, and reports ROC-AUC,
accuracy, precision, recall, F1, PR-AUC, log loss, and confusion matrix.

Example:
    python src/run_step9_model_evaluation.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TEST_PATH = PROJECT_DIR / "output" / "test.parquet"
DEFAULT_MODEL_PATH = (
    PROJECT_DIR / "output" / "optuna_tuned_models" / "xgboost_optuna.joblib"
)
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "output" / "step9_test_model_evaluation"
DEFAULT_TARGET_COL = "label"
DEFAULT_THRESHOLD = 0.5
DEFAULT_THRESHOLDS = [
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
]
ID_COLS = ["user_id", "item_id"]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate final model generalization on held-out test.parquet."
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
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Model artifact path. The artifact should contain model and feature_cols.",
    )
    parser.add_argument(
        "--prediction-path",
        type=Path,
        default=None,
        help=(
            "Optional existing prediction CSV/Parquet. When provided, the script "
            "evaluates this file instead of re-predicting from model-path."
        ),
    )
    parser.add_argument(
        "--prediction-prob-col",
        default="purchase_probability",
        help="Probability column name when --prediction-path is provided.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for metrics, predictions, and report outputs.",
    )
    parser.add_argument(
        "--target-col",
        default=None,
        help="Target column in test-path. Defaults to model artifact target_col or label.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Classification threshold. Defaults to model artifact selected_threshold, "
            "then artifact threshold, then 0.5."
        ),
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=DEFAULT_THRESHOLDS,
        help="Thresholds for diagnostic precision/recall/F1 analysis.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500_000,
        help="Prediction batch size when model-path is used.",
    )
    parser.add_argument(
        "--save-predictions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to save row-level test predictions.",
    )
    return parser.parse_args()


def load_model_artifact(model_path: Path) -> tuple[Any, list[str], str, float, str]:
    """Load model artifact and return model, features, target, threshold, source."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact does not exist: {model_path}")

    artifact = joblib.load(model_path)
    if isinstance(artifact, dict):
        model = artifact.get("model")
        feature_cols = artifact.get("feature_cols")
        target_col = artifact.get("target_col", DEFAULT_TARGET_COL)
        if "selected_threshold" in artifact:
            threshold = float(artifact["selected_threshold"])
            threshold_source = artifact.get(
                "selected_threshold_source",
                "model_artifact_selected_threshold",
            )
        elif "threshold" in artifact:
            threshold = float(artifact["threshold"])
            threshold_source = "model_artifact_threshold"
        else:
            threshold = DEFAULT_THRESHOLD
            threshold_source = "default_0_5"
    else:
        model = artifact
        feature_cols = None
        target_col = DEFAULT_TARGET_COL
        threshold = DEFAULT_THRESHOLD
        threshold_source = "default_0_5"

    if model is None:
        raise ValueError("Model artifact does not contain a valid model object.")
    if not feature_cols:
        raise ValueError("Model artifact must contain non-empty feature_cols.")

    return model, list(feature_cols), str(target_col), threshold, str(threshold_source)


def validate_thresholds(thresholds: list[float]) -> list[float]:
    """Validate and de-duplicate thresholds while preserving numeric order."""
    unique_thresholds = sorted({float(value) for value in thresholds})
    invalid = [value for value in unique_thresholds if value < 0 or value > 1]
    if invalid:
        raise ValueError(f"Thresholds must be within [0, 1], got: {invalid}")
    return unique_thresholds


def load_validation_data(val_path: Path, target_col: str) -> pd.DataFrame:
    """Read validation/test parquet and validate target column."""
    if not val_path.exists():
        raise FileNotFoundError(f"Validation file does not exist: {val_path}")
    val_df = pd.read_parquet(val_path)
    if target_col not in val_df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found in {val_path}. "
            f"Available columns: {list(val_df.columns)}"
        )
    return val_df


def build_feature_matrix(val_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Build model feature DataFrame in training-time column order."""
    missing_cols = sorted(set(feature_cols) - set(val_df.columns))
    if missing_cols:
        raise ValueError(f"Validation data is missing feature columns: {missing_cols}")

    X = val_df[feature_cols].copy()
    non_numeric_cols = [
        col for col in feature_cols if not pd.api.types.is_numeric_dtype(X[col])
    ]
    if non_numeric_cols:
        raise ValueError(f"Feature columns must be numeric: {non_numeric_cols}")
    if X.isna().any().any():
        missing_summary = X.isna().sum()
        missing_summary = missing_summary[missing_summary > 0].to_dict()
        raise ValueError(
            f"Validation feature matrix contains missing values: {missing_summary}"
        )
    return X.astype(np.float32)


def predict_positive_probability(
    model: Any,
    X: pd.DataFrame,
    batch_size: int,
) -> np.ndarray:
    """Predict positive-class probabilities in batches."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if not hasattr(model, "predict_proba"):
        raise TypeError(f"{type(model).__name__} does not support predict_proba().")

    probabilities: list[np.ndarray] = []
    for start in range(0, len(X), batch_size):
        stop = min(start + batch_size, len(X))
        batch_prob = model.predict_proba(X.iloc[start:stop])[:, 1]
        probabilities.append(np.asarray(batch_prob, dtype=np.float32))
        print(f"Predicted rows {start:,} - {stop:,}", flush=True)
    return (
        np.concatenate(probabilities)
        if probabilities
        else np.array([], dtype=np.float32)
    )


def load_existing_predictions(
    prediction_path: Path,
    val_df: pd.DataFrame,
    prob_col: str,
) -> np.ndarray:
    """Load existing prediction probabilities and align with val_df order."""
    if not prediction_path.exists():
        raise FileNotFoundError(f"Prediction file does not exist: {prediction_path}")

    suffix = prediction_path.suffix.lower()
    if suffix == ".csv":
        pred_df = pd.read_csv(prediction_path)
    elif suffix in {".parquet", ".pq"}:
        pred_df = pd.read_parquet(prediction_path)
    else:
        raise ValueError(f"Unsupported prediction file suffix: {suffix}")

    if prob_col not in pred_df.columns:
        raise ValueError(f"Prediction probability column not found: {prob_col}")

    if len(pred_df) == len(val_df) and not all(
        col in pred_df.columns for col in ID_COLS
    ):
        return pred_df[prob_col].to_numpy(dtype=np.float32)

    if all(col in pred_df.columns for col in ID_COLS) and all(
        col in val_df.columns for col in ID_COLS
    ):
        pred_df = pred_df[ID_COLS + [prob_col]].copy()
        if pred_df.duplicated(ID_COLS).any():
            raise ValueError(
                "Prediction file contains duplicated user_id,item_id rows."
            )
        aligned = val_df[ID_COLS].merge(
            pred_df, on=ID_COLS, how="left", validate="one_to_one"
        )
        if aligned[prob_col].isna().any():
            missing_count = int(aligned[prob_col].isna().sum())
            raise ValueError(
                f"{missing_count:,} validation rows cannot match predictions."
            )
        return aligned[prob_col].to_numpy(dtype=np.float32)

    if len(pred_df) != len(val_df):
        raise ValueError(
            "Prediction file must either have the same row count as val data "
            "or contain user_id,item_id columns for alignment."
        )
    return pred_df[prob_col].to_numpy(dtype=np.float32)


def metrics_at_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """Compute classification metrics at one threshold."""
    y_pred = (y_prob >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def compute_summary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """Compute ranking metrics plus fixed-threshold classification metrics."""
    threshold_metrics = metrics_at_threshold(y_true, y_prob, threshold)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc_ap": float(average_precision_score(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, np.clip(y_prob, 1e-15, 1 - 1e-15))),
        "threshold": threshold_metrics["threshold"],
        "accuracy": threshold_metrics["accuracy"],
        "precision": threshold_metrics["precision"],
        "recall": threshold_metrics["recall"],
        "f1": threshold_metrics["f1"],
        "tn": threshold_metrics["tn"],
        "fp": threshold_metrics["fp"],
        "fn": threshold_metrics["fn"],
        "tp": threshold_metrics["tp"],
    }


def build_prediction_output(
    val_df: pd.DataFrame,
    target_col: str,
    y_prob: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    """Build row-level prediction output."""
    id_cols = [col for col in ID_COLS if col in val_df.columns]
    output_cols = id_cols + [target_col]
    prediction_df = val_df[output_cols].copy()
    prediction_df = prediction_df.rename(columns={target_col: "y_true"})
    prediction_df["purchase_probability"] = y_prob
    prediction_df["prediction_label"] = (
        prediction_df["purchase_probability"] >= threshold
    ).astype(np.int8)
    prediction_df["threshold"] = threshold
    if "user_id" in prediction_df.columns:
        prediction_df["rank_in_user"] = (
            prediction_df.groupby("user_id")["purchase_probability"]
            .rank(method="first", ascending=False)
            .astype("int32")
        )
        prediction_df = prediction_df.sort_values(["user_id", "rank_in_user"])
    return prediction_df.reset_index(drop=True)


def to_markdown_table(df: pd.DataFrame) -> str:
    """Convert a DataFrame to a compact markdown table."""
    table_df = df.copy()
    for col in table_df.select_dtypes(include=["float"]).columns:
        table_df[col] = table_df[col].map(lambda value: f"{value:.6f}")
    markdown_table = table_df.to_csv(index=False, sep="|").replace("|", " | ")
    table_lines = markdown_table.strip().splitlines()
    header = f"| {table_lines[0]} |"
    separator = "| " + " | ".join(["---"] * len(table_df.columns)) + " |"
    body = [f"| {line} |" for line in table_lines[1:]]
    return "\n".join([header, separator, *body])


def write_report(
    output_dir: Path,
    setup: dict[str, Any],
    summary_df: pd.DataFrame,
    threshold_df: pd.DataFrame,
) -> None:
    """Write a Chinese markdown report for Step 9."""
    best_f1_row = threshold_df.sort_values(
        ["f1", "precision"],
        ascending=[False, False],
    ).iloc[0]
    report_lines = [
        "# Step9 模型效果综合评估报告",
        "",
        "## 1. 评估目标",
        "",
        (
            "本步骤使用独立测试集评估最终模型的泛化能力。评估过程不重新训练模型，"
            "不使用测试集选择模型参数；测试集只用于最终性能报告。"
        ),
        "",
        "## 2. 评估配置",
        "",
        f"- 测试集路径：`{setup['test_path']}`",
        f"- 模型路径：`{setup['model_path']}`",
        f"- 预测文件路径：`{setup['prediction_path']}`",
        f"- 输出目录：`{setup['output_dir']}`",
        f"- 测试集样本数：{setup['n_rows']:,}",
        f"- 正样本数：{setup['positive_count']:,}",
        f"- 负样本数：{setup['negative_count']:,}",
        f"- 正样本占比：{setup['positive_rate']:.6%}",
        f"- 特征数：{setup['feature_count']}",
        f"- 标签列：`{setup['target_col']}`",
        f"- 固定评估阈值：{setup['threshold']}",
        f"- 阈值来源：`{setup['threshold_source']}`",
        "",
        "## 3. 核心评估指标",
        "",
        to_markdown_table(summary_df),
        "",
        "说明：ROC-AUC 衡量模型整体排序能力，Accuracy、Precision、Recall 和 F1 衡量固定阈值下的分类效果。由于购买预测正负样本极不平衡，PR-AUC 也作为补充指标，用于观察模型对正样本的识别能力。",
        "",
        "## 4. 阈值诊断",
        "",
        to_markdown_table(threshold_df),
        "",
        (
            f"在测试集上的 F1 诊断最优阈值为 {best_f1_row['threshold']:.2f}，"
            f"对应 F1 为 {best_f1_row['f1']:.6f}。该结果只用于误差分析和阈值敏感性说明，"
            "不能反向用于重新选择模型参数，否则会引入测试集信息泄露。"
        ),
        "",
        "## 5. 产出物",
        "",
        "| 文件 | 说明 |",
        "| --- | --- |",
        "| `step9_generalization_metrics.csv` | 固定阈值下的综合评估指标 |",
        "| `step9_threshold_metrics.csv` | 多阈值 Precision/Recall/F1 诊断结果 |",
        "| `step9_test_predictions.csv` | 测试集逐样本预测概率与预测标签 |",
        "| `step9_evaluation_setup.json` | 本次评估的数据、模型、阈值和耗时配置 |",
        "| `step9_model_evaluation_report.md` | 本 Markdown 报告 |",
        "",
    ]
    (output_dir / "step9_model_evaluation_report.md").write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )


def main() -> None:
    """Run Step 9 model evaluation."""
    args = parse_args()
    start_time = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    thresholds = validate_thresholds(args.thresholds)

    model, feature_cols, artifact_target_col, artifact_threshold, threshold_source = (
        load_model_artifact(args.model_path)
    )
    target_col = args.target_col or artifact_target_col or DEFAULT_TARGET_COL
    threshold = (
        float(args.threshold) if args.threshold is not None else artifact_threshold
    )
    if threshold < 0 or threshold > 1:
        raise ValueError(f"threshold must be within [0, 1], got: {threshold}")
    if args.threshold is not None:
        threshold_source = "command_line"

    val_df = load_validation_data(args.test_path, target_col)
    y_true = val_df[target_col].to_numpy(dtype=np.int8, copy=True)
    if len(np.unique(y_true)) != 2:
        raise ValueError(
            "Evaluation target must contain both positive and negative labels."
        )

    if args.prediction_path is not None:
        print(f"Loading existing predictions: {args.prediction_path}", flush=True)
        y_prob = load_existing_predictions(
            args.prediction_path,
            val_df,
            args.prediction_prob_col,
        )
        prediction_source = "existing_prediction_file"
    else:
        print(f"Loading test features: {args.test_path}", flush=True)
        X_val = build_feature_matrix(val_df, feature_cols)
        print(f"Predicting with model: {args.model_path}", flush=True)
        y_prob = predict_positive_probability(model, X_val, args.batch_size)
        prediction_source = "model_predict_proba"

    if len(y_prob) != len(y_true):
        raise ValueError(
            f"Prediction length mismatch: predictions={len(y_prob)}, labels={len(y_true)}"
        )

    summary = compute_summary_metrics(y_true, y_prob, threshold)
    summary.update(
        {
            "model_path": str(args.model_path),
            "test_path": str(args.test_path),
            "prediction_source": prediction_source,
            "target_col": target_col,
            "threshold_source": threshold_source,
            "n_rows": int(len(y_true)),
            "positive_count": int(y_true.sum()),
            "negative_count": int(len(y_true) - y_true.sum()),
            "positive_rate": float(y_true.mean()),
            "feature_count": int(len(feature_cols)),
        }
    )
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(args.output_dir / "step9_generalization_metrics.csv", index=False)

    threshold_df = pd.DataFrame(
        [metrics_at_threshold(y_true, y_prob, value) for value in thresholds]
    )
    threshold_df.insert(0, "model_path", str(args.model_path))
    threshold_df.to_csv(args.output_dir / "step9_threshold_metrics.csv", index=False)

    if args.save_predictions:
        prediction_df = build_prediction_output(val_df, target_col, y_prob, threshold)
        prediction_df.to_csv(
            args.output_dir / "step9_test_predictions.csv",
            index=False,
        )

    elapsed_seconds = time.time() - start_time
    setup = {
        "test_path": str(args.test_path),
        "model_path": str(args.model_path),
        "prediction_path": str(args.prediction_path) if args.prediction_path else None,
        "output_dir": str(args.output_dir),
        "target_col": target_col,
        "feature_cols": feature_cols,
        "feature_count": len(feature_cols),
        "threshold": threshold,
        "threshold_source": threshold_source,
        "thresholds": thresholds,
        "batch_size": args.batch_size,
        "save_predictions": args.save_predictions,
        "prediction_source": prediction_source,
        "n_rows": int(len(y_true)),
        "positive_count": int(y_true.sum()),
        "negative_count": int(len(y_true) - y_true.sum()),
        "positive_rate": float(y_true.mean()),
        "elapsed_seconds": elapsed_seconds,
    }
    (args.output_dir / "step9_evaluation_setup.json").write_text(
        json.dumps(setup, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    write_report(args.output_dir, setup, summary_df, threshold_df)

    print("\nStep9 model evaluation finished.", flush=True)
    print(summary_df.to_string(index=False), flush=True)
    print(f"\nOutputs saved to: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()

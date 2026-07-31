"""Train controlled baseline classifiers and export comparison artifacts.

Model comparison is performed on an internal split sampled from train.parquet.
The untouched val.parquet is used only for final reporting.
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
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
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
from xgboost import XGBClassifier

from traditional_ml_split import (
    numeric_feature_columns,
    sample_by_label_time_strata,
    smote_to_ratio,
    split_within_time_strata,
    to_xy,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_PATH = PROJECT_DIR / "output" / "train.parquet"
DEFAULT_FINAL_TRAIN_PATH = PROJECT_DIR / "output" / "train_smote_r10.parquet"
DEFAULT_VAL_PATH = PROJECT_DIR / "output" / "val.parquet"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "output" / "baseline_models"
TARGET_COL = "label"
RANDOM_STATE = 42
THRESHOLD = 0.5
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument(
        "--final-train-path",
        type=Path,
        default=DEFAULT_FINAL_TRAIN_PATH,
        help="Full SMOTE r10 training set used after baseline model selection.",
    )
    parser.add_argument("--val-path", type=Path, default=DEFAULT_VAL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--selection-sample-size", type=int, default=500_000)
    parser.add_argument("--selection-val-frac", type=float, default=0.2)
    parser.add_argument("--smote-ratio", type=float, default=0.10)
    return parser.parse_args()


def build_models() -> dict[str, Any]:
    return {
        "logistic_regression": LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            max_iter=1000,
            class_weight=None,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            objective="binary:logistic",
            eval_metric=["logloss", "auc", "aucpr"],
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        ),
        "lightgbm": LGBMClassifier(
            objective="binary",
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=0.0,
            class_weight=None,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        ),
    }


def evaluate(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, Any]:
    y_pred = (y_prob >= THRESHOLD).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc_ap": average_precision_score(y_true, y_prob),
        "log_loss": log_loss(y_true, np.clip(y_prob, 1e-15, 1 - 1e-15)),
        "accuracy_at_0_5": accuracy_score(y_true, y_pred),
        "precision_at_0_5": precision_score(y_true, y_pred, zero_division=0),
        "recall_at_0_5": recall_score(y_true, y_pred, zero_division=0),
        "f1_at_0_5": f1_score(y_true, y_pred, zero_division=0),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def predict_positive_probability(model: Any, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    raise TypeError(f"{type(model).__name__} does not support predict_proba().")


def best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: list[float] = DEFAULT_THRESHOLDS,
) -> dict[str, float]:
    """Select a threshold on internal selection validation only."""
    rows = []
    for threshold in thresholds:
        pred = (y_prob >= threshold).astype(np.int8)
        rows.append(
            {
                "threshold": threshold,
                "precision": precision_score(y_true, pred, zero_division=0),
                "recall": recall_score(y_true, pred, zero_division=0),
                "f1": f1_score(y_true, pred, zero_division=0),
            }
        )
    return max(rows, key=lambda row: (row["f1"], row["precision"]))


def feature_effects(
    model_name: str, model: Any, feature_cols: list[str]
) -> pd.DataFrame:
    if model_name == "logistic_regression":
        values = np.abs(model.coef_[0])
        raw_values = model.coef_[0]
        return pd.DataFrame(
            {
                "feature": feature_cols,
                "importance": values,
                "coefficient": raw_values,
            }
        ).sort_values("importance", ascending=False)

    if hasattr(model, "feature_importances_"):
        return pd.DataFrame(
            {
                "feature": feature_cols,
                "importance": model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

    return pd.DataFrame({"feature": feature_cols, "importance": np.nan})


def write_report(
    output_dir: Path,
    setup: dict[str, Any],
    selection_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
) -> None:
    def to_markdown_table(df: pd.DataFrame) -> str:
        table_df = df.copy()
        for col in table_df.select_dtypes(include=["float"]).columns:
            table_df[col] = table_df[col].map(lambda x: f"{x:.6f}")
        markdown_table = table_df.to_csv(index=False, sep="|").replace("|", " | ")
        table_lines = markdown_table.strip().splitlines()
        header = f"| {table_lines[0]} |"
        separator = "| " + " | ".join(["---"] * len(table_df.columns)) + " |"
        body = [f"| {line} |" for line in table_lines[1:]]
        return "\n".join([header, separator, *body])

    lines = [
        "# Baseline Model Comparison",
        "",
        "## Controlled Setup",
        "",
        f"- Raw train for model selection: `{setup['train_path']}`",
        f"- Final train for refit: `{setup['final_train_path']}`",
        f"- Final validation: `{setup['val_path']}`",
        f"- Features: {len(setup['feature_cols'])} common numeric columns",
        f"- Raw train shape: {tuple(setup['raw_train_shape'])}",
        f"- Selection sample shape: {tuple(setup['selection_sample_shape'])}",
        f"- Selection train shape before SMOTE: {tuple(setup['selection_train_shape'])}, "
        f"positive rate: {setup['selection_train_pos_rate_before_smote']:.6f}",
        f"- Selection train positive rate after SMOTE: {setup['selection_train_pos_rate_after_smote']:.6f}",
        f"- Selection validation shape: {tuple(setup['selection_val_shape'])}, "
        f"positive rate: {setup['selection_val_pos_rate']:.6f}",
        f"- Final train shape: {tuple(setup['final_train_shape'])}, positive rate: {setup['final_train_pos_rate']:.6f}",
        f"- Final validation shape: {tuple(setup['final_val_shape'])}, positive rate: {setup['final_val_pos_rate']:.6f}",
        f"- Random state: {RANDOM_STATE}",
        f"- Default classification threshold: {THRESHOLD}",
        "- Model/threshold selection uses only the internal selection validation split.",
        "- Final validation is used once for reporting and is not used to choose models or thresholds.",
        "",
        "## Internal Selection Metrics",
        "",
        to_markdown_table(selection_df),
        "",
        "## Final Validation Metrics",
        "",
        to_markdown_table(metrics_df),
        "",
        "## Feature Columns",
        "",
        ", ".join(setup["feature_cols"]),
        "",
    ]
    (output_dir / "baseline_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw_train = pd.read_parquet(args.train_path)
    final_train = pd.read_parquet(args.final_train_path)
    final_val = pd.read_parquet(args.val_path)
    feature_cols = numeric_feature_columns(raw_train, final_train, final_val)

    selection_sample = sample_by_label_time_strata(
        raw_train,
        sample_size=min(args.selection_sample_size, len(raw_train)),
        random_state=RANDOM_STATE,
    )
    selection_train, selection_val = split_within_time_strata(
        selection_sample,
        val_frac=args.selection_val_frac,
    )
    X_selection_train, y_selection_train, selection_smote_info = smote_to_ratio(
        selection_train,
        feature_cols,
        target_pos_ratio=args.smote_ratio,
        random_state=RANDOM_STATE,
    )
    X_selection_val, y_selection_val = to_xy(selection_val, feature_cols)
    X_final_train, y_final_train = to_xy(final_train, feature_cols)
    X_val, y_val = to_xy(final_val, feature_cols)

    setup = {
        "train_path": str(args.train_path),
        "final_train_path": str(args.final_train_path),
        "val_path": str(args.val_path),
        "output_dir": str(args.output_dir),
        "feature_cols": feature_cols,
        "raw_train_shape": raw_train.shape,
        "selection_sample_shape": selection_sample.shape,
        "selection_train_shape": selection_train.shape,
        "selection_val_shape": selection_val.shape,
        "selection_train_pos_rate_before_smote": float(selection_train[TARGET_COL].mean()),
        "selection_train_pos_rate_after_smote": float(y_selection_train.mean()),
        "selection_val_pos_rate": float(y_selection_val.mean()),
        "selection_smote_info": selection_smote_info,
        "final_train_shape": final_train.shape,
        "final_val_shape": final_val.shape,
        "final_train_pos_rate": float(y_final_train.mean()),
        "final_val_pos_rate": float(y_val.mean()),
        "random_state": RANDOM_STATE,
        "threshold": THRESHOLD,
        "selection_policy": (
            "sample 500k from train.parquet by label × time stratum; "
            "split each time stratum into selection train/validation; "
            "apply SMOTE only to selection train"
        ),
        "final_policy": (
            "refit selected baseline configurations on full train_smote_r10.parquet; "
            "evaluate once on full val.parquet"
        ),
    }
    (args.output_dir / "baseline_setup.json").write_text(
        json.dumps(setup, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    selection_rows = []
    metrics_rows = []
    params = {}
    for model_name in build_models().keys():
        print(f"\nSelecting baseline {model_name} on internal split ...", flush=True)
        selection_model = build_models()[model_name]
        selection_model.fit(X_selection_train, y_selection_train)
        selection_prob = predict_positive_probability(selection_model, X_selection_val)
        selection_threshold = best_threshold(y_selection_val, selection_prob)
        selection_rows.append(
            {
                "model": model_name,
                "selection_rows": len(y_selection_val),
                "selection_positive_count": int(y_selection_val.sum()),
                **evaluate(y_selection_val, selection_prob),
                "selected_threshold": selection_threshold["threshold"],
                "selected_threshold_f1": selection_threshold["f1"],
            }
        )

        print(f"Training final {model_name} on full SMOTE r10 train ...", flush=True)
        model = build_models()[model_name]
        start = time.time()
        model.fit(X_final_train, y_final_train)
        train_seconds = time.time() - start

        val_prob = predict_positive_probability(model, X_val)
        row = {
            "model": model_name,
            "train_seconds": train_seconds,
            **evaluate(y_val, val_prob),
            "selected_threshold": selection_threshold["threshold"],
            "selected_threshold_source": "internal_selection_validation",
        }
        metrics_rows.append(row)

        joblib.dump(
            {
                "model": model,
                "feature_cols": feature_cols,
                "target_col": TARGET_COL,
                "threshold": THRESHOLD,
                "selected_threshold": selection_threshold["threshold"],
                "selected_threshold_source": "internal_selection_validation",
                "selection_metrics": selection_rows[-1],
                "params": model.get_params(),
            },
            args.output_dir / f"{model_name}_baseline.joblib",
        )
        feature_effects(model_name, model, feature_cols).to_csv(
            args.output_dir / f"{model_name}_feature_effects.csv",
            index=False,
        )
        params[model_name] = model.get_params()
        print(f"Finished {model_name}: {train_seconds:.2f}s", flush=True)

    selection_df = pd.DataFrame(selection_rows).sort_values("pr_auc_ap", ascending=False)
    selection_df.to_csv(
        args.output_dir / "baseline_selection_metrics.csv",
        index=False,
    )
    metrics_df = pd.DataFrame(metrics_rows).sort_values("pr_auc_ap", ascending=False)
    metrics_df.to_csv(args.output_dir / "baseline_metrics.csv", index=False)
    (args.output_dir / "baseline_params.json").write_text(
        json.dumps(params, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    write_report(args.output_dir, setup, selection_df, metrics_df)
    print("\nValidation metrics:")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()

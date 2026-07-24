"""Train controlled baseline classifiers and export comparison artifacts."""

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


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_PATH = PROJECT_DIR / "output" / "train_smote_r10.parquet"
DEFAULT_VAL_PATH = PROJECT_DIR / "output" / "val.parquet"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "output" / "baseline_models"
TARGET_COL = "label"
EXCLUDE_COLS = {
    TARGET_COL,
    "user_id",
    "item_id",
    "last_time",
    "buy_path_type",
    "behavior_type",
    "item_category",
}
RANDOM_STATE = 42
THRESHOLD = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--val-path", type=Path, default=DEFAULT_VAL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_data(train_path: Path, val_path: Path) -> dict[str, Any]:
    train = pd.read_parquet(train_path)
    val = pd.read_parquet(val_path)

    if TARGET_COL not in train.columns or TARGET_COL not in val.columns:
        raise ValueError(f"Both datasets must contain `{TARGET_COL}`.")

    common_cols = [c for c in train.columns if c in val.columns]
    feature_cols = []
    dropped_non_numeric = []
    for col in common_cols:
        if col in EXCLUDE_COLS:
            continue
        if pd.api.types.is_numeric_dtype(train[col]) and pd.api.types.is_numeric_dtype(
            val[col]
        ):
            feature_cols.append(col)
        else:
            dropped_non_numeric.append(col)

    if not feature_cols:
        raise ValueError("No common numeric feature columns found.")

    X_train = train[feature_cols].to_numpy(dtype=np.float32, copy=True)
    y_train = train[TARGET_COL].to_numpy(dtype=np.int8, copy=True)
    X_val = val[feature_cols].to_numpy(dtype=np.float32, copy=True)
    y_val = val[TARGET_COL].to_numpy(dtype=np.int8, copy=True)

    if np.isnan(X_train).any() or np.isnan(X_val).any():
        raise ValueError("Missing values found. Add a shared imputation step first.")

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "feature_cols": feature_cols,
        "train_shape": train.shape,
        "val_shape": val.shape,
        "train_pos_rate": float(y_train.mean()),
        "val_pos_rate": float(y_val.mean()),
        "dropped_train_only": sorted(set(train.columns) - set(val.columns)),
        "dropped_val_only": sorted(set(val.columns) - set(train.columns)),
        "dropped_non_numeric": dropped_non_numeric,
    }


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
    metrics_df: pd.DataFrame,
) -> None:
    table_df = metrics_df.copy()
    for col in table_df.select_dtypes(include=["float"]).columns:
        table_df[col] = table_df[col].map(lambda x: f"{x:.6f}")
    markdown_table = table_df.to_csv(index=False, sep="|").replace("|", " | ")
    table_lines = markdown_table.strip().splitlines()
    header = f"| {table_lines[0]} |"
    separator = "| " + " | ".join(["---"] * len(table_df.columns)) + " |"
    body = [f"| {line} |" for line in table_lines[1:]]

    lines = [
        "# Baseline Model Comparison",
        "",
        "## Controlled Setup",
        "",
        f"- Train: `{setup['train_path']}`",
        f"- Validation: `{setup['val_path']}`",
        f"- Features: {len(setup['feature_cols'])} common numeric columns",
        f"- Train shape: {tuple(setup['train_shape'])}, positive rate: {setup['train_pos_rate']:.6f}",
        f"- Validation shape: {tuple(setup['val_shape'])}, positive rate: {setup['val_pos_rate']:.6f}",
        f"- Random state: {RANDOM_STATE}",
        f"- Classification threshold: {THRESHOLD}",
        "- No extra class weights; the same SMOTE r10 training set is used for all models.",
        "",
        "## Validation Metrics",
        "",
        "\n".join([header, separator, *body]),
        "",
        "## Feature Columns",
        "",
        ", ".join(setup["feature_cols"]),
        "",
        "## Dropped Columns",
        "",
        f"- Train-only columns: {setup['dropped_train_only']}",
        f"- Validation-only columns: {setup['dropped_val_only']}",
        f"- Non-numeric common columns: {setup['dropped_non_numeric']}",
        "",
    ]
    (output_dir / "baseline_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = load_data(args.train_path, args.val_path)
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    feature_cols = data["feature_cols"]

    setup = {
        "train_path": str(args.train_path),
        "val_path": str(args.val_path),
        "output_dir": str(args.output_dir),
        "feature_cols": feature_cols,
        "train_shape": data["train_shape"],
        "val_shape": data["val_shape"],
        "train_pos_rate": data["train_pos_rate"],
        "val_pos_rate": data["val_pos_rate"],
        "dropped_train_only": data["dropped_train_only"],
        "dropped_val_only": data["dropped_val_only"],
        "dropped_non_numeric": data["dropped_non_numeric"],
        "random_state": RANDOM_STATE,
        "threshold": THRESHOLD,
    }
    (args.output_dir / "baseline_setup.json").write_text(
        json.dumps(setup, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    metrics_rows = []
    params = {}
    for model_name, model in build_models().items():
        print(f"\nTraining {model_name} ...", flush=True)
        start = time.time()
        model.fit(X_train, y_train)
        train_seconds = time.time() - start

        val_prob = predict_positive_probability(model, X_val)
        row = {
            "model": model_name,
            "train_seconds": train_seconds,
            **evaluate(y_val, val_prob),
        }
        metrics_rows.append(row)

        joblib.dump(
            {
                "model": model,
                "feature_cols": feature_cols,
                "target_col": TARGET_COL,
                "threshold": THRESHOLD,
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

    metrics_df = pd.DataFrame(metrics_rows).sort_values("pr_auc_ap", ascending=False)
    metrics_df.to_csv(args.output_dir / "baseline_metrics.csv", index=False)
    (args.output_dir / "baseline_params.json").write_text(
        json.dumps(params, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    write_report(args.output_dir, setup, metrics_df)
    print("\nValidation metrics:")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()

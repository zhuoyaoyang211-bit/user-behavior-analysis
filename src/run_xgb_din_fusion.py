"""Compare XGBoost Optuna with fixed XGBoost + DIN fusion candidates.

This is a diagnostic Part7 experiment.  The weights are fixed before looking at
validation metrics, and val.parquet is used only for final reporting.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_DIR / "output"
DEFAULT_VAL_PATH = OUTPUT_DIR / "val.parquet"
DEFAULT_XGBOOST_PATH = OUTPUT_DIR / "optuna_tuned_models" / "xgboost_optuna.joblib"
DEFAULT_DIN_PRED_PATH = (
    OUTPUT_DIR
    / "part7_din_stacking"
    / "din_full_train_validation_predictions.csv"
)
DEFAULT_OUTPUT_DIR = OUTPUT_DIR / "xgb_din_fusion"
TARGET_COL = "label"
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
DEFAULT_DIN_WEIGHTS = [0.05, 0.10, 0.20, 0.30, 0.50]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-path", type=Path, default=DEFAULT_VAL_PATH)
    parser.add_argument("--xgboost-path", type=Path, default=DEFAULT_XGBOOST_PATH)
    parser.add_argument("--din-pred-path", type=Path, default=DEFAULT_DIN_PRED_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=100_000)
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Also save per-row fusion scores. This CSV is large and slower to write.",
    )
    parser.add_argument(
        "--din-weights",
        nargs="+",
        type=float,
        default=DEFAULT_DIN_WEIGHTS,
        help="Fixed DIN weights used in XGBoost + DIN blends.",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=DEFAULT_THRESHOLDS,
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for path in (args.val_path, args.xgboost_path, args.din_pred_path):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    invalid = [weight for weight in args.din_weights if not 0 < weight < 1]
    if invalid:
        raise ValueError(f"DIN weights must be in (0, 1): {invalid}")


def predict_in_batches(model: Any, X: np.ndarray, batch_size: int) -> np.ndarray:
    parts = []
    for start in range(0, len(X), batch_size):
        stop = min(start + batch_size, len(X))
        parts.append(model.predict_proba(X[start:stop])[:, 1])
    return np.concatenate(parts)


def percentile_rank(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average", pct=True).to_numpy(dtype=np.float64)


def threshold_metrics(
    y_true: np.ndarray,
    score: np.ndarray,
    thresholds: list[float],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    for threshold in thresholds:
        pred = (score >= threshold).astype(np.int8)
        tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
        rows.append(
            {
                "threshold": threshold,
                "precision": precision_score(y_true, pred, zero_division=0),
                "recall": recall_score(y_true, pred, zero_division=0),
                "f1": f1_score(y_true, pred, zero_division=0),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
        )
    best = max(rows, key=lambda row: (row["f1"], row["precision"]))
    return rows, best


def score_summary(
    name: str,
    score_type: str,
    y_true: np.ndarray,
    score: np.ndarray,
    thresholds: list[float],
    fixed_threshold: float = 0.5,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    score = np.clip(score, 1e-15, 1.0 - 1e-15)
    pred = (score >= fixed_threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    threshold_rows, best = threshold_metrics(y_true, score, thresholds)
    summary = {
        "model": name,
        "score_type": score_type,
        "roc_auc": roc_auc_score(y_true, score),
        "pr_auc_ap": average_precision_score(y_true, score),
        "log_loss": log_loss(y_true, score),
        "fixed_threshold": fixed_threshold,
        "precision_at_fixed_threshold": precision_score(
            y_true,
            pred,
            zero_division=0,
        ),
        "recall_at_fixed_threshold": recall_score(y_true, pred, zero_division=0),
        "f1_at_fixed_threshold": f1_score(y_true, pred, zero_division=0),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "best_threshold_on_validation_diagnostic": best["threshold"],
        "best_f1_on_validation_diagnostic": best["f1"],
    }
    tagged_threshold_rows = [
        {
            "model": name,
            "score_type": score_type,
            **row,
        }
        for row in threshold_rows
    ]
    return summary, tagged_threshold_rows


def write_report(
    output_dir: Path,
    setup: dict[str, Any],
    metrics_df: pd.DataFrame,
) -> None:
    view = metrics_df.copy()
    for col in view.select_dtypes(include=["float"]).columns:
        view[col] = view[col].map(lambda value: f"{value:.6f}")
    columns = [
        "model",
        "score_type",
        "din_weight",
        "roc_auc",
        "pr_auc_ap",
        "pr_auc_delta_vs_xgboost",
        "log_loss",
        "f1_at_fixed_threshold",
        "best_threshold_on_validation_diagnostic",
        "best_f1_on_validation_diagnostic",
        "beats_xgboost_pr_auc",
    ]
    table = view[columns]
    header = "| " + " | ".join(table.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(table.columns)) + " |"
    body = ["| " + " | ".join(map(str, row)) + " |" for row in table.to_numpy()]
    lines = [
        "# XGBoost + DIN Fusion Diagnostic",
        "",
        "This experiment uses XGBoost Optuna as the baseline and tests fixed "
        "XGBoost + DIN weights. The validation set is used only for reporting; "
        "weights are not optimized on validation.",
        "",
        "## Setup",
        "",
        f"- Validation data: `{setup['val_path']}`",
        f"- XGBoost artifact: `{setup['xgboost_path']}`",
        f"- DIN predictions: `{setup['din_pred_path']}`",
        f"- Validation rows: {setup['validation_rows']}",
        f"- Validation positives: {setup['validation_positive_count']}",
        f"- Fixed DIN weights: {setup['din_weights']}",
        "",
        "## Results",
        "",
        header,
        separator,
        *body,
        "",
        "Probability blends are deployable fixed-score blends. Rank blends are "
        "diagnostic only; they test whether DIN contributes useful ordering "
        "information independent of probability calibration.",
        "",
    ]
    (output_dir / "xgb_din_fusion_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    val = pd.read_parquet(args.val_path)
    y_true = val[TARGET_COL].to_numpy(dtype=np.int8, copy=True)
    xgb_payload = joblib.load(args.xgboost_path)
    feature_cols = xgb_payload["feature_cols"]
    missing_cols = sorted(set(feature_cols) - set(val.columns))
    if missing_cols:
        raise ValueError(f"Missing validation feature columns: {missing_cols}")
    X_val = val[feature_cols].to_numpy(dtype=np.float32, copy=True)
    if np.isnan(X_val).any():
        raise ValueError("Missing values found in XGBoost validation features.")

    din_pred = pd.read_csv(args.din_pred_path)
    if len(din_pred) != len(val):
        raise ValueError(
            f"DIN prediction row count {len(din_pred)} != validation rows {len(val)}."
        )
    if TARGET_COL in din_pred.columns:
        din_label = din_pred[TARGET_COL].to_numpy(dtype=np.int8, copy=True)
        if not np.array_equal(din_label, y_true):
            raise ValueError("DIN prediction labels do not match val.parquet labels.")
    if "din_probability" not in din_pred.columns:
        raise ValueError("DIN prediction file must contain `din_probability`.")

    xgb_prob = predict_in_batches(
        xgb_payload["model"],
        X_val,
        args.batch_size,
    )
    din_prob = din_pred["din_probability"].to_numpy(dtype=np.float64, copy=True)
    din_prob = np.clip(din_prob, 1e-15, 1.0 - 1e-15)

    xgb_rank = percentile_rank(xgb_prob)
    din_rank = percentile_rank(din_prob)

    scores: dict[str, tuple[str, float | None, np.ndarray]] = {
        "xgboost_optuna": ("probability", 0.0, xgb_prob),
        "din_full_train": ("probability", 1.0, din_prob),
    }
    for din_weight in args.din_weights:
        xgb_weight = 1.0 - din_weight
        scores[f"xgb_{xgb_weight:.2f}_din_{din_weight:.2f}_prob"] = (
            "probability_blend",
            din_weight,
            xgb_weight * xgb_prob + din_weight * din_prob,
        )
        scores[f"xgb_{xgb_weight:.2f}_din_{din_weight:.2f}_rank"] = (
            "rank_blend_diagnostic",
            din_weight,
            xgb_weight * xgb_rank + din_weight * din_rank,
        )

    summary_rows = []
    threshold_rows = []
    prediction_cols = None
    if args.save_predictions:
        prediction_cols = {
            TARGET_COL: y_true,
            "xgboost_probability": xgb_prob,
            "din_probability": din_prob,
        }
    for name, (score_type, din_weight, score) in scores.items():
        summary, rows = score_summary(
            name=name,
            score_type=score_type,
            y_true=y_true,
            score=score,
            thresholds=args.thresholds,
        )
        summary["din_weight"] = din_weight
        summary_rows.append(summary)
        threshold_rows.extend(rows)
        if prediction_cols is not None and name not in {"xgboost_optuna", "din_full_train"}:
            prediction_cols[name] = score

    metrics_df = pd.DataFrame(summary_rows)
    baseline_pr_auc = float(
        metrics_df.loc[metrics_df["model"] == "xgboost_optuna", "pr_auc_ap"].iloc[0]
    )
    metrics_df["pr_auc_delta_vs_xgboost"] = (
        metrics_df["pr_auc_ap"] - baseline_pr_auc
    )
    metrics_df["beats_xgboost_pr_auc"] = metrics_df["pr_auc_ap"] > baseline_pr_auc
    metrics_df = metrics_df.sort_values(
        ["pr_auc_ap", "roc_auc"],
        ascending=[False, False],
    )

    metrics_df.to_csv(args.output_dir / "xgb_din_fusion_metrics.csv", index=False)
    pd.DataFrame(threshold_rows).to_csv(
        args.output_dir / "xgb_din_fusion_threshold_metrics.csv",
        index=False,
    )
    if prediction_cols is not None:
        pd.DataFrame(prediction_cols).to_csv(
            args.output_dir / "xgb_din_fusion_predictions.csv",
            index=False,
        )

    setup = {
        "val_path": str(args.val_path),
        "xgboost_path": str(args.xgboost_path),
        "din_pred_path": str(args.din_pred_path),
        "output_dir": str(args.output_dir),
        "validation_rows": len(y_true),
        "validation_positive_count": int(y_true.sum()),
        "feature_cols": feature_cols,
        "din_weights": args.din_weights,
        "thresholds": args.thresholds,
        "save_predictions": args.save_predictions,
        "policy": (
            "Fixed-weight diagnostic only; validation is not used to optimize "
            "weights or train a meta-model."
        ),
    }
    (args.output_dir / "xgb_din_fusion_setup.json").write_text(
        json.dumps(setup, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    write_report(args.output_dir, setup, metrics_df)

    print("\nXGBoost + DIN fusion metrics:")
    print(metrics_df.to_string(index=False))
    print(f"\nOutputs written to {args.output_dir}")


if __name__ == "__main__":
    main()

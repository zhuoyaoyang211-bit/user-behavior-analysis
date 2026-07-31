"""OOF Stacking for the selected LightGBM and XGBoost models.

Workflow:
1. Sample a representative subset from the original training set.
2. Train fold-specific LightGBM/XGBoost models and generate out-of-fold
   predictions for the meta-model training rows.
3. Fit a logistic-regression meta-model on logit-transformed OOF predictions.
4. Score the untouched validation set with the selected base model artifacts.
5. Compare LightGBM, XGBoost, equal-weight blending, and OOF Stacking.

The validation set is never used to fit the meta-model or choose its threshold.
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
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier


PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_DIR / "output"
DEFAULT_TRAIN_PATH = OUTPUT_DIR / "train.parquet"
DEFAULT_VAL_PATH = OUTPUT_DIR / "val.parquet"
DEFAULT_BASELINE_DIR = OUTPUT_DIR / "baseline_models"
DEFAULT_TREE_DIR = OUTPUT_DIR / "optuna_tuned_models"
DEFAULT_OUTPUT_DIR = OUTPUT_DIR / "tree_stacking"
TARGET_COL = "label"
RANDOM_STATE = 42
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
EXCLUDE_COLUMNS = {
    "user_id",
    "item_id",
    "label",
    "last_time",
    "buy_path_type",
    "behavior_type",
    "item_category",
    "is_power_user",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--val-path", type=Path, default=DEFAULT_VAL_PATH)
    parser.add_argument("--tree-dir", type=Path, default=DEFAULT_TREE_DIR)
    parser.add_argument(
        "--lightgbm-path",
        type=Path,
        default=DEFAULT_BASELINE_DIR / "lightgbm_baseline.joblib",
        help="Selected LightGBM artifact. Default uses the stronger baseline LightGBM.",
    )
    parser.add_argument(
        "--xgboost-path",
        type=Path,
        default=DEFAULT_TREE_DIR / "xgboost_optuna.joblib",
        help="Selected XGBoost artifact. Default uses the stronger Optuna XGBoost.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--oof-sample-size",
        type=int,
        default=500_000,
        help="Number of original training rows used to create OOF meta-features.",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--fold-smote-ratio",
        type=float,
        default=0.10,
        help="Positive ratio produced by SMOTE inside each OOF fit fold.",
    )
    parser.add_argument("--n-jobs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=100_000)
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=DEFAULT_THRESHOLDS,
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.oof_sample_size <= 0:
        raise ValueError("--oof-sample-size must be positive.")
    if args.folds < 2:
        raise ValueError("--folds must be at least 2.")
    if not 0 < args.fold_smote_ratio < 1:
        raise ValueError("--fold-smote-ratio must be in (0, 1).")
    if args.n_jobs == 0:
        raise ValueError("--n-jobs cannot be 0.")
    for model_name, path in {
        "lightgbm": args.lightgbm_path,
        "xgboost": args.xgboost_path,
    }.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing selected {model_name} model: {path}")


def build_time_strata(frame: pd.DataFrame) -> pd.Series:
    """Build a compact date/day-type/hour stratum for representative sampling."""
    dt = pd.to_datetime(frame["last_time"])
    special_dates = {pd.Timestamp("2025-12-12").date()}
    day_type = np.select(
        [
            dt.dt.date.isin(special_dates),
            dt.dt.dayofweek >= 5,
        ],
        ["special", "weekend"],
        default="weekday",
    )
    hour_bin = pd.cut(
        dt.dt.hour,
        bins=[-1, 5, 11, 17, 23],
        labels=["night", "morning", "afternoon", "evening"],
    )
    return (
        dt.dt.date.astype(str)
        + "|"
        + pd.Series(day_type, index=frame.index).astype(str)
        + "|"
        + hour_bin.astype(str)
    )


def proportional_allocations(counts: pd.Series, target: int) -> pd.Series:
    """Allocate an exact target in proportion to group sizes."""
    if target > int(counts.sum()):
        raise ValueError("Allocation target exceeds available rows.")
    raw = counts.astype(float) * (target / float(counts.sum()))
    allocations = np.floor(raw).astype("int64")
    remaining = target - int(allocations.sum())
    order = (raw - allocations).sort_values(ascending=False).index
    for key in order:
        if remaining == 0:
            break
        if allocations.loc[key] < counts.loc[key]:
            allocations.loc[key] += 1
            remaining -= 1
    if remaining:
        raise RuntimeError("Could not allocate the requested sample size.")
    return allocations


def sample_training_rows(
    frame: pd.DataFrame,
    sample_size: int,
) -> pd.DataFrame:
    """Sample by label and time stratum without touching validation data."""
    sample_size = min(sample_size, len(frame))
    if sample_size == len(frame):
        return frame.reset_index(drop=True)

    work = frame.copy()
    work["_time_stratum"] = build_time_strata(work)
    label_alloc = proportional_allocations(
        work[TARGET_COL].value_counts().sort_index(),
        sample_size,
    )
    parts = []
    for label, label_target in label_alloc.items():
        label_rows = work[work[TARGET_COL] == label]
        strata_counts = label_rows["_time_stratum"].value_counts().sort_index()
        strata_alloc = proportional_allocations(strata_counts, int(label_target))
        for stratum, n_rows in strata_alloc.items():
            if n_rows <= 0:
                continue
            rows = label_rows[label_rows["_time_stratum"] == stratum]
            parts.append(rows.sample(int(n_rows), random_state=RANDOM_STATE))
    return (
        pd.concat(parts)
        .sample(frac=1.0, random_state=RANDOM_STATE)
        .drop(columns=["_time_stratum"])
        .reset_index(drop=True)
    )


def load_base_payloads(
    lightgbm_path: Path,
    xgboost_path: Path,
) -> dict[str, dict[str, Any]]:
    payloads = {
        "lightgbm": joblib.load(lightgbm_path),
        "xgboost": joblib.load(xgboost_path),
    }
    if payloads["lightgbm"]["feature_cols"] != payloads["xgboost"]["feature_cols"]:
        raise ValueError("LightGBM and XGBoost feature columns do not match.")
    payloads["lightgbm"]["artifact_path"] = str(lightgbm_path)
    payloads["xgboost"]["artifact_path"] = str(xgboost_path)
    return payloads


def get_feature_columns(
    train: pd.DataFrame,
    val: pd.DataFrame,
    payloads: dict[str, dict[str, Any]],
) -> list[str]:
    payload_cols = payloads["lightgbm"]["feature_cols"]
    missing = sorted(set(payload_cols) - set(train.columns) - set(val.columns))
    if missing:
        raise ValueError(f"Missing tuned-model feature columns: {missing}")
    feature_cols = [
        col
        for col in payload_cols
        if col in train.columns and col in val.columns and col not in EXCLUDE_COLUMNS
    ]
    if feature_cols != payload_cols:
        raise ValueError("Feature columns differ from the tuned model artifact.")
    return feature_cols


def smote_training_rows(
    X: np.ndarray,
    y: np.ndarray,
    positive_ratio: float = 0.10,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply SMOTE inside one OOF fit fold to match Optuna's 10% training ratio."""
    positive_count = int(y.sum())
    negative_count = len(y) - positive_count
    current_ratio = positive_count / len(y)
    if positive_count < 2 or negative_count == 0 or positive_ratio <= current_ratio:
        return X, y

    minority_majority_ratio = positive_ratio / (1.0 - positive_ratio)
    smote = SMOTE(
        random_state=RANDOM_STATE,
        sampling_strategy=minority_majority_ratio,
        k_neighbors=min(5, positive_count - 1),
    )
    return smote.fit_resample(X, y)


def build_fold_model(
    model_name: str,
    payload: dict[str, Any],
    n_jobs: int,
) -> Any:
    params = dict(payload["params"])
    params["n_jobs"] = n_jobs
    params["random_state"] = RANDOM_STATE
    if model_name == "lightgbm":
        params.update({"objective": "binary", "verbose": -1})
        return LGBMClassifier(**params)
    params.update(
        {
            "objective": "binary:logistic",
            "eval_metric": ["logloss", "auc", "aucpr"],
            "tree_method": "hist",
            "verbosity": 0,
        }
    )
    return XGBClassifier(**params)


def predict_in_batches(model: Any, X: np.ndarray, batch_size: int) -> np.ndarray:
    probabilities = []
    for start in range(0, len(X), batch_size):
        stop = min(start + batch_size, len(X))
        probabilities.append(model.predict_proba(X[start:stop])[:, 1])
    return np.concatenate(probabilities)


def meta_feature_matrix(predictions: dict[str, np.ndarray]) -> np.ndarray:
    """Build calibrated meta-features from base-model probabilities."""
    cols = []
    for model_name in ("lightgbm", "xgboost"):
        probability = np.clip(predictions[model_name], 1e-6, 1.0 - 1e-6)
        cols.append(np.log(probability / (1.0 - probability)))
    return np.column_stack(cols)


def make_oof_predictions(
    sample: pd.DataFrame,
    feature_cols: list[str],
    payloads: dict[str, dict[str, Any]],
    folds: int,
    fold_smote_ratio: float,
    n_jobs: int,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, np.ndarray], pd.DataFrame]:
    X = sample[feature_cols].to_numpy(dtype=np.float32, copy=True)
    y = sample[TARGET_COL].to_numpy(dtype=np.int8, copy=True)
    if np.isnan(X).any():
        raise ValueError("Missing values found in OOF features.")

    splitter = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    oof = {
        "lightgbm": np.zeros(len(sample), dtype=np.float64),
        "xgboost": np.zeros(len(sample), dtype=np.float64),
    }
    fold_rows = []
    for fold, (fit_idx, holdout_idx) in enumerate(splitter.split(X, y), start=1):
        print(f"OOF fold {fold}/{folds}: fit={len(fit_idx):,}, holdout={len(holdout_idx):,}", flush=True)
        X_fit, y_fit = smote_training_rows(
            X[fit_idx],
            y[fit_idx],
            positive_ratio=fold_smote_ratio,
        )
        fold_start = time.time()
        for model_name in ("lightgbm", "xgboost"):
            model = build_fold_model(model_name, payloads[model_name], n_jobs)
            model.fit(X_fit, y_fit)
            oof[model_name][holdout_idx] = predict_in_batches(
                model,
                X[holdout_idx],
                batch_size,
            )
        fold_rows.append(
            {
                "fold": fold,
                "fit_rows_before_oversampling": len(fit_idx),
                "fit_rows_after_smote": len(X_fit),
                "fit_positive_ratio_after_smote": float(y_fit.mean()),
                "holdout_rows": len(holdout_idx),
                "holdout_positive_count": int(y[holdout_idx].sum()),
                "seconds": time.time() - fold_start,
            }
        )
        del X_fit, y_fit
    return y, oof, pd.DataFrame(fold_rows)


def score_metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
    thresholds: list[float],
    threshold: float,
) -> dict[str, Any]:
    y_pred = (probability >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    threshold_rows = []
    for candidate in thresholds:
        candidate_pred = (probability >= candidate).astype(np.int8)
        ttn, tfp, tfn, ttp = confusion_matrix(y_true, candidate_pred).ravel()
        threshold_rows.append(
            {
                "threshold": candidate,
                "precision": precision_score(y_true, candidate_pred, zero_division=0),
                "recall": recall_score(y_true, candidate_pred, zero_division=0),
                "f1": f1_score(y_true, candidate_pred, zero_division=0),
                "tn": int(ttn),
                "fp": int(tfp),
                "fn": int(tfn),
                "tp": int(ttp),
            }
        )
    best = max(threshold_rows, key=lambda row: (row["f1"], row["precision"]))
    return {
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "pr_auc_ap": float(average_precision_score(y_true, probability)),
        "log_loss": float(log_loss(y_true, np.clip(probability, 1e-15, 1 - 1e-15))),
        "precision_at_threshold": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_at_threshold": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_at_threshold": float(f1_score(y_true, y_pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "threshold": threshold,
        "best_threshold": float(best["threshold"]),
        "best_f1": float(best["f1"]),
        "threshold_rows": threshold_rows,
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()

    print("Loading training and validation data ...", flush=True)
    train = pd.read_parquet(args.train_path)
    val = pd.read_parquet(args.val_path)
    payloads = load_base_payloads(
        lightgbm_path=args.lightgbm_path,
        xgboost_path=args.xgboost_path,
    )
    feature_cols = get_feature_columns(train, val, payloads)
    sample = sample_training_rows(train, args.oof_sample_size)
    print(
        f"OOF sample rows={len(sample):,}, positive={int(sample[TARGET_COL].sum()):,}",
        flush=True,
    )

    y_oof, oof_base, fold_df = make_oof_predictions(
        sample=sample,
        feature_cols=feature_cols,
        payloads=payloads,
        folds=args.folds,
        fold_smote_ratio=args.fold_smote_ratio,
        n_jobs=args.n_jobs,
        batch_size=args.batch_size,
    )
    oof_matrix = meta_feature_matrix(oof_base)
    meta_model = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        random_state=RANDOM_STATE,
    )
    meta_model.fit(oof_matrix, y_oof)
    oof_stacking_probability = meta_model.predict_proba(oof_matrix)[:, 1]
    oof_metrics = score_metrics(
        y_oof,
        oof_stacking_probability,
        args.thresholds,
        threshold=0.5,
    )
    selected_threshold = oof_metrics["best_threshold"]

    val_X = val[feature_cols].to_numpy(dtype=np.float32, copy=True)
    val_y = val[TARGET_COL].to_numpy(dtype=np.int8, copy=True)
    if np.isnan(val_X).any():
        raise ValueError("Missing values found in validation features.")
    val_base = {
        name: predict_in_batches(payloads[name]["model"], val_X, args.batch_size)
        for name in ("lightgbm", "xgboost")
    }
    val_matrix = meta_feature_matrix(val_base)
    val_stack = meta_model.predict_proba(val_matrix)[:, 1]
    # The transparent baseline must average probabilities, not logit features.
    val_equal = (val_base["lightgbm"] + val_base["xgboost"]) / 2.0

    probabilities = {
        "lightgbm": val_base["lightgbm"],
        "xgboost": val_base["xgboost"],
        "tree_equal": val_equal,
        "stacking": val_stack,
    }
    summary_rows = []
    threshold_rows = []
    for name, probability in probabilities.items():
        metrics = score_metrics(
            val_y,
            probability,
            args.thresholds,
            threshold=selected_threshold if name == "stacking" else 0.5,
        )
        summary_rows.append(
            {
                "evaluation_set": "validation",
                "model": name,
                "roc_auc": metrics["roc_auc"],
                "pr_auc_ap": metrics["pr_auc_ap"],
                "log_loss": metrics["log_loss"],
                "precision_at_threshold": metrics["precision_at_threshold"],
                "recall_at_threshold": metrics["recall_at_threshold"],
                "f1_at_threshold": metrics["f1_at_threshold"],
                "threshold": metrics["threshold"],
                "best_threshold_on_validation": metrics["best_threshold"],
                "best_f1_on_validation": metrics["best_f1"],
                "tn": metrics["tn"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "tp": metrics["tp"],
            }
        )
        for row in metrics["threshold_rows"]:
            threshold_rows.append({"model": name, **row})

    pd.DataFrame(summary_rows).to_csv(
        args.output_dir / "stacking_validation_metrics.csv",
        index=False,
    )
    pd.DataFrame(threshold_rows).to_csv(
        args.output_dir / "stacking_validation_threshold_metrics.csv",
        index=False,
    )
    pd.DataFrame(
        {
            "lightgbm": oof_base["lightgbm"],
            "xgboost": oof_base["xgboost"],
            "stacking": oof_stacking_probability,
            "label": y_oof,
        }
    ).to_csv(args.output_dir / "stacking_oof_predictions.csv", index=False)
    pd.DataFrame(
        {
            **probabilities,
            "label": val_y,
        }
    ).to_csv(args.output_dir / "stacking_validation_predictions.csv", index=False)
    fold_df.to_csv(args.output_dir / "stacking_oof_folds.csv", index=False)
    joblib.dump(meta_model, args.output_dir / "stacking_meta_model.joblib")

    setup = {
        "train_path": str(args.train_path),
        "val_path": str(args.val_path),
        "tree_dir": str(args.tree_dir),
        "lightgbm_path": str(args.lightgbm_path),
        "xgboost_path": str(args.xgboost_path),
        "output_dir": str(args.output_dir),
        "feature_cols": feature_cols,
        "oof_sample_size": len(sample),
        "oof_positive_count": int(y_oof.sum()),
        "folds": args.folds,
        "n_jobs": args.n_jobs,
        "oversampling": (
            f"SMOTE to {args.fold_smote_ratio:.0%} positive ratio inside each "
            "OOF fit fold; OOF holdout and validation keep original distribution"
        ),
        "meta_model": "LogisticRegression without class_weight",
        "meta_features": "logit-transformed LightGBM and XGBoost probabilities",
        "meta_training": "OOF predictions only; validation excluded",
        "stacking_threshold": selected_threshold,
        "elapsed_seconds": time.time() - start,
    }
    (args.output_dir / "stacking_setup.json").write_text(
        json.dumps(setup, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print("\nValidation results:", flush=True)
    print(pd.DataFrame(summary_rows).to_string(index=False), flush=True)
    print(f"\nOutputs written to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()

"""Tune tree baseline models with Optuna and export the best single model.

Optuna tuning is performed on an internal split sampled from train.parquet.
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
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
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

from sequence_modeling.metrics import compute_threshold_metrics
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
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "output" / "optuna_tuned_models"
DEFAULT_BASELINE_METRICS_PATH = (
    PROJECT_DIR / "output" / "baseline_models" / "baseline_metrics.csv"
)
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
PRIMARY_METRIC = "pr_auc_ap"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument(
        "--final-train-path",
        type=Path,
        default=DEFAULT_FINAL_TRAIN_PATH,
        help="Full SMOTE r10 training set used after Optuna selects parameters.",
    )
    parser.add_argument("--val-path", type=Path, default=DEFAULT_VAL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--baseline-metrics-path",
        type=Path,
        default=DEFAULT_BASELINE_METRICS_PATH,
        help="Existing baseline_metrics.csv used for the final comparison.",
    )
    parser.add_argument(
        "--models",
        default="lightgbm,xgboost",
        help="Comma-separated model list. Supported: lightgbm,xgboost.",
    )
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument(
        "--timeout-per-model",
        type=int,
        default=None,
        help="Optional Optuna timeout in seconds for each model.",
    )
    parser.add_argument("--tune-sample-size", type=int, default=500_000)
    parser.add_argument("--tune-val-frac", type=float, default=0.2)
    parser.add_argument("--smote-ratio", type=float, default=0.10)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Thread count passed to LightGBM/XGBoost.",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=DEFAULT_THRESHOLDS,
        help="Probability thresholds used for post-training F1 analysis.",
    )
    return parser.parse_args()


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


def suggest_lightgbm_params(trial: optuna.Trial, n_jobs: int) -> dict[str, Any]:
    max_depth = trial.suggest_int("max_depth", 3, 10)
    max_leaves = min(255, (2**max_depth) - 1)
    return {
        "objective": "binary",
        "n_estimators": trial.suggest_int("n_estimators", 200, 900, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": max_depth,
        "num_leaves": trial.suggest_int("num_leaves", 7, max_leaves),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 120),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
        "random_state": RANDOM_STATE,
        "n_jobs": n_jobs,
        "verbose": -1,
    }


def suggest_xgboost_params(trial: optuna.Trial, n_jobs: int) -> dict[str, Any]:
    grow_policy = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
    max_depth = trial.suggest_int("max_depth", 3, 10)
    params: dict[str, Any] = {
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "auc", "aucpr"],
        "n_estimators": trial.suggest_int("n_estimators", 200, 900, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": max_depth,
        "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 20.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 1e-8, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "grow_policy": grow_policy,
        "tree_method": "hist",
        "random_state": RANDOM_STATE,
        "n_jobs": n_jobs,
        "verbosity": 0,
    }
    if grow_policy == "lossguide":
        params["max_leaves"] = trial.suggest_int("max_leaves", 16, 255)
    return params


def build_model(model_name: str, params: dict[str, Any]) -> Any:
    if model_name == "lightgbm":
        return LGBMClassifier(**params)
    if model_name == "xgboost":
        return XGBClassifier(**params)
    raise ValueError(f"Unsupported model: {model_name}")


def objective_factory(
    model_name: str,
    X_tune: np.ndarray,
    y_tune: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_jobs: int,
) -> Any:
    def objective(trial: optuna.Trial) -> float:
        if model_name == "lightgbm":
            params = suggest_lightgbm_params(trial, n_jobs)
        elif model_name == "xgboost":
            params = suggest_xgboost_params(trial, n_jobs)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        model = build_model(model_name, params)
        model.fit(X_tune, y_tune)
        val_prob = predict_positive_probability(model, X_val)
        return average_precision_score(y_val, val_prob)

    return objective


def tune_model(
    model_name: str,
    X_tune: np.ndarray,
    y_tune: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int,
    timeout_per_model: int | None,
    n_jobs: int,
    output_dir: Path,
) -> dict[str, Any]:
    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=f"{model_name}_{PRIMARY_METRIC}",
    )
    study.optimize(
        objective_factory(model_name, X_tune, y_tune, X_val, y_val, n_jobs),
        n_trials=n_trials,
        timeout=timeout_per_model,
        show_progress_bar=True,
    )

    trials_df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    trials_df.to_csv(output_dir / f"{model_name}_optuna_trials.csv", index=False)

    best = {
        "model": model_name,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "n_trials": len(study.trials),
    }
    (output_dir / f"{model_name}_best_params.json").write_text(
        json.dumps(best, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return best


def feature_effects(model_name: str, model: Any, feature_cols: list[str]) -> pd.DataFrame:
    if hasattr(model, "feature_importances_"):
        return pd.DataFrame(
            {
                "feature": feature_cols,
                "importance": model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

    return pd.DataFrame({"feature": feature_cols, "importance": np.nan})


def read_baseline_metrics(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    baseline_df = pd.read_csv(path)
    baseline_df.insert(0, "source", "baseline")
    baseline_df["rank_scope"] = "comparison"
    return baseline_df


def compute_model_threshold_metrics(
    model_name: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: list[float],
) -> pd.DataFrame:
    threshold_df = pd.DataFrame(compute_threshold_metrics(y_true, y_prob, thresholds))
    threshold_df.insert(0, "model", f"{model_name}_optuna")
    return threshold_df


def best_threshold_summary(threshold_df: pd.DataFrame) -> dict[str, Any]:
    best_row = threshold_df.sort_values(
        ["f1", "precision"],
        ascending=[False, False],
    ).iloc[0]
    return {
        "best_f1_threshold": best_row["threshold"],
        "best_f1": best_row["f1"],
        "precision_at_best_f1": best_row["precision"],
        "recall_at_best_f1": best_row["recall"],
    }


def metrics_at_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "selected_threshold_precision": precision_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "selected_threshold_recall": recall_score(y_true, y_pred, zero_division=0),
        "selected_threshold_f1": f1_score(y_true, y_pred, zero_division=0),
        "selected_threshold_tn": int(tn),
        "selected_threshold_fp": int(fp),
        "selected_threshold_fn": int(fn),
        "selected_threshold_tp": int(tp),
    }


def complete_model_params(
    model_name: str,
    best_params: dict[str, Any],
    n_jobs: int,
) -> dict[str, Any]:
    params = best_params.copy()
    params.update({"random_state": RANDOM_STATE, "n_jobs": n_jobs})
    if model_name == "lightgbm":
        params.update({"objective": "binary", "verbose": -1})
    elif model_name == "xgboost":
        params.update(
            {
                "objective": "binary:logistic",
                "eval_metric": ["logloss", "auc", "aucpr"],
                "tree_method": "hist",
                "verbosity": 0,
            }
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return params


def write_report(
    output_dir: Path,
    setup: dict[str, Any],
    tuned_metrics_df: pd.DataFrame,
    selection_threshold_metrics_df: pd.DataFrame,
    threshold_metrics_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    best_model_name: str,
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
        "# Optuna Tuned Model Comparison",
        "",
        "## Controlled Setup",
        "",
        f"- Raw train for Optuna search: `{setup['train_path']}`",
        f"- Final train for refit: `{setup['final_train_path']}`",
        f"- Final validation: `{setup['val_path']}`",
        f"- Features: {len(setup['feature_cols'])} common numeric columns",
        f"- Raw train shape: {tuple(setup['raw_train_shape'])}",
        f"- Tuning sample shape: {tuple(setup['tune_sample_shape'])}",
        f"- Tuning train shape before SMOTE: {tuple(setup['tune_train_shape'])}, "
        f"positive rate: {setup['tune_train_pos_rate_before_smote']:.6f}",
        f"- Tuning train positive rate after SMOTE: {setup['tune_train_pos_rate_after_smote']:.6f}",
        f"- Tuning validation shape: {tuple(setup['tune_val_shape'])}, "
        f"positive rate: {setup['tune_val_pos_rate']:.6f}",
        f"- Final train shape: {tuple(setup['final_train_shape'])}, positive rate: {setup['final_train_pos_rate']:.6f}",
        f"- Final validation shape: {tuple(setup['final_val_shape'])}, positive rate: {setup['final_val_pos_rate']:.6f}",
        f"- Random state: {RANDOM_STATE}",
        f"- Default classification threshold: {THRESHOLD}",
        f"- Thresholds for F1 analysis: {setup['thresholds']}",
        f"- Optuna objective: maximize `{PRIMARY_METRIC}` on internal tuning validation",
        "- Final validation is used once for reporting and is not used to choose parameters or thresholds.",
        f"- Best single model: `{best_model_name}`",
        "",
        "## Tuned Models",
        "",
        to_markdown_table(tuned_metrics_df),
        "",
        "## Internal Threshold Selection",
        "",
        to_markdown_table(selection_threshold_metrics_df),
        "",
        "## Final Validation Threshold Diagnostics",
        "",
        to_markdown_table(threshold_metrics_df),
        "",
        "## Baseline vs Tuned",
        "",
        to_markdown_table(comparison_df),
        "",
    ]
    (output_dir / "optuna_tuned_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def normalize_model_names(model_arg: str) -> list[str]:
    names = [name.strip().lower() for name in model_arg.split(",") if name.strip()]
    supported = {"lightgbm", "xgboost"}
    unsupported = sorted(set(names) - supported)
    if unsupported:
        raise ValueError(f"Unsupported models: {unsupported}. Supported: {sorted(supported)}")
    if not names:
        raise ValueError("At least one model must be selected.")
    return names


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_names = normalize_model_names(args.models)

    raw_train = pd.read_parquet(args.train_path)
    final_train = pd.read_parquet(args.final_train_path)
    final_val = pd.read_parquet(args.val_path)
    feature_cols = numeric_feature_columns(raw_train, final_train, final_val)

    tune_sample = sample_by_label_time_strata(
        raw_train,
        sample_size=min(args.tune_sample_size, len(raw_train)),
        random_state=RANDOM_STATE,
    )
    tune_train, tune_val = split_within_time_strata(
        tune_sample,
        val_frac=args.tune_val_frac,
    )
    X_tune_train, y_tune_train, tune_smote_info = smote_to_ratio(
        tune_train,
        feature_cols,
        target_pos_ratio=args.smote_ratio,
        random_state=RANDOM_STATE,
    )
    X_tune_val, y_tune_val = to_xy(tune_val, feature_cols)
    X_final_train, y_final_train = to_xy(final_train, feature_cols)
    X_val, y_val = to_xy(final_val, feature_cols)

    setup = {
        "train_path": str(args.train_path),
        "final_train_path": str(args.final_train_path),
        "val_path": str(args.val_path),
        "output_dir": str(args.output_dir),
        "baseline_metrics_path": str(args.baseline_metrics_path),
        "models": model_names,
        "n_trials": args.n_trials,
        "timeout_per_model": args.timeout_per_model,
        "tune_sample_size": args.tune_sample_size,
        "tune_val_frac": args.tune_val_frac,
        "smote_ratio": args.smote_ratio,
        "feature_cols": feature_cols,
        "raw_train_shape": raw_train.shape,
        "tune_sample_shape": tune_sample.shape,
        "tune_train_shape": tune_train.shape,
        "tune_val_shape": tune_val.shape,
        "tune_train_pos_rate_before_smote": float(tune_train[TARGET_COL].mean()),
        "tune_train_pos_rate_after_smote": float(y_tune_train.mean()),
        "tune_val_pos_rate": float(y_tune_val.mean()),
        "tune_smote_info": tune_smote_info,
        "final_train_shape": final_train.shape,
        "final_val_shape": final_val.shape,
        "final_train_pos_rate": float(y_final_train.mean()),
        "final_val_pos_rate": float(y_val.mean()),
        "random_state": RANDOM_STATE,
        "threshold": THRESHOLD,
        "thresholds": args.thresholds,
        "primary_metric": PRIMARY_METRIC,
        "selection_policy": (
            "sample 500k from train.parquet by label × time stratum; "
            "split each time stratum into tune train/validation; "
            "apply SMOTE only to tune train"
        ),
        "final_policy": (
            "refit best Optuna parameters on full train_smote_r10.parquet; "
            "evaluate once on full val.parquet"
        ),
    }
    (args.output_dir / "optuna_setup.json").write_text(
        json.dumps(setup, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    tuned_rows = []
    selection_threshold_frames = []
    threshold_metric_frames = []
    best_params_by_model = {}
    fitted_models = {}
    for model_name in model_names:
        print(f"\nTuning {model_name} with Optuna ...", flush=True)
        tune_start = time.time()
        best = tune_model(
            model_name=model_name,
            X_tune=X_tune_train,
            y_tune=y_tune_train,
            X_val=X_tune_val,
            y_val=y_tune_val,
            n_trials=args.n_trials,
            timeout_per_model=args.timeout_per_model,
            n_jobs=args.n_jobs,
            output_dir=args.output_dir,
        )
        tune_seconds = time.time() - tune_start
        best_params_by_model[model_name] = best

        final_params = complete_model_params(
            model_name=model_name,
            best_params=best["best_params"],
            n_jobs=args.n_jobs,
        )

        print(f"Selecting threshold for {model_name} on internal split ...", flush=True)
        selection_model = build_model(model_name, final_params)
        selection_model.fit(X_tune_train, y_tune_train)
        selection_prob = predict_positive_probability(selection_model, X_tune_val)
        selection_threshold_df = compute_model_threshold_metrics(
            model_name=model_name,
            y_true=y_tune_val,
            y_prob=selection_prob,
            thresholds=args.thresholds,
        )
        selection_threshold_df["threshold_source"] = "internal_tuning_validation"
        selection_threshold_frames.append(selection_threshold_df)
        selected_threshold_summary = best_threshold_summary(selection_threshold_df)
        selected_threshold = float(selected_threshold_summary["best_f1_threshold"])

        print(f"Refitting {model_name} with best params on full SMOTE r10 train ...", flush=True)

        model = build_model(model_name, final_params)
        train_start = time.time()
        model.fit(X_final_train, y_final_train)
        train_seconds = time.time() - train_start
        val_prob = predict_positive_probability(model, X_val)
        threshold_df = compute_model_threshold_metrics(
            model_name=model_name,
            y_true=y_val,
            y_prob=val_prob,
            thresholds=args.thresholds,
        )
        threshold_df["threshold_source"] = "final_validation_diagnostic_only"
        threshold_metric_frames.append(threshold_df)
        final_val_threshold_summary = best_threshold_summary(threshold_df)

        row = {
            "source": "optuna_tuned",
            "rank_scope": "comparison",
            "model": f"{model_name}_optuna",
            "tune_seconds": tune_seconds,
            "train_seconds": train_seconds,
            "n_trials": best["n_trials"],
            **evaluate(y_val, val_prob),
            "selected_threshold": selected_threshold,
            "selected_threshold_source": "internal_tuning_validation",
            "selection_best_f1": selected_threshold_summary["best_f1"],
            "selection_precision_at_best_f1": selected_threshold_summary[
                "precision_at_best_f1"
            ],
            "selection_recall_at_best_f1": selected_threshold_summary[
                "recall_at_best_f1"
            ],
            **metrics_at_threshold(y_val, val_prob, selected_threshold),
            "final_val_best_f1_threshold_diagnostic": final_val_threshold_summary[
                "best_f1_threshold"
            ],
            "final_val_best_f1_diagnostic": final_val_threshold_summary["best_f1"],
        }
        tuned_rows.append(row)
        fitted_models[model_name] = {
            "model": model,
            "params": final_params,
            "metrics": row,
            "selected_threshold_summary": selected_threshold_summary,
        }

        joblib.dump(
            {
                "model": model,
                "feature_cols": feature_cols,
                "target_col": TARGET_COL,
                "threshold": THRESHOLD,
                "thresholds": args.thresholds,
                "selected_threshold": selected_threshold,
                "selected_threshold_source": "internal_tuning_validation",
                "best_f1_threshold": selected_threshold,
                "params": final_params,
                "optuna_best": best,
                "primary_metric": PRIMARY_METRIC,
                "selection_threshold_summary": selected_threshold_summary,
            },
            args.output_dir / f"{model_name}_optuna.joblib",
        )
        feature_effects(model_name, model, feature_cols).to_csv(
            args.output_dir / f"{model_name}_optuna_feature_effects.csv",
            index=False,
        )
        print(
            f"Finished {model_name}: tuned {tune_seconds:.2f}s, "
            f"refit {train_seconds:.2f}s, {PRIMARY_METRIC}={row[PRIMARY_METRIC]:.6f}",
            flush=True,
        )

    tuned_metrics_df = pd.DataFrame(tuned_rows).sort_values(
        PRIMARY_METRIC,
        ascending=False,
    )
    tuned_metrics_df.to_csv(args.output_dir / "optuna_tuned_metrics.csv", index=False)
    selection_threshold_metrics_df = pd.concat(
        selection_threshold_frames,
        ignore_index=True,
    ).sort_values(
        ["model", "f1", "precision"],
        ascending=[True, False, False],
    )
    selection_threshold_metrics_df.to_csv(
        args.output_dir / "optuna_selection_threshold_metrics.csv",
        index=False,
    )
    threshold_metrics_df = pd.concat(
        threshold_metric_frames,
        ignore_index=True,
    ).sort_values(
        ["model", "f1", "precision"],
        ascending=[True, False, False],
    )
    threshold_metrics_df.to_csv(
        args.output_dir / "optuna_threshold_metrics.csv",
        index=False,
    )
    (args.output_dir / "optuna_best_params.json").write_text(
        json.dumps(best_params_by_model, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    baseline_df = read_baseline_metrics(args.baseline_metrics_path)
    comparison_df = pd.concat([baseline_df, tuned_metrics_df], ignore_index=True)
    comparison_df = comparison_df.sort_values(PRIMARY_METRIC, ascending=False)
    comparison_df.to_csv(args.output_dir / "baseline_vs_optuna_metrics.csv", index=False)

    best_row = tuned_metrics_df.iloc[0]
    best_base_name = str(best_row["model"]).removesuffix("_optuna")
    best_payload = fitted_models[best_base_name]
    joblib.dump(
        {
            "model": best_payload["model"],
            "model_name": str(best_row["model"]),
            "feature_cols": feature_cols,
            "target_col": TARGET_COL,
            "threshold": THRESHOLD,
            "thresholds": args.thresholds,
            "selected_threshold": best_payload["metrics"]["selected_threshold"],
            "selected_threshold_source": "internal_tuning_validation",
            "best_f1_threshold": best_payload["metrics"]["selected_threshold"],
            "params": best_payload["params"],
            "metrics": best_payload["metrics"],
            "primary_metric": PRIMARY_METRIC,
            "selection_threshold_summary": best_payload["selected_threshold_summary"],
        },
        args.output_dir / "best_single_model.joblib",
    )

    write_report(
        output_dir=args.output_dir,
        setup=setup,
        tuned_metrics_df=tuned_metrics_df,
        selection_threshold_metrics_df=selection_threshold_metrics_df,
        threshold_metrics_df=threshold_metrics_df,
        comparison_df=comparison_df,
        best_model_name=str(best_row["model"]),
    )

    print("\nTuned validation metrics:")
    print(tuned_metrics_df.to_string(index=False))
    print("\nInternal threshold selection metrics:")
    print(selection_threshold_metrics_df.to_string(index=False))
    print("\nTuned threshold metrics:")
    print(threshold_metrics_df.to_string(index=False))
    print("\nBaseline vs tuned validation metrics:")
    print(comparison_df.to_string(index=False))
    print(f"\nBest single model saved to: {args.output_dir / 'best_single_model.joblib'}")


if __name__ == "__main__":
    main()

"""Evaluate traditional baseline models under multiple thresholds."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from common.logger import get_logger
from config import get_config
from sequence_modeling.metrics import compute_threshold_metrics


logger = get_logger(__name__)
TARGET_COL = "label"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    project_root = get_config().project_root
    output_dir = project_root / "output"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-paths",
        nargs="+",
        type=Path,
        default=[
            output_dir / "baseline_models" / "logistic_regression_baseline.joblib",
            output_dir / "baseline_models" / "xgboost_baseline.joblib",
            output_dir / "baseline_models" / "lightgbm_baseline.joblib",
        ],
    )
    parser.add_argument("--val-path", type=Path, default=output_dir / "val.parquet")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=output_dir / "baseline_threshold_analysis",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[
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
        ],
    )
    return parser.parse_args()


def infer_model_name(model_path: Path) -> str:
    """Infer model name from baseline checkpoint path."""
    name = model_path.stem
    return name.removesuffix("_baseline")


def score_model(
    model_path: Path,
    val: pd.DataFrame,
    thresholds: list[float],
) -> pd.DataFrame:
    """Score one saved baseline model under multiple thresholds.

    Args:
        model_path: Saved joblib baseline artifact path.
        val: Validation DataFrame.
        thresholds: Probability thresholds to evaluate.

    Returns:
        Threshold metrics DataFrame for the model.
    """
    artifact = joblib.load(model_path)
    model = artifact["model"]
    feature_cols = artifact["feature_cols"]
    X_val = val[feature_cols].to_numpy(dtype="float32", copy=True)
    y_val = val[TARGET_COL].to_numpy(dtype="int8", copy=True)
    y_prob = model.predict_proba(X_val)[:, 1]
    metrics = pd.DataFrame(compute_threshold_metrics(y_val, y_prob, thresholds))
    metrics.insert(0, "model", infer_model_name(model_path))
    metrics["baseline_version_path"] = str(model_path)
    return metrics


def write_report(output_dir: Path, metrics_df: pd.DataFrame) -> None:
    """Write markdown threshold analysis report."""
    view = metrics_df.copy()
    for col in view.select_dtypes(include=["float"]).columns:
        view[col] = view[col].map(lambda value: f"{value:.6f}")
    columns = [
        "model",
        "threshold",
        "precision",
        "recall",
        "f1",
        "accuracy",
        "tn",
        "fp",
        "fn",
        "tp",
    ]
    view = view[columns]
    header = "| " + " | ".join(view.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(view.columns)) + " |"
    body = ["| " + " | ".join(map(str, row)) + " |" for row in view.to_numpy()]
    lines = [
        "# Traditional Baseline Threshold Analysis",
        "",
        "This report evaluates fixed probability thresholds for saved baseline "
        "models without retraining.",
        "",
        header,
        separator,
        *body,
        "",
    ]
    (output_dir / "traditional_threshold_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    """Evaluate saved traditional baselines under multiple thresholds."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading validation data from %s", args.val_path)
    val = pd.read_parquet(args.val_path)
    rows = []
    for model_path in args.model_paths:
        logger.info("Scoring %s", model_path)
        rows.append(score_model(model_path, val, args.thresholds))

    metrics_df = pd.concat(rows, ignore_index=True).sort_values(
        ["model", "f1", "precision"],
        ascending=[True, False, False],
    )
    metrics_path = args.output_dir / "traditional_threshold_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    write_report(args.output_dir, metrics_df)
    logger.info("Threshold metrics saved to %s", metrics_path)
    logger.info("\n%s", metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()

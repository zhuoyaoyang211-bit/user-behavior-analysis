"""Evaluate traditional baseline models under multiple thresholds.

Saved model artifacts carry thresholds selected on the internal train split.
The full validation threshold sweep here is diagnostic only.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

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
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Score one saved baseline model under multiple thresholds.

    Args:
        model_path: Saved joblib baseline artifact path.
        val: Validation DataFrame.
        thresholds: Probability thresholds to evaluate.

    Returns:
        Threshold sweep DataFrame plus the row for the artifact-selected threshold.
    """
    artifact = joblib.load(model_path)
    model = artifact["model"]
    feature_cols = artifact["feature_cols"]
    model_name = infer_model_name(model_path)
    X_val = val[feature_cols].to_numpy(dtype="float32", copy=True)
    y_val = val[TARGET_COL].to_numpy(dtype="int8", copy=True)
    y_prob = model.predict_proba(X_val)[:, 1]
    metrics = pd.DataFrame(compute_threshold_metrics(y_val, y_prob, thresholds))
    metrics.insert(0, "model", model_name)
    metrics["baseline_version_path"] = str(model_path)
    metrics["threshold_source"] = "final_validation_diagnostic_only"

    selected_threshold = artifact.get(
        "selected_threshold",
        artifact.get("best_f1_threshold", artifact.get("threshold", 0.5)),
    )
    selected_source = artifact.get(
        "selected_threshold_source",
        "artifact_threshold_or_default",
    )
    selected_metrics = compute_threshold_metrics(
        y_val,
        y_prob,
        [float(selected_threshold)],
    )[0]
    selected_row = {
        "model": model_name,
        **selected_metrics,
        "selected_threshold_source": selected_source,
        "baseline_version_path": str(model_path),
    }
    return metrics, selected_row


def write_report(
    output_dir: Path,
    selected_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
) -> None:
    """Write markdown threshold analysis report."""
    def to_markdown_table(df: pd.DataFrame) -> str:
        view = df.copy()
        for col in view.select_dtypes(include=["float"]).columns:
            view[col] = view[col].map(lambda value: f"{value:.6f}")
        header = "| " + " | ".join(view.columns) + " |"
        separator = "| " + " | ".join(["---"] * len(view.columns)) + " |"
        body = ["| " + " | ".join(map(str, row)) + " |" for row in view.to_numpy()]
        return "\n".join([header, separator, *body])

    selected_columns = [
        "model",
        "threshold",
        "selected_threshold_source",
        "precision",
        "recall",
        "f1",
        "accuracy",
        "tn",
        "fp",
        "fn",
        "tp",
    ]
    diagnostic_columns = [
        "model",
        "threshold",
        "threshold_source",
        "precision",
        "recall",
        "f1",
        "accuracy",
        "tn",
        "fp",
        "fn",
        "tp",
    ]
    lines = [
        "# Traditional Baseline Threshold Analysis",
        "",
        "Saved model artifacts carry thresholds selected on the internal selection "
        "validation split. The full validation threshold sweep below is diagnostic "
        "only and is not used to choose thresholds.",
        "",
        "## Artifact-Selected Thresholds on Final Validation",
        "",
        to_markdown_table(selected_df[selected_columns]),
        "",
        "## Final Validation Threshold Diagnostics",
        "",
        to_markdown_table(metrics_df[diagnostic_columns]),
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
    selected_rows = []
    for model_path in args.model_paths:
        logger.info("Scoring %s", model_path)
        threshold_df, selected_row = score_model(model_path, val, args.thresholds)
        rows.append(threshold_df)
        selected_rows.append(selected_row)

    metrics_df = pd.concat(rows, ignore_index=True).sort_values(
        ["model", "f1", "precision"],
        ascending=[True, False, False],
    )
    selected_df = pd.DataFrame(selected_rows).sort_values(
        ["f1", "precision"],
        ascending=[False, False],
    )
    selected_path = args.output_dir / "traditional_selected_threshold_metrics.csv"
    selected_df.to_csv(selected_path, index=False)
    metrics_path = args.output_dir / "traditional_threshold_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    write_report(args.output_dir, selected_df, metrics_df)
    logger.info("Selected threshold metrics saved to %s", selected_path)
    logger.info("Threshold metrics saved to %s", metrics_path)
    logger.info("\n%s", selected_df.to_string(index=False))
    logger.info("\n%s", metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()

"""
Validation Metrics & Visualisation
====================================
- Aggregate metrics across walk-forward folds
- Calibration curves (model vs market)
- Per-fold metric stability plots
- Feature importance chart
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve


def print_aggregate_metrics(fold_results: list) -> None:
    """Print summary statistics across all walk-forward folds."""
    df = pd.DataFrame(fold_results)

    print(f"\n{'=' * 65}")
    print("  AGGREGATE METRICS (mean +/- std across folds)")
    print(f"{'=' * 65}")
    print(
        f"  Model Log Loss:   "
        f"{df['model_log_loss'].mean():.4f}  +/- {df['model_log_loss'].std():.4f}"
    )
    print(
        f"  Market Log Loss:  "
        f"{df['market_log_loss'].mean():.4f}  +/- {df['market_log_loss'].std():.4f}"
    )
    print(
        f"  Model Brier:      "
        f"{df['model_brier'].mean():.4f}  +/- {df['model_brier'].std():.4f}"
    )
    print(
        f"  Market Brier:     "
        f"{df['market_brier'].mean():.4f}  +/- {df['market_brier'].std():.4f}"
    )
    print(f"{'=' * 65}\n")


def plot_calibration(
    predictions: pd.DataFrame,
    output_dir: str = "outputs",
) -> None:
    """
    Plot calibration curves for model and market probabilities.

    A well-calibrated model's curve follows the diagonal:
    when the model says 20%, horses actually win ~20% of the time.
    """
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Model calibration ---
    prob_true, prob_pred = calibration_curve(
        predictions["win"],
        predictions["predicted_prob"],
        n_bins=10,
        strategy="quantile",
    )
    axes[0].plot(prob_pred, prob_true, "o-", color="#2196F3", label="Model")
    axes[0].plot([0, 1], [0, 1], "--", color="gray", label="Perfect")
    axes[0].set_xlabel("Predicted Probability")
    axes[0].set_ylabel("Observed Win Frequency")
    axes[0].set_title("Model Calibration")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # --- Market calibration ---
    prob_true_m, prob_pred_m = calibration_curve(
        predictions["win"],
        predictions["market_implied_prob"],
        n_bins=10,
        strategy="quantile",
    )
    axes[1].plot(prob_pred_m, prob_true_m, "o-", color="#FF5722", label="Market")
    axes[1].plot([0, 1], [0, 1], "--", color="gray", label="Perfect")
    axes[1].set_xlabel("Implied Probability")
    axes[1].set_ylabel("Observed Win Frequency")
    axes[1].set_title("Market Calibration")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "calibration_curve.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_fold_metrics(
    fold_results: list,
    output_dir: str = "outputs",
) -> None:
    """Plot per-fold log loss and Brier score to assess stability over time."""
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(fold_results)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = df["fold"]

    # --- Log Loss ---
    axes[0].plot(x, df["model_log_loss"], "o-", color="#2196F3", label="Model")
    axes[0].plot(x, df["market_log_loss"], "o-", color="#FF5722", label="Market")
    axes[0].set_xlabel("Fold")
    axes[0].set_ylabel("Log Loss")
    axes[0].set_title("Log Loss per Fold (lower = better)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # --- Brier Score ---
    axes[1].plot(x, df["model_brier"], "o-", color="#2196F3", label="Model")
    axes[1].plot(x, df["market_brier"], "o-", color="#FF5722", label="Market")
    axes[1].set_xlabel("Fold")
    axes[1].set_ylabel("Brier Score")
    axes[1].set_title("Brier Score per Fold (lower = better)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "fold_metrics.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_feature_importance(
    importance_df: pd.DataFrame,
    output_dir: str = "outputs",
) -> None:
    """Horizontal bar chart of LightGBM split-based feature importance."""
    os.makedirs(output_dir, exist_ok=True)
    imp = importance_df.sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(imp["feature"], imp["importance"], color="#2196F3")
    ax.set_xlabel("Importance (split count)")
    ax.set_title("LightGBM Feature Importance")
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    path = os.path.join(output_dir, "feature_importance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def save_predictions(
    predictions: pd.DataFrame,
    output_dir: str = "outputs",
) -> None:
    """Save per-horse predictions to CSV."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "predictions.csv")
    predictions.to_csv(path, index=False)
    print(f"  Saved: {path}  ({len(predictions):,} rows)")

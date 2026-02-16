"""
Horse Racing Win Probability Model - Proof of Concept
======================================================

End-to-end pipeline:
    1. Generate synthetic horse racing data (replace with real data loader)
    2. Engineer time-aware features (no data leakage)
    3. Walk-forward validation with LightGBM
    4. Evaluate: log loss, Brier score, calibration, market comparison

Usage:
    python main.py
"""

import os
import time

from data.generate_synthetic_data import generate_horse_racing_data
from features.engineering import engineer_features
from model.walk_forward import walk_forward_validate
from validation.metrics import (
    print_aggregate_metrics,
    plot_calibration,
    plot_fold_metrics,
    plot_feature_importance,
    save_predictions,
)


def main():
    os.makedirs("outputs", exist_ok=True)
    t0 = time.time()

    # ==================================================================
    # Step 1 — Data
    # ==================================================================
    print("=" * 65)
    print("  STEP 1: Generating synthetic horse racing data")
    print("=" * 65)
    df = generate_horse_racing_data(n_races=5000)
    df.to_csv("data/synthetic_races.csv", index=False)
    print(
        f"  {len(df):,} rows  |  {df['race_id'].nunique():,} races  |  "
        f"{df['horse_id'].nunique()} horses"
    )
    print(
        f"  Date range: {df['race_date'].min().date()} to "
        f"{df['race_date'].max().date()}"
    )

    # ==================================================================
    # Step 2 — Feature Engineering
    # ==================================================================
    print(f"\n{'=' * 65}")
    print("  STEP 2: Engineering time-aware features")
    print("=" * 65)
    df = engineer_features(df)
    print(f"  Feature matrix shape: {df.shape}")

    # ==================================================================
    # Step 3 — Walk-Forward Validation
    # ==================================================================
    print(f"\n{'=' * 65}")
    print("  STEP 3: Walk-forward validation with LightGBM")
    print("=" * 65)
    fold_results, predictions, importance = walk_forward_validate(df)

    # ==================================================================
    # Step 4 — Metrics & Output
    # ==================================================================
    print_aggregate_metrics(fold_results)

    print("  Generating plots...")
    plot_calibration(predictions)
    plot_fold_metrics(fold_results)
    plot_feature_importance(importance)
    save_predictions(predictions)

    # --- Sample output for one race ---
    print(f"\n{'=' * 65}")
    print("  SAMPLE OUTPUT (first race in validation set)")
    print("=" * 65)
    sample_race_id = predictions["race_id"].iloc[0]
    sample = predictions[predictions["race_id"] == sample_race_id].copy()
    sample = sample.sort_values("predicted_prob", ascending=False)
    cols = [
        "horse_id", "predicted_prob", "fair_odds",
        "market_implied_prob", "sp_odds", "finish_position", "win",
    ]
    print(sample[cols].to_string(index=False))

    # --- Feature importance ---
    print(f"\n{'=' * 65}")
    print("  FEATURE IMPORTANCE (top 10)")
    print("=" * 65)
    print(importance.head(10).to_string(index=False))

    elapsed = time.time() - t0
    print(f"\nPipeline completed in {elapsed:.1f}s")


if __name__ == "__main__":
    main()

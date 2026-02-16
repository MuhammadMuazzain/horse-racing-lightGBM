"""
Walk-Forward Validation
========================
Expanding-window time-series cross-validation for horse racing.

Design:
    Fold 1:  Train [2021-01 -> 2022-01]  |  Val [2022-01-08 -> 2022-04-08]
    Fold 2:  Train [2021-01 -> 2022-04]  |  Val [2022-04-15 -> 2022-07-15]
    ...
    Fold N:  Train [all prior]           |  Val [latest 3 months]

    - Training window EXPANDS (all historical data is kept)
    - 7-day gap between train and val prevents boundary bleed
    - Validation windows are non-overlapping
"""

import lightgbm as lgb
import pandas as pd
import numpy as np
from datetime import timedelta
from dateutil.relativedelta import relativedelta
from sklearn.metrics import log_loss, brier_score_loss

from config import (
    LGBM_PARAMS,
    FEATURE_COLUMNS,
    TARGET,
    EARLY_STOPPING_ROUNDS,
    WALK_FORWARD_MIN_TRAIN_MONTHS,
    WALK_FORWARD_FOLD_MONTHS,
    WALK_FORWARD_GAP_DAYS,
)


def generate_folds(df: pd.DataFrame) -> list:
    """
    Generate time-based expanding-window fold boundaries.

    Returns:
        List of (train_cutoff, val_start, val_end) date tuples.
    """
    min_date = df["race_date"].min()
    max_date = df["race_date"].max()

    train_cutoff = min_date + relativedelta(months=WALK_FORWARD_MIN_TRAIN_MONTHS)
    gap = timedelta(days=WALK_FORWARD_GAP_DAYS)
    folds = []

    while True:
        val_start = train_cutoff + gap
        val_end = val_start + relativedelta(months=WALK_FORWARD_FOLD_MONTHS)

        if val_end > max_date + timedelta(days=1):
            break

        folds.append((train_cutoff, val_start, val_end))
        train_cutoff = val_end

    return folds


def walk_forward_validate(df: pd.DataFrame):
    """
    Run full walk-forward validation with LightGBM.

    Returns:
        fold_results: List of per-fold metric dicts.
        predictions:  DataFrame of all out-of-sample predictions.
        importance:   DataFrame of feature importances (from last fold).
    """
    df = df.copy()
    df["race_date"] = pd.to_datetime(df["race_date"])

    folds = generate_folds(df)
    fold_results = []
    all_val_preds = []
    last_model = None

    print(f"\n{'=' * 65}")
    print(f"  WALK-FORWARD VALIDATION  |  {len(folds)} folds")
    print(f"{'=' * 65}\n")

    for fold_idx, (train_cutoff, val_start, val_end) in enumerate(folds):

        # --- Strict time-based split ---
        train = df[df["race_date"] < train_cutoff]
        val = df[(df["race_date"] >= val_start) & (df["race_date"] < val_end)]

        if len(val) == 0:
            continue

        X_train = train[FEATURE_COLUMNS]
        y_train = train[TARGET]
        X_val = val[FEATURE_COLUMNS]
        y_val = val[TARGET]

        # --- Train LightGBM ---
        model = lgb.LGBMClassifier(**LGBM_PARAMS)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        last_model = model

        # --- Predicted win probabilities ---
        y_prob = model.predict_proba(X_val)[:, 1]

        # --- Metrics: Model ---
        model_ll = log_loss(y_val, y_prob)
        model_bs = brier_score_loss(y_val, y_prob)

        # --- Metrics: Market benchmark ---
        market_probs = val["market_implied_prob"].clip(0.01, 0.99)
        market_ll = log_loss(y_val, market_probs)
        market_bs = brier_score_loss(y_val, market_probs)

        fold_result = {
            "fold": fold_idx + 1,
            "train_cutoff": train_cutoff.strftime("%Y-%m-%d"),
            "val_start": val_start.strftime("%Y-%m-%d"),
            "val_end": val_end.strftime("%Y-%m-%d"),
            "n_train": len(train),
            "n_val": len(val),
            "model_log_loss": round(model_ll, 4),
            "market_log_loss": round(market_ll, 4),
            "model_brier": round(model_bs, 4),
            "market_brier": round(market_bs, 4),
        }
        fold_results.append(fold_result)

        # --- Store predictions ---
        val_preds = val[
            ["race_id", "race_date", "horse_id", "finish_position",
             "win", "sp_odds", "market_implied_prob"]
        ].copy()
        val_preds["predicted_prob"] = y_prob
        val_preds["fair_odds"] = np.round(1.0 / np.maximum(y_prob, 0.001), 2)
        val_preds["fold"] = fold_idx + 1
        all_val_preds.append(val_preds)

        # --- Print fold summary ---
        print(
            f"  Fold {fold_idx + 1:>2d}  "
            f"Train < {train_cutoff.strftime('%Y-%m-%d')}  |  "
            f"Val [{val_start.strftime('%Y-%m-%d')} -> {val_end.strftime('%Y-%m-%d')}]  |  "
            f"n_train={len(train):>6,}  n_val={len(val):>5,}"
        )
        print(
            f"           "
            f"LogLoss  model={model_ll:.4f}  market={market_ll:.4f}  |  "
            f"Brier  model={model_bs:.4f}  market={market_bs:.4f}"
        )

    # --- Feature importance from last model ---
    importance = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "importance": last_model.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    predictions = pd.concat(all_val_preds, ignore_index=True)

    return fold_results, predictions, importance

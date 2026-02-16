"""
Configuration for Horse Racing Win Probability Model.
All hyperparameters and feature definitions in one place.
"""

# =============================================================================
# Walk-Forward Validation
# =============================================================================
WALK_FORWARD_MIN_TRAIN_MONTHS = 12   # minimum training data before first fold
WALK_FORWARD_FOLD_MONTHS = 3         # validation window size (months)
WALK_FORWARD_GAP_DAYS = 7            # gap between train end and val start

# =============================================================================
# Feature Engineering
# =============================================================================
ROLLING_WINDOW = 10                  # last N starts for horse rolling features
JOCKEY_TRAINER_WINDOW_DAYS = 90      # time window for jockey/trainer strike rates

# =============================================================================
# LightGBM Hyperparameters
# =============================================================================
LGBM_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "n_estimators": 300,
}

EARLY_STOPPING_ROUNDS = 30

# =============================================================================
# Feature Columns (used for training)
# =============================================================================
FEATURE_COLUMNS = [
    "horse_win_rate_last10",
    "horse_place_rate_last10",
    "horse_avg_finish_last10",
    "horse_starts_count",
    "horse_distance_win_rate",
    "horse_track_win_rate",
    "jockey_strike_rate_90d",
    "trainer_strike_rate_90d",
    "days_since_last_run",
    "weight_carried",
    "barrier",
    "field_size",
    "age",
    "race_class",
    "distance_f",
    "market_implied_prob",
]

TARGET = "win"

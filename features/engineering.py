"""
Time-Aware Feature Engineering
===============================
CRITICAL DESIGN PRINCIPLE:
    Every feature is computed using ONLY data available *before* each race date.
    No future information ever leaks into any feature value.

Implementation approach:
    - groupby(horse_id) + shift(1) ensures the current race is excluded
    - Rolling windows use only prior starts
    - Jockey/trainer rates use a strict date < cutoff filter
"""

import pandas as pd
import numpy as np

from config import ROLLING_WINDOW, JOCKEY_TRAINER_WINDOW_DAYS


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all features with strict time-awareness.

    Args:
        df: Raw race data sorted by date.

    Returns:
        DataFrame with feature columns appended.
    """
    df = df.copy()
    df["race_date"] = pd.to_datetime(df["race_date"])
    df = df.sort_values(["race_date", "race_id", "horse_id"]).reset_index(drop=True)

    n = ROLLING_WINDOW

    # ------------------------------------------------------------------
    # 1. Horse rolling features (last N starts)
    #    shift(1) guarantees current race is NOT included.
    # ------------------------------------------------------------------
    df["horse_win_rate_last10"] = (
        df.groupby("horse_id")["win"]
        .transform(lambda x: x.shift(1).rolling(n, min_periods=1).mean())
    )

    df["horse_place_rate_last10"] = (
        df.groupby("horse_id")["place"]
        .transform(lambda x: x.shift(1).rolling(n, min_periods=1).mean())
    )

    df["horse_avg_finish_last10"] = (
        df.groupby("horse_id")["finish_position"]
        .transform(lambda x: x.shift(1).rolling(n, min_periods=1).mean())
    )

    # Career starts before this race
    df["horse_starts_count"] = df.groupby("horse_id").cumcount()

    # ------------------------------------------------------------------
    # 2. Horse performance at this distance (expanding, shifted)
    # ------------------------------------------------------------------
    df["_dist_key"] = df["horse_id"].astype(str) + "_" + df["distance_f"].astype(str)
    df["horse_distance_win_rate"] = (
        df.groupby("_dist_key")["win"]
        .transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
    )

    # ------------------------------------------------------------------
    # 3. Horse performance at this track (expanding, shifted)
    # ------------------------------------------------------------------
    df["_track_key"] = df["horse_id"].astype(str) + "_" + df["track"]
    df["horse_track_win_rate"] = (
        df.groupby("_track_key")["win"]
        .transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
    )

    # ------------------------------------------------------------------
    # 4. Days since last run
    # ------------------------------------------------------------------
    df["_prev_date"] = df.groupby("horse_id")["race_date"].shift(1)
    df["days_since_last_run"] = (df["race_date"] - df["_prev_date"]).dt.days

    # ------------------------------------------------------------------
    # 5. Jockey & Trainer strike rates (rolling time window)
    #    Uses only races strictly BEFORE each race date.
    # ------------------------------------------------------------------
    print("    Computing jockey strike rates (time-windowed)...")
    df["jockey_strike_rate_90d"] = _time_windowed_strike_rate(
        df, group_col="jockey_id", target_col="win",
        window_days=JOCKEY_TRAINER_WINDOW_DAYS,
    )

    print("    Computing trainer strike rates (time-windowed)...")
    df["trainer_strike_rate_90d"] = _time_windowed_strike_rate(
        df, group_col="trainer_id", target_col="win",
        window_days=JOCKEY_TRAINER_WINDOW_DAYS,
    )

    # ------------------------------------------------------------------
    # 6. Market implied probability (normalised within each race)
    # ------------------------------------------------------------------
    df["_raw_implied"] = 1.0 / df["sp_odds"]
    df["market_implied_prob"] = (
        df["_raw_implied"]
        / df.groupby("race_id")["_raw_implied"].transform("sum")
    )

    # ------------------------------------------------------------------
    # Clean up temp columns
    # ------------------------------------------------------------------
    drop_cols = [c for c in df.columns if c.startswith("_")]
    df = df.drop(columns=drop_cols)

    # Fill NaN for first-time runners (no prior history)
    avg_field = df["field_size"].mean()
    df = df.fillna({
        "horse_win_rate_last10": 0.0,
        "horse_place_rate_last10": 0.0,
        "horse_avg_finish_last10": avg_field / 2,
        "horse_distance_win_rate": 0.0,
        "horse_track_win_rate": 0.0,
        "days_since_last_run": 90.0,
        "jockey_strike_rate_90d": 0.0,
        "trainer_strike_rate_90d": 0.0,
    })

    return df


def _time_windowed_strike_rate(
    df: pd.DataFrame,
    group_col: str,
    target_col: str,
    window_days: int,
) -> pd.Series:
    """
    Compute strike rate for a group (jockey/trainer) over a rolling time window.

    For each race date, looks back `window_days` and computes the group's
    win rate using ONLY races in that historical window.

    NOTE: This iterates over unique race dates. Acceptable for a PoC dataset
    (~1,000 dates). For production-scale data, use merge_asof or vectorised
    cumulative logic.
    """
    result = pd.Series(np.nan, index=df.index, dtype=float)
    dates = sorted(df["race_date"].unique())

    for date in dates:
        mask = df["race_date"] == date
        window_start = date - pd.Timedelta(days=window_days)

        hist = df[(df["race_date"] < date) & (df["race_date"] >= window_start)]

        if len(hist) == 0:
            result.loc[mask] = 0.0
            continue

        group_rates = hist.groupby(group_col)[target_col].mean()
        current_groups = df.loc[mask, group_col]
        result.loc[mask] = current_groups.map(group_rates).fillna(0.0).values

    return result

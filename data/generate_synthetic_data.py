"""
Synthetic Horse Racing Data Generator
======================================
Generates a realistic synthetic dataset for model development and testing.

In production, this module is replaced by a data loader for real historical data.
The synthetic data preserves the schema and statistical properties needed to
validate the full pipeline (feature engineering, walk-forward split, etc.).
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_horse_racing_data(n_races: int = 5000, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic horse racing data with realistic structure.

    Each row represents one horse in one race, with columns for race conditions,
    runner details, finishing position, win/place flags, and starting price odds.

    Args:
        n_races: Number of races to generate (~6-14 runners each).
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with ~50,000 rows (n_races x avg field size).
    """
    np.random.seed(seed)

    # --- Entity pools ---
    n_horses = 800
    n_jockeys = 60
    n_trainers = 80

    tracks = [
        "Ascot", "Cheltenham", "Newmarket", "York",
        "Goodwood", "Doncaster", "Epsom", "Sandown",
    ]
    goings = ["Firm", "Good to Firm", "Good", "Good to Soft", "Soft", "Heavy"]
    distances_f = [5, 6, 7, 8, 10, 12, 14, 16, 20]

    # --- Latent ability scores (not observable — drive outcomes) ---
    horse_ability = np.random.normal(0, 1.0, n_horses)
    jockey_skill = np.random.normal(0, 0.5, n_jockeys)
    trainer_skill = np.random.normal(0, 0.3, n_trainers)

    # --- Horse characteristics (fixed per horse) ---
    horse_pref_distance = np.random.choice(distances_f, n_horses)
    horse_pref_going_idx = np.random.randint(0, len(goings), n_horses)
    horse_trainer = np.random.randint(0, n_trainers, n_horses)
    horse_pref_jockey = np.random.randint(0, n_jockeys, n_horses)
    horse_base_age = np.random.randint(2, 7, n_horses)

    # --- Generate races over 3 years ---
    start_date = datetime(2021, 1, 1)
    days_span = 365 * 3
    records = []

    for race_idx in range(n_races):
        race_date = start_date + timedelta(
            days=int(race_idx * days_span / n_races)
        )
        track = np.random.choice(tracks)
        going_idx = np.random.randint(0, len(goings))
        going = goings[going_idx]
        distance = int(np.random.choice(distances_f))
        race_class = int(np.random.randint(1, 7))
        field_size = int(np.random.randint(6, 15))

        runners = np.random.choice(n_horses, field_size, replace=False)

        scores = []
        runner_records = []

        for pos_in_field, h_idx in enumerate(runners):
            trainer_id = int(horse_trainer[h_idx])

            if np.random.random() < 0.70:
                jockey_id = int(horse_pref_jockey[h_idx])
            else:
                jockey_id = int(np.random.randint(0, n_jockeys))

            dist_penalty = -0.1 * abs(distance - horse_pref_distance[h_idx])
            going_penalty = -0.15 * abs(going_idx - horse_pref_going_idx[h_idx])

            years_into_sim = (race_date - start_date).days / 365.25
            age = int(horse_base_age[h_idx] + years_into_sim)

            weight = round(np.random.uniform(8.0, 10.0), 1)
            barrier = int(pos_in_field + 1)

            noise = np.random.normal(0, 1.5)
            score = (
                horse_ability[h_idx]
                + jockey_skill[jockey_id]
                + trainer_skill[trainer_id]
                + dist_penalty
                + going_penalty
                + noise
            )

            scores.append(score)
            runner_records.append({
                "race_id": race_idx + 1,
                "race_date": race_date.strftime("%Y-%m-%d"),
                "track": track,
                "distance_f": distance,
                "going": going,
                "race_class": race_class,
                "field_size": field_size,
                "horse_id": int(h_idx + 1),
                "jockey_id": jockey_id + 1,
                "trainer_id": trainer_id + 1,
                "age": age,
                "weight_carried": weight,
                "barrier": barrier,
            })

        # Rank by latent score to determine finishing order
        order = np.argsort(scores)[::-1]
        for rank, idx in enumerate(order):
            runner_records[idx]["finish_position"] = rank + 1
            runner_records[idx]["win"] = int(rank == 0)
            runner_records[idx]["place"] = int(rank < 3)

        # Generate starting-price odds (correlated with latent score)
        scores_arr = np.array(scores)
        raw_probs = np.exp(scores_arr - scores_arr.max())
        raw_probs = raw_probs / raw_probs.sum()
        overround = 1.15
        sp_odds = np.round(1.0 / (raw_probs * overround), 2)
        sp_odds = np.maximum(sp_odds, 1.10)

        for i, rec in enumerate(runner_records):
            rec["sp_odds"] = float(sp_odds[i])

        records.extend(runner_records)

    df = pd.DataFrame(records)
    df["race_date"] = pd.to_datetime(df["race_date"])
    return df


if __name__ == "__main__":
    df = generate_horse_racing_data()
    print(f"Generated {len(df):,} rows  |  {df['race_id'].nunique():,} races  |  "
          f"{df['horse_id'].nunique()} horses")
    print(f"Date range: {df['race_date'].min().date()} to {df['race_date'].max().date()}")
    print(f"\nWin rate: {df['win'].mean():.3f}  (expected ~1/field_size)")
    print(f"\nSample:\n{df.head(10).to_string(index=False)}")

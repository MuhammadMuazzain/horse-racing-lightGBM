# Horse Racing Win Probability Model (V1 — Proof of Concept)

A structured, production-quality pipeline for predicting horse racing win probabilities using **LightGBM** with strict **time-based validation** and **market comparison**.

## Key Design Principles

| Principle | Implementation |
|-----------|---------------|
| **No data leakage** | All features computed using only data available *before* each race |
| **Walk-forward validation** | Expanding-window time-series CV — no random splits |
| **Calibrated probabilities** | Output is a true P(win), not just a ranking |
| **Market benchmark** | Model predictions compared against SP implied probabilities |

## Project Structure

```
horse-racing-lightGBM/
├── config.py                       # All hyperparameters and feature definitions
├── main.py                         # End-to-end pipeline entrypoint
├── requirements.txt
│
├── data/
│   └── generate_synthetic_data.py  # Synthetic data (replace with real loader)
│
├── features/
│   └── engineering.py              # Time-aware feature engineering
│
├── model/
│   └── walk_forward.py             # Walk-forward validation + LightGBM training
│
├── validation/
│   └── metrics.py                  # Evaluation metrics + visualisation
│
└── outputs/                        # Generated at runtime
    ├── predictions.csv             # Per-horse win probabilities + fair odds
    ├── calibration_curve.png
    ├── fold_metrics.png
    └── feature_importance.png
```

## Methodology

### Feature Engineering (`features/engineering.py`)

Every feature is computed with a strict **time guard** — only historical data before each race date is used.

| Feature | Description | Leakage Prevention |
|---------|-------------|--------------------|
| `horse_win_rate_last10` | Win rate in last 10 starts | `groupby(horse).shift(1).rolling(10)` |
| `horse_place_rate_last10` | Place rate (top 3) in last 10 starts | Same shift+rolling pattern |
| `horse_avg_finish_last10` | Average finishing position | Same shift+rolling pattern |
| `horse_distance_win_rate` | Win rate at today's distance | `shift(1).expanding()` |
| `horse_track_win_rate` | Win rate at today's track | `shift(1).expanding()` |
| `jockey_strike_rate_90d` | Jockey win rate last 90 days | Strict `date < cutoff` filter |
| `trainer_strike_rate_90d` | Trainer win rate last 90 days | Strict `date < cutoff` filter |
| `days_since_last_run` | Days since horse's previous race | `groupby(horse).shift(1)` |
| `market_implied_prob` | Normalised 1/SP within race | Derived from SP odds |

### Walk-Forward Validation (`model/walk_forward.py`)

```
Fold 1:  Train [2021-01 → 2022-01]  |  Val [2022-01-08 → 2022-04-08]
Fold 2:  Train [2021-01 → 2022-04]  |  Val [2022-04-15 → 2022-07-15]
Fold 3:  Train [2021-01 → 2022-07]  |  Val [2022-07-22 → 2022-10-22]
...
```

- **Expanding window**: training data grows with each fold (all history is kept)
- **7-day gap**: between training cutoff and validation start
- **Non-overlapping** validation windows (3 months each)
- **Early stopping** on validation fold to prevent overfitting

### Evaluation Metrics

- **Log Loss** — measures probability quality (penalises confident wrong predictions)
- **Brier Score** — mean squared error of probability predictions
- **Calibration Curve** — does "20% predicted" mean "20% actual wins"?
- **Model vs Market** — are we more accurate than the betting market?

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python main.py
```

This will:
1. Generate synthetic race data (~50k rows, 5000 races)
2. Engineer time-aware features
3. Run walk-forward validation (7 folds)
4. Print metrics and generate plots in `outputs/`

## Output Format

Each row in `outputs/predictions.csv`:

| Column | Description |
|--------|-------------|
| `race_id` | Unique race identifier |
| `horse_id` | Unique horse identifier |
| `predicted_prob` | Model's estimated P(win) |
| `fair_odds` | 1 / predicted_prob |
| `market_implied_prob` | Market-derived P(win) from SP |
| `sp_odds` | Starting price decimal odds |
| `finish_position` | Actual finishing position |
| `win` | 1 if won, 0 otherwise |

## Adapting for Real Data

To swap in real historical data:

1. Replace `generate_horse_racing_data()` in `main.py` with a data loader
2. Ensure the DataFrame has these columns: `race_id`, `race_date`, `track`, `distance_f`, `going`, `race_class`, `field_size`, `horse_id`, `jockey_id`, `trainer_id`, `age`, `weight_carried`, `barrier`, `finish_position`, `win`, `place`, `sp_odds`
3. Everything else (features, validation, metrics) works unchanged

## Dependencies

- LightGBM
- pandas / numpy
- scikit-learn
- matplotlib

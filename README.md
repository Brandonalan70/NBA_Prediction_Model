# NBA In-Game Win Probability Model

A machine learning system that predicts NBA win probability in real time using play-by-play data. The model is a stacking ensemble (Logistic Regression + Random Forest + XGBoost) trained on four seasons of NBA play-by-play events and achieves **79.7% accuracy** and **0.879 ROC-AUC** on held-out games.

An interactive Streamlit dashboard lets you replay any game play-by-play, visualize the win probability curve, and run what-if lineup substitution scenarios.

---

## Features

- **Stacking ensemble model** combining Logistic Regression, Random Forest, and XGBoost base learners
- **19 features** including score margin, time remaining, per-game team stats, ELO differential, and engineered differentials
- **Game-level train/test split** to prevent data leakage across 1,750 games (4 seasons)
- **Well-calibrated probabilities** (ECE = 0.023) — suitable for use as actual probabilities, not just rankings
- **Interactive dashboard** with live replay and lineup what-if analysis
- **HPC training script** (SLURM) for retraining on a cluster

---

## Model Performance

| Metric | Stacking Ensemble | Logistic Baseline |
|---|---|---|
| Accuracy | **79.7%** | 79.6% |
| ROC-AUC | **0.879** | 0.877 |
| Brier Score | **0.141** | 0.141 |
| ECE | **0.023** | — |
| Brier Skill Score | **0.425** | — |

Performance by quarter (Brier score, lower is better):

| Q1 | Q2 | Q3 | Q4 | Clutch (last 5 min, ≤5 pts) |
|---|---|---|---|---|
| 0.185 | 0.157 | 0.132 | 0.093 | 0.083 |

---

## Project Structure

```
NBA_Prediction_Model/
├── dashboard/
│   └── app.py                  # Streamlit web dashboard
├── src/
│   ├── data/
│   │   ├── scrape.py           # Fetch PBP data from the NBA API
│   │   └── preprocess.py       # Clean and engineer features from raw PBP CSVs
│   └── models/
│       ├── train.py            # Train the stacking ensemble (main script)
│       ├── train_logistic.py   # Logistic regression baseline
│       ├── evaluate.py         # ROC curve comparison across models
│       ├── predict.py          # Terminal live-game streamer with win probability
│       └── test.py             # Quick model validation
├── notebooks/
│   └── data_exploration.ipynb  # Data quality checks and EDA
├── scripts/
│   └── train_hpc.slurm         # SLURM job script for PSC Bridges-2
├── results/
│   ├── stacking_model_results.json
│   ├── calibration_plot.png
│   ├── roc_curve.png
│   ├── model_comparison.png
│   ├── performance_by_quarter.png
│   ├── probability_distribution.png
│   └── archive/                # Previous model run outputs
├── data/
│   ├── raw/                    # Full season PBP CSVs + ELO ratings (local only)
│   │   ├── nba_elo.csv
│   │   ├── pbp_2022_23_regularseason/
│   │   ├── pbp_2023_24_regularseason/
│   │   ├── pbp_2024_25_regularseason/
│   │   └── pbp_2025_26_regularseason/
│   ├── processed/
│   │   └── nba_pbp.csv         # Preprocessed feature matrix
│   └── sample/                 # Small sample files for development/testing
│       ├── nba_elo_sample.csv  # ELO data from 2020–present (~14k rows)
│       └── pbp_sample/         # 5 games per season across all 4 seasons
└── NBA_Model_Report.pdf        # Final report
```

---

## Installation

```bash
git clone (https://github.com/Brandonalan70/NBA_Prediction_Model)
cd NBA_Prediction_Model
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```


---

## Usage

### Run the dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard has two tabs:

- **Live Replay** — select a game and step through plays while watching the win probability update in real time
- **What-If** — freeze a moment in the game, swap a player in or out, and see how the lineup change shifts win probability

> Before running, update the `MODEL_PATH`, `SCALER_PATH`, and `DATA_DIR` constants near the top of `dashboard/app.py` to point to your local paths.

### Train the model

```bash
python src/models/train.py
```

Trains the stacking ensemble on all seasons in `data/raw/`, saves the model and scaler to `results/`, and writes performance plots and a JSON metrics file.

To retrain on a SLURM cluster:

```bash
sbatch scripts/train_hpc.slurm
```

### Scrape new season data

```bash
python src/data/scrape.py
```

Fetches play-by-play CSVs from the NBA API for a configured season and saves them to `data/raw/`. Edit the `SEASON` and `SEASON_TYPE` constants at the top of the script before running.

### Preprocess raw data

```bash
python src/data/preprocess.py
```

Joins all raw PBP CSVs with ELO ratings, builds the feature matrix, and writes `data/processed/nba_pbp.csv`.

### Stream a game in the terminal

```bash
python src/models/predict.py
```

Presents a list of available game IDs, then streams play-by-play events with live win probability estimates printed to stdout.

---

## Data

| File | Description | Size |
|---|---|---|
| `data/raw/nba_elo.csv` | Historical NBA ELO ratings (FiveThirtyEight) | ~15 MB |
| `data/raw/pbp_*_regularseason/` | Individual game PBP CSVs, one file per game | ~48 MB / season |
| `data/processed/nba_pbp.csv` | Merged + feature-engineered dataset | ~4 KB |
| `data/sample/nba_elo_sample.csv` | ELO data from 2020 onward (for dev/testing) | ~700 KB |
| `data/sample/pbp_sample/` | 5 games per season (for dev/testing) | ~3.7 MB |

Full raw data covers the **2022–23 through 2025–26 regular seasons** (~4,900 individual game files).

---

## Model Artifacts

Trained artifacts live in `results/`:

| File | Description |
|---|---|
| `nba_stacking_scaler.pkl` | Fitted `StandardScaler` for the 19 input features |
| `stacking_model_results.json` | Full metrics, feature names, baseline comparisons |

The serialized model itself (`nba_stacking_model4.pkl`) is excluded from git due to size. Retrain locally with `src/models/train.py` or request the file directly.

---

## Requirements

See `requirements.txt`. Key dependencies:

- `scikit-learn >= 1.2`
- `xgboost >= 1.7`
- `streamlit >= 1.20`
- `nba_api >= 1.3`

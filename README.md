# Wildfire Hit Prediction

Wildfire spread prediction pipeline built for the **WiDS Datathon 2026**. Given satellite-derived fire perimeter features at a snapshot in time, the models predict whether an active fire will reach a target structure within 12, 24, 48, or 72 hours.

---

## Problem

The dataset contains one row per fire-event snapshot with:
- **`event`** — binary label: did the fire hit the structure? (31% positive rate)
- **`time_to_hit_hours`** — time until impact (censored at observation end for `event=0`)
- Dozens of engineered features covering perimeter counts, area growth, spread speed, bearing, and temporal resolution flags.

Four binary classification targets are derived from these two columns:

| Target | Meaning |
|--------|---------|
| `y_12` | Fire hits within 12 hours |
| `y_24` | Fire hits within 24 hours |
| `y_48` | Fire hits within 48 hours |
| `y_72` | Fire hits within 72 hours |



---

## Pipeline

### 1. Data Cleaning (`src/cleaning.py`)

Run once to inspect and clean the raw data:
- Reports missing values, duplicate rows, class balance, and outliers (3×IQR method)
- Checks skewness of all numeric features
- Saves `data/train_clean.csv` and `data/test_clean.csv`

### 2. Train/Val Split (`src/split.py`)

Run once after cleaning:
- Stratified 80/20 split on `event` to preserve the ~31% positive rate
- Saves `data/train_split.csv` and `data/val_split.csv`

### 3. Main Pipeline (`main.py`)

Runs EDA, trains both models, evaluates, and saves all figures:

```
python main.py
```

**Steps inside `main.py`:**
1. Load `train_split.csv`, `val_split.csv`, `test_clean.csv`
2. Create horizon labels (`y_12`, `y_24`, `y_48`, `y_72`)
3. Run EDA — feature summary table, histograms, correlation heatmaps, boxplots
4. Train and evaluate **Logistic Regression** on all four horizons
5. Train and evaluate **Random Forest** on all four horizons
6. Generate comparison plots: AUC by horizon, ROC curves (24h), RF feature importances

---

## Features Used

| Feature | Type | Description |
|---------|------|-------------|
| `num_perimeters_0_5h` | numeric | Number of perimeter records in first 0–5 hours |
| `low_temporal_resolution_0_5h` | binary flag | 1 if temporal resolution is low in the early window |
| `log1p_area_first` | numeric | Log-transformed initial fire area (hectares) |
| `log_area_ratio_0_5h` | numeric | Log ratio of area growth over 0–5 hour window |
| `centroid_speed_m_per_h` | numeric | Speed of fire centroid movement (m/h) |
| `spread_bearing_sin` | numeric | Sine component of fire spread bearing |
| `spread_bearing_cos` | numeric | Cosine component of fire spread bearing |
| `dist_min_ci_0_5h` | numeric | Min distance to closest ignition in 0–5h window |
| `closing_speed_m_per_h` | numeric | Rate at which fire is closing toward structure (m/h) |
| `alignment_cos` | numeric | Cosine of angle between spread direction and structure direction |
| `event_start_hour` | numeric | Hour of day the fire event started |
| `event_start_month` | numeric | Month the fire event started |

---

## Models

### Logistic Regression
- Preprocessing: median imputation → StandardScaler
- `class_weight="balanced"` to handle class imbalance
- One model trained per horizon (4 total)

### Random Forest
- Preprocessing: median imputation only (tree-based, scale-invariant)
- 300 estimators, uncapped depth, `min_samples_leaf=5`, `class_weight="balanced"`
- One model trained per horizon (4 total)

Both models are evaluated with **ROC-AUC** and a full classification report at threshold 0.5.

---

## Requirements

```
pandas
numpy
scikit-learn
matplotlib
```

Install with:

```bash
pip install pandas numpy scikit-learn matplotlib
```

---

## Usage

```bash
# Step 1 — clean raw data (run once)
python src/cleaning.py

# Step 2 — create train/val split (run once)
python src/split.py

# Step 3 — run full pipeline
python main.py
```

All figures are saved to `outputs/figures/` and the feature summary to `outputs/feature_summary.csv`.

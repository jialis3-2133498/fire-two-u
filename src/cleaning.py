import pandas as pd
import numpy as np


train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# ──────────────── MISSING VALUES ────────────────
print("=== Missing values (train) ===")
miss = train.isnull().sum()
print(miss[miss > 0] if miss.any() else "No missing values")

print("\n=== Missing values (test) ===")
miss_t = test.isnull().sum()
print(miss_t[miss_t > 0] if miss_t.any() else "No missing values")

# ──────────────── DUPLICATE ROWS ────────────────
print(
    f"\nDuplicate rows — train: \
        {train.duplicated().sum()}, test: {test.duplicated().sum()}")

# ──────────────── TARGET VARIABLE OVERVIEW ────────────────
# Two targets: 'event' (binary) and 'time_to_hit_hours' (continuous)
print("\n=== Target: event (class balance) ===")
print(train['event'].value_counts())
print(f"Positive rate: {train['event'].mean():.2%}")

print("\n=== Target: time_to_hit_hours ===")
print(train['time_to_hit_hours'].describe())
# Note: for events that never hit (event=0),
# time_to_hit_hours is the censoring time
print(
    f"\ntime_to_hit_hours for event=1:\n\
        {train[train['event'] == 1]['time_to_hit_hours'].describe()}")
print(
    f"\ntime_to_hit_hours for event=0:\n\
        {train[train['event'] == 0]['time_to_hit_hours'].describe()}")

# ──────────────── FEATURE COLUMNS ────────────────
# Separate IDs, targets, and features
drop_cols = ['event_id', 'event', 'time_to_hit_hours']
feat_cols = [c for c in train.columns if c not in drop_cols]

X_train = train[feat_cols].copy()
X_test = test[[c for c in feat_cols if c in test.columns]].copy()

# Check for columns in train but not test (should only be targets)
only_in_train = set(train.columns) - set(test.columns)
print(f"\nColumns only in train (expected: targets): {only_in_train}")

# ──────────────── OUTLIER DETECTION ────────────────
print("\n=== Outlier check (IQR method) ===")
numeric_cols = X_train.select_dtypes(include=np.number).columns
outlier_summary = {}
for col in numeric_cols:
    q1, q3 = X_train[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    n_out = ((X_train[col] < q1 - 3*iqr) | (X_train[col] > q3 + 3*iqr)).sum()
    if n_out > 0:
        outlier_summary[col] = n_out
print(pd.Series(outlier_summary).sort_values(ascending=False))

# ──────────────── SKEWNESS CHECK ────────────────
print("\n=== Highly skewed features (|skew| > 1) ===")
skew = X_train[numeric_cols].skew().abs().sort_values(ascending=False)
print(skew[skew > 1])

# ──────────────── FLAG COLUMN CHECK ────────────────
# low_temporal_resolution_0_5h is a binary flag — confirm it
print("\n=== low_temporal_resolution_0_5h value counts ===")
print(train['low_temporal_resolution_0_5h'].value_counts())

# ──────────────── CORRELATION WITH TARGET ────────────────
print("\n=== Top correlations with 'event' ===")
corr = train[feat_cols + ['event']].corr()['event'].drop('event').abs()
print(corr.sort_values(ascending=False).head(10))

# ──────────────── SAVE CLEANED SPLIT ────────────────
# No imputation needed if no missing values; otherwise add here.
# Keep event_id for submission matching.
train_clean = train.copy()
test_clean = test.copy()

train_clean.to_csv("data/train_clean.csv", index=False)
test_clean.to_csv("data/test_clean.csv", index=False)
print("\nSaved train_clean.csv and test_clean.csv")
import pandas as pd
from sklearn.model_selection import train_test_split

train = pd.read_csv("data/train_clean.csv")

# ── DEFINE FEATURES AND TARGETS ───────────────────────────────────────────────
drop_cols = ['event_id', 'event', 'time_to_hit_hours']
feat_cols = [c for c in train.columns if c not in drop_cols]

X = train[feat_cols]
y_event = train['event']                  # classification target
y_time  = train['time_to_hit_hours']      # survival time target

# ── STRATIFIED SPLIT ──────────────────────────────────────────────────────────
# Stratify on 'event' to preserve the 31% positive rate in both splits
X_train, X_val, y_event_train, y_event_val, y_time_train, y_time_val = train_test_split(
    X, y_event, y_time,
    test_size=0.2,   # <-- split by 80% and 20%g
    random_state=42,
    shuffle=True,          # <-- explicit random shuffle before splitting
    stratify=y_event       # <-- preserves class balance after shuffle
)

# ── VERIFY BALANCE ────────────────────────────────────────────────────────────
print(f"Train size: {len(X_train)} | Val size: {len(X_val)}")
print(f"\nEvent rate — full: {y_event.mean():.2%}")
print(f"Event rate — train split: {y_event_train.mean():.2%}")
print(f"Event rate — val split:   {y_event_val.mean():.2%}")

# ── SAVE ──────────────────────────────────────────────────────────────────────
train_split = X_train.copy()
train_split['event']            = y_event_train
train_split['time_to_hit_hours'] = y_time_train

val_split = X_val.copy()
val_split['event']              = y_event_val
val_split['time_to_hit_hours']  = y_time_val

train_split.to_csv("data/train_split.csv", index=False)
val_split.to_csv("data/val_split.csv", index=False)
print("\nSaved train_split.csv and val_split.csv")
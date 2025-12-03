import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pickle

from clusters import assign_cluster_id

CLUSTER_FEATURE_COLUMNS = [
    "learning_days_ratio",
    "daily_reviews",
    "reviews_per_learning_day",
    "accuracy",
]

# Load data
print("📂 Loading training data...")
df = pd.read_csv('learning_sessions_data.csv')
print(f"✅ Loaded {len(df)} training samples\n")

# Cluster feature preparation
missing_cluster_cols = [col for col in CLUSTER_FEATURE_COLUMNS if col not in df.columns]
if missing_cluster_cols:
    print("ℹ️  Source CSV misses detailed Anki metrics – creating proxy columns for clustering.")
    df = df.copy()
    if 'learning_days_ratio' not in df.columns:
        df['learning_days_ratio'] = 1 / (1 + df['days_since_last_session'].fillna(0))
    if 'daily_reviews' not in df.columns:
        df['daily_reviews'] = df['total_session_duration'].fillna(0) / 3.0
    if 'reviews_per_learning_day' not in df.columns:
        df['reviews_per_learning_day'] = df['daily_reviews'] * df['learning_days_ratio']
    if 'accuracy' not in df.columns:
        baseline = df.get('concentration_score', df['concentration_baseline'])
        df['accuracy'] = (baseline / 10.0).clip(0.5, 0.98)

def _row_to_cluster_id(row):
    feature_dict = {col: row[col] for col in CLUSTER_FEATURE_COLUMNS}
    return assign_cluster_id(feature_dict)

print("📊 Assigning cluster ids to training data...")
df['cluster_id'] = df.apply(_row_to_cluster_id, axis=1)

# FEATURE ENGINEERING
# One-hot encoding for time_of_day
df_encoded = pd.get_dummies(df, columns=['time_of_day'], prefix='time')

# Select features (X)
feature_columns = [
    'total_session_duration',
    'time_morning', 'time_afternoon', 'time_evening', 'time_night',
    'concentration_baseline',
    'days_since_last_session',
    'previous_session_rating',
    'cluster_id'
]

X = df_encoded[feature_columns]

# Select targets (y)
y_work_blocks = df['optimal_work_blocks']
y_work_duration = df['work_block_duration']
y_break_duration = df['break_duration']
y_next_session = df['next_session_recommendation_hours']

# Train-test split (80% train, 20% test)
X_train, X_test, y_wb_train, y_wb_test = train_test_split(
    X, y_work_blocks, test_size=0.2, random_state=42
)
_, _, y_wd_train, y_wd_test = train_test_split(
    X, y_work_duration, test_size=0.2, random_state=42
)
_, _, y_bd_train, y_bd_test = train_test_split(
    X, y_break_duration, test_size=0.2, random_state=42
)
_, _, y_ns_train, y_ns_test = train_test_split(
    X, y_next_session, test_size=0.2, random_state=42
)

# Feature scaling (important for Ridge Regression)
print("🔧 Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# TRAIN MODELS
print("\n🤖 Training Ridge Regression models...\n")

# Model 1: optimal number of work blocks
print("1️⃣ Model for work blocks...")
model_work_blocks = Ridge(alpha=1.0)  # alpha = regularization strength
model_work_blocks.fit(X_train_scaled, y_wb_train)
y_wb_pred = model_work_blocks.predict(X_test_scaled)
print(f"   R² Score: {r2_score(y_wb_test, y_wb_pred):.3f}")
print(f"   RMSE: {np.sqrt(mean_squared_error(y_wb_test, y_wb_pred)):.3f}")

# Model 2: work-block duration
print("\n2️⃣ Model for work-block duration...")
model_work_duration = Ridge(alpha=1.0)
model_work_duration.fit(X_train_scaled, y_wd_train)
y_wd_pred = model_work_duration.predict(X_test_scaled)
print(f"   R² Score: {r2_score(y_wd_test, y_wd_pred):.3f}")
print(f"   RMSE: {np.sqrt(mean_squared_error(y_wd_test, y_wd_pred)):.3f}")

# Model 3: break duration
print("\n3️⃣ Model for break duration...")
model_break_duration = Ridge(alpha=1.0)
model_break_duration.fit(X_train_scaled, y_bd_train)
y_bd_pred = model_break_duration.predict(X_test_scaled)
print(f"   R² Score: {r2_score(y_bd_test, y_bd_pred):.3f}")
print(f"   RMSE: {np.sqrt(mean_squared_error(y_bd_test, y_bd_pred)):.3f}")

# Model 4: next session recommendation
print("\n4️⃣ Model for next session...")
model_next_session = Ridge(alpha=1.0)
model_next_session.fit(X_train_scaled, y_ns_train)
y_ns_pred = model_next_session.predict(X_test_scaled)
print(f"   R² Score: {r2_score(y_ns_test, y_ns_pred):.3f}")
print(f"   RMSE: {np.sqrt(mean_squared_error(y_ns_test, y_ns_pred)):.3f}")

# SAVE MODELS
print("\n💾 Saving models and scaler...")

models = {
    'scaler': scaler,
    'work_blocks': model_work_blocks,
    'work_duration': model_work_duration,
    'break_duration': model_break_duration,
    'next_session': model_next_session,
    'feature_columns': feature_columns
}

with open('learning_models.pkl', 'wb') as f:
    pickle.dump(models, f)

print("✅ Saved all models to 'learning_models.pkl'")

# EXAMPLE PREDICTION
print("\n" + "="*60)
print("📊 EXAMPLE PREDICTION")
print("="*60)

# Example: 120-minute session, morning, high concentration
example = pd.DataFrame([{
    'total_session_duration': 120,
    'time_morning': 1,
    'time_afternoon': 0,
    'time_evening': 0,
    'time_night': 0,
    'concentration_baseline': 8.0,
    'days_since_last_session': 1,
    'previous_session_rating': 7.5,
    'cluster_id': assign_cluster_id({
        'learning_days_ratio': 0.55,
        'daily_reviews': 40,
        'reviews_per_learning_day': 70,
        'accuracy': 0.82,
    })
}])

example_scaled = scaler.transform(example)

pred_blocks = model_work_blocks.predict(example_scaled)[0]
pred_work = model_work_duration.predict(example_scaled)[0]
pred_break = model_break_duration.predict(example_scaled)[0]
pred_next = model_next_session.predict(example_scaled)[0]

print(f"\n📥 INPUT:")
print(f"   Planned session: 120 minutes")
print(f"   Time of day: morning")
print(f"   Concentration: 8.0/10")
print(f"   Days since last session: 1")

print(f"\n📤 PREDICTION:")
print(f"   Recommended number of work blocks: {int(round(pred_blocks))}")
print(f"   Length per work block: {int(round(pred_work))} minutes")
print(f"   Length per break: {int(round(pred_break))} minutes")
print(f"   Next session in: {pred_next:.1f} hours")

print("\n" + "="*60)
print("✅ Training complete!")

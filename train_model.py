import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pickle

# Load data
print("📂 Loading training data...")
df = pd.read_csv('learning_sessions_data.csv')
print(f"✅ Loaded {len(df)} training samples\n")

# FEATURE ENGINEERING
# One-hot encoding for time_of_day
df_encoded = pd.get_dummies(df, columns=['time_of_day'], prefix='time')

# Select features (X)
feature_columns = [
    'total_session_duration',
    'time_morning', 'time_afternoon', 'time_evening', 'time_night',
    'concentration_baseline',
    'days_since_last_session',
    'previous_session_rating'
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
    'previous_session_rating': 7.5
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

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pickle

# Daten laden
print("📂 Lade Trainingsdaten...")
df = pd.read_csv('learning_sessions_data.csv')
print(f"✅ {len(df)} Trainingsbeispiele geladen\n")

# FEATURE ENGINEERING
# One-Hot Encoding für time_of_day
df_encoded = pd.get_dummies(df, columns=['time_of_day'], prefix='time')

# Features auswählen (X)
feature_columns = [
    'total_session_duration',
    'time_morning', 'time_afternoon', 'time_evening', 'time_night',
    'concentration_baseline',
    'days_since_last_session',
    'previous_session_rating'
]

X = df_encoded[feature_columns]

# Targets auswählen (y)
y_work_blocks = df['optimal_work_blocks']
y_work_duration = df['work_block_duration']
y_break_duration = df['break_duration']
y_next_session = df['next_session_recommendation_hours']

# Train-Test Split (80% Training, 20% Test)
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

# Feature Scaling (wichtig für Ridge Regression!)
print("🔧 Skaliere Features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# MODELLE TRAINIEREN
print("\n🤖 Trainiere Ridge Regression Modelle...\n")

# Modell 1: Optimale Anzahl Arbeitsblöcke
print("1️⃣ Modell für Arbeitsblöcke...")
model_work_blocks = Ridge(alpha=1.0)  # alpha = Regularisierungsstärke
model_work_blocks.fit(X_train_scaled, y_wb_train)
y_wb_pred = model_work_blocks.predict(X_test_scaled)
print(f"   R² Score: {r2_score(y_wb_test, y_wb_pred):.3f}")
print(f"   RMSE: {np.sqrt(mean_squared_error(y_wb_test, y_wb_pred)):.3f}")

# Modell 2: Arbeitsblock-Dauer
print("\n2️⃣ Modell für Arbeitsblock-Dauer...")
model_work_duration = Ridge(alpha=1.0)
model_work_duration.fit(X_train_scaled, y_wd_train)
y_wd_pred = model_work_duration.predict(X_test_scaled)
print(f"   R² Score: {r2_score(y_wd_test, y_wd_pred):.3f}")
print(f"   RMSE: {np.sqrt(mean_squared_error(y_wd_test, y_wd_pred)):.3f}")

# Modell 3: Pausen-Dauer
print("\n3️⃣ Modell für Pausen-Dauer...")
model_break_duration = Ridge(alpha=1.0)
model_break_duration.fit(X_train_scaled, y_bd_train)
y_bd_pred = model_break_duration.predict(X_test_scaled)
print(f"   R² Score: {r2_score(y_bd_test, y_bd_pred):.3f}")
print(f"   RMSE: {np.sqrt(mean_squared_error(y_bd_test, y_bd_pred)):.3f}")

# Modell 4: Nächste Session Empfehlung
print("\n4️⃣ Modell für nächste Session...")
model_next_session = Ridge(alpha=1.0)
model_next_session.fit(X_train_scaled, y_ns_train)
y_ns_pred = model_next_session.predict(X_test_scaled)
print(f"   R² Score: {r2_score(y_ns_test, y_ns_pred):.3f}")
print(f"   RMSE: {np.sqrt(mean_squared_error(y_ns_test, y_ns_pred)):.3f}")

# MODELLE SPEICHERN
print("\n💾 Speichere Modelle und Scaler...")

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

print("✅ Alle Modelle gespeichert in 'learning_models.pkl'")

# BEISPIEL-VORHERSAGE
print("\n" + "="*60)
print("📊 BEISPIEL-VORHERSAGE")
print("="*60)

# Beispiel: 120 Minuten Session, morgens, hohe Konzentration
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
print(f"   Geplante Session: 120 Minuten")
print(f"   Tageszeit: Morgen")
print(f"   Konzentration: 8.0/10")
print(f"   Tage seit letzter Session: 1")

print(f"\n📤 VORHERSAGE:")
print(f"   Empfohlene Anzahl Lernblöcke: {int(round(pred_blocks))}")
print(f"   Länge pro Lernblock: {int(round(pred_work))} Minuten")
print(f"   Länge pro Pause: {int(round(pred_break))} Minuten")
print(f"   Nächste Session in: {pred_next:.1f} Stunden")

print("\n" + "="*60)
print("✅ Training abgeschlossen!")
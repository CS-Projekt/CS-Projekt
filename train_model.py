# this file was created with some knowledge of the course at HSG "Machine Learning in Finance" 
# and some YouTube tutorials on Ridge Regression, but the code was written from scratch by us

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pickle

# Load data
print(" Loading training data...")
df = pd.read_csv('learning_sessions_data.csv')
print(f"Loaded {len(df)} training samples\n")

# Feature engineering
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

X = df_encoded[feature_columns] # Features for prediction

# Select targets (y)
y_work_blocks = df['optimal_work_blocks']
y_work_duration = df['work_block_duration']
y_break_duration = df['break_duration']
y_next_session = df['next_session_recommendation_hours']

# Train test split (80% train, 20% test)
X_train, X_test, y_wb_train, y_wb_test = train_test_split(
    X, y_work_blocks, test_size=0.2, random_state=42    
)                                                     # Split for work blocks
_, _, y_wd_train, y_wd_test = train_test_split(
    X, y_work_duration, test_size=0.2, random_state=42
)                                                    # Split for work duration        
_, _, y_bd_train, y_bd_test = train_test_split(
    X, y_break_duration, test_size=0.2, random_state=42
)                                                   # Split for break duration
_, _, y_ns_train, y_ns_test = train_test_split(
    X, y_next_session, test_size=0.2, random_state=42
)                                                # Split for next session       

# Feature scaling (important for Ridge Regression)
print("Scaling features...")
scaler = StandardScaler() # Initialize scaler
X_train_scaled = scaler.fit_transform(X_train) # Fit and transform training data
X_test_scaled = scaler.transform(X_test) # Transform test data

# TRAIN MODELS
print("\n Training Ridge Regression models...\n")

# Model 1: optimal number of work blocks
print("Model for work blocks...")
model_work_blocks = Ridge(alpha=1.0)  # alpha = regularization strength, here we use 1.0, because data is not too noisy and we want to avoid overfitting and multicollinearity
model_work_blocks.fit(X_train_scaled, y_wb_train)   # Train model
y_wb_pred = model_work_blocks.predict(X_test_scaled)    # Predict on test set
print(f"   R² Score: {r2_score(y_wb_test, y_wb_pred):.3f}") # Evaluate R squared
print(f"   RMSE: {np.sqrt(mean_squared_error(y_wb_test, y_wb_pred)):.3f}") # Evaluate RMSE

# Model 2: work-block duration
print("\n Model for work-block duration...")
model_work_duration = Ridge(alpha=1.0) # here we also choose alpha = 1.0, because we want to avoid overfitting and multicollinearity
model_work_duration.fit(X_train_scaled, y_wd_train)  # Train model
y_wd_pred = model_work_duration.predict(X_test_scaled)  # Predict on test set
print(f"   R² Score: {r2_score(y_wd_test, y_wd_pred):.3f}") # Evaluate R squared
print(f"   RMSE: {np.sqrt(mean_squared_error(y_wd_test, y_wd_pred)):.3f}")  # Evaluate RMSE

# Model 3: break duration
print("\n Model for break duration...") 
model_break_duration = Ridge(alpha=1.0)   # here we also choose alpha = 1.0, because we want to avoid overfitting and multicollinearity
model_break_duration.fit(X_train_scaled, y_bd_train) # Train model
y_bd_pred = model_break_duration.predict(X_test_scaled) 
print(f"   R² Score: {r2_score(y_bd_test, y_bd_pred):.3f}")
print(f"   RMSE: {np.sqrt(mean_squared_error(y_bd_test, y_bd_pred)):.3f}")

# Model 4: next session recommendation
print("\n4 Model for next session...")  
model_next_session = Ridge(alpha=1.0)   # here we also choose alpha = 1.0, because we want to avoid overfitting and multicollinearity
model_next_session.fit(X_train_scaled, y_ns_train) # Train model
y_ns_pred = model_next_session.predict(X_test_scaled)   # Predict on test set
print(f"   R² Score: {r2_score(y_ns_test, y_ns_pred):.3f}")  # Evaluate R squared
print(f"   RMSE: {np.sqrt(mean_squared_error(y_ns_test, y_ns_pred)):.3f}") # Evaluate RMSE

# SAVE MODELS
print("\n Saving models and scaler...")

models = {
    'scaler': scaler,
    'work_blocks': model_work_blocks,
    'work_duration': model_work_duration,
    'break_duration': model_break_duration,
    'next_session': model_next_session,
    'feature_columns': feature_columns
}

with open('learning_models.pkl', 'wb') as f:
    pickle.dump(models, f)  # Save all models and scaler in pkl file

print("Saved all models to 'learning_models.pkl'") # we made the print statements that we can see in the console the progress of the training and saving process


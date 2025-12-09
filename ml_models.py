# DISCLAIMER: This ml_models.py was created with the help of AI, because when we tried there were a lot of errors 
# and the AI could help us to fix them and supported us to write some new lines. Also AI helped us understanding how to integrate some features.

from __future__ import annotations

import pickle
import sqlite3
from pathlib import Path
from typing import Dict, Any

import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


# Define paths and feature columns
MODEL_PATH = Path("learning_models.pkl")   # Path to save/load the trained models
DB_PATH = Path("learning_plan.db")  # Path to the SQLite database with training data

TIME_OF_DAY_VALUES = ["morning", "afternoon", "evening", "night"] # Possible values for time_of_day
BASE_FEATURES = [ 
    "total_session_duration", # Base features for prediction
    "concentration_baseline", 
    "days_since_last_session",
    "previous_session_rating",
    "cluster_id",
] 
TIME_FEATURES = [f"time_{key}" for key in TIME_OF_DAY_VALUES] # Time features 
FEATURE_COLUMNS = BASE_FEATURES + TIME_FEATURES # attatches all features together


# Fetch training data from the SQLite database
def _fetch_training_frame(db_path: Path) -> pd.DataFrame:
    conn = sqlite3.connect(db_path) # Connect to the database
    try:
        df = pd.read_sql_query(           # Fetch data from learning_samples table
            """
            SELECT
                total_session_duration,
                time_of_day,
                concentration_baseline,
                days_since_last_session,
                previous_session_rating,
                optimal_work_blocks,
                work_block_duration,
                break_duration,
                next_session_recommendation_hours,
                cluster_id
            FROM learning_samples
            """,
            conn,
        )
    finally:
        conn.close() # Ensure the connection is closed
    if df.empty:
        raise ValueError("The learning_samples table is empty.") # Check if data is empty
    missing = [col for col in FEATURE_COLUMNS if col not in df.columns] # Check for missing columns
    if missing:
        for col in missing:   # Add missing columns with default value 0
            df[col] = 0
    return df  # Return the fetched data frame

# Prepare features by one-hot encoding (Matching names to numbers) time_of_day (This part was created completley by AI)
def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "time_of_day" not in df.columns:
        raise ValueError("Missing 'time_of_day' column in features.")
    dummies = pd.get_dummies(df["time_of_day"], prefix="time") 
    for col in TIME_FEATURES:  # Ensure all time features are present
        if col not in dummies.columns:             # If a time feature is missing, add it with default value 0
            dummies[col] = 0
    df = pd.concat([df[BASE_FEATURES], dummies[TIME_FEATURES]], axis=1) # Combine base features with time features
    return df # Return the prepared feature frame

# Train regression models for study plan prediction 
def train_models_from_db(db_path: Path = DB_PATH) -> Dict[str, Any]: # Train models and return them
    df = _fetch_training_frame(db_path)
    feature_frame = _prepare_features(df) 

    scaler = StandardScaler() # Initialize scaler
    X = scaler.fit_transform(feature_frame) # Scale features

    models: Dict[str, Any] = {
        "scaler": scaler,
        "feature_columns": FEATURE_COLUMNS, # Store feature columns for later use
    }
# End of AI created part

# Define target variables for regression
    targets = {
        "work_duration": df["work_block_duration"].astype(float),   # Ensure target variables are float
        "break_duration": df["break_duration"].astype(float),
        "next_session": df["next_session_recommendation_hours"].astype(float),
        "work_blocks": df["optimal_work_blocks"].astype(float),
    }

    for name, target in targets.items(): # Train a Ridge Regression model for each target
        model = Ridge(alpha=1.0)   # Regularization strength set to 1.0 to avoid overfitting and multicollinearity
        model.fit(X, target) # Train model
        models[name] = model # Store trained model

    with MODEL_PATH.open("wb") as f: # Save all models and scaler to a pickle file
        pickle.dump(models, f)
    return models   # Return the trained models

# Load trained models
def load_models() -> Dict[str, Any]: # Load models from pickle file
    if not MODEL_PATH.exists(): 
        raise FileNotFoundError(
            f"No trained model found at {MODEL_PATH}. Train it first via the sidebar button." # Raise error if model file does not exist
        )
    with MODEL_PATH.open("rb") as f:
        return pickle.load(f) # Load models from pickle file


def _clamp(value: float, minimum: float, maximum: float) -> float: # Clamp value between minimum and maximum
    return max(minimum, min(maximum, value)) 

# Predict study plan based on features
def predict_plan(     
    models: Dict[str, Any], # Trained models and scaler
    features: Dict[str, Any], # Input features for prediction
    desired_total_duration: int, # Desired total duration of the study session
) -> Dict[str, Any]:
    df = pd.DataFrame([features])
    prepared = _prepare_features(df) # Prepare features for prediction

    feature_columns = models["feature_columns"] # Get feature columns used during training
    for col in feature_columns:  # Ensure all required feature columns are present
        if col not in prepared.columns:
            prepared[col] = 0
    prepared = prepared[feature_columns] # Reorder columns to match training

    scaler: StandardScaler = models["scaler"] # Load scaler
    X = scaler.transform(prepared) # Scale features

    work_duration = int(
        round(
            _clamp(models["work_duration"].predict(X)[0], minimum=15, maximum=60) # Clamp work duration between 15 and 60 minutes, because shorter or longer durations are not effective (based on research)
        )
    )
    break_duration = int(
        round(
            _clamp(models["break_duration"].predict(X)[0], minimum=5, maximum=20) # Clamp break duration between 5 and 20 minutes, because shorter or longer breaks are not effective (based on research)
        )
    )
    next_session_hours = float(
        _clamp(models["next_session"].predict(X)[0], minimum=2, maximum=36) # Clamp next session recommendation between 2 and 36 hours
    )
    predicted_blocks = int(round(models["work_blocks"].predict(X)[0])) # Predict optimal number of work blocks
    if predicted_blocks < 1:                # Ensure at least one work block is scheduled
        cycle = work_duration + break_duration # Calculate cycle duration
        predicted_blocks = max(
            1,
            int((desired_total_duration + break_duration) / max(cycle, 1)), # Ensure at least one work block is scheduled
        )

    schedule = [] # Build the study schedule
    total_calculated = 0 # Total calculated duration, starts at 0, because nothing is scheduled yet
    for block in range(predicted_blocks): # Loop through each work block
        schedule.append(
            {"type": "Study", "duration": work_duration, "block": block + 1} # Add study block to schedule
        )
        total_calculated += work_duration # Update total calculated duration
        if block < predicted_blocks - 1:
            schedule.append(
                {"type": "Break", "duration": break_duration, "block": block + 1} # Add break to schedule
            )
            total_calculated += break_duration # Update total calculated duration

# Return the predicted study plan
    return {
        "blocks": predicted_blocks,
        "work_duration": work_duration,
        "break_duration": break_duration,
        "next_session_hours": next_session_hours,
        "total_duration": desired_total_duration,
        "actual_duration": total_calculated,
        "schedule": schedule,
    }

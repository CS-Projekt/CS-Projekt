"""
Utility helpers to train and use Ridge Regression models for the study planner.
"""

from __future__ import annotations

import pickle
import sqlite3
from pathlib import Path
from typing import Dict, Any

import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


MODEL_PATH = Path("learning_models.pkl")
DB_PATH = Path("learning_plan.db")

TIME_OF_DAY_VALUES = ["morning", "afternoon", "evening", "night"]
BASE_FEATURES = [
    "total_session_duration",
    "concentration_baseline",
    "days_since_last_session",
    "previous_session_rating",
    "cluster_id",
]
TIME_FEATURES = [f"time_{key}" for key in TIME_OF_DAY_VALUES]
FEATURE_COLUMNS = BASE_FEATURES + TIME_FEATURES


def _fetch_training_frame(db_path: Path) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(
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
        conn.close()
    if df.empty:
        raise ValueError("The learning_samples table is empty.")
    missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing:
        # populate placeholders; actual values will be created via dummies.
        for col in missing:
            df[col] = 0
    return df


def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "time_of_day" not in df.columns:
        raise ValueError("Missing 'time_of_day' column in features.")
    dummies = pd.get_dummies(df["time_of_day"], prefix="time")
    for col in TIME_FEATURES:
        if col not in dummies.columns:
            dummies[col] = 0
    df = pd.concat([df[BASE_FEATURES], dummies[TIME_FEATURES]], axis=1)
    return df


def train_models_from_db(db_path: Path = DB_PATH) -> Dict[str, Any]:
    df = _fetch_training_frame(db_path)
    feature_frame = _prepare_features(df)

    scaler = StandardScaler()
    X = scaler.fit_transform(feature_frame)

    models: Dict[str, Any] = {
        "scaler": scaler,
        "feature_columns": FEATURE_COLUMNS,
    }

    targets = {
        "work_duration": df["work_block_duration"].astype(float),
        "break_duration": df["break_duration"].astype(float),
        "next_session": df["next_session_recommendation_hours"].astype(float),
        "work_blocks": df["optimal_work_blocks"].astype(float),
    }

    for name, target in targets.items():
        model = Ridge(alpha=1.0)
        model.fit(X, target)
        models[name] = model

    with MODEL_PATH.open("wb") as f:
        pickle.dump(models, f)
    return models


def load_models() -> Dict[str, Any]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No trained model found at {MODEL_PATH}. Train it first via the sidebar button."
        )
    with MODEL_PATH.open("rb") as f:
        return pickle.load(f)


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def predict_plan(
    models: Dict[str, Any],
    features: Dict[str, Any],
    desired_total_duration: int,
) -> Dict[str, Any]:
    df = pd.DataFrame([features])
    prepared = _prepare_features(df)

    feature_columns = models["feature_columns"]
    for col in feature_columns:
        if col not in prepared.columns:
            prepared[col] = 0
    prepared = prepared[feature_columns]

    scaler: StandardScaler = models["scaler"]
    X = scaler.transform(prepared)

    work_duration = int(
        round(
            _clamp(models["work_duration"].predict(X)[0], minimum=15, maximum=60)
        )
    )
    break_duration = int(
        round(
            _clamp(models["break_duration"].predict(X)[0], minimum=5, maximum=20)
        )
    )
    next_session_hours = float(
        _clamp(models["next_session"].predict(X)[0], minimum=2, maximum=36)
    )
    predicted_blocks = int(round(models["work_blocks"].predict(X)[0]))
    if predicted_blocks < 1:
        cycle = work_duration + break_duration
        predicted_blocks = max(
            1,
            int((desired_total_duration + break_duration) / max(cycle, 1)),
        )

    schedule = []
    total_calculated = 0
    for block in range(predicted_blocks):
        schedule.append(
            {"type": "Study", "duration": work_duration, "block": block + 1}
        )
        total_calculated += work_duration
        if block < predicted_blocks - 1:
            schedule.append(
                {"type": "Break", "duration": break_duration, "block": block + 1}
            )
            total_calculated += break_duration

    return {
        "blocks": predicted_blocks,
        "work_duration": work_duration,
        "break_duration": break_duration,
        "next_session_hours": next_session_hours,
        "total_duration": desired_total_duration,
        "actual_duration": total_calculated,
        "schedule": schedule,
    }

"""
database.py – central data API for the AI Study Timer

Main responsibilities:
- Set up and manage a local SQLite database
- Provide a clean Python API for:
    * users
    * study sessions (from the app)
    * learning_samples (from learning_sessions_data.csv)
"""

import sqlite3
from contextlib import contextmanager
from typing import List, Dict, Any, Optional
from datetime import datetime
import csv
import os

DB_PATH = "study_timer.db"


# ---------------------------------------------------------
#  Connection & Setup
# ---------------------------------------------------------
@contextmanager
def get_connection():
    """Context manager for a DB connection with row factory."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    """Create all required tables if they do not exist yet."""
    with get_connection() as conn:
        cur = conn.cursor()

        # Users
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                study_field TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        # Study sessions created via the app
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                subject TEXT,
                duration_minutes INTEGER NOT NULL,
                pause_minutes INTEGER,
                focus_score INTEGER,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                source TEXT DEFAULT 'app',
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
            """
        )

        # Training data table that mirrors learning_sessions_data.csv
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS learning_samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,

                total_session_duration INTEGER,
                time_of_day TEXT,
                time_of_day_encoded INTEGER,
                concentration_baseline REAL,
                days_since_last_session INTEGER,
                previous_session_rating REAL,
                optimal_work_blocks INTEGER,
                work_block_duration INTEGER,
                break_duration INTEGER,
                concentration_score REAL,
                next_session_recommendation_hours REAL
            )
            """
        )


# ---------------------------------------------------------
#  User API
# ---------------------------------------------------------
def create_user(name: str, study_field: Optional[str] = None) -> int:
    """Create a new user and return its ID."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO users (name, study_field) VALUES (?, ?)",
            (name, study_field),
        )
        return cur.lastrowid


def get_user(user_id: int) -> Optional[Dict[str, Any]]:
    """Fetch a user by ID (or None if not found)."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        row = cur.fetchone()
        return dict(row) if row else None


def get_or_create_default_user() -> int:
    """
    For the MVP: get an existing default user or create one.

    Useful when you do not have a full login system yet.
    """
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT id FROM users ORDER BY id LIMIT 1")
        row = cur.fetchone()
        if row:
            return row["id"]

        cur.execute(
            "INSERT INTO users (name, study_field) VALUES (?, ?)",
            ("Default User", "Unknown"),
        )
        return cur.lastrowid


# ---------------------------------------------------------
#  Session API (for the app)
# ---------------------------------------------------------
def create_session(
    user_id: int,
    duration_minutes: int,
    pause_minutes: int,
    focus_score: Optional[int] = None,
    subject: Optional[str] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    source: str = "app",
) -> int:
    """
    Store a new study session.
    """
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO sessions (
                user_id, subject, duration_minutes, pause_minutes,
                focus_score, start_time, end_time, source
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                subject,
                duration_minutes,
                pause_minutes,
                focus_score,
                start_time.isoformat() if isinstance(start_time, datetime) else start_time,
                end_time.isoformat() if isinstance(end_time, datetime) else end_time,
                source,
            ),
        )
        return cur.lastrowid


def get_sessions_for_user(
    user_id: int,
    limit: Optional[int] = None,
    order: str = "DESC",
) -> List[Dict[str, Any]]:
    """
    Fetch sessions for a given user, e.g. for visualizations.

    Parameters
    ----------
    user_id : int
        ID of the user whose sessions you want.
    limit : Optional[int]
        Maximum number of rows (None = no limit).
    order : str
        'ASC' for oldest first, 'DESC' for newest first.
    """
    order = order.upper()
    if order not in ("ASC", "DESC"):
        order = "DESC"

    sql = f"""
        SELECT *
        FROM sessions
        WHERE user_id = ?
        ORDER BY start_time {order}
    """
    if limit is not None:
        sql += " LIMIT ?"
        params = (user_id, limit)
    else:
        params = (user_id,)

    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(sql, params)
        rows = cur.fetchall()
        return [dict(r) for r in rows]


# ---------------------------------------------------------
#  CSV-based training data (learning_sessions_data.csv)
# ---------------------------------------------------------
CSV_COLUMNS = [
    "total_session_duration",
    "time_of_day",
    "time_of_day_encoded",
    "concentration_baseline",
    "days_since_last_session",
    "previous_session_rating",
    "optimal_work_blocks",
    "work_block_duration",
    "break_duration",
    "concentration_score",
    "next_session_recommendation_hours",
]


def import_learning_sessions_csv(
    csv_path: str = "learning_sessions_data.csv",
    clear_existing: bool = False,
) -> int:
    """
    Import learning_sessions_data.csv into the 'learning_samples' table.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file.
    clear_existing : bool
        If True, delete all existing rows in learning_samples first.

    Returns
    -------
    int
        Number of successfully imported rows.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    with get_connection() as conn:
        cur = conn.cursor()

        if clear_existing:
            cur.execute("DELETE FROM learning_samples")

        imported = 0

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            # Ensure all expected columns are present
            missing = [c for c in CSV_COLUMNS if c not in reader.fieldnames]
            if missing:
                raise ValueError(
                    f"The following columns are missing in the CSV: {missing}\n"
                    f"Found columns: {reader.fieldnames}"
                )

            for row in reader:
                # Helper converters: empty string -> None, otherwise int/float
                def to_int(val: str) -> Optional[int]:
                    val = val.strip()
                    if val == "":
                        return None
                    return int(float(val))

                def to_float(val: str) -> Optional[float]:
                    val = val.strip()
                    if val == "":
                        return None
                    return float(val)

                cur.execute(
                    """
                    INSERT INTO learning_samples (
                        total_session_duration,
                        time_of_day,
                        time_of_day_encoded,
                        concentration_baseline,
                        days_since_last_session,
                        previous_session_rating,
                        optimal_work_blocks,
                        work_block_duration,
                        break_duration,
                        concentration_score,
                        next_session_recommendation_hours
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        to_int(row["total_session_duration"]),
                        row["time_of_day"].strip() or None,
                        to_int(row["time_of_day_encoded"]),
                        to_float(row["concentration_baseline"]),
                        to_int(row["days_since_last_session"]),
                        to_float(row["previous_session_rating"]),
                        to_int(row["optimal_work_blocks"]),
                        to_int(row["work_block_duration"]),
                        to_int(row["break_duration"]),
                        to_float(row["concentration_score"]),
                        to_float(row["next_session_recommendation_hours"]),
                    ),
                )
                imported += 1

        return imported


def get_learning_samples() -> List[Dict[str, Any]]:
    """
    Fetch all entries from learning_samples, e.g. to train your ML model.
    """
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM learning_samples")
        rows = cur.fetchall()
        return [dict(r) for r in rows]


# ---------------------------------------------------------
#  Dev helper – reset DB (only for local development!)
# ---------------------------------------------------------
def reset_database(confirm: bool = False) -> None:
    """
    WARNING: completely delete the DB file.

    Only use this during development if you want to reset everything.
    """
    if not confirm:
        raise RuntimeError(
            "Call reset_database(confirm=True) if you really want to delete the DB."
        )
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print("Database file deleted:", DB_PATH)
    else:
        print("No database file found:", DB_PATH)


if __name__ == "__main__":
    # Small self-test when running this file directly
    init_db()
    user_id = get_or_create_default_user()
    print("Default user ID:", user_id)

    if os.path.exists("learning_sessions_data.csv"):
        n = import_learning_sessions_csv("learning_sessions_data.csv")
        print(f"Imported {n} learning samples from CSV.")
    else:
        print("learning_sessions_data.csv not found in current folder.")

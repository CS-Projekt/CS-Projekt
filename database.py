

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional


DB_PATH = "learning_plan.db"


@contextmanager
def get_connection():
    # Context manager provides an SQLite connection, commits after successful
    # Automatically blocks and always closes the connection cleanly.
    """Yield a SQLite connection that auto-commits on success."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()

#Creation of the database if it does not yet exist
def init_db() -> None:
    """Create all required tables if they are missing."""
    with get_connection() as conn: #returns a database connection
        cur = conn.cursor()

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
                next_session_recommendation_hours REAL,
                cluster_id INTEGER DEFAULT 1
            )
            """
        )
        cur.execute("PRAGMA table_info(learning_samples)")
        columns = {row[1] for row in cur.fetchall()}
        if "cluster_id" not in columns:
            cur.execute("ALTER TABLE learning_samples ADD COLUMN cluster_id INTEGER DEFAULT 1")
            cur.execute("UPDATE learning_samples SET cluster_id = 1 WHERE cluster_id IS NULL")


#Creates a new user in the “users” table
def create_user(name: str, study_field: Optional[str] = None) -> int:
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
#Insert new record in “users”
            "INSERT INTO users (name, study_field) VALUES (?, ?)",
            (name, study_field),
        )
        return cur.lastrowid #Return the ID of the newly created user

#Retrieves a user with a given ID from the database
def get_user(user_id: int) -> Optional[Dict[str, Any]]:
    with get_connection() as conn:
        cur = conn.cursor()
        # Select matching users based on their ID
        cur.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        row = cur.fetchone()
        return dict(row) if row else None


def get_or_create_default_user() -> int:
    """
    For the single-user Streamlit app we always work with a fallback user.
    """
    # Check whether any user already exists
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT id FROM users ORDER BY id LIMIT 1")
        row = cur.fetchone()
    #If a user exists: use their ID as the “Default User”
        if row:
            return row["id"]
    #If no user exists yet: create a new default user
        cur.execute(
            "INSERT INTO users (name, study_field) VALUES (?, ?)",
            ("Default User", "Unknown"),
        )
        return cur.lastrowid


# -- sessions ---------------------------------------------------------------
# Creates a new entry for a completed learning unit.
#Parameters:
#- user_id: ID of the user who owns the session
#- duration_minutes: Duration of the learning unit in minutes
#- pause_minutes: Duration of breaks in minutes
#- focus_score: Optional focus score for the session
#- subject: Optional subject/topic of the session
#- start_time: Optional start time of the session
#- end_time: Optional end time of the session
#- source: Source of the data (default: “app”)
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
    """Persist a completed study session."""
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

#Retrieves a user's saved learning sessions from the database.
#Parameters:
#- user_id: ID of the user whose sessions are to be loaded
#- limit: maximum number of sessions returned (None = all)
#- order: Sort order by start_time (“ASC” or ‘DESC’, default: “DESC”)
#Return value:
#- List of dictionaries, each entry corresponds to a session row from the DB
def get_sessions_for_user(
    user_id: int,
    limit: Optional[int] = None,
    order: str = "DESC",
) -> List[Dict[str, Any]]:
    """Fetch stored sessions for analytics or exports."""
    valid_order = "DESC" if order.upper() != "ASC" else "ASC"
    sql = [
        "SELECT * FROM sessions WHERE user_id = ? ORDER BY start_time",
        valid_order,
    ]
    params: List[Any] = [user_id]
    if limit is not None:
        sql.append("LIMIT ?")
        params.append(limit)

    query = " ".join(sql)
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(query, tuple(params))
        rows = cur.fetchall()
        return [dict(row) for row in rows]


# Inserts a complete ML training sample (e.g., a CSV row) into the learning_samples table.
def insert_learning_sample(sample: Dict[str, Any]) -> int:
    """Convenience helper for scripts that ingest CSV samples."""
    expected_columns = [
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
        "cluster_id",
    ]
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            f"""
            INSERT INTO learning_samples ({", ".join(expected_columns)})
            VALUES ({", ".join("?" for _ in expected_columns)})
            """,
            tuple(sample[col] for col in expected_columns),
        )
        return cur.lastrowid

#Retrieves the most recently saved ML training samples from the database, optionally limited by limit.
def fetch_learning_samples(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    with get_connection() as conn:
        cur = conn.cursor()
        sql = "SELECT * FROM learning_samples ORDER BY id DESC"
        params: List[Any] = []
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        cur.execute(sql, tuple(params))
        rows = cur.fetchall()
        return [dict(row) for row in rows]

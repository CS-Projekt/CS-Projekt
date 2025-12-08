# train_clustering.py is completley created by AI but reviewed and verified by us

from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import sqlite3
from joblib import dump
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
    "learning_days_ratio",
    "daily_reviews",
    "reviews_per_learning_day",
    "accuracy",
]

# Define path to the SQLite database
DB_PATH = Path("learning_plan.db")

# Canonical mapping that the rest of the app expects.
CANONICAL_CLUSTER_ORDER = {
    "marathoner": 0,
    "planner": 1,
    "sprinter": 2,
}
CLUSTER_ID_TO_NAME = {v: k for k, v in CANONICAL_CLUSTER_ORDER.items()}

# Build mapping from learned labels to canonical cluster IDs
def _build_label_mapping(
    predicted_labels: pd.Series,
    human_labels: pd.Series,
    n_clusters: int,
) -> Dict[int, int]:
    """
    Use the known cluster names in the CSV to force the learned labels
    into the canonical ordering (0=Marathoner, 1=Planner, 2=Sprinter).
    """
    normalized = human_labels.str.lower().str.strip()
    if normalized.isnull().any():
        raise ValueError("The CSV must contain a 'cluster' column with non-empty names.")

    predicted = pd.Series(predicted_labels, index=human_labels.index)
    label_map: Dict[int, int] = {}
    used_learned = set()
    used_canonical = set()

    for canonical_name, canonical_id in CANONICAL_CLUSTER_ORDER.items():
        mask = normalized == canonical_name
        if not mask.any():
            continue

        counts = predicted[mask].value_counts()
        for learned_label in counts.index:
            if learned_label not in used_learned:
                label_map[learned_label] = canonical_id
                used_learned.add(learned_label)
                used_canonical.add(canonical_id)
                break

    remaining_canonical = [
        cid for cid in CANONICAL_CLUSTER_ORDER.values() if cid not in used_canonical
    ]
    remaining_learned = [
        label for label in range(n_clusters) if label not in used_learned
    ]

    for learned_label, canonical_id in zip(remaining_learned, remaining_canonical):
        label_map[learned_label] = canonical_id
        used_learned.add(learned_label)
        used_canonical.add(canonical_id)

    if len(label_map) != n_clusters:
        missing = [label for label in range(n_clusters) if label not in label_map]
        raise RuntimeError(f"Could not assign canonical ids for cluster labels: {missing}")

    return label_map

def _load_db_samples(db_path: Path) -> pd.DataFrame:
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found at {db_path.resolve()}")
    conn = sqlite3.connect(db_path)
    query = """
        SELECT
            total_session_duration,
            days_since_last_session,
            concentration_baseline,
            previous_session_rating,
            optimal_work_blocks,
            work_block_duration,
            break_duration,
            concentration_score,
            cluster_id
        FROM learning_samples
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    if df.empty:
        raise ValueError("learning_samples table is empty; cannot train clustering.")

    df["learning_days_ratio"] = 1 / (1 + df["days_since_last_session"].fillna(0))
    df["daily_reviews"] = df["total_session_duration"].fillna(0) / 3.0
    df["reviews_per_learning_day"] = df["daily_reviews"] * df["learning_days_ratio"]
    df["accuracy"] = (
        df["concentration_score"].fillna(df["concentration_baseline"]) / 10.0
    ).clip(0.5, 0.98)

    df["cluster"] = df["cluster_id"].map(CLUSTER_ID_TO_NAME).fillna("planner")
    return df[FEATURE_COLUMNS + ["cluster"]]


def _load_training_dataframe(csv_path: Optional[str]) -> pd.DataFrame:
    fallback_reason = None
    try:
        df = _load_db_samples(DB_PATH)
    except Exception as db_err:
        fallback_reason = f"[train_clustering] Falling back to CSV due to DB error: {db_err}"
        df = None

    if df is None or df["cluster"].nunique() < len(CANONICAL_CLUSTER_ORDER):
        if csv_path is None:
            if df is None:
                raise RuntimeError("No DB data available and no CSV fallback configured.")
            raise RuntimeError(
                "DB data does not provide all canonical clusters and no CSV fallback was provided."
            )
        csv_file = Path(csv_path)
        if not csv_file.exists():
            raise FileNotFoundError(f"Could not find CSV file at {csv_file.resolve()}")

        csv_df = pd.read_csv(csv_file)
        if fallback_reason:
            print(fallback_reason)
        else:
            print(
                "[train_clustering] DB data misses some cluster labels; augmenting with CSV fallback."
            )

        df = csv_df if df is None else pd.concat([df, csv_df], ignore_index=True)

    return df


# Train KMeans clustering and save artifacts
def train_and_save_clustering(
    csv_path: str = "cluster_dummy_data.csv",
    artifact_path: str = "cluster_artifacts.joblib",
    n_clusters: int = 3,
) -> None:
    """
    Train KMeans + scaler and persist them for use inside the Streamlit app.
    """
    df = _load_training_dataframe(csv_path)
    missing_columns = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns for clustering: {missing_columns}")

    if "cluster" not in df.columns:
        raise ValueError("Training data needs a 'cluster' column with the human-readable label.")

    X = df[FEATURE_COLUMNS]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    kmeans.fit(X_scaled)

    predicted_labels = kmeans.predict(X_scaled)
    label_mapping = _build_label_mapping(
        predicted_labels=predicted_labels,
        human_labels=df["cluster"],
        n_clusters=n_clusters,
    )

    artifacts = {
        "scaler": scaler,
        "kmeans": kmeans,
        "feature_columns": FEATURE_COLUMNS,
        "label_mapping": label_mapping,
    }

    artifact_file = Path(artifact_path)
    dump(artifacts, artifact_file)
    print(f" Saved scaler + KMeans to '{artifact_file.resolve()}'")

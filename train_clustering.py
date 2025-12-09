# DISCLAIMER: train_clustering.py is completley created by AI but reviewed, commented and verified by us

from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import sqlite3
from joblib import dump
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Define feature columns for clustering
FEATURE_COLUMNS = [
    "learning_days_ratio",
    "daily_reviews",
    "reviews_per_learning_day",
    "accuracy",
]

# Define path to the SQLite database
DB_PATH = Path("learning_plan.db")

# assign canonical cluster IDs
CANONICAL_CLUSTER_ORDER = {
    "marathoner": 0,
    "planner": 1,
    "sprinter": 2,
}
CLUSTER_ID_TO_NAME = {v: k for k, v in CANONICAL_CLUSTER_ORDER.items()} # Reverse mapping

# Build mapping from learned labels to canonical cluster IDs
def _build_label_mapping(
    predicted_labels: pd.Series, # predicted cluster labels from KMeans
    human_labels: pd.Series, # human-readable cluster names from CSV
    n_clusters: int,  # number of clusters
) -> Dict[int, int]: # Build mapping from learned labels to canonical cluster IDs
    """
    Use the known cluster names in the CSV to force the learned labels
    into the canonical ordering (0=Marathoner, 1=Planner, 2=Sprinter).
    """
    normalized = human_labels.str.lower().str.strip() # Normalize human labels
    if normalized.isnull().any(): # Check for missing cluster names
        raise ValueError("The CSV must contain a 'cluster' column with non-empty names.")

    predicted = pd.Series(predicted_labels, index=human_labels.index) # Series of predicted labels
    label_map: Dict[int, int] = {} # Mapping from learned label to canonical ID
    used_learned = set() # Track used learned labels
    used_canonical = set() # Track used canonical IDs

    for canonical_name, canonical_id in CANONICAL_CLUSTER_ORDER.items(): # Loop through canonical clusters
        mask = normalized == canonical_name # Find samples matching the canonical name
        if not mask.any(): 
            continue 

        counts = predicted[mask].value_counts()     # Count predicted labels for these samples
        for learned_label in counts.index:
            if learned_label not in used_learned:
                label_map[learned_label] = canonical_id # Assign mapping
                used_learned.add(learned_label) # Mark learned label as used
                used_canonical.add(canonical_id) # Mark canonical ID as used
                break

    remaining_canonical = [ 
        cid for cid in CANONICAL_CLUSTER_ORDER.values() if cid not in used_canonical
    ] # Find remaining canonical IDs
    remaining_learned = [
        label for label in range(n_clusters) if label not in used_learned
    ] # Find remaining learned labels

    for learned_label, canonical_id in zip(remaining_learned, remaining_canonical): # Assign remaining mappings
        label_map[learned_label] = canonical_id
        used_learned.add(learned_label)
        used_canonical.add(canonical_id)

    if len(label_map) != n_clusters: # Verify all clusters are mapped
        missing = [label for label in range(n_clusters) if label not in label_map]
        raise RuntimeError(f"Could not assign canonical ids for cluster labels: {missing}")

    return label_map # Return the mapping

def _load_db_samples(db_path: Path) -> pd.DataFrame: # Load training samples from the database
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found at {db_path.resolve()}") # Check if database file exists
    conn = sqlite3.connect(db_path) # Connect to the SQLite database
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
    df = pd.read_sql_query(query, conn) # Execute query and load data into DataFrame
    conn.close() # Close the database connection
    if df.empty: # Check if the DataFrame is empty
        raise ValueError("learning_samples table is empty; cannot train clustering.")

    df["learning_days_ratio"] = 1 / (1 + df["days_since_last_session"].fillna(0)) # Calculate learning days ratio
    df["daily_reviews"] = df["total_session_duration"].fillna(0) / 3.0 # Estimate daily reviews
    df["reviews_per_learning_day"] = df["daily_reviews"] * df["learning_days_ratio"] # Calculate reviews per learning day
    df["accuracy"] = ( # Estimate accuracy
        df["concentration_score"].fillna(df["concentration_baseline"]) / 10.0 # Normalize score
    ).clip(0.5, 0.98) # Clamp accuracy between 0.5 and 0.98

    df["cluster"] = df["cluster_id"].map(CLUSTER_ID_TO_NAME).fillna("planner") # Map cluster IDs to names, default to "planner"
    return df[FEATURE_COLUMNS + ["cluster"]] # Return relevant features and cluster names


def _load_training_dataframe(csv_path: Optional[str]) -> pd.DataFrame: # Load training DataFrame from DB or CSV
    fallback_reason = None # Reason for fallback to CSV
    try:
        df = _load_db_samples(DB_PATH) # Attempt to load samples from the database
    except Exception as db_err:     # On any error, fallback to CSV
        fallback_reason = f"[train_clustering] Falling back to CSV due to DB error: {db_err}"
        df = None # Set df to None to indicate failure

    if df is None or df["cluster"].nunique() < len(CANONICAL_CLUSTER_ORDER): # Check if all canonical clusters are present
        if csv_path is None: #  No CSV fallback provided
            if df is None:
                raise RuntimeError("No DB data available and no CSV fallback configured.") # Raise error if no data at all
            raise RuntimeError( # Raise error if some clusters are missing
                "DB data does not provide all canonical clusters and no CSV fallback was provided."
            )
        csv_file = Path(csv_path) # Path to the CSV file
        if not csv_file.exists():
            raise FileNotFoundError(f"Could not find CSV file at {csv_file.resolve()}")

        csv_df = pd.read_csv(csv_file) # Load data from CSV
        if fallback_reason:
            print(fallback_reason) # Log the reason for fallback
        else:
            print(
                "[train_clustering] DB data misses some cluster labels; augmenting with CSV fallback."
            )

        df = csv_df if df is None else pd.concat([df, csv_df], ignore_index=True) # Combine DB and CSV data

    return df


# Train KMeans clustering and save artifacts
def train_and_save_clustering(
    csv_path: str = "cluster_dummy_data.csv", # Path to CSV fallback for training data
    artifact_path: str = "cluster_artifacts.joblib", # Path to save trained artifacts
    n_clusters: int = 3, # Number of clusters for KMeans
) -> None:
    """
    Train KMeans + scaler and persist them for use inside the Streamlit app.
    """
    df = _load_training_dataframe(csv_path) # Load training data
    missing_columns = [col for col in FEATURE_COLUMNS if col not in df.columns] # Check for missing feature columns
    if missing_columns: # Raise error if any required columns are missing
        raise ValueError(f"Missing required columns for clustering: {missing_columns}")

    if "cluster" not in df.columns: # Ensure 'cluster' column is present
        raise ValueError("Training data needs a 'cluster' column with the human-readable label.")

    X = df[FEATURE_COLUMNS] # Extract features for clustering

    scaler = StandardScaler() # Initialize scaler
    X_scaled = scaler.fit_transform(X) # Scale features

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20) # Initialize KMeans
    kmeans.fit(X_scaled) # Fit KMeans to scaled data

    predicted_labels = kmeans.predict(X_scaled) # Predict cluster labels
    label_mapping = _build_label_mapping(   # Build mapping from learned labels to canonical IDs
        predicted_labels=predicted_labels, # predicted cluster labels
        human_labels=df["cluster"], # human-readable cluster names
        n_clusters=n_clusters, # number of clusters
    )

    artifacts = { # Store artifacts to be saved
        "scaler": scaler,
        "kmeans": kmeans,
        "feature_columns": FEATURE_COLUMNS,
        "label_mapping": label_mapping,
    }

    artifact_file = Path(artifact_path) # Path to save artifacts
    dump(artifacts, artifact_file)
    print(f" Saved scaler + KMeans to '{artifact_file.resolve()}'")

# train_clustering.py is completley created by AI but reviewed and verified by us

from pathlib import Path
from typing import Dict

import pandas as pd
from joblib import dump
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
    "learning_days_ratio",
    "daily_reviews",
    "reviews_per_learning_day",
    "accuracy",
]

# Canonical mapping that the rest of the app expects.
CANONICAL_CLUSTER_ORDER = {
    "marathoner": 0,
    "planner": 1,
    "sprinter": 2,
}

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
    for canonical_name, canonical_id in CANONICAL_CLUSTER_ORDER.items():
        mask = normalized == canonical_name
        if not mask.any():
            raise ValueError(f"No samples found for cluster '{canonical_name}' in the CSV.")

        # Count which learned label most often represents this canonical cluster.
        counts = predicted[mask].value_counts()
        learned_label = counts.idxmax()
        label_map[learned_label] = canonical_id

    if len(label_map) != n_clusters:
        missing = [label for label in range(n_clusters) if label not in label_map]
        raise RuntimeError(f"Could not assign canonical ids for cluster labels: {missing}")

    return label_map

# Train KMeans clustering and save artifacts
def train_and_save_clustering(
    csv_path: str = "cluster_dummy_data.csv",
    artifact_path: str = "cluster_artifacts.joblib",
    n_clusters: int = 3,
) -> None:
    """
    Train KMeans + scaler and persist them for use inside the Streamlit app.
    """
    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"Could not find CSV file at {csv_file.resolve()}")

    df = pd.read_csv(csv_file)

    missing_columns = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in the CSV: {missing_columns}")

    if "cluster" not in df.columns:
        raise ValueError("The CSV needs a 'cluster' column with the human-readable label.")

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

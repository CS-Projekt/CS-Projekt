#DISCLAIMER: Some functions in this clusters.py were created with the help of AI, 
# but they were reviewed, commented and verified by us as well as written from scratch by us.

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from joblib import load

# Define path to cluster artifacts
ARTIFACT_PATH = Path("cluster_artifacts.joblib")

# Define cluster keys and profiles
class ClusterKey(str, Enum):
    SPRINTER = "sprinter"
    MARATHONER = "marathoner"
    PLANNER = "planner"  # structured planner

# Map cluster IDs to ClusterKeys
CLUSTER_ID_TO_KEY = {
    0: ClusterKey.MARATHONER,
    1: ClusterKey.PLANNER,
    2: ClusterKey.SPRINTER,
}

# Define cluster profile dataclass
@dataclass
class ClusterProfile:
    key: ClusterKey
    name: str
    description: str
    recommendation: str

# Define cluster profiles
CLUSTERS = {
    ClusterKey.SPRINTER: ClusterProfile(
        key=ClusterKey.SPRINTER,
        name="Sprinter",
        description=(
            "Studies often and in rather short sessions. Many reviews, "
            "shorter intervals, and relatively high study frequency."
        ),
        recommendation="Recommended block length: 20–30 minutes focus, 5 minutes break."
    ),
    ClusterKey.MARATHONER: ClusterProfile(
        key=ClusterKey.MARATHONER,
        name="Marathoner",
        description=(
            "Studies rarely but very intensively. High number of reviews on study days, "
            "long intervals, and high recall rate."
        ),
        recommendation="Recommended block length: 60+ minutes focus, 10–15 minutes break."
    ),
    ClusterKey.PLANNER: ClusterProfile(
        key=ClusterKey.PLANNER,
        name="Structured Planner",
        description=(
            "Studies regularly in medium-sized blocks with solid consistency. "
            "Neither extreme peaks nor long breaks."
        ),
        recommendation="Recommended block length: 35–50 minutes focus, 5–10 minutes break."
    ),
}

# Cache for loaded artifacts (This line was created with the help of AI)
_CACHED_ARTIFACTS: Optional[Dict] = None

# Load cluster artifacts with caching (This function was created with the help of AI)
def _load_cluster_artifacts(force_reload: bool = False) -> Dict:
    """Load scaler + kmeans + feature columns once and cache them."""
    global _CACHED_ARTIFACTS 
    if _CACHED_ARTIFACTS is not None and not force_reload:  # Check cache
        return _CACHED_ARTIFACTS # Return cached artifacts

    if not ARTIFACT_PATH.exists(): # Check if artifact file exists
        raise FileNotFoundError(
            f"Missing cluster artifacts at {ARTIFACT_PATH.resolve()}. "
            "Run train_clustering.train_and_save_clustering() first."
        )

    artifacts = load(ARTIFACT_PATH) # Load artifacts from file
    required_keys = {"scaler", "kmeans", "feature_columns", "label_mapping"}
    if not required_keys.issubset(artifacts.keys()):
        raise ValueError("Artifact file is missing required keys. Please retrain clusters.")

    _CACHED_ARTIFACTS = artifacts # Cache loaded artifacts
    return artifacts

# Prepare feature vector for clustering (This function was created with the help of AI)
def _prepare_feature_vector(features: dict, feature_columns: List[str]) -> np.ndarray:
    """Create aligned feature vector based on the persisted column order."""
    try:
        values = [float(features[column]) for column in feature_columns] # Extract and convert features
    except KeyError as exc:
        missing = exc.args[0] # Identify missing feature
        raise KeyError(f"Missing feature '{missing}' needed for clustering.") from exc
    except (TypeError, ValueError) as exc: # Handle non-numeric features
        raise ValueError("All cluster features must be numeric.") from exc
    return np.array([values]) # Return as 2D array for scaler

# Assign cluster ID based on features (This function was created with the help of AI)
def assign_cluster_id(features: dict, force_reload: bool = False) -> int:
    """
    Return the canonical cluster id (0 marathoner, 1 planner, 2 sprinter)
    for the provided feature dictionary.
    """
    artifacts = _load_cluster_artifacts(force_reload=force_reload) # Load artifacts
    vector = _prepare_feature_vector(features, artifacts["feature_columns"]) # Prepare feature vector
    scaled_vector = artifacts["scaler"].transform(vector) # Scale features
    learned_label = int(artifacts["kmeans"].predict(scaled_vector)[0]) # Predict cluster label
    label_mapping: Dict[int, int] = artifacts["label_mapping"] # Get label mapping
    if learned_label not in label_mapping: # Validate learned label
        raise ValueError(f"Unexpected cluster label {learned_label}. Retrain clustering.")
    return label_mapping[learned_label] # Return mapped cluster ID

# Assign cluster key based on features 
def assign_cluster_from_features(features: dict) -> ClusterKey:
    """Return the ClusterKey for UI descriptions."""
    cluster_id = assign_cluster_id(features)
    return CLUSTER_ID_TO_KEY.get(cluster_id, ClusterKey.PLANNER) # Default to PLANNER if unknown

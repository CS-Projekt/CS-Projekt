# clusters.py
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from joblib import load


ARTIFACT_PATH = Path("cluster_artifacts.joblib")


class ClusterKey(str, Enum):
    SPRINTER = "sprinter"
    MARATHONER = "marathoner"
    PLANNER = "planner"  # structured planner


CLUSTER_ID_TO_KEY = {
    0: ClusterKey.MARATHONER,
    1: ClusterKey.PLANNER,
    2: ClusterKey.SPRINTER,
}


@dataclass
class ClusterProfile:
    key: ClusterKey
    name: str
    description: str
    recommendation: str


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
        recommendation="Recommended block length: 60–75 minutes focus, 10–15 minutes break."
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


_CACHED_ARTIFACTS: Optional[Dict] = None


def _load_cluster_artifacts(force_reload: bool = False) -> Dict:
    """Load scaler + kmeans + feature columns once and cache them."""
    global _CACHED_ARTIFACTS
    if _CACHED_ARTIFACTS is not None and not force_reload:
        return _CACHED_ARTIFACTS

    if not ARTIFACT_PATH.exists():
        raise FileNotFoundError(
            f"Missing cluster artifacts at {ARTIFACT_PATH.resolve()}. "
            "Run train_clustering.train_and_save_clustering() first."
        )

    artifacts = load(ARTIFACT_PATH)
    required_keys = {"scaler", "kmeans", "feature_columns", "label_mapping"}
    if not required_keys.issubset(artifacts.keys()):
        raise ValueError("Artifact file is missing required keys. Please retrain clusters.")

    _CACHED_ARTIFACTS = artifacts
    return artifacts


def _prepare_feature_vector(features: dict, feature_columns: List[str]) -> np.ndarray:
    """Create aligned feature vector based on the persisted column order."""
    try:
        values = [float(features[column]) for column in feature_columns]
    except KeyError as exc:
        missing = exc.args[0]
        raise KeyError(f"Missing feature '{missing}' needed for clustering.") from exc
    except (TypeError, ValueError) as exc:
        raise ValueError("All cluster features must be numeric.") from exc
    return np.array([values])


def assign_cluster_id(features: dict, force_reload: bool = False) -> int:
    """
    Return the canonical cluster id (0 marathoner, 1 planner, 2 sprinter)
    for the provided feature dictionary.
    """
    artifacts = _load_cluster_artifacts(force_reload=force_reload)
    vector = _prepare_feature_vector(features, artifacts["feature_columns"])
    scaled_vector = artifacts["scaler"].transform(vector)
    learned_label = int(artifacts["kmeans"].predict(scaled_vector)[0])
    label_mapping: Dict[int, int] = artifacts["label_mapping"]
    if learned_label not in label_mapping:
        raise ValueError(f"Unexpected cluster label {learned_label}. Retrain clustering.")
    return label_mapping[learned_label]


def assign_cluster_from_features(features: dict) -> ClusterKey:
    """Return the ClusterKey enum for UI descriptions."""
    cluster_id = assign_cluster_id(features)
    return CLUSTER_ID_TO_KEY.get(cluster_id, ClusterKey.PLANNER)

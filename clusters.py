# clusters.py
from dataclasses import dataclass
from enum import Enum


class ClusterKey(str, Enum):
    SPRINTER = "sprinter"
    MARATHONER = "marathoner"
    PLANNER = "planner"  # structured planner


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


def assign_cluster_from_features(features: dict) -> ClusterKey:
    """
    Takes key metrics and assigns a cluster.

    Expected keys:
      - learning_days_ratio
      - reviews_per_learning_day
      - daily_reviews
      - accuracy
    """
    ldr = features.get("learning_days_ratio", 0.0)
    rp_ld = features.get("reviews_per_learning_day", 0.0)
    dr = features.get("daily_reviews", 0.0)
    acc = features.get("accuracy", 0.0)

    # Marathoner – few study days, but high output and good accuracy
    if ldr < 0.2 and rp_ld > 80 and acc >= 0.8:
        return ClusterKey.MARATHONER

    # Sprinter – relatively many study days, moderate intensity
    if ldr >= 0.3 and 20 <= dr <= 80:
        return ClusterKey.SPRINTER

    # Otherwise: structured planner
    return ClusterKey.PLANNER

# clusters.py
from dataclasses import dataclass
from enum import Enum


class ClusterKey(str, Enum):
    SPRINTER = "sprinter"
    MARATHONER = "marathoner"
    PLANNER = "planner"  # strukturierter Planer


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
            "Lernt oft und in eher kurzen Einheiten. Viele Wiederholungen, "
            "kürzere Intervalle und eher hohe Lernfrequenz."
        ),
        recommendation="Empfohlene Blocklänge: 20–30 Minuten Fokus, 5 Minuten Pause."
    ),
    ClusterKey.MARATHONER: ClusterProfile(
        key=ClusterKey.MARATHONER,
        name="Marathoner",
        description=(
            "Lernt selten, dafür sehr intensiv. Hohe Anzahl Wiederholungen an Lerntagen, "
            "lange Intervalle und hohe Erinnerungsquote."
        ),
        recommendation="Empfohlene Blocklänge: 60–75 Minuten Fokus, 10–15 Minuten Pause."
    ),
    ClusterKey.PLANNER: ClusterProfile(
        key=ClusterKey.PLANNER,
        name="Strukturierter Planer",
        description=(
            "Lernt regelmäßig in mittelgroßen Blöcken mit solider Konstanz. "
            "Weder extreme Peaks noch lange Pausen."
        ),
        recommendation="Empfohlene Blocklänge: 35–50 Minuten Fokus, 5–10 Minuten Pause."
    ),
}


def assign_cluster_from_features(features: dict) -> ClusterKey:
    """
    Nimmt Kennzahlen und ordnet einem Cluster zu.

    Erwartete Keys:
      - learning_days_ratio
      - reviews_per_learning_day
      - daily_reviews
      - accuracy
    """
    ldr = features.get("learning_days_ratio", 0.0)
    rp_ld = features.get("reviews_per_learning_day", 0.0)
    dr = features.get("daily_reviews", 0.0)
    acc = features.get("accuracy", 0.0)

    # Marathoner – seltene Lerntage, aber viel Output und gute Quote
    if ldr < 0.2 and rp_ld > 80 and acc >= 0.8:
        return ClusterKey.MARATHONER

    # Sprinter – relativ viele Lerntage, moderate Intensität
    if ldr >= 0.3 and 20 <= dr <= 80:
        return ClusterKey.SPRINTER

    # Sonst: strukturierter Planer
    return ClusterKey.PLANNER

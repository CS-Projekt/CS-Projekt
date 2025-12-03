import os
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import requests


DEFAULT_API_BASE = "internal"
MODEL_PATH = Path("learning_models.pkl")
_LOCAL_MODELS = None


class PredictionAPIError(RuntimeError):
    """Raised when the ML API cannot be reached or returns an error."""


def _get_api_base() -> str:
    base = os.getenv("ML_API_BASE", DEFAULT_API_BASE).strip()
    return base or DEFAULT_API_BASE


def _is_http_base(base: str) -> bool:
    return base.startswith("http://") or base.startswith("https://")


def _request(method: str, path: str, timeout: float = 10, **kwargs):
    url = f"{_get_api_base()}{path}"
    try:
        response = requests.request(method=method, url=url, timeout=timeout, **kwargs)
        response.raise_for_status()
        return response
    except requests.RequestException as exc:
        raise PredictionAPIError(f"Request to ML API failed ({url}): {exc}") from exc


def predict_session_via_api(features: Dict[str, Any], desired_total_duration: Optional[int] = None):
    base = _get_api_base()
    if not _is_http_base(base):
        return _local_predict_session(features, desired_total_duration)

    params = {}
    if desired_total_duration is not None:
        params["desired_total_duration"] = desired_total_duration
    response = _request(
        "POST",
        "/predict-session",
        json=features,
        params=params or None,
        timeout=10,
    )
    return response.json()


def predict_anki_via_api(pdf_path: str):
    base = _get_api_base()
    if not _is_http_base(base):
        raise PredictionAPIError("predict_anki_via_api requires a running HTTP API.")
    with open(pdf_path, "rb") as f:
        files = {"file": ("anki.pdf", f, "application/pdf")}
        response = _request(
            "POST",
            "/predict-anki",
            files=files,
            timeout=30,
        )
    return response.json()


def is_api_available(timeout: float = 0.5) -> bool:
    """
    Return True if the ML API responds to /healthz. When the client is
    configured to use the internal models, the API is considered available.
    """
    base = _get_api_base()
    if not _is_http_base(base):
        try:
            _load_local_models()
            return True
        except FileNotFoundError:
            return False

    try:
        _request("GET", "/healthz", timeout=timeout)
        return True
    except PredictionAPIError:
        return False


def _load_local_models():
    global _LOCAL_MODELS
    if _LOCAL_MODELS is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Could not find local models at {MODEL_PATH.resolve()}. Run train_model.py first."
            )
        with MODEL_PATH.open("rb") as f:
            _LOCAL_MODELS = pickle.load(f)
    return _LOCAL_MODELS


def _local_predict_session(features: Dict[str, Any], desired_total_duration: Optional[int]) -> Dict[str, Any]:
    models = _load_local_models()
    df = pd.DataFrame(
        [
            {
                "total_session_duration": features["total_session_duration"],
                "time_morning": features.get("time_morning", 0),
                "time_afternoon": features.get("time_afternoon", 0),
                "time_evening": features.get("time_evening", 0),
                "time_night": features.get("time_night", 0),
                "concentration_baseline": features["concentration_baseline"],
                "days_since_last_session": features["days_since_last_session"],
                "previous_session_rating": features["previous_session_rating"],
                "cluster_id": features.get("cluster_id", 1),
            }
        ]
    )
    scaler = models["scaler"]
    X = scaler.transform(df)

    work_pred = int(round(models["work_duration"].predict(X)[0]))
    break_pred = int(round(models["break_duration"].predict(X)[0]))
    next_pred = float(models["next_session"].predict(X)[0])

    work_pred = max(15, min(45, work_pred))
    break_pred = max(5, min(15, break_pred))

    total_duration = (
        desired_total_duration
        if desired_total_duration is not None
        else features["total_session_duration"]
    )
    cycle_duration = work_pred + break_pred
    pred_blocks = max(1, int((total_duration + break_pred) / cycle_duration))

    schedule = []
    total_calculated = 0
    for block in range(pred_blocks):
        schedule.append({"type": "Study", "duration": work_pred, "block": block + 1})
        total_calculated += work_pred
        if block < pred_blocks - 1:
            schedule.append({"type": "Break", "duration": break_pred, "block": block + 1})
            total_calculated += break_pred

    return {
        "work_duration": work_pred,
        "break_duration": break_pred,
        "next_session_hours": next_pred,
        "blocks": pred_blocks,
        "schedule": schedule,
        "total_calculated_duration": total_calculated,
    }

import requests
from typing import Any, Dict

API_BASE = "http://localhost:8000"

def predict_session_via_api(features: Dict[str, Any], desired_total_duration: int = None):
    params = {}
    if desired_total_duration is not None:
        params['desired_total_duration'] = desired_total_duration
    resp = requests.post(f"{API_BASE}/predict-session", json=features, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()

def predict_anki_via_api(pdf_path: str):
    with open(pdf_path, "rb") as f:
        files = {"file": ("anki.pdf", f, "application/pdf")}
        resp = requests.post(f"{API_BASE}/predict-anki", files=files, timeout=30)
    resp.raise_for_status()
    return resp.json()
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
import math
import uvicorn
from typing import Optional, List
from anki_utils import extract_features_from_anki_pdf
import clusters
from clusters import assign_cluster_id, CLUSTER_ID_TO_KEY

app = FastAPI(title="Study Plan ML API")

# Load models at startup
MODEL_PATH = "learning_models.pkl"
try:
    with open(MODEL_PATH, "rb") as f:
        MODELS = pickle.load(f)
except FileNotFoundError:
    MODELS = None

class SessionFeatures(BaseModel):
    total_session_duration: int
    time_morning: int = 0
    time_afternoon: int = 0
    time_evening: int = 0
    time_night: int = 0
    concentration_baseline: float
    days_since_last_session: int
    previous_session_rating: float
    cluster_id: Optional[int] = None

class PredictResponse(BaseModel):
    work_duration: int
    break_duration: int
    next_session_hours: float
    blocks: int
    schedule: List[dict]
    total_calculated_duration: int

@app.on_event("startup")
def check_models():
    if MODELS is None:
        # let startup succeed but warn endpoints
        app.state.models_loaded = False
    else:
        app.state.models_loaded = True

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def resolve_cluster_id(value: Optional[int]) -> int:
    if value is None:
        return 1  # structured planner as neutral default
    if value not in (0, 1, 2):
        raise HTTPException(status_code=400, detail="cluster_id must be 0, 1, or 2.")
    return value

@app.post("/predict-session", response_model=PredictResponse)
def predict_session(features: SessionFeatures, desired_total_duration: Optional[int] = None):
    if not app.state.models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded. Run train_model.py first to create learning_models.pkl")

    # Build dataframe the same way app.py does
    df = pd.DataFrame([{
        'total_session_duration': features.total_session_duration,
        'time_morning': features.time_morning,
        'time_afternoon': features.time_afternoon,
        'time_evening': features.time_evening,
        'time_night': features.time_night,
        'concentration_baseline': features.concentration_baseline,
        'days_since_last_session': features.days_since_last_session,
        'previous_session_rating': features.previous_session_rating,
        'cluster_id': resolve_cluster_id(features.cluster_id),
    }])

    scaler = MODELS['scaler']
    X = scaler.transform(df)

    work_pred = int(round(MODELS['work_duration'].predict(X)[0]))
    break_pred = int(round(MODELS['break_duration'].predict(X)[0]))
    next_pred = float(MODELS['next_session'].predict(X)[0])

    # Apply same clamping logic as app.py
    work_pred = clamp(work_pred, 15, 45)
    break_pred = clamp(break_pred, 5, 15)

    # Compute number of blocks / schedule based on desired_total_duration or total_session_duration
    total_duration = desired_total_duration if desired_total_duration is not None else features.total_session_duration
    cycle = work_pred + break_pred
    blocks = max(1, int((total_duration + break_pred) / cycle))

    schedule = []
    total_calc = 0
    for block in range(blocks):
        schedule.append({'type': 'Study', 'duration': work_pred, 'block': block + 1})
        total_calc += work_pred
        if block < blocks - 1:
            schedule.append({'type': 'Break', 'duration': break_pred, 'block': block + 1})
            total_calc += break_pred

    return {
        'work_duration': int(work_pred),
        'break_duration': int(break_pred),
        'next_session_hours': float(next_pred),
        'blocks': blocks,
        'schedule': schedule,
        'total_calculated_duration': int(total_calc)
    }

@app.post("/predict-anki")
async def predict_anki(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files supported")

    # read into memory (anki_utils expects file-like object or path)
    contents = await file.read()
    try:
        # anki_utils.extract_features_from_anki_pdf accepts bytes or file-like
        features = extract_features_from_anki_pdf(contents)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error extracting features: {e}")

    # assign cluster
    cluster_id = assign_cluster_id(features)
    cluster_key = CLUSTER_ID_TO_KEY.get(cluster_id, clusters.ClusterKey.PLANNER)
    profile = clusters.CLUSTERS[cluster_key]

    return {
        "features": features,
        "cluster_key": str(cluster_key),
        "cluster_id": cluster_id,
        "profile": {
            "name": profile.name,
            "description": profile.description,
            "recommendation": profile.recommendation
        }
    }


@app.get("/healthz")
def healthcheck():
    """Lightweight endpoint so clients can detect whether the API is running."""
    return {"status": "ok"}

if __name__ == "__main__":
    # For local dev only; use uvicorn in production
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)

from fastapi.responses import RedirectResponse

@app.get("/", include_in_schema=False)
def docs_redirect():
    return RedirectResponse(url="/docs")

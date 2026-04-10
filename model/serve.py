# model/serve.py — complete clean version

# ── Imports ───────────────────────────────────────────────────────────
import os
import json
import time
import redis
import mlflow
import joblib
import numpy as np
import pandas as pd
import warnings
from datetime import datetime
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from mlflow.tracking import MlflowClient
import uvicorn

warnings.filterwarnings("ignore", message="X does not have valid feature names")
load_dotenv("../.env")

# ── Config ────────────────────────────────────────────────────────────
MLFLOW_URI    = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME    = "fraud-classifier"
REDIS_HOST    = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT    = int(os.getenv("REDIS_PORT", "6379"))
FEATURE_NAMES = [f"V{i}" for i in range(1, 29)] + ["Amount"]

# ── Global variables ──────────────────────────────────────────────────
model        = None
scaler       = None
redis_client = None

# ── Lifespan (startup + shutdown) ─────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # everything before yield = startup
    # everything after yield  = shutdown
    global model, scaler, redis_client

    print("Starting up...")

    # Connect to Redis
    redis_client = redis.Redis(
        host=REDIS_HOST, port=REDIS_PORT, decode_responses=True
    )
    redis_client.ping()
    print("Redis connected")

    # Load scaler
    scaler = joblib.load("scaler.pkl")
    print("Scaler loaded")

    # Load model from MLflow
    mlflow.set_tracking_uri(MLFLOW_URI)

    client = MlflowClient()
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    # search_model_versions returns all registered versions newest first
    # if empty → model was never registered → run train.py first

    if not versions:
        raise RuntimeError(
            f"No model named '{MODEL_NAME}' found in MLflow. "
            f"Run python train.py first."
        )

    latest_version = versions[0].version
    # versions[0] = most recently registered version object
    # .version = the version number as a string e.g. "1"

    model_uri = f"models:/{MODEL_NAME}/{latest_version}"
    # explicit version number works in all MLflow versions
    # safer than "latest" alias which is deprecated in MLflow 3.x

    model = mlflow.xgboost.load_model(model_uri)
    print(f"Model loaded — version {latest_version}")
    print("Server ready\n")

    yield
    # server runs here — handles all requests until Ctrl+C

    # shutdown
    redis_client.close()
    print("Connections closed.")

# ── Create app — ONCE, with lifespan attached ─────────────────────────
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time fraud prediction with drift monitoring",
    version="1.0.0",
    lifespan=lifespan
    # lifespan= is how FastAPI 0.93+ handles startup/shutdown
    # replaces the deprecated @app.on_event("startup")
)

# ── Request / Response schemas ────────────────────────────────────────
class PredictRequest(BaseModel):
    transaction_id: int
    # pydantic validates this — sending a string causes automatic 422 error

class PredictResponse(BaseModel):
    transaction_id:    int
    fraud_probability: float
    prediction:        int
    threshold_used:    float
    model_version:     str
    latency_ms:        float

# ── Health check ──────────────────────────────────────────────────────
@app.get("/health")
async def health():
    # GET /health — used by load balancers to check if service is alive
    # always returns 200 OK if server is up
    return {
        "status":    "healthy",
        "model":     MODEL_NAME,
        "timestamp": datetime.utcnow().isoformat()
    }

# ── Prediction endpoint ───────────────────────────────────────────────
@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    start_time = time.time()

    # Fetch features from Redis
    redis_key    = f"transaction:{request.transaction_id}"
    raw_features = redis_client.hgetall(redis_key)
    # hgetall returns {} if key doesn't exist

    if not raw_features:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Transaction {request.transaction_id} not found in Redis. "
                f"Make sure consumer.py is running and has processed this ID."
            )
        )

    # Build feature DataFrame — same column order as training
    feature_values = [float(raw_features[name]) for name in FEATURE_NAMES]
    # Redis stores everything as strings → float() converts "0.34" → 0.34
    # order must match FEATURE_NAMES exactly — model is order-sensitive

    features_df = pd.DataFrame([feature_values], columns=FEATURE_NAMES)
    # already scaled by consumer.py — do NOT scale again here

    # Run inference
    fraud_prob = float(model.predict_proba(features_df)[0][1])
    # predict_proba → [[prob_legit, prob_fraud]]
    # [0][1] = first row, second column = fraud probability

    threshold  = 0.5
    prediction = int(fraud_prob >= threshold)
    # True/False → 1/0

    latency_ms = round((time.time() - start_time) * 1000, 2)

    # Log to .jsonl file (drift detector reads this in Step 5)
    log_entry = {
        "transaction_id":    request.transaction_id,
        "fraud_probability": fraud_prob,
        "prediction":        prediction,
        "timestamp":         datetime.utcnow().isoformat(),
        "latency_ms":        latency_ms,
        "model_version":     MODEL_NAME,
    }
    with open("predictions_log.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")
    # "a" = append mode — never overwrites existing entries
    # .jsonl = one JSON object per line — easy to stream and parse

    return PredictResponse(
        transaction_id    = request.transaction_id,
        fraud_probability = round(fraud_prob, 4),
        prediction        = prediction,
        threshold_used    = threshold,
        model_version     = MODEL_NAME,
        latency_ms        = latency_ms,
    )

# ── Entry point ───────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "serve:app",
        host="0.0.0.0",
        port=8000,
        reload=True
        # reload=True = restart on file save — dev only, never production
    )
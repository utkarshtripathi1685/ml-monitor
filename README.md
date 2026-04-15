# ML Monitor — Real-Time ML System Monitor & Auto-Retraining Pipeline

A production-grade ML monitoring system that detects data drift in real-time and triggers automatic model retraining. Built to simulate how fraud detection systems work at scale.

## Architecture
```
CSV Data → Kafka Producer → Kafka Topic → Consumer → Redis Feature Store
↓
FastAPI Serving Layer
(MLflow model registry)
↓
predictions_log.jsonl
↓
Drift Detector (PSI + KS)
↓
drift_log.jsonl
```

## Tech Stack

- **Streaming**: Apache Kafka + Zookeeper (Confluent)
- **Feature Store**: Redis
- **ML**: XGBoost, scikit-learn, MLflow (experiment tracking + model registry)
- **Serving**: FastAPI + Uvicorn
- **Drift Detection**: PSI (Population Stability Index) + KS Test (scipy)
- **Infrastructure**: Docker + docker-compose
- **Language**: Python 3.11

## What It Does

1. **Streams** synthetic fraud transactions through Kafka at ~20 events/second
2. **Transforms** raw features using a fitted StandardScaler and stores them in Redis
3. **Serves** real-time fraud predictions via a REST API (sub-5ms latency)
4. **Monitors** live feature distributions against training baseline every 60 seconds
5. **Detects** drift using PSI > 0.2 threshold and KS test p-value < 0.05
6. **Alerts** when significant drift is detected across any feature

## Key Engineering Decisions

**Why Kafka instead of direct API calls?**
Decouples the producer and consumer — if the consumer crashes, no data is lost. Kafka retains messages so the consumer resumes from exactly where it left off.

**Why load the model once at startup?**
Loading an XGBoost model takes ~2-3 seconds. Loading per-request would make every prediction unusably slow. FastAPI's lifespan context manager loads it once into memory.

**Why PSI + KS test together?**
PSI catches overall distribution shift (industry standard in banking/finance). KS test provides statistical significance — p < 0.05 means the shift is unlikely to be random noise. Using both reduces false positives.

**Why store scaled features in Redis instead of raw?**
The serving layer needs features in the same scale as training. Scaling once in the consumer (rather than on every prediction request) reduces serving latency and ensures consistency.

## Running Locally

**Prerequisites**: Docker Desktop, Python 3.11

```bash
# 1. Start infrastructure
docker-compose up -d

# 2. Train baseline model
cd model && python train.py

# 3. Start feature pipeline (separate terminals)
cd ingestor && python consumer.py
cd ingestor && python producer.py

# 4. Start prediction API
cd model && python serve.py

# 5. Test predictions
cd model && python test_api.py

# 6. Run drift detector
cd monitor && python detector.py

# 7. Inject drift to test detection
cd monitor && python simulate_drift.py
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service health check |
| `/predict` | POST | Returns fraud probability for a transaction |
| `/docs` | GET | Auto-generated Swagger UI |

**Sample prediction request:**
```json
POST /predict
{"transaction_id": 500}
```

**Sample response:**
```json
{
  "transaction_id": 500,
  "fraud_probability": 0.8934,
  "prediction": 1,
  "threshold_used": 0.5,
  "model_version": "fraud-classifier",
  "latency_ms": 4.21
}
```

## Drift Detection

The detector compares live feature distributions against training baseline using:

- **PSI < 0.1**: Stable, no action needed
- **PSI 0.1–0.2**: Minor shift, monitor closely  
- **PSI > 0.2**: Significant drift → retrain triggered
- **KS p-value < 0.05**: Statistically significant distribution change

## Project Structure
```
ml-monitor/
├── model/          # XGBoost training, MLflow logging, FastAPI server
├── ingestor/       # Kafka producer + Redis consumer
├── monitor/        # Drift detection engine + drift simulator
├── dashboard/      # Next.js live dashboard (coming soon)
├── infra/          # Terraform AWS deployment (coming soon)
└── docker-compose.yml
```

## What's Next

- [ ] Airflow DAG for automated retraining on drift detection
- [ ] AWS deployment (Lambda + DynamoDB + SNS alerts)
- [ ] Next.js real-time dashboard with WebSocket updates
- [ ] Terraform IaC for reproducible cloud infrastructure
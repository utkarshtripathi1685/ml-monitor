# model/train.py — correct order of operations

# ── 1. All imports at the very top ───────────────────────────────────
import pandas as pd
import numpy as np
import mlflow
import mlflow.xgboost
import joblib                          # ← must be here at the top with other imports
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
import os

# ── 2. Config ─────────────────────────────────────────────────────────
MLFLOW_URI      = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "fraud-detection"
MODEL_NAME      = "fraud-classifier"

# ── 3. Connect to MLflow ──────────────────────────────────────────────
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# ── 4. Load data ──────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv("creditcard.csv")
print(f"Shape: {df.shape}")
print(f"Fraud rate: {df['Class'].mean():.2%}")

# ── 5. Split features and label ───────────────────────────────────────
X = df.drop(columns=["Class"])
y = df["Class"]

# ── 6. Train/test split ───────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")

# ── 7. Scale features ─────────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit + transform on train
X_test_scaled  = scaler.transform(X_test)         # transform only on test

# ── 8. Save scaler right after fitting ───────────────────────────────
joblib.dump(scaler, "scaler.pkl")
# save immediately after fit_transform so it's never forgotten
# consumer.py loads this exact file to scale live transactions
print("Scaler saved to scaler.pkl")

# ── 9. Define params ──────────────────────────────────────────────────
params = {
    "n_estimators":       100,
    "max_depth":          6,
    "learning_rate":      0.1,
    "scale_pos_weight":   49,
    "random_state":       42,
    "eval_metric":        "auc",
}

# ── 10. Train + log to MLflow ─────────────────────────────────────────
with mlflow.start_run(run_name="xgboost-baseline") as run:
    print(f"MLflow run ID: {run.info.run_id}")

    mlflow.log_params(params)

    print("Training model...")
    model = XGBClassifier(**params)
    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_test_scaled, y_test)],
        verbose=50
    )

    # Evaluate
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    y_pred       = model.predict(X_test_scaled)
    auc          = roc_auc_score(y_test, y_pred_proba)
    f1           = f1_score(y_test, y_pred, average="binary")

    print(f"\nAUC: {auc:.4f}")
    print(f"F1:  {f1:.4f}")
    print(classification_report(y_test, y_pred))

    mlflow.log_metric("auc", auc)
    mlflow.log_metric("f1",  f1)

    # Log model
    mlflow.xgboost.log_model(
        xgb_model=model,
        name="model",
        registered_model_name=MODEL_NAME,
    )

    # Save and log baseline stats for drift detection
    baseline_stats = pd.DataFrame(X_train_scaled).describe()
    baseline_stats.to_csv("baseline_stats.csv")
    mlflow.log_artifact("baseline_stats.csv")

    print(f"\nModel registered as '{MODEL_NAME}' in MLflow")
    print(f"Open http://localhost:5000 to see your run")
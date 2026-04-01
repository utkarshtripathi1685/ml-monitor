# model/train.py
# This is the heart of Step 2.
# It trains an XGBoost model and tracks everything in MLflow.

# ── Imports ──────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import mlflow                           # experiment tracking
import mlflow.xgboost                   # mlflow's built-in XGBoost logger
from xgboost import XGBClassifier       # the model we're training
from sklearn.model_selection import train_test_split
# train_test_split = splits data into training set and test set
from sklearn.metrics import (
    roc_auc_score,    # AUC = Area Under Curve, best metric for imbalanced data
    f1_score,         # balance between precision and recall
    classification_report  # full breakdown: precision, recall, F1 per class
)
from sklearn.preprocessing import StandardScaler
# StandardScaler = normalises features to mean=0, std=1
# XGBoost doesn't strictly need this but it helps convergence
import os             # to read environment variables

# ── Config ───────────────────────────────────────────────────────────
# os.getenv reads from environment variables (our .env file)
# the second argument is the default if the variable isn't set
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "fraud-detection"
MODEL_NAME = "fraud-classifier"   # name in the MLflow model registry

# ── Connect to MLflow ────────────────────────────────────────────────
mlflow.set_tracking_uri(MLFLOW_URI)
# tells MLflow where the server is
# all subsequent mlflow.log_* calls go to this server

mlflow.set_experiment(EXPERIMENT_NAME)
# experiments are like folders in MLflow
# if "fraud-detection" doesn't exist, MLflow creates it automatically

# ── Load data ────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv("creditcard.csv")
# pd.read_csv reads a CSV file into a DataFrame
# df is convention for "dataframe" — just a variable name

print(f"Shape: {df.shape}")           # (100000, 30) — rows × columns
print(f"Fraud rate: {df['Class'].mean():.2%}")
# df['Class'] selects the "Class" column
# .mean() on a 0/1 column = fraction of 1s = fraud rate
# :.2% = format as percentage with 2 decimal places

# ── Split features and label ─────────────────────────────────────────
X = df.drop(columns=["Class"])
# drop the label column — everything else is a feature
# X = input to the model (features)

y = df["Class"]
# y = what we want to predict (0 or 1)
# convention: X for features, y for labels

# ── Train/test split ─────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,       # 20% of data goes to test set, 80% for training
    random_state=42,     # fixed seed = reproducible split
    stratify=y           # keep same fraud ratio in both splits
    # without stratify, you might get 0 fraud cases in test set by bad luck
)

print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")

# ── Scale features ───────────────────────────────────────────────────
scaler = StandardScaler()
# StandardScaler transforms each feature to: (value - mean) / std_dev
# result: every feature has mean=0 and std=1
# why? so no single feature dominates just because it has bigger numbers
# e.g. "Amount" ranges 0–10000, "V1" ranges -3 to 3 — unfair without scaling

X_train_scaled = scaler.fit_transform(X_train)
# fit = learn the mean and std from training data
# transform = apply the scaling
# IMPORTANT: fit only on training data — never on test data
# fitting on test data = "data leakage" = your model cheats

X_test_scaled = scaler.transform(X_test)
# transform only (no fit) — use the SAME mean/std learned from training
# this simulates real-world: you won't know future data's distribution

# ── Define model parameters ──────────────────────────────────────────
params = {
    "n_estimators": 100,
    # number of trees in the ensemble
    # more trees = better accuracy but slower training

    "max_depth": 6,
    # how deep each tree can grow
    # deeper = captures more patterns but risks overfitting

    "learning_rate": 0.1,
    # how much each new tree corrects the previous ones
    # lower = more conservative, needs more trees

    "scale_pos_weight": 49,
    # handles class imbalance: ratio of negatives to positives
    # 98% legit / 2% fraud ≈ 49:1
    # tells XGBoost to penalise missing frauds 49× more

    "random_state": 42,
    "eval_metric": "auc",
    # metric used internally during training to evaluate each tree
    
}

# ── Train inside an MLflow run ───────────────────────────────────────
# "with mlflow.start_run()" is a context manager
# everything inside this block is tracked as one experiment "run"
# when the block exits, the run is automatically closed and saved

with mlflow.start_run(run_name="xgboost-baseline") as run:
    print(f"MLflow run ID: {run.info.run_id}")
    # every run gets a unique ID — you can reference it later

    # ── Log parameters ───────────────────────────────────────────────
    mlflow.log_params(params)
    # saves all our hyperparameters to MLflow
    # you'll see these in the MLflow UI under "Parameters"
    # this is why you can always reproduce a run — params are recorded

    # ── Train the model ──────────────────────────────────────────────
    print("Training model...")
    model = XGBClassifier(**params)
    # **params unpacks the dict as keyword arguments
    # same as: XGBClassifier(n_estimators=100, max_depth=6, ...)

    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_test_scaled, y_test)],
        # eval_set = data to evaluate on after each tree
        # XGBoost prints loss after each round — useful to watch
        verbose=50
        # print evaluation every 50 rounds (not every single tree)
    )

    # ── Evaluate ─────────────────────────────────────────────────────
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    # predict_proba returns [[prob_class0, prob_class1], ...]
    # [:, 1] selects the second column = probability of fraud
    # we want probabilities, not just 0/1, for AUC calculation

    y_pred = model.predict(X_test_scaled)
    # predict returns hard labels: 0 or 1

    auc = roc_auc_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred, average="binary")
    # average="binary" = F1 for the positive class (fraud)

    print(f"\nAUC: {auc:.4f}")
    print(f"F1:  {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # ── Log metrics ──────────────────────────────────────────────────
    mlflow.log_metric("auc", auc)
    mlflow.log_metric("f1", f1)
    # metrics appear in MLflow UI — you can compare across runs
    # if you retrain tomorrow with different params, you'll see both runs

    # ── Log the model ────────────────────────────────────────────────
    mlflow.xgboost.log_model(
    xgb_model=model,
    # xgb_model = keyword argument for the model object in MLflow 3.x
    name="model",
    # "name" replaces "artifact_path" in MLflow 3.x
    registered_model_name=MODEL_NAME,
    # this stays the same — registers it in the model registry
    )
    # MLflow saves the model + a conda.yaml + requirements.txt automatically
    # so anyone can reproduce the exact environment

    # ── Save training baseline stats ─────────────────────────────────
    # CRITICAL: we save the training data's feature distributions now.
    # In Step 3 (drift detection), we'll compare live data against these.
    # This is the "baseline" — what normal data looks like.
    baseline_stats = pd.DataFrame(X_train_scaled).describe()
    # .describe() = count, mean, std, min, 25%, 50%, 75%, max per column
    baseline_stats.to_csv("baseline_stats.csv")
    mlflow.log_artifact("baseline_stats.csv")
    # log_artifact saves any file to MLflow storage
    # our drift detector will download this file later to compare against

    print(f"\nModel registered as '{MODEL_NAME}' in MLflow")
    print(f"Open http://localhost:5000 to see your run")
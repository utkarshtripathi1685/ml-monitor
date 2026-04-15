# airflow/dags/retrain_pipeline.py
# Automated retraining DAG.
# Triggered when drift is detected.
# Tasks: check_drift → retrain → evaluate → promote_if_better

from airflow import DAG
# DAG = Directed Acyclic Graph — the workflow definition
# "Directed" = tasks have a defined order
# "Acyclic" = no circular dependencies (A→B→A is not allowed)
# "Graph" = tasks are nodes, dependencies are edges

from airflow.operators.python import PythonOperator
# PythonOperator = run a Python function as a task
# other operators exist: BashOperator, DockerOperator, etc.

from datetime import datetime, timedelta
import os
import json
import mlflow
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score
from xgboost import XGBClassifier
from mlflow.tracking import MlflowClient
import joblib

# ── DAG default arguments ─────────────────────────────────────────────
default_args = {
    "owner": "ml-team",
    # who owns this DAG — shown in UI

    "retries": 1,
    # if a task fails, retry once before marking it failed

    "retry_delay": timedelta(minutes=2),
    # wait 2 minutes between retries
    # gives transient issues (network blip, resource spike) time to resolve

    "email_on_failure": False,
    # don't try to send email (we haven't configured SMTP)
}

# ── Config ────────────────────────────────────────────────────────────
MLFLOW_URI    = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
# note: inside Docker, we reach MLflow as "mlflow:5000" not "localhost:5000"
# "mlflow" is the service name in docker-compose — Docker DNS resolves it

MODEL_NAME    = "fraud-classifier"
DATA_PATH     = "/opt/airflow/model/creditcard.csv"
DRIFT_LOG     = "/opt/airflow/model/drift_log.jsonl"
FEATURE_NAMES = [f"V{i}" for i in range(1, 29)] + ["Amount"]
PSI_THRESHOLD = 0.2

# ── Task 1: Check if drift was detected ──────────────────────────────
def check_drift(**context):
    """
    Reads the latest entry in drift_log.jsonl.
    If drift detected → proceed with retraining.
    If stable → skip remaining tasks.

    **context = Airflow passes metadata about the current run here
    context["ti"] = TaskInstance — used to push/pull data between tasks
    """
    ti = context["ti"]
    # ti = TaskInstance object
    # ti.xcom_push = store a value that other tasks can read
    # ti.xcom_pull = read a value stored by another task
    # XCom = "cross-communication" — Airflow's way to pass data between tasks

    if not os.path.exists(DRIFT_LOG):
        print("No drift log found. Skipping retraining.")
        ti.xcom_push(key="drift_detected", value=False)
        return

    # Read the most recent drift check result
    with open(DRIFT_LOG, "r") as f:
        lines = f.readlines()

    if not lines:
        ti.xcom_push(key="drift_detected", value=False)
        return

    latest = json.loads(lines[-1].strip())
    # lines[-1] = last line = most recent drift check
    # json.loads = string → dict

    drift_detected = latest.get("overall_drift", False)
    max_psi        = latest.get("max_psi", 0)
    drifted_features = latest.get("drifted_features", [])

    print(f"Latest drift check: drift={drift_detected}, max_psi={max_psi}")
    print(f"Drifted features: {drifted_features}")

    ti.xcom_push(key="drift_detected",   value=drift_detected)
    ti.xcom_push(key="max_psi",          value=max_psi)
    ti.xcom_push(key="drifted_features", value=drifted_features)
    # push these values so downstream tasks can read them

    if drift_detected:
        print(f"DRIFT DETECTED — triggering retraining pipeline")
    else:
        print("No drift — retraining not needed")

# ── Task 2: Retrain the model ─────────────────────────────────────────
def retrain_model(**context):
    """
    Retrains the XGBoost model on fresh data.
    Logs new run to MLflow but does NOT promote yet.
    Pushes new model version number for the next task.
    """
    ti = context["ti"]

    drift_detected = ti.xcom_pull(key="drift_detected", task_ids="check_drift")
    # xcom_pull = read value pushed by another task
    # task_ids = which task pushed it

    if not drift_detected:
        print("No drift detected upstream. Skipping retraining.")
        return

    print("Starting retraining...")

    # ── Load data ─────────────────────────────────────────────────────
    df = pd.read_csv(DATA_PATH)
    X  = df.drop(columns=["Class"])
    y  = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── Scale ─────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # Save updated scaler
    joblib.dump(scaler, "/opt/airflow/model/scaler.pkl")

    # ── Train with slightly adjusted params ───────────────────────────
    # In a real system you'd run hyperparameter search here
    # For now we adjust scale_pos_weight based on current class balance
    fraud_rate = y_train.mean()
    scale_pos_weight = (1 - fraud_rate) / fraud_rate
    # dynamically recalculate class weight ratio from fresh data
    # if fraud patterns have shifted, this ratio may have changed

    params = {
        "n_estimators":     150,
        # slightly more trees than baseline (100) — more capacity
        "max_depth":        6,
        "learning_rate":    0.1,
        "scale_pos_weight": scale_pos_weight,
        "random_state":     42,
        "eval_metric":      "auc",
    }

    # ── Log to MLflow as a new run ────────────────────────────────────
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("fraud-detection")

    with mlflow.start_run(run_name="retrain-on-drift") as run:
        mlflow.log_params(params)
        mlflow.log_param("trigger", "drift_detection")
        mlflow.log_param("scale_pos_weight_dynamic", scale_pos_weight)
        # log why this run was triggered — useful for audit trail

        model = XGBClassifier(**params)
        model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_test_scaled, y_test)],
            verbose=False
            # verbose=False = don't print per-tree progress
        )

        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        y_pred       = model.predict(X_test_scaled)
        auc = roc_auc_score(y_test, y_pred_proba)
        f1  = f1_score(y_test, y_pred, average="binary")

        mlflow.log_metric("auc", auc)
        mlflow.log_metric("f1",  f1)

        # Register as new version — NOT promoted yet
        mlflow.xgboost.log_model(
            xgb_model=model,
            name="model",
            registered_model_name=MODEL_NAME,
        )

        print(f"New model trained — AUC: {auc:.4f}, F1: {f1:.4f}")
        print(f"MLflow run: {run.info.run_id}")

        # Push metrics for next task to compare
        ti.xcom_push(key="new_auc", value=auc)
        ti.xcom_push(key="new_f1",  value=f1)
        ti.xcom_push(key="run_id",  value=run.info.run_id)

# ── Task 3: Evaluate — compare new vs champion ────────────────────────
def evaluate_model(**context):
    """
    Compares the newly trained model against the current champion.
    Pushes a boolean: should_promote.
    """
    ti = context["ti"]

    drift_detected = ti.xcom_pull(key="drift_detected", task_ids="check_drift")
    if not drift_detected:
        return

    new_auc = ti.xcom_pull(key="new_auc", task_ids="retrain_model")
    new_f1  = ti.xcom_pull(key="new_f1",  task_ids="retrain_model")

    if new_auc is None:
        print("No new model metrics found. Skipping promotion.")
        ti.xcom_push(key="should_promote", value=False)
        return

    # Get current champion's metrics from MLflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = MlflowClient()

    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    # returns all versions newest first
    # versions[0] = newest (just trained), versions[1] = previous champion

    if len(versions) < 2:
        # only one version exists — it's both new and champion
        # always promote in this case
        print("Only one model version exists. Promoting by default.")
        ti.xcom_push(key="should_promote", value=True)
        return

    champion_version = versions[1]
    # versions[0] = newest (just trained)
    # versions[1] = previous champion

    # Get the champion's run metrics
    champion_run = mlflow.get_run(champion_version.run_id)
    # mlflow.get_run = fetches run metadata including logged metrics
    champion_auc = champion_run.data.metrics.get("auc", 0)

    print(f"Champion AUC: {champion_auc:.4f}")
    print(f"New model AUC: {new_auc:.4f}")

    # Promote only if new model is meaningfully better
    # require at least 0.5% improvement to avoid promoting noise
    should_promote = new_auc > champion_auc + 0.005
    # 0.005 = minimum improvement threshold
    # prevents constantly swapping models for tiny random fluctuations

    ti.xcom_push(key="should_promote",  value=should_promote)
    ti.xcom_push(key="champion_auc",    value=champion_auc)

    if should_promote:
        print(f"New model is better by {(new_auc - champion_auc):.4f} AUC. Will promote.")
    else:
        print(f"New model not better enough. Keeping champion.")

# ── Task 4: Promote if better ─────────────────────────────────────────
def promote_if_better(**context):
    """
    If should_promote=True, tags the new version as 'champion' in MLflow.
    The serving layer will pick it up on next restart.
    """
    ti = context["ti"]

    drift_detected = ti.xcom_pull(key="drift_detected", task_ids="check_drift")
    if not drift_detected:
        print("No drift — nothing to promote.")
        return

    should_promote = ti.xcom_pull(key="should_promote", task_ids="evaluate_model")

    if not should_promote:
        print("New model not promoted. Champion unchanged.")
        return

    mlflow.set_tracking_uri(MLFLOW_URI)
    client = MlflowClient()

    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    new_version = versions[0].version
    # versions[0] = most recently registered = just trained

    # Tag the new version as champion
    client.set_model_version_tag(
        name=MODEL_NAME,
        version=new_version,
        key="champion",
        value="true"
        # tags are key-value metadata on model versions
        # the serving layer can filter by this tag
    )

    print(f"Model version {new_version} promoted to champion!")
    print(f"Serving layer will use this model on next load.")

    # Log the promotion event
    promotion_record = {
        "timestamp":   datetime.utcnow().isoformat(),
        "new_version": new_version,
        "new_auc":     ti.xcom_pull(key="new_auc",      task_ids="retrain_model"),
        "champion_auc":ti.xcom_pull(key="champion_auc", task_ids="evaluate_model"),
        "trigger":     "drift_detection",
    }
    print(f"Promotion record: {promotion_record}")

# ── Define the DAG ────────────────────────────────────────────────────
with DAG(
    dag_id="fraud_model_retraining",
    # dag_id = unique name shown in Airflow UI

    default_args=default_args,

    description="Retrain fraud model when drift is detected",

    schedule_interval="0 * * * *",
    # cron expression: "0 * * * *" = at minute 0 of every hour
    # format: minute hour day month weekday
    # other examples:
    # "*/15 * * * *" = every 15 minutes
    # "0 2 * * *"    = every day at 2am
    # None           = only triggered manually

    start_date=datetime(2024, 1, 1),
    # Airflow won't schedule runs before this date

    catchup=False,
    # catchup=False = don't run all missed intervals since start_date
    # True would try to run every hour since Jan 1 2024 — not what we want

    tags=["ml", "monitoring", "fraud"],
    # tags appear in the UI for filtering
) as dag:

    # ── Define tasks ──────────────────────────────────────────────────
    t1 = PythonOperator(
        task_id="check_drift",
        # task_id = unique name within this DAG
        python_callable=check_drift,
        # python_callable = the function to run
    )

    t2 = PythonOperator(
        task_id="retrain_model",
        python_callable=retrain_model,
    )

    t3 = PythonOperator(
        task_id="evaluate_model",
        python_callable=evaluate_model,
    )

    t4 = PythonOperator(
        task_id="promote_if_better",
        python_callable=promote_if_better,
    )

    # ── Set task order ────────────────────────────────────────────────
    t1 >> t2 >> t3 >> t4
    # >> operator = "then run"
    # t1 must complete successfully before t2 starts
    # t2 must complete before t3, and so on
    # this is what makes it a DAG — defined execution order
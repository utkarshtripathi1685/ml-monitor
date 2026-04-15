# monitor/detector.py
# Drift detection engine.
# Runs periodically, compares live feature distributions
# against the training baseline, fires alerts on significant drift.

import os
import json
import time
import numpy as np
import pandas as pd
from scipy import stats
# scipy.stats = statistical functions
# we use ks_2samp (two-sample KS test) from here

import redis
import warnings
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv("../.env")

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────
REDIS_HOST         = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT         = int(os.getenv("REDIS_PORT", "6379"))
BASELINE_STATS_CSV = "../model/baseline_stats.csv"
PREDICTIONS_LOG    = "../model/predictions_log.jsonl"
DRIFT_LOG          = "drift_log.jsonl"

# PSI thresholds — industry standard from banking
PSI_STABLE     = 0.1
# PSI < 0.1: distribution is stable, no action needed
PSI_ALERT      = 0.2
# PSI > 0.2: significant drift detected, trigger retraining

FEATURE_NAMES  = [f"V{i}" for i in range(1, 29)] + ["Amount"]
CHECK_INTERVAL = 60
# run drift check every 60 seconds (use 3600 in production = every hour)

# ── Connect to Redis ──────────────────────────────────────────────────
redis_client = redis.Redis(
    host=REDIS_HOST, port=REDIS_PORT, decode_responses=True
)
redis_client.ping()
print("Connected to Redis")

# ── Load baseline stats ───────────────────────────────────────────────
baseline_df = pd.read_csv(BASELINE_STATS_CSV, index_col=0)
# baseline_df = the .describe() output we saved during training
# rows: count, mean, std, min, 25%, 50%, 75%, max
# columns: V1, V2, ..., V28, Amount

print(f"Baseline loaded: {baseline_df.shape[1]} features")

# ─────────────────────────────────────────────────────────────────────
# PSI FUNCTION
# PSI = Population Stability Index
# Measures how much a distribution has shifted between two samples.
# Originally from credit risk modeling — now standard in ML monitoring.
# ─────────────────────────────────────────────────────────────────────
def calculate_psi(baseline_values, current_values, n_bins=10):
    """
    Calculate PSI between baseline and current distributions.

    baseline_values: numpy array of training feature values
    current_values:  numpy array of live feature values
    n_bins:          number of buckets to divide distribution into

    Returns: PSI score (float)
    """

    # Create bin edges from baseline distribution
    # np.percentile creates evenly-spaced quantile boundaries
    # e.g. for n_bins=10: [10th, 20th, 30th, ..., 90th] percentile
    breakpoints = np.percentile(
        baseline_values,
        np.linspace(0, 100, n_bins + 1)
        # linspace(0, 100, 11) = [0, 10, 20, ..., 100]
        # these are the percentile points for bin edges
    )

    # Remove duplicate breakpoints (can happen when many values are equal)
    breakpoints = np.unique(breakpoints)

    if len(breakpoints) < 3:
        # not enough unique values to create meaningful bins
        return 0.0

    # Count how many values fall into each bin
    baseline_counts, _ = np.histogram(baseline_values, bins=breakpoints)
    current_counts,  _ = np.histogram(current_values,  bins=breakpoints)
    # np.histogram returns (counts, bin_edges)
    # we only need counts, so we discard bin_edges with _

    # Convert counts to percentages
    # add tiny epsilon (1e-4) to avoid division by zero
    # if a bin is empty, log(0) would be -infinity
    baseline_pct = baseline_counts / len(baseline_values) + 1e-4
    current_pct  = current_counts  / len(current_values)  + 1e-4

    # PSI formula: sum of (current% - baseline%) * ln(current% / baseline%)
    # each term measures how much one bin has shifted
    # multiplying by the log ratio makes large shifts penalised more
    psi = np.sum(
        (current_pct - baseline_pct) * np.log(current_pct / baseline_pct)
    )

    return float(psi)

# ─────────────────────────────────────────────────────────────────────
# KS TEST FUNCTION
# KS = Kolmogorov-Smirnov
# Statistical test: "are these two samples from the same distribution?"
# Returns p-value — if p < 0.05, distributions are significantly different
# ─────────────────────────────────────────────────────────────────────
def calculate_ks(baseline_values, current_values):
    """
    Run two-sample KS test.
    Returns (statistic, p_value)
    statistic: max distance between the two CDFs (0 to 1)
    p_value:   probability of seeing this difference by chance
               p < 0.05 means drift is statistically significant
    """
    statistic, p_value = stats.ks_2samp(baseline_values, current_values)
    # ks_2samp = two-sample KS test
    # compares the empirical CDFs (cumulative distribution functions)
    # of the two arrays
    # statistic = maximum vertical distance between the two CDFs
    # p_value   = probability this gap happened by random chance
    return float(statistic), float(p_value)

# ─────────────────────────────────────────────────────────────────────
# FETCH LIVE FEATURES FROM REDIS
# ─────────────────────────────────────────────────────────────────────
def fetch_live_features(sample_size=500):
    """
    Fetch recent feature values from Redis.
    Returns a DataFrame with shape (n_samples, n_features)
    """
    # Get all keys matching "transaction:*"
    keys = redis_client.keys("transaction:*")
    # redis_client.keys(pattern) = returns all keys matching the pattern
    # "transaction:*" = any key starting with "transaction:"
    # "*" is a wildcard

    if len(keys) < 50:
        # not enough data to run a meaningful drift check
        print(f"Only {len(keys)} transactions in Redis. Need at least 50.")
        return None

    # Sample up to sample_size keys randomly
    sample_keys = np.random.choice(keys, size=min(sample_size, len(keys)), replace=False)
    # np.random.choice = randomly pick elements from an array
    # replace=False = no duplicates in the sample

    rows = []
    for key in sample_keys:
        features = redis_client.hgetall(key)
        # hgetall = get all fields of a Redis Hash as a dict
        if features:
            row = {name: float(features.get(name, 0)) for name in FEATURE_NAMES}
            # features.get(name, 0) = get value or 0 if missing
            # float() converts Redis string → number
            rows.append(row)

    if not rows:
        return None

    return pd.DataFrame(rows)
    # shape: (n_samples, 29) — one row per transaction

# ─────────────────────────────────────────────────────────────────────
# FETCH PREDICTION DRIFT FROM LOG
# ─────────────────────────────────────────────────────────────────────
def fetch_prediction_stats(hours_back=1):
    """
    Read recent predictions from predictions_log.jsonl
    Returns dict with fraud_rate and avg_confidence
    """
    if not os.path.exists(PREDICTIONS_LOG):
        return None

    cutoff = datetime.utcnow() - timedelta(hours=hours_back)
    # cutoff = only look at predictions from the last N hours
    # timedelta = represents a duration of time

    recent_probs = []

    with open(PREDICTIONS_LOG, "r") as f:
        for line in f:
            # each line is one JSON object
            entry = json.loads(line.strip())
            # json.loads = string → dict
            # .strip() = remove trailing newline

            ts = datetime.fromisoformat(entry["timestamp"])
            # fromisoformat = parse "2024-01-01T12:00:00" → datetime object

            if ts > cutoff:
                recent_probs.append(entry["fraud_probability"])

    if len(recent_probs) < 10:
        return None

    return {
        "count":            len(recent_probs),
        "fraud_rate":       float(np.mean([p > 0.5 for p in recent_probs])),
        # mean of True/False list = fraction of fraud predictions
        "avg_confidence":   float(np.mean(recent_probs)),
        "high_conf_frauds": int(sum(p > 0.8 for p in recent_probs)),
        # count of very high confidence fraud predictions
    }

# ─────────────────────────────────────────────────────────────────────
# MAIN DRIFT CHECK FUNCTION
# ─────────────────────────────────────────────────────────────────────
def run_drift_check():
    """
    Run a full drift check across all features.
    Logs results to drift_log.jsonl.
    Returns True if drift detected, False if stable.
    """
    print(f"\n{'='*50}")
    print(f"Running drift check at {datetime.utcnow().isoformat()}")

    # Fetch live features from Redis
    live_df = fetch_live_features(sample_size=500)
    if live_df is None:
        print("Not enough live data. Skipping.")
        return False

    print(f"Fetched {len(live_df)} live samples from Redis")

    # Extract baseline mean and std for synthetic baseline generation
    # We use baseline stats to reconstruct approximate training distribution
    drift_results   = {}
    drifted_features = []

    for feature in FEATURE_NAMES:
        # Get baseline distribution parameters from saved stats
        baseline_mean = float(baseline_df.loc["mean", feature])
        baseline_std  = float(baseline_df.loc["std",  feature])
        # baseline_df.loc["mean", feature] = value at row "mean", column feature

        # Reconstruct approximate baseline sample using normal distribution
        baseline_sample = np.random.normal(
            loc=baseline_mean,
            scale=baseline_std,
            size=1000
        )
        # loc = mean, scale = std dev
        # size = how many values to generate
        # this gives us a sample that matches the training distribution shape

        # Get live values for this feature
        live_values = live_df[feature].values
        # .values = convert pandas Series to numpy array

        # Calculate PSI
        psi = calculate_psi(baseline_sample, live_values)

        # Calculate KS test
        ks_stat, ks_pvalue = calculate_ks(baseline_sample, live_values)

        drift_results[feature] = {
            "psi":       round(psi, 4),
            "ks_stat":   round(ks_stat, 4),
            "ks_pvalue": round(ks_pvalue, 4),
            "drifted":   psi > PSI_ALERT or ks_pvalue < 0.05,
            # drifted = True if EITHER PSI is high OR KS test says different
        }

        if drift_results[feature]["drifted"]:
            drifted_features.append(feature)

    # Check prediction drift
    pred_stats = fetch_prediction_stats(hours_back=1)

    # Build summary
    max_psi = max(v["psi"] for v in drift_results.values())
    # max PSI across all features — overall drift score

    overall_drift = len(drifted_features) > 0

    summary = {
        "timestamp":          datetime.utcnow().isoformat(),
        "overall_drift":      overall_drift,
        "drifted_features":   drifted_features,
        "n_drifted":          len(drifted_features),
        "max_psi":            round(max_psi, 4),
        "feature_results":    drift_results,
        "prediction_stats":   pred_stats,
    }

    # Save to drift log
    with open(DRIFT_LOG, "a") as f:
        f.write(json.dumps(summary) + "\n")

    # Print summary
    print(f"Features checked:  {len(FEATURE_NAMES)}")
    print(f"Features drifted:  {len(drifted_features)}")
    print(f"Max PSI:           {max_psi:.4f}")

    if drifted_features:
        print(f"\nDRIFT ALERT — drifted features: {drifted_features[:5]}")
        print("Action: consider retraining the model")
    else:
        print("Status: STABLE — no significant drift detected")

    if pred_stats:
        print(f"Prediction stats:  {pred_stats['count']} predictions, "
              f"fraud rate: {pred_stats['fraud_rate']:.2%}")

    return overall_drift

# ─────────────────────────────────────────────────────────────────────
# SCHEDULER LOOP
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Drift detector started. Checking every {CHECK_INTERVAL}s\n")

    while True:
        try:
            run_drift_check()
        except Exception as e:
            # catch any unexpected error so the loop keeps running
            # in production you'd send this error to a logging service
            print(f"Error during drift check: {e}")

        print(f"\nNext check in {CHECK_INTERVAL} seconds...")
        time.sleep(CHECK_INTERVAL)
        # time.sleep = pause execution for N seconds
        # the detector wakes up, checks, sleeps, repeats forever
# monitor/simulate_drift.py
# Injects artificially drifted data into Redis
# so we can test that the detector actually catches drift.
# In production you'd never do this — drift happens naturally.

import redis
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv("../.env")

redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", "6379")),
    decode_responses=True
)

FEATURE_NAMES = [f"V{i}" for i in range(1, 29)] + ["Amount"]

print("Injecting drifted transactions into Redis...")

# Inject 200 transactions with severely shifted Amount and V1
# Amount shifted from mean~100 to mean~5000 (unusual large transactions)
# V1 shifted by +3 standard deviations (unusual pattern)
for i in range(200):
    drifted_features = {}

    for feature in FEATURE_NAMES:
        if feature == "Amount":
            # normal Amount mean ~100, inject mean ~5000
            drifted_features[feature] = float(np.random.exponential(5000))
            # exponential distribution with very high mean
        elif feature == "V1":
            # shift V1 by 3 standard deviations
            drifted_features[feature] = float(np.random.normal(3.0, 1.0))
            # normal data has V1 ~ N(0, 1), we shift to N(3, 1)
        else:
            # other features stay normal
            drifted_features[feature] = float(np.random.normal(0, 1))

    # Store with high transaction IDs to not overwrite real data
    redis_key = f"transaction:{200000 + i}"
    redis_client.hset(redis_key, mapping=drifted_features)
    redis_client.expire(redis_key, 86400)

print(f"Injected 200 drifted transactions (IDs 200000-200199)")
print("Now run detector.py — it should detect drift in Amount and V1")
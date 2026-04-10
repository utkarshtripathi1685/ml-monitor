# model/test_api.py
import requests   # pip install requests if needed

BASE_URL = "http://localhost:8000"

# Test health
resp = requests.get(f"{BASE_URL}/health")
print("Health:", resp.json())

# Test a few predictions
for tx_id in [0, 100, 500, 1000, 5000]:
    resp = requests.post(
        f"{BASE_URL}/predict",
        json={"transaction_id": tx_id}
        # json= automatically sets Content-Type: application/json
        # and serializes the dict to JSON
    )
    if resp.status_code == 200:
        data = resp.json()
        label = "FRAUD" if data["prediction"] == 1 else "legit"
        print(
            f"Transaction {tx_id:>5} | "
            f"Fraud prob: {data['fraud_probability']:.4f} | "
            f"{label} | "
            f"{data['latency_ms']}ms"
        )
    else:
        print(f"Transaction {tx_id}: {resp.status_code} — {resp.json()}")
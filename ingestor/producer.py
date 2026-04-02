# ingestor/producer.py — only the import and producer creation changes
# replace the top section:

import json
import time
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from confluent_kafka import Producer
# Producer replaces KafkaProducer

load_dotenv("../.env")

KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")
KAFKA_TOPIC  = os.getenv("KAFKA_TOPIC",  "transactions")

# ── Create producer ───────────────────────────────────────────────────
producer = Producer({
    "bootstrap.servers": KAFKA_BROKER,
    # confluent-kafka config dict — same dot notation as consumer
})

def delivery_report(err, msg):
    # delivery_report = callback function called after each message send
    # confluent-kafka is async by default — send() returns immediately
    # this callback tells you if the message actually arrived
    # err = None means success, otherwise contains the error
    if err:
        print(f"Delivery failed for {msg.key()}: {err}")
    # we don't print success — too noisy for 100k messages

print(f"Producer connected to Kafka at {KAFKA_BROKER}")
print(f"Sending to topic: '{KAFKA_TOPIC}'")

df = pd.read_csv("../model/creditcard.csv")
print(f"Loaded {len(df)} transactions. Starting stream...\n")

for index, row in df.iterrows():
    event = {
        "transaction_id": int(index),
        "timestamp":      time.time(),
        "features": {
            col: float(row[col])
            for col in df.columns
            if col != "Class"
        },
        "label": int(row["Class"]),
    }

    producer.produce(
        KAFKA_TOPIC,
        # produce() replaces send() in confluent-kafka
        value=json.dumps(event).encode("utf-8"),
        # confluent-kafka doesn't have value_serializer
        # so we manually serialize: dict → JSON string → bytes
        key=str(index).encode("utf-8"),
        on_delivery=delivery_report,
        # on_delivery = callback for delivery confirmation
    )

    producer.poll(0)
    # poll(0) = process any pending callbacks without waiting
    # this drains the delivery report queue so it doesn't grow forever
    # call it frequently inside the loop

    if index % 100 == 0:
        fraud_flag = "FRAUD" if event["label"] == 1 else "legit"
        print(f"Sent transaction {index:>6} | Amount: ${event['features'].get('Amount', 0):>8.2f} | {fraud_flag}")

    time.sleep(0.05)

producer.flush()
# flush() = wait until ALL pending messages are delivered
# blocks until delivery_report has been called for every message
print("\nAll transactions sent.")
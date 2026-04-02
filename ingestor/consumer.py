# ingestor/consumer.py
# full file — replace your current version with this

import json
import redis
import pandas as pd
import numpy as np
import joblib
import os
import warnings
from dotenv import load_dotenv
from confluent_kafka import Consumer, KafkaError
# confluent_kafka = official Confluent client
# Consumer = the consumer class (replaces KafkaConsumer)
# KafkaError = error codes we check during message polling

warnings.filterwarnings("ignore", message="X does not have valid feature names")

load_dotenv("../.env")

# ── Config ───────────────────────────────────────────────────────────
KAFKA_BROKER = os.getenv("KAFKA_BROKER",  "localhost:9092")
KAFKA_TOPIC  = os.getenv("KAFKA_TOPIC",   "transactions")
REDIS_HOST   = os.getenv("REDIS_HOST",    "localhost")
REDIS_PORT   = int(os.getenv("REDIS_PORT", "6379"))

# ── Connect to Redis ─────────────────────────────────────────────────
redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    decode_responses=True,
)
redis_client.ping()
print("Connected to Redis")

# ── Load scaler ───────────────────────────────────────────────────────
scaler = joblib.load("../model/scaler.pkl")
print("Scaler loaded")

# ── Create Kafka consumer ─────────────────────────────────────────────
consumer = Consumer({
    "bootstrap.servers": KAFKA_BROKER,
    # confluent-kafka uses a config dict instead of keyword arguments
    # keys use dots instead of underscores: "bootstrap.servers" not "bootstrap_servers"

    "group.id": "feature-store-group",
    # same concept as before — consumer group for offset tracking

    "auto.offset.reset": "earliest",
    # start from beginning if no saved offset exists

    "enable.auto.commit": True,
    # automatically commit offsets every 5 seconds
})

consumer.subscribe([KAFKA_TOPIC])
# subscribe() takes a LIST of topics (you can subscribe to multiple)
# in kafka-python it was passed to the constructor, here it's separate

print(f"Listening on topic '{KAFKA_TOPIC}'...\n")

# ── Process messages ──────────────────────────────────────────────────
feature_names = [f"V{i}" for i in range(1, 29)] + ["Amount"]
# define once outside the loop — no need to recreate every iteration

try:
    while True:
        # confluent-kafka uses poll() instead of iteration
        # poll(timeout) = wait up to 1.0 second for a new message
        # returns None if no message arrived in that time
        msg = consumer.poll(1.0)

        if msg is None:
            # no message in the last second — just keep waiting
            continue

        if msg.error():
            # msg.error() returns None if message is fine
            # returns a KafkaError object if something went wrong
            if msg.error().code() == KafkaError._PARTITION_EOF:
                # _PARTITION_EOF = reached end of partition (not a real error)
                # happens when consumer catches up to producer
                # just means "no new messages right now"
                continue
            else:
                # real error — print and stop
                print(f"Kafka error: {msg.error()}")
                break

        # ── Deserialize message ───────────────────────────────────────
        event = json.loads(msg.value().decode("utf-8"))
        # msg.value() = raw bytes (confluent-kafka doesn't auto-deserialize)
        # .decode("utf-8") = bytes → string
        # json.loads() = string → Python dict
        # same logic as before, just manual instead of automatic

        transaction_id = event["transaction_id"]
        features       = event["features"]

        # ── Scale features ────────────────────────────────────────────
        raw_df = pd.DataFrame(
            [[features[col] for col in feature_names]],
            columns=feature_names
        )
        scaled_values = scaler.transform(raw_df)
        scaled_dict   = {
            name: float(scaled_values[0][i])
            for i, name in enumerate(feature_names)
        }

        # ── Store in Redis ────────────────────────────────────────────
        redis_key = f"transaction:{transaction_id}"
        redis_client.hset(redis_key, mapping=scaled_dict)
        redis_client.expire(redis_key, 3600)

        if transaction_id % 100 == 0:
            print(f"Stored transaction {transaction_id:>6} → Redis key: {redis_key}")

except KeyboardInterrupt:
    # catching KeyboardInterrupt (Ctrl+C) means we exit cleanly
    # instead of showing a scary traceback
    print("\nStopped by user.")

finally:
    consumer.close()
    # finally block ALWAYS runs — even after an error or Ctrl+C
    # close() = cleanly disconnect from Kafka
    # tells the broker this consumer is leaving the group gracefully
    print("Consumer closed.")
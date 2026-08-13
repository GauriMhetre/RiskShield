import time
import json
import logging
from datetime import datetime
from pathlib import Path
import pandas as pd
from kafka import KafkaProducer
from kafka.errors import KafkaError

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def load_sample_transactions(csv_path="data/processed/synthetic_transactions.csv", sample_size=200, seed=42) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset at {csv_path}. Please ensure Phase 1 synthetic data generation has been run.")
    
    df = pd.read_csv(path)
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=seed)
    return df

def build_kafka_producer(bootstrap_servers="localhost:9092") -> KafkaProducer:
    try:
        producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            request_timeout_ms=5000,
        )
        return producer
    except Exception:
        logger.error(f"ERROR: Failed to connect to Kafka at {bootstrap_servers}.")
        logger.error("Please confirm Kafka is running via 'docker-compose up -d kafka'.")
        raise

def transaction_to_message(row) -> dict:
    """
    Converts a DataFrame row to a ScoreRequest-compatible message dictionary.
    
    IMPORTANT: We stamp the 'created_at' field with the CURRENT time (datetime.now()) 
    rather than reusing the CSV's original historical timestamp. 
    Why? Because we are simulating a live stream of transactions happening *now*.
    If we replayed stale historical timestamps into a live pipeline, time-window 
    features (like velocity) would break, as the consumer would think these 
    transactions occurred years ago.
    """
    return {
        "txn_id": str(row["transaction_id"]),
        "user_id": str(row["user_id"]),
        "amount": float(row["amount"]),
        "currency": str(row["currency"]),
        "merchant_id": str(row["merchant_id"]),
        "device_id": str(row["device_id"]),
        "ip_address": str(row["ip_address"]),
        "country": str(row["country"]),
        "latitude": float(row["latitude"]) if pd.notnull(row["latitude"]) else 0.0,
        "longitude": float(row["longitude"]) if pd.notnull(row["longitude"]) else 0.0,
        "created_at": datetime.now().isoformat()
    }

def produce_stream(producer, df, topic="transactions", delay_seconds=0.3) -> None:
    total = len(df)
    for i, (_, row) in enumerate(df.iterrows(), 1):
        msg = transaction_to_message(row)
        try:
            producer.send(topic, value=msg)
            logger.info(f"Sent {i}/{total}: txn_id={msg['txn_id']} user_id={msg['user_id']} amount={msg['amount']}")
        except KafkaError as e:
            logger.error(f"Failed to send message: {e}")
        
        time.sleep(delay_seconds)

def main():
    logger.info("Starting simulated transaction stream...")
    df = load_sample_transactions()
    producer = build_kafka_producer()
    
    try:
        produce_stream(producer, df)
    except KeyboardInterrupt:
        logger.info("\nStreaming interrupted by user.")
    finally:
        # Why flush? The KafkaProducer sends messages asynchronously. If the script exits 
        # immediately after the last .send(), any messages still sitting in the internal 
        # memory buffers will be silently dropped. producer.flush() explicitly waits for 
        # all buffered messages to be fully acknowledged by the broker before closing.
        producer.flush()
        producer.close()
        logger.info("Producer finished, all messages flushed.")

if __name__ == "__main__":
    main()

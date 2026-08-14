import json
import logging
from datetime import datetime
from uuid import UUID

from kafka import KafkaConsumer
from kafka.errors import NoBrokersAvailable

from backend.app.db.repository import (
    get_recent_transactions,
    get_user_profile,
    save_scored_transaction,
    save_transaction,
    upsert_user_profile,
)
from backend.app.db.session import SessionLocal
from backend.app.ml.model_loader import ModelLoader
from ml.features import TransactionInput, compute_features
from ml.features import UserProfile as UserProfileFeatures

logging.basicConfig(level=logging.INFO, format='%(asctime)s INFO consumer - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

def build_kafka_consumer(topic="transactions", bootstrap_servers="localhost:9092", group_id="riskshield-scoring-consumer") -> KafkaConsumer:
    """
    Builds a KafkaConsumer that connects to the broker and decodes JSON messages.
    
    The group_id identifies this consumer as part of a specific "consumer group". 
    Kafka tracks the offset (which messages have been read) per group_id. Using a 
    stable, explicit name ensures that if the consumer restarts, it resumes from 
    where it left off instead of reprocessing everything or skipping messages.
    """
    try:
        consumer = KafkaConsumer(
            topic,
            bootstrap_servers=bootstrap_servers,
            group_id=group_id,
            value_deserializer=lambda v: json.loads(v.decode('utf-8'))
        )
        return consumer
    except NoBrokersAvailable:
        logger.error(f"ERROR: Failed to connect to Kafka at {bootstrap_servers}.")
        logger.error("Please confirm Kafka is running via 'docker-compose up -d kafka'.")
        raise
    except Exception as e:
        logger.error(f"Failed to build KafkaConsumer: {e}")
        raise

def score_message(session, model_loader: ModelLoader, message: dict) -> dict:
    """
    Scores a single transaction message from Kafka.
    
    IMPORTANT: This function's logic must always mirror backend/app/api/score.py's 
    scoring sequence. If score.py's logic ever changes, this function needs the same 
    update. This duplication-of-SEQUENCE (not duplication of underlying feature/model 
    logic, which is genuinely shared) is an acknowledged, deliberate tradeoff of having 
    two entry points (HTTP and Kafka) into the same scoring pipeline.
    """
    # Parse incoming message data
    user_id_uuid = UUID(message["user_id"])
    created_at = datetime.fromisoformat(message["created_at"])
    
    # ====================================================================
    # Step 1: Look up the user's profile from PostgreSQL
    # ====================================================================
    db_profile = get_user_profile(session, user_id_uuid)
    
    if db_profile is None:
        db_profile = upsert_user_profile(
            session,
            user_id=user_id_uuid,
            profile_updates={
                "avg_txn_amount": 0.0,
                "std_txn_amount": 0.0,
                "txn_count": 0,
            },
        )

    # ====================================================================
    # Step 2: Query recent transactions for velocity features
    # ====================================================================
    recent_txns_24h = get_recent_transactions(
        session,
        user_id=user_id_uuid,
        window_hours=24.0,
        before_timestamp=created_at,
    )
    recent_txn_timestamps = [txn.created_at for txn in recent_txns_24h]

    # ====================================================================
    # Step 3: Convert repository data (ORM model) to feature engineering format
    # ====================================================================
    profile_for_features = UserProfileFeatures(
        user_id=str(db_profile.user_id),
        avg_amount=float(db_profile.avg_txn_amount),
        std_amount=float(db_profile.std_txn_amount),
        known_device_ids=[db_profile.last_device_id] if db_profile.last_device_id else [],
        home_country=db_profile.last_country or "",
        home_latitude=float(db_profile.last_latitude) if db_profile.last_latitude else None,
        home_longitude=float(db_profile.last_longitude) if db_profile.last_longitude else None,
        recent_txn_timestamps=recent_txn_timestamps,
    )

    # ====================================================================
    # Step 4: Build a TransactionInput object
    # ====================================================================
    transaction = TransactionInput(
        transaction_id=message["txn_id"],
        amount=message["amount"],
        device_id=message["device_id"],
        country=message["country"],
        latitude=message["latitude"],
        longitude=message["longitude"],
        created_at=created_at,
    )

    # ====================================================================
    # Step 5: Compute features
    # ====================================================================
    feature_dict = compute_features(transaction, profile_for_features)

    # ====================================================================
    # Step 6 & 7: Predict fraud probability
    # ====================================================================
    risk_score = model_loader.predict_proba(feature_dict)

    # ====================================================================
    # Step 8: Determine if flagged based on threshold
    # ====================================================================
    threshold = model_loader.get_threshold()
    flagged = risk_score >= threshold

    # ====================================================================
    # Step 10: Save the transaction to the database
    # ====================================================================
    saved_txn = save_transaction(
        session,
        transaction_data={
            "user_id": user_id_uuid,
            "amount": message["amount"],
            "currency": message["currency"],
            "merchant_id": message.get("merchant_id"),
            "device_id": message["device_id"],
            "ip_address": message.get("ip_address"),
            "country": message["country"],
            "latitude": message["latitude"],
            "longitude": message["longitude"],
            "created_at": created_at,
        },
    )

    # ====================================================================
    # Step 11: Save the scoring decision to the database
    # ====================================================================
    save_scored_transaction(
        session,
        scored_data={
            "txn_id": saved_txn.txn_id,
            "risk_score": float(risk_score),
            "flagged": flagged,
            "model_version": model_loader.get_model_version(),
            "feature_snapshot": feature_dict,
            "shap_values": None,
        },
    )

    return {
        "txn_id": message["txn_id"],
        "risk_score": float(risk_score),
        "flagged": flagged
    }


def main():
    # Load model once at startup
    model_loader = ModelLoader(model_version="model_v1")
    consumer = build_kafka_consumer()
    
    logger.info("Consumer started, listening on topic 'transactions'...")
    
    for message in consumer:
        # Create a new session per message
        session = SessionLocal()
        try:
            msg_dict = message.value
            summary = score_message(session, model_loader, msg_dict)
            session.commit()
            logger.info(f"Scored txn_id={summary['txn_id']} risk_score={summary['risk_score']:.4f} flagged={summary['flagged']}")
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to process message: {e} | Raw message: {message.value}")
        finally:
            session.close()

if __name__ == "__main__":
    main()

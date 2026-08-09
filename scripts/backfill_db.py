"""
Database backfill script for RiskShield.

Loads synthetic transactions from data/processed/synthetic_transactions.csv
and populates the PostgreSQL database with realistic historical data.

This is a ONE-TIME, manually-run script (not part of the running application).
It does NOT score these historical transactions (that's why scored_transactions
remains empty after this script completes) — Task 6 or user exploration will
selectively score backfilled users live via /score if desired.

Key design decisions:
  - Computes each user's FINAL end-state profile across their full history
    (unlike ml/batch_features.py's leakage-safe expanding-window used for training)
    because the goal here is populating the database as if this history already
    happened, not training a leakage-safe model.
  - Clears and re-inserts all backfill data each run (simplest, most predictable)
    rather than incremental upserts.
  - Batches commits during transaction insertion for reasonable performance
    (not one per row, not one giant commit).
"""

import sys
import csv
import time
import logging
from uuid import UUID
from datetime import datetime
from pathlib import Path
from collections import defaultdict

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.app.db.session import SessionLocal
from backend.app.db.models import UserProfile, Transaction

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


def load_synthetic_data(csv_path: str = "data/processed/synthetic_transactions.csv") -> list[dict]:
    """
    Load synthetic transactions from CSV.

    Args:
        csv_path: Path to the synthetic transactions CSV

    Returns:
        List of transaction dicts with parsed created_at datetime

    Raises:
        FileNotFoundError: If CSV is missing
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Synthetic data CSV not found at {csv_path}.\n"
            f"Run: python data/generate_synthetic.py first to create it."
        )

    transactions = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Parse created_at as datetime
            row["created_at"] = datetime.fromisoformat(row["created_at"])
            transactions.append(row)

    logger.info(f"Loaded {len(transactions)} synthetic transactions")
    return transactions


def derive_user_profiles(transactions: list[dict]) -> list[dict]:
    """
    Derive user profiles from transaction history.

    Groups by user_id and computes:
      - avg_txn_amount: mean of amounts
      - std_txn_amount: standard deviation of amounts (0 for single transactions)
      - txn_count: number of transactions
      - last_device_id, last_country, last_latitude, last_longitude: from most recent transaction

    IMPORTANT DESIGN NOTE:
    Unlike ml/batch_features.py's leakage-safe expanding-window feature computation
    (which walks through time chronologically during MODEL TRAINING to avoid leakage),
    this backfill computes each user's FINAL end-state profile across their FULL
    history, because the goal here is populating the database as if this history
    already happened. We're not training a model — we're populating an end-state
    database for realistic feature computation during live /score calls. These are
    deliberately different and simpler computations suited to their respective jobs.

    Args:
        transactions: List of transaction dicts with parsed created_at

    Returns:
        List of user profile dicts, one per unique user_id
    """
    profiles_by_user = defaultdict(list)

    # Group transactions by user
    for txn in transactions:
        user_id = txn["user_id"]
        profiles_by_user[user_id].append(txn)

    profiles = []
    for user_id, user_txns in profiles_by_user.items():
        amounts = [float(txn["amount"]) for txn in user_txns]

        # Compute statistics
        avg_amount = sum(amounts) / len(amounts)
        if len(amounts) > 1:
            variance = sum((x - avg_amount) ** 2 for x in amounts) / len(amounts)
            std_amount = variance ** 0.5
        else:
            std_amount = 0.0

        # Get most recent transaction (by created_at)
        last_txn = max(user_txns, key=lambda t: t["created_at"])

        profiles.append(
            {
                "user_id": UUID(user_id),
                "avg_txn_amount": round(avg_amount, 2),
                "std_txn_amount": round(std_amount, 2),
                "txn_count": len(user_txns),
                "last_device_id": last_txn.get("device_id"),
                "last_country": last_txn.get("country"),
                "last_latitude": float(last_txn["latitude"]) if last_txn.get("latitude") else None,
                "last_longitude": float(last_txn["longitude"]) if last_txn.get("longitude") else None,
            }
        )

    logger.info(f"Derived profiles for {len(profiles)} users")
    return profiles


def clear_backfill_data(session) -> None:
    """
    Clear all existing backfilled data from the database.

    Deletes rows in the correct foreign-key order:
    1. Delete from transactions (references user_profile)
    2. Delete from user_profile

    WARNING: This is DESTRUCTIVE. Use only for backfill/dev scripts.
    Never expose this operation via an API route.

    Args:
        session: SQLAlchemy session
    """
    try:
        # Delete transactions first (they reference user_profile via FK)
        txn_count = session.query(Transaction).delete()
        session.commit()

        # Then delete user profiles
        profile_count = session.query(UserProfile).delete()
        session.commit()

        logger.info(
            f"Cleared {txn_count} existing transactions, "
            f"{profile_count} existing user_profile rows"
        )
    except Exception as e:
        session.rollback()
        raise RuntimeError(f"Failed to clear backfill data: {e}")


def backfill_user_profiles(session, profiles: list[dict]) -> int:
    """
    Insert user profiles into the database.

    Uses SQLAlchemy bulk_insert_mappings for reasonable performance
    (not one commit per row).

    Args:
        session: SQLAlchemy session
        profiles: List of profile dicts

    Returns:
        Count of profiles inserted
    """
    try:
        session.bulk_insert_mappings(UserProfile, profiles)
        session.commit()
        logger.info(f"Inserted {len(profiles)} user profiles")
        return len(profiles)
    except Exception as e:
        session.rollback()
        raise RuntimeError(f"Failed to backfill user profiles: {e}")


def backfill_transactions(session, transactions: list[dict], batch_size: int = 1000) -> int:
    """
    Insert transactions into the database in batches.

    Commits once per batch (not per row, not all at once) for reasonable
    performance and memory usage.

    Args:
        session: SQLAlchemy session
        transactions: List of transaction dicts (with parsed created_at)
        batch_size: Number of rows to insert per commit

    Returns:
        Total count of transactions inserted
    """
    total_inserted = 0

    try:
        for i in range(0, len(transactions), batch_size):
            batch = transactions[i : i + batch_size]

            # Convert to Transaction objects, mapping CSV columns to model fields
            txn_objects = [
                Transaction(
                    txn_id=UUID(txn["transaction_id"]),
                    user_id=UUID(txn["user_id"]),
                    amount=float(txn["amount"]),
                    currency=txn["currency"],
                    merchant_id=txn.get("merchant_id"),
                    device_id=txn.get("device_id"),
                    ip_address=txn.get("ip_address"),
                    country=txn.get("country"),
                    latitude=float(txn["latitude"]) if txn.get("latitude") else None,
                    longitude=float(txn["longitude"]) if txn.get("longitude") else None,
                    created_at=txn["created_at"],
                )
                for txn in batch
            ]

            session.bulk_insert_mappings(
                Transaction,
                [
                    {
                        "txn_id": obj.txn_id,
                        "user_id": obj.user_id,
                        "amount": obj.amount,
                        "currency": obj.currency,
                        "merchant_id": obj.merchant_id,
                        "device_id": obj.device_id,
                        "ip_address": obj.ip_address,
                        "country": obj.country,
                        "latitude": obj.latitude,
                        "longitude": obj.longitude,
                        "created_at": obj.created_at,
                    }
                    for obj in txn_objects
                ],
            )
            session.commit()

            total_inserted += len(batch)
            logger.info(f"Inserted {total_inserted}/{len(transactions)} transactions...")

        logger.info(f"Inserted {total_inserted} transactions total")
        return total_inserted

    except Exception as e:
        session.rollback()
        logger.error(f"Failed to insert batch at row {i}: {e}")
        raise RuntimeError(f"Transaction batch insert failed: {e}")


def main():
    """
    Orchestrate the backfill: load → clear → populate profiles → populate transactions.

    Measures and reports elapsed time using perf_counter (consistent with Phase 4's
    timing approach).
    """
    t_start = time.perf_counter()
    session = SessionLocal()

    try:
        # Load synthetic data
        transactions = load_synthetic_data()

        # Derive user profiles from transaction history
        profiles = derive_user_profiles(transactions)

        # Clear any existing backfilled data (safe to re-run)
        clear_backfill_data(session)

        # Backfill user profiles
        profile_count = backfill_user_profiles(session, profiles)

        # Backfill transactions
        transaction_count = backfill_transactions(session, transactions)

        # Report
        elapsed = time.perf_counter() - t_start
        logger.info(
            f"\n✓ Backfill complete: {profile_count} users, "
            f"{transaction_count} transactions, elapsed {elapsed:.2f}s"
        )

    except Exception as e:
        logger.error(f"✗ Backfill failed: {e}")
        raise
    finally:
        session.close()


if __name__ == "__main__":
    main()

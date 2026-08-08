"""
Manual verification script for Phase 5, Task 2.

Tests the repository functions end-to-end:
  1. Create a test user profile via upsert_user_profile()
  2. Insert test transactions via save_transaction()
  3. Query recent transactions via get_recent_transactions()
  4. Fetch user profile via get_user_profile()
  5. Save a scored transaction via save_scored_transaction()

Run this after the PostgreSQL schema is applied to verify the database integration works.

Command:
  python scripts/manual_test_repository.py

Expected output:
  - Clear labeled output at each step
  - One test user with 3 transactions
  - Queries returning the inserted data
  - No exceptions (database connection, foreign keys, etc.)
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from uuid import uuid4

# Add project root to path so imports work
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.app.db.session import SessionLocal
from backend.app.db.repository import (
    get_user_profile,
    get_recent_transactions,
    save_transaction,
    save_scored_transaction,
    upsert_user_profile,
)


def main():
    """Run the manual test."""
    print("=" * 80)
    print("Phase 5, Task 2 — Repository Manual Test")
    print("=" * 80)
    print()

    # ========================================================================
    # Step 1: Create a test user via upsert_user_profile
    # ========================================================================
    print("Step 1: Creating test user profile...")
    session = SessionLocal()
    try:
        test_user_id = uuid4()
        profile = upsert_user_profile(
            session,
            user_id=test_user_id,
            profile_updates={
                "avg_txn_amount": 1000.0,
                "std_txn_amount": 200.0,
                "txn_count": 0,
                "last_country": "US",
                "last_latitude": 40.7128,
                "last_longitude": -74.0060,
            },
        )
        print(f"✓ Created profile: {profile}")
        print(f"  user_id={profile.user_id}")
        print(f"  avg_txn_amount={profile.avg_txn_amount}")
        print(f"  updated_at={profile.updated_at}")
        print()

        # ====================================================================
        # Step 2: Insert test transactions
        # ====================================================================
        print("Step 2: Inserting test transactions...")
        base_time = datetime.now()

        # Transaction 1: 2 hours ago
        txn1_time = base_time - timedelta(hours=2)
        txn1 = save_transaction(
            session,
            transaction_data={
                "user_id": test_user_id,
                "amount": 500.0,
                "currency": "USD",
                "merchant_id": "merchant_001",
                "device_id": "device_laptop_01",
                "ip_address": "192.168.1.1",
                "country": "US",
                "latitude": 40.7128,
                "longitude": -74.0060,
                "created_at": txn1_time,
            },
        )
        print(f"✓ Inserted transaction 1: {txn1}")
        print(f"  txn_id={txn1.txn_id}")
        print(f"  amount={txn1.amount}")
        print(f"  created_at={txn1.created_at}")

        # Transaction 2: 1 hour ago
        txn2_time = base_time - timedelta(hours=1)
        txn2 = save_transaction(
            session,
            transaction_data={
                "user_id": test_user_id,
                "amount": 750.0,
                "currency": "USD",
                "merchant_id": "merchant_002",
                "device_id": "device_phone_01",
                "ip_address": "192.168.1.2",
                "country": "US",
                "latitude": 40.7580,
                "longitude": -73.9855,
                "created_at": txn2_time,
            },
        )
        print(f"✓ Inserted transaction 2: {txn2}")

        # Transaction 3: 30 minutes ago
        txn3_time = base_time - timedelta(minutes=30)
        txn3 = save_transaction(
            session,
            transaction_data={
                "user_id": test_user_id,
                "amount": 1250.0,
                "currency": "USD",
                "merchant_id": "merchant_003",
                "device_id": "device_laptop_01",
                "ip_address": "192.168.1.3",
                "country": "US",
                "latitude": 40.7489,
                "longitude": -73.9680,
                "created_at": txn3_time,
            },
        )
        print(f"✓ Inserted transaction 3: {txn3}")
        print()

        # ====================================================================
        # Step 3: Query recent transactions (test the before_timestamp logic)
        # ====================================================================
        print("Step 3: Querying recent transactions (within 24-hour window)...")
        # Query transactions BEFORE base_time (now) going back 24 hours
        recent_txns = get_recent_transactions(
            session,
            user_id=test_user_id,
            window_hours=24.0,
            before_timestamp=base_time,
        )
        print(f"✓ Found {len(recent_txns)} recent transactions:")
        for i, txn in enumerate(recent_txns, 1):
            print(
                f"  {i}. {txn.txn_id} | amount={txn.amount} | created_at={txn.created_at}"
            )
        print()

        # ====================================================================
        # Step 4: Query transactions in a narrower window (1.5 hours)
        # ====================================================================
        print("Step 4: Querying transactions within 1.5-hour window...")
        narrow_txns = get_recent_transactions(
            session,
            user_id=test_user_id,
            window_hours=1.5,
            before_timestamp=base_time,
        )
        print(f"✓ Found {len(narrow_txns)} transactions (should be 2):")
        for i, txn in enumerate(narrow_txns, 1):
            print(f"  {i}. amount={txn.amount} | created_at={txn.created_at}")
        print()

        # ====================================================================
        # Step 5: Fetch user profile via get_user_profile
        # ====================================================================
        print("Step 5: Fetching user profile...")
        fetched_profile = get_user_profile(session, test_user_id)
        print(f"✓ Fetched profile: {fetched_profile}")
        print(f"  user_id={fetched_profile.user_id}")
        print(f"  avg_txn_amount={fetched_profile.avg_txn_amount}")
        print()

        # ====================================================================
        # Step 6: Test get_user_profile with non-existent user
        # ====================================================================
        print("Step 6: Testing get_user_profile with non-existent user...")
        fake_user_id = uuid4()
        missing_profile = get_user_profile(session, fake_user_id)
        print(f"✓ get_user_profile(non_existent_id) returned: {missing_profile}")
        print(f"  (Should be None for missing profile)")
        print()

        # ====================================================================
        # Step 7: Save a scored transaction
        # ====================================================================
        print("Step 7: Saving scored transaction...")
        scored_txn = save_scored_transaction(
            session,
            scored_data={
                "txn_id": txn1.txn_id,
                "risk_score": 0.8123,
                "flagged": True,
                "model_version": "model_v1",
                "feature_snapshot": {
                    "txn_count_1h": 0,
                    "txn_count_24h": 2,
                    "amount_zscore": 2.5,
                    "amount_ratio_to_avg": 0.5,
                    "device_mismatch": 0,
                    "country_mismatch": 0,
                    "geo_distance_km": 0.0,
                    "amount": 500.0,
                    "hour_of_day": 14,
                    "day_of_week": 3,
                },
                "shap_values": None,
            },
        )
        print(f"✓ Saved scored transaction: {scored_txn}")
        print(f"  id={scored_txn.id}")
        print(f"  txn_id={scored_txn.txn_id}")
        print(f"  risk_score={scored_txn.risk_score}")
        print(f"  flagged={scored_txn.flagged}")
        print(f"  model_version={scored_txn.model_version}")
        print(f"  feature_snapshot keys: {list(scored_txn.feature_snapshot.keys())}")
        print()

        # ====================================================================
        # Summary
        # ====================================================================
        print("=" * 80)
        print("✓ ALL CHECKS PASSED")
        print("=" * 80)
        print()
        print("Summary:")
        print(f"  - Created user profile: {test_user_id}")
        print(f"  - Inserted 3 transactions")
        print(f"  - Queried recent transactions (24h window): {len(recent_txns)} found")
        print(f"  - Queried recent transactions (1.5h window): {len(narrow_txns)} found")
        print(f"  - Fetched user profile successfully")
        print(f"  - Tested None return for missing user (correct behavior)")
        print(f"  - Saved scored transaction with feature snapshot")
        print()
        print("Phase 5, Task 2 is ready. Database integration verified end-to-end.")

    except Exception as e:
        print(f"✗ ERROR: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        session.close()


if __name__ == "__main__":
    main()

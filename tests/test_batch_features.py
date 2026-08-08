"""
Unit tests for ml/batch_features.py batch feature computation.

These tests verify the correctness of batch processing, including the CRITICAL
anti-leakage safeguards that ensure each row only sees past data, not future data.
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from ml.batch_features import compute_features_batch


# =============================================================================
# Helper: Create small hand-crafted test DataFrames
# =============================================================================


def create_simple_batch_df():
    """
    Create a small hand-crafted DataFrame for testing.
    
    2 users, 5-6 rows each, controlled timestamps for testing.
    """
    rows = [
        # User A: 3 transactions
        {
            "transaction_id": "txn_a1",
            "user_id": "user_a",
            "amount": 1000.0,
            "device_id": "dev_1",
            "country": "US",
            "latitude": 40.7128,
            "longitude": -74.0060,
            "created_at": datetime(2026, 8, 1, 10, 0, 0),
            "label": 0,
        },
        {
            "transaction_id": "txn_a2",
            "user_id": "user_a",
            "amount": 1100.0,
            "device_id": "dev_1",
            "country": "US",
            "latitude": 40.7128,
            "longitude": -74.0060,
            "created_at": datetime(2026, 8, 1, 11, 0, 0),
            "label": 0,
        },
        {
            "transaction_id": "txn_a3",
            "user_id": "user_a",
            "amount": 5000.0,  # Large spike later
            "device_id": "dev_1",
            "country": "US",
            "latitude": 40.7128,
            "longitude": -74.0060,
            "created_at": datetime(2026, 8, 1, 20, 0, 0),
            "label": 0,
        },
        # User B: 3 transactions
        {
            "transaction_id": "txn_b1",
            "user_id": "user_b",
            "amount": 500.0,
            "device_id": "dev_2",
            "country": "IN",
            "latitude": 19.0760,
            "longitude": 72.8777,
            "created_at": datetime(2026, 8, 1, 9, 0, 0),
            "label": 0,
        },
        {
            "transaction_id": "txn_b2",
            "user_id": "user_b",
            "amount": 510.0,
            "device_id": "dev_2",
            "country": "IN",
            "latitude": 19.0760,
            "longitude": 72.8777,
            "created_at": datetime(2026, 8, 1, 15, 0, 0),
            "label": 0,
        },
        {
            "transaction_id": "txn_b3",
            "user_id": "user_b",
            "amount": 520.0,
            "device_id": "dev_2",
            "country": "IN",
            "latitude": 19.0760,
            "longitude": 72.8777,
            "created_at": datetime(2026, 8, 1, 22, 0, 0),
            "label": 0,
        },
    ]
    
    df = pd.DataFrame(rows)
    return df


# =============================================================================
# Test: Row count preservation
# =============================================================================


def test_compute_features_batch_row_count_preserved():
    """Test that output has the same number of rows as input."""
    df = create_simple_batch_df()
    input_row_count = len(df)
    
    features_df = compute_features_batch(
        df,
        user_id_col="user_id",
        amount_col="amount",
        device_id_col="device_id",
        country_col="country",
        latitude_col="latitude",
        longitude_col="longitude",
        timestamp_col="created_at",
    )
    
    output_row_count = len(features_df)
    assert output_row_count == input_row_count, (
        f"Row count mismatch: input={input_row_count}, output={output_row_count}"
    )


# =============================================================================
# Test: Feature column presence
# =============================================================================


def test_compute_features_batch_has_required_columns():
    """Test that output has exactly the 10 feature columns."""
    df = create_simple_batch_df()
    
    features_df = compute_features_batch(
        df,
        user_id_col="user_id",
        amount_col="amount",
        device_id_col="device_id",
        country_col="country",
        latitude_col="latitude",
        longitude_col="longitude",
        timestamp_col="created_at",
    )
    
    expected_feature_cols = {
        "txn_count_1h",
        "txn_count_24h",
        "amount_zscore",
        "amount_ratio_to_avg",
        "device_mismatch",
        "country_mismatch",
        "geo_distance_km",
        "amount",
        "hour_of_day",
        "day_of_week",
    }
    
    # Check that all expected columns are present
    for col in expected_feature_cols:
        assert col in features_df.columns, f"Missing feature column: {col}"
    
    # Also verify passthrough columns (user_id, label)
    assert "user_id" in features_df.columns, "Missing passthrough: user_id"
    assert "label" in features_df.columns, "Missing passthrough: label"


# =============================================================================
# CRITICAL TEST: No future leakage
# =============================================================================


def test_compute_features_batch_no_future_leakage():
    """
    THE MOST IMPORTANT TEST: Verify that a row does NOT see future data.
    
    Scenario:
    - User A has a small early transaction (1000) and a huge later transaction (5000)
    - The early transaction's amount_zscore should NOT reflect knowledge of the later spike
    - If it did, that would be data leakage (the model learned a pattern that doesn't exist
      in production, where we only have past data when scoring)
    
    This test directly validates the "no row sees future data" design from Task 3.
    """
    df = create_simple_batch_df()
    
    features_df = compute_features_batch(
        df,
        user_id_col="user_id",
        amount_col="amount",
        device_id_col="device_id",
        country_col="country",
        latitude_col="latitude",
        longitude_col="longitude",
        timestamp_col="created_at",
    )
    
    # Find User A's early transaction (txn_a1, created_at 10:00 on Aug 1)
    early_txn_row = features_df[features_df["transaction_id"] == "txn_a1"].iloc[0]
    early_zscore = early_txn_row["amount_zscore"]
    early_amount = early_txn_row["amount"]
    
    # Find User A's later transaction (txn_a3, created_at 20:00 on Aug 1)
    late_txn_row = features_df[features_df["transaction_id"] == "txn_a3"].iloc[0]
    late_zscore = late_txn_row["amount_zscore"]
    late_amount = late_txn_row["amount"]
    
    print(f"Early txn: amount={early_amount}, zscore={early_zscore}")
    print(f"Late txn: amount={late_amount}, zscore={late_zscore}")
    
    # The early transaction's zscore should be close to 0
    # (it's the first in the batch, so avg=1000, std=0, zscore=0)
    assert early_zscore == pytest.approx(0.0, abs=0.01), (
        f"Early transaction (first in user history) should have zscore ~0, got {early_zscore}. "
        f"This suggests the model saw future data (the 5000 spike later)!"
    )
    
    # The late transaction's zscore should be high
    # (avg of 1000 and 1100 = 1050, std > 0, so 5000 is several std devs away)
    assert late_zscore > 5.0, (
        f"Late transaction (5000) should have high zscore > 5.0, got {late_zscore}. "
        f"Check that it's computed from prior txns only, not including future data."
    )
    
    # The difference proves no leakage: early txn is normal, late txn is flagged
    assert early_zscore < late_zscore, (
        f"Early zscore should be < late zscore. Got early={early_zscore}, late={late_zscore}. "
        f"This pattern proves no leakage: early txn doesn't 'know' about the later spike."
    )


# =============================================================================
# Test: First transaction gets fallback values
# =============================================================================


def test_compute_features_batch_first_transaction_fallback():
    """
    Test that a user's very first transaction gets fallback values.
    
    This should match the single-transaction behavior tested in test_features.py.
    """
    df = create_simple_batch_df()
    
    features_df = compute_features_batch(
        df,
        user_id_col="user_id",
        amount_col="amount",
        device_id_col="device_id",
        country_col="country",
        latitude_col="latitude",
        longitude_col="longitude",
        timestamp_col="created_at",
    )
    
    # User A's first transaction: txn_a1 (created 2026-08-01 10:00)
    first_txn_a = features_df[features_df["transaction_id"] == "txn_a1"].iloc[0]
    
    # For the very first transaction, velocity should be 0
    assert first_txn_a["txn_count_1h"] == 0, (
        f"First txn should have txn_count_1h=0, got {first_txn_a['txn_count_1h']}"
    )
    assert first_txn_a["txn_count_24h"] == 0, (
        f"First txn should have txn_count_24h=0, got {first_txn_a['txn_count_24h']}"
    )
    
    # Amount deviation should be 0 (no baseline yet)
    assert first_txn_a["amount_zscore"] == 0.0, (
        f"First txn should have amount_zscore=0.0, got {first_txn_a['amount_zscore']}"
    )
    assert first_txn_a["amount_ratio_to_avg"] == 0.0, (
        f"First txn should have amount_ratio_to_avg=0.0, got {first_txn_a['amount_ratio_to_avg']}"
    )
    
    # Device mismatch should be 1 (new user, no prior devices)
    assert first_txn_a["device_mismatch"] == 1, (
        f"First txn should have device_mismatch=1, got {first_txn_a['device_mismatch']}"
    )
    
    # Country mismatch should be 0 (no home country yet, so no mismatch)
    assert first_txn_a["country_mismatch"] == 0, (
        f"First txn should have country_mismatch=0, got {first_txn_a['country_mismatch']}"
    )
    
    # Geo distance should be 0.0 (no prior location, so 0 distance)
    assert first_txn_a["geo_distance_km"] == 0.0, (
        f"First txn should have geo_distance_km=0.0, got {first_txn_a['geo_distance_km']}"
    )


# =============================================================================
# Test: Input DataFrame not mutated
# =============================================================================


def test_compute_features_batch_does_not_mutate_input():
    """Test that the input DataFrame is not modified."""
    df = create_simple_batch_df()
    df_copy = df.copy()
    
    # Call the function
    features_df = compute_features_batch(
        df,
        user_id_col="user_id",
        amount_col="amount",
        device_id_col="device_id",
        country_col="country",
        latitude_col="latitude",
        longitude_col="longitude",
        timestamp_col="created_at",
    )
    
    # Verify input is unchanged
    pd.testing.assert_frame_equal(df, df_copy, check_dtype=True, check_names=True)

"""
Unit tests for ml/features.py low-level and high-level feature functions.

These tests verify the correctness of individual feature computation functions
and the combined compute_features() wrapper, with emphasis on edge cases and
the prevention of data leakage in training scenarios.
"""

import pytest
from datetime import datetime, timedelta
from ml.features import (
    compute_velocity,
    compute_amount_deviation,
    compute_device_mismatch,
    compute_country_mismatch,
    compute_geo_distance,
    compute_features,
    TransactionInput,
    UserProfile,
)


# =============================================================================
# Tests for compute_velocity()
# =============================================================================


def test_compute_velocity_happy_path():
    """Test velocity calculation with a known set of timestamps in a window."""
    base_time = datetime(2026, 8, 7, 12, 0, 0)
    
    # Create timestamps: 0.5 hours ago, 2 hours ago, 25 hours ago
    prior_timestamps = [
        base_time - timedelta(minutes=30),    # 0.5 hours ago - should be IN 1h window
        base_time - timedelta(hours=2),       # 2 hours ago - should be IN 24h window but NOT 1h
        base_time - timedelta(hours=25),      # 25 hours ago - should be OUT of 24h window
    ]
    
    # In 1-hour window: should count only the 30-minute-ago transaction
    count_1h = compute_velocity(prior_timestamps, base_time, 1.0)
    assert count_1h == 1, f"Expected 1 transaction in 1-hour window, got {count_1h}"
    
    # In 24-hour window: should count the 0.5h and 2h transactions, not the 25h
    count_24h = compute_velocity(prior_timestamps, base_time, 24.0)
    assert count_24h == 2, f"Expected 2 transactions in 24-hour window, got {count_24h}"


def test_compute_velocity_empty_list():
    """Test that empty timestamp list returns 0."""
    base_time = datetime(2026, 8, 7, 12, 0, 0)
    count = compute_velocity([], base_time, 1.0)
    assert count == 0, f"Expected 0 for empty list, got {count}"


def test_compute_velocity_current_timestamp_is_counted():
    """
    Test of actual behavior: if current_timestamp appears in the list, it IS counted.
    
    NOTE: This is actually a potential bug/issue. The function docstring says:
      "MUST NOT include the current transaction being scored"
    But the implementation uses >= comparison, so if the current timestamp is
    accidentally in the list, it WILL be counted.
    
    In practice, this shouldn't happen because batch_features.py and the live API
    correctly exclude the current row when building recent_txn_timestamps. However,
    this test documents the actual behavior, which is: the timestamp IS counted if present.
    """
    base_time = datetime(2026, 8, 7, 12, 0, 0)
    
    # Manually construct a list that accidentally includes the current time
    prior_timestamps = [
        base_time - timedelta(minutes=30),
        base_time,  # The current transaction time
        base_time - timedelta(hours=2),
    ]
    
    count = compute_velocity(prior_timestamps, base_time, 24.0)
    # Actual behavior: counts 3 (including the current timestamp due to >= comparison)
    assert count == 3, (
        f"Expected 3 (including base_time due to >= semantics), got {count}"
    )


def test_compute_velocity_boundary_inclusive():
    """
    Test that a timestamp exactly on the window boundary is included (>= semantics).
    
    Window is [current_time - window_hours, current_time), so a transaction
    exactly window_hours ago should be included.
    """
    base_time = datetime(2026, 8, 7, 12, 0, 0)
    
    # A transaction exactly 1 hour ago
    prior_timestamps = [base_time - timedelta(hours=1.0)]
    
    count = compute_velocity(prior_timestamps, base_time, 1.0)
    # Should be included (window uses >= semantics)
    assert count == 1, (
        f"Transaction at window boundary should be included (>= semantics). "
        f"Expected 1, got {count}"
    )


# =============================================================================
# Tests for compute_amount_deviation()
# =============================================================================


def test_compute_amount_deviation_happy_path():
    """
    Test amount deviation with known values.
    
    Example: user avg=1000, std=200, current amount=1200
    zscore = (1200 - 1000) / 200 = 200 / 200 = 1.0
    ratio = 1200 / 1000 = 1.2
    """
    amount = 1200.0
    user_avg = 1000.0
    user_std = 200.0
    
    result = compute_amount_deviation(amount, user_avg, user_std)
    
    assert result["amount_zscore"] == pytest.approx(1.0), (
        f"Expected zscore=1.0, got {result['amount_zscore']}"
    )
    assert result["amount_ratio_to_avg"] == pytest.approx(1.2), (
        f"Expected ratio=1.2, got {result['amount_ratio_to_avg']}"
    )


def test_compute_amount_deviation_zero_std():
    """
    Test that zero standard deviation returns amount_zscore=0.0, not error/inf/nan.
    
    This is the "new user with no variance" case.
    """
    amount = 1500.0
    user_avg = 1000.0
    user_std = 0.0  # No variance
    
    result = compute_amount_deviation(amount, user_avg, user_std)
    
    # Should return 0.0, not raise an error or return inf/nan
    assert result["amount_zscore"] == 0.0, (
        f"Expected zscore=0.0 for zero std, got {result['amount_zscore']}"
    )
    assert isinstance(result["amount_zscore"], float), (
        f"Expected float, got {type(result['amount_zscore'])}"
    )


def test_compute_amount_deviation_zero_avg():
    """
    Test that zero average amount returns amount_ratio_to_avg=0.0, not error/inf/nan.
    
    This is the "brand-new user" case where we have no baseline.
    """
    amount = 500.0
    user_avg = 0.0  # No transactions yet
    user_std = 0.0
    
    result = compute_amount_deviation(amount, user_avg, user_std)
    
    # Should return 0.0, not raise a ZeroDivisionError or return inf/nan
    assert result["amount_ratio_to_avg"] == 0.0, (
        f"Expected ratio=0.0 for zero avg, got {result['amount_ratio_to_avg']}"
    )
    assert isinstance(result["amount_ratio_to_avg"], float), (
        f"Expected float, got {type(result['amount_ratio_to_avg'])}"
    )


# =============================================================================
# Tests for compute_device_mismatch()
# =============================================================================


def test_compute_device_mismatch_known_device():
    """Test that a device in the known list returns 0 (no mismatch)."""
    current_device = "dev_primary"
    known_devices = ["dev_primary", "dev_mobile"]
    
    result = compute_device_mismatch(current_device, known_devices)
    assert result == 0, f"Expected 0 for known device, got {result}"


def test_compute_device_mismatch_unknown_device():
    """Test that a device NOT in the known list returns 1 (mismatch)."""
    current_device = "dev_new"
    known_devices = ["dev_primary", "dev_mobile"]
    
    result = compute_device_mismatch(current_device, known_devices)
    assert result == 1, f"Expected 1 for unknown device, got {result}"


def test_compute_device_mismatch_empty_known_list():
    """
    Test that an empty known_device_ids list returns 1.
    
    This is the "brand-new user" case: all devices are technically unknown.
    """
    current_device = "dev_any"
    known_devices = []
    
    result = compute_device_mismatch(current_device, known_devices)
    assert result == 1, (
        f"Expected 1 for empty known_devices (new user), got {result}"
    )


# =============================================================================
# Tests for compute_country_mismatch()
# =============================================================================


def test_compute_country_mismatch_same_country():
    """Test that matching countries return 0 (no mismatch)."""
    current_country = "US"
    home_country = "US"
    
    result = compute_country_mismatch(current_country, home_country)
    assert result == 0, f"Expected 0 for matching countries, got {result}"


def test_compute_country_mismatch_different_country():
    """Test that different countries return 1 (mismatch)."""
    current_country = "SG"
    home_country = "US"
    
    result = compute_country_mismatch(current_country, home_country)
    assert result == 1, f"Expected 1 for different countries, got {result}"


def test_compute_country_mismatch_none_home_country():
    """
    Test that None home_country returns 0 (no mismatch).
    
    This is the "brand-new user" case: they have no established home country yet.
    """
    current_country = "IN"
    home_country = None
    
    result = compute_country_mismatch(current_country, home_country)
    assert result == 0, (
        f"Expected 0 for None home_country (new user), got {result}"
    )


# =============================================================================
# Tests for compute_geo_distance()
# =============================================================================


def test_compute_geo_distance_real_world_example():
    """
    Test haversine distance on a real-world example: New York to London.
    
    Real-world distance: approximately 5570 km
    We use pytest.approx to allow for small floating-point variance.
    """
    # New York coordinates
    lat1, lon1 = 40.7128, -74.0060
    # London coordinates
    lat2, lon2 = 51.5074, -0.1278
    
    distance = compute_geo_distance(lat1, lon1, lat2, lon2)
    
    # Assert within ~50 km of the real distance (to account for Earth model variations)
    assert distance == pytest.approx(5570, abs=50), (
        f"Expected ~5570 km for NY-London, got {distance}"
    )


def test_compute_geo_distance_identical_coordinates():
    """Test that identical coordinates return approximately 0 distance."""
    lat, lon = 40.7128, -74.0060
    
    distance = compute_geo_distance(lat, lon, lat, lon)
    
    # Should be very close to 0 (allowing for floating-point arithmetic)
    assert distance == pytest.approx(0.0, abs=0.01), (
        f"Expected ~0 km for identical coordinates, got {distance}"
    )


def test_compute_geo_distance_none_coordinates():
    """
    Test that None coordinates return 0.0.
    
    This is the "brand-new user" case: no prior location history.
    """
    lat1, lon1 = 40.7128, -74.0060
    # Current transaction has no prior location
    lat2, lon2 = None, None
    
    distance = compute_geo_distance(lat1, lon1, lat2, lon2)
    assert distance == 0.0, (
        f"Expected 0.0 for None coordinates, got {distance}"
    )


# =============================================================================
# Tests for compute_features() - the combined wrapper
# =============================================================================


def test_compute_features_contract_exact_keys():
    """
    Contract test: verify that compute_features() returns EXACTLY the 10 expected keys.
    
    This test should FAIL loudly if any key is added, removed, or renamed.
    """
    transaction = TransactionInput(
        transaction_id="txn_001",
        amount=1500.0,
        device_id="dev_primary",
        country="US",
        latitude=40.7128,
        longitude=-74.0060,
        created_at=datetime(2026, 8, 7, 14, 30, 0),
    )
    
    profile = UserProfile(
        user_id="user_001",
        avg_amount=1000.0,
        std_amount=200.0,
        known_device_ids=["dev_primary"],
        home_country="US",
        home_latitude=40.7128,
        home_longitude=-74.0060,
        recent_txn_timestamps=[datetime(2026, 8, 7, 13, 0, 0)],
    )
    
    features = compute_features(transaction, profile)
    
    expected_keys = {
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
    
    actual_keys = set(features.keys())
    assert actual_keys == expected_keys, (
        f"Feature keys mismatch.\n"
        f"Expected: {sorted(expected_keys)}\n"
        f"Got: {sorted(actual_keys)}\n"
        f"Missing: {sorted(expected_keys - actual_keys)}\n"
        f"Extra: {sorted(actual_keys - expected_keys)}"
    )


def test_compute_features_brand_new_user_no_exception():
    """
    Test that a brand-new user (empty history) does not raise any exception.
    
    All fallback values should be sensible (0 or 0.0).
    """
    transaction = TransactionInput(
        transaction_id="txn_new",
        amount=2000.0,
        device_id="dev_new",
        country="IN",
        latitude=19.0760,
        longitude=72.8777,
        created_at=datetime(2026, 8, 7, 10, 0, 0),
    )
    
    # Brand-new user: no history, no prior location, no home country
    profile = UserProfile(
        user_id="user_new",
        avg_amount=0.0,
        std_amount=0.0,
        known_device_ids=[],
        home_country=None,
        home_latitude=None,
        home_longitude=None,
        recent_txn_timestamps=[],
    )
    
    # This should NOT raise an exception
    features = compute_features(transaction, profile)
    
    # Verify all features are present and numeric
    assert len(features) == 10, f"Expected 10 features, got {len(features)}"
    
    # Verify fallback values for a brand-new user
    assert features["txn_count_1h"] == 0, "New user should have 0 txn_count_1h"
    assert features["txn_count_24h"] == 0, "New user should have 0 txn_count_24h"
    assert features["amount_zscore"] == 0.0, "New user should have 0.0 amount_zscore"
    assert features["amount_ratio_to_avg"] == 0.0, "New user should have 0.0 amount_ratio_to_avg"
    assert features["device_mismatch"] == 1, "New user device should be flagged as unknown"
    assert features["country_mismatch"] == 0, "New user should have 0 country_mismatch"
    assert features["geo_distance_km"] == 0.0, "New user should have 0.0 geo_distance"
    
    # Verify all values are numeric (not None or NaN)
    for key, value in features.items():
        assert isinstance(value, (int, float)), (
            f"Feature '{key}' should be numeric, got {type(value)}: {value}"
        )


def test_compute_features_invalid_created_at():
    """
    Test that compute_features() raises an AttributeError if created_at is a string
    instead of a datetime object. This documents that date parsing must happen
    upstream (e.g., via Pydantic or Pandas).
    """
    transaction = TransactionInput(
        transaction_id="txn_002",
        amount=100.0,
        device_id="dev_1",
        country="US",
        latitude=40.7,
        longitude=-74.0,
        created_at="2026-08-07T12:00:00Z",  # String instead of datetime
    )
    profile = UserProfile(
        user_id="user_002",
        avg_amount=0.0,
        std_amount=0.0,
        known_device_ids=[],
        home_country=None,
        home_latitude=None,
        home_longitude=None,
        recent_txn_timestamps=[],
    )
    
    with pytest.raises(AttributeError):
        compute_features(transaction, profile)

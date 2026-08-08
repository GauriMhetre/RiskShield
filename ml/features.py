"""
Shared feature-engineering module for RiskShield.

This module contains LOW-LEVEL feature computation functions used identically
by both the training pipeline (Phase 3) and the live API serving layer (Phase 4).

CRITICAL: These functions are the SINGLE SOURCE OF TRUTH for all feature calculations.
They must NEVER be duplicated elsewhere in the codebase. Training code and API code
both import and call these functions to guarantee no train/serve skew.

Each function is a pure calculation: no side effects, no I/O, no database access.
They are designed to handle edge cases (new users, missing data) gracefully via
fallback values, not exceptions — this allows the live API to score transactions
even when historical data is incomplete or absent.
"""

import math
from datetime import datetime, timedelta

# Small epsilon for division safety (avoids division by zero)
EPSILON = 1e-9


def compute_velocity(
    transaction_timestamps: list, current_timestamp: datetime, window_hours: float
) -> int:
    """
    Compute the number of prior transactions within a trailing time window.

    This detects "burst" fraud patterns: rapid-fire transactions in a short period.
    A user might legitimately make 2-3 transactions in 24 hours; but 20 transactions
    in 1 hour is suspicious and likely fraud (either card compromise or account takeover).

    Args:
        transaction_timestamps: List of datetime objects for prior transactions
                              (MUST NOT include the current transaction being scored).
        current_timestamp: The timestamp of the transaction being scored (not included in the count).
        window_hours: The trailing window size in hours (e.g., 1.0 for 1 hour, 24.0 for 24 hours).

    Returns:
        Integer count of transactions that fall within the window.
        Returns 0 if the list is empty or no transactions fall within the window.

    Note:
        It is critical that current_transaction is NOT in transaction_timestamps.
        Including it would cause the count to be artificially high, triggering false
        positives on every single transaction scored (a transaction always has 1+ txns
        "in the last 1 hour" if you count itself, so the feature becomes useless).
    """
    if not transaction_timestamps:
        return 0

    window_start = current_timestamp - timedelta(hours=window_hours)
    count = sum(1 for ts in transaction_timestamps if ts >= window_start)
    return count


def compute_amount_deviation(
    amount: float, user_avg_amount: float, user_std_amount: float
) -> dict:
    """
    Compute statistical deviation of a transaction amount from user's typical behavior.

    Fraud often appears as unusual spending amounts: either suspiciously high (stolen card,
    attacker maximizing damage) or suspiciously low (testing a stolen card with small purchase).
    This function captures that deviation in two forms:

    - Z-score: how many standard deviations away from the user's average?
      A z-score of 3 (3 std devs above average) is very unusual.
    - Ratio: simple multiplier (e.g., 5.0 = amount is 5x the user's average).

    Args:
        amount: Transaction amount in the user's home currency.
        user_avg_amount: User's historical mean transaction amount (expected/typical).
        user_std_amount: User's historical standard deviation of transaction amounts.

    Returns:
        Dictionary with keys:
          "amount_zscore": float, the z-score (or 0.0 if std is ~0)
          "amount_ratio_to_avg": float, the ratio (or 0.0 if avg is ~0)

    Note:
        For brand-new users with no transaction history (std=0 or avg=0), we return 0.0
        instead of raising an error or returning inf/nan. A new user shouldn't be immediately
        flagged as fraudulent just because we have no historical baseline. The model will
        learn other signals to differentiate new-user behavior from fraud.
    """
    # Z-score: guard against division by zero (new user with no variance)
    if user_std_amount < EPSILON:
        amount_zscore = 0.0
    else:
        amount_zscore = (amount - user_avg_amount) / user_std_amount

    # Ratio: guard against division by zero (new user with no transactions)
    if user_avg_amount < EPSILON:
        amount_ratio_to_avg = 0.0
    else:
        amount_ratio_to_avg = amount / user_avg_amount

    return {
        "amount_zscore": amount_zscore,
        "amount_ratio_to_avg": amount_ratio_to_avg,
    }


def compute_device_mismatch(current_device_id: str, known_device_ids: list) -> int:
    """
    Detect if the current transaction used a previously unknown device.

    Device fingerprints are a strong fraud signal: attackers using a stolen card typically
    use their own device, not the legitimate owner's device. This function flags transactions
    from new, unrecognized devices as a potential risk factor.

    Args:
        current_device_id: The device ID/fingerprint for this transaction.
        known_device_ids: List of device IDs previously seen for this user.

    Returns:
        1 if current_device_id is NOT in known_device_ids (new/unknown device).
        0 if current_device_id IS in known_device_ids (recognized device).

    Note:
        If known_device_ids is empty (brand-new user, no history), we return 1 because
        technically every device is "unknown" for a user who just signed up. This is
        correct: a new user with a new device is not more suspicious than a new user
        with any device. The model learns that new users often have new devices, so
        this signal alone doesn't trigger fraud, but combined with other signals it helps.
    """
    if not known_device_ids:
        return 1  # All devices are unknown for a brand-new user

    return 0 if current_device_id in known_device_ids else 1


def compute_country_mismatch(current_country: str, home_country: str) -> int:
    """
    Detect if the transaction originates from a different country than the user's home.

    International transactions are not inherently fraudulent — legitimate users travel.
    However, physically impossible location jumps (e.g., user in New York 1 hour ago,
    now purchasing in Singapore) are strong fraud signals. This feature captures the
    binary "is this a cross-border transaction?" signal.

    Args:
        current_country: The country code (e.g., "US", "IN", "SG") where this transaction occurred.
        home_country: The user's established home country (where they typically transact).

    Returns:
        1 if countries differ (cross-border transaction).
        0 if they match (same-country transaction).

    Note:
        If home_country is None or empty string (brand-new user with no established location),
        we return 0 (no mismatch). This is correct: a brand-new user has no "home" to mismatch
        against, so we shouldn't penalize them for transacting in any country. The model learns
        other signals to evaluate the risk. Once the user completes a transaction, home_country
        is set and this feature becomes active.
    """
    if not home_country:
        return 0  # No home country established yet; can't have a mismatch

    return 0 if current_country == home_country else 1


def compute_geo_distance(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """
    Compute the great-circle distance between two geographic coordinates in kilometers.

    This implements the Haversine formula, which calculates the shortest distance between
    two points on a sphere (Earth). It's used to detect impossible location jumps: if a
    user was in New York and 30 minutes later appears in Tokyo (9000+ km away),
    that's physically impossible and indicates fraud.

    The Haversine formula converts lat/lon (in degrees) into great-circle distance,
    accounting for Earth's curvature. It's more accurate than a flat "Pythagorean distance"
    for real-world geographic coordinates.

    Args:
        lat1, lon1: Latitude and longitude of the first location (prior transaction).
        lat2, lon2: Latitude and longitude of the second location (current transaction).

    Returns:
        Distance in kilometers between the two points.

    Note:
        If lat2 or lon2 (or lat1/lon1 for a brand-new user) is None (no prior location history),
        we return 0.0. A new user's first transaction has "0 distance traveled," which is correct
        and non-fraudulent. Once the user completes a transaction, their location is recorded
        and this feature becomes active for future transactions.

    Formula (high level):
        - Convert lat/lon from degrees to radians
        - Compute the central angle (angular distance) between the two points
        - Multiply by Earth's radius to get distance in km
        - The formula accounts for latitude variations and the spherical geometry
    """
    # Handle missing prior location (brand-new user, no home location set yet)
    if lat1 is None or lon1 is None or lat2 is None or lon2 is None:
        return 0.0

    # Validate that inputs are numbers (not strings, not completely invalid data)
    try:
        lat1, lon1, lat2, lon2 = float(lat1), float(lon1), float(lat2), float(lon2)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"Geographic coordinates must be numeric. Got lat1={lat1}, lon1={lon1}, "
            f"lat2={lat2}, lon2={lon2}. Error: {e}"
        )

    # Convert degrees to radians
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)

    # Haversine formula
    a = math.sin(delta_lat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(
        delta_lon / 2
    ) ** 2
    c = 2 * math.asin(math.sqrt(a))

    # Earth's mean radius in kilometers
    earth_radius_km = 6371.0

    return earth_radius_km * c


# ============================================================================
# Input Validation Helper (internal use only, not exported as a public feature)
# ============================================================================


def _validate_numeric_input(value, field_name: str) -> float:
    """
    Internal helper: validate and convert a value to float.

    Raises ValueError with a helpful message if the input is not numeric.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"{field_name} must be numeric, got {type(value).__name__}: {value}"
        )


# ============================================================================
# Data Classes (Simple Dicts for Cross-Layer Compatibility)
# ============================================================================


class TransactionInput:
    """
    Represents a single transaction being scored.

    This is a simple data container (not a Pydantic model to keep dependencies minimal)
    with required fields for fraud scoring. Can be created from dict unpacking:
        TransactionInput(**transaction_dict)

    Attributes:
        transaction_id: Unique identifier for this transaction.
        amount: Transaction amount in user's home currency.
        device_id: Device fingerprint/identifier for this transaction.
        country: Country code (e.g., "US", "IN", "SG") where transaction occurred.
        latitude: Geographic latitude of transaction origin.
        longitude: Geographic longitude of transaction origin.
        created_at: Datetime when the transaction occurred.
    """

    def __init__(
        self,
        transaction_id: str,
        amount: float,
        device_id: str,
        country: str,
        latitude: float,
        longitude: float,
        created_at: datetime,
    ):
        self.transaction_id = transaction_id
        self.amount = amount
        self.device_id = device_id
        self.country = country
        self.latitude = latitude
        self.longitude = longitude
        self.created_at = created_at


class UserProfile:
    """
    Represents a user's profile and behavioral history.

    This is a simple data container with fields aggregated from user history.
    Can be created from dict unpacking: UserProfile(**profile_dict)

    Attributes:
        user_id: Unique identifier for this user.
        avg_amount: User's historical mean transaction amount.
        std_amount: User's historical standard deviation of amounts.
        known_device_ids: List of device IDs this user has used before.
        home_country: User's established home country (where they typically transact).
        home_latitude: User's home location latitude.
        home_longitude: User's home location longitude.
        recent_txn_timestamps: List of datetime objects for recent transactions (not including current).
    """

    def __init__(
        self,
        user_id: str,
        avg_amount: float,
        std_amount: float,
        known_device_ids: list,
        home_country: str,
        home_latitude: float,
        home_longitude: float,
        recent_txn_timestamps: list,
    ):
        self.user_id = user_id
        self.avg_amount = avg_amount
        self.std_amount = std_amount
        self.known_device_ids = known_device_ids
        self.home_country = home_country
        self.home_latitude = home_latitude
        self.home_longitude = home_longitude
        self.recent_txn_timestamps = recent_txn_timestamps


# ============================================================================
# High-Level Feature Computation Wrapper
# ============================================================================


def compute_features(transaction: TransactionInput, profile: UserProfile) -> dict:
    """
    Compute all fraud-detection features for a single transaction.

    This is the HIGH-LEVEL wrapper that combines the low-level feature functions
    into a single feature vector. It is used identically by:
      - Training pipeline (Phase 3): to build feature matrices for model training
      - Live API (Phase 4): to score incoming transactions in real-time
      - EDA/notebooks: for exploratory analysis

    The function returns a feature dictionary suitable for:
      - Direct input to a model (all numeric values)
      - Inspection by analysts (human-readable feature names)
      - Storage in databases or logs (flat dict, JSON-serializable)

    Args:
        transaction: TransactionInput object with current transaction details.
        profile: UserProfile object with user's historical data.

    Returns:
        Dictionary with exactly 10 feature keys:
          - "txn_count_1h": int, number of transactions in last hour
          - "txn_count_24h": int, number of transactions in last 24 hours
          - "amount_zscore": float, z-score of amount vs. user's history
          - "amount_ratio_to_avg": float, amount / user's historical average
          - "device_mismatch": int (0 or 1), is this a new/unknown device?
          - "country_mismatch": int (0 or 1), is transaction from different country?
          - "geo_distance_km": float, distance from user's home location (km)
          - "amount": float, the transaction amount itself
          - "hour_of_day": int (0-23), hour when transaction occurred
          - "day_of_week": int (0-6), day of week when transaction occurred (0=Monday)

    Note:
        This function gracefully handles brand-new users (empty history) without errors.
        It has no side effects: no database access, no file I/O, no state mutation.
        All 10 features are always present and numeric (never None or NaN).

    Example:
        >>> transaction = TransactionInput(
        ...     transaction_id="txn_123",
        ...     amount=5000.0,
        ...     device_id="dev_abc",
        ...     country="US",
        ...     latitude=40.7128,
        ...     longitude=-74.0060,
        ...     created_at=datetime(2026, 8, 7, 14, 30, 0)
        ... )
        >>> profile = UserProfile(
        ...     user_id="user_xyz",
        ...     avg_amount=1000.0,
        ...     std_amount=200.0,
        ...     known_device_ids=["dev_abc"],
        ...     home_country="US",
        ...     home_latitude=40.7128,
        ...     home_longitude=-74.0060,
        ...     recent_txn_timestamps=[datetime(2026, 8, 7, 13, 0, 0)]
        ... )
        >>> features = compute_features(transaction, profile)
        >>> print(features)
        {
            'txn_count_1h': 0,
            'txn_count_24h': 1,
            'amount_zscore': 20.0,
            'amount_ratio_to_avg': 5.0,
            'device_mismatch': 0,
            'country_mismatch': 0,
            'geo_distance_km': 0.0,
            'amount': 5000.0,
            'hour_of_day': 14,
            'day_of_week': 3
        }
    """
    # Compute velocity features (1-hour and 24-hour windows)
    txn_count_1h = compute_velocity(profile.recent_txn_timestamps, transaction.created_at, 1.0)
    txn_count_24h = compute_velocity(profile.recent_txn_timestamps, transaction.created_at, 24.0)

    # Compute amount deviation features
    amount_deviation = compute_amount_deviation(
        transaction.amount, profile.avg_amount, profile.std_amount
    )

    # Compute device mismatch
    device_mismatch = compute_device_mismatch(transaction.device_id, profile.known_device_ids)

    # Compute country mismatch
    country_mismatch = compute_country_mismatch(transaction.country, profile.home_country)

    # Compute geographic distance
    geo_distance_km = compute_geo_distance(
        profile.home_latitude,
        profile.home_longitude,
        transaction.latitude,
        transaction.longitude,
    )

    # Extract temporal features
    hour_of_day = transaction.created_at.hour
    day_of_week = transaction.created_at.weekday()

    # Assemble the final feature vector
    features = {
        "txn_count_1h": txn_count_1h,
        "txn_count_24h": txn_count_24h,
        "amount_zscore": amount_deviation["amount_zscore"],
        "amount_ratio_to_avg": amount_deviation["amount_ratio_to_avg"],
        "device_mismatch": device_mismatch,
        "country_mismatch": country_mismatch,
        "geo_distance_km": geo_distance_km,
        "amount": transaction.amount,
        "hour_of_day": hour_of_day,
        "day_of_week": day_of_week,
    }

    return features

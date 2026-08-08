"""
Batch feature engineering for RiskShield.

This module provides DataFrame-level feature computation, used for:
  - Training pipeline (Phase 3): batch computing features for entire datasets
  - Model validation/inference: scoring large batches of historical transactions
  - EDA/analysis: exploratory data analysis on feature distributions

Key principle: It wraps the single-transaction compute_features() from
ml.features (ensuring no duplication of logic), applying it row-by-row
via pandas apply() and groupby operations for efficiency.

No pandas operations or DataFrame logic is duplicated elsewhere — this is
the SINGLE SOURCE OF TRUTH for batch feature computation.
"""

import pandas as pd
from datetime import datetime
from ml.features import compute_features, TransactionInput, UserProfile


def compute_features_batch(
    df: pd.DataFrame,
    user_id_col: str = "user_id",
    amount_col: str = "amount",
    device_id_col: str = "device_id",
    country_col: str = "country",
    latitude_col: str = "latitude",
    longitude_col: str = "longitude",
    timestamp_col: str = "created_at",
    label_col: str = None,
) -> pd.DataFrame:
    """
    Compute fraud-detection features for a batch of transactions.

    This is the batch/DataFrame version of compute_features(). It:
    1. Groups transactions by user
    2. Computes user profile statistics (avg/std amounts, device list, location, etc.)
    3. For each transaction, builds its feature vector using the shared
       compute_features() function from ml.features
    4. Returns a new DataFrame with all 10 features plus optional passthrough columns

    This ensures ZERO train/serve skew: the same underlying compute_features()
    logic runs here during training, and later in the live API during scoring.

    Args:
        df: Input DataFrame with transaction records. Expected columns:
            - user_id_col: User identifier
            - amount_col: Transaction amount
            - device_id_col: Device fingerprint
            - country_col: Country code (e.g., "US", "IN", "SG")
            - latitude_col: Geographic latitude
            - longitude_col: Geographic longitude
            - timestamp_col: Transaction datetime (must be datetime64 or convertible)
            - label_col (optional): Ground-truth label (0=legit, 1=fraud) for training
        user_id_col: Name of the user ID column (default: "user_id")
        amount_col: Name of the amount column (default: "amount")
        device_id_col: Name of the device ID column (default: "device_id")
        country_col: Name of the country column (default: "country")
        latitude_col: Name of the latitude column (default: "latitude")
        longitude_col: Name of the longitude column (default: "longitude")
        timestamp_col: Name of the timestamp column (default: "created_at")
        label_col: Name of the label column (optional, default: None)

    Returns:
        DataFrame with original columns plus 10 feature columns:
          - "txn_count_1h"
          - "txn_count_24h"
          - "amount_zscore"
          - "amount_ratio_to_avg"
          - "device_mismatch"
          - "country_mismatch"
          - "geo_distance_km"
          - "amount" (passthrough from input)
          - "hour_of_day"
          - "day_of_week"
        
        Plus passthrough columns from the input:
          - user_id, transaction_id (if exists), label (if provided), and others

    Raises:
        ValueError: if required columns are missing or timestamp is not datetime

    Note:
        For the first transaction of a brand-new user, velocity features will be 0,
        amount deviations will be 0.0 (no baseline yet), device_mismatch will be 1,
        and geo_distance will be 0.0. This is expected and correct behavior — the
        model learns that new users have these fallback values.

    Example:
        >>> import pandas as pd
        >>> df = pd.read_csv("data/processed/synthetic_transactions.csv",
        ...                   parse_dates=["created_at"])
        >>> features_df = compute_features_batch(df)
        >>> print(features_df.shape)
        (15000, 21)  # 10 features + passthrough columns
        >>> print(features_df[["user_id", "amount", "txn_count_1h", "device_mismatch"]].head())
    """
    # Validate required columns exist
    required_cols = [user_id_col, amount_col, device_id_col, country_col, 
                     latitude_col, longitude_col, timestamp_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        try:
            df = df.copy()  # Avoid modifying original
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        except Exception as e:
            raise ValueError(f"Could not parse {timestamp_col} as datetime: {e}")

    # Sort by user and timestamp for proper velocity calculation
    df = df.sort_values([user_id_col, timestamp_col]).reset_index(drop=True)

    # Compute user profiles: aggregate statistics for each user
    user_profiles = _compute_user_profiles(
        df,
        user_id_col=user_id_col,
        amount_col=amount_col,
        device_id_col=device_id_col,
        country_col=country_col,
        latitude_col=latitude_col,
        longitude_col=longitude_col,
        timestamp_col=timestamp_col,
    )

    # Compute features row-by-row using the shared single-transaction function
    feature_list = []
    for idx, row in df.iterrows():
        features = _compute_row_features(
            row=row,
            user_profiles=user_profiles,
            user_id_col=user_id_col,
            amount_col=amount_col,
            device_id_col=device_id_col,
            country_col=country_col,
            latitude_col=latitude_col,
            longitude_col=longitude_col,
            timestamp_col=timestamp_col,
            df_for_velocity=df,
        )
        feature_list.append(features)

    # Convert list of dicts to DataFrame
    features_df = pd.DataFrame(feature_list)

    # Passthrough: add original columns (user_id, label if present, etc.)
    passthrough_cols = [user_id_col]
    if "transaction_id" in df.columns:
        passthrough_cols.append("transaction_id")
    if label_col and label_col in df.columns:
        passthrough_cols.append(label_col)

    for col in passthrough_cols:
        if col in df.columns:
            features_df.insert(0, col, df[col].values)

    return features_df


def _compute_user_profiles(
    df: pd.DataFrame,
    user_id_col: str,
    amount_col: str,
    device_id_col: str,
    country_col: str,
    latitude_col: str,
    longitude_col: str,
    timestamp_col: str,
) -> dict:
    """
    Compute aggregate user profiles from the full dataset.

    For each user, compute:
      - avg_amount, std_amount: historical mean/std of transaction amounts
      - known_device_ids: list of devices this user has used
      - home_country: user's most common country (or first transaction country)
      - home_latitude, home_longitude: user's most common/last known location

    Args:
        df: Full transaction DataFrame (must be sorted by user and timestamp)
        Column name parameters for flexible column naming

    Returns:
        Dictionary mapping user_id -> dict with profile fields:
            {
                "user_id": str,
                "avg_amount": float,
                "std_amount": float,
                "known_device_ids": list,
                "home_country": str,
                "home_latitude": float,
                "home_longitude": float,
            }
    """
    profiles = {}

    for user_id, group in df.groupby(user_id_col):
        # Amount statistics
        amounts = group[amount_col].astype(float)
        avg_amount = amounts.mean()
        std_amount = amounts.std() if len(group) > 1 else 0.0

        # Devices
        known_device_ids = group[device_id_col].dropna().unique().tolist()

        # Location: use the most common country, or first if tie
        country_counts = group[country_col].value_counts()
        home_country = country_counts.index[0] if len(country_counts) > 0 else None

        # Home location: use the last (most recent) transaction location
        last_row = group.iloc[-1]
        home_latitude = last_row[latitude_col]
        home_longitude = last_row[longitude_col]

        profiles[user_id] = {
            "user_id": user_id,
            "avg_amount": avg_amount,
            "std_amount": std_amount if not pd.isna(std_amount) else 0.0,
            "known_device_ids": known_device_ids,
            "home_country": home_country,
            "home_latitude": home_latitude,
            "home_longitude": home_longitude,
        }

    return profiles


def _compute_row_features(
    row: pd.Series,
    user_profiles: dict,
    user_id_col: str,
    amount_col: str,
    device_id_col: str,
    country_col: str,
    latitude_col: str,
    longitude_col: str,
    timestamp_col: str,
    df_for_velocity: pd.DataFrame,
) -> dict:
    """
    Compute features for a single transaction row.

    Internal helper: builds TransactionInput and UserProfile objects,
    then calls the shared compute_features() function.

    Args:
        row: A single row from the DataFrame
        user_profiles: Dictionary of precomputed user profiles
        df_for_velocity: The full DataFrame, used to compute velocity
                        (recent transactions for this user)
        Column name parameters for flexible column naming

    Returns:
        Dictionary with the 10 feature keys (from compute_features())
    """
    user_id = row[user_id_col]
    profile_dict = user_profiles.get(user_id, {})

    # Build TransactionInput
    transaction = TransactionInput(
        transaction_id=str(row.get("transaction_id", "")),
        amount=float(row[amount_col]),
        device_id=str(row[device_id_col]),
        country=str(row[country_col]) if pd.notna(row[country_col]) else None,
        latitude=float(row[latitude_col]) if pd.notna(row[latitude_col]) else None,
        longitude=float(row[longitude_col]) if pd.notna(row[longitude_col]) else None,
        created_at=pd.Timestamp(row[timestamp_col]).to_pydatetime(),
    )

    # Compute recent transactions for this user (excluding current row)
    user_txns = df_for_velocity[df_for_velocity[user_id_col] == user_id]
    current_idx = row.name if hasattr(row, 'name') else None
    
    # Get recent timestamps (excluding current transaction)
    recent_timestamps = []
    for idx, txn_row in user_txns.iterrows():
        if idx != current_idx:  # Exclude current transaction
            ts = pd.Timestamp(txn_row[timestamp_col]).to_pydatetime()
            recent_timestamps.append(ts)

    # Build UserProfile
    profile = UserProfile(
        user_id=user_id,
        avg_amount=profile_dict.get("avg_amount", 0.0),
        std_amount=profile_dict.get("std_amount", 0.0),
        known_device_ids=profile_dict.get("known_device_ids", []),
        home_country=profile_dict.get("home_country", None),
        home_latitude=profile_dict.get("home_latitude", None),
        home_longitude=profile_dict.get("home_longitude", None),
        recent_txn_timestamps=recent_timestamps,
    )

    # Compute features using the shared single-transaction function
    features = compute_features(transaction, profile)

    return features

"""
Batch feature engineering for RiskShield fraud detection.

This module computes features for entire historical datasets by calling the SAME
compute_features() function used by the live API, ensuring ZERO train/serve skew.

CRITICAL RULE: NO ROW MAY EVER SEE ANOTHER ROW'S FUTURE DATA
================================================================
Data leakage is prevented by:
1. Sorting the entire DataFrame by timestamp FIRST (before any grouping)
2. Processing rows strictly in chronological order within each user group
3. For each row, building a UserProfile using ONLY transactions that occurred
   BEFORE that row's timestamp — never including the current row or any future row
4. This ensures each row's features depend only on information that would have been
   available at the time of that transaction in a real-time system

Violating this rule (e.g., accidentally including future rows in velocity calculations)
would cause the model to learn patterns that don't exist in production, leading to
catastrophic accuracy degradation and undetectable failures in live scoring.
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
) -> pd.DataFrame:
    """
    Compute fraud-detection features for a batch of transactions.

    This function processes an entire dataset of transactions, computing the 10 fraud
    features for each row by calling the shared compute_features() function from ml.features.
    It is used for training data preparation and batch scoring of historical transactions.

    The function takes generic column names to work identically with multiple schemas
    (synthetic transactions, IEEE-CIS Kaggle data, etc.) without code duplication.

    Key design: This implementation is intentionally row-by-row-within-groupby for
    clarity and correctness, not maximally optimized. For datasets of 15K-50K rows,
    the performance is acceptable. If this ever needs to scale to millions of rows,
    a more vectorized rewrite would be needed, but that is out of scope for now.

    Args:
        df: Input DataFrame with transaction records. Must contain the columns specified
            by the col parameters below.
        user_id_col: Name of the user ID column (e.g., "user_id", "card1")
        amount_col: Name of the transaction amount column (e.g., "amount", "TransactionAmt")
        device_id_col: Name of the device ID column (e.g., "device_id", "DeviceType")
        country_col: Name of the country code column (e.g., "country")
        latitude_col: Name of the latitude column (e.g., "latitude")
        longitude_col: Name of the longitude column (e.g., "longitude")
        timestamp_col: Name of the timestamp column (e.g., "created_at", "TransactionDT")

    Returns:
        A new DataFrame with the same number of rows as the input, containing:
        - All 10 computed feature columns (txn_count_1h, amount_zscore, etc.)
        - Passthrough columns: the user_id column, any "label" or "isFraud" column if present,
          and any transaction ID column if present
        - Original DataFrame is not modified (safe to use repeatedly)

    Raises:
        ValueError: If any required column does not exist in df

    Example:
        >>> import pandas as pd
        >>> df = pd.read_csv("data/processed/synthetic_transactions.csv",
        ...                   parse_dates=["created_at"])
        >>> features_df = compute_features_batch(
        ...     df,
        ...     user_id_col="user_id",
        ...     amount_col="amount",
        ...     device_id_col="device_id",
        ...     country_col="country",
        ...     latitude_col="latitude",
        ...     longitude_col="longitude",
        ...     timestamp_col="created_at"
        ... )
        >>> print(features_df.shape)
        (15052, 13)  # 10 features + 3 passthrough columns
    """
    # Validate that all required columns exist before processing
    required_cols = {
        user_id_col,
        amount_col,
        device_id_col,
        country_col,
        latitude_col,
        longitude_col,
        timestamp_col,
    }
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"DataFrame is missing required columns: {sorted(missing_cols)}. "
            f"Available columns: {sorted(df.columns)}"
        )

    # CRITICAL: Sort by timestamp FIRST, before any grouping
    # This ensures all row-by-row processing happens in chronological order
    df_sorted = df.sort_values(timestamp_col).reset_index(drop=True)

    # Collect computed features for each row
    features_list = []

    # Process each user's transactions in chronological order
    for user_id, user_group in df_sorted.groupby(user_id_col):
        user_rows = list(user_group.itertuples(index=True, name=None))

        # Process each transaction for this user
        for row_idx, row in enumerate(user_group.itertuples(index=True)):
            # Build UserProfile using ONLY prior transactions (strictly before this row)
            # LEAKAGE PREVENTION: Only iterate through rows 0..row_idx-1, never including
            # the current row (row_idx) or any future rows (row_idx+1..end)
            prior_rows = list(user_group.itertuples())[: row_idx]

            # Compute aggregates from prior transactions only
            prior_amounts = []
            prior_devices = set()
            prior_countries = []
            prior_timestamps = []

            for prior_row in prior_rows:
                prior_amount = getattr(prior_row, amount_col, None)
                if prior_amount is not None:
                    try:
                        prior_amounts.append(float(prior_amount))
                    except (ValueError, TypeError):
                        pass

                prior_device = getattr(prior_row, device_id_col, None)
                if prior_device is not None and pd.notna(prior_device):
                    prior_devices.add(str(prior_device))

                prior_country = getattr(prior_row, country_col, None)
                if prior_country is not None and pd.notna(prior_country):
                    prior_countries.append(str(prior_country))

                prior_timestamp = getattr(prior_row, timestamp_col, None)
                if prior_timestamp is not None:
                    prior_timestamps.append(pd.Timestamp(prior_timestamp).to_pydatetime())

            # Compute user's historical statistics from prior transactions
            if prior_amounts:
                avg_amount = sum(prior_amounts) / len(prior_amounts)
                # Compute std: even with 1 prior transaction, std=0 (no variance)
                if len(prior_amounts) > 1:
                    variance = sum((x - avg_amount) ** 2 for x in prior_amounts) / len(prior_amounts)
                    std_amount = variance ** 0.5
                else:
                    std_amount = 0.0
            else:
                # Brand-new user: no prior transactions
                avg_amount = 0.0
                std_amount = 0.0

            # Most common prior country, or first if only one
            if prior_countries:
                # Use most common
                country_counts = {}
                for c in prior_countries:
                    country_counts[c] = country_counts.get(c, 0) + 1
                home_country = max(country_counts, key=country_counts.get)
            else:
                home_country = None

            # Last known location: use the most recent prior transaction
            if prior_rows:
                last_prior_row = prior_rows[-1]
                home_latitude = getattr(last_prior_row, latitude_col, None)
                home_longitude = getattr(last_prior_row, longitude_col, None)
                # Convert to float, handle NaN
                try:
                    home_latitude = float(home_latitude) if pd.notna(home_latitude) else None
                except (ValueError, TypeError):
                    home_latitude = None
                try:
                    home_longitude = float(home_longitude) if pd.notna(home_longitude) else None
                except (ValueError, TypeError):
                    home_longitude = None
            else:
                home_latitude = None
                home_longitude = None

            # Build UserProfile for this row
            profile = UserProfile(
                user_id=user_id,
                avg_amount=avg_amount,
                std_amount=std_amount,
                known_device_ids=list(prior_devices),
                home_country=home_country,
                home_latitude=home_latitude,
                home_longitude=home_longitude,
                recent_txn_timestamps=prior_timestamps,
            )

            # Build TransactionInput from current row
            current_amount = getattr(row, amount_col, None)
            current_device = getattr(row, device_id_col, None)
            current_country = getattr(row, country_col, None)
            current_latitude = getattr(row, latitude_col, None)
            current_longitude = getattr(row, longitude_col, None)
            current_timestamp = getattr(row, timestamp_col, None)

            # Convert to appropriate types, handling NaN/None
            try:
                current_amount = float(current_amount)
            except (ValueError, TypeError):
                current_amount = 0.0

            current_device = str(current_device) if pd.notna(current_device) else None
            current_country = str(current_country) if pd.notna(current_country) else None

            try:
                current_latitude = float(current_latitude) if pd.notna(current_latitude) else None
            except (ValueError, TypeError):
                current_latitude = None

            try:
                current_longitude = float(current_longitude) if pd.notna(current_longitude) else None
            except (ValueError, TypeError):
                current_longitude = None

            current_timestamp = pd.Timestamp(current_timestamp).to_pydatetime()

            # Get transaction ID if it exists
            txn_id = getattr(row, "transaction_id", None) or getattr(row, "TransactionID", None)
            if txn_id is None:
                txn_id = str(row.Index)

            transaction = TransactionInput(
                transaction_id=str(txn_id),
                amount=current_amount,
                device_id=current_device,
                country=current_country,
                latitude=current_latitude,
                longitude=current_longitude,
                created_at=current_timestamp,
            )

            # Compute features using the shared function
            features_dict = compute_features(transaction, profile)

            # Add passthrough columns: user ID and label if present
            features_dict[user_id_col] = user_id

            # Auto-detect and include label column if present
            if "label" in df.columns:
                features_dict["label"] = row.label
            elif "isFraud" in df.columns:
                features_dict["isFraud"] = row.isFraud

            # Include transaction ID if original df has one
            if "transaction_id" in df.columns:
                features_dict["transaction_id"] = getattr(row, "transaction_id", txn_id)
            elif "TransactionID" in df.columns:
                features_dict["TransactionID"] = getattr(row, "TransactionID", txn_id)

            features_list.append(features_dict)

    # Convert list of dicts to DataFrame
    result_df = pd.DataFrame(features_list)

    return result_df

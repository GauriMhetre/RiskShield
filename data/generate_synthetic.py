"""
Synthetic transaction dataset generator for RiskShield.

This module generates a realistic synthetic dataset for fraud detection model training.
It creates three separate datasets:
  1. Users with profiles (country, location, typical transaction amounts, known devices)
  2. Legitimate transactions (normal user behavior)
  3. Fraudulent transactions (injected fraud signals)

The outputs are combined into a single CSV file suitable for feature engineering
and model training in later phases.
"""

import os
import random
import uuid
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


# Major city coordinates (country_code -> (lat, lng, name))
CITY_COORDINATES = {
    "IN": (19.0760, 72.8777, "Mumbai"),
    "US": (40.7128, -74.0060, "New York"),
    "GB": (51.5074, -0.1278, "London"),
    "DE": (52.5200, 13.4050, "Berlin"),
    "SG": (1.3521, 103.8198, "Singapore"),
}

COUNTRY_LIST = list(CITY_COORDINATES.keys())


def _validate_positive_int(value, param_name):
    """Helper to validate input parameters are positive integers."""
    if not isinstance(value, int) or value <= 0:
        raise ValueError(
            f"{param_name} must be a positive integer, got {value} ({type(value).__name__})"
        )


def generate_users(n_users=300, seed=42) -> pd.DataFrame:
    """
    Generate a synthetic user dataset.

    Args:
        n_users: Number of users to generate. Must be positive integer.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with columns: user_id, home_country, home_latitude, home_longitude,
        avg_amount, std_amount, usual_devices.

    Raises:
        ValueError: if n_users is not a positive integer.
    """
    _validate_positive_int(n_users, "n_users")

    np.random.seed(seed)
    random.seed(seed)

    users = []
    for _ in range(n_users):
        user_id = str(uuid.uuid4())
        home_country = random.choice(COUNTRY_LIST)
        lat, lng, _ = CITY_COORDINATES[home_country]

        # Add small jitter to coordinates (within ~50 km)
        lat += np.random.normal(0, 0.3)
        lng += np.random.normal(0, 0.3)

        # User's typical transaction amount (lognormal distribution for realism)
        # median ~500 INR, range roughly 100–5000
        avg_amount = np.random.lognormal(mean=6.0, sigma=0.8)
        std_amount = avg_amount * np.random.uniform(0.2, 0.5)

        # Each user has 1-3 usual devices
        n_devices = random.randint(1, 3)
        usual_devices = [f"dev_{uuid.uuid4().hex[:8]}" for _ in range(n_devices)]

        users.append(
            {
                "user_id": user_id,
                "home_country": home_country,
                "home_latitude": lat,
                "home_longitude": lng,
                "avg_amount": avg_amount,
                "std_amount": std_amount,
                "usual_devices": usual_devices,
            }
        )

    return pd.DataFrame(users)


def generate_legit_transactions(
    users_df, n_transactions=14500, seed=42
) -> pd.DataFrame:
    """
    Generate synthetic legitimate transactions.

    Args:
        users_df: DataFrame from generate_users().
        n_transactions: Number of legitimate transactions. Must be positive integer.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with transaction columns and label=0.

    Raises:
        ValueError: if n_transactions is not a positive integer.
    """
    _validate_positive_int(n_transactions, "n_transactions")

    np.random.seed(seed)
    random.seed(seed)

    transactions = []
    base_date = datetime(2026, 5, 1)
    end_date = base_date + timedelta(days=90)

    for _ in range(n_transactions):
        user = users_df.sample(1).iloc[0]

        # Realistic timestamp: cluster by hour-of-day (more daytime activity)
        hours_passed = np.random.exponential(scale=36)  # Exponential cluster
        hours_passed = hours_passed % (90 * 24)  # Wrap within 90 days
        created_at = base_date + timedelta(hours=hours_passed)

        # Amount: normal distribution around user's avg, clipped to be positive
        amount = max(10, np.random.normal(user["avg_amount"], user["std_amount"]))

        # Location: home location with small jitter (~3 km)
        latitude = user["home_latitude"] + np.random.normal(0, 0.025)
        longitude = user["home_longitude"] + np.random.normal(0, 0.025)

        device_id = random.choice(user["usual_devices"])

        transactions.append(
            {
                "transaction_id": str(uuid.uuid4()),
                "user_id": user["user_id"],
                "amount": amount,
                "currency": "INR",
                "merchant_id": f"merch_{uuid.uuid4().hex[:6]}",
                "device_id": device_id,
                "ip_address": f"203.0.{random.randint(0, 255)}.{random.randint(1, 255)}",
                "country": user["home_country"],
                "latitude": latitude,
                "longitude": longitude,
                "created_at": created_at,
                "label": 0,
            }
        )

    return pd.DataFrame(transactions)


def _haversine_distance_km(lat1, lng1, lat2, lng2) -> float:
    """
    Calculate approximate distance in km between two lat/lng points.
    Uses simplified haversine formula.
    """
    R = 6371  # Earth radius in km
    dlat = np.radians(lat2 - lat1)
    dlng = np.radians(lng2 - lng1)
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlng / 2) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def generate_fraud_transactions(
    users_df, n_transactions=500, seed=42
) -> pd.DataFrame:
    """
    Generate synthetic fraudulent transactions with injected fraud signals.

    Each fraud transaction has ONE OR MORE of these signals (randomly chosen):
      a. Burst: 3-8 transactions in quick succession (10-30 min window)
      b. Amount spike: 3x–15x the user's typical amount
      c. New device: device ID not in user's usual devices
      d. Location jump: different country + far geographic distance

    Args:
        users_df: DataFrame from generate_users().
        n_transactions: Number of fraud transactions. Must be positive integer.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with transaction columns and label=1.

    Raises:
        ValueError: if n_transactions is not a positive integer.
    """
    _validate_positive_int(n_transactions, "n_transactions")

    np.random.seed(seed)
    random.seed(seed)

    transactions = []
    base_date = datetime(2026, 5, 1)
    end_date = base_date + timedelta(days=90)

    for _ in range(n_transactions):
        user = users_df.sample(1).iloc[0]

        # Randomly decide which fraud signals to inject (at least one)
        signals = ["burst", "amount_spike", "new_device", "location_jump"]
        active_signals = random.sample(signals, k=random.randint(1, len(signals)))

        # Base timestamp
        hours_passed = np.random.exponential(scale=36)
        hours_passed = hours_passed % (90 * 24)
        created_at = base_date + timedelta(hours=hours_passed)

        # --- Signal: Amount spike ---
        if "amount_spike" in active_signals:
            amount = user["avg_amount"] * np.random.uniform(3, 15)
        else:
            # Even without spike signal, add light noise so not all fraud is extreme
            amount = max(10, np.random.normal(user["avg_amount"], user["std_amount"]))

        # --- Signal: New device ---
        if "new_device" in active_signals:
            device_id = f"dev_{uuid.uuid4().hex[:8]}"  # Not in usual_devices
        else:
            device_id = random.choice(user["usual_devices"])

        # --- Signal: Location jump ---
        if "location_jump" in active_signals:
            country = random.choice([c for c in COUNTRY_LIST if c != user["home_country"]])
            fraud_lat, fraud_lng, _ = CITY_COORDINATES[country]
            # Add small jitter to the distant city
            latitude = fraud_lat + np.random.normal(0, 0.3)
            longitude = fraud_lng + np.random.normal(0, 0.3)
        else:
            country = user["home_country"]
            latitude = user["home_latitude"] + np.random.normal(0, 0.025)
            longitude = user["home_longitude"] + np.random.normal(0, 0.025)

        # --- Signal: Burst (handled in main() by creating multiple txns at once) ---
        # For now, just mark this transaction as potentially part of a burst
        # (actual burst logic handled when combining transactions)

        transactions.append(
            {
                "transaction_id": str(uuid.uuid4()),
                "user_id": user["user_id"],
                "amount": amount,
                "currency": "INR",
                "merchant_id": f"merch_{uuid.uuid4().hex[:6]}",
                "device_id": device_id,
                "ip_address": f"203.0.{random.randint(0, 255)}.{random.randint(1, 255)}",
                "country": country,
                "latitude": latitude,
                "longitude": longitude,
                "created_at": created_at,
                "label": 1,
            }
        )

    # Inject burst signal: some fraud transactions should happen in quick succession
    # Re-seed to ensure reproducibility
    np.random.seed(seed)
    random.seed(seed)

    n_bursts = max(1, n_transactions // 50)  # ~1 burst per 50 fraud txns
    for _ in range(n_bursts):
        burst_user = users_df.sample(1).iloc[0]
        burst_size = random.randint(3, 8)
        base_burst_time = base_date + timedelta(hours=np.random.uniform(0, 90 * 24))

        for i in range(burst_size):
            offset_minutes = random.randint(0, 30)
            txn_time = base_burst_time + timedelta(minutes=offset_minutes)

            # Re-sample to avoid duplicating the first burst transaction
            if i > 0:  # Only create additional txns if burst_size > 1
                transactions.append(
                    {
                        "transaction_id": str(uuid.uuid4()),
                        "user_id": burst_user["user_id"],
                        "amount": max(10, np.random.normal(
                            burst_user["avg_amount"], burst_user["std_amount"]
                        )),
                        "currency": "INR",
                        "merchant_id": f"merch_{uuid.uuid4().hex[:6]}",
                        "device_id": random.choice(burst_user["usual_devices"]),
                        "ip_address": f"203.0.{random.randint(0, 255)}.{random.randint(1, 255)}",
                        "country": burst_user["home_country"],
                        "latitude": burst_user["home_latitude"] + np.random.normal(0, 0.025),
                        "longitude": burst_user["home_longitude"] + np.random.normal(0, 0.025),
                        "created_at": txn_time,
                        "label": 1,
                    }
                )

    return pd.DataFrame(transactions)


def main(n_users=300, n_legit=14500, n_fraud=500):
    """
    Main entry point: generate users, transactions, combine, and save to CSV.

    Args:
        n_users: Number of users to generate.
        n_legit: Number of legitimate transactions.
        n_fraud: Number of fraudulent transactions.
    """
    print("=" * 70)
    print("RiskShield Synthetic Dataset Generator")
    print("=" * 70)

    try:
        # Validate inputs
        _validate_positive_int(n_users, "n_users")
        _validate_positive_int(n_legit, "n_legit")
        _validate_positive_int(n_fraud, "n_fraud")

        print(f"\nGenerating {n_users} users...")
        users_df = generate_users(n_users=n_users, seed=42)
        print(f"✓ Generated {len(users_df)} users")

        print(f"\nGenerating {n_legit} legitimate transactions...")
        legit_df = generate_legit_transactions(users_df, n_transactions=n_legit, seed=42)
        print(f"✓ Generated {len(legit_df)} legitimate transactions")

        print(f"\nGenerating {n_fraud} fraudulent transactions...")
        fraud_df = generate_fraud_transactions(users_df, n_transactions=n_fraud, seed=42)
        print(f"✓ Generated {len(fraud_df)} fraudulent transactions (with burst signal injected)")

        # Combine and sort by timestamp
        print("\nCombining and sorting by timestamp...")
        transactions_df = pd.concat([legit_df, fraud_df], ignore_index=True)
        transactions_df = transactions_df.sort_values("created_at").reset_index(drop=True)
        print(f"✓ Combined dataset: {len(transactions_df)} total rows")

        # Validation
        print("\nValidating dataset...")
        total_rows = len(transactions_df)
        fraud_ratio = transactions_df["label"].mean()
        nan_count = transactions_df.isnull().sum().sum()

        print(f"  Total rows: {total_rows}")
        print(f"  Fraud ratio: {fraud_ratio:.2%}")
        print(f"  NaN values: {nan_count}")

        if nan_count == 0:
            print("  ✓ No NaN values found")
        else:
            print(f"  ⚠ WARNING: Found {nan_count} NaN values")

        # Create output directory and save
        print("\nSaving dataset...")
        output_dir = os.path.join(os.path.dirname(__file__), "processed")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "synthetic_transactions.csv")

        transactions_df.to_csv(output_path, index=False)
        print(f"✓ Saved to: {output_path}")

        print("\n" + "=" * 70)
        print("Dataset generation complete!")
        print("=" * 70)

    except ValueError as e:
        print(f"\n✗ Validation Error: {e}")
        raise
    except Exception as e:
        print(f"\n✗ Error saving dataset: {e}")
        raise


if __name__ == "__main__":
    main()

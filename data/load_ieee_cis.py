"""
IEEE-CIS Kaggle dataset loader for RiskShield.

This module loads and processes the IEEE-CIS Credit Card Fraud Detection
dataset from Kaggle. The raw files (train_transaction.csv and train_identity.csv)
are expected to be present in data/raw/ — they are not downloaded by this script.

The module provides utilities to:
  1. Load raw CSV files from data/raw/
  2. Join transaction and identity data
  3. Sample a manageable subset for model training
  4. Summarize data quality and structure
"""

import os
import pandas as pd


COLUMNS_TO_KEEP = [
    "TransactionID",
    "isFraud",
    "TransactionAmt",
    "TransactionDT",
    "card1",
    "DeviceType",
    "DeviceInfo",
    "addr1",
    "addr2",
    "P_emaildomain",
]


def load_raw(raw_dir="data/raw") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load raw transaction and identity CSV files from data/raw/.

    Args:
        raw_dir: Path to the directory containing the raw CSV files.
                Defaults to "data/raw".

    Returns:
        Tuple of (transaction_df, identity_df) as pandas DataFrames.

    Raises:
        FileNotFoundError: if either train_transaction.csv or train_identity.csv
                          is missing from raw_dir. Provides a helpful message
                          directing the user to download the dataset from Kaggle.
    """
    transaction_path = os.path.join(raw_dir, "train_transaction.csv")
    identity_path = os.path.join(raw_dir, "train_identity.csv")

    # Check for missing files
    if not os.path.exists(transaction_path):
        raise FileNotFoundError(
            f"train_transaction.csv not found at {transaction_path}\n"
            "Please download the IEEE-CIS Kaggle dataset from:\n"
            "  https://www.kaggle.com/c/ieee-fraud-detection/data\n"
            "Extract the files to data/raw/ and try again."
        )

    if not os.path.exists(identity_path):
        raise FileNotFoundError(
            f"train_identity.csv not found at {identity_path}\n"
            "Please download the IEEE-CIS Kaggle dataset from:\n"
            "  https://www.kaggle.com/c/ieee-fraud-detection/data\n"
            "Extract the files to data/raw/ and try again."
        )

    print(f"Loading transaction data from {transaction_path}...")
    transaction_df = pd.read_csv(transaction_path)
    print(f"  ✓ Loaded {len(transaction_df)} transaction records")

    print(f"Loading identity data from {identity_path}...")
    identity_df = pd.read_csv(identity_path)
    print(f"  ✓ Loaded {len(identity_df)} identity records")

    return transaction_df, identity_df


def join_and_sample(
    transaction_df, identity_df, sample_size=50000, seed=42
) -> pd.DataFrame:
    """
    Join transaction and identity data, then sample.

    Performs a left join on TransactionID so that transactions without
    identity data are retained (with NaN values in identity columns).
    Then randomly samples up to sample_size rows.

    Args:
        transaction_df: DataFrame from load_raw() (transactions).
        identity_df: DataFrame from load_raw() (identity data).
        sample_size: Number of rows to sample. If the joined DataFrame has
                    fewer rows, all are returned.
        seed: Random seed for reproducibility.

    Returns:
        Sampled and joined DataFrame with columns from COLUMNS_TO_KEEP.
    """
    print("\nJoining transaction and identity data (left join on TransactionID)...")
    joined_df = transaction_df.merge(
        identity_df, on="TransactionID", how="left"
    )
    print(f"  ✓ Joined: {len(joined_df)} rows")

    # Select only columns we care about
    available_columns = [col for col in COLUMNS_TO_KEEP if col in joined_df.columns]
    joined_df = joined_df[available_columns]

    # Sample
    actual_sample_size = min(sample_size, len(joined_df))
    print(f"\nSampling {actual_sample_size} rows (seed={seed})...")
    sampled_df = joined_df.sample(n=actual_sample_size, random_state=seed)
    print(f"  ✓ Sampled: {len(sampled_df)} rows")

    return sampled_df


def summarize(df) -> None:
    """
    Print a summary of the dataset quality and structure.

    Args:
        df: DataFrame to summarize (typically the output of join_and_sample()).
    """
    print("\n" + "=" * 70)
    print("Dataset Summary")
    print("=" * 70)

    print(f"\nTotal rows: {len(df)}")

    fraud_ratio = df["isFraud"].mean() if "isFraud" in df.columns else 0.0
    print(f"Fraud ratio: {fraud_ratio:.2%}")

    # Percentage of rows with non-null DeviceType (shows identity data sparsity)
    if "DeviceType" in df.columns:
        device_type_pct = df["DeviceType"].notna().mean() * 100
        print(f"Rows with DeviceType (identity data coverage): {device_type_pct:.1f}%")

    print("\nNull values per column:")
    null_counts = df.isnull().sum()
    for col in COLUMNS_TO_KEEP:
        if col in df.columns:
            null_count = null_counts[col]
            null_pct = (null_count / len(df)) * 100
            print(f"  {col}: {null_count} ({null_pct:.1f}%)")

    print("\n" + "=" * 70)


def main():
    """
    Main entry point: load, join, sample, summarize, and save.
    """
    print("=" * 70)
    print("IEEE-CIS Kaggle Dataset Loader")
    print("=" * 70)

    try:
        # Load raw files
        transaction_df, identity_df = load_raw(raw_dir="data/raw")

        # Join and sample
        sampled_df = join_and_sample(
            transaction_df, identity_df, sample_size=50000, seed=42
        )

        # Summarize
        summarize(sampled_df)

        # Save
        print("\nSaving sampled dataset...")
        output_dir = os.path.join("data", "processed")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "ieee_cis_sample.csv")
        sampled_df.to_csv(output_path, index=False)
        print(f"  ✓ Saved to: {output_path}")

        print("\n" + "=" * 70)
        print("Dataset loading complete!")
        print("=" * 70)

    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        raise
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        raise


if __name__ == "__main__":
    main()

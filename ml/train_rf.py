"""
Random Forest baseline training for RiskShield fraud detection.

Phase 3, Task 2: Train a Random Forest classifier as a comparison baseline
to the existing XGBoost model, using the same time-based split and evaluation approach.

This task:
  - Loads the synthetic dataset (same as XGBoost baseline)
  - Uses the same time-based (chronological) 70/15/15 train/val/test split
  - Trains Random Forest with class_weight='balanced' for imbalance handling
  - Generates predictions on validation and test sets
  - Compares rough validation metrics to XGBoost baseline

This is NOT a full evaluation. Task 4 will compute proper PR-AUC, F1, confusion matrix, etc.
Goal here is to have a second trained model for side-by-side comparison on the val set.

References:
  - ml/train_baseline.py — XGBoost baseline (Phase 3, Task 1)
  - XGBoost results: val accuracy ~96%, fraud detection count varies with threshold
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import pickle

# ML libraries
from sklearn.ensemble import RandomForestClassifier
from ml.batch_features import compute_features_batch


def load_and_featurize(csv_path: str = "data/processed/synthetic_transactions.csv") -> pd.DataFrame:
    """
    Load synthetic dataset and compute all 10 fraud-detection features.
    
    Args:
        csv_path: Path to data/processed/synthetic_transactions.csv
    
    Returns:
        DataFrame with original columns plus 10 feature columns and timestamp preserved.
    
    Raises:
        FileNotFoundError: If CSV not found; user should run data/generate_synthetic.py
    """
    try:
        print("Loading synthetic dataset...")
        df = pd.read_csv(csv_path, parse_dates=["created_at"])
        print(f"  Loaded {len(df)} transactions")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Cannot find {csv_path}.\n"
            "Please generate synthetic data first:\n"
            "  python data/generate_synthetic.py\n"
            "Then re-run this script."
        )
    
    print("Computing features using batch pipeline...")
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
    print(f"  Computed features for {len(features_df)} rows")
    
    # Preserve timestamp for time-based splitting
    # (compute_features_batch doesn't pass it through, so we add it back)
    features_df["created_at"] = df["created_at"].values
    
    return features_df


def time_based_split(
    df: pd.DataFrame, 
    timestamp_col: str = "created_at",
    train_pct: float = 0.70,
    val_pct: float = 0.15,
) -> tuple:
    """
    Perform a time-based (chronological) split of the dataset.
    
    Splits the data by timestamp into consecutive time periods:
    - Train: first 70% of the time range
    - Validation: next 15%
    - Test: final 15%
    
    This ensures no test data is "older" than training data (no temporal leakage).
    
    IMPORTANT: The featurized output from compute_features_batch() retains row order
    (sorted by timestamp internally), so we split by index position rather than by
    timestamp value. If timestamp_col is not in df (compute_features_batch may drop it),
    this function relies on the already-sorted row order.
    
    Args:
        df: DataFrame with a timestamp column
        timestamp_col: Name of the timestamp column (used for validation/printing, not split logic)
        train_pct, val_pct: Percentages (test_pct = 1.0 - train_pct - val_pct)
    
    Returns:
        Tuple of (df_train, df_val, df_test)
    
    Raises:
        ValueError: If train_pct + val_pct >= 1.0
    """
    if train_pct + val_pct >= 1.0:
        raise ValueError(
            f"train_pct ({train_pct}) + val_pct ({val_pct}) must sum to < 1.0"
        )
    
    test_pct = 1.0 - train_pct - val_pct
    
    # Split by index position (assumes rows already sorted by timestamp)
    n = len(df)
    train_end = int(n * train_pct)
    val_end = train_end + int(n * val_pct)
    
    df_train = df.iloc[:train_end].copy()
    df_val = df.iloc[train_end:val_end].copy()
    df_test = df.iloc[val_end:].copy()
    
    print(f"\nTime-based split ({train_pct*100:.0f}/{val_pct*100:.0f}/{test_pct*100:.0f}):")
    print(f"  Train: {len(df_train)} rows ({len(df_train)/n*100:.1f}%)")
    print(f"  Val:   {len(df_val)} rows ({len(df_val)/n*100:.1f}%)")
    print(f"  Test:  {len(df_test)} rows ({len(df_test)/n*100:.1f}%)")
    
    # Fraud ratio per split
    if "label" in df_train.columns:
        fraud_train = df_train["label"].mean()
        fraud_val = df_val["label"].mean()
        fraud_test = df_test["label"].mean()
        print(f"  Fraud ratio - train: {fraud_train:.2%}, val: {fraud_val:.2%}, test: {fraud_test:.2%}")
    
    return df_train, df_val, df_test


def train_baseline_rf(
    train_df: pd.DataFrame,
    feature_columns: list,
    label_column: str = "label",
) -> RandomForestClassifier:
    """
    Train a baseline Random Forest classifier with class_weight='balanced'.
    
    Random Forest is chosen as a baseline because it:
    - Handles non-linear relationships and feature interactions naturally
    - Doesn't require normalization/scaling
    - Is interpretable (feature importance via .feature_importances_)
    - Provides a reasonable point of comparison to XGBoost
    
    class_weight='balanced' re-weights the loss so mistakes on the minority (fraud)
    class count more, without altering the data itself (unlike SMOTE, which is tested in a later task).
    
    Args:
        train_df: Training DataFrame with feature and label columns
        feature_columns: List of feature column names to use (exactly these 10)
        label_column: Name of the label/target column (default "label")
    
    Returns:
        Trained RandomForestClassifier model
    
    Raises:
        ValueError: If any feature column doesn't exist in train_df
    """
    # Validate that all feature columns exist
    missing_cols = set(feature_columns) - set(train_df.columns)
    if missing_cols:
        raise ValueError(
            f"Missing feature columns in train_df: {sorted(missing_cols)}. "
            f"Available: {sorted(train_df.columns)}"
        )
    
    print("\nTraining Random Forest baseline with class_weight='balanced'...")
    
    X_train = train_df[feature_columns].values
    y_train = train_df[label_column].values
    
    # Log class distribution
    n_legit = (y_train == 0).sum()
    n_fraud = (y_train == 1).sum()
    fraud_ratio = n_fraud / len(y_train) * 100
    print(f"  Training set: {n_legit} legitimate, {n_fraud} fraud ({fraud_ratio:.2f}% fraud)")
    print(f"  Fraud ratio: 1:{n_legit/n_fraud:.1f}")
    
    # Train Random Forest with balanced class weights
    model = RandomForestClassifier(
        n_estimators=100,           # Standard number of trees for a baseline
        max_depth=None,             # Grow trees to their natural depth (not as limiting as XGBoost depth=6)
        min_samples_split=2,        # Standard sklearn default
        min_samples_leaf=1,         # Standard sklearn default
        class_weight='balanced',    # Penalize mistakes on minority (fraud) class more
        random_state=42,            # Reproducibility
        n_jobs=-1,                  # Use all CPU cores
    )
    
    model.fit(X_train, y_train)
    print("  ✓ Model trained")
    
    return model


def generate_predictions(
    model,
    df: pd.DataFrame,
    feature_columns: list,
    set_name: str = "set",
    label_column: str = "label",
) -> tuple:
    """
    Generate probability predictions on a dataset.
    
    Args:
        model: Trained classifier (RandomForestClassifier or XGBClassifier)
        df: DataFrame with feature columns and (optionally) label column
        feature_columns: List of feature column names
        set_name: Name for logging (e.g., "validation")
        label_column: Name of the label column for ground truth (optional)
    
    Returns:
        Tuple of (y_pred_proba, accuracy, fraud_pred_count, fraud_actual_count)
    """
    X = df[feature_columns].values
    y_pred_proba = model.predict_proba(X)[:, 1]  # Probability of fraud (class 1)
    
    print(f"\n{set_name.capitalize()} predictions:")
    print(f"  Min probability: {y_pred_proba.min():.4f}")
    print(f"  Max probability: {y_pred_proba.max():.4f}")
    print(f"  Mean probability: {y_pred_proba.mean():.4f}")
    print(f"  Median probability: {np.median(y_pred_proba):.4f}")
    
    # Rough sanity check: at threshold 0.5
    y_pred_binary = (y_pred_proba >= 0.5).astype(int)
    fraud_pred_count = y_pred_binary.sum()
    
    accuracy = None
    fraud_actual_count = None
    if label_column in df.columns:
        y_actual = df[label_column].values
        accuracy = (y_pred_binary == y_actual).mean()
        fraud_actual_count = (y_actual == 1).sum()
        print(f"  Accuracy (threshold 0.5): {accuracy:.4f}")
        print(f"  Predicted fraud: {fraud_pred_count}, Actual fraud: {fraud_actual_count}")
    
    return y_pred_proba, accuracy, fraud_pred_count, fraud_actual_count


def main():
    """Main entry point for Random Forest baseline training."""
    print("=" * 70)
    print("Phase 3, Task 2: Random Forest Baseline Training on Synthetic Dataset")
    print("=" * 70)
    
    # Load and engineer features
    features_df = load_and_featurize()
    
    # Define feature columns (the 10 computed features from compute_features())
    feature_cols = [
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
    ]
    
    # Time-based split
    df_train, df_val, df_test = time_based_split(features_df)
    
    # Train model
    model = train_baseline_rf(df_train, feature_cols)
    
    # Generate predictions on validation and test sets
    print("\n" + "=" * 70)
    print("ROUGH Sanity Checks (NOT full evaluation)")
    print("=" * 70)
    print("Task 4 will compute proper PR-AUC, F1, confusion matrix, etc.")
    
    y_val_pred, val_acc, val_fraud_pred, val_fraud_actual = generate_predictions(
        model, df_val, feature_cols, "validation"
    )
    y_test_pred, test_acc, test_fraud_pred, test_fraud_actual = generate_predictions(
        model, df_test, feature_cols, "test"
    )
    
    # Sanity checks
    print("\n" + "=" * 70)
    print("Sanity Checks")
    print("=" * 70)
    
    # Check predictions are in valid range
    assert (y_val_pred >= 0).all() and (y_val_pred <= 1).all(), "Validation predictions out of range"
    assert (y_test_pred >= 0).all() and (y_test_pred <= 1).all(), "Test predictions out of range"
    print("✓ Predictions in valid range [0, 1]")
    
    # Check no NaN predictions
    assert not np.isnan(y_val_pred).any(), "NaN in validation predictions"
    assert not np.isnan(y_test_pred).any(), "NaN in test predictions"
    print("✓ No NaN values in predictions")
    
    # Feature importance (unique to tree-based models)
    print("\n" + "=" * 70)
    print("Feature Importance (Random Forest)")
    print("=" * 70)
    importances = model.feature_importances_
    sorted_indices = np.argsort(importances)[::-1]
    for idx in sorted_indices[:5]:
        print(f"  {feature_cols[idx]:25s}: {importances[idx]:.4f}")
    
    print("\n" + "=" * 70)
    print("✓ Random Forest baseline complete")
    print("=" * 70)
    print("\nComparison Notes (for Task 4):")
    print("  - Validation accuracy: {:.4f}".format(val_acc or 0))
    print("  - Validation fraud detected: {}/{}".format(val_fraud_pred or 0, val_fraud_actual or 0))
    print("  - Test accuracy: {:.4f}".format(test_acc or 0))
    print("  - Test fraud detected: {}/{}".format(test_fraud_pred or 0, test_fraud_actual or 0))
    print("\nCompare these rough metrics to the XGBoost baseline (train_baseline.py)")
    print("Full evaluation (PR-AUC, F1, confusion matrix) comes in Task 4")


if __name__ == "__main__":
    main()

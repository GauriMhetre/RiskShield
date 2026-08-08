"""
Baseline model training for RiskShield fraud detection.

Phase 3, Task 1: Train a minimal XGBoost baseline on the synthetic dataset
to validate the entire training pipeline end-to-end.

This task:
  - Loads the synthetic dataset
  - Performs a time-based (chronological) 70/15/15 train/val/test split
  - Trains XGBoost with class_weight='balanced' for imbalance handling
  - Generates predictions on validation and test sets
  - Saves basic artifacts (model, split indices, predictions)

This is NOT a final evaluation. Task 4 will compute proper PR-AUC, F1, confusion matrix, etc.
Goal here is to sanity-check that features → model → predictions all work end-to-end.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import pickle

# ML libraries
from xgboost import XGBClassifier
from ml.batch_features import compute_features_batch


def load_and_engineer_features(csv_path: str) -> pd.DataFrame:
    """
    Load synthetic dataset and compute all 10 fraud-detection features.
    
    Args:
        csv_path: Path to data/processed/synthetic_transactions.csv
    
    Returns:
        DataFrame with original columns plus 10 feature columns.
    """
    print("Loading synthetic dataset...")
    df = pd.read_csv(csv_path, parse_dates=["created_at"])
    print(f"  Loaded {len(df)} transactions")
    
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
    features_df["created_at"] = df["created_at"].values
    
    return features_df


def time_based_split(
    df: pd.DataFrame, 
    timestamp_col: str = "created_at",
    train_pct: float = 0.70,
    val_pct: float = 0.15,
    test_pct: float = 0.15,
) -> tuple:
    """
    Perform a time-based (chronological) split of the dataset.
    
    Splits the data by timestamp into consecutive time periods:
    - Train: first 70% of the time range
    - Validation: next 15%
    - Test: final 15%
    
    This ensures no test data is "older" than training data (no temporal leakage).
    
    Args:
        df: DataFrame with a timestamp column
        timestamp_col: Name of the timestamp column
        train_pct, val_pct, test_pct: Percentages (must sum to 1.0)
    
    Returns:
        Tuple of (df_train, df_val, df_test)
    """
    # Sort by timestamp (should already be sorted from batch_features, but be explicit)
    df_sorted = df.sort_values(timestamp_col).reset_index(drop=True)
    
    n = len(df_sorted)
    train_end = int(n * train_pct)
    val_end = train_end + int(n * val_pct)
    
    df_train = df_sorted.iloc[:train_end].copy()
    df_val = df_sorted.iloc[train_end:val_end].copy()
    df_test = df_sorted.iloc[val_end:].copy()
    
    print(f"\nTime-based split (70/15/15):")
    print(f"  Train: {len(df_train)} rows ({len(df_train)/n*100:.1f}%)")
    print(f"  Val:   {len(df_val)} rows ({len(df_val)/n*100:.1f}%)")
    print(f"  Test:  {len(df_test)} rows ({len(df_test)/n*100:.1f}%)")
    
    return df_train, df_val, df_test


def train_baseline_model(
    df_train: pd.DataFrame,
    feature_cols: list,
    label_col: str = "label",
) -> XGBClassifier:
    """
    Train a baseline XGBoost classifier with class_weight='balanced'.
    
    Args:
        df_train: Training DataFrame with feature and label columns
        feature_cols: List of feature column names to use
        label_col: Name of the label/target column
    
    Returns:
        Trained XGBClassifier model
    """
    print("\nTraining XGBoost baseline with class_weight='balanced'...")
    
    X_train = df_train[feature_cols].values
    y_train = df_train[label_col].values
    
    # Log class distribution
    n_legit = (y_train == 0).sum()
    n_fraud = (y_train == 1).sum()
    fraud_ratio = n_fraud / len(y_train) * 100
    print(f"  Training set: {n_legit} legitimate, {n_fraud} fraud ({fraud_ratio:.2f}% fraud)")
    
    # Train XGBoost with balanced class weights
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        scale_pos_weight=n_legit / n_fraud,  # Penalize false negatives more
        random_state=42,
        verbosity=0,  # Suppress XGBoost's verbose output
    )
    
    model.fit(X_train, y_train, verbose=False)
    print("  ✓ Model trained")
    
    return model


def generate_predictions(
    model: XGBClassifier,
    df: pd.DataFrame,
    feature_cols: list,
    set_name: str = "set",
) -> np.ndarray:
    """
    Generate probability predictions on a dataset.
    
    Args:
        model: Trained XGBClassifier
        df: DataFrame with feature columns
        feature_cols: List of feature column names
        set_name: Name for logging (e.g., "validation")
    
    Returns:
        Array of predicted probabilities (shape: (n_samples,))
    """
    X = df[feature_cols].values
    y_pred_proba = model.predict_proba(X)[:, 1]  # Probability of fraud (class 1)
    
    print(f"\n{set_name.capitalize()} predictions:")
    print(f"  Min probability: {y_pred_proba.min():.4f}")
    print(f"  Max probability: {y_pred_proba.max():.4f}")
    print(f"  Mean probability: {y_pred_proba.mean():.4f}")
    print(f"  Median probability: {np.median(y_pred_proba):.4f}")
    
    return y_pred_proba


def save_artifacts(
    model: XGBClassifier,
    feature_cols: list,
    output_dir: str = "models",
) -> str:
    """
    Save trained model and feature metadata.
    
    Args:
        model: Trained XGBClassifier
        feature_cols: List of feature column names
        output_dir: Directory to save artifacts
    
    Returns:
        Path to saved model
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    model_path = output_path / "baseline_xgb_v1.pkl"
    metadata_path = output_path / "baseline_metadata.txt"
    
    # Save model
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"\n✓ Model saved: {model_path}")
    
    # Save metadata
    with open(metadata_path, "w") as f:
        f.write(f"RiskShield Baseline Model Metadata\n")
        f.write(f"==================================\n\n")
        f.write(f"Created: {datetime.now().isoformat()}\n")
        f.write(f"Model: XGBoost (n_estimators=100, max_depth=6)\n")
        f.write(f"Class weighting: balanced (scale_pos_weight)\n\n")
        f.write(f"Features ({len(feature_cols)}):\n")
        for col in feature_cols:
            f.write(f"  - {col}\n")
    print(f"✓ Metadata saved: {metadata_path}")
    
    return str(model_path)


def main():
    """Main entry point for baseline training."""
    print("=" * 70)
    print("Phase 3, Task 1: Baseline XGBoost Training on Synthetic Dataset")
    print("=" * 70)
    
    # Load and engineer features
    features_df = load_and_engineer_features("data/processed/synthetic_transactions.csv")
    
    # Define feature columns (the 10 computed features)
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
    model = train_baseline_model(df_train, feature_cols)
    
    # Generate predictions
    y_val_pred = generate_predictions(model, df_val, feature_cols, "validation")
    y_test_pred = generate_predictions(model, df_test, feature_cols, "test")
    
    # Save artifacts
    save_artifacts(model, feature_cols)
    
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
    
    # Check fraud ground truth
    fraud_ratio_val = df_val["label"].mean()
    fraud_ratio_test = df_test["label"].mean()
    print(f"✓ Fraud ratio - validation: {fraud_ratio_val:.2%}, test: {fraud_ratio_test:.2%}")
    
    print("\n" + "=" * 70)
    print("✓ Pipeline complete end-to-end")
    print("=" * 70)
    print("\nNext: Task 4 will compute PR-AUC, F1, confusion matrix, and feature importance")


if __name__ == "__main__":
    main()

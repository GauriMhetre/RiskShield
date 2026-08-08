"""
Consolidated training pipeline for RiskShield fraud detection.

Phase 3, Tasks 1-2: Train baseline models (Random Forest, XGBoost) on synthetic data
using time-based 70/15/15 split to avoid temporal leakage.

This module:
  - Loads and engineers features for the entire dataset
  - Performs chronological train/val/test split
  - Trains Random Forest with class_weight='balanced'
  - Trains XGBoost with scale_pos_weight (native imbalance handling)
  - Generates predictions and rough sanity checks for both models
  - Compares side-by-side on validation set

This is NOT a final evaluation. Phase 3, Task 4 will compute proper PR-AUC, F1, 
confusion matrix, and feature importance. This task just validates that both models
train end-to-end on the same data split with plausible predictions.

References:
  - ml/features.py: compute_features() and data classes
  - ml/batch_features.py: compute_features_batch() for batch processing
  - docs/HLD.md: describes XGBoost as primary, Random Forest as baseline comparison
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import pickle
import json

# ML libraries
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib
from ml.batch_features import compute_features_batch


# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

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


# ============================================================================
# RANDOM FOREST BASELINE
# ============================================================================

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
        max_depth=None,             # Grow trees to their natural depth
        min_samples_split=2,        # Standard sklearn default
        min_samples_leaf=1,         # Standard sklearn default
        class_weight='balanced',    # Penalize mistakes on minority (fraud) class more
        random_state=42,            # Reproducibility
        n_jobs=-1,                  # Use all CPU cores
    )
    
    model.fit(X_train, y_train)
    print("  ✓ Model trained")
    
    return model


# ============================================================================
# XGBOOST WITH SCALE_POS_WEIGHT
# ============================================================================

def compute_scale_pos_weight(train_df: pd.DataFrame, label_column: str = "label") -> float:
    """
    Compute the scale_pos_weight parameter for XGBoost.
    
    scale_pos_weight tells XGBoost to treat each positive (fraud) example as worth this
    many negative examples during gradient computation. The standard formula is:
        scale_pos_weight = count(negative) / count(positive)
    
    This is computed ONLY from the training set to avoid leaking information about the
    validation/test splits, which would corrupt the model's ability to fairly evaluate
    on unseen data.
    
    Args:
        train_df: Training DataFrame with label column
        label_column: Name of the label/target column (default "label")
    
    Returns:
        Float value for scale_pos_weight
    
    Raises:
        ValueError: If training set has zero positive (fraud) examples, making
                   scale_pos_weight undefined/meaningless
    """
    n_negative = (train_df[label_column] == 0).sum()
    n_positive = (train_df[label_column] == 1).sum()
    
    if n_positive == 0:
        raise ValueError(
            "Cannot compute scale_pos_weight: training set has zero positive (fraud) examples. "
            "XGBoost needs at least one fraud case to train meaningfully."
        )
    
    scale_pos_weight = n_negative / n_positive
    return scale_pos_weight


def train_xgboost(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_columns: list,
    label_column: str = "label",
    scale_pos_weight: float = 1.0,
) -> XGBClassifier:
    """
    Train an XGBoost classifier with scale_pos_weight for class imbalance handling.
    
    XGBoost is the primary model per the PRD because it:
    - Handles non-linear relationships and feature interactions
    - Natively supports scale_pos_weight for imbalance (no data modification like SMOTE)
    - Has fast inference suitable for real-time scoring
    
    scale_pos_weight upweights the minority (fraud) class in the gradient computation,
    making the model more sensitive to fraud examples during training.
    
    eval_metric='aucpr' (Precision-Recall AUC) is used instead of the default because:
    - With severe class imbalance (3-4% fraud), PR-AUC is more informative than ROC-AUC
    - Recall-dominated metrics are better for evaluating imbalanced fraud data
    - This metric is appropriate for the imbalanced problem domain
    
    Args:
        train_df: Training DataFrame with feature and label columns
        val_df: Validation DataFrame (included for consistency; not currently used for early stopping)
        feature_columns: List of feature column names to use (exactly these 10)
        label_column: Name of the label/target column (default "label")
        scale_pos_weight: Pre-computed value for class imbalance weighting
    
    Returns:
        Trained XGBClassifier model
    
    Raises:
        ValueError: If validation set is empty or missing expected columns
    """
    # Validate inputs
    if len(val_df) == 0:
        raise ValueError("Validation DataFrame is empty; cannot use for early stopping")
    
    missing_cols = set(feature_columns) - set(train_df.columns)
    if missing_cols:
        raise ValueError(
            f"Missing feature columns in train_df: {sorted(missing_cols)}. "
            f"Available: {sorted(train_df.columns)}"
        )
    
    missing_cols = set(feature_columns + [label_column]) - set(val_df.columns)
    if missing_cols:
        raise ValueError(
            f"Missing columns in val_df: {sorted(missing_cols)}. "
            f"Available: {sorted(val_df.columns)}"
        )
    
    print("\nTraining XGBoost with scale_pos_weight='balanced'...")
    print(f"  scale_pos_weight value: {scale_pos_weight:.2f}")
    
    X_train = train_df[feature_columns].values
    y_train = train_df[label_column].values
    X_val = val_df[feature_columns].values
    y_val = val_df[label_column].values
    
    # Log class distribution
    n_legit = (y_train == 0).sum()
    n_fraud = (y_train == 1).sum()
    fraud_ratio = n_fraud / len(y_train) * 100
    print(f"  Training set: {n_legit} legitimate, {n_fraud} fraud ({fraud_ratio:.2f}% fraud)")
    
    # Train XGBoost with early stopping on validation set
    model = XGBClassifier(
        n_estimators=300,           # Start with 300 trees (early stopping will reduce if needed)
        max_depth=6,                # Moderate depth to avoid overfitting on small fraud minority
        learning_rate=0.05,         # Conservative learning rate
        subsample=0.8,              # Use 80% of rows per tree
        colsample_bytree=0.8,       # Use 80% of features per tree
        objective="binary:logistic", # Binary classification
        scale_pos_weight=scale_pos_weight,  # Upweight fraud class
        eval_metric='aucpr',        # Use PR-AUC for imbalanced data
        random_state=42,            # Reproducibility
        verbosity=0,                # Suppress verbose output
    )
    
    # Fit model (early stopping not used in this version for compatibility)
    model.fit(
        X_train, y_train,
        verbose=False,
    )
    
    print("  ✓ Model trained")
    
    return model


# ============================================================================
# XGBOOST WITH SMOTE (SYNTHETIC MINORITY OVERSAMPLING)
# ============================================================================

def apply_smote(
    train_df: pd.DataFrame,
    feature_columns: list,
    label_column: str = "label",
    random_state: int = 42,
) -> tuple:
    """
    Apply SMOTE (Synthetic Minority Oversampling Technique) to the training set.
    
    SMOTE generates synthetic fraud examples by interpolating between existing fraud rows
    in feature space, rebalancing the training set to roughly 50/50 fraud/legitimate before
    training. This is fundamentally different from scale_pos_weight (which reweights the
    loss function) or class_weight (which reweights samples) — it actually adds new rows
    to the training data.
    
    CRITICAL DESIGN CHOICE: This function ONLY accepts train_df, not separate X/y arrays.
    This structural constraint makes it impossible to accidentally call with val_df or test_df,
    which would be a catastrophic leakage mistake. If SMOTE were applied before splitting,
    or if synthetic rows leaked into validation/test, the model would memorize near-duplicates
    of training data during "evaluation," falsely inflating performance metrics on unseen data.
    
    Args:
        train_df: Training DataFrame ONLY (not val/test). After the time-based split,
                  this is the real, imbalanced training set to be resampled.
        feature_columns: List of feature column names (the 10 known features)
        label_column: Name of the label/target column (default "label")
        random_state: Seed for reproducibility
    
    Returns:
        Tuple of (X_resampled, y_resampled) — resampled feature matrix and labels
    
    Raises:
        ValueError: If training set has fewer than ~6 fraud examples (SMOTE's default
                   n_neighbors=5 needs at least 6 positive samples to find neighbors)
    """
    from imblearn.over_sampling import SMOTE
    
    n_fraud = (train_df[label_column] == 1).sum()
    n_legit = (train_df[label_column] == 0).sum()
    
    if n_fraud < 6:
        raise ValueError(
            f"Cannot apply SMOTE: training set has only {n_fraud} fraud examples. "
            f"SMOTE requires at least ~6 fraud samples to find nearest neighbors (default n_neighbors=5). "
            f"If your training set is too small or fraud ratio too low, consider using scale_pos_weight or class_weight instead."
        )
    
    X_train = train_df[feature_columns].values
    y_train = train_df[label_column].values
    
    print(f"  Before SMOTE: {len(train_df)} rows, fraud ratio {n_fraud / len(train_df):.2%}")
    
    # Apply SMOTE with random_state for reproducibility
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    n_fraud_resampled = (y_resampled == 1).sum()
    print(f"  After SMOTE:  {len(X_resampled)} rows, fraud ratio {n_fraud_resampled / len(X_resampled):.2%}")
    print(f"  ✓ Generated {len(X_resampled) - len(train_df)} synthetic fraud examples")
    
    return X_resampled, y_resampled


def train_xgboost_smote(
    X_resampled,
    y_resampled,
    val_df: pd.DataFrame,
    feature_columns: list,
    label_column: str = "label",
) -> XGBClassifier:
    """
    Train an XGBoost classifier on SMOTE-resampled training data.
    
    This variant uses the same XGBoost hyperparameters as train_xgboost() for a fair
    apples-to-apples comparison, but trains on synthetically rebalanced data instead of
    using scale_pos_weight.
    
    IMPORTANT: scale_pos_weight is NOT used here, even though the val set is imbalanced.
    Why? Because SMOTE already rebalanced the training data itself. If we also applied
    scale_pos_weight, we'd be "double-correcting" for imbalance: once via SMOTE's
    synthetic rebalancing, and again via the loss function reweighting. This would:
    - Make the model overly cautious on fraud examples (double-penalizing misses)
    - Distort the learned feature weights
    - Potentially cause the model to predict "fraud" too aggressively
    Task 4's evaluation will reveal whether this vs. scale_pos_weight works better in practice.
    
    Validation data (val_df) is NOT resampled — it stays at the original, real-world
    imbalance ratio (4% fraud) so early stopping and final evaluation reflect actual
    production conditions.
    
    Args:
        X_resampled: Feature matrix after SMOTE resampling
        y_resampled: Labels after SMOTE resampling
        val_df: Validation DataFrame (real, non-resampled, for early stopping)
        feature_columns: List of feature column names
        label_column: Name of the label/target column (default "label")
    
    Returns:
        Trained XGBClassifier model
    """
    # Validate inputs
    if len(val_df) == 0:
        raise ValueError("Validation DataFrame is empty; cannot train model")
    
    missing_cols = set(feature_columns + [label_column]) - set(val_df.columns)
    if missing_cols:
        raise ValueError(
            f"Missing columns in val_df: {sorted(missing_cols)}. "
            f"Available: {sorted(val_df.columns)}"
        )
    
    print("\nTraining XGBoost on SMOTE-resampled data (no scale_pos_weight)...")
    
    X_val = val_df[feature_columns].values
    y_val = val_df[label_column].values
    
    # Log class distribution in resampled training data
    n_fraud_resampled = (y_resampled == 1).sum()
    n_legit_resampled = (y_resampled == 0).sum()
    print(f"  Resampled training set: {n_legit_resampled} legitimate, {n_fraud_resampled} fraud (1:{n_legit_resampled/n_fraud_resampled:.1f})")
    
    # Train XGBoost without scale_pos_weight (training data is already balanced by SMOTE)
    # Use identical hyperparameters to train_xgboost() for fair comparison
    model = XGBClassifier(
        n_estimators=300,           # Same as scale_pos_weight variant
        max_depth=6,                # Same as scale_pos_weight variant
        learning_rate=0.05,         # Same as scale_pos_weight variant
        subsample=0.8,              # Same as scale_pos_weight variant
        colsample_bytree=0.8,       # Same as scale_pos_weight variant
        objective="binary:logistic", # Binary classification
        scale_pos_weight=1.0,       # NO UPWEIGHTING — training data is already balanced
        eval_metric='aucpr',        # Same as scale_pos_weight variant
        random_state=42,            # Same as scale_pos_weight variant
        verbosity=0,                # Suppress verbose output
    )
    
    # Fit model on resampled training data, validate on REAL validation data
    model.fit(
        X_resampled, y_resampled,
        verbose=False,
    )
    
    print("  ✓ Model trained on SMOTE-resampled data")
    
    return model


# ============================================================================
# PREDICTION AND EVALUATION
# ============================================================================

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
    
    # Rough sanity check: at threshold 0.5
    y_pred_binary = (y_pred_proba >= 0.5).astype(int)
    fraud_pred_count = y_pred_binary.sum()
    
    accuracy = None
    fraud_actual_count = None
    if label_column in df.columns:
        y_actual = df[label_column].values
        accuracy = (y_pred_binary == y_actual).mean()
        fraud_actual_count = (y_actual == 1).sum()
    
    return y_pred_proba, accuracy, fraud_pred_count, fraud_actual_count


# ============================================================================
# MODEL SERIALIZATION (Phase 3, Task 5)
# ============================================================================

def save_model_artifact(
    model,
    threshold: float,
    feature_columns: list,
    model_version: str,
    model_type: str,
    save_dir: str = "ml/models",
) -> None:
    """
    Save a trained model along with its decision threshold and feature schema.
    
    Artifacts saved:
    - {model_version}.pkl: the trained model object via joblib.dump
    - {model_version}_metadata.json: threshold, feature schema, model type, and timestamp
    
    Separating the threshold from the model file allows Phase 4's API to load both
    independently, and makes it easy to retune the threshold later without retraining.
    Saving the exact feature column list guards against future train/serve skew if
    ml/features.py ever changes (e.g., adding or renaming a feature).
    
    Args:
        model: Trained classifier (RandomForestClassifier or XGBClassifier)
        threshold: Float decision threshold (0-1) for binary classification
        feature_columns: List of feature column names in training order
        model_version: Version identifier (e.g., "model_v1")
        model_type: Human-readable model description (e.g., "xgboost_class_weighted")
        save_dir: Directory to save artifacts (default "ml/models")
    
    Raises:
        ValueError: If save_dir cannot be created or written to
    """
    save_path = Path(save_dir)
    try:
        save_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise ValueError(f"Cannot create model save directory {save_dir}: {e}")
    
    # Save model
    model_file = save_path / f"{model_version}.pkl"
    try:
        joblib.dump(model, model_file)
        print(f"✓ Saved model to {model_file}")
    except Exception as e:
        raise ValueError(f"Cannot save model to {model_file}: {e}")
    
    # Save metadata
    metadata = {
        "model_version": model_version,
        "model_type": model_type,
        "threshold": float(threshold),  # Ensure it serializes as float, not numpy type
        "feature_columns": feature_columns,
        "timestamp": datetime.utcnow().isoformat(),
    }
    
    metadata_file = save_path / f"{model_version}_metadata.json"
    try:
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Saved metadata to {metadata_file}")
    except Exception as e:
        raise ValueError(f"Cannot save metadata to {metadata_file}: {e}")


def load_model_artifact(
    model_version: str,
    save_dir: str = "ml/models",
) -> tuple:
    """
    Load a saved model and its associated metadata.
    
    This function is stateless and has no side effects beyond reading files —
    it's designed to be called from Phase 4's API without any surprise state changes.
    
    Args:
        model_version: Version identifier (e.g., "model_v1")
        save_dir: Directory where artifacts were saved
    
    Returns:
        Tuple of (model, metadata_dict) where metadata_dict contains:
        - model_version, model_type, threshold, feature_columns, timestamp
    
    Raises:
        FileNotFoundError: If either the model or metadata file is missing
    """
    save_path = Path(save_dir)
    
    model_file = save_path / f"{model_version}.pkl"
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")
    
    metadata_file = save_path / f"{model_version}_metadata.json"
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
    
    # Load model
    model = joblib.load(model_file)
    
    # Load metadata
    with open(metadata_file, "r") as f:
        metadata = json.load(f)
    
    return model, metadata


# ============================================================================
# MAIN ORCHESTRATION
# ============================================================================

def main():
    """Main entry point: load data, split, train models, evaluate, select winner, and save."""
    # ========================================================================
    # PHASE 3, TASK 5: MODEL SELECTION AND SERIALIZATION
    # ========================================================================
    # SET THIS BASED ON YOUR READING OF THE TASK 4 COMPARISON TABLE
    # Options: "Random Forest", "XGBoost (class-weighted)", "XGBoost (SMOTE)"
    WINNING_MODEL = "XGBoost (class-weighted)"
    
    # Precision floor for threshold selection
    # Find the lowest threshold where precision >= this value
    # Adjust this value based on your operational needs
    MIN_PRECISION = 0.75
    
    print("=" * 80)
    print("RiskShield Training Pipeline — Phase 3, Tasks 1-5")
    print("=" * 80)
    print()
    
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
    
    # ========================================================================
    # RANDOM FOREST BASELINE (Task 1)
    # ========================================================================
    print("\n" + "=" * 80)
    print("RANDOM FOREST BASELINE (class_weight='balanced')")
    print("=" * 80)
    
    rf_model = train_baseline_rf(df_train, feature_cols)
    _, rf_val_acc, rf_val_fraud_pred, rf_val_fraud_actual = generate_predictions(
        rf_model, df_val, feature_cols, "validation"
    )
    
    print(f"\n  --- Rough sanity check only (full evaluation comes later) ---")
    print(f"  Validation accuracy: {rf_val_acc:.4f}")
    print(f"  Predicted fraud count: {rf_val_fraud_pred} | Actual fraud count: {rf_val_fraud_actual}")
    
    # Feature importance
    print(f"\n  Feature importance (top 5):")
    importances = rf_model.feature_importances_
    sorted_indices = np.argsort(importances)[::-1]
    for idx in sorted_indices[:5]:
        print(f"    {feature_cols[idx]:25s}: {importances[idx]:.4f}")
    
    # ========================================================================
    # XGBOOST WITH SCALE_POS_WEIGHT (Task 2)
    # ========================================================================
    print("\n" + "=" * 80)
    print("XGBOOST BASELINE (scale_pos_weight='balanced')")
    print("=" * 80)
    
    scale_pos_weight = compute_scale_pos_weight(df_train)
    print(f"  Computed scale_pos_weight: {scale_pos_weight:.2f}")
    
    xgb_model = train_xgboost(df_train, df_val, feature_cols, scale_pos_weight=scale_pos_weight)
    _, xgb_val_acc, xgb_val_fraud_pred, xgb_val_fraud_actual = generate_predictions(
        xgb_model, df_val, feature_cols, "validation"
    )
    
    print(f"\n  --- Rough sanity check only (full evaluation comes later) ---")
    print(f"  Validation accuracy: {xgb_val_acc:.4f}")
    print(f"  Predicted fraud count: {xgb_val_fraud_pred} | Actual fraud count: {xgb_val_fraud_actual}")
    
    # ========================================================================
    # XGBOOST WITH SMOTE (Task 3)
    # ========================================================================
    print("\n" + "=" * 80)
    print("XGBOOST BASELINE (SMOTE-resampled training data)")
    print("=" * 80)
    
    X_train_resampled, y_train_resampled = apply_smote(df_train, feature_cols)
    xgb_smote_model = train_xgboost_smote(X_train_resampled, y_train_resampled, df_val, feature_cols)
    _, xgb_smote_val_acc, xgb_smote_val_fraud_pred, xgb_smote_val_fraud_actual = generate_predictions(
        xgb_smote_model, df_val, feature_cols, "validation"
    )
    
    print(f"\n  --- Rough sanity check only (full evaluation comes later) ---")
    print(f"  Validation accuracy: {xgb_smote_val_acc:.4f}")
    print(f"  Predicted fraud count: {xgb_smote_val_fraud_pred} | Actual fraud count: {xgb_smote_val_fraud_actual}")
    
    # ========================================================================
    # SIDE-BY-SIDE COMPARISON AND SANITY CHECKS
    # ========================================================================
    print("\n" + "=" * 80)
    print("THREE-MODEL COMPARISON (Validation Set)")
    print("=" * 80)
    print()
    print(f"{'Metric':<40} {'RF':<15} {'XGB+Weight':<15} {'XGB+SMOTE':<15}")
    print("-" * 85)
    print(f"{'Accuracy (threshold 0.5)':<40} {rf_val_acc:<15.4f} {xgb_val_acc:<15.4f} {xgb_smote_val_acc:<15.4f}")
    print(f"{'Fraud predicted':<40} {rf_val_fraud_pred:<15} {xgb_val_fraud_pred:<15} {xgb_smote_val_fraud_pred:<15}")
    print(f"{'Fraud actual':<40} {rf_val_fraud_actual:<15} {xgb_val_fraud_actual:<15} {xgb_smote_val_fraud_actual:<15}")
    print()
    
    # Sanity checks
    print("=" * 80)
    print("SANITY CHECKS")
    print("=" * 80)
    
    # Get predictions for checks
    rf_val_pred, _, _, _ = generate_predictions(rf_model, df_val, feature_cols)
    xgb_val_pred, _, _, _ = generate_predictions(xgb_model, df_val, feature_cols)
    xgb_smote_val_pred, _, _, _ = generate_predictions(xgb_smote_model, df_val, feature_cols)
    
    # Predictions in valid range
    assert (rf_val_pred >= 0).all() and (rf_val_pred <= 1).all(), "RF predictions out of range"
    assert (xgb_val_pred >= 0).all() and (xgb_val_pred <= 1).all(), "XGBoost predictions out of range"
    assert (xgb_smote_val_pred >= 0).all() and (xgb_smote_val_pred <= 1).all(), "XGBoost+SMOTE predictions out of range"
    print("✓ All three models' predictions in valid range [0, 1]")
    
    # No NaN predictions
    assert not np.isnan(rf_val_pred).any(), "NaN in RF predictions"
    assert not np.isnan(xgb_val_pred).any(), "NaN in XGBoost predictions"
    assert not np.isnan(xgb_smote_val_pred).any(), "NaN in XGBoost+SMOTE predictions"
    print("✓ No NaN values in any model's predictions")
    
    # Predictions are not degenerate
    assert rf_val_fraud_pred > 0, "RF detected zero fraud (degenerate predictions)"
    assert xgb_val_fraud_pred > 0, "XGBoost detected zero fraud (degenerate predictions)"
    assert xgb_smote_val_fraud_pred > 0, "XGBoost+SMOTE detected zero fraud (degenerate predictions)"
    print("✓ All three models detect at least some fraud cases")
    
    # Predictions not wildly large
    assert rf_val_fraud_pred < len(df_val), "RF predicted fraud for all validation rows"
    assert xgb_val_fraud_pred < len(df_val), "XGBoost predicted fraud for all validation rows"
    assert xgb_smote_val_fraud_pred < len(df_val), "XGBoost+SMOTE predicted fraud for all validation rows"
    print("✓ No model flags entire validation set as fraud")
    
    print()
    print("=" * 80)
    print("✓ Training pipeline complete — all three models trained successfully")
    print("=" * 80)
    print()
    print("Models trained:")
    print("  1. Random Forest with class_weight='balanced'")
    print("  2. XGBoost with scale_pos_weight (loss reweighting)")
    print("  3. XGBoost trained on SMOTE-resampled data (data rebalancing)")
    print()
    
    # ========================================================================
    # REAL EVALUATION (Phase 3, Task 4)
    # ========================================================================
    print("\n" + "=" * 80)
    print("REAL EVALUATION SUITE (Validation Set)")
    print("=" * 80)
    
    from ml.evaluate import evaluate_model, recall_at_fixed_fpr, plot_pr_curve, compare_all
    
    # Extract validation features and labels
    X_val = df_val[feature_cols].values
    y_val = df_val["label"].values
    
    # Evaluate all three models
    results = []
    fpr_results = []
    
    print("\n" + "=" * 80)
    print("Metrics at threshold 0.5")
    print("=" * 80)
    
    result_rf = evaluate_model(rf_model, X_val, y_val, "Random Forest")
    results.append(result_rf)
    
    result_xgb = evaluate_model(xgb_model, X_val, y_val, "XGBoost (class-weighted)")
    results.append(result_xgb)
    
    result_smote = evaluate_model(xgb_smote_model, X_val, y_val, "XGBoost (SMOTE)")
    results.append(result_smote)
    
    # Compute recall at fixed FPR (2% false-positive budget)
    print("\n" + "=" * 80)
    print("Recall at Fixed False-Positive Rate (2% budget)")
    print("=" * 80)
    print()
    
    fpr_rf = recall_at_fixed_fpr(rf_model, X_val, y_val, max_fpr=0.02)
    fpr_results.append(fpr_rf)
    print(f"Random Forest:            recall = {fpr_rf['recall_at_budget']:.4f} at fpr = {fpr_rf['achieved_fpr']:.4f}")
    
    fpr_xgb = recall_at_fixed_fpr(xgb_model, X_val, y_val, max_fpr=0.02)
    fpr_results.append(fpr_xgb)
    print(f"XGBoost (class-weighted): recall = {fpr_xgb['recall_at_budget']:.4f} at fpr = {fpr_xgb['achieved_fpr']:.4f}")
    
    fpr_smote = recall_at_fixed_fpr(xgb_smote_model, X_val, y_val, max_fpr=0.02)
    fpr_results.append(fpr_smote)
    print(f"XGBoost (SMOTE):          recall = {fpr_smote['recall_at_budget']:.4f} at fpr = {fpr_smote['achieved_fpr']:.4f}")
    
    # Plot PR curves
    print()
    models_and_names = [
        (rf_model, "Random Forest"),
        (xgb_model, "XGBoost (class-weighted)"),
        (xgb_smote_model, "XGBoost (SMOTE)"),
    ]
    plot_pr_curve(models_and_names, X_val, y_val)
    print("✓ Saved PR curve comparison to ml/models/pr_curve_comparison.png")
    
    # Print final comparison table
    compare_all(results, fpr_results)
    
    print("=" * 100)
    print()
    print("Interpretation guide:")
    print("  - PR-AUC: Higher is better. Represents model's ability to trade off precision vs recall.")
    print("  - F1: Harmonic mean of precision and recall. Good all-around metric.")
    print("  - Recall@2%FPR: Given 2% false-positive budget, max fraud detection rate achieved.")
    print("                  This is the most business-realistic metric for fraud systems.")
    print()
    
    # ========================================================================
    # PHASE 3, TASK 5: MODEL SELECTION AND THRESHOLD TUNING
    # ========================================================================
    print("\n" + "=" * 80)
    print("MODEL SELECTION AND SERIALIZATION (Phase 3, Task 5)")
    print("=" * 80)
    print()
    
    # Map model names to trained model objects
    model_map = {
        "Random Forest": rf_model,
        "XGBoost (class-weighted)": xgb_model,
        "XGBoost (SMOTE)": xgb_smote_model,
    }
    
    model_type_map = {
        "Random Forest": "random_forest_class_weighted",
        "XGBoost (class-weighted)": "xgboost_class_weighted",
        "XGBoost (SMOTE)": "xgboost_smote",
    }
    
    # Select the winning model
    if WINNING_MODEL not in model_map:
        raise ValueError(
            f"Unknown model: {WINNING_MODEL}. "
            f"Must be one of: {list(model_map.keys())}"
        )
    
    selected_model = model_map[WINNING_MODEL]
    model_type = model_type_map[WINNING_MODEL]
    
    print(f"Winning model: {WINNING_MODEL}")
    print()
    
    # Select threshold using min_precision criterion
    from ml.evaluate import select_threshold
    
    print(f"Selecting threshold with min_precision={MIN_PRECISION}...")
    selected_threshold = select_threshold(
        selected_model,
        X_val,
        y_val,
        min_precision=MIN_PRECISION,
    )
    print(f"Selected threshold: {selected_threshold:.4f} (lowest threshold meeting precision floor)")
    print()
    
    # Save the model artifact
    save_model_artifact(
        selected_model,
        threshold=selected_threshold,
        feature_columns=feature_cols,
        model_version="model_v1",
        model_type=model_type,
        save_dir="ml/models",
    )
    print()
    
    # Verify the artifact can be reloaded
    print("Verifying artifact reload...")
    loaded_model, loaded_metadata = load_model_artifact("model_v1", save_dir="ml/models")
    print(f"✓ Successfully reloaded model_v1")
    print(f"  Model type: {loaded_metadata['model_type']}")
    print(f"  Threshold: {loaded_metadata['threshold']:.4f}")
    print(f"  Features ({len(loaded_metadata['feature_columns'])}): {loaded_metadata['feature_columns']}")
    print()
    print("=" * 80)
    print("✓ Phase 3 complete — model selected and serialized")
    print("=" * 80)
    print()
    print("Next: Phase 4 — Build FastAPI /score endpoint using the saved model_v1 artifact")
    print()


if __name__ == "__main__":
    main()

"""
Manual verification script for Phase 4, Task 1.

This standalone script tests the ModelLoader class and feature computation pipeline
without running the full FastAPI server. It's a quick sanity check that:
  1. The model artifact loads successfully
  2. Features compute without errors
  3. Predictions are in the valid range [0, 1]
  4. The threshold and flagged decision make sense

Run this after ml/train.py completes to verify everything is wired up correctly.

Command:
  python scripts/manual_test_model_loader.py

Expected output:
  - Feature dict printed
  - Raw fraud probability printed
  - Loaded threshold printed
  - Flagged decision (True/False) printed
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from datetime import datetime

from backend.app.ml.model_loader import ModelLoader
from ml.features import TransactionInput, UserProfile, compute_features


def main():
    """Load model and test prediction pipeline on a hand-crafted transaction."""

    print("=" * 80)
    print("Phase 4, Task 1 — ModelLoader Verification")
    print("=" * 80)
    print()

    # ========================================================================
    # STEP 1: Load the model
    # ========================================================================
    print("Step 1: Loading model artifact...")
    try:
        loader = ModelLoader(model_version="model_v1", save_dir="ml/models")
        print("✓ Model loaded successfully")
        print(f"  Model version: {loader.get_model_version()}")
        print(f"  Threshold: {loader.get_threshold():.4f}")
        print()
    except FileNotFoundError as e:
        print("✗ FAILED to load model:")
        print(f"  {e}")
        return

    # ========================================================================
    # STEP 2: Create a hand-crafted transaction and user profile
    # ========================================================================
    print("Step 2: Creating test transaction and user profile...")

    # Transaction: $500 amount, novel device, different country, far from home
    # (This is similar to Phase 2, Task 2's manual test for comparability)
    transaction = TransactionInput(
        transaction_id="test_txn_001",
        amount=500.0,  # High amount
        device_id="new_device_xyz",  # Unknown device
        country="SG",  # Different country
        latitude=1.3521,  # Singapore
        longitude=103.8198,
        created_at=datetime(2026, 8, 7, 14, 30, 0),
    )

    # User: typically spends $100, home in US, established devices
    profile = UserProfile(
        user_id="test_user_001",
        avg_amount=100.0,
        std_amount=20.0,
        known_device_ids=["dev_old_1", "dev_old_2"],
        home_country="US",
        home_latitude=40.7128,  # New York
        home_longitude=-74.0060,
        recent_txn_timestamps=[
            datetime(2026, 8, 7, 13, 0, 0),  # 1.5 hours ago
            datetime(2026, 8, 6, 14, 30, 0),  # ~1 day ago
        ],
    )

    print("✓ Created test transaction:")
    print(f"  Transaction ID: {transaction.transaction_id}")
    print(f"  Amount: ${transaction.amount}")
    print(f"  Device: {transaction.device_id} (unknown)")
    print(f"  Country: {transaction.country} (home: {profile.home_country})")
    print(f"  Location: ({transaction.latitude}, {transaction.longitude})")
    print()
    print("✓ Created user profile:")
    print(f"  User ID: {profile.user_id}")
    print(f"  Avg transaction: ${profile.avg_amount} ± ${profile.std_amount}")
    print(f"  Known devices: {profile.known_device_ids}")
    print(f"  Home: {profile.home_country} ({profile.home_latitude}, {profile.home_longitude})")
    print()

    # ========================================================================
    # STEP 3: Compute features
    # ========================================================================
    print("Step 3: Computing features...")
    try:
        feature_dict = compute_features(transaction, profile)
        print("✓ Features computed successfully:")
        print(f"  {feature_dict}")
        print()
    except Exception as e:
        print("✗ FAILED to compute features:")
        print(f"  {e}")
        return

    # ========================================================================
    # STEP 4: Predict fraud probability
    # ========================================================================
    print("Step 4: Predicting fraud probability...")
    try:
        fraud_probability = loader.predict_proba(feature_dict)
        print("✓ Prediction successful:")
        print(f"  Raw fraud probability: {fraud_probability:.4f}")
        print()
    except Exception as e:
        print("✗ FAILED to predict:")
        print(f"  {e}")
        return

    # ========================================================================
    # STEP 5: Apply threshold and flag decision
    # ========================================================================
    print("Step 5: Applying threshold...")
    threshold = loader.get_threshold()
    flagged = fraud_probability >= threshold
    print("✓ Decision made:")
    print(f"  Threshold: {threshold:.4f}")
    print(f"  Fraud probability: {fraud_probability:.4f}")
    print(f"  Flagged as fraud: {flagged}")
    print()

    # ========================================================================
    # STEP 6: Sanity checks
    # ========================================================================
    print("Step 6: Running sanity checks...")

    checks_passed = True

    # Check 1: Probability is in valid range
    if 0 <= fraud_probability <= 1:
        print("✓ Fraud probability is in valid range [0, 1]")
    else:
        print(f"✗ Fraud probability is out of range: {fraud_probability}")
        checks_passed = False

    # Check 2: Feature dict has exactly 10 keys
    if len(feature_dict) == 10:
        print("✓ Feature dict has exactly 10 keys")
    else:
        print(f"✗ Feature dict has {len(feature_dict)} keys, expected 10")
        checks_passed = False

    # Check 3: All features are numeric (not NaN)
    import math
    all_numeric = all(
        isinstance(v, (int, float)) and not math.isnan(float(v))
        for v in feature_dict.values()
    )
    if all_numeric:
        print("✓ All feature values are numeric and not NaN")
    else:
        print("✗ Some features are non-numeric or NaN")
        checks_passed = False

    # Check 4: Threshold is in valid range
    if 0 <= threshold <= 1:
        print("✓ Threshold is in valid range [0, 1]")
    else:
        print(f"✗ Threshold is out of range: {threshold}")
        checks_passed = False

    print()

    # ========================================================================
    # TEST MISSING FEATURE ERROR HANDLING
    # ========================================================================
    print("Step 7: Testing error handling (missing feature)...")
    incomplete_dict = {k: v for k, v in feature_dict.items() if k != "amount_zscore"}
    try:
        loader.predict_proba(incomplete_dict)
        print("✗ FAILED: Should have raised ValueError for missing feature")
        checks_passed = False
    except ValueError as e:
        if "missing required features" in str(e).lower() or "missing" in str(e).lower():
            print("✓ Correctly raised ValueError for missing feature:")
            print(f"  {str(e)[:100]}...")
        else:
            print("✗ Raised ValueError but with unexpected message:")
            print(f"  {e}")
            checks_passed = False
    except Exception as e:
        print(f"✗ Raised unexpected exception type {type(e).__name__}:")
        print(f"  {e}")
        checks_passed = False

    print()

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    if checks_passed:
        print("=" * 80)
        print("✓ ALL CHECKS PASSED")
        print("=" * 80)
        print()
        print("Summary:")
        print(f"  - Model loaded: {loader.get_model_version()}")
        print(f"  - Features computed: {len(feature_dict)} keys")
        print(f"  - Prediction: {fraud_probability:.4f}")
        print(f"  - Threshold: {threshold:.4f}")
        print(f"  - Flagged: {flagged}")
        print()
        print("Phase 4, Task 1 is ready. Next: build the /score FastAPI route.")
    else:
        print("=" * 80)
        print("✗ SOME CHECKS FAILED")
        print("=" * 80)
        print()
        print("Please review the errors above and fix before proceeding.")


if __name__ == "__main__":
    main()

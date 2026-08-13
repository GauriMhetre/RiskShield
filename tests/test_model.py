"""
Phase 7, Task 3: Model Regression Tests
Regression tests for the saved model artifact.
"""
import pytest
import numpy as np
from datetime import datetime

from ml.train import load_model_artifact
from ml.features import TransactionInput, UserProfile, compute_features


@pytest.fixture(scope="module")
def loaded_model():
    """
    Module-scoped fixture to load the model artifact once.
    Returns both the model and its metadata.
    """
    model, metadata = load_model_artifact("model_v1", "ml/models")
    return model, metadata


def test_model_artifact_loads_successfully(loaded_model):
    """
    Basic sanity check — model and metadata both loaded, metadata contains all expected keys.
    """
    model, metadata = loaded_model
    
    assert model is not None
    assert metadata is not None
    
    expected_keys = {"model_version", "model_type", "threshold", "feature_columns", "timestamp"}
    assert expected_keys.issubset(set(metadata.keys())), f"Missing keys in metadata: {expected_keys - set(metadata.keys())}"


def test_feature_schema_contract(loaded_model):
    """
    Call compute_features() with a hand-built TransactionInput/UserProfile example, 
    get its actual returned dict keys, and assert this set EXACTLY matches 
    metadata['feature_columns'] from the loaded model.
    
    This is the automated enforcement of the "feature contract" concept introduced back 
    in Phase 2, Task 2, now checked against the actual saved artifact rather than just 
    asserted in isolation.
    """
    _, metadata = loaded_model
    
    txn = TransactionInput(
        transaction_id="test_contract",
        amount=100.0,
        device_id="dev_1",
        country="US",
        latitude=40.0,
        longitude=-74.0,
        created_at=datetime(2026, 8, 7, 12, 0, 0)
    )
    profile = UserProfile(
        user_id="user_1",
        avg_amount=100.0,
        std_amount=10.0,
        known_device_ids=["dev_1"],
        home_country="US",
        home_latitude=40.0,
        home_longitude=-74.0,
        recent_txn_timestamps=[]
    )
    
    features = compute_features(txn, profile)
    
    # Assert that the exact same set of feature keys are present in both
    assert set(features.keys()) == set(metadata["feature_columns"])


def test_obviously_fraudulent_transaction_scores_high(loaded_model):
    """
    Build a hand-crafted TransactionInput + UserProfile with strong fraud signals, 
    compute its features, get its risk_score, assert risk_score > 0.5.
    """
    model, metadata = loaded_model
    
    txn = TransactionInput(
        transaction_id="fraud_test",
        amount=50000.0, # Large amount relative to average
        device_id="unknown_dev_999", # Unrecognized device
        country="IN", # Mismatched country
        latitude=20.0, # Mismatched geo
        longitude=77.0,
        created_at=datetime(2026, 8, 7, 12, 0, 0)
    )
    profile = UserProfile(
        user_id="user_fraud",
        avg_amount=50.0,
        std_amount=5.0,
        known_device_ids=["dev_known"],
        home_country="US",
        home_latitude=40.0,
        home_longitude=-74.0,
        recent_txn_timestamps=[]
    )
    
    features = compute_features(txn, profile)
    feature_array = np.array([features[col] for col in metadata["feature_columns"]]).reshape(1, -1)
    risk_score = model.predict_proba(feature_array)[0, 1]
    
    assert risk_score > 0.5


def test_obviously_normal_transaction_scores_low(loaded_model):
    """
    Build a hand-crafted TransactionInput + UserProfile representing a routine transaction, 
    assert risk_score < 0.5.
    """
    model, metadata = loaded_model
    
    txn = TransactionInput(
        transaction_id="normal_test",
        amount=52.0, # Close to average
        device_id="dev_known", # Known device
        country="US", # Matching country
        latitude=40.0, # Matching geo
        longitude=-74.0,
        created_at=datetime(2026, 8, 7, 12, 0, 0)
    )
    profile = UserProfile(
        user_id="user_normal",
        avg_amount=50.0,
        std_amount=5.0,
        known_device_ids=["dev_known"],
        home_country="US",
        home_latitude=40.0,
        home_longitude=-74.0,
        recent_txn_timestamps=[]
    )
    
    features = compute_features(txn, profile)
    feature_array = np.array([features[col] for col in metadata["feature_columns"]]).reshape(1, -1)
    risk_score = model.predict_proba(feature_array)[0, 1]
    
    assert risk_score < 0.5


def test_fraud_example_scores_higher_than_normal_example(loaded_model):
    """
    Using the two examples from tests 4 and 5, assert the fraud example's risk_score 
    is strictly greater than the normal example's risk_score.
    """
    model, metadata = loaded_model
    
    # Fraud Example
    fraud_txn = TransactionInput(
        transaction_id="fraud_test",
        amount=50000.0,
        device_id="unknown_dev_999",
        country="IN",
        latitude=20.0,
        longitude=77.0,
        created_at=datetime(2026, 8, 7, 12, 0, 0)
    )
    fraud_profile = UserProfile(
        user_id="user_fraud",
        avg_amount=50.0,
        std_amount=5.0,
        known_device_ids=["dev_known"],
        home_country="US",
        home_latitude=40.0,
        home_longitude=-74.0,
        recent_txn_timestamps=[]
    )
    fraud_features = compute_features(fraud_txn, fraud_profile)
    fraud_array = np.array([fraud_features[col] for col in metadata["feature_columns"]]).reshape(1, -1)
    fraud_score = model.predict_proba(fraud_array)[0, 1]
    
    # Normal Example
    normal_txn = TransactionInput(
        transaction_id="normal_test",
        amount=52.0,
        device_id="dev_known",
        country="US",
        latitude=40.0,
        longitude=-74.0,
        created_at=datetime(2026, 8, 7, 12, 0, 0)
    )
    normal_profile = UserProfile(
        user_id="user_normal",
        avg_amount=50.0,
        std_amount=5.0,
        known_device_ids=["dev_known"],
        home_country="US",
        home_latitude=40.0,
        home_longitude=-74.0,
        recent_txn_timestamps=[]
    )
    normal_features = compute_features(normal_txn, normal_profile)
    normal_array = np.array([normal_features[col] for col in metadata["feature_columns"]]).reshape(1, -1)
    normal_score = model.predict_proba(normal_array)[0, 1]
    
    assert fraud_score > normal_score


def test_threshold_is_valid_probability(loaded_model):
    """
    Assert 0.0 <= metadata['threshold'] <= 1.0 (a basic sanity guard against a corrupted/malformed metadata file).
    """
    _, metadata = loaded_model
    
    assert 0.0 <= metadata["threshold"] <= 1.0

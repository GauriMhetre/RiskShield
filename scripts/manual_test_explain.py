import os
import sys
from datetime import datetime

# Ensure the root directory is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml.explain import build_explainer, explain_prediction
from ml.features import TransactionInput, UserProfile, compute_features
from ml.train import load_model_artifact


def run_test():
    print("Loading model...")
    # Load model_v1 explicitly
    model, metadata = load_model_artifact("model_v1")
    feature_columns = metadata["feature_columns"]
    
    print("Building SHAP TreeExplainer...")
    explainer = build_explainer(model)
    
    # --- Obviously Fraudulent Example ---
    # From Phase 2/4/7: large amount deviation + new device + country mismatch
    fraud_txn = TransactionInput(
        transaction_id="txn_fraud",
        amount=10500.0,
        device_id="unknown_device_999",
        country="RU",
        latitude=55.7558,
        longitude=37.6173,
        created_at=datetime(2026, 8, 14, 12, 0, 0)
    )
    fraud_profile = UserProfile(
        user_id="user_victim",
        avg_amount=500.0,
        std_amount=500.0, # Z-score will be 20.0
        known_device_ids=["iphone_12_main"],
        home_country="US",
        home_latitude=40.7128,
        home_longitude=-74.0060,
        recent_txn_timestamps=[]
    )
    
    fraud_features = compute_features(fraud_txn, fraud_profile)
    fraud_shap = explain_prediction(explainer, fraud_features, feature_columns, top_n=3)
    
    print("\n=== Obviously Fraudulent Example ===")
    print("Top 3 contributing features:")
    for item in fraud_shap:
        print(f"  {item['feature']}: shap_value={item['shap_value']:+.2f}, feature_value={item['feature_value']}")
        
    # --- Obviously Normal Example ---
    # From Phase 2/4/7: routine amount + known device + matching country
    normal_txn = TransactionInput(
        transaction_id="txn_normal",
        amount=400.0,
        device_id="iphone_12_main",
        country="US",
        latitude=40.7300, # Close to home
        longitude=-73.9900,
        created_at=datetime(2026, 8, 14, 13, 0, 0)
    )
    normal_profile = UserProfile(
        user_id="user_normal",
        avg_amount=500.0,
        std_amount=500.0, # Z-score will be -0.2
        known_device_ids=["iphone_12_main"],
        home_country="US",
        home_latitude=40.7128,
        home_longitude=-74.0060,
        recent_txn_timestamps=[]
    )
    
    normal_features = compute_features(normal_txn, normal_profile)
    normal_shap = explain_prediction(explainer, normal_features, feature_columns, top_n=3)
    
    print("\n=== Obviously Normal Example ===")
    print("Top 3 contributing features:")
    for item in normal_shap:
        print(f"  {item['feature']}: shap_value={item['shap_value']:+.2f}, feature_value={item['feature_value']}")

if __name__ == "__main__":
    run_test()

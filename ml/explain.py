"""
SHAP explainability module.
"""
import numpy as np
import shap

# This mapping should stay in sync with any similar readable-label mapping 
# already used in the frontend's placeholder heuristic (Phase 6, Task 5), 
# since both describe the same features to a human reader.
FEATURE_READABLE_NAMES = {
    "txn_count_1h": "1H Transaction Count",
    "txn_count_24h": "24H Transaction Count",
    "amount_zscore": "Amount Z-Score",
    "amount_ratio_to_avg": "Amount Ratio to Avg",
    "device_mismatch": "Device Mismatch",
    "country_mismatch": "Country Mismatch",
    "geo_distance_km": "Geo Distance Km",
    "amount": "Amount",
    "hour_of_day": "Hour of Day",
    "day_of_week": "Day of Week"
}

def build_explainer(model) -> shap.TreeExplainer:
    """
    Constructs and returns a shap.TreeExplainer wrapping the given trained model.
    This should be constructed ONCE per model and reused, mirroring ModelLoader's
    'load once' pattern from Phase 4, to avoid unnecessary re-initialization overhead.
    """
    return shap.TreeExplainer(model)

def explain_prediction(explainer, feature_dict: dict, feature_columns: list, top_n: int = 3) -> list[dict]:
    """
    Explain a prediction using SHAP TreeExplainer.
    
    Converts feature_dict into a correctly-ordered array using feature_columns. This applies
    the SAME ordering discipline as ModelLoader.predict_proba() from Phase 4, Task 1, 
    since we must never trust dictionary order.
    """
    if top_n < 1:
        raise ValueError("top_n must be at least 1")
    
    missing_keys = [col for col in feature_columns if col not in feature_dict]
    if missing_keys:
        raise ValueError(f"feature_dict is missing required columns: {missing_keys}")
    
    # Create the ordered feature array (shape: 1 x n_features)
    feature_array = np.array([[feature_dict[col] for col in feature_columns]])
    
    # We use explainer.shap_values(X) instead of the newer explainer(X) callable interface
    # because we only need the raw contribution values (numpy arrays), not the full 
    # Explanation object overhead, and shap_values() remains stable and faster for this.
    shap_vals = explainer.shap_values(feature_array)
    
    # Handle SHAP output shape differences between model types (binary classification)
    if isinstance(shap_vals, list):
        # RandomForest format: list of arrays, [0] for negative class, [1] for positive class (fraud)
        # We extract the positive class index [1], then the first (and only) sample [0]
        shap_vals = shap_vals[1][0]
    else:
        # XGBoost format for binary classification: single array of shape (n_samples, n_features)
        # representing log-odds contributions to the positive class.
        if len(shap_vals.shape) == 3:
            # Fallback for some versions where binary might return 3D array (1, n_features, 2)
            shap_vals = shap_vals[0, :, 1]
        else:
            shap_vals = shap_vals[0]
            
    results = []
    for i, col in enumerate(feature_columns):
        # Convert numpy types to native Python types for JSON serializability
        # shap_value is cast to plain float
        val = float(shap_vals[i])
        feat_val = feature_dict[col]
        
        readable_name = FEATURE_READABLE_NAMES.get(col, col)
        
        results.append({
            "feature": readable_name,
            "shap_value": val,
            "feature_value": feat_val
        })
        
    # Sort by absolute SHAP value descending
    results.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
    
    return results[:top_n]

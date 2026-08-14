import json
import os
import sys
from datetime import datetime

import numpy as np

# Ensure the root directory is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml.train import load_and_featurize, time_based_split

NUMERIC_FEATURES = [
    "txn_count_1h", "txn_count_24h", "amount_zscore",
    "amount_ratio_to_avg", "geo_distance_km", "amount"
]
BINARY_FEATURES = ["device_mismatch", "country_mismatch"]
INFO_FEATURES = ["hour_of_day", "day_of_week"]

def main():
    print("Loading training data...")
    # Load identical data and split as train.py
    features_df = load_and_featurize()
    train_df, _, _ = time_based_split(features_df)
    
    print(f"Training set: {len(train_df)} rows")
    
    distribution = {
        "metadata": {
            "generated_at": datetime.utcnow().isoformat(),
            "n_training_rows": len(train_df)
        },
        "numeric_features": {},
        "binary_features": {},
        "info_features": {}
    }
    
    # 1. Numeric features (10 quantile-based bins)
    for col in NUMERIC_FEATURES:
        # Compute 0th to 100th percentiles in 10 steps (11 edges)
        percentiles = np.linspace(0, 100, 11)
        raw_edges = np.nanpercentile(train_df[col], percentiles)
        edges = np.unique(raw_edges).tolist()
        
        if len(edges) == 1:
            edges = [edges[0] - 1e-9, edges[0] + 1e-9]
            
        # Ensure the first and last edges encompass any future live data
        edges[0] = -float('inf')
        edges[-1] = float('inf')
        
        # Compute proportions in bins
        counts, _ = np.histogram(train_df[col], bins=edges)
        proportions = (counts / len(train_df)).tolist()
        
        distribution["numeric_features"][col] = {
            "bin_edges": edges,
            "proportions": proportions
        }
        
    print(f"Saved bin edges and proportions for {len(NUMERIC_FEATURES)} numeric features")
    
    # 2. Binary features (proportion of 1s)
    for col in BINARY_FEATURES:
        prop_1s = (train_df[col] == 1).mean()
        distribution["binary_features"][col] = {
            "proportion_1s": float(prop_1s)
        }
        
    print(f"Saved proportions for {len(BINARY_FEATURES)} binary features")
    
    # 3. Info features (value counts)
    for col in INFO_FEATURES:
        counts = train_df[col].value_counts(normalize=True).to_dict()
        counts = {str(k): float(v) for k, v in counts.items()}
        distribution["info_features"][col] = counts
        
    print("Saved distributions for hour_of_day, day_of_week (informational)")
    
    os.makedirs("ml/models", exist_ok=True)
    out_path = "ml/models/training_distribution.json"
    with open(out_path, "w") as f:
        json.dump(distribution, f, indent=2)
        
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()

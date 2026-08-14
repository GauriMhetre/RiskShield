import json
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import select

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend.app.db.models import ScoredTransaction
from backend.app.db.session import SessionLocal


def compute_psi(expected_proportions: list, actual_proportions: list) -> float:
    """
    Compute Population Stability Index (PSI) between two sets of bin proportions.
    
    Formula: sum( (actual% - expected%) * ln(actual% / expected%) )
    We add a small epsilon (1e-6) to any proportion to avoid division-by-zero
    or log(0) errors. This is a common, necessary PSI implementation detail.
    """
    eps = 1e-6
    expected = np.array(expected_proportions) + eps
    actual = np.array(actual_proportions) + eps
    
    # Normalize back to 1.0 after adding epsilon
    expected = expected / expected.sum()
    actual = actual / actual.sum()
    
    psi_values = (actual - expected) * np.log(actual / expected)
    return float(np.sum(psi_values))

def load_recent_scored_data(session, hours_back: int = 24) -> pd.DataFrame:
    """
    Queries ScoredTransaction rows with scored_at within the last hours_back hours,
    and extracts feature_snapshot into a DataFrame.
    """
    cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
    stmt = select(ScoredTransaction).where(ScoredTransaction.scored_at >= cutoff_time)
    
    rows = session.execute(stmt).scalars().all()
    
    if len(rows) < 30:
        print(f"Not enough recent scored transactions (found {len(rows)}, need at least 30) to compute reliable drift.")
        print("PSI on a tiny sample is unreliable and misleading. Please generate more scored transactions.")
        return pd.DataFrame()
        
    features = [row.feature_snapshot for row in rows if row.feature_snapshot]
    return pd.DataFrame(features)

def check_numeric_feature_drift(feature_name, live_values, training_distribution) -> dict:
    """
    Buckets live_values into the SAME bin edges stored in training_distribution,
    computes actual proportions, and computes PSI.
    """
    dist = training_distribution["numeric_features"][feature_name]
    edges = dist["bin_edges"]
    expected_props = dist["proportions"]
    
    # Bucket live data using expected edges
    counts, _ = np.histogram(live_values.dropna(), bins=edges)
    actual_props = (counts / len(live_values)).tolist()
    
    psi = compute_psi(expected_props, actual_props)
    
    severity = "none"
    if psi > 0.25:
        severity = "significant"
    elif psi > 0.1:
        severity = "moderate"
        
    return {
        "feature": feature_name,
        "psi": psi,
        "severity": severity
    }

def check_binary_feature_drift(feature_name, live_values, training_distribution) -> dict:
    """
    Compares the live proportion of 1s against the stored training proportion.
    Thresholds: > 0.25 absolute difference -> significant, > 0.1 -> moderate.
    """
    dist = training_distribution["binary_features"][feature_name]
    expected_prop = dist["proportion_1s"]
    
    actual_prop = (live_values == 1).mean()
    
    diff = abs(actual_prop - expected_prop)
    
    severity = "none"
    if diff > 0.25:
        severity = "significant"
    elif diff > 0.1:
        severity = "moderate"
        
    return {
        "feature": feature_name + " (diff)",
        "psi": diff,
        "severity": severity
    }

def main():
    dist_path = "ml/models/training_distribution.json"
    if not os.path.exists(dist_path):
        print(f"Error: {dist_path} missing.")
        print("Please run scripts/generate_training_distribution.py first.")
        sys.exit(1)
        
    with open(dist_path, "r") as f:
        training_distribution = json.load(f)
        
    db = SessionLocal()
    try:
        # We'll use 720 hours back to ensure we grab all testing data from this portfolio project.
        # In a real environment, 24 hours is typical for daily monitoring.
        hours_back = 720
        df_live = load_recent_scored_data(db, hours_back=hours_back)
        
        if df_live.empty:
            sys.exit(0)
            
        print(f"\nChecking drift over last {hours_back}h ({len(df_live)} scored transactions)\n")
        print(f"{'Feature':<25} {'PSI/Diff':<10} {'Severity':<15}")
        print("-" * 55)
        
        results = []
        
        # Check Numeric
        for feature in training_distribution["numeric_features"]:
            if feature in df_live.columns:
                res = check_numeric_feature_drift(feature, df_live[feature], training_distribution)
                results.append(res)
                
        # Check Binary
        for feature in training_distribution["binary_features"]:
            if feature in df_live.columns:
                res = check_binary_feature_drift(feature, df_live[feature], training_distribution)
                results.append(res)
                
        # Print report
        significant_features = []
        moderate_features = []
        
        for res in results:
            print(f"{res['feature']:<25} {res['psi']:<10.4f} {res['severity']:<15}")
            if res["severity"] == "significant":
                significant_features.append(res["feature"])
            elif res["severity"] == "moderate":
                moderate_features.append(res["feature"])
                
        print("\n=== Summary ===")
        if significant_features:
            print(f"{len(significant_features)} features show SIGNIFICANT drift: {', '.join(significant_features)}")
        if moderate_features:
            print(f"{len(moderate_features)} features show MODERATE drift: {', '.join(moderate_features)}")
            
        if not significant_features and not moderate_features:
            print(f"No significant drift detected across {len(df_live)} recent transactions.")
            
    finally:
        db.close()

if __name__ == "__main__":
    main()

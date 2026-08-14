import pandas as pd
from sqlalchemy.orm import Session

from backend.app.db.models import FeedbackLabel, ScoredTransaction
from backend.app.db.session import SessionLocal
from ml.evaluate import evaluate_model, recall_at_fixed_fpr
from ml.train import (
    compute_scale_pos_weight,
    load_and_featurize,
    load_model_artifact,
    save_model_artifact,
    time_based_split,
    train_xgboost,
)


def load_feedback_augmented_data(session: Session) -> pd.DataFrame:
    """
    Queries FeedbackLabel joined with ScoredTransaction on txn_id.
    """
    feedbacks = session.query(FeedbackLabel, ScoredTransaction).join(
        ScoredTransaction, FeedbackLabel.txn_id == ScoredTransaction.txn_id
    ).all()
    
    if not feedbacks:
        raise ValueError("Not enough feedback rows (found 0, need at least 10)")
        
    rows = []
    for fb, st in feedbacks:
        # Extract features (already computed, so we don't recompute)
        row = st.feature_snapshot.copy()
        row["txn_id"] = str(fb.txn_id)
        row["decided_at"] = fb.decided_at
        row["label"] = 1 if fb.decision == "confirmed_fraud" else 0
        rows.append(row)
        
    df = pd.DataFrame(rows)
    
    # If a txn_id has multiple feedback rows, we use only the MOST RECENT one.
    # This handles the schema design deliberately allowing multiple analyst decisions over time.
    df = df.sort_values("decided_at", ascending=False).drop_duplicates(subset=["txn_id"], keep="first")
    
    # Drop intermediate tracking columns
    df = df.drop(columns=["txn_id", "decided_at"])
    
    if len(df) < 10:
        raise ValueError(f"Not enough feedback rows (found {len(df)}, need at least 10)")
        
    print(f"\n  Loaded {len(df)} feedback-labeled transactions "
          f"({(df['label'] == 1).sum()} confirmed_fraud, {(df['label'] == 0).sum()} false_positive)")
          
    return df

def retrain_with_feedback(original_train_df: pd.DataFrame, feedback_df: pd.DataFrame, feature_columns: list, label_column: str, model_type: str):
    # This task assumes the current model_type is XGBoost class-weighted 
    # (the most common outcome of Phase 3's comparison)
    if model_type != "xgboost_class_weighted":
        print(f"Error: This retrain script currently only supports 'xgboost_class_weighted', but current model is '{model_type}'.")
        print("Stopping retrain.")
        return None
        
    # Combines original_train_df's feature columns + label with feedback_df
    augmented_df = pd.concat([original_train_df, feedback_df], ignore_index=True)
    
    fraud_ratio = augmented_df[label_column].mean()
    print(f"  Combined training set: {len(augmented_df)} rows ({len(original_train_df)} original + {len(feedback_df)} feedback), fraud ratio {fraud_ratio:.4f}")
    
    # Recompute scale_pos_weight for the augmented set (not reusing the original weight, 
    # since the class balance may have shifted)
    scale_pos_weight = compute_scale_pos_weight(augmented_df, label_column)
    print(f"  Computed scale_pos_weight: {scale_pos_weight:.2f}")
    
    # Train new model
    model = train_xgboost(augmented_df, augmented_df, feature_columns, label_column, scale_pos_weight)
    
    return model

def compare_models(old_model, new_model, test_df, feature_columns, label_column, old_threshold):
    X_test = test_df[feature_columns].values
    y_test = test_df[label_column].values
    
    # A more thorough approach would re-tune the threshold for the new model too, 
    # but using the same threshold first gives a clean, simple comparison.
    
    # We suppress standard evaluate_model prints by capturing stdout? No, evaluate_model prints are fine.
    # But the prompt wants a clean comparison table. Let's let them print and then print our own table.
    print("\n--- Evaluating Current (v1) ---")
    old_eval = evaluate_model(old_model, X_test, y_test, "Current (v1)", old_threshold)
    print("\n--- Evaluating Retrained (candidate) ---")
    new_eval = evaluate_model(new_model, X_test, y_test, "Retrained (candidate)", old_threshold)
    
    old_fpr = recall_at_fixed_fpr(old_model, X_test, y_test)
    new_fpr = recall_at_fixed_fpr(new_model, X_test, y_test)
    
    print("\n  === MODEL COMPARISON ===")
    print(f"  {'Model':<22} {'PR-AUC':<8} {'F1':<6} {'Precision':<10} {'Recall':<8} {'Recall@2%FPR':<12}")
    
    for eval_res, fpr_res in [(old_eval, old_fpr), (new_eval, new_fpr)]:
        print(f"  {eval_res['model_name']:<22} {eval_res['pr_auc']:<8.3f} {eval_res['f1']:<6.2f} "
              f"{eval_res['precision']:<10.2f} {eval_res['recall']:<8.2f} {fpr_res['recall_at_budget']:<12.2f}")

PROMOTE = False

def main():
    print("=" * 80)
    print("RiskShield Feedback Retraining Pipeline — Phase 12, Task 3")
    print("=" * 80)
    
    features_df = load_and_featurize()
    
    feature_cols = [
        "txn_count_1h", "txn_count_24h", "amount_zscore", "amount_ratio_to_avg", 
        "device_mismatch", "country_mismatch", "geo_distance_km", "amount", 
        "hour_of_day", "day_of_week"
    ]
    
    # Reproducing the EXACT same train/val/test split as Phase 3
    df_train, df_val, df_test = time_based_split(features_df)
    
    session = SessionLocal()
    try:
        feedback_df = load_feedback_augmented_data(session)
    except ValueError as e:
        print(f"\n{e}")
        return
    finally:
        session.close()
        
    try:
        old_model, metadata = load_model_artifact("model_v1")
    except FileNotFoundError:
        print("\nCould not find model_v1. Train Phase 3 first.")
        return
        
    model_type = metadata.get("model_type")
    old_threshold = metadata.get("threshold", 0.5)
    
    new_model = retrain_with_feedback(df_train, feedback_df, feature_cols, "label", model_type)
    if new_model is None:
        return
        
    compare_models(old_model, new_model, df_test, feature_cols, "label", old_threshold)
    
    print("\nReview the comparison above. To promote this model, set PROMOTE = True and re-run this script.")
    if PROMOTE:
        save_model_artifact(new_model, old_threshold, feature_cols, "model_v2", model_type)
        print("Model promoted and saved as model_v2.")
    else:
        print("This was a dry run — no model was saved.")

if __name__ == "__main__":
    main()

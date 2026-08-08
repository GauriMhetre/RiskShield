"""
Real evaluation suite for RiskShield fraud detection models.

Phase 3, Task 4: Compute proper evaluation metrics (PR-AUC, F1, precision, recall,
confusion matrix, recall@fixed-FPR) for all three model variants against a held-out
validation set, enabling fair comparison and informed model selection.

This module is separate from ml/train.py because training and evaluation are
conceptually distinct responsibilities, even though they're both part of the pipeline.

Metrics focus on the fraud class (pos_label=1) throughout, since fraud detection
is an imbalanced problem where the minority class is what we care about.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)


def evaluate_model(
    model,
    X_val,
    y_val,
    model_name: str,
    threshold: float = 0.5,
) -> dict:
    """
    Compute comprehensive evaluation metrics for a trained fraud-detection model.
    
    Metrics computed:
    - PR-AUC: Precision-Recall Area Under Curve (optimal for imbalanced data)
    - Precision: Of flagged transactions, what fraction are actually fraud?
    - Recall: Of actual fraud transactions, what fraction did we catch?
    - F1: Harmonic mean of precision and recall (fraud class)
    - Confusion Matrix: TP/FP/TN/FN breakdown
    - Predicted fraud count: How many transactions flagged as fraud?
    
    All metrics focus on pos_label=1 (fraud class), since fraud detection is
    inherently an imbalanced problem where the minority class is what matters.
    
    Args:
        model: Trained classifier (RandomForestClassifier or XGBClassifier)
        X_val: Validation feature matrix (numpy array or similar)
        y_val: Validation labels (numpy array, 0/1)
        model_name: Human-readable name for printing (e.g., "XGBoost+SMOTE")
        threshold: Decision threshold for binary classification (default 0.5)
    
    Returns:
        Dict with keys:
        - model_name: Name passed in
        - pr_auc: Precision-Recall AUC (0-1, higher is better)
        - threshold: Threshold used for binary predictions
        - precision: True positives / (true positives + false positives)
        - recall: True positives / (true positives + false negatives)
        - f1: Harmonic mean of precision and recall
        - confusion_matrix: 2D list [[TN, FP], [FN, TP]]
        - predicted_fraud_count: Number of transactions flagged as fraud
        - actual_fraud_count: Number of actual fraud transactions in val set
    
    Raises:
        ValueError: If X_val/y_val are empty or model not fitted
    """
    if len(X_val) == 0 or len(y_val) == 0:
        raise ValueError("Validation set is empty; cannot evaluate model")
    
    # Get predicted probabilities (fraud class)
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    
    # Binary predictions at the given threshold
    y_pred_binary = (y_pred_proba >= threshold).astype(int)
    
    # Compute metrics
    pr_auc = average_precision_score(y_val, y_pred_proba)
    precision = precision_score(y_val, y_pred_binary, pos_label=1, zero_division=0)
    recall = recall_score(y_val, y_pred_binary, pos_label=1, zero_division=0)
    f1 = f1_score(y_val, y_pred_binary, pos_label=1, zero_division=0)
    cm = confusion_matrix(y_val, y_pred_binary)  # [[TN, FP], [FN, TP]]
    
    # Count predictions and actual fraud
    predicted_fraud_count = (y_pred_binary == 1).sum()
    actual_fraud_count = (y_val == 1).sum()
    
    # Build result dict
    result = {
        "model_name": model_name,
        "pr_auc": pr_auc,
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm.tolist(),  # Convert to list for clean printing
        "predicted_fraud_count": predicted_fraud_count,
        "actual_fraud_count": actual_fraud_count,
    }
    
    # Print formatted output
    print(f"\n{model_name}:")
    print(f"  PR-AUC: {pr_auc:.4f}")
    print(f"  Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    print(f"  Confusion matrix: [[{tn}, {fp}], [{fn}, {tp}]]")
    print(f"  Predicted fraud: {predicted_fraud_count} | Actual fraud: {actual_fraud_count}")
    
    return result


def recall_at_fixed_fpr(
    model,
    X_val,
    y_val,
    max_fpr: float = 0.02,
) -> dict:
    """
    Find the best achievable recall within a fixed false-positive rate budget.
    
    This is the most business-realistic metric for fraud systems: instead of picking
    an arbitrary threshold, ask "if I can only tolerate flagging 2% of legitimate
    transactions (a real operational constraint — analysts can only review so much),
    what's the best recall I can get within that budget?"
    
    Uses scikit-learn's roc_curve to find the full range of thresholds and their
    associated FPR/TPR values, then searches for the best TPR (recall) where
    FPR <= max_fpr.
    
    Args:
        model: Trained classifier
        X_val: Validation feature matrix
        y_val: Validation labels (0/1)
        max_fpr: Maximum acceptable false-positive rate (default 0.02 = 2%)
    
    Returns:
        Dict with keys:
        - max_fpr: The budget passed in
        - achieved_fpr: The actual FPR at the chosen threshold (typically <= max_fpr)
        - recall_at_budget: Best achievable recall within the budget
        - threshold_used: The threshold that achieved this recall
    
    Raises:
        ValueError: If no threshold achieves FPR <= max_fpr (rare, but caught gracefully)
    """
    if len(X_val) == 0 or len(y_val) == 0:
        raise ValueError("Validation set is empty; cannot compute recall@FPR")
    
    # Get predicted probabilities
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    
    # Compute ROC curve: get FPR, TPR, and thresholds for all possible decision points
    fpr, tpr, thresholds = roc_curve(y_val, y_pred_proba)
    
    # Find the best (highest) TPR where FPR <= max_fpr
    valid_indices = np.where(fpr <= max_fpr)[0]
    
    if len(valid_indices) == 0:
        raise ValueError(
            f"No threshold achieves FPR <= {max_fpr:.4f} on this validation set. "
            f"Consider raising the FPR budget (currently {max_fpr:.2%})."
        )
    
    # Among valid thresholds, pick the one with highest TPR (recall)
    best_idx = valid_indices[np.argmax(tpr[valid_indices])]
    
    result = {
        "max_fpr": max_fpr,
        "achieved_fpr": fpr[best_idx],
        "recall_at_budget": tpr[best_idx],
        "threshold_used": thresholds[best_idx],
    }
    
    return result


def plot_pr_curve(
    models_and_names,
    X_val,
    y_val,
    save_path: str = "ml/models/pr_curve_comparison.png",
) -> None:
    """
    Plot Precision-Recall curves for multiple models on a single figure.
    
    The PR curve shows the tradeoff between precision (y-axis) and recall (x-axis)
    across all possible decision thresholds. A curve further toward the top-right
    (high precision AND high recall simultaneously) indicates a better-performing
    model.
    
    Args:
        models_and_names: List of (model, name) tuples for models to plot
        X_val: Validation feature matrix
        y_val: Validation labels (0/1)
        save_path: Path to save the figure (default ml/models/pr_curve_comparison.png)
    
    Side effects:
        - Creates ml/models/ directory if it doesn't exist
        - Saves figure to save_path
    """
    # Create output directory if needed
    output_dir = Path(save_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create figure
    plt.figure(figsize=(10, 7))
    
    # Plot PR curve for each model
    for model, name in models_and_names:
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        precision, recall, _ = precision_recall_curve(y_val, y_pred_proba)
        pr_auc = average_precision_score(y_val, y_pred_proba)
        plt.plot(recall, precision, label=f"{name} (PR-AUC={pr_auc:.3f})", linewidth=2)
    
    # Labels and formatting
    plt.xlabel("Recall (True Positive Rate)", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("Precision-Recall Curves — Model Comparison", fontsize=14, fontweight="bold")
    plt.legend(loc="best", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    
    # Save without showing (this is a script, not interactive)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def compare_all(results, fpr_results) -> None:
    """
    Print a comprehensive side-by-side comparison table for all models.
    
    This is the key decision-support output of the evaluation suite. Shows all
    important metrics in one human-readable table.
    
    Args:
        results: List of dicts from evaluate_model() (one per model)
        fpr_results: List of dicts from recall_at_fixed_fpr() (one per model)
    """
    print("\n" + "=" * 100)
    print("COMPARISON TABLE — All Models, All Metrics")
    print("=" * 100)
    print()
    
    # Build table header and rows
    header = f"{'Model':<25} {'PR-AUC':<10} {'F1':<10} {'Precision':<12} {'Recall':<10} {'Recall@2%FPR':<15}"
    print(header)
    print("-" * 100)
    
    for result, fpr_result in zip(results, fpr_results):
        model_name = result["model_name"]
        pr_auc = result["pr_auc"]
        f1 = result["f1"]
        precision = result["precision"]
        recall = result["recall"]
        recall_at_budget = fpr_result["recall_at_budget"]
        
        row = f"{model_name:<25} {pr_auc:<10.4f} {f1:<10.4f} {precision:<12.4f} {recall:<10.4f} {recall_at_budget:<15.4f}"
        print(row)
    
    print()


def select_threshold(
    model,
    X_val,
    y_val,
    min_precision: float = None,
    target_recall_at_fpr: dict = None,
) -> float:
    """
    Select an optimal decision threshold for the fraud classifier.
    
    Instead of using the default 0.5 threshold (which is arbitrary and rarely optimal),
    this function looks up a threshold from the PR curve that matches a chosen business
    criterion: either a minimum acceptable precision, or the threshold computed from
    a fixed false-positive budget (from recall_at_fixed_fpr()).
    
    If min_precision is used:
    - Finds the LOWEST threshold where precision >= min_precision
    - Why lowest, not highest? A higher threshold requires more certainty to flag fraud,
      catching fewer cases (lower recall). The lowest threshold that still meets precision
      guards maximizes the fraud you catch subject to keeping false alarms under control.
      In fraud detection, catching more fraud within a precision budget is almost always
      better than being overly conservative.
    
    If target_recall_at_fpr is used:
    - Directly uses the 'threshold_used' from recall_at_fixed_fpr()'s output
    - This threshold was already optimized for a fixed false-positive budget,
      the most business-realistic criterion for fraud systems
    
    Args:
        model: Trained classifier with predict_proba method
        X_val: Validation feature matrix
        y_val: Validation labels (0/1)
        min_precision: Minimum acceptable precision (e.g., 0.75)
        target_recall_at_fpr: Dict from recall_at_fixed_fpr() containing 'threshold_used'
    
    Returns:
        Float threshold value between 0 and 1
    
    Raises:
        ValueError: If neither or both of min_precision/target_recall_at_fpr are provided,
                   or if no threshold meets the min_precision constraint
    """
    if (min_precision is None and target_recall_at_fpr is None) or \
       (min_precision is not None and target_recall_at_fpr is not None):
        raise ValueError(
            "Exactly one of min_precision or target_recall_at_fpr must be provided, not both or neither."
        )
    
    # If using target_recall_at_fpr, just return its pre-computed threshold
    if target_recall_at_fpr is not None:
        return target_recall_at_fpr["threshold_used"]
    
    # Otherwise, use min_precision: find lowest threshold meeting the constraint
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_pred_proba)
    
    # Find indices where precision >= min_precision
    # Note: precision_recall_curve returns precisions with one extra element,
    # so we align thresholds carefully
    valid_indices = np.where(precisions[:-1] >= min_precision)[0]
    
    if len(valid_indices) == 0:
        raise ValueError(
            f"No threshold achieves precision >= {min_precision:.3f} on this validation set. "
            f"Max achievable precision: {precisions.max():.3f}. "
            f"Consider lowering min_precision."
        )
    
    # Pick the LOWEST threshold among those meeting the precision floor
    # (this maximizes recall subject to the precision constraint)
    best_idx = valid_indices[np.argmin(thresholds[valid_indices])]
    selected_threshold = thresholds[best_idx]
    
    return selected_threshold

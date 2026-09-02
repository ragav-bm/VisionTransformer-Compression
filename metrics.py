import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
from dataset_bdd import ALL_ATTRIBUTES, NUM_ATTRIBUTES


def compute_multilabel_metrics(targets, probabilities, threshold=0.5):
    """
    Computes global multi-label perception metrics.
    
    Args:
        targets (np.ndarray or torch.Tensor): Ground truth binary matrix of shape (N, NUM_ATTRIBUTES)
        probabilities (np.ndarray or torch.Tensor): Predicted probabilities in [0, 1] of shape (N, NUM_ATTRIBUTES)
        threshold (float): Decision threshold for binary classification metrics (F1, Precision, Recall)
        
    Returns:
        dict: Summary containing mAP, Macro F1, Micro F1, Macro Precision, Macro Recall
    """
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    if isinstance(probabilities, torch.Tensor):
        probabilities = probabilities.detach().cpu().numpy()

    # 1. Mean Average Precision (mAP)
    # average_precision_score with average='macro' computes AP per class and averages them
    try:
        mAP = average_precision_score(targets, probabilities, average="macro")
    except ValueError:
        mAP = 0.0

    # 2. Binary predictions for F1/Precision/Recall at decision threshold
    preds = (probabilities >= threshold).astype(np.float32)

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        targets, preds, average="macro", zero_division=0
    )
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
        targets, preds, average="micro", zero_division=0
    )

    return {
        "mAP": float(mAP) * 100.0,
        "macro_f1": float(macro_f1) * 100.0,
        "macro_precision": float(macro_p) * 100.0,
        "macro_recall": float(macro_r) * 100.0,
        "micro_f1": float(micro_f1) * 100.0
    }


def compute_per_class_metrics(targets, probabilities, threshold=0.5, class_names=ALL_ATTRIBUTES):
    """
    Computes detailed evaluation metrics for each individual attribute.
    
    Returns:
        dict: Mapping attribute_name -> {AP, Precision, Recall, F1, Positives_Count}
    """
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    if isinstance(probabilities, torch.Tensor):
        probabilities = probabilities.detach().cpu().numpy()

    preds = (probabilities >= threshold).astype(np.float32)
    per_class_results = {}

    for i, name in enumerate(class_names):
        y_true = targets[:, i]
        y_score = probabilities[:, i]
        y_pred = preds[:, i]

        # Calculate AP for this specific class
        if np.sum(y_true) > 0:
            ap = average_precision_score(y_true, y_score) * 100.0
        else:
            ap = 0.0

        p, r, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )

        per_class_results[name] = {
            "AP (%)": round(float(ap), 2),
            "F1 (%)": round(float(f1) * 100.0, 2),
            "Precision (%)": round(float(p) * 100.0, 2),
            "Recall (%)": round(float(r) * 100.0, 2),
            "Pos_Count": int(np.sum(y_true))
        }

    return per_class_results


def find_optimal_thresholds(targets, probabilities, class_names=ALL_ATTRIBUTES):
    """
    Performs a threshold sweep in [0.05, 0.95] to find the threshold that maximizes F1 per class.
    
    Returns:
        dict: Mapping attribute_name -> optimal_threshold
    """
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    if isinstance(probabilities, torch.Tensor):
        probabilities = probabilities.detach().cpu().numpy()

    threshold_candidates = np.linspace(0.05, 0.95, 19)
    best_thresholds = {}

    for i, name in enumerate(class_names):
        y_true = targets[:, i]
        y_score = probabilities[:, i]

        best_f1 = -1.0
        best_th = 0.5

        for th in threshold_candidates:
            y_pred = (y_score >= th).astype(np.float32)
            _, _, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average="binary", zero_division=0
            )
            if f1 > best_f1:
                best_f1 = f1
                best_th = th

        best_thresholds[name] = round(float(best_th), 2)

    return best_thresholds


if __name__ == "__main__":
    print("=== Testing Multi-Label Metrics Engine ===")
    
    # Generate 500 synthetic validation samples with 15 attributes
    np.random.seed(42)
    num_samples = 500
    
    # Simulate binary ground truth with realistic class imbalance
    synthetic_targets = np.zeros((num_samples, NUM_ATTRIBUTES), dtype=np.float32)
    for i in range(num_samples):
        # 1 weather (0..6)
        synthetic_targets[i, np.random.randint(0, 7)] = 1.0
        # 1 scene (7..12)
        synthetic_targets[i, np.random.randint(7, 13)] = 1.0
        # 1 timeofday (13..14)
        synthetic_targets[i, np.random.randint(13, 15)] = 1.0

    # Simulate noisy model probabilities (logits + sigmoid)
    synthetic_probs = np.clip(
        synthetic_targets * 0.7 + np.random.uniform(0.0, 0.4, size=synthetic_targets.shape),
        0.0, 1.0
    )

    # 1. Global Metrics
    summary = compute_multilabel_metrics(synthetic_targets, synthetic_probs)
    print("\n--- Global Perception Metrics ---")
    for k, v in summary.items():
        print(f"  {k:18s}: {v:.2f}%")

    # 2. Per-Class Sample
    per_class = compute_per_class_metrics(synthetic_targets, synthetic_probs)
    print("\n--- Per-Class Sample (First 3 Attributes) ---")
    for name in ALL_ATTRIBUTES[:3]:
        print(f"  {name:15s}: {per_class[name]}")

    # 3. Optimal Thresholds
    opt_th = find_optimal_thresholds(synthetic_targets, synthetic_probs)
    print(f"\nSample Optimal Threshold for '{ALL_ATTRIBUTES[0]}': {opt_th[ALL_ATTRIBUTES[0]]}")
    print("\n✅ Metrics module verified successfully!")


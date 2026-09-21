"""
Detection Performance Optimization & Alert Calibration Suite for Cyber-JEPA (Phase 7).

Provides:
1. Validation-Calibrated Operating Thresholds (tau*):
   - Youden's J Statistic: Maximizes TPR - FPR (optimal balanced operating point).
   - Cost-Sensitive F_beta: F2 (recall-prioritized) or F0.5 (precision-prioritized).
   - FPR-Constrained Threshold: Maximum threshold maintaining target FPR <= 5%.
   - Recall-Constrained Threshold: Maximum threshold maintaining target Recall >= 95%.
2. Causal Temporal Evidence Accumulation:
   - Exponential moving average smoothing: p_bar_t = alpha * p_t + (1 - alpha) * p_bar_{t-1}
   - Grouped per trajectory to ensure zero temporal boundary leakage.
   - Eliminates isolated single-step alert jitter and blue restoration artifacts.
3. End-to-End Comparative Evaluation:
   - Direct before-and-after comparison of operational detection rates and false alarms.
"""

from typing import Any
import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    roc_curve,
)


def calibrate_decision_threshold(
    y_true_val: np.ndarray,
    y_prob_val: np.ndarray,
    criterion: str = "youden",
) -> float:
    """
    Compute optimal decision threshold tau* on validation split.

    Args:
        y_true_val: Ground truth binary labels on validation set [N]
        y_prob_val: Predicted attack probabilities on validation set [N]
        criterion: One of 'youden', 'f2', 'f05', 'fpr_target_05', 'recall_target_95'

    Returns:
        Optimal scalar decision threshold in [0.01, 0.99].
    """
    if len(np.unique(y_true_val)) < 2:
        return 0.50

    fpr, tpr, roc_thresh = roc_curve(y_true_val, y_prob_val)

    if criterion == "youden":
        # Youden's J statistic: maximize TPR - FPR
        j_scores = tpr - fpr
        best_idx = int(np.argmax(j_scores))
        best_thresh = float(roc_thresh[best_idx])
    elif criterion == "f2":
        # Cost-sensitive F2: prioritize recall over precision (beta=2.0)
        candidate_thresholds = np.linspace(0.05, 0.95, 91)
        best_score = -1.0
        best_thresh = 0.50
        for th in candidate_thresholds:
            p_bin = (y_prob_val >= th).astype(int)
            score = fbeta_score(y_true_val, p_bin, beta=2.0, zero_division=0)
            if score > best_score:
                best_score = score
                best_thresh = float(th)
    elif criterion == "f05":
        # Precision-prioritized F0.5 (beta=0.5)
        candidate_thresholds = np.linspace(0.05, 0.95, 91)
        best_score = -1.0
        best_thresh = 0.50
        for th in candidate_thresholds:
            p_bin = (y_prob_val >= th).astype(int)
            score = fbeta_score(y_true_val, p_bin, beta=0.5, zero_division=0)
            if score > best_score:
                best_score = score
                best_thresh = float(th)
    elif criterion == "fpr_target_05":
        # Maximum threshold where FPR <= 0.05
        valid = np.where(fpr <= 0.05)[0]
        if len(valid) > 0:
            best_thresh = float(roc_thresh[valid[-1]])
        else:
            best_thresh = 0.50
    elif criterion == "recall_target_95":
        # Maximum threshold where TPR >= 0.95
        valid = np.where(tpr >= 0.95)[0]
        if len(valid) > 0:
            best_thresh = float(roc_thresh[valid[0]])
        else:
            best_thresh = 0.50
    else:
        best_thresh = 0.50

    return float(np.clip(best_thresh, 0.05, 0.95))


def apply_temporal_smoothing(
    probs: np.ndarray,
    trajectory_ids: list[str],
    alpha: float = 0.60,
) -> np.ndarray:
    """
    Apply causal exponential moving average smoothing within each trajectory.
    
    p_bar_t = alpha * p_t + (1 - alpha) * p_bar_{t-1}
    with p_bar_0 = p_0.

    Args:
        probs: 1D array of predicted probabilities [N]
        trajectory_ids: List of trajectory identifiers [N]
        alpha: Weight for current observation in (0, 1]. alpha=1.0 is no smoothing.

    Returns:
        1D array of smoothed probabilities [N]
    """
    smoothed = np.zeros_like(probs, dtype=float)
    unique_trajs = sorted(list(set(trajectory_ids)))

    for traj in unique_trajs:
        idx = [i for i, tid in enumerate(trajectory_ids) if tid == traj]
        traj_p = probs[idx]
        s_val = float(traj_p[0])
        for step_i, orig_i in enumerate(idx):
            if step_i == 0:
                s_val = float(traj_p[step_i])
            else:
                s_val = float(alpha * traj_p[step_i] + (1.0 - alpha) * s_val)
            smoothed[orig_i] = s_val

    return smoothed


def evaluate_optimized_detection(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    trajectory_ids: list[str],
    threshold: float = 0.50,
    temporal_alpha: float | None = None,
) -> dict[str, Any]:
    """
    Compute operational metrics with calibrated threshold and optional temporal smoothing.

    Args:
        y_true: Ground truth binary labels [N]
        y_prob: Raw predicted probabilities [N]
        trajectory_ids: Trajectory IDs [N]
        threshold: Operating decision threshold tau
        temporal_alpha: Optional EMA smoothing factor (e.g. 0.60)

    Returns:
        Dictionary of operational attack and detection metrics.
    """
    if temporal_alpha is not None and 0.0 < temporal_alpha < 1.0:
        eval_probs = apply_temporal_smoothing(y_prob, trajectory_ids, alpha=temporal_alpha)
    else:
        eval_probs = y_prob

    preds = (eval_probs >= threshold).astype(int)

    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    tot_att = tp + fn
    tot_clean = tn + fp
    det_rate = float((tp / max(1, tot_att)) * 100.0) if tot_att > 0 else 100.0
    fa_rate = float((fp / max(1, tot_clean)) * 100.0) if tot_clean > 0 else 0.0
    prec = float((tp / max(1, tp + fp)) * 100.0)
    macro_f1 = float(f1_score(y_true, preds, average="macro", zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_true, preds))

    return {
        "operating_threshold": float(threshold),
        "temporal_smoothing_alpha": float(temporal_alpha) if temporal_alpha is not None else 1.0,
        "total_attacks": tot_att,
        "attacks_caught": tp,
        "attacks_missed": fn,
        "clean_steps": tot_clean,
        "false_alarms": fp,
        "attack_detection_rate_pct": det_rate,
        "false_alarm_rate_pct": fa_rate,
        "precision_pct": prec,
        "macro_f1": macro_f1,
        "balanced_accuracy": bal_acc,
    }

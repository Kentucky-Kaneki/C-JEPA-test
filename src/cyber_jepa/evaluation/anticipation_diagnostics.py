"""
Anticipation Gap Diagnostic Suite grounded in NIST SP 800-61 Rev. 2.

NIST SP 800-61 Rev. 2 defines:
- Precursor: A sign that an incident may occur in the future (forecasted risk at t+k).
- Indicator: A sign that an incident has occurred or is occurring currently (telemetry at t).

The Anticipation Gap Benchmark tests whether the JEPA future rollout z_hat_{t+k}
provides genuine predictive foresight over and above the current telemetry representation z_t:
    Delta_anticipation = Metric(h_future(z_hat_{t+k})) - Metric(h_current(z_t))

If Delta_anticipation > 0, the world model actively synthesizes new anticipatory information
that is not linearly accessible from the latest observation alone.
"""

from typing import Any
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, balanced_accuracy_score

from cyber_jepa.evaluation.probes import compute_bootstrap_ci


def evaluate_anticipation_gap(
    train_z_current: np.ndarray,
    train_z_future: np.ndarray,
    train_labels: np.ndarray,
    test_z_current: np.ndarray,
    test_z_future: np.ndarray,
    test_labels: np.ndarray,
    c_val: float = 1.0,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Train and evaluate paired precursor (future rollout) vs indicator (current state) probes.
    
    Args:
        train_z_current: [N_train, D] frozen context representations z_t
        train_z_future: [N_train, D] predicted future representations z_hat_{t+k}
        train_labels: [N_train] binary compromise ground-truth at t+k
        test_z_current: [N_test, D] test context representations z_t
        test_z_future: [N_test, D] test predicted future representations z_hat_{t+k}
        test_labels: [N_test] test binary compromise ground-truth at t+k
        c_val: LogisticRegression regularization parameter
        seed: Random seed for solver and bootstrapping
    """
    # 1. Current State Baseline Probe (Indicator Probe: z_t -> y_{t+k})
    clf_current = LogisticRegression(C=c_val, max_iter=500, random_state=seed)
    clf_current.fit(train_z_current, train_labels)
    preds_current = clf_current.predict(test_z_current)
    probs_current = clf_current.predict_proba(test_z_current)[:, 1] if len(np.unique(test_labels)) > 1 else np.zeros(len(test_labels))

    f1_curr, f1_curr_low, f1_curr_high = compute_bootstrap_ci(
        test_labels, preds_current, lambda y, p: f1_score(y, p, average="macro", zero_division=0), seed=seed
    )
    auroc_curr = float(roc_auc_score(test_labels, probs_current)) if len(np.unique(test_labels)) > 1 else 0.5
    bal_acc_curr = float(balanced_accuracy_score(test_labels, preds_current))

    # 2. Anticipatory Future Rollout Probe (Precursor Probe: z_hat_{t+k} -> y_{t+k})
    clf_future = LogisticRegression(C=c_val, max_iter=500, random_state=seed)
    clf_future.fit(train_z_future, train_labels)
    preds_future = clf_future.predict(test_z_future)
    probs_future = clf_future.predict_proba(test_z_future)[:, 1] if len(np.unique(test_labels)) > 1 else np.zeros(len(test_labels))

    f1_fut, f1_fut_low, f1_fut_high = compute_bootstrap_ci(
        test_labels, preds_future, lambda y, p: f1_score(y, p, average="macro", zero_division=0), seed=seed
    )
    auroc_fut = float(roc_auc_score(test_labels, probs_future)) if len(np.unique(test_labels)) > 1 else 0.5
    bal_acc_fut = float(balanced_accuracy_score(test_labels, preds_future))

    # 3. Paired Anticipation Gap
    delta_f1 = float(f1_fut - f1_curr)
    delta_auroc = float(auroc_fut - auroc_curr)
    delta_bal_acc = float(bal_acc_fut - bal_acc_curr)

    return {
        "current_indicator_probe": {
            "macro_f1": float(f1_curr),
            "macro_f1_ci_95": [float(f1_curr_low), float(f1_curr_high)],
            "auroc": float(auroc_curr),
            "balanced_accuracy": float(bal_acc_curr),
        },
        "future_precursor_probe": {
            "macro_f1": float(f1_fut),
            "macro_f1_ci_95": [float(f1_fut_low), float(f1_fut_high)],
            "auroc": float(auroc_fut),
            "balanced_accuracy": float(bal_acc_fut),
        },
        "anticipation_gap": {
            "delta_macro_f1": delta_f1,
            "delta_auroc": delta_auroc,
            "delta_balanced_accuracy": delta_bal_acc,
            "has_anticipation_advantage": bool(delta_f1 > 0 and delta_auroc >= 0),
        },
    }

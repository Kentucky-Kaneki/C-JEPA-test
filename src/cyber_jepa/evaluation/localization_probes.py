"""
Multi-Host Compromise Localization Diagnostic Suite grounded in MITRE ATT&CK v14.

Maps CybORG 3.1 Scenario1b hosts to Cyber Kill Chain & MITRE ATT&CK tactical tiers:
1. User Tier (Initial Access & Reconnaissance): User1, User2, User3, User4
2. Enterprise Tier (Lateral Movement & Privilege Escalation): Enterprise0, Enterprise1, Enterprise2
3. Operational Tier (Crown Jewel & Operational Impact): Op_Server0, Op_Host0, Op_Host1, Op_Host2

Evaluates whether predicted future latents z_hat_{t+k} accurately localize which
specific network host is compromised along the kill chain.
Includes base-rate fallacy metrics (Arp et al., USENIX Security 2022; Sommer & Paxson, IEEE S&P 2010):
- True positive rate (Recall) vs False Positive Rate (FPR)
- FPR @ 95% Recall on natural unbalanced test distributions
"""

from typing import Any
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_recall_curve,
    auc,
)

from cyber_jepa.data.dataset import MONITORED_HOSTS

# MITRE ATT&CK Kill-Chain Subnet Tiers
MITRE_TIERS = {
    "user_tier": ["User1", "User2", "User3", "User4"],
    "enterprise_tier": ["Enterprise0", "Enterprise1", "Enterprise2"],
    "operational_tier": ["Op_Server0", "Op_Host0", "Op_Host1", "Op_Host2"],
}


def compute_fpr_at_recall(y_true: np.ndarray, y_prob: np.ndarray, target_recall: float = 0.95) -> float:
    """Compute False Positive Rate at a target Recall threshold (Base-rate fallacy audit)."""
    if len(np.unique(y_true)) < 2:
        return 0.0
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    # Find smallest threshold where recall >= target_recall
    valid_idx = np.where(recalls >= target_recall)[0]
    if len(valid_idx) == 0:
        threshold = 0.0
    else:
        best_idx = valid_idx[-1]
        threshold = thresholds[min(best_idx, len(thresholds) - 1)]

    y_pred = (y_prob >= threshold).astype(int)
    negatives = (y_true == 0)
    fp = np.sum((y_pred == 1) & negatives)
    n = np.sum(negatives)
    return float(fp / max(1, n))


def evaluate_multi_host_localization(
    train_latents: np.ndarray,
    train_host_compromised: np.ndarray,
    test_latents: np.ndarray,
    test_host_compromised: np.ndarray,
    host_names: list[str] = MONITORED_HOSTS,
    c_val: float = 1.0,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Train per-host linear probes and compute localization metrics across MITRE tiers.
    
    Args:
        train_latents: [N_train, D] predicted future latents z_hat_{t+k}
        train_host_compromised: [N_train, num_hosts] binary compromise flags
        test_latents: [N_test, D] predicted future latents z_hat_{t+k}
        test_host_compromised: [N_test, num_hosts] binary compromise flags
        host_names: List of monitored host names
        c_val: Regularization parameter
        seed: Random seed
    """
    results_by_host: dict[str, dict[str, float]] = {}
    predicted_probs_matrix = np.zeros_like(test_host_compromised, dtype=np.float32)

    for idx, host in enumerate(host_names):
        y_train = train_host_compromised[:, idx]
        y_test = test_host_compromised[:, idx]

        n_pos_train = int(np.sum(y_train))
        n_pos_test = int(np.sum(y_test))

        # Check if host exhibits variation in both splits
        if len(np.unique(y_train)) < 2:
            results_by_host[host] = {
                "macro_f1": 1.0 if n_pos_test == 0 else 0.0,
                "auroc": 1.0 if n_pos_test == 0 else 0.5,
                "fpr_at_95_recall": 0.0,
                "pr_auc": 0.0,
                "train_positives": n_pos_train,
                "test_positives": n_pos_test,
                "status": "constant_in_training",
            }
            continue

        clf = LogisticRegression(C=c_val, max_iter=500, random_state=seed)
        clf.fit(train_latents, y_train)

        probs = clf.predict_proba(test_latents)[:, 1]
        predicted_probs_matrix[:, idx] = probs
        preds = clf.predict(test_latents)

        f1 = float(f1_score(y_test, preds, average="macro", zero_division=0))
        auroc = float(roc_auc_score(y_test, probs)) if len(np.unique(y_test)) > 1 else 1.0
        fpr_95 = compute_fpr_at_recall(y_test, probs, target_recall=0.95)

        # Precision-Recall AUC
        if len(np.unique(y_test)) > 1:
            p_curve, r_curve, _ = precision_recall_curve(y_test, probs)
            pr_auc = float(auc(r_curve, p_curve))
        else:
            pr_auc = 1.0

        results_by_host[host] = {
            "macro_f1": f1,
            "auroc": auroc,
            "fpr_at_95_recall": fpr_95,
            "pr_auc": pr_auc,
            "train_positives": n_pos_train,
            "test_positives": n_pos_test,
            "status": "evaluated",
        }

    # Tier-level aggregations
    tier_results = {}
    for tier_name, tier_hosts in MITRE_TIERS.items():
        eval_hosts = [h for h in tier_hosts if h in results_by_host and results_by_host[h]["status"] == "evaluated"]
        if eval_hosts:
            tier_results[tier_name] = {
                "mean_auroc": float(np.mean([results_by_host[h]["auroc"] for h in eval_hosts])),
                "mean_macro_f1": float(np.mean([results_by_host[h]["macro_f1"] for h in eval_hosts])),
                "mean_fpr_95": float(np.mean([results_by_host[h]["fpr_at_95_recall"] for h in eval_hosts])),
                "active_hosts": eval_hosts,
            }
        else:
            tier_results[tier_name] = {
                "mean_auroc": 1.0,
                "mean_macro_f1": 1.0,
                "mean_fpr_95": 0.0,
                "active_hosts": [],
            }

    # Top-1 Host Attribution Precision (when any host is compromised in test step)
    any_compromised = np.any(test_host_compromised > 0, axis=1)
    if np.sum(any_compromised) > 0:
        top_predicted_hosts = np.argmax(predicted_probs_matrix[any_compromised], axis=1)
        true_compromised_sub = test_host_compromised[any_compromised]
        # Check if the highest risk host was indeed compromised
        correct_attributions = np.array([
            true_compromised_sub[i, top_predicted_hosts[i]] > 0
            for i in range(len(top_predicted_hosts))
        ])
        top1_attribution_accuracy = float(np.mean(correct_attributions))
    else:
        top1_attribution_accuracy = 1.0

    return {
        "by_host": results_by_host,
        "by_mitre_tier": tier_results,
        "top1_host_attribution_accuracy": top1_attribution_accuracy,
        "overall_mean_auroc": float(np.mean([r["auroc"] for r in results_by_host.values() if r["status"] == "evaluated"])),
        "overall_mean_macro_f1": float(np.mean([r["macro_f1"] for r in results_by_host.values() if r["status"] == "evaluated"])),
    }

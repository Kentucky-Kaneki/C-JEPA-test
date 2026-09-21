"""
Emergent Learning, Out-of-Distribution (OOD) Generalization & Zero-Label Anomaly Suite.

Provides:
1. Unsupervised Zero-Label Latent Anomaly Detection:
   - Measures latent drift from clean reference baseline centroid.
   - Evaluates Cosine and Euclidean latent distance against ground-truth attacks
     WITHOUT ANY LABELED PROBE TRAINING (pure emergent self-supervised detection).
   - Reports AUROC, PR-AUC, and True Positive Rate at 95% Specificity (FPR <= 5%).
2. Cross-Policy Zero-Shot Transfer:
   - B-line (Targeted) -> Meander (Stealth OOD): tests generalization to unseen exploratory evasion.
   - Meander (Stealth) -> B-line (Targeted OOD): tests generalization to rapid linear killchains.
   - Computes Zero-Shot Retention Rate (% of in-distribution F1 preserved on unseen attacks).
3. MITRE ATT&CK Multi-Tier Host Localization:
   - Evaluates attribution across User Tier (Initial Access), Enterprise Tier (Lateral Movement),
     and Operational Tier (Critical Sabotage).
"""

from typing import Any
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)

from cyber_jepa.data.dataset import MONITORED_HOSTS
from cyber_jepa.evaluation.localization_probes import evaluate_multi_host_localization, MITRE_TIERS


def evaluate_zero_label_latent_anomaly(
    clean_reference_latents: np.ndarray,
    test_latents: np.ndarray,
    test_labels: np.ndarray,
) -> dict[str, Any]:
    """
    Evaluate emergent zero-label anomaly detection using latent distance from clean centroid.

    Args:
        clean_reference_latents: [N_clean, D] known clean baseline states (e.g. t <= 5 in training)
        test_latents: [N_test, D] test latent representations
        test_labels: [N_test] ground truth binary attack labels

    Returns:
        Dictionary with zero-label AUROC, PR-AUC, and operating metrics.
    """
    if len(clean_reference_latents) == 0 or len(test_latents) == 0:
        return {}

    # Compute reference centroid
    centroid = np.mean(clean_reference_latents, axis=0)

    # 1. Cosine Distance: 1 - (z . c) / (||z|| * ||c||)
    norm_test = test_latents / np.maximum(1e-12, np.linalg.norm(test_latents, axis=1, keepdims=True))
    norm_c = centroid / max(1e-12, np.linalg.norm(centroid))
    cosine_dist = 1.0 - (norm_test @ norm_c)

    # 2. Euclidean Distance: ||z - c||_2
    euclidean_dist = np.linalg.norm(test_latents - centroid, axis=1)

    has_both = len(np.unique(test_labels)) > 1
    cos_auroc = float(roc_auc_score(test_labels, cosine_dist)) if has_both else 0.5
    cos_prauc = float(average_precision_score(test_labels, cosine_dist)) if has_both else 0.0

    euc_auroc = float(roc_auc_score(test_labels, euclidean_dist)) if has_both else 0.5
    euc_prauc = float(average_precision_score(test_labels, euclidean_dist)) if has_both else 0.0

    # Detection Rate at 95% Specificity (FPR <= 0.05) using Cosine distance
    tpr_at_95_spec = 0.0
    if has_both:
        fpr, tpr, _ = roc_curve(test_labels, cosine_dist)
        valid = np.where(fpr <= 0.05)[0]
        if len(valid) > 0:
            tpr_at_95_spec = float(tpr[valid[-1]])

    return {
        "cosine_distance": {
            "auroc": cos_auroc,
            "pr_auc": cos_prauc,
            "tpr_at_95_specificity": tpr_at_95_spec,
            "clean_mean_distance": float(np.mean(cosine_dist[test_labels == 0])) if np.sum(test_labels == 0) > 0 else 0.0,
            "attack_mean_distance": float(np.mean(cosine_dist[test_labels == 1])) if np.sum(test_labels == 1) > 0 else 0.0,
        },
        "euclidean_distance": {
            "auroc": euc_auroc,
            "pr_auc": euc_prauc,
            "clean_mean_distance": float(np.mean(euclidean_dist[test_labels == 0])) if np.sum(test_labels == 0) > 0 else 0.0,
            "attack_mean_distance": float(np.mean(euclidean_dist[test_labels == 1])) if np.sum(test_labels == 1) > 0 else 0.0,
        },
        "emergent_detection_viable": bool(cos_auroc >= 0.75),
    }


def evaluate_cross_policy_transfer(
    train_latents: np.ndarray,
    train_labels: np.ndarray,
    test_latents_indist: np.ndarray,
    test_labels_indist: np.ndarray,
    test_latents_ood: np.ndarray,
    test_labels_ood: np.ndarray,
    source_policy_name: str,
    target_policy_name: str,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Train probe on source policy and evaluate zero-shot transfer to unseen target policy.

    Args:
        train_latents: [N_train, D] from source policy
        train_labels: [N_train] binary labels
        test_latents_indist: [N_test_in, D] from source policy
        test_labels_indist: [N_test_in]
        test_latents_ood: [N_test_ood, D] from unseen target policy
        test_labels_ood: [N_test_ood]
        source_policy_name: Name of source policy (e.g. 'bline')
        target_policy_name: Name of unseen target policy (e.g. 'meander')

    Returns:
        Comparative dictionary with in-distribution vs zero-shot OOD metrics.
    """
    clf = LogisticRegression(C=1.0, max_iter=500, random_state=seed)
    clf.fit(train_latents, train_labels)

    # 1. In-distribution evaluation
    pred_in = clf.predict(test_latents_indist)
    prob_in = clf.predict_proba(test_latents_indist)[:, 1] if len(np.unique(test_labels_indist)) > 1 else np.zeros(len(test_labels_indist))
    f1_in = float(f1_score(test_labels_indist, pred_in, average="macro", zero_division=0))
    auroc_in = float(roc_auc_score(test_labels_indist, prob_in)) if len(np.unique(test_labels_indist)) > 1 else 0.5
    cm_in = confusion_matrix(test_labels_indist, pred_in, labels=[0, 1])
    det_in = float((cm_in[1, 1] / max(1, cm_in[1, 1] + cm_in[1, 0])) * 100.0)

    # 2. Zero-shot Out-of-Distribution evaluation
    pred_ood = clf.predict(test_latents_ood)
    prob_ood = clf.predict_proba(test_latents_ood)[:, 1] if len(np.unique(test_labels_ood)) > 1 else np.zeros(len(test_labels_ood))
    f1_ood = float(f1_score(test_labels_ood, pred_ood, average="macro", zero_division=0))
    auroc_ood = float(roc_auc_score(test_labels_ood, prob_ood)) if len(np.unique(test_labels_ood)) > 1 else 0.5
    cm_ood = confusion_matrix(test_labels_ood, pred_ood, labels=[0, 1])
    det_ood = float((cm_ood[1, 1] / max(1, cm_ood[1, 1] + cm_ood[1, 0])) * 100.0)
    fa_ood = float((cm_ood[0, 1] / max(1, cm_ood[0, 0] + cm_ood[0, 1])) * 100.0)

    retention_rate = float((f1_ood / max(1e-6, f1_in)) * 100.0)

    return {
        "source_policy": source_policy_name,
        "target_policy_unseen": target_policy_name,
        "in_distribution": {
            "macro_f1": f1_in,
            "auroc": auroc_in,
            "detection_rate_pct": det_in,
        },
        "zero_shot_ood": {
            "macro_f1": f1_ood,
            "auroc": auroc_ood,
            "detection_rate_pct": det_ood,
            "false_alarm_rate_pct": fa_ood,
            "attacks_caught": int(cm_ood[1, 1]),
            "attacks_missed": int(cm_ood[1, 0]),
            "total_attacks": int(cm_ood[1, 1] + cm_ood[1, 0]),
        },
        "zero_shot_retention_rate_pct": retention_rate,
        "generalization_penalty_delta_f1": float(f1_ood - f1_in),
    }


def evaluate_mitre_tier_summary(
    train_latents: np.ndarray,
    train_host_comp: np.ndarray,
    test_latents: np.ndarray,
    test_host_comp: np.ndarray,
    host_names: list[str] = MONITORED_HOSTS,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Execute MITRE ATT&CK host compromise localization and aggregate by tactical tier.

    Returns:
        Structured breakdown across User Tier, Enterprise Tier, and Operational Tier.
    """
    loc_results = evaluate_multi_host_localization(
        train_latents=train_latents,
        train_host_compromised=train_host_comp,
        test_latents=test_latents,
        test_host_compromised=test_host_comp,
        host_names=host_names,
        seed=seed,
    )

    by_host = loc_results.get("by_host", {})
    tier_summary: dict[str, Any] = {}

    for tier_name, tier_hosts in MITRE_TIERS.items():
        tier_f1s = []
        tier_aurocs = []
        tier_fprs = []
        for h in tier_hosts:
            if h in by_host:
                h_res = by_host[h]
                tier_f1s.append(h_res.get("macro_f1", 0.0))
                tier_aurocs.append(h_res.get("auroc", 0.5))
                tier_fprs.append(h_res.get("fpr_at_95_recall", 1.0))

        tier_summary[tier_name] = {
            "mean_macro_f1": float(np.mean(tier_f1s)) if tier_f1s else 0.0,
            "mean_auroc": float(np.mean(tier_aurocs)) if tier_aurocs else 0.5,
            "mean_fpr_at_95_recall": float(np.mean(tier_fprs)) if tier_fprs else 1.0,
            "monitored_hosts": tier_hosts,
        }

    return {
        "overall_mean_macro_f1": loc_results.get("overall_mean_macro_f1", 0.0),
        "overall_mean_auroc": loc_results.get("overall_mean_auroc", 0.0),
        "tier_summary": tier_summary,
        "by_host": by_host,
    }

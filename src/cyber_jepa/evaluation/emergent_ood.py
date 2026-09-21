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
from cyber_jepa.evaluation.detection_optimization import (
    apply_temporal_smoothing,
    calibrate_decision_threshold,
)
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


def evaluate_operational_zero_day_detection(
    clean_reference_latents: np.ndarray,
    test_latents: np.ndarray,
    test_labels: np.ndarray,
    test_host_comp: np.ndarray | None = None,
    quantiles: list[float] | None = None,
) -> dict[str, Any]:
    """
    Evaluate real-world operational zero-day detection metrics without labeled attack training.
    
    Thresholds are calibrated strictly on clean reference baseline telemetry at designated quantiles
    (e.g., 90th, 95th, 98th percentile of benign latent drift).
    
    Args:
        clean_reference_latents: [N_clean, D] Uncompromised baseline states (e.g. early steps in training)
        test_latents: [N_test, D] Test latent representations
        test_labels: [N_test] Ground truth binary attack labels (Op_Server0 compromise)
        test_host_comp: Optional [N_test, num_hosts] per-host compromise indicators
        quantiles: List of clean calibration percentiles (default: [0.80, 0.85, 0.90, 0.95, 0.98])
        
    Returns:
        Structured dictionary containing raw counts (attacks caught/missed), detection rates,
        false alarm rates, and benign false alarm audit per operating quantile.
    """
    if len(clean_reference_latents) == 0 or len(test_latents) == 0:
        return {}

    if quantiles is None:
        quantiles = [0.80, 0.85, 0.90, 0.95, 0.98]

    # Compute reference centroid
    centroid = np.mean(clean_reference_latents, axis=0)

    # Cosine distance from centroid
    norm_test = test_latents / np.maximum(1e-12, np.linalg.norm(test_latents, axis=1, keepdims=True))
    norm_c = centroid / max(1e-12, np.linalg.norm(centroid))
    test_cos_dist = 1.0 - (norm_test @ norm_c)

    norm_clean = clean_reference_latents / np.maximum(1e-12, np.linalg.norm(clean_reference_latents, axis=1, keepdims=True))
    clean_cos_dist = 1.0 - (norm_clean @ norm_c)

    has_both = len(np.unique(test_labels)) > 1
    cos_auroc = float(roc_auc_score(test_labels, test_cos_dist)) if has_both else 0.5
    cos_prauc = float(average_precision_score(test_labels, test_cos_dist)) if has_both else 0.0

    # Perimeter intrusion and true benign masks
    if test_host_comp is not None and len(test_host_comp) == len(test_labels):
        any_comp = (np.sum(test_host_comp > 0, axis=1) > 0).astype(int)
        true_benign = (test_labels == 0) & (any_comp == 0)
    else:
        any_comp = None
        true_benign = (test_labels == 0)

    operating_points = {}
    for q in quantiles:
        th = float(np.percentile(clean_cos_dist, q * 100.0))
        preds = (test_cos_dist >= th).astype(int)

        cm = confusion_matrix(test_labels, preds, labels=[0, 1])
        tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

        tot_att = tp + fn
        tot_clean = tn + fp
        det_rate = float((tp / max(1, tot_att)) * 100.0) if tot_att > 0 else 100.0
        far = float((fp / max(1, tot_clean)) * 100.0) if tot_clean > 0 else 0.0
        f1 = float(f1_score(test_labels, preds, average="macro", zero_division=0))
        prec = float((tp / max(1, tp + fp)) * 100.0)
        bal_acc = float(balanced_accuracy_score(test_labels, preds))

        point_info: dict[str, Any] = {
            "calibration_quantile": float(q),
            "threshold_value": th,
            "crown_jewel_attacks_caught": tp,
            "crown_jewel_attacks_missed": fn,
            "crown_jewel_total_attacks": tot_att,
            "crown_jewel_detection_rate_pct": det_rate,
            "nominal_clean_steps": tot_clean,
            "nominal_false_alarms": fp,
            "nominal_false_alarm_rate_pct": far,
            "macro_f1": f1,
            "precision_pct": prec,
            "balanced_accuracy": bal_acc,
        }

        if any_comp is not None:
            tp_any = int(np.sum((any_comp == 1) & (preds == 1)))
            fn_any = int(np.sum((any_comp == 1) & (preds == 0)))
            tot_any = tp_any + fn_any
            perim_det_rate = float((tp_any / max(1, tot_any)) * 100.0) if tot_any > 0 else 100.0

            tb_steps = int(np.sum(true_benign))
            tb_fps = int(np.sum(true_benign & (preds == 1)))
            tb_far = float((tb_fps / max(1, tb_steps)) * 100.0) if tb_steps > 0 else 0.0

            point_info["perimeter_intrusions_caught"] = tp_any
            point_info["perimeter_intrusions_missed"] = fn_any
            point_info["perimeter_total_intrusions"] = tot_any
            point_info["perimeter_detection_rate_pct"] = perim_det_rate
            point_info["true_benign_steps"] = tb_steps
            point_info["true_benign_false_alarms"] = tb_fps
            point_info["true_benign_false_alarm_rate_pct"] = tb_far

        key_name = f"q_{int(round(q * 100))}"
        operating_points[key_name] = point_info

    return {
        "auroc": cos_auroc,
        "pr_auc": cos_prauc,
        "clean_mean_distance": float(np.mean(clean_cos_dist)),
        "test_clean_mean_distance": float(np.mean(test_cos_dist[test_labels == 0])) if np.sum(test_labels == 0) > 0 else 0.0,
        "test_attack_mean_distance": float(np.mean(test_cos_dist[test_labels == 1])) if np.sum(test_labels == 1) > 0 else 0.0,
        "operating_points": operating_points,
    }


def evaluate_jepa_energy_anomaly(
    clean_energies: np.ndarray,
    test_energies: np.ndarray,
    test_labels: np.ndarray,
    test_host_comp: np.ndarray | None = None,
    quantiles: list[float] | None = None,
) -> dict[str, Any]:
    """
    Evaluate unsupervised threat detection via JEPA prediction energy E(x, y, a).
    
    In Yann LeCun's JEPA formulation, Energy measures incompatibility between the predicted
    future latent z_hat and actual observed future target latent z_target:
        E(x, y, a) = 1 - cos(z_hat, z_target)
        
    Clean transitions adhere to the forward dynamics model (low energy).
    Zero-day attacks violate the normal transition manifold (high energy).
    """
    if len(clean_energies) == 0 or len(test_energies) == 0:
        return {}

    if quantiles is None:
        quantiles = [0.80, 0.85, 0.90, 0.95, 0.98]

    has_both = len(np.unique(test_labels)) > 1
    auroc = float(roc_auc_score(test_labels, test_energies)) if has_both else 0.5
    prauc = float(average_precision_score(test_labels, test_energies)) if has_both else 0.0

    operating_points = {}
    for q in quantiles:
        th = float(np.percentile(clean_energies, q * 100.0))
        preds = (test_energies >= th).astype(int)
        cm = confusion_matrix(test_labels, preds, labels=[0, 1])
        tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

        tot_att = tp + fn
        tot_clean = tn + fp
        det_rate = float((tp / max(1, tot_att)) * 100.0) if tot_att > 0 else 100.0
        far = float((fp / max(1, tot_clean)) * 100.0) if tot_clean > 0 else 0.0
        f1 = float(f1_score(test_labels, preds, average="macro", zero_division=0))

        operating_points[f"q_{int(round(q * 100))}"] = {
            "calibration_quantile": float(q),
            "threshold_value": th,
            "attacks_caught": tp,
            "attacks_missed": fn,
            "total_attacks": tot_att,
            "detection_rate_pct": det_rate,
            "false_alarms": fp,
            "false_alarm_rate_pct": far,
            "macro_f1": f1,
        }

    return {
        "auroc": auroc,
        "pr_auc": prauc,
        "clean_mean_energy": float(np.mean(clean_energies)),
        "attack_mean_energy": float(np.mean(test_energies[test_labels == 1])) if np.sum(test_labels == 1) > 0 else 0.0,
        "benign_mean_energy": float(np.mean(test_energies[test_labels == 0])) if np.sum(test_labels == 0) > 0 else 0.0,
        "operating_points": operating_points,
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
    val_latents_indist: np.ndarray | None = None,
    val_labels_indist: np.ndarray | None = None,
    test_trajectories_ood: list[str] | None = None,
    temporal_alpha: float = 0.60,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Train probe on source policy and evaluate zero-shot transfer to unseen target policy,
    incorporating validation threshold calibration and causal temporal smoothing.
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

    # 2. Threshold Calibration on Validation Split (if provided)
    if val_latents_indist is not None and val_labels_indist is not None and len(np.unique(val_labels_indist)) > 1:
        val_probs = clf.predict_proba(val_latents_indist)[:, 1]
        tau_youden = calibrate_decision_threshold(val_labels_indist, val_probs, criterion="youden")
        tau_f2 = calibrate_decision_threshold(val_labels_indist, val_probs, criterion="f2")
    else:
        tau_youden = 0.50
        tau_f2 = 0.50

    # 3. Zero-shot Out-of-Distribution evaluation
    prob_ood = clf.predict_proba(test_latents_ood)[:, 1] if len(np.unique(test_labels_ood)) > 1 else np.zeros(len(test_labels_ood))
    auroc_ood = float(roc_auc_score(test_labels_ood, prob_ood)) if len(np.unique(test_labels_ood)) > 1 else 0.5

    def _eval_ood_point(probs: np.ndarray, thresh: float) -> dict[str, Any]:
        preds = (probs >= thresh).astype(int)
        cm = confusion_matrix(test_labels_ood, preds, labels=[0, 1])
        tp = int(cm[1, 1])
        fn = int(cm[1, 0])
        fp = int(cm[0, 1])
        tn = int(cm[0, 0])
        tot_att = tp + fn
        tot_clean = tn + fp
        det_r = float((tp / max(1, tot_att)) * 100.0) if tot_att > 0 else 100.0
        fa_r = float((fp / max(1, tot_clean)) * 100.0) if tot_clean > 0 else 0.0
        f1_val = float(f1_score(test_labels_ood, preds, average="macro", zero_division=0))
        return {
            "threshold": float(thresh),
            "macro_f1": f1_val,
            "detection_rate_pct": det_r,
            "false_alarm_rate_pct": fa_r,
            "attacks_caught": tp,
            "attacks_missed": fn,
            "total_attacks": tot_att,
            "false_alarms": fp,
        }

    ood_base = _eval_ood_point(prob_ood, 0.50)
    ood_youden = _eval_ood_point(prob_ood, tau_youden)
    ood_f2 = _eval_ood_point(prob_ood, tau_f2)

    # Optional temporal smoothing on OOD episodes
    if test_trajectories_ood is not None and len(test_trajectories_ood) == len(test_labels_ood):
        smoothed_ood = apply_temporal_smoothing(prob_ood, test_trajectories_ood, alpha=temporal_alpha)
        ood_smoothed = _eval_ood_point(smoothed_ood, tau_youden)
    else:
        ood_smoothed = ood_youden

    retention_rate = float((ood_youden["macro_f1"] / max(1e-6, f1_in)) * 100.0)

    return {
        "source_policy": source_policy_name,
        "target_policy_unseen": target_policy_name,
        "in_distribution": {
            "macro_f1": f1_in,
            "auroc": auroc_in,
            "detection_rate_pct": det_in,
        },
        "zero_shot_ood": {
            "macro_f1": ood_base["macro_f1"],
            "auroc": auroc_ood,
            "detection_rate_pct": ood_base["detection_rate_pct"],
            "false_alarm_rate_pct": ood_base["false_alarm_rate_pct"],
            "attacks_caught": ood_base["attacks_caught"],
            "attacks_missed": ood_base["attacks_missed"],
            "total_attacks": ood_base["total_attacks"],
        },
        "calibrated_ood": {
            "calibrated_tau_youden": tau_youden,
            "calibrated_tau_f2": tau_f2,
            "baseline_tau_050": ood_base,
            "youden_calibrated": ood_youden,
            "f2_recall_calibrated": ood_f2,
            "smoothed_and_youden": ood_smoothed,
        },
        "zero_shot_retention_rate_pct": retention_rate,
        "generalization_penalty_delta_f1": float(ood_youden["macro_f1"] - f1_in),
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

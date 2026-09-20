"""
Detection & Attack Attribution Benchmark Diagnostics for Cyber-JEPA (Phase 6B).

Provides:
1. Operational Attack Accounting:
   - Direct counts: Total Attacks, Attacks Detected (TP), Attacks Missed (FN),
     Clean Steps (TN), False Alarms (FP).
   - Percentages: Detection Rate %, False Alarm Rate %, Precision %.
2. Adversary Tactic / Policy Breakdown:
   - Separate attack detection rates for B-line (targeted) vs. Meander (exploratory).
3. Incident-Level Early Detection:
   - Incident coverage % and mean precursor lead steps.
4. Academic Grounded Abstractions:
   - Macro F1 with 95% Bootstrap CI (B=1,000).
   - Balanced Accuracy, AUROC, PR-AUC (Average Precision).
   - FPR@95% Recall (Arp et al., USENIX 2022 / Sommer & Paxson, S&P 2010).
5. Anti-Shortcut / "Clever Hans" Audits (Geirhos et al., 2020):
   - Empirical Majority Class Baseline.
   - Representation Gain: JEPA Latents vs. Raw Uncompressed Telemetry (ΔF1).
   - Dynamic-only Transition Isolation (RMS > 1e-4).
   - Non-Parametric k-NN Probe (k=5) vs. Linear Probe.
"""

from typing import Any
import numpy as np
import torch
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.neighbors import KNeighborsClassifier

from cyber_jepa.evaluation.probes import compute_bootstrap_ci


def compute_fpr_at_recall(labels: np.ndarray, probs: np.ndarray, target_recall: float = 0.95) -> float:
    """Compute False Positive Rate at target recall (e.g. 95%) adhering to Arp et al. (2022)."""
    if len(np.unique(labels)) < 2:
        return 0.0
    fpr, tpr, _ = roc_curve(labels, probs)
    idx = np.where(tpr >= target_recall)[0]
    return float(fpr[idx[0]]) if len(idx) > 0 else 1.0


def evaluate_detection_metrics(
    labels: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Compute full operational attack accounting and academic performance metrics.

    Args:
        labels: Ground truth binary labels [N] (1 = Attack / Compromised, 0 = Clean)
        preds: Predicted binary labels [N]
        probs: Predicted attack probabilities [N]
        seed: Random seed for bootstrap confidence intervals

    Returns:
        Dictionary with operational counts, rates, and academic metrics.
    """
    total_samples = len(labels)
    if total_samples == 0:
        return {}

    # 1. Operational Counts & Confusion Matrix
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    total_attacks = tp + fn
    clean_steps = tn + fp

    attack_detection_rate = float((tp / max(1, total_attacks)) * 100.0)
    false_alarm_rate = float((fp / max(1, clean_steps)) * 100.0)
    precision_pct = float((tp / max(1, tp + fp)) * 100.0)

    # 2. Academic Performance Metrics
    f1_pt, f1_low, f1_high = compute_bootstrap_ci(
        labels, preds, lambda y, p: f1_score(y, p, average="macro", zero_division=0), seed=seed
    )

    bal_acc = float(balanced_accuracy_score(labels, preds))

    has_both_classes = len(np.unique(labels)) > 1
    auroc = float(roc_auc_score(labels, probs)) if has_both_classes else 0.5
    pr_auc = float(average_precision_score(labels, probs)) if has_both_classes else 0.0
    fpr_95 = compute_fpr_at_recall(labels, probs, target_recall=0.95)

    return {
        # Operational Counts ("At Its Simplest")
        "total_samples": total_samples,
        "total_attacks": total_attacks,
        "attacks_detected": tp,
        "attacks_missed": fn,
        "clean_steps": clean_steps,
        "clean_correct": tn,
        "false_alarms": fp,
        # Operational Rates
        "attack_detection_rate_pct": attack_detection_rate,
        "false_alarm_rate_pct": false_alarm_rate,
        "precision_pct": precision_pct,
        # Academic Abstractions
        "macro_f1": float(f1_pt),
        "macro_f1_ci_95": [float(f1_low), float(f1_high)],
        "balanced_accuracy": bal_acc,
        "auroc": auroc,
        "pr_auc": pr_auc,
        "fpr_at_95_recall": fpr_95,
    }


def evaluate_adversary_policy_breakdown(
    labels: np.ndarray,
    preds: np.ndarray,
    trajectory_ids: list[str],
) -> dict[str, Any]:
    """
    Break down attack detection rates by Red adversary policy (B-line vs. Meander).

    Args:
        labels: Ground truth binary labels [N]
        preds: Predicted binary labels [N]
        trajectory_ids: List of trajectory identifiers [N]

    Returns:
        Dictionary with per-policy attack detection statistics.
    """
    bline_mask = np.array(["bline" in str(tid) for tid in trajectory_ids], dtype=bool)
    meander_mask = np.array(["meander" in str(tid) for tid in trajectory_ids], dtype=bool)

    def _stats_for_subset(mask: np.ndarray) -> dict[str, Any]:
        if np.sum(mask) == 0:
            return {
                "total_attacks": 0,
                "detected": 0,
                "missed": 0,
                "clean_steps": 0,
                "false_alarms": 0,
                "detection_rate_pct": 0.0,
                "false_alarm_rate_pct": 0.0,
            }
        sub_y = labels[mask]
        sub_p = preds[mask]
        cm = confusion_matrix(sub_y, sub_p, labels=[0, 1])
        tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
        tot_att = tp + fn
        tot_clean = tn + fp
        det_rate = float((tp / max(1, tot_att)) * 100.0) if tot_att > 0 else 100.0
        fa_rate = float((fp / max(1, tot_clean)) * 100.0) if tot_clean > 0 else 0.0
        return {
            "total_attacks": tot_att,
            "detected": tp,
            "missed": fn,
            "clean_steps": tot_clean,
            "false_alarms": fp,
            "detection_rate_pct": det_rate,
            "false_alarm_rate_pct": fa_rate,
        }

    return {
        "bline_targeted": _stats_for_subset(bline_mask),
        "meander_stealth": _stats_for_subset(meander_mask),
    }


def evaluate_incident_early_detection(
    labels: np.ndarray,
    preds: np.ndarray,
    trajectory_ids: list[str],
    t_steps: list[int] | None = None,
) -> dict[str, Any]:
    """
    Evaluate incident-level detection: Does the system alert before or at compromise?

    Args:
        labels: Ground truth binary labels [N]
        preds: Predicted binary labels [N]
        trajectory_ids: Trajectory ID for each sample [N]
        t_steps: Optional step index within the trajectory [N]

    Returns:
        Incident coverage %, timely vs. late breakdown, clean trajectory false alarm rate,
        and mean lead steps before impact.
    """
    unique_trajs = sorted(list(set(trajectory_ids)))
    total_incidents = 0
    detected_incidents = 0
    timely_incidents = 0
    late_incidents = 0
    clean_trajectories = 0
    false_alarm_trajectories = 0
    lead_steps_list = []

    for traj in unique_trajs:
        idx = [i for i, tid in enumerate(trajectory_ids) if tid == traj]
        traj_labels = labels[idx]
        traj_preds = preds[idx]
        traj_steps = [t_steps[i] for i in idx] if t_steps is not None else list(range(len(idx)))

        # An attack incident occurs if any step in the trajectory has label == 1
        if np.any(traj_labels == 1):
            total_incidents += 1
            # First ground-truth attack step
            first_attack_idx = int(np.where(traj_labels == 1)[0][0])
            first_attack_step = traj_steps[first_attack_idx]

            # First alert raised by model
            alert_indices = np.where(traj_preds == 1)[0]
            if len(alert_indices) > 0:
                first_alert_step = traj_steps[int(alert_indices[0])]
                detected_incidents += 1
                lead = first_attack_step - first_alert_step
                lead_steps_list.append(lead)
                if lead >= 0:
                    timely_incidents += 1
                else:
                    late_incidents += 1
        else:
            clean_trajectories += 1
            if np.any(traj_preds == 1):
                false_alarm_trajectories += 1

    missed_incidents = total_incidents - detected_incidents
    coverage_pct = float((detected_incidents / max(1, total_incidents)) * 100.0)
    timely_pct = float((timely_incidents / max(1, total_incidents)) * 100.0)
    traj_far_pct = float((false_alarm_trajectories / max(1, clean_trajectories)) * 100.0) if clean_trajectories > 0 else 0.0
    mean_lead = float(np.mean(lead_steps_list)) if lead_steps_list else 0.0

    return {
        "total_attack_incidents": total_incidents,
        "incidents_detected": detected_incidents,
        "timely_detected_incidents": timely_incidents,
        "late_detected_incidents": late_incidents,
        "missed_incidents": missed_incidents,
        "incident_coverage_pct": coverage_pct,
        "timely_coverage_pct": timely_pct,
        "clean_trajectories": clean_trajectories,
        "false_alarm_trajectories": false_alarm_trajectories,
        "trajectory_false_alarm_rate_pct": traj_far_pct,
        "mean_lead_steps": mean_lead,
    }


def run_clever_hans_audit(
    train_latents: np.ndarray,
    train_labels: np.ndarray,
    test_latents: np.ndarray,
    test_labels: np.ndarray,
    raw_train_features: np.ndarray,
    raw_test_features: np.ndarray,
    test_rms_deltas: np.ndarray,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Execute Anti-Shortcut / Clever Hans Audits to prove model is not exploiting trivial artifacts:
    1. Majority Class Baseline: Verifies model outperforms trivial empirical prior guessing.
    2. Representation Gain (ΔF1): Compares probe on raw uncompressed features vs. JEPA latents.
    3. Dynamic vs. Static Transition Breakdown: Tests whether accuracy holds during non-zero shifts.
    4. Non-Parametric k-NN Probe (k=5): Verifies intrinsic geometric clustering.
    """
    # 1. Majority Baseline
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(train_latents, train_labels)
    dummy_preds = dummy.predict(test_latents)
    dummy_f1 = float(f1_score(test_labels, dummy_preds, average="macro", zero_division=0))

    # 2. Raw Telemetry Feature Probe
    raw_clf = LogisticRegression(C=1.0, max_iter=500, random_state=seed)
    raw_clf.fit(raw_train_features, train_labels)
    raw_preds = raw_clf.predict(raw_test_features)
    raw_f1 = float(f1_score(test_labels, raw_preds, average="macro", zero_division=0))

    # JEPA Latent Probe
    jepa_clf = LogisticRegression(C=1.0, max_iter=500, random_state=seed)
    jepa_clf.fit(train_latents, train_labels)
    jepa_preds = jepa_clf.predict(test_latents)
    jepa_f1 = float(f1_score(test_labels, jepa_preds, average="macro", zero_division=0))

    repr_gain = float(jepa_f1 - raw_f1)

    # 3. Dynamic vs. Static Breakdown (RMS > 1e-4)
    dynamic_mask = (test_rms_deltas > 1e-4)
    if np.sum(dynamic_mask) > 0 and len(np.unique(test_labels[dynamic_mask])) > 1:
        dyn_f1 = float(f1_score(test_labels[dynamic_mask], jepa_preds[dynamic_mask], average="macro", zero_division=0))
    else:
        dyn_f1 = jepa_f1

    if np.sum(~dynamic_mask) > 0 and len(np.unique(test_labels[~dynamic_mask])) > 1:
        stat_f1 = float(f1_score(test_labels[~dynamic_mask], jepa_preds[~dynamic_mask], average="macro", zero_division=0))
    else:
        stat_f1 = jepa_f1

    # 4. Non-Parametric k-NN Probe (k=5)
    # Downsample train latents if large to avoid slow evaluation
    max_knn_train = 5000
    if len(train_latents) > max_knn_train:
        np.random.seed(seed)
        sub_idx = np.random.choice(len(train_latents), size=max_knn_train, replace=False)
        knn_train_x = train_latents[sub_idx]
        knn_train_y = train_labels[sub_idx]
    else:
        knn_train_x = train_latents
        knn_train_y = train_labels

    knn = KNeighborsClassifier(n_neighbors=5, metric="cosine")
    knn.fit(knn_train_x, knn_train_y)
    knn_preds = knn.predict(test_latents)
    knn_f1 = float(f1_score(test_labels, knn_preds, average="macro", zero_division=0))
    knn_bal_acc = float(balanced_accuracy_score(test_labels, knn_preds))

    return {
        "majority_baseline_macro_f1": dummy_f1,
        "raw_features_macro_f1": raw_f1,
        "jepa_latents_macro_f1": jepa_f1,
        "representation_gain_f1": repr_gain,
        "dynamic_transitions_f1": dyn_f1,
        "static_transitions_f1": stat_f1,
        "knn_nonparametric_macro_f1": knn_f1,
        "knn_nonparametric_bal_acc": knn_bal_acc,
        "passed_anti_shortcut_audit": bool(repr_gain >= -0.02 and jepa_f1 > dummy_f1 + 0.10 and dyn_f1 > 0.60),
    }

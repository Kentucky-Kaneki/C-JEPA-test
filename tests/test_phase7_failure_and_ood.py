"""
Unit tests for Phase 7: Failure Mode Forensics, Detection Optimization, and Emergent OOD Generalization.
"""

import pytest
import numpy as np

from cyber_jepa.evaluation.failure_analysis import run_comprehensive_failure_analysis
from cyber_jepa.evaluation.detection_optimization import (
    calibrate_decision_threshold,
    apply_temporal_smoothing,
    evaluate_multi_target_probing,
    evaluate_optimized_detection,
)
from cyber_jepa.evaluation.emergent_ood import (
    evaluate_cross_policy_transfer,
    evaluate_jepa_energy_anomaly,
    evaluate_mitre_tier_summary,
    evaluate_operational_zero_day_detection,
    evaluate_zero_label_latent_anomaly,
)


def test_perimeter_vs_crown_jewel_failure_accounting():
    """Verify failure analysis correctly decouples early perimeter detections from true false alarms."""
    # 10 samples:
    # 0..3: Crown jewel attacks (y_true=1)
    # 4..6: Crown jewel clean (y_true=0), BUT hosts 0/1 are compromised (Perimeter breach)
    # 7..9: Completely benign (y_true=0), 0 hosts compromised
    y_true = np.array([1, 1, 1, 1,   0, 0, 0,   0, 0, 0])

    # Predictions:
    # 0..2 caught, 3 missed
    # 4, 5 flagged as attack (Perimeter breach detections!)
    # 6 ignored
    # 7 flagged as attack (TRUE False Alarm)
    # 8, 9 ignored
    y_pred = np.array([1, 1, 1, 0,   1, 1, 0,   1, 0, 0])
    y_prob = np.array([0.9, 0.8, 0.7, 0.3,   0.65, 0.60, 0.20,   0.55, 0.10, 0.05])

    # Host compromised matrix [10, 4 hosts]
    host_comp = np.zeros((10, 4), dtype=float)
    host_comp[0:4, 3] = 1.0   # Op_Server0 compromised
    host_comp[4, 0] = 1.0     # Enterprise0 compromised
    host_comp[5, 1] = 1.0     # Enterprise1 compromised
    host_comp[6, 0] = 1.0     # Enterprise0 compromised
    # 7, 8, 9 have zero hosts compromised

    trajs = ["ep_bline_1"] * 5 + ["ep_meander_1"] * 5

    res = run_comprehensive_failure_analysis(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        host_compromised=host_comp,
        trajectory_ids=trajs,
    )

    audit = res["perimeter_vs_crown_jewel_audit"]
    # Nominal clean steps (where Op_Server0 is clean) = 6
    assert audit["nominal_clean_steps"] == 6
    # Nominal false alarms (model flagged alert while Op_Server0 clean) = 3 (samples 4, 5, 7)
    assert audit["nominal_false_alarms"] == 3
    # True benign steps (where zero hosts compromised) = 3 (samples 7, 8, 9)
    assert audit["true_benign_steps_zero_hosts_comp"] == 3
    # Early perimeter detections: samples 4, 5
    assert audit["early_perimeter_detections"] == 2
    # True false alarms: sample 7
    assert audit["true_false_alarms_zero_hosts_comp"] == 1
    assert np.isclose(audit["true_false_alarm_rate_pct"], (1 / 3) * 100.0)

    miss = res["borderline_miss_stratification"]
    # 1 missed attack (sample 3, prob 0.30)
    assert miss["total_missed_attacks"] == 1
    assert miss["prob_bin_30_to_40"]["count"] == 1


def test_blue_defense_action_audit():
    """Verify Blue action interference audit correctly calculates disruption FAR ratio."""
    y_true = np.array([0, 0, 0, 0, 1, 1])
    y_pred = np.array([1, 1, 0, 0, 1, 1])  # 2 FPs on clean steps (indices 0, 1)
    y_prob = np.array([0.8, 0.7, 0.1, 0.2, 0.9, 0.95])
    host_comp = np.zeros((6, 2))
    trajs = ["t1"] * 6
    actions = ["Restore", "Remove", "Sleep", "Monitor", "Sleep", "Restore"]

    res = run_comprehensive_failure_analysis(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        host_compromised=host_comp,
        trajectory_ids=trajs,
        action_types=actions,
    )

    blue_audit = res["blue_defense_action_audit"]
    assert "disruptive_actions_restore_remove" in blue_audit
    assert "passive_actions_other" in blue_audit
    assert blue_audit["disruptive_actions_restore_remove"]["false_alarm_rate_pct"] == 100.0
    assert blue_audit["passive_actions_other"]["false_alarm_rate_pct"] == 0.0
    assert blue_audit["disruption_far_ratio"] > 1.0


def test_calibrate_decision_threshold():
    """Verify threshold calibration algorithms compute optimal thresholds across all criteria."""
    np.random.seed(42)
    # Synthetic validation set with clean and attack distributions
    y_val = np.array([0] * 50 + [1] * 50)
    # Clean probs centered around 0.20, attack probs centered around 0.70
    p_val = np.clip(np.concatenate([np.random.normal(0.20, 0.10, 50), np.random.normal(0.70, 0.10, 50)]), 0.01, 0.99)

    # Youden's J
    tau_youden = calibrate_decision_threshold(y_val, p_val, criterion="youden")
    assert 0.30 <= tau_youden <= 0.60

    # F2 (recall-prioritized) should select lower threshold
    tau_f2 = calibrate_decision_threshold(y_val, p_val, criterion="f2")
    assert 0.10 <= tau_f2 <= 0.50

    # F0.5 (precision-prioritized)
    tau_f05 = calibrate_decision_threshold(y_val, p_val, criterion="f05")
    assert 0.20 <= tau_f05 <= 0.90

    # FPR target 5%
    tau_fpr05 = calibrate_decision_threshold(y_val, p_val, criterion="fpr_target_05")
    assert 0.20 <= tau_fpr05 <= 0.90

    # Recall target 95%
    tau_rec95 = calibrate_decision_threshold(y_val, p_val, criterion="recall_target_95")
    assert 0.10 <= tau_rec95 <= 0.65


def test_apply_temporal_smoothing():
    """Verify causal temporal exponential smoothing within trajectory boundaries and edge cases."""
    # Two trajectories of 3 steps each
    probs = np.array([0.2, 0.8, 0.2,   0.1, 0.9, 0.9])
    trajs = ["t1", "t1", "t1",   "t2", "t2", "t2"]

    smoothed = apply_temporal_smoothing(probs, trajs, alpha=0.5)

    # Trajectory 1:
    # step 0: 0.2
    # step 1: 0.5 * 0.8 + 0.5 * 0.2 = 0.5
    # step 2: 0.5 * 0.2 + 0.5 * 0.5 = 0.35
    assert np.isclose(smoothed[0], 0.2)
    assert np.isclose(smoothed[1], 0.5)
    assert np.isclose(smoothed[2], 0.35)

    # Trajectory 2 starts fresh at step 3 (zero cross-trajectory contamination!):
    # step 3: 0.1
    # step 4: 0.5 * 0.9 + 0.5 * 0.1 = 0.5
    # step 5: 0.5 * 0.9 + 0.5 * 0.5 = 0.7
    assert np.isclose(smoothed[3], 0.1)
    assert np.isclose(smoothed[4], 0.5)
    assert np.isclose(smoothed[5], 0.7)

    # Edge case 1: Single step trajectory
    single_p = np.array([0.42])
    single_t = ["t_single"]
    res_single = apply_temporal_smoothing(single_p, single_t, alpha=0.5)
    assert np.isclose(res_single[0], 0.42)

    # Edge case 2: alpha = 1.0 (no smoothing, identity)
    res_identity = apply_temporal_smoothing(probs, trajs, alpha=1.0)
    assert np.allclose(res_identity, probs)


def test_evaluate_multi_target_probing():
    """Verify dual-target probing evaluates both crown jewel and perimeter intrusion targets."""
    np.random.seed(42)
    N_tr, N_te, D = 100, 50, 64
    train_z = np.random.randn(N_tr, D)
    test_z = np.random.randn(N_te, D)

    train_labels = (train_z[:, 0] > 0).astype(int)
    test_labels = (test_z[:, 0] > 0).astype(int)

    train_host_comp = np.zeros((N_tr, 4))
    test_host_comp = np.zeros((N_te, 4))
    train_host_comp[train_z[:, 1] > 0, 0] = 1.0
    test_host_comp[test_z[:, 1] > 0, 0] = 1.0

    res = evaluate_multi_target_probing(
        train_latents=train_z,
        train_labels=train_labels,
        train_host_comp=train_host_comp,
        test_latents=test_z,
        test_labels=test_labels,
        test_host_comp=test_host_comp,
        seed=42,
    )

    assert "crown_jewel_probe" in res
    assert "perimeter_intrusion_probe" in res
    assert "detection_rate_pct" in res["crown_jewel_probe"]
    assert "detection_rate_pct" in res["perimeter_intrusion_probe"]
    assert "macro_f1" in res["crown_jewel_probe"]
    assert "auroc" in res["perimeter_intrusion_probe"]


def test_evaluate_zero_label_latent_anomaly():
    """Verify emergent zero-label anomaly detection using latent centroid distance."""
    np.random.seed(42)
    N_clean, N_test, D = 100, 200, 64

    # Known clean reference baseline centered at non-zero cluster (mean 1.0)
    clean_ref = np.random.normal(1.0, 0.2, (N_clean, D))

    # Test set: 100 clean, 100 attack (shifted by +2.0 along first 10 dims)
    test_clean = np.random.normal(1.0, 0.2, (100, D))
    test_attack = np.random.normal(1.0, 0.2, (100, D))
    test_attack[:, :10] += 2.0

    test_z = np.vstack([test_clean, test_attack])
    test_y = np.array([0] * 100 + [1] * 100)

    res = evaluate_zero_label_latent_anomaly(clean_ref, test_z, test_y)

    assert "cosine_distance" in res
    assert res["cosine_distance"]["auroc"] > 0.85
    assert res["cosine_distance"]["pr_auc"] > 0.80
    assert res["emergent_detection_viable"] is True


def test_evaluate_cross_policy_transfer():
    """Verify zero-shot cross-policy transfer and retention rate calculation."""
    np.random.seed(42)
    N_train, N_test, D = 200, 100, 64

    # Source policy training data
    train_z = np.random.randn(N_train, D)
    train_y = (train_z[:, 0] > 0).astype(int)

    # In-distribution test data
    test_z_in = np.random.randn(N_test, D)
    test_y_in = (test_z_in[:, 0] > 0).astype(int)

    # Unseen target policy test data (with slight domain rotation/noise)
    test_z_ood = np.random.randn(N_test, D)
    test_z_ood[:, 0] += 0.5 * test_z_ood[:, 1]
    test_y_ood = (test_z_ood[:, 0] > 0).astype(int)

    transfer_res = evaluate_cross_policy_transfer(
        train_latents=train_z,
        train_labels=train_y,
        test_latents_indist=test_z_in,
        test_labels_indist=test_y_in,
        test_latents_ood=test_z_ood,
        test_labels_ood=test_y_ood,
        source_policy_name="bline",
        target_policy_name="meander",
        seed=42,
    )

    assert "zero_shot_retention_rate_pct" in transfer_res
    assert transfer_res["in_distribution"]["macro_f1"] > 0.70
    assert transfer_res["zero_shot_ood"]["macro_f1"] > 0.60
    assert transfer_res["zero_shot_retention_rate_pct"] > 50.0


def test_evaluate_mitre_tier_summary():
    """Verify MITRE tier summarization correctly groups hosts into tactical tiers."""
    np.random.seed(42)
    from cyber_jepa.data.dataset import MONITORED_HOSTS
    N_tr, N_te, D = 100, 50, 64

    train_z = np.random.randn(N_tr, D)
    test_z = np.random.randn(N_te, D)

    num_hosts = len(MONITORED_HOSTS)
    train_host = np.random.randint(0, 2, (N_tr, num_hosts)).astype(float)
    test_host = np.random.randint(0, 2, (N_te, num_hosts)).astype(float)

    res = evaluate_mitre_tier_summary(
        train_latents=train_z,
        train_host_comp=train_host,
        test_latents=test_z,
        test_host_comp=test_host,
        host_names=MONITORED_HOSTS,
        seed=42,
    )

    assert "tier_summary" in res
    assert "user_tier" in res["tier_summary"]
    assert "enterprise_tier" in res["tier_summary"]
    assert "operational_tier" in res["tier_summary"]
    assert "mean_macro_f1" in res["tier_summary"]["user_tier"]
    assert "mean_auroc" in res["tier_summary"]["enterprise_tier"]
    assert 0.0 <= res["tier_summary"]["operational_tier"]["mean_auroc"] <= 1.0


def test_evaluate_operational_zero_day_detection():
    """Verify operational zero-day detection computes raw counts and benign false alarm audit."""
    np.random.seed(42)
    N_clean, N_test, D = 100, 200, 64
    clean_ref = np.random.normal(1.0, 0.2, (N_clean, D))

    test_clean = np.random.normal(1.0, 0.2, (100, D))
    test_attack = np.random.normal(1.0, 0.2, (100, D))
    test_attack[:, :10] += 2.0
    test_z = np.vstack([test_clean, test_attack])
    test_y = np.array([0] * 100 + [1] * 100)

    # Perimeter indicators: 50 of the clean samples have a perimeter host compromised
    host_comp = np.zeros((200, 4))
    host_comp[50:100, 0] = 1.0   # 50 early perimeter intrusions while crown jewel clean
    host_comp[100:200, 3] = 1.0  # All attacks have crown jewel compromised

    res = evaluate_operational_zero_day_detection(
        clean_reference_latents=clean_ref,
        test_latents=test_z,
        test_labels=test_y,
        test_host_comp=host_comp,
        quantiles=[0.90, 0.95],
    )

    assert "auroc" in res
    assert res["auroc"] > 0.85
    assert "operating_points" in res
    assert "q_90" in res["operating_points"]
    assert "q_95" in res["operating_points"]

    q90 = res["operating_points"]["q_90"]
    assert "crown_jewel_attacks_caught" in q90
    assert "crown_jewel_detection_rate_pct" in q90
    assert "perimeter_detection_rate_pct" in q90
    assert "true_benign_steps" in q90
    assert q90["true_benign_steps"] == 50  # 100 clean - 50 perimeter = 50 true benign


def test_evaluate_jepa_energy_anomaly():
    """Verify JEPA prediction energy anomaly detection computes operational points."""
    np.random.seed(42)
    N_clean, N_test = 100, 200

    # Low prediction error energy on clean transitions (0.05 +- 0.02)
    clean_energy = np.random.normal(0.05, 0.02, N_clean)

    # Test set: 100 low energy clean, 100 high energy attack (0.35 +- 0.05)
    test_clean_energy = np.random.normal(0.05, 0.02, 100)
    test_attack_energy = np.random.normal(0.35, 0.05, 100)
    test_energy = np.concatenate([test_clean_energy, test_attack_energy])
    test_y = np.array([0] * 100 + [1] * 100)

    res = evaluate_jepa_energy_anomaly(
        clean_energies=clean_energy,
        test_energies=test_energy,
        test_labels=test_y,
        quantiles=[0.90, 0.95],
    )

    assert "auroc" in res
    assert res["auroc"] > 0.95
    assert "operating_points" in res
    assert "q_90" in res["operating_points"]
    assert res["operating_points"]["q_90"]["detection_rate_pct"] > 90.0


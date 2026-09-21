"""
Unit tests for Phase 7: Failure Mode Forensics, Detection Optimization, and Emergent OOD Generalization.
"""

import pytest
import numpy as np

from cyber_jepa.evaluation.failure_analysis import run_comprehensive_failure_analysis
from cyber_jepa.evaluation.detection_optimization import (
    calibrate_decision_threshold,
    apply_temporal_smoothing,
    evaluate_optimized_detection,
)
from cyber_jepa.evaluation.emergent_ood import (
    evaluate_zero_label_latent_anomaly,
    evaluate_cross_policy_transfer,
    evaluate_mitre_tier_summary,
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


def test_calibrate_decision_threshold():
    """Verify threshold calibration algorithms compute optimal thresholds."""
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

    # Recall target 95%
    tau_rec95 = calibrate_decision_threshold(y_val, p_val, criterion="recall_target_95")
    assert 0.10 <= tau_rec95 <= 0.65


def test_apply_temporal_smoothing():
    """Verify causal temporal exponential smoothing within trajectory boundaries."""
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

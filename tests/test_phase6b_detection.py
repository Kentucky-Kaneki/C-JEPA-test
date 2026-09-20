"""
Unit tests for Phase 6B: Scaled Red Agent Detection Diagnostics & Anti-Shortcut Audits.
"""

import pytest
import numpy as np
import torch

from cyber_jepa.evaluation.detection_benchmarks import (
    compute_fpr_at_recall,
    evaluate_adversary_policy_breakdown,
    evaluate_detection_metrics,
    evaluate_incident_early_detection,
    run_clever_hans_audit,
)
from cyber_jepa.evaluation.scale_benchmarks import (
    SCALE_SPECS,
    ScaledDatasetWrapper,
)


def test_scaled_dataset_wrapper_8_scales():
    """Verify ScaledDatasetWrapper correctly slices or replicates across all 8 scales (5 to 500 hosts)."""
    class DummyDataset:
        def __len__(self):
            return 10

        def __getitem__(self, idx):
            return {
                "history_flat": torch.randn(4, 52),
                "target_flat": torch.randn(52),
                "label": 1 if idx % 2 == 0 else 0,
                "host_compromised": torch.zeros(11),
                "action_seq": torch.zeros(4, dtype=torch.long),
                "trajectory_id": f"traj_bline_1001_{idx:03d}",
                "transition_id": f"trans_{idx}",
                "rms_delta": 0.35,
            }

    base_ds = DummyDataset()
    test_scales = ["5", "10", "13", "25", "50", "100", "250", "500"]

    for scale_key in test_scales:
        spec = SCALE_SPECS[scale_key]
        num_hosts = spec["num_hosts"]
        exp_dim = spec["obs_dim"]

        wrapper = ScaledDatasetWrapper(base_ds, scale=scale_key)
        assert len(wrapper) == 10
        item = wrapper[0]

        assert item["history_flat"].shape == (4, exp_dim), f"Scale {scale_key} hist shape mismatch"
        assert item["target_flat"].shape == (exp_dim,), f"Scale {scale_key} target shape mismatch"
        assert item["host_compromised"].shape == (num_hosts,), f"Scale {scale_key} host_comp mismatch"
        assert item["trajectory_id"] == "traj_bline_1001_000"


def test_operational_detection_metrics_accounting():
    """Verify exact count tracking: Total Attacks, Detected (TP), Missed (FN), False Alarms (FP)."""
    # 100 samples: 30 attacks, 70 clean
    labels = np.array([1] * 30 + [0] * 70)
    # Predictions: 27 of 30 attacks caught (3 missed); 7 false alarms
    preds = np.array([1] * 27 + [0] * 3 + [1] * 7 + [0] * 63)
    probs = np.clip(preds * 0.8 + 0.1, 0.0, 1.0)

    res = evaluate_detection_metrics(labels, preds, probs, seed=42)

    assert res["total_samples"] == 100
    assert res["total_attacks"] == 30
    assert res["attacks_detected"] == 27
    assert res["attacks_missed"] == 3
    assert res["clean_steps"] == 70
    assert res["clean_correct"] == 63
    assert res["false_alarms"] == 7

    assert np.isclose(res["attack_detection_rate_pct"], 90.0)
    assert np.isclose(res["false_alarm_rate_pct"], 10.0)
    assert np.isclose(res["precision_pct"], (27 / 34) * 100.0)
    assert res["macro_f1"] > 0.80
    assert res["auroc"] > 0.85
    assert res["pr_auc"] > 0.60


def test_adversary_policy_breakdown():
    """Verify B-line vs Meander detection rate breakdown."""
    # 4 samples: 2 bline, 2 meander
    labels = np.array([1, 0, 1, 0])
    preds = np.array([1, 0, 0, 0]) # Caught bline attack, missed meander attack
    trajs = ["traj_bline_001", "traj_bline_002", "traj_meander_001", "traj_meander_002"]

    breakdown = evaluate_adversary_policy_breakdown(labels, preds, trajs)

    assert breakdown["bline_targeted"]["total_attacks"] == 1
    assert breakdown["bline_targeted"]["detected"] == 1
    assert breakdown["bline_targeted"]["detection_rate_pct"] == 100.0

    assert breakdown["meander_stealth"]["total_attacks"] == 1
    assert breakdown["meander_stealth"]["detected"] == 0
    assert breakdown["meander_stealth"]["detection_rate_pct"] == 0.0


def test_incident_early_detection():
    """Verify incident coverage, timely vs late alerts, and clean trajectory false alarm rate."""
    # ep1: attack incident, timely detected (lead=1)
    # ep2: attack incident, missed (no alert)
    # ep3: clean trajectory, no false alarm
    # ep4: clean trajectory, false alarm
    labels = np.array([0, 0, 1, 1,   0, 1,   0, 0,   0, 0])
    preds =  np.array([0, 1, 1, 1,   0, 0,   0, 0,   1, 0])
    trajs = ["ep1", "ep1", "ep1", "ep1", "ep2", "ep2", "ep3", "ep3", "ep4", "ep4"]
    t_steps = [10, 11, 12, 13,   20, 21,   30, 31,   40, 41]

    res = evaluate_incident_early_detection(labels, preds, trajs, t_steps)

    assert res["total_attack_incidents"] == 2
    assert res["incidents_detected"] == 1
    assert res["timely_detected_incidents"] == 1
    assert res["late_detected_incidents"] == 0
    assert res["missed_incidents"] == 1
    assert res["incident_coverage_pct"] == 50.0
    assert res["timely_coverage_pct"] == 50.0
    assert res["clean_trajectories"] == 2
    assert res["false_alarm_trajectories"] == 1
    assert res["trajectory_false_alarm_rate_pct"] == 50.0
    assert res["mean_lead_steps"] == 1.0


def test_clever_hans_audit():
    """Verify anti-shortcut audits: majority baseline, representation gain, dynamic F1, k-NN."""
    np.random.seed(42)
    N_train, N_test, D = 100, 50, 64
    train_latents = np.random.randn(N_train, D)
    test_latents = np.random.randn(N_test, D)

    # Correlate label with first dimension
    train_labels = (train_latents[:, 0] > 0).astype(int)
    test_labels = (test_latents[:, 0] > 0).astype(int)

    raw_train = np.random.randn(N_train, 128)
    raw_test = np.random.randn(N_test, 128)
    rms_deltas = np.array([0.25 if i % 2 == 0 else 0.0 for i in range(N_test)])

    audit = run_clever_hans_audit(
        train_latents, train_labels, test_latents, test_labels,
        raw_train, raw_test, rms_deltas, seed=42
    )

    assert "majority_baseline_macro_f1" in audit
    assert "representation_gain_f1" in audit
    assert "dynamic_transitions_f1" in audit
    assert "knn_nonparametric_macro_f1" in audit
    assert audit["jepa_latents_macro_f1"] > audit["majority_baseline_macro_f1"]

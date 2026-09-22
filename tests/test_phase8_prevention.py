"""
Unit tests for Phase 8: Operational Prevention, Lead-Time Forensics, and Closed-Loop Containment.
"""

import numpy as np
import pytest

from cyber_jepa.evaluation.prevention_analysis import (
    evaluate_prevention_lead_time,
    evaluate_closed_loop_prevention,
    evaluate_multi_quantile_prevention,
)


def test_evaluate_prevention_lead_time_synthetic():
    """Verify lead-time calculation properly measures Delta t = t_breach - t_alert across episodes."""
    # 4 trajectories:
    # ep_early: Alert at t=3, CJ breach at t=7 (Lead = 4 steps!)
    # ep_late:  Alert at t=8, CJ breach at t=5 (Alert too late, Lead = 0)
    # ep_miss:  No alert,     CJ breach at t=6 (Missed, Lead = 0)
    # ep_clean: No alert,     No breach (Pristine baseline)

    trajs = (
        ["ep_early"] * 5
        + ["ep_late"] * 5
        + ["ep_miss"] * 5
        + ["ep_clean"] * 5
    )
    t_contexts = np.array(
        [1, 3, 5, 7, 9]
        + [1, 3, 5, 7, 9]
        + [1, 3, 5, 7, 9]
        + [1, 3, 5, 7, 9]
    )

    # Labels for Op_Server0 breach:
    # ep_early: breaches at t=7 (index 3)
    # ep_late:  breaches at t=5 (index 7)
    # ep_miss:  breaches at t=7 (index 13)
    # ep_clean: all 0
    labels = np.zeros(20, dtype=int)
    labels[3] = 1
    labels[4] = 1
    labels[7] = 1
    labels[8] = 1
    labels[9] = 1
    labels[13] = 1
    labels[14] = 1

    # Anomaly scores (threshold = 0.50):
    # ep_early: triggers alert at t=3 (score 0.8)
    # ep_late:  triggers alert at t=7 (score 0.8, but breach was at t=5!)
    # ep_miss:  never reaches 0.50
    # ep_clean: never reaches 0.50
    scores = np.zeros(20, dtype=float)
    scores[1] = 0.8  # ep_early t=3
    scores[2] = 0.9
    scores[8] = 0.8  # ep_late t=7
    scores[9] = 0.9

    res = evaluate_prevention_lead_time(
        test_labels=labels,
        anomaly_scores=scores,
        trajectory_ids=trajs,
        t_contexts=t_contexts,
        threshold=0.50,
    )

    assert res["total_attack_episodes"] == 3
    assert res["episodes_alerted_before_cj"] == 1
    assert np.isclose(res["early_warning_rate_pct"], (1 / 3) * 100.0)
    assert res["mean_lead_time_steps"] == 4.0  # t_breach (7) - t_alert (3) = 4
    assert res["median_lead_time_steps"] == 4.0
    assert res["episodes_with_lead_ge_3_steps"] == 1
    assert np.isclose(res["lead_ge_3_steps_rate_pct"], (1 / 3) * 100.0)
    assert res["episodes_with_lead_ge_5_steps"] == 0


def test_evaluate_closed_loop_prevention():
    """Verify closed-loop containment accounting (preservation rate & false intervention rate)."""
    # 5 trajectories:
    # ep_prevented: Alert at t=3, CJ breach at t=7 (Lateral movement severed -> PREVENTED)
    # ep_late:      Alert at t=7, CJ breach at t=5 (Alert late -> UNCONTAINED)
    # ep_clean_ok:  No alerts, no breaches (Clean baseline preserved)
    # ep_clean_fp:  Alert at t=3, no breaches (False intervention on pristine network)

    trajs = (
        ["ep_prev"] * 3
        + ["ep_late"] * 3
        + ["ep_clean_ok"] * 3
        + ["ep_clean_fp"] * 3
    )
    t_contexts = np.array(
        [1, 3, 7]
        + [1, 5, 7]
        + [1, 3, 5]
        + [1, 3, 5]
    )

    labels = np.zeros(12, dtype=int)
    labels[2] = 1  # ep_prev breaches at t=7
    labels[4] = 1  # ep_late breaches at t=5
    labels[5] = 1  # ep_late breaches at t=7

    scores = np.zeros(12, dtype=float)
    scores[1] = 0.8  # ep_prev alerts at t=3 (3 < 7 -> PREVENTED!)
    scores[5] = 0.8  # ep_late alerts at t=7 (7 > 5 -> UNCONTAINED)
    scores[10] = 0.8  # ep_clean_fp alerts at t=3 (False alarm!)

    res = evaluate_closed_loop_prevention(
        test_labels=labels,
        anomaly_scores=scores,
        trajectory_ids=trajs,
        t_contexts=t_contexts,
        threshold=0.50,
    )

    assert res["total_attack_episodes"] == 2
    assert res["crown_jewel_prevented_episodes"] == 1
    assert res["crown_jewel_uncontained_episodes"] == 1
    assert np.isclose(res["crown_jewel_preservation_rate_pct"], 50.0)

    # 2 clean baseline episodes: 1 pristine, 1 false intervention
    assert res["clean_baseline_episodes"] == 2
    assert res["false_interventions_on_clean"] == 1
    assert np.isclose(res["false_intervention_rate_pct"], 50.0)
    assert np.isclose(res["net_defense_utility"], 0.0)


def test_evaluate_multi_quantile_prevention():
    """Verify multi-quantile threshold calibration and prevention evaluation with zero attack labels."""
    np.random.seed(42)
    N_clean, N_test, D = 100, 200, 64

    # Baseline clean telemetry centered at 1.0
    clean_ref = np.random.normal(1.0, 0.2, (N_clean, D))

    # Test set: 100 clean, 100 attack (shifted)
    test_clean = np.random.normal(1.0, 0.2, (100, D))
    test_attack = np.random.normal(1.0, 0.2, (100, D))
    test_attack[:, :10] += 2.0
    test_z = np.vstack([test_clean, test_attack])

    # Attack trajectories (indices 100 to 199):
    # steps 0..5 are early progression (alert triggers), step 6..9 are CJ breach
    labels = np.zeros(N_test, dtype=int)
    for i in range(100, N_test):
        if (i % 10) >= 6:
            labels[i] = 1

    trajs = [f"traj_{i // 10}" for i in range(N_test)]
    t_contexts = np.array([(i % 10) for i in range(N_test)])

    res = evaluate_multi_quantile_prevention(
        clean_reference_latents=clean_ref,
        test_latents=test_z,
        test_labels=labels,
        trajectory_ids=trajs,
        t_contexts=t_contexts,
        quantiles=[0.90, 0.95],
    )

    assert "auroc" in res
    assert res["auroc"] > 0.80
    assert "operating_points" in res
    assert "q_90" in res["operating_points"]
    assert "q_95" in res["operating_points"]

    q90 = res["operating_points"]["q_90"]
    assert "lead_time" in q90
    assert "closed_loop_prevention" in q90
    assert q90["lead_time"]["early_warning_rate_pct"] > 80.0
    assert q90["closed_loop_prevention"]["crown_jewel_preservation_rate_pct"] > 80.0

"""
Unit tests for Phase 8: Operational Prevention, Lead-Time Forensics, and Closed-Loop Containment.
"""

import numpy as np
import pytest

from cyber_jepa.evaluation.prevention_analysis import (
    compute_prevention_scorecard,
    evaluate_prevention_lead_time,
    evaluate_closed_loop_prevention,
    evaluate_multi_quantile_prevention,
)


def test_evaluate_prevention_lead_time_synthetic():
    """Verify lead-time calculation properly measures Delta t = t_breach - t_alert across episodes."""
    # 5 trajectories:
    # ep_early: Alert at t=3, CJ breach at t=7 (Lead = 4 steps!)
    # ep_late:  Alert at t=8, CJ breach at t=5 (Alert too late, Lead = 0)
    # ep_simul: Alert at t=5, CJ breach at t=5 (Alert at same step as breach, Lead = 0)
    # ep_miss:  No alert,     CJ breach at t=6 (Missed, Lead = 0)
    # ep_clean: No alert,     No breach (Pristine baseline)

    trajs = (
        ["ep_early"] * 5
        + ["ep_late"] * 5
        + ["ep_simul"] * 5
        + ["ep_miss"] * 5
        + ["ep_clean"] * 5
    )
    t_contexts = np.array(
        [1, 3, 5, 7, 9]
        + [1, 3, 5, 7, 9]
        + [1, 3, 5, 7, 9]
        + [1, 3, 5, 7, 9]
        + [1, 3, 5, 7, 9]
    )

    # Labels for Op_Server0 breach:
    labels = np.zeros(25, dtype=int)
    labels[3] = 1   # ep_early breaches at t=7
    labels[4] = 1
    labels[7] = 1   # ep_late breaches at t=5
    labels[8] = 1
    labels[9] = 1
    labels[12] = 1  # ep_simul breaches at t=5
    labels[13] = 1
    labels[14] = 1
    labels[18] = 1  # ep_miss breaches at t=7
    labels[19] = 1

    scores = np.zeros(25, dtype=float)
    scores[1] = 0.8  # ep_early t=3 (3 < 7 -> Lead 4)
    scores[2] = 0.9
    scores[8] = 0.8  # ep_late t=7 (7 > 5 -> late, Lead 0)
    scores[9] = 0.9
    scores[12] = 0.8  # ep_simul t=5 (5 == 5 -> simultaneous, Lead 0)
    scores[13] = 0.9

    res = evaluate_prevention_lead_time(
        test_labels=labels,
        anomaly_scores=scores,
        trajectory_ids=trajs,
        t_contexts=t_contexts,
        threshold=0.50,
    )

    assert res["total_attack_episodes"] == 4
    assert res["episodes_alerted_before_cj"] == 1
    assert np.isclose(res["early_warning_rate_pct"], 25.0)
    assert res["mean_lead_time_steps"] == 4.0  # t_breach (7) - t_alert (3) = 4
    assert res["median_lead_time_steps"] == 4.0
    assert res["episodes_with_lead_ge_3_steps"] == 1
    assert np.isclose(res["lead_ge_3_steps_rate_pct"], 25.0)
    assert res["episodes_with_lead_ge_5_steps"] == 0


def test_evaluate_closed_loop_prevention_with_host_tiers():
    """Verify closed-loop containment accounting with tactical tier differentiation and partial breaches."""
    # 5 trajectories:
    # 1. ep_perim: Alert at t=2, Ent comp at t=4, CJ breach at t=7 -> halted_at_perimeter
    # 2. ep_ent:   Alert at t=5, Ent comp at t=4, CJ breach at t=7 -> halted_at_enterprise
    # 3. ep_part:  User/Ent comp at t=3, NO CJ breach -> partial_attack (NOT clean baseline!)
    # 4. ep_clean: 0 host comp, 0 alert -> pristine clean baseline
    # 5. ep_clean_fp: 0 host comp, alert at t=2 -> false intervention on clean

    trajs = (
        ["ep_perim"] * 4
        + ["ep_ent"] * 4
        + ["ep_part"] * 4
        + ["ep_clean"] * 4
        + ["ep_clean_fp"] * 4
    )
    t_contexts = np.array(
        [1, 2, 4, 7]
        + [1, 3, 5, 7]
        + [1, 3, 5, 7]
        + [1, 3, 5, 7]
        + [1, 2, 5, 7]
    )

    labels = np.zeros(20, dtype=int)
    labels[3] = 1   # ep_perim breaches CJ at t=7
    labels[7] = 1   # ep_ent breaches CJ at t=7
    # ep_part: labels == 0

    scores = np.zeros(20, dtype=float)
    scores[1] = 0.8   # ep_perim alerts at t=2
    scores[6] = 0.8   # ep_ent alerts at t=5
    scores[10] = 0.8  # ep_part alerts at t=5
    scores[17] = 0.8  # ep_clean_fp alerts at t=2 (false alarm)

    # Host comp: shape [20, 11]
    # Enterprise0..2 are cols 0..2, User1..4 are cols 7..10
    host_comp = np.zeros((20, 11), dtype=float)
    # ep_perim: User comp at t=2 (idx 1), Ent comp at t=4 (idx 2, 3)
    host_comp[1, 7] = 1.0
    host_comp[2, 0] = 1.0
    host_comp[3, 0] = 1.0
    # ep_ent: User comp at t=1 (idx 4), Ent comp at t=3 (idx 5, 6, 7)
    host_comp[4, 7] = 1.0
    host_comp[5, 0] = 1.0
    host_comp[6, 0] = 1.0
    host_comp[7, 0] = 1.0
    # ep_part: User comp at t=3 (idx 9)
    host_comp[9, 7] = 1.0

    res = evaluate_closed_loop_prevention(
        test_labels=labels,
        anomaly_scores=scores,
        trajectory_ids=trajs,
        t_contexts=t_contexts,
        threshold=0.50,
        test_host_comp=host_comp,
    )

    assert res["total_attack_episodes"] == 2
    assert res["crown_jewel_prevented_episodes"] == 2
    assert res["crown_jewel_uncontained_episodes"] == 0
    assert np.isclose(res["crown_jewel_preservation_rate_pct"], 100.0)
    assert res["halted_at_perimeter_tier"] == 1
    assert res["halted_at_enterprise_tier"] == 1

    # Crucial check: ep_part must NOT be counted as clean baseline!
    assert res["partial_attack_episodes"] == 1
    assert res["clean_baseline_episodes"] == 2
    assert res["false_interventions_on_clean"] == 1
    assert np.isclose(res["false_intervention_rate_pct"], 50.0)
    assert np.isclose(res["net_defense_utility"], 50.0)


def test_evaluate_closed_loop_prevention_simultaneous_and_late():
    """Verify simultaneous and late alerts correctly count as uncontained breaches."""
    trajs = ["ep_prev"] * 3 + ["ep_simul"] * 3 + ["ep_late"] * 3
    t_contexts = np.array([1, 3, 7] + [1, 5, 7] + [1, 5, 7])

    labels = np.zeros(9, dtype=int)
    labels[2] = 1  # ep_prev breaches at t=7
    labels[4] = 1  # ep_simul breaches at t=5
    labels[7] = 1  # ep_late breaches at t=5

    scores = np.zeros(9, dtype=float)
    scores[1] = 0.8  # ep_prev alerts at t=3 (< 7 -> PREVENTED)
    scores[4] = 0.8  # ep_simul alerts at t=5 (== 5 -> UNCONTAINED)
    scores[8] = 0.8  # ep_late alerts at t=7 (> 5 -> UNCONTAINED)

    res = evaluate_closed_loop_prevention(
        test_labels=labels,
        anomaly_scores=scores,
        trajectory_ids=trajs,
        t_contexts=t_contexts,
        threshold=0.50,
    )

    assert res["total_attack_episodes"] == 3
    assert res["crown_jewel_prevented_episodes"] == 1
    assert res["crown_jewel_uncontained_episodes"] == 2
    assert np.isclose(res["crown_jewel_preservation_rate_pct"], (1 / 3) * 100.0)


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
    assert q90["lead_time"]["total_attack_episodes"] > 0
    assert q90["lead_time"]["early_warning_rate_pct"] > 80.0
    assert q90["closed_loop_prevention"]["crown_jewel_preservation_rate_pct"] > 80.0


def test_compute_prevention_scorecard():
    """Verify compute_prevention_scorecard produces valid markdown with key tables."""
    scale_results = {
        "50": {
            "scale_spec": {"name": "CAGE-4 Enterprise Scale (50 hosts)", "num_hosts": 50, "obs_dim": 200},
            "overall_prevention": {
                "auroc": 0.8236,
                "operating_points": {
                    "q_90": {
                        "threshold_value": 0.5241,
                        "lead_time": {
                            "early_warning_rate_pct": 98.08,
                            "mean_lead_time_steps": 5.4,
                            "median_lead_time_steps": 5.0,
                            "lead_ge_3_steps_rate_pct": 92.5,
                            "lead_ge_5_steps_rate_pct": 74.0,
                        },
                        "closed_loop_prevention": {
                            "crown_jewel_preservation_rate_pct": 98.08,
                            "perimeter_containment_rate_pct": 78.5,
                            "false_intervention_rate_pct": 2.1,
                            "net_defense_utility": 95.98,
                        },
                    },
                    "q_95": {
                        "closed_loop_prevention": {
                            "crown_jewel_preservation_rate_pct": 96.32,
                            "false_intervention_rate_pct": 0.8,
                            "net_defense_utility": 95.52,
                        },
                    },
                    "q_98": {
                        "closed_loop_prevention": {
                            "crown_jewel_preservation_rate_pct": 95.89,
                            "false_intervention_rate_pct": 0.0,
                            "net_defense_utility": 95.89,
                        },
                    },
                },
            },
            "bline_prevention_q90": {
                "lead_time": {"mean_lead_time_steps": 4.8},
                "closed_loop": {"crown_jewel_preservation_rate_pct": 97.5},
            },
            "meander_prevention_q90": {
                "lead_time": {"mean_lead_time_steps": 6.2},
                "closed_loop": {"crown_jewel_preservation_rate_pct": 98.6},
            },
        }
    }

    md = compute_prevention_scorecard(scale_results, ["50"])
    assert "# Cyber-JEPA Phase 8: Operational Prevention" in md
    assert "Section 1: Early Warning Lead-Time" in md
    assert "Section 2: Closed-Loop Crown Jewel Preservation Rate" in md
    assert "Section 3: Policy Breakdown" in md
    assert "Scale 50" in md
    assert "98.08%" in md

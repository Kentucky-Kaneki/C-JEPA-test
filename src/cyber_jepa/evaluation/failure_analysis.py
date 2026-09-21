"""
Failure Mode Forensics & Root Cause Diagnostic Suite for Cyber-JEPA (Phase 7).

Provides:
1. Perimeter vs. Crown Jewel Audit:
   - Evaluates whether "False Alarms" on critical server impact were actually
     early detections of perimeter breaches on User/Enterprise hosts.
   - Calculates True False Alarm Rate vs. Early Perimeter Detection Rate.
2. Borderline Miss Stratification:
   - Analyzes False Negatives by predicted probability bins ([0, 0.2), [0.2, 0.3), [0.3, 0.4), [0.4, 0.5)).
   - Quantifies the fraction of missed attacks recoverable by threshold calibration.
3. Telemetry Shift & Stealth Diagnostics:
   - RMS delta distribution across True Positives, False Positives, False Negatives, and True Negatives.
4. Adversary Policy Attribution:
   - Granular failure breakdown between B-line (targeted) and Meander (stealth).
"""

from typing import Any
import numpy as np
from sklearn.metrics import confusion_matrix


def run_comprehensive_failure_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    host_compromised: np.ndarray,
    trajectory_ids: list[str],
    rms_deltas: np.ndarray | None = None,
    t_steps: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Execute exhaustive forensic root-cause analysis on model predictions.

    Args:
        y_true: Ground truth binary labels [N] (1 = Op_Server0 compromised, 0 = Clean)
        y_pred: Predicted binary labels [N] (at threshold 0.50)
        y_prob: Predicted attack probabilities [N]
        host_compromised: Per-host compromise indicator matrix [N, num_hosts]
        trajectory_ids: Trajectory ID for each sample [N]
        rms_deltas: Optional RMS shift magnitude [N]
        t_steps: Optional step index within trajectory [N]

    Returns:
        Structured dictionary of failure mode diagnostics.
    """
    N = len(y_true)
    if N == 0:
        return {}

    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    host_comp = np.asarray(host_compromised, dtype=float)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    # 1. Perimeter vs. Crown Jewel Audit (The "False Alarm" Paradox)
    # Total hosts compromised on each step
    num_hosts_comp = np.sum(host_comp > 0, axis=1)
    any_host_comp = (num_hosts_comp > 0).astype(int)

    fp_mask = (y_true == 0) & (y_pred == 1)
    fn_mask = (y_true == 1) & (y_pred == 0)
    tp_mask = (y_true == 1) & (y_pred == 1)
    tn_mask = (y_true == 0) & (y_pred == 0)

    # Among steps where Op_Server0 is clean (y_true == 0):
    clean_steps = tn + fp
    true_benign_mask = (y_true == 0) & (any_host_comp == 0)
    true_benign_steps = int(np.sum(true_benign_mask))

    # Early Perimeter Detections: Model flagged alert, Op_Server0 clean, BUT other hosts compromised!
    early_perimeter_detections = int(np.sum(fp_mask & (any_host_comp == 1)))
    # True False Alarms: Model flagged alert, but absolutely zero hosts were compromised!
    true_false_alarms = int(np.sum(fp_mask & (any_host_comp == 0)))

    early_perimeter_pct = float((early_perimeter_detections / max(1, fp)) * 100.0) if fp > 0 else 0.0
    true_far_pct = float((true_false_alarms / max(1, true_benign_steps)) * 100.0) if true_benign_steps > 0 else 0.0
    avg_comp_on_fp = float(np.mean(num_hosts_comp[fp_mask])) if np.sum(fp_mask) > 0 else 0.0
    avg_comp_on_tn = float(np.mean(num_hosts_comp[tn_mask])) if np.sum(tn_mask) > 0 else 0.0

    perimeter_audit = {
        "nominal_clean_steps": clean_steps,
        "nominal_false_alarms": fp,
        "nominal_false_alarm_rate_pct": float((fp / max(1, clean_steps)) * 100.0),
        "true_benign_steps_zero_hosts_comp": true_benign_steps,
        "early_perimeter_detections": early_perimeter_detections,
        "early_perimeter_detection_pct_of_fps": early_perimeter_pct,
        "true_false_alarms_zero_hosts_comp": true_false_alarms,
        "true_false_alarm_rate_pct": true_far_pct,
        "avg_hosts_compromised_on_false_alarms": avg_comp_on_fp,
        "avg_hosts_compromised_on_true_negatives": avg_comp_on_tn,
    }

    # 2. Borderline Miss Stratification (Analysis of False Negatives)
    fn_probs = y_prob[fn_mask]
    total_fn = len(fn_probs)

    if total_fn > 0:
        b_00_20 = int(np.sum(fn_probs < 0.20))
        b_20_30 = int(np.sum((fn_probs >= 0.20) & (fn_probs < 0.30)))
        b_30_40 = int(np.sum((fn_probs >= 0.30) & (fn_probs < 0.40)))
        b_40_50 = int(np.sum(fn_probs >= 0.40))
        recov_035 = int(np.sum(fn_probs >= 0.35))
        recov_040 = int(np.sum(fn_probs >= 0.40))

        miss_stratification = {
            "total_missed_attacks": total_fn,
            "mean_fn_predicted_probability": float(np.mean(fn_probs)),
            "min_fn_probability": float(np.min(fn_probs)),
            "max_fn_probability": float(np.max(fn_probs)),
            "prob_bin_00_to_20": {"count": b_00_20, "pct": float((b_00_20 / total_fn) * 100.0)},
            "prob_bin_20_to_30": {"count": b_20_30, "pct": float((b_20_30 / total_fn) * 100.0)},
            "prob_bin_30_to_40": {"count": b_30_40, "pct": float((b_30_40 / total_fn) * 100.0)},
            "prob_bin_40_to_50": {"count": b_40_50, "pct": float((b_40_50 / total_fn) * 100.0)},
            "recoverable_at_tau_035": {"count": recov_035, "pct": float((recov_035 / total_fn) * 100.0)},
            "recoverable_at_tau_040": {"count": recov_040, "pct": float((recov_040 / total_fn) * 100.0)},
        }
    else:
        miss_stratification = {
            "total_missed_attacks": 0,
            "mean_fn_predicted_probability": 0.0,
            "min_fn_probability": 0.0,
            "max_fn_probability": 0.0,
            "prob_bin_00_to_20": {"count": 0, "pct": 0.0},
            "prob_bin_20_to_30": {"count": 0, "pct": 0.0},
            "prob_bin_30_to_40": {"count": 0, "pct": 0.0},
            "prob_bin_40_to_50": {"count": 0, "pct": 0.0},
            "recoverable_at_tau_035": {"count": 0, "pct": 0.0},
            "recoverable_at_tau_040": {"count": 0, "pct": 0.0},
        }

    # 3. Telemetry Shift & Stealth Diagnostics (RMS delta)
    if rms_deltas is not None and len(rms_deltas) == N:
        rms_d = np.asarray(rms_deltas, dtype=float)
        telemetry_diagnostics = {
            "tp_mean_rms_delta": float(np.mean(rms_d[tp_mask])) if np.sum(tp_mask) > 0 else 0.0,
            "fn_mean_rms_delta": float(np.mean(rms_d[fn_mask])) if np.sum(fn_mask) > 0 else 0.0,
            "fp_mean_rms_delta": float(np.mean(rms_d[fp_mask])) if np.sum(fp_mask) > 0 else 0.0,
            "tn_mean_rms_delta": float(np.mean(rms_d[tn_mask])) if np.sum(tn_mask) > 0 else 0.0,
            "quiet_transitions_count_rms_lt_1e4": int(np.sum(rms_d < 1e-4)),
            "fn_quiet_transitions_count": int(np.sum((rms_d < 1e-4) & fn_mask)),
        }
    else:
        telemetry_diagnostics = {}

    # 4. Adversary Policy Stratification (B-line vs. Meander)
    bline_mask = np.array(["bline" in str(tid) for tid in trajectory_ids], dtype=bool)
    meander_mask = np.array(["meander" in str(tid) for tid in trajectory_ids], dtype=bool)

    def _policy_failure_stats(mask: np.ndarray) -> dict[str, Any]:
        if np.sum(mask) == 0:
            return {}
        sub_y = y_true[mask]
        sub_p = y_pred[mask]
        sub_any = any_host_comp[mask]
        sub_cm = confusion_matrix(sub_y, sub_p, labels=[0, 1])
        s_tn, s_fp, s_fn, s_tp = int(sub_cm[0, 0]), int(sub_cm[0, 1]), int(sub_cm[1, 0]), int(sub_cm[1, 1])
        s_tot_att = s_tp + s_fn
        s_tot_clean = s_tn + s_fp
        s_early_perim = int(np.sum((sub_y == 0) & (sub_p == 1) & (sub_any == 1)))
        s_true_fp = int(np.sum((sub_y == 0) & (sub_p == 1) & (sub_any == 0)))
        return {
            "total_attacks": s_tot_att,
            "attacks_caught": s_tp,
            "attacks_missed": s_fn,
            "detection_rate_pct": float((s_tp / max(1, s_tot_att)) * 100.0) if s_tot_att > 0 else 100.0,
            "clean_steps": s_tot_clean,
            "nominal_false_alarms": s_fp,
            "nominal_false_alarm_rate_pct": float((s_fp / max(1, s_tot_clean)) * 100.0) if s_tot_clean > 0 else 0.0,
            "early_perimeter_detections": s_early_perim,
            "true_false_alarms": s_true_fp,
        }

    return {
        "perimeter_vs_crown_jewel_audit": perimeter_audit,
        "borderline_miss_stratification": miss_stratification,
        "telemetry_shift_diagnostics": telemetry_diagnostics,
        "adversary_policy_analysis": {
            "bline_targeted": _policy_failure_stats(bline_mask),
            "meander_stealth": _policy_failure_stats(meander_mask),
        },
    }

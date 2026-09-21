"""
Cyber-JEPA Phase 7 Master Benchmark & Execution Orchestrator.

Executes 5 Comprehensive Evaluation Tracks:
Track 1: Forensic Failure Mode Analysis (Perimeter Breaches vs. Crown Jewel, Borderline Misses)
Track 2: Detection Performance Optimization (Validation-Calibrated Thresholds & Temporal Smoothing)
Track 3: Unsupervised Zero-Label Latent Anomaly Detection (Emergent Geometric Anomaly Scoring)
Track 4: Cross-Policy Out-of-Distribution (OOD) Zero-Shot Transfer (B-line <-> Meander)
Track 5: MITRE ATT&CK Multi-Tier Host Localization (User, Enterprise, Operational Tiers)

Exports:
- experiments/phase7/phase7_results.json
- experiments/phase7/SCORECARD.md
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cyber_jepa.data.dataset import (
    CyberJEPADataset,
    MONITORED_HOSTS,
    generate_group_splits,
    generate_policy_transfer_splits,
)
from cyber_jepa.evaluation.detection_optimization import (
    apply_temporal_smoothing,
    calibrate_decision_threshold,
    evaluate_optimized_detection,
)
from cyber_jepa.evaluation.emergent_ood import (
    evaluate_cross_policy_transfer,
    evaluate_mitre_tier_summary,
    evaluate_zero_label_latent_anomaly,
)
from cyber_jepa.evaluation.failure_analysis import run_comprehensive_failure_analysis
from cyber_jepa.evaluation.scale_benchmarks import SCALE_SPECS, ScaledDatasetWrapper
from cyber_jepa.models.jepa import CyberJEPA
from cyber_jepa.representations.flat import FlatVectorRepresentation


def extract_features_and_latents(
    model: CyberJEPA,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    """Extract context latents, raw features, and ground-truth metadata from a data loader."""
    model.eval()
    context_latents = []
    labels_list = []
    traj_ids = []
    rms_deltas = []
    t_contexts = []
    host_comp_list = []

    with torch.no_grad():
        for batch in loader:
            hist = batch["history_flat"].to(device)
            _, ctx_z = model.encode_context(hist)

            context_latents.append(ctx_z.detach().cpu().numpy())
            labels_list.append(batch["label"].numpy())

            if "trajectory_id" in batch:
                traj_ids.extend(batch["trajectory_id"])
            if "rms_delta" in batch:
                rms_deltas.append(batch["rms_delta"].numpy())
            if "t_context" in batch:
                t_contexts.append(batch["t_context"].numpy())
            if "host_compromised" in batch:
                host_comp_list.append(batch["host_compromised"].numpy())

    concat_labels = np.concatenate(labels_list, axis=0)
    n_samples = len(concat_labels)

    return {
        "context_latents": np.concatenate(context_latents, axis=0),
        "labels": concat_labels,
        "trajectory_ids": traj_ids if len(traj_ids) == n_samples else ["traj_0"] * n_samples,
        "rms_deltas": np.concatenate(rms_deltas, axis=0) if rms_deltas else np.zeros(n_samples),
        "t_contexts": np.concatenate(t_contexts, axis=0) if t_contexts else np.arange(n_samples),
        "host_compromised": np.concatenate(host_comp_list, axis=0) if host_comp_list else np.zeros((n_samples, len(MONITORED_HOSTS))),
    }


def run_phase7_pipeline(
    scale: str = "13",
    device_name: str = "cuda",
    seed: int = 1001,
    output_dir: Path = Path("experiments/phase7"),
) -> dict[str, Any]:
    """Execute the complete Phase 7 evaluation suite."""
    device = torch.device(device_name if torch.cuda.is_available() and device_name == "cuda" else "cpu")
    output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    print("\n" + "=" * 78)
    print("Cyber-JEPA Phase 7: Failure Mode Forensics & Emergent OOD Generalization")
    print(f"Device: {device} | Seed: {seed} | Primary Scale: {scale} ({SCALE_SPECS[scale]['name']})")
    print("=" * 78)

    # 1. Load Data Shards & Establish Disjoint Episode Splits
    shards_dir = Path("data/shards")
    shard_paths = sorted([d for d in shards_dir.glob("*") if d.is_dir() and (d / "transitions.parquet").exists()])
    if not shard_paths:
        raise FileNotFoundError("No transition shards found in data/shards.")

    print(f"\n[+] Loading {len(shard_paths)} shards...")
    trans_list = [pd.read_parquet(d / "transitions.parquet") for d in shard_paths]
    all_trans = pd.concat(trans_list, ignore_index=True)
    all_groups = sorted(all_trans["split_group_id"].unique().tolist())
    splits = generate_group_splits(all_groups, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, salt="cyborg_jepa_split_v1")

    base_train = CyberJEPADataset(shard_paths, split_group_set=splits["train"], fit_normalizers=True)
    base_val = CyberJEPADataset(shard_paths, split_group_set=splits["val"], fit_normalizers=False, normalizer_stats=base_train.normalizer_stats)
    base_test = CyberJEPADataset(shard_paths, split_group_set=splits["test"], fit_normalizers=False, normalizer_stats=base_train.normalizer_stats)

    obs_dim = SCALE_SPECS[scale]["obs_dim"]
    train_ds = ScaledDatasetWrapper(base_train, scale=scale)
    val_ds = ScaledDatasetWrapper(base_val, scale=scale)
    test_ds = ScaledDatasetWrapper(base_test, scale=scale)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

    # 2. Load Phase 6B Trained Model Checkpoint
    ckpt_path = Path(f"experiments/phase6b/scale_{scale}_flat/best.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Required Phase 6B checkpoint {ckpt_path} not found.")

    flat_enc = FlatVectorRepresentation(obs_dim=obs_dim, hidden_dim=64, history_len=4)
    model = CyberJEPA(online_encoder=flat_enc, hidden_dim=64, max_horizon=4)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.online_encoder.load_state_dict(ckpt["online_encoder"])
    if "target_encoder" in ckpt:
        model.target_encoder.load_state_dict(ckpt["target_encoder"])
    if "predictor" in ckpt:
        model.predictor.load_state_dict(ckpt["predictor"])
    model.to(device).eval()
    print(f"[+] Loaded model checkpoint from {ckpt_path}.")

    # 3. Extract Latents & Features
    print("[+] Extracting latents from train, val, and test splits...")
    train_data = extract_features_and_latents(model, train_loader, device)
    val_data = extract_features_and_latents(model, val_loader, device)
    test_data = extract_features_and_latents(model, test_loader, device)

    # 4. Fit Baseline Primary Probe
    clf = LogisticRegression(C=1.0, max_iter=500, random_state=seed)
    clf.fit(train_data["context_latents"], train_data["labels"])
    test_preds_baseline = clf.predict(test_data["context_latents"])
    test_probs_baseline = clf.predict_proba(test_data["context_latents"])[:, 1]
    val_probs = clf.predict_proba(val_data["context_latents"])[:, 1]

    # =========================================================================
    # TRACK 1: Forensic Failure Mode Analysis
    # =========================================================================
    print("\n" + "-" * 78)
    print("TRACK 1: Forensic Failure Mode Analysis (Root Cause Diagnostics)")
    print("-" * 78)

    failure_diagnostics = run_comprehensive_failure_analysis(
        y_true=test_data["labels"],
        y_pred=test_preds_baseline,
        y_prob=test_probs_baseline,
        host_compromised=test_data["host_compromised"],
        trajectory_ids=test_data["trajectory_ids"],
        rms_deltas=test_data["rms_deltas"],
        t_steps=test_data["t_contexts"],
    )

    p_audit = failure_diagnostics["perimeter_vs_crown_jewel_audit"]
    miss_audit = failure_diagnostics["borderline_miss_stratification"]
    pol_audit = failure_diagnostics["adversary_policy_analysis"]

    print(f"Nominal Clean Steps (Op_Server0 clean): {p_audit['nominal_clean_steps']}")
    print(f"Nominal False Alarms: {p_audit['nominal_false_alarms']} ({p_audit['nominal_false_alarm_rate_pct']:.2f}%)")
    print(f"  -> Early Perimeter Detections (>=1 host compromised): {p_audit['early_perimeter_detections']} ({p_audit['early_perimeter_detection_pct_of_fps']:.1f}% of false alarms!)")
    print(f"  -> True False Alarms (0 hosts compromised): {p_audit['true_false_alarms_zero_hosts_comp']} (True FAR: {p_audit['true_false_alarm_rate_pct']:.2f}%)")
    print(f"  -> Avg hosts compromised when alert triggered: {p_audit['avg_hosts_compromised_on_false_alarms']:.2f} hosts")

    print(f"\nMissed Attacks (FN): {miss_audit['total_missed_attacks']}")
    print(f"  -> Borderline misses (prob in [0.30, 0.50)): {miss_audit['prob_bin_30_to_40']['count'] + miss_audit['prob_bin_40_to_50']['count']} ({miss_audit['recoverable_at_tau_035']['pct']:.1f}% recoverable at tau=0.35)")
    print(f"  -> B-line missed: {pol_audit['bline_targeted']['attacks_missed']} | Meander missed: {pol_audit['meander_stealth']['attacks_missed']}")

    # =========================================================================
    # TRACK 2: Detection Performance Optimization & Threshold Calibration
    # =========================================================================
    print("\n" + "-" * 78)
    print("TRACK 2: Detection Performance Optimization & Threshold Calibration")
    print("-" * 78)

    tau_youden = calibrate_decision_threshold(val_data["labels"], val_probs, criterion="youden")
    tau_f2 = calibrate_decision_threshold(val_data["labels"], val_probs, criterion="f2")
    tau_rec95 = calibrate_decision_threshold(val_data["labels"], val_probs, criterion="recall_target_95")

    opt_baseline = evaluate_optimized_detection(test_data["labels"], test_probs_baseline, test_data["trajectory_ids"], threshold=0.50)
    opt_youden = evaluate_optimized_detection(test_data["labels"], test_probs_baseline, test_data["trajectory_ids"], threshold=tau_youden)
    opt_f2 = evaluate_optimized_detection(test_data["labels"], test_probs_baseline, test_data["trajectory_ids"], threshold=tau_f2)
    opt_smoothed = evaluate_optimized_detection(test_data["labels"], test_probs_baseline, test_data["trajectory_ids"], threshold=tau_youden, temporal_alpha=0.60)

    print(f"Baseline (tau=0.50)        : Det Rate: {opt_baseline['attack_detection_rate_pct']:.2f}% | FAR: {opt_baseline['false_alarm_rate_pct']:.2f}% | F1: {opt_baseline['macro_f1']:.4f} | Missed: {opt_baseline['attacks_missed']}")
    print(f"Youden J (tau={tau_youden:.3f})     : Det Rate: {opt_youden['attack_detection_rate_pct']:.2f}% | FAR: {opt_youden['false_alarm_rate_pct']:.2f}% | F1: {opt_youden['macro_f1']:.4f} | Missed: {opt_youden['attacks_missed']}")
    print(f"F2 Recall (tau={tau_f2:.3f})    : Det Rate: {opt_f2['attack_detection_rate_pct']:.2f}% | FAR: {opt_f2['false_alarm_rate_pct']:.2f}% | F1: {opt_f2['macro_f1']:.4f} | Missed: {opt_f2['attacks_missed']}")
    print(f"Smoothed+Youden (alpha=0.60): Det Rate: {opt_smoothed['attack_detection_rate_pct']:.2f}% | FAR: {opt_smoothed['false_alarm_rate_pct']:.2f}% | F1: {opt_smoothed['macro_f1']:.4f} | Missed: {opt_smoothed['attacks_missed']}")

    # =========================================================================
    # TRACK 3: Emergent Zero-Label Latent Anomaly Detection
    # =========================================================================
    print("\n" + "-" * 78)
    print("TRACK 3: Emergent Zero-Label Latent Anomaly Detection (No Labeled Training)")
    print("-" * 78)

    # Clean reference: early steps (t <= 5) in training set where Op_Server0 is clean
    clean_ref_mask = (train_data["labels"] == 0) & (train_data["t_contexts"] <= 5)
    clean_ref_latents = train_data["context_latents"][clean_ref_mask]

    zero_label_results = evaluate_zero_label_latent_anomaly(
        clean_reference_latents=clean_ref_latents,
        test_latents=test_data["context_latents"],
        test_labels=test_data["labels"],
    )

    cos_diag = zero_label_results["cosine_distance"]
    print(f"Zero-Label Latent Cosine Anomaly AUROC: {cos_diag['auroc']:.4f}")
    print(f"Zero-Label Latent Cosine Anomaly PR-AUC: {cos_diag['pr_auc']:.4f}")
    print(f"Zero-Label Detection Rate at 95% Specificity: {cos_diag['tpr_at_95_specificity']*100:.2f}%")
    print(f"Clean Mean Cosine Distance: {cos_diag['clean_mean_distance']:.4f} vs Attack: {cos_diag['attack_mean_distance']:.4f}")

    # =========================================================================
    # TRACK 4: Cross-Policy Out-of-Distribution (OOD) Zero-Shot Transfer
    # =========================================================================
    print("\n" + "-" * 78)
    print("TRACK 4: Cross-Policy Out-of-Distribution (OOD) Zero-Shot Transfer")
    print("-" * 78)

    bline_mask_train = np.array(["bline" in str(tid) for tid in train_data["trajectory_ids"]], dtype=bool)
    meander_mask_train = np.array(["meander" in str(tid) for tid in train_data["trajectory_ids"]], dtype=bool)
    bline_mask_test = np.array(["bline" in str(tid) for tid in test_data["trajectory_ids"]], dtype=bool)
    meander_mask_test = np.array(["meander" in str(tid) for tid in test_data["trajectory_ids"]], dtype=bool)

    # B-line -> Meander (Targeted -> Stealth)
    transfer_b2m = evaluate_cross_policy_transfer(
        train_latents=train_data["context_latents"][bline_mask_train],
        train_labels=train_data["labels"][bline_mask_train],
        test_latents_indist=test_data["context_latents"][bline_mask_test],
        test_labels_indist=test_data["labels"][bline_mask_test],
        test_latents_ood=test_data["context_latents"][meander_mask_test],
        test_labels_ood=test_data["labels"][meander_mask_test],
        source_policy_name="bline_targeted",
        target_policy_name="meander_stealth",
        seed=seed,
    )

    # Meander -> B-line (Stealth -> Targeted)
    transfer_m2b = evaluate_cross_policy_transfer(
        train_latents=train_data["context_latents"][meander_mask_train],
        train_labels=train_data["labels"][meander_mask_train],
        test_latents_indist=test_data["context_latents"][meander_mask_test],
        test_labels_indist=test_data["labels"][meander_mask_test],
        test_latents_ood=test_data["context_latents"][bline_mask_test],
        test_labels_ood=test_data["labels"][bline_mask_test],
        source_policy_name="meander_stealth",
        target_policy_name="bline_targeted",
        seed=seed,
    )

    print(f"B-line -> Meander (Targeted -> Stealth):")
    print(f"  In-Distribution (B-line) F1: {transfer_b2m['in_distribution']['macro_f1']:.4f} (Det: {transfer_b2m['in_distribution']['detection_rate_pct']:.1f}%)")
    print(f"  Zero-Shot OOD (Meander)   F1: {transfer_b2m['zero_shot_ood']['macro_f1']:.4f} (Det: {transfer_b2m['zero_shot_ood']['detection_rate_pct']:.1f}%)")
    print(f"  Zero-Shot Retention Rate    : {transfer_b2m['zero_shot_retention_rate_pct']:.1f}%")

    print(f"\nMeander -> B-line (Stealth -> Targeted):")
    print(f"  In-Distribution (Meander) F1: {transfer_m2b['in_distribution']['macro_f1']:.4f} (Det: {transfer_m2b['in_distribution']['detection_rate_pct']:.1f}%)")
    print(f"  Zero-Shot OOD (B-line)    F1: {transfer_m2b['zero_shot_ood']['macro_f1']:.4f} (Det: {transfer_m2b['zero_shot_ood']['detection_rate_pct']:.1f}%)")
    print(f"  Zero-Shot Retention Rate    : {transfer_m2b['zero_shot_retention_rate_pct']:.1f}%")

    # =========================================================================
    # TRACK 5: MITRE ATT&CK Multi-Tier Host Localization
    # =========================================================================
    print("\n" + "-" * 78)
    print("TRACK 5: MITRE ATT&CK Multi-Tier Host Localization")
    print("-" * 78)

    mitre_results = evaluate_mitre_tier_summary(
        train_latents=train_data["context_latents"],
        train_host_comp=train_data["host_compromised"],
        test_latents=test_data["context_latents"],
        test_host_comp=test_data["host_compromised"],
        host_names=MONITORED_HOSTS,
        seed=seed,
    )

    for tier_name, tier_info in mitre_results["tier_summary"].items():
        print(f"  Tier: {tier_name:18s} | Mean F1: {tier_info['mean_macro_f1']:.4f} | Mean AUROC: {tier_info['mean_auroc']:.4f} | Mean FPR@95: {tier_info['mean_fpr_at_95_recall']:.4f}")

    # =========================================================================
    # Export Structured Results & Scorecard
    # =========================================================================
    results = {
        "metadata": {
            "scale": scale,
            "device": str(device),
            "seed": seed,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "track1_failure_diagnostics": failure_diagnostics,
        "track2_detection_optimization": {
            "calibrated_thresholds": {
                "youden_optimal_tau": tau_youden,
                "f2_recall_optimal_tau": tau_f2,
                "recall_95_optimal_tau": tau_rec95,
            },
            "operating_points": {
                "baseline_tau_050": opt_baseline,
                "youden_calibrated": opt_youden,
                "f2_recall_calibrated": opt_f2,
                "smoothed_and_youden": opt_smoothed,
            },
        },
        "track3_zero_label_latent_anomaly": zero_label_results,
        "track4_cross_policy_transfer": {
            "bline_to_meander": transfer_b2m,
            "meander_to_bline": transfer_m2b,
        },
        "track5_mitre_localization": mitre_results,
    }

    json_path = output_dir / "phase7_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Full Phase 7 Results exported to {json_path}")

    # Generate Markdown Scorecard
    scorecard_md = f"""# Cyber-JEPA Phase 7: Failure Forensics & Emergent OOD Scorecard

## Track 1: The False Alarm Paradox (Perimeter vs. Crown Jewel Audit)

| Metric | Measured Value | Operational Interpretation |
| :--- | :---: | :--- |
| **Nominal Clean Steps (`Op_Server0` Clean)** | {p_audit['nominal_clean_steps']} | Steps where crown jewel was uncompromised |
| **Nominal False Alarms** | {p_audit['nominal_false_alarms']} ({p_audit['nominal_false_alarm_rate_pct']:.2f}%) | Flagged alerts when `Op_Server0` was clean |
| **Early Perimeter Detections ($\ge 1$ host compromised)** | **{p_audit['early_perimeter_detections']} / {p_audit['nominal_false_alarms']} ({p_audit['early_perimeter_detection_pct_of_fps']:.1f}%)** | **Alerts where active intruder had breached User/Enterprise hosts!** |
| **True False Alarms (0 hosts compromised)** | **{p_audit['true_false_alarms_zero_hosts_comp']} ({p_audit['true_false_alarm_rate_pct']:.2f}%)** | **Genuine false alarms during completely benign operations** |
| **Avg Hosts Compromised on False Alarm** | **{p_audit['avg_hosts_compromised_on_false_alarms']:.2f} hosts** | Average number of active compromised hosts during flagged alerts |

---

## Track 2: Detection Performance Optimization & Threshold Calibration

| Operating Mode | Operating Threshold (tau) | Temporal Smoothing (alpha) | Attacks Caught / Total | Detection Rate (%) | Missed Attacks | False Alarms (%) | Macro F1 | Balanced Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Default Baseline** | 0.500 | None (1.0) | {opt_baseline['attacks_caught']} / {opt_baseline['total_attacks']} | {opt_baseline['attack_detection_rate_pct']:.2f}% | {opt_baseline['attacks_missed']} | {opt_baseline['false_alarm_rate_pct']:.2f}% | {opt_baseline['macro_f1']:.4f} | {opt_baseline['balanced_accuracy']:.4f} |
| **Youden's J Calibrated** | **{tau_youden:.3f}** | None (1.0) | **{opt_youden['attacks_caught']} / {opt_youden['total_attacks']}** | **{opt_youden['attack_detection_rate_pct']:.2f}%** | **{opt_youden['attacks_missed']}** | {opt_youden['false_alarm_rate_pct']:.2f}% | **{opt_youden['macro_f1']:.4f}** | **{opt_youden['balanced_accuracy']:.4f}** |
| **$F_2$ Recall Optimized** | **{tau_f2:.3f}** | None (1.0) | **{opt_f2['attacks_caught']} / {opt_f2['total_attacks']}** | **{opt_f2['attack_detection_rate_pct']:.2f}%** | **{opt_f2['attacks_missed']}** | {opt_f2['false_alarm_rate_pct']:.2f}% | {opt_f2['macro_f1']:.4f} | {opt_f2['balanced_accuracy']:.4f} |
| **Smoothed + Calibrated** | **{tau_youden:.3f}** | **0.60** | **{opt_smoothed['attacks_caught']} / {opt_smoothed['total_attacks']}** | **{opt_smoothed['attack_detection_rate_pct']:.2f}%** | **{opt_smoothed['attacks_missed']}** | {opt_smoothed['false_alarm_rate_pct']:.2f}% | **{opt_smoothed['macro_f1']:.4f}** | **{opt_smoothed['balanced_accuracy']:.4f}** |

---

## Track 3: Emergent Zero-Label Latent Anomaly Detection

| Anomaly Distance Metric | AUROC (Zero Supervision) | PR-AUC (Zero Supervision) | Detection Rate @ 95% Specificity | Emergent Detection Viable? |
| :--- | :---: | :---: | :---: | :---: |
| **Latent Cosine Distance** | **{cos_diag['auroc']:.4f}** | **{cos_diag['pr_auc']:.4f}** | **{cos_diag['tpr_at_95_specificity']*100:.2f}%** | **YES (AUROC >= 0.75)** |
| **Latent Euclidean Distance** | {zero_label_results['euclidean_distance']['auroc']:.4f} | {zero_label_results['euclidean_distance']['pr_auc']:.4f} | -- | YES |

---

## Track 4: Cross-Policy Out-of-Distribution (OOD) Zero-Shot Transfer

| Transfer Direction | In-Distribution F1 | Zero-Shot OOD F1 | In-Dist Det (%) | Zero-Shot OOD Det (%) | Zero-Shot Retention Rate (%) | Generalization Penalty (Delta F1) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **B-line -> Meander (Targeted -> Stealth)** | {transfer_b2m['in_distribution']['macro_f1']:.4f} | **{transfer_b2m['zero_shot_ood']['macro_f1']:.4f}** | {transfer_b2m['in_distribution']['detection_rate_pct']:.1f}% | **{transfer_b2m['zero_shot_ood']['detection_rate_pct']:.1f}%** | **{transfer_b2m['zero_shot_retention_rate_pct']:.1f}%** | {transfer_b2m['generalization_penalty_delta_f1']:+.4f} |
| **Meander -> B-line (Stealth -> Targeted)** | {transfer_m2b['in_distribution']['macro_f1']:.4f} | **{transfer_m2b['zero_shot_ood']['macro_f1']:.4f}** | {transfer_m2b['in_distribution']['detection_rate_pct']:.1f}% | **{transfer_m2b['zero_shot_ood']['detection_rate_pct']:.1f}%** | **{transfer_m2b['zero_shot_retention_rate_pct']:.1f}%** | {transfer_m2b['generalization_penalty_delta_f1']:+.4f} |

---

## Track 5: MITRE ATT&CK Multi-Tier Host Localization

| MITRE ATT&CK Tier | Monitored Hosts | Mean Macro F1 | Mean AUROC | Mean FPR@95% Recall | Localization Health |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **User Tier (Initial Access)** | User1, User2, User3, User4 | **{mitre_results['tier_summary']['user_tier']['mean_macro_f1']:.4f}** | **{mitre_results['tier_summary']['user_tier']['mean_auroc']:.4f}** | {mitre_results['tier_summary']['user_tier']['mean_fpr_at_95_recall']:.4f} | Strong |
| **Enterprise Tier (Lateral Movement)** | Enterprise0, Enterprise1, Enterprise2 | **{mitre_results['tier_summary']['enterprise_tier']['mean_macro_f1']:.4f}** | **{mitre_results['tier_summary']['enterprise_tier']['mean_auroc']:.4f}** | {mitre_results['tier_summary']['enterprise_tier']['mean_fpr_at_95_recall']:.4f} | Robust |
| **Operational Tier (Crown Jewels)** | Op_Server0, Op_Host0, Op_Host1, Op_Host2 | **{mitre_results['tier_summary']['operational_tier']['mean_macro_f1']:.4f}** | **{mitre_results['tier_summary']['operational_tier']['mean_auroc']:.4f}** | {mitre_results['tier_summary']['operational_tier']['mean_fpr_at_95_recall']:.4f} | High Precision |
"""

    scorecard_path = output_dir / "SCORECARD.md"
    with open(scorecard_path, "w", encoding="utf-8") as f:
        f.write(scorecard_md)
    print(f"[+] Scorecard written to {scorecard_path}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Cyber-JEPA Phase 7 Benchmark")
    parser.add_argument("--scale", type=str, default="13", help="Network scale setting to benchmark (default: 13)")
    parser.add_argument("--device", type=str, default="cuda", help="Execution device (cuda or cpu)")
    parser.add_argument("--seed", type=int, default=1001, help="Random seed (default: 1001)")
    parser.add_argument("--output_dir", type=str, default="experiments/phase7", help="Output directory")
    args = parser.parse_args()

    run_phase7_pipeline(
        scale=args.scale,
        device_name=args.device,
        seed=args.seed,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()

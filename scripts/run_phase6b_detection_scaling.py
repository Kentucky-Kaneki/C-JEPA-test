"""
Cyber-JEPA Phase 6B: Scaled Red Agent Detection Benchmark (5 to 500 Hosts).

Evaluates Flat Vector JEPA across 8 network scale settings:
- 5 hosts (20 dims) - Micro Branch Subnet
- 10 hosts (40 dims) - Core Subnet Tier
- 13 hosts (52 dims) - CybORG Scenario 1b Standard
- 25 hosts (100 dims) - Multi-Subnet Department
- 50 hosts (200 dims) - CAGE-4 Enterprise Scale
- 100 hosts (400 dims) - Campus Division Network
- 250 hosts (1000 dims) - Large Multi-Building Enterprise
- 500 hosts (2000 dims) - Global Corporate HQ / University

Assesses:
1. Operational Attack Accounting: Total Attacks, Attacks Detected (TP), Attacks Missed (FN),
   Detection Rate %, False Alarm Rate %, Precision %.
2. Adversary Tactic Breakdown: B-line (targeted) vs. Meander (exploratory).
3. Incident-Level Early Detection: Incident coverage % and mean lead steps.
4. Academic Grounded Metrics: Macro F1 (95% CI), Balanced Acc, AUROC, PR-AUC, FPR@95% Recall.
5. Anti-Shortcut / "Clever Hans" Audits: Majority baseline, Representation gain vs. raw features,
   Dynamic vs. static transition breakdown, Non-parametric k-NN clustering.
6. Anticipation Gap: Prediction rollout lead-time detection (z_hat_{t+k} vs. z_{t+k}).
7. Latent Geometry: Effective Rank (H) and Wang-Isola Uniformity.
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any

# Filter cosmetic framework warnings from polluting stderr
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Ensure deterministic CuBLAS algorithms and src on path
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader

from cyber_jepa.data.dataset import CyberJEPADataset, generate_group_splits
from cyber_jepa.evaluation.detection_benchmarks import (
    evaluate_adversary_policy_breakdown,
    evaluate_detection_metrics,
    evaluate_incident_early_detection,
    run_clever_hans_audit,
)
from cyber_jepa.evaluation.metrics import (
    compute_effective_rank,
    compute_wang_isola_uniformity,
)
from cyber_jepa.evaluation.scale_benchmarks import (
    SCALE_SPECS,
    ScaledDatasetWrapper,
    benchmark_inference_latency_and_memory,
)
from cyber_jepa.models.jepa import CyberJEPA
from cyber_jepa.representations.flat import FlatVectorRepresentation
from cyber_jepa.training.trainer import Trainer
from cyber_jepa.utils.reproducibility import seed_worker, set_deterministic_seed


def to_serializable(val: Any) -> Any:
    """Recursively convert non-serializable objects (slice, numpy, torch, etc.) to JSON-safe structures."""
    if isinstance(val, slice):
        return {"start": val.start, "stop": val.stop, "step": val.step}
    if isinstance(val, (np.integer, np.int64, np.int32)):
        return int(val)
    if isinstance(val, (np.floating, np.float32, np.float64)):
        return float(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, torch.Tensor):
        return val.detach().cpu().tolist()
    if isinstance(val, dict):
        return {str(k): to_serializable(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [to_serializable(v) for v in val]
    return val


def extract_features_and_latents(
    model: CyberJEPA,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    """Extract context latents z_t, predicted rollout latents z_hat_{t+k}, raw inputs, and metadata."""
    model.eval()
    pred_latents = []
    context_latents = []
    target_latents = []
    raw_obs_list = []
    labels_list = []
    traj_ids = []
    rms_deltas = []
    t_contexts = []

    with torch.no_grad():
        for batch in loader:
            hist = batch["history_flat"].to(device)
            actions = batch["action_seq"].to(device)
            target = batch["target_flat"].to(device)

            _, pred_z, tgt_z = model(hist, actions, target)
            _, ctx_z = model.encode_context(hist)

            pred_latents.append(pred_z.detach().cpu().numpy())
            context_latents.append(ctx_z.detach().cpu().numpy())
            target_latents.append(tgt_z.detach().cpu().numpy())

            # Use last history step observation as raw telemetry representation [B, obs_dim]
            raw_obs_list.append(hist[:, -1, :].detach().cpu().numpy())
            labels_list.append(batch["label"].numpy())

            if "trajectory_id" in batch:
                traj_ids.extend(batch["trajectory_id"])
            if "rms_delta" in batch:
                rms_deltas.append(batch["rms_delta"].numpy())
            if "t_context" in batch:
                t_contexts.append(batch["t_context"].numpy())

    concat_labels = np.concatenate(labels_list, axis=0)
    n_samples = len(concat_labels)

    return {
        "pred_latents": np.concatenate(pred_latents, axis=0),
        "context_latents": np.concatenate(context_latents, axis=0),
        "target_latents": np.concatenate(target_latents, axis=0),
        "raw_features": np.concatenate(raw_obs_list, axis=0),
        "labels": concat_labels,
        "trajectory_ids": traj_ids if len(traj_ids) == n_samples else ["traj_0"] * n_samples,
        "rms_deltas": np.concatenate(rms_deltas, axis=0) if rms_deltas else np.zeros(n_samples),
        "t_contexts": np.concatenate(t_contexts, axis=0) if t_contexts else np.arange(n_samples),
    }


def run_phase6b_benchmark(
    scales: list[str],
    epochs: int = 10,
    seed: int = 1001,
    device_name: str = "cuda",
    output_dir: Path = Path("experiments/phase6b"),
) -> dict[str, Any]:
    """Execute complete Phase 6B scaled red agent detection benchmark suite."""
    device = torch.device(device_name if torch.cuda.is_available() and device_name == "cuda" else "cpu")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*78}")
    print(f"Cyber-JEPA Phase 6B: Scaled Red Agent Detection Benchmark (5 to 500 Hosts)")
    print(f"Device: {device} | Seed: {seed} | Epochs: {epochs} | Scales: {scales}")
    print(f"{'='*78}\n", flush=True)

    gen = set_deterministic_seed(seed, warn_only=True)

    # 1. Shards & Strict Disjoint Group Splits (Zero Data Leakage Protocol)
    shards_dir = Path("data/shards")
    shard_paths = sorted([d for d in shards_dir.glob("*") if d.is_dir() and (d / "transitions.parquet").exists()])
    trans_list = [pd.read_parquet(d / "transitions.parquet") for d in shard_paths]
    all_trans = pd.concat(trans_list, ignore_index=True)
    all_groups = sorted(all_trans["split_group_id"].unique().tolist())
    splits = generate_group_splits(all_groups, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)

    print(f"[+] Loaded {len(shard_paths)} shards ({len(all_trans)} transitions).")
    print(f"[+] Train groups: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])} (Disjoint Episode-Level).\n")

    # Base datasets: fitted normalizers strictly on train split
    base_train_ds = CyberJEPADataset(shard_dirs=shard_paths, split_group_set=splits["train"], horizon=4, history_len=4, fit_normalizers=True)
    base_val_ds = CyberJEPADataset(shard_dirs=shard_paths, split_group_set=splits["val"], horizon=4, history_len=4, fit_normalizers=False, normalizer_stats=base_train_ds.normalizer_stats)
    base_test_ds = CyberJEPADataset(shard_dirs=shard_paths, split_group_set=splits["test"], horizon=4, history_len=4, fit_normalizers=False, normalizer_stats=base_train_ds.normalizer_stats)

    results: dict[str, Any] = {
        "metadata": {
            "device": str(device),
            "seed": seed,
            "epochs": epochs,
            "scales": scales,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "scales": {},
    }

    scorecard_rows = []

    for scale in scales:
        scale_key = str(scale)
        spec = SCALE_SPECS.get(scale_key, {
            "num_hosts": int(scale),
            "obs_dim": int(scale) * 4,
            "name": f"Scale {scale} ({scale} hosts)",
        })
        num_hosts = spec["num_hosts"]
        obs_dim = spec["obs_dim"]
        scale_name = spec["name"]

        print(f"\n{'-'*78}")
        print(f"BENCHMARK SCALE: {scale_key.upper()} ({scale_name} | {num_hosts} hosts | {obs_dim} features)")
        print(f"{'-'*78}")

        train_ds = ScaledDatasetWrapper(base_train_ds, scale=scale_key, target_hosts=num_hosts)
        val_ds = ScaledDatasetWrapper(base_val_ds, scale=scale_key, target_hosts=num_hosts)
        test_ds = ScaledDatasetWrapper(base_test_ds, scale=scale_key, target_hosts=num_hosts)

        median_thresh = train_ds.compute_median_dynamic_rms()
        train_sampler = train_ds.get_balanced_sampler(threshold=median_thresh, generator=gen)

        train_loader = DataLoader(train_ds, batch_size=64, sampler=train_sampler, worker_init_fn=seed_worker, generator=gen)
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

        # -------------------------------------------------------------
        # 1. Flat Vector JEPA Model Architecture & Optimization
        # -------------------------------------------------------------
        flat_encoder = FlatVectorRepresentation(obs_dim=obs_dim, hidden_dim=64, history_len=4)
        flat_jepa = CyberJEPA(
            online_encoder=flat_encoder,
            hidden_dim=64,
            max_horizon=4,
            loss_norm_mode="batch_center",
            vicreg_var_weight=1.0,
            vicreg_cov_weight=0.04,
        )

        run_dir = output_dir / f"scale_{scale_key}_flat"
        opt = torch.optim.AdamW(flat_jepa.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(train_loader) // 2, eta_min=1e-6)
        trainer = Trainer(
            model=flat_jepa,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=opt,
            scheduler=sched,
            run_dir=run_dir,
            device=device,
            max_epochs=epochs,
            min_epochs=min(6, epochs),
            patience=4,
            accum_steps=2,
            monitor_metric="val_pred_loss",
        )

        t_start = time.perf_counter()
        history = trainer.fit()
        train_time = float(time.perf_counter() - t_start)
        epochs_trained = len(history["train_loss"])
        print(f"[+] Model trained in {train_time:.2f}s ({epochs_trained} epochs).")

        # Load best checkpoint
        best_ckpt_path = run_dir / "best.pt"
        if best_ckpt_path.exists():
            ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
            flat_jepa.online_encoder.load_state_dict(ckpt["online_encoder"])
            flat_jepa.target_encoder.load_state_dict(ckpt["target_encoder"])
            flat_jepa.predictor.load_state_dict(ckpt["predictor"])
            if "action_encoder" in ckpt and hasattr(flat_jepa, "action_encoder"):
                flat_jepa.action_encoder.load_state_dict(ckpt["action_encoder"])

        # -------------------------------------------------------------
        # 2. Latency, Throughput & Memory Profiling
        # -------------------------------------------------------------
        print(f"[+] Profiling inference latency & memory...")
        prof_gpu = benchmark_inference_latency_and_memory(flat_jepa, (64, 4, obs_dim), device=device, num_warmup=25, num_iters=200)
        prof_cpu = benchmark_inference_latency_and_memory(flat_jepa, (64, 4, obs_dim), device=torch.device("cpu"), num_warmup=10, num_iters=50)

        # -------------------------------------------------------------
        # 3. Latent Representation & Feature Extraction
        # -------------------------------------------------------------
        print(f"[+] Extracting train/test latent representations...")
        train_data = extract_features_and_latents(flat_jepa, train_loader, device)
        test_data = extract_features_and_latents(flat_jepa, test_loader, device)

        # -------------------------------------------------------------
        # 4. Latent Geometry Diagnostics (Wang-Isola Uniformity & Effective Rank)
        # -------------------------------------------------------------
        test_z_torch = torch.tensor(test_data["context_latents"], dtype=torch.float32)
        eff_rank = float(compute_effective_rank(test_z_torch))
        uniformity = float(compute_wang_isola_uniformity(test_z_torch))

        # -------------------------------------------------------------
        # 5. Primary Detection Probe Training & Evaluation
        # -------------------------------------------------------------
        print(f"[+] Evaluating operational attack accounting & detection metrics...")
        clf = LogisticRegression(C=1.0, max_iter=500, random_state=seed)
        clf.fit(train_data["context_latents"], train_data["labels"])

        test_preds = clf.predict(test_data["context_latents"])
        test_probs = clf.predict_proba(test_data["context_latents"])[:, 1]

        detection_metrics = evaluate_detection_metrics(test_data["labels"], test_preds, test_probs, seed=seed)
        policy_breakdown = evaluate_adversary_policy_breakdown(test_data["labels"], test_preds, test_data["trajectory_ids"])
        incident_metrics = evaluate_incident_early_detection(
            test_data["labels"], test_preds, test_data["trajectory_ids"], test_data["t_contexts"]
        )

        # -------------------------------------------------------------
        # 6. Anti-Shortcut / "Clever Hans" Audits
        # -------------------------------------------------------------
        print(f"[+] Running Anti-Shortcut / Clever Hans Audits...")
        clever_hans = run_clever_hans_audit(
            train_latents=train_data["context_latents"],
            train_labels=train_data["labels"],
            test_latents=test_data["context_latents"],
            test_labels=test_data["labels"],
            raw_train_features=train_data["raw_features"],
            raw_test_features=test_data["raw_features"],
            test_rms_deltas=test_data["rms_deltas"],
            seed=seed,
        )

        # -------------------------------------------------------------
        # 7. Anticipation Gap (Rollout Prediction z_hat_{t+4} vs. Target z_{t+4})
        # -------------------------------------------------------------
        clf_pred = LogisticRegression(C=1.0, max_iter=500, random_state=seed)
        clf_pred.fit(train_data["pred_latents"], train_data["labels"])
        test_pred_preds = clf_pred.predict(test_data["pred_latents"])
        test_pred_probs = clf_pred.predict_proba(test_data["pred_latents"])[:, 1]
        anticipation_metrics = evaluate_detection_metrics(test_data["labels"], test_pred_preds, test_pred_probs, seed=seed)

        anticipation_gap = {
            "anticipation_macro_f1": anticipation_metrics["macro_f1"],
            "anticipation_auroc": anticipation_metrics["auroc"],
            "delta_f1_vs_realtime": float(anticipation_metrics["macro_f1"] - detection_metrics["macro_f1"]),
            "delta_auroc_vs_realtime": float(anticipation_metrics["auroc"] - detection_metrics["auroc"]),
        }

        # -------------------------------------------------------------
        # 8. Compile Scale Results
        # -------------------------------------------------------------
        scale_summary = {
            "spec": spec,
            "hardware_profiling": {
                "num_params": prof_gpu["num_params"],
                "model_size_mb": prof_gpu["model_size_mb"],
                "train_time_sec": train_time,
                "epochs_trained": epochs_trained,
                "gpu_mean_latency_ms": prof_gpu["mean_latency_ms"],
                "gpu_p50_latency_ms": prof_gpu["p50_latency_ms"],
                "gpu_p95_latency_ms": prof_gpu["p95_latency_ms"],
                "gpu_throughput_fps": prof_gpu["throughput_samples_per_sec"],
                "gpu_peak_vram_mb": prof_gpu["peak_vram_mb"],
                "cpu_mean_latency_ms": prof_cpu["mean_latency_ms"],
                "cpu_p50_latency_ms": prof_cpu["p50_latency_ms"],
            },
            "latent_geometry": {
                "effective_rank": eff_rank,
                "wang_isola_uniformity": uniformity,
            },
            "operational_detection": detection_metrics,
            "adversary_policy_breakdown": policy_breakdown,
            "incident_early_detection": incident_metrics,
            "anti_shortcut_audits": clever_hans,
            "anticipation_gap": anticipation_gap,
        }

        results["scales"][scale_key] = scale_summary

        # Print scale console summary
        print(f"\n--- SCALE {scale_key.upper()} ({num_hosts} hosts) OPERATIONAL DETECTION ---")
        print(f"Total Attacks Evaluated: {detection_metrics['total_attacks']} | Clean Steps: {detection_metrics['clean_steps']}")
        print(f"Attacks Caught: {detection_metrics['attacks_detected']} ({detection_metrics['attack_detection_rate_pct']:.2f}%) | Missed: {detection_metrics['attacks_missed']}")
        print(f"False Alarms: {detection_metrics['false_alarms']} ({detection_metrics['false_alarm_rate_pct']:.2f}%) | Precision: {detection_metrics['precision_pct']:.2f}%")
        print(f"Macro F1: {detection_metrics['macro_f1']:.4f} [95% CI: {detection_metrics['macro_f1_ci_95'][0]:.4f}, {detection_metrics['macro_f1_ci_95'][1]:.4f}]")
        print(f"AUROC: {detection_metrics['auroc']:.4f} | PR-AUC: {detection_metrics['pr_auc']:.4f} | FPR@95: {detection_metrics['fpr_at_95_recall']:.4f}")
        print(f"B-line Detection: {policy_breakdown['bline_targeted']['detection_rate_pct']:.2f}% | Meander Detection: {policy_breakdown['meander_stealth']['detection_rate_pct']:.2f}%")
        print(f"Incident Early Det: {incident_metrics['timely_detected_incidents']}/{incident_metrics['total_attack_incidents']} ({incident_metrics['timely_coverage_pct']:.1f}%) | Mean Lead: {incident_metrics['mean_lead_steps']:+.1f} steps")
        print(f"Clean Traj False Alarms: {incident_metrics['false_alarm_trajectories']}/{incident_metrics['clean_trajectories']} ({incident_metrics['trajectory_false_alarm_rate_pct']:.1f}%)")
        print(f"Representation Gain (vs Raw): {clever_hans['representation_gain_f1']:+.4f} F1 | k-NN F1: {clever_hans['knn_nonparametric_macro_f1']:.4f}")
        print(f"Latency: {prof_gpu['mean_latency_ms']:.3f}ms GPU ({prof_gpu['throughput_samples_per_sec']:.0f} FPS) | VRAM: {prof_gpu['peak_vram_mb']:.1f} MB")

        scorecard_rows.append({
            "Scale": f"{num_hosts} hosts",
            "Obs Dim": obs_dim,
            "Latency (ms)": f"{prof_gpu['mean_latency_ms']:.2f}",
            "Attacks Caught / Tot": f"{detection_metrics['attacks_detected']} / {detection_metrics['total_attacks']}",
            "Det Rate (%)": f"{detection_metrics['attack_detection_rate_pct']:.1f}%",
            "False Alarms (%)": f"{detection_metrics['false_alarm_rate_pct']:.1f}%",
            "Macro F1": f"{detection_metrics['macro_f1']:.4f}",
            "AUROC": f"{detection_metrics['auroc']:.4f}",
            "PR-AUC": f"{detection_metrics['pr_auc']:.4f}",
            "FPR@95": f"{detection_metrics['fpr_at_95_recall']:.4f}",
            "B-line Det (%)": f"{policy_breakdown['bline_targeted']['detection_rate_pct']:.1f}%",
            "Meander Det (%)": f"{policy_breakdown['meander_stealth']['detection_rate_pct']:.1f}%",
            "Incident Cov (%)": f"{incident_metrics['incident_coverage_pct']:.1f}%",
            "Mean Lead": f"{incident_metrics['mean_lead_steps']:+.1f}",
            "Repr Gain": f"{clever_hans['representation_gain_f1']:+.4f}",
        })

    # Save JSON results
    out_json = output_dir / "phase6b_detection_scaling.json"
    with open(out_json, "w") as f:
        json.dump(to_serializable(results), f, indent=2)
    print(f"\n[+] Full Phase 6B Detection Benchmark saved to {out_json}")

    # Generate Markdown Scorecard
    out_md = output_dir / "SCORECARD.md"
    scorecard_df = pd.DataFrame(scorecard_rows)
    with open(out_md, "w") as f:
        f.write("# Phase 6B: Scaled Red Agent Detection Scorecard (5 to 500 Hosts)\n\n")
        f.write(scorecard_df.to_markdown(index=False))
        f.write("\n")
    print(f"[+] Scorecard written to {out_md}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Cyber-JEPA Phase 6B Scaled Red Agent Detection Benchmark")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs per model")
    parser.add_argument("--seed", type=int, default=1001, help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cuda", help="Target device ('cuda' or 'cpu')")
    parser.add_argument("--scales", nargs="+", default=["5", "10", "13", "25", "50", "100", "250", "500"], help="Scales to evaluate")
    parser.add_argument("--output_dir", type=str, default="experiments/phase6b", help="Output directory")

    args = parser.parse_args()
    run_phase6b_benchmark(
        scales=args.scales,
        epochs=args.epochs,
        seed=args.seed,
        device_name=args.device,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()

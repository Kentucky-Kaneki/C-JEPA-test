"""
Cyber-JEPA Phase 6A: Multi-Scale Tokenized vs. Flat JEPA Benchmark Runner.

Executes 'The Bet' across 3 standardized network scales:
- Small: 3 hosts (12 dims, Enterprise subnet)
- Medium: 13 hosts (52 dims, Scenario 1b standard)
- Large: 45 hosts (180 dims, CAGE-4 9-subnet enterprise standard)

Evaluates:
1. Retraining & convergence time (seconds on CUDA).
2. Inference latency (ms per step: mean, p50, p95, throughput) on GPU & CPU.
3. Peak GPU VRAM memory footprint (MB) & parameter scaling.
4. Downstream detection efficacy: Macro F1, 95% bootstrap CI, AUROC, FPR@95% Recall.
5. Self-Attention interpretability: Shannon entropy, entropy collapse under attack,
   and attention focus ratio on compromised hosts.
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
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader

from cyber_jepa.data.dataset import CyberJEPADataset, generate_group_splits
from cyber_jepa.evaluation.attention_analysis import (
    analyze_attack_attention_focus,
    compute_attention_entropy,
    extract_host_attention_distribution,
)
from cyber_jepa.evaluation.probes import compute_bootstrap_ci
from cyber_jepa.evaluation.scale_benchmarks import (
    SCALE_SPECS,
    ScaledDatasetWrapper,
    benchmark_inference_latency_and_memory,
)
from cyber_jepa.models.jepa import CyberJEPA
from cyber_jepa.representations.flat import FlatVectorRepresentation
from cyber_jepa.representations.host import ModernHostTokenRepresentation
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


def compute_fpr_at_recall(labels: np.ndarray, probs: np.ndarray, target_recall: float = 0.95) -> float:
    """Compute False Positive Rate at target recall (e.g. 95%) adhering to Arp et al. (2022)."""
    if len(np.unique(labels)) < 2:
        return 0.0
    fpr, tpr, _ = roc_curve(labels, probs)
    idx = np.where(tpr >= target_recall)[0]
    return float(fpr[idx[0]]) if len(idx) > 0 else 1.0


def extract_latents(
    model: CyberJEPA,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract future predictions z_hat_{t+k}, context z_t, labels, and host compromise matrix."""
    model.eval()
    pred_latents = []
    context_latents = []
    labels = []
    host_comp_list = []

    with torch.no_grad():
        for batch in loader:
            hist = batch["history_flat"].to(device)
            actions = batch["action_seq"].to(device)
            target = batch["target_flat"].to(device)

            _, pred_z, _ = model(hist, actions, target)
            _, ctx_z = model.encode_context(hist)

            pred_latents.append(pred_z.detach().cpu().numpy())
            context_latents.append(ctx_z.detach().cpu().numpy())
            labels.append(batch["label"].numpy())
            if "host_compromised" in batch:
                host_comp_list.append(batch["host_compromised"].numpy())

    P = np.concatenate(pred_latents, axis=0)
    C = np.concatenate(context_latents, axis=0)
    Y = np.concatenate(labels, axis=0)
    H = np.concatenate(host_comp_list, axis=0) if host_comp_list else np.zeros((len(Y), 1))
    return P, C, Y, H


def evaluate_detection_probe(
    train_latents: np.ndarray,
    train_labels: np.ndarray,
    test_latents: np.ndarray,
    test_labels: np.ndarray,
    seed: int = 42,
) -> dict[str, Any]:
    """Train linear probe on frozen latents and evaluate classification metrics."""
    if len(np.unique(train_labels)) < 2 or len(np.unique(test_labels)) < 2:
        return {
            "macro_f1": 0.0,
            "macro_f1_ci_95": [0.0, 0.0],
            "auroc": 0.5,
            "balanced_accuracy": 0.5,
            "fpr_at_95_recall": 1.0,
        }

    clf = LogisticRegression(C=1.0, max_iter=500, random_state=seed)
    clf.fit(train_latents, train_labels)

    preds = clf.predict(test_latents)
    probs = clf.predict_proba(test_latents)[:, 1]

    f1_pt, f1_low, f1_high = compute_bootstrap_ci(
        test_labels, preds, lambda y, p: f1_score(y, p, average="macro", zero_division=0), seed=seed
    )
    auroc = float(roc_auc_score(test_labels, probs))
    bal_acc = float(balanced_accuracy_score(test_labels, preds))
    fpr_95 = compute_fpr_at_recall(test_labels, probs, target_recall=0.95)

    return {
        "macro_f1": float(f1_pt),
        "macro_f1_ci_95": [float(f1_low), float(f1_high)],
        "auroc": auroc,
        "balanced_accuracy": bal_acc,
        "fpr_at_95_recall": fpr_95,
    }


def run_phase6a_benchmark(
    scales: list[str],
    epochs: int = 15,
    seed: int = 1001,
    device_name: str = "cuda",
    output_dir: Path = Path("experiments/phase6a"),
) -> dict[str, Any]:
    """Execute complete Phase 6A multi-scale token vs flat benchmark suite."""
    device = torch.device(device_name if torch.cuda.is_available() and device_name == "cuda" else "cpu")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*75}")
    print(f"Cyber-JEPA Phase 6A: Tokenized vs. Flat Multi-Scale Scaling Benchmark")
    print(f"Device: {device} | Seed: {seed} | Epochs: {epochs} | Scales: {scales}")
    print(f"{'='*75}\n", flush=True)

    gen = set_deterministic_seed(seed, warn_only=True)

    # 1. Shards & Group Splits
    shards_dir = Path("data/shards")
    shard_paths = sorted([d for d in shards_dir.glob("*") if d.is_dir() and (d / "transitions.parquet").exists()])
    trans_list = [pd.read_parquet(d / "transitions.parquet") for d in shard_paths]
    all_trans = pd.concat(trans_list, ignore_index=True)
    all_groups = sorted(all_trans["split_group_id"].unique().tolist())
    splits = generate_group_splits(all_groups, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)

    print(f"[+] Loaded {len(shard_paths)} shards ({len(all_trans)} transitions).")
    print(f"[+] Train groups: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}.\n")

    # Base datasets (with normalizer fit on train)
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

    for scale in scales:
        spec = SCALE_SPECS[scale]
        num_hosts = spec["num_hosts"]
        obs_dim = spec["obs_dim"]
        scale_name = spec["name"]

        print(f"\n{'-'*75}")
        print(f"BENCHMARK SCALE: {scale.upper()} ({scale_name} | {num_hosts} hosts | {obs_dim} features)")
        print(f"{'-'*75}")

        train_ds = ScaledDatasetWrapper(base_train_ds, scale=scale)
        val_ds = ScaledDatasetWrapper(base_val_ds, scale=scale)
        test_ds = ScaledDatasetWrapper(base_test_ds, scale=scale)

        median_thresh = train_ds.compute_median_dynamic_rms()
        train_sampler = train_ds.get_balanced_sampler(threshold=median_thresh, generator=gen)

        train_loader = DataLoader(train_ds, batch_size=64, sampler=train_sampler, worker_init_fn=seed_worker, generator=gen)
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

        scale_results: dict[str, Any] = {
            "spec": spec,
            "flat_jepa": {},
            "token_jepa": {},
            "comparison": {},
        }

        # -------------------------------------------------------------
        # (A) Flat Vector JEPA Pipeline
        # -------------------------------------------------------------
        print(f"\n[+] Training Flat Vector JEPA (obs_dim={obs_dim}) on {device}...")
        flat_encoder = FlatVectorRepresentation(obs_dim=obs_dim, hidden_dim=64, history_len=4)
        flat_jepa = CyberJEPA(
            online_encoder=flat_encoder,
            hidden_dim=64,
            max_horizon=4,
            loss_norm_mode="batch_center",
            vicreg_var_weight=1.0,
            vicreg_cov_weight=0.04,
        )

        flat_run_dir = output_dir / f"{scale}_flat"
        flat_opt = torch.optim.AdamW(flat_jepa.parameters(), lr=1e-3, weight_decay=1e-4)
        flat_sched = torch.optim.lr_scheduler.CosineAnnealingLR(flat_opt, T_max=epochs * len(train_loader) // 2, eta_min=1e-6)
        flat_trainer = Trainer(
            model=flat_jepa,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=flat_opt,
            scheduler=flat_sched,
            run_dir=flat_run_dir,
            device=device,
            max_epochs=epochs,
            min_epochs=min(6, epochs),
            patience=4,
            accum_steps=2,
            monitor_metric="val_pred_loss",
        )

        t_start = time.perf_counter()
        flat_history = flat_trainer.fit()
        flat_train_time = float(time.perf_counter() - t_start)
        print(f"[+] Flat JEPA trained in {flat_train_time:.2f}s ({len(flat_history['train_loss'])} epochs).")

        # Load best checkpoint
        best_flat = flat_run_dir / "best.pt"
        if best_flat.exists():
            ckpt = torch.load(best_flat, map_location=device, weights_only=False)
            flat_jepa.online_encoder.load_state_dict(ckpt["online_encoder"])
            flat_jepa.target_encoder.load_state_dict(ckpt["target_encoder"])
            flat_jepa.predictor.load_state_dict(ckpt["predictor"])
            if "action_encoder" in ckpt and hasattr(flat_jepa, "action_encoder"):
                flat_jepa.action_encoder.load_state_dict(ckpt["action_encoder"])

        # Latency & Memory Profile
        print(f"[+] Profiling Flat JEPA inference latency & memory...")
        flat_prof_gpu = benchmark_inference_latency_and_memory(flat_jepa, (64, 4, obs_dim), device=device, num_warmup=25, num_iters=200)
        flat_prof_cpu = benchmark_inference_latency_and_memory(flat_jepa, (64, 4, obs_dim), device=torch.device("cpu"), num_warmup=10, num_iters=50)

        # Probes & Detection
        train_P, _, train_Y, _ = extract_latents(flat_jepa, train_loader, device)
        test_P, _, test_Y, _ = extract_latents(flat_jepa, test_loader, device)
        flat_detection = evaluate_detection_probe(train_P, train_Y, test_P, test_Y, seed=seed)

        final_flat_val_loss = (
            float(flat_history["val_pred_loss"][-1])
            if flat_history.get("val_pred_loss")
            else (float(flat_history["val_loss"][-1]) if flat_history.get("val_loss") else 0.0)
        )

        scale_results["flat_jepa"] = {
            "train_time_sec": flat_train_time,
            "epochs_trained": len(flat_history["train_loss"]),
            "final_val_pred_loss": final_flat_val_loss,
            "num_params": flat_prof_gpu["num_params"],
            "model_size_mb": flat_prof_gpu["model_size_mb"],
            "gpu_mean_latency_ms": flat_prof_gpu["mean_latency_ms"],
            "gpu_p50_latency_ms": flat_prof_gpu["p50_latency_ms"],
            "gpu_p95_latency_ms": flat_prof_gpu["p95_latency_ms"],
            "gpu_throughput_fps": flat_prof_gpu["throughput_samples_per_sec"],
            "gpu_peak_vram_mb": flat_prof_gpu["peak_vram_mb"],
            "cpu_mean_latency_ms": flat_prof_cpu["mean_latency_ms"],
            "cpu_p50_latency_ms": flat_prof_cpu["p50_latency_ms"],
            "detection": flat_detection,
        }

        # -------------------------------------------------------------
        # (B) Modernized Host Token JEPA Pipeline
        # -------------------------------------------------------------
        print(f"\n[+] Training Modern Host Token JEPA (num_hosts={num_hosts}) on {device}...")
        token_encoder = ModernHostTokenRepresentation(num_hosts=num_hosts, features_per_host=4, hidden_dim=64, history_len=4)
        token_jepa = CyberJEPA(
            online_encoder=token_encoder,
            hidden_dim=64,
            max_horizon=4,
            loss_norm_mode="batch_center",
            vicreg_var_weight=1.0,
            vicreg_cov_weight=0.04,
        )

        token_run_dir = output_dir / f"{scale}_token"
        token_opt = torch.optim.AdamW(token_jepa.parameters(), lr=1e-3, weight_decay=1e-4)
        token_sched = torch.optim.lr_scheduler.CosineAnnealingLR(token_opt, T_max=epochs * len(train_loader) // 2, eta_min=1e-6)
        token_trainer = Trainer(
            model=token_jepa,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=token_opt,
            scheduler=token_sched,
            run_dir=token_run_dir,
            device=device,
            max_epochs=epochs,
            min_epochs=min(6, epochs),
            patience=4,
            accum_steps=2,
            monitor_metric="val_pred_loss",
        )

        t_start = time.perf_counter()
        token_history = token_trainer.fit()
        token_train_time = float(time.perf_counter() - t_start)
        print(f"[+] Token JEPA trained in {token_train_time:.2f}s ({len(token_history['train_loss'])} epochs).")

        # Load best checkpoint
        best_token = token_run_dir / "best.pt"
        if best_token.exists():
            ckpt = torch.load(best_token, map_location=device, weights_only=False)
            token_jepa.online_encoder.load_state_dict(ckpt["online_encoder"])
            token_jepa.target_encoder.load_state_dict(ckpt["target_encoder"])
            token_jepa.predictor.load_state_dict(ckpt["predictor"])
            if "action_encoder" in ckpt and hasattr(token_jepa, "action_encoder"):
                token_jepa.action_encoder.load_state_dict(ckpt["action_encoder"])

        # Latency & Memory Profile
        print(f"[+] Profiling Token JEPA inference latency & memory...")
        token_prof_gpu = benchmark_inference_latency_and_memory(token_jepa, (64, 4, obs_dim), device=device, num_warmup=25, num_iters=200)
        token_prof_cpu = benchmark_inference_latency_and_memory(token_jepa, (64, 4, obs_dim), device=torch.device("cpu"), num_warmup=10, num_iters=50)

        # Probes & Detection
        train_P_tok, _, train_Y_tok, _ = extract_latents(token_jepa, train_loader, device)
        test_P_tok, _, test_Y_tok, test_H_tok = extract_latents(token_jepa, test_loader, device)
        token_detection = evaluate_detection_probe(train_P_tok, train_Y_tok, test_P_tok, test_Y_tok, seed=seed)

        # -------------------------------------------------------------
        # (C) Self-Attention Analysis & Focus Quantification
        # -------------------------------------------------------------
        print(f"[+] Performing Self-Attention Analysis & Host Attribution across test set...")
        all_test_hist = torch.cat([b["history_flat"] for b in test_loader], dim=0)

        # Extract attention distributions (batched internally in chunks of 64)
        host_probs, raw_attn = extract_host_attention_distribution(token_jepa, all_test_hist, device, batch_size=64)
        attention_analysis = analyze_attack_attention_focus(host_probs, test_H_tok)

        # Mean host attention probability across clean vs attack steps
        any_attack = np.any(test_H_tok > 0, axis=-1)
        mean_attn_clean = np.mean(host_probs[~any_attack], axis=0).tolist() if np.sum(~any_attack) > 0 else [0.0] * num_hosts
        mean_attn_attack = np.mean(host_probs[any_attack], axis=0).tolist() if np.sum(any_attack) > 0 else [0.0] * num_hosts

        final_token_val_loss = (
            float(token_history["val_pred_loss"][-1])
            if token_history.get("val_pred_loss")
            else (float(token_history["val_loss"][-1]) if token_history.get("val_loss") else 0.0)
        )

        scale_results["token_jepa"] = {
            "train_time_sec": token_train_time,
            "epochs_trained": len(token_history["train_loss"]),
            "final_val_pred_loss": final_token_val_loss,
            "num_params": token_prof_gpu["num_params"],
            "model_size_mb": token_prof_gpu["model_size_mb"],
            "gpu_mean_latency_ms": token_prof_gpu["mean_latency_ms"],
            "gpu_p50_latency_ms": token_prof_gpu["p50_latency_ms"],
            "gpu_p95_latency_ms": token_prof_gpu["p95_latency_ms"],
            "gpu_throughput_fps": token_prof_gpu["throughput_samples_per_sec"],
            "gpu_peak_vram_mb": token_prof_gpu["peak_vram_mb"],
            "cpu_mean_latency_ms": token_prof_cpu["mean_latency_ms"],
            "cpu_p50_latency_ms": token_prof_cpu["p50_latency_ms"],
            "detection": token_detection,
            "attention_analysis": attention_analysis,
            "mean_attention_clean_per_host": mean_attn_clean,
            "mean_attention_attack_per_host": mean_attn_attack,
        }

        # Comparative Summary Ratios
        lat_ratio = token_prof_gpu["mean_latency_ms"] / max(1e-6, flat_prof_gpu["mean_latency_ms"])
        train_time_ratio = token_train_time / max(1e-6, flat_train_time)
        f1_diff = flat_detection["macro_f1"] - token_detection["macro_f1"]
        auroc_diff = flat_detection["auroc"] - token_detection["auroc"]

        scale_results["comparison"] = {
            "latency_slowdown_ratio": lat_ratio,
            "training_time_ratio": train_time_ratio,
            "f1_advantage_flat": f1_diff,
            "auroc_advantage_flat": auroc_diff,
            "attention_focus_ratio": attention_analysis["attention_focus_ratio"],
            "entropy_collapse_bits": attention_analysis["entropy_reduction_under_attack"],
        }

        print(f"\n--- {scale.upper()} SUMMARY ---")
        print(f"Flat JEPA:  Train={flat_train_time:.1f}s | Latency={flat_prof_gpu['mean_latency_ms']:.3f}ms | F1={flat_detection['macro_f1']:.4f} | AUROC={flat_detection['auroc']:.4f} | FPR@95={flat_detection['fpr_at_95_recall']:.4f}")
        print(f"Token JEPA: Train={token_train_time:.1f}s | Latency={token_prof_gpu['mean_latency_ms']:.3f}ms | F1={token_detection['macro_f1']:.4f} | AUROC={token_detection['auroc']:.4f} | FPR@95={token_detection['fpr_at_95_recall']:.4f}")
        print(f"Token Attention Focus Ratio: {attention_analysis['attention_focus_ratio']:.2f}x | Entropy Reduction: {attention_analysis['entropy_reduction_under_attack']:.3f} bits")
        print(f"Flat vs Token Speedup: {lat_ratio:.2f}x faster inference | F1 Advantage: {f1_diff:+.4f}")

        results["scales"][scale] = scale_results

    # Save final JSON results
    out_file = output_dir / "phase6a_results.json"
    with open(out_file, "w") as f:
        json.dump(to_serializable(results), f, indent=2)
    print(f"\n[+] Full Phase 6A Multi-Scale Benchmark saved to {out_file}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Cyber-JEPA Phase 6A Multi-Scale Token vs Flat Benchmark")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs per model per scale")
    parser.add_argument("--seed", type=int, default=1001, help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cuda", help="Target execution device ('cuda' or 'cpu')")
    parser.add_argument("--scales", nargs="+", default=["small", "medium", "large"], help="Scales to evaluate")
    parser.add_argument("--output_dir", type=str, default="experiments/phase6a", help="Output directory")

    args = parser.parse_args()
    run_phase6a_benchmark(
        scales=args.scales,
        epochs=args.epochs,
        seed=args.seed,
        device_name=args.device,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()

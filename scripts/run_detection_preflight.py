"""
Unified Pre-Flight Detection Runner & Multi-Seed Benchmark for Cyber-JEPA.

Grounded in Industry & Academic Standards:
1. NIST SP 800-61 Rev. 2: Precursor (future rollout z_hat_{t+k}) vs Indicator (current telemetry z_t) Anticipation Gap.
2. MITRE ATT&CK v14: Subnet-tiered compromise localization (User, Enterprise, Operational tiers).
3. Arp et al. (USENIX Security 2022) & Sommer-Paxson (IEEE S&P 2010): Base-rate fallacy avoidance, natural test distribution, FPR@95% recall, and multi-seed statistical aggregation (N=3 seeds: 1001, 2003, 3005).
"""

import sys, os, json, argparse
from typing import Any
from pathlib import Path

# Ensure deterministic CuBLAS algorithms and src on path
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from cyber_jepa.utils.reproducibility import set_deterministic_seed, seed_worker
from cyber_jepa.data.dataset import (
    CyberJEPADataset,
    MONITORED_HOSTS,
    generate_group_splits,
    generate_policy_transfer_splits,
)
from cyber_jepa.representations.flat import FlatVectorRepresentation
from cyber_jepa.models.jepa import CyberJEPA
from cyber_jepa.training.trainer import Trainer
from cyber_jepa.evaluation.probes import LinearProbeEvaluator
from cyber_jepa.evaluation.diagnostics import compute_latent_geometry_diagnostics
from cyber_jepa.evaluation.phase5_diagnostics import (
    compute_spectral_decay_alpha,
    compute_wang_isola_uniformity,
    compute_fisher_discriminant_ratio,
    evaluate_knn_probe,
    evaluate_action_sensitivity,
    evaluate_predictive_skill_and_persistence,
)
from cyber_jepa.evaluation.anticipation_diagnostics import evaluate_anticipation_gap
from cyber_jepa.evaluation.localization_probes import evaluate_multi_host_localization


def extract_preflight_latents(
    model: CyberJEPA,
    loader: DataLoader,
    device: torch.device,
    dynamic_threshold: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract comprehensive latent representations:
    - P: Predicted future latents z_hat_{t+k} [N, D]
    - C: Current context latents z_t [N, D] (online encoder on O_{t-3:t})
    - T: Target future latents z_{t+k} [N, D] (target encoder on O_{t+k})
    - Pers: Persistence latents z_t^{target} [N, D] (target encoder on O_t)
    - Y: Crown jewel binary labels [N]
    - H: Multi-host compromise matrix [N, num_hosts]
    - Dyn: Dynamic transition boolean flags [N]
    """
    model.eval()
    pred_latents = []
    curr_latents = []
    target_latents = []
    persistence_latents = []
    labels = []
    host_compromised_list = []
    dynamic_flags = []

    with torch.no_grad():
        for batch in loader:
            hist = batch["history_flat"].to(device)
            actions = batch["action_seq"].to(device)
            target = batch["target_flat"].to(device)

            # 1. Forward pass: future prediction & target encoding
            _, pred_z, target_z = model(hist, actions, target)

            # 2. Current context latent z_t from online encoder + aggregator
            _, curr_z = model.encode_context(hist)

            # 3. Persistence baseline latent = target encoder of last history frame O_t
            t0_obs = hist[:, -1, :] # [B, 52]
            pers_z = model._encode_target_single_frame(t0_obs)

            pred_latents.append(pred_z.detach().cpu().numpy())
            curr_latents.append(curr_z.detach().cpu().numpy())
            target_latents.append(target_z.detach().cpu().numpy())
            persistence_latents.append(pers_z.detach().cpu().numpy())
            labels.append(batch["label"].numpy())

            if "host_compromised" in batch:
                host_compromised_list.append(batch["host_compromised"].numpy())

            if "rms_delta" in batch:
                rms = batch["rms_delta"]
                if isinstance(rms, torch.Tensor):
                    dyn = (rms >= dynamic_threshold).cpu().numpy()
                else:
                    dyn = np.asarray(rms) >= dynamic_threshold
                dynamic_flags.append(dyn)
            else:
                diff = (target.cpu() - hist[:, -1, :].cpu()).numpy()
                rms = np.sqrt(np.mean(diff ** 2, axis=1))
                dynamic_flags.append(rms >= dynamic_threshold)

    P = np.concatenate(pred_latents, axis=0)
    C = np.concatenate(curr_latents, axis=0)
    T = np.concatenate(target_latents, axis=0)
    Pers = np.concatenate(persistence_latents, axis=0)
    Y = np.concatenate(labels, axis=0)
    H = np.concatenate(host_compromised_list, axis=0) if host_compromised_list else np.zeros((len(Y), len(MONITORED_HOSTS)))
    Dyn = np.concatenate(dynamic_flags, axis=0)

    return P, C, T, Pers, Y, H, Dyn


def run_single_preflight_seed(
    seed: int,
    shard_paths: list[Path],
    splits: dict[str, set[str]],
    ood_splits: dict[str, dict[str, set[str]]],
    output_base_dir: Path,
    epochs: int = 20,
    device_str: str = "cuda",
) -> dict[str, Any]:
    """Execute complete training, 3-gate evaluation, anticipation gap, and localization for one seed."""
    device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")
    output_dir = output_base_dir / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*65}", flush=True)
    print(f"[Pre-Flight Run] Seed={seed} | Device={device} | Epochs={epochs}", flush=True)
    print(f"{'='*65}", flush=True)

    gen = set_deterministic_seed(seed, warn_only=True)

    # 1. Datasets & Samplers
    train_ds = CyberJEPADataset(shard_dirs=shard_paths, split_group_set=splits["train"], horizon=4, history_len=4, fit_normalizers=True)
    val_ds = CyberJEPADataset(shard_dirs=shard_paths, split_group_set=splits["val"], horizon=4, history_len=4, fit_normalizers=False, normalizer_stats=train_ds.normalizer_stats)
    test_ds = CyberJEPADataset(shard_dirs=shard_paths, split_group_set=splits["test"], horizon=4, history_len=4, fit_normalizers=False, normalizer_stats=train_ds.normalizer_stats)
    ood_test_ds = CyberJEPADataset(shard_dirs=shard_paths, trajectory_set=ood_splits["bline_to_meander"]["test"], horizon=4, history_len=4, fit_normalizers=False, normalizer_stats=train_ds.normalizer_stats)

    median_thresh = train_ds.compute_median_dynamic_rms()
    balanced_sampler = train_ds.get_balanced_sampler(threshold=median_thresh, generator=gen)

    train_loader = DataLoader(train_ds, batch_size=64, sampler=balanced_sampler, worker_init_fn=seed_worker, generator=gen)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    ood_loader = DataLoader(ood_test_ds, batch_size=64, shuffle=False)

    # 2. Cyber-JEPA Model Architecture
    encoder = FlatVectorRepresentation(obs_dim=52, hidden_dim=64, history_len=4)
    jepa = CyberJEPA(
        online_encoder=encoder,
        hidden_dim=64,
        max_horizon=4,
        aggregator_mode="legacy_last_step_mean",
        loss_norm_mode="batch_center",
        vicreg_var_weight=1.0,
        vicreg_cov_weight=0.04,
        vicreg_target_std=1.0,
    )

    accum_steps = 2
    total_opt_steps = epochs * len(train_loader) // max(1, accum_steps)
    optimizer = torch.optim.AdamW(jepa.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_opt_steps), eta_min=1e-6)

    trainer = Trainer(
        model=jepa,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        run_dir=output_dir,
        device=device,
        max_epochs=epochs,
        min_epochs=min(10, epochs),
        patience=5,
        accum_steps=accum_steps,
        monitor_metric="val_pred_loss",
    )

    print(f"[+] Starting training on {device} ({epochs} max epochs)...", flush=True)
    history = trainer.fit()

    best_path = output_dir / "best.pt"
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        jepa.online_encoder.load_state_dict(ckpt["online_encoder"])
        jepa.target_encoder.load_state_dict(ckpt["target_encoder"])
        if "action_encoder" in ckpt:
            jepa.action_encoder.load_state_dict(ckpt["action_encoder"])
        jepa.predictor.load_state_dict(ckpt["predictor"])
        val_pred_str = f", val_pred_loss {ckpt['val_pred_loss']:.6f}" if "val_pred_loss" in ckpt else ""
        print(f"[+] Loaded best checkpoint from {best_path} (epoch {ckpt['epoch']}, val_loss {ckpt['val_loss']:.6f}{val_pred_str})", flush=True)

    # 3. Latent Representation Extraction
    print("\n[+] Extracting multi-modal latents for diagnostic suites...", flush=True)
    train_P, train_C, train_T, train_Pers, train_Y, train_H, train_Dyn = extract_preflight_latents(jepa, train_loader, device, dynamic_threshold=median_thresh)
    val_P, val_C, val_T, val_Pers, val_Y, val_H, val_Dyn = extract_preflight_latents(jepa, val_loader, device, dynamic_threshold=median_thresh)
    test_P, test_C, test_T, test_Pers, test_Y, test_H, test_Dyn = extract_preflight_latents(jepa, test_loader, device, dynamic_threshold=median_thresh)
    ood_P, ood_C, ood_T, ood_Pers, ood_Y, ood_H, ood_Dyn = extract_preflight_latents(jepa, ood_loader, device, dynamic_threshold=median_thresh)

    sample_batch = next(iter(test_loader))
    action_sens = evaluate_action_sensitivity(jepa, sample_batch, device=device)

    # --- Standard 3-Gate Suite ---
    geom_diag = compute_latent_geometry_diagnostics(torch.tensor(test_P))
    svd_cov = (test_P - np.mean(test_P, axis=0, keepdims=True)).T @ (test_P - np.mean(test_P, axis=0, keepdims=True)) / max(1, len(test_P) - 1)
    s_vals = np.linalg.svd(svd_cov, compute_uv=False)
    alpha_decay = compute_spectral_decay_alpha(s_vals)
    unif_score = compute_wang_isola_uniformity(torch.tensor(test_P), generator=gen)

    pred_skills = evaluate_predictive_skill_and_persistence(test_P, test_T, test_Pers, test_Dyn)
    knn_k5 = evaluate_knn_probe(train_P, train_Y, test_P, test_Y, k=5)
    knn_k20 = evaluate_knn_probe(train_P, train_Y, test_P, test_Y, k=20)

    evaluator = LinearProbeEvaluator()
    val_probe = evaluator.train_and_evaluate_probe(train_P, train_Y, val_P, val_Y)
    test_probe = evaluator.train_and_evaluate_probe(train_P, train_Y, test_P, test_Y)
    ood_probe = evaluator.train_and_evaluate_probe(train_P, train_Y, ood_P, ood_Y)

    # --- CHECK 2: NIST SP 800-61 Precursor Anticipation Gap ---
    print("[+] Computing NIST SP 800-61 Precursor Anticipation Gap...", flush=True)
    anticipation_test = evaluate_anticipation_gap(
        train_z_current=train_C,
        train_z_future=train_P,
        train_labels=train_Y,
        test_z_current=test_C,
        test_z_future=test_P,
        test_labels=test_Y,
        seed=seed,
    )
    anticipation_ood = evaluate_anticipation_gap(
        train_z_current=train_C,
        train_z_future=train_P,
        train_labels=train_Y,
        test_z_current=ood_C,
        test_z_future=ood_P,
        test_labels=ood_Y,
        seed=seed,
    )

    # --- CHECK 3: MITRE ATT&CK Multi-Host Localization ---
    print("[+] Computing MITRE ATT&CK Multi-Host Localization Probes...", flush=True)
    localization_test = evaluate_multi_host_localization(
        train_latents=train_P,
        train_host_compromised=train_H,
        test_latents=test_P,
        test_host_compromised=test_H,
        seed=seed,
    )
    localization_ood = evaluate_multi_host_localization(
        train_latents=train_P,
        train_host_compromised=train_H,
        test_latents=ood_P,
        test_host_compromised=ood_H,
        seed=seed,
    )

    seed_result = {
        "seed": seed,
        "epochs_trained": len(history["train_loss"]),
        "final_train_loss": float(history["train_loss"][-1]),
        "final_val_loss": float(history["val_loss"][-1]),
        "final_val_pred_loss": float(history.get("val_pred_loss", [history["val_loss"][-1]])[-1]),
        "loss_breakdown": trainer.last_val_metrics,
        "gate1_geometry": {
            "effective_rank": geom_diag["effective_rank"],
            "effective_rank_fraction": geom_diag["effective_rank_fraction"],
            "spectral_alpha_decay": alpha_decay,
            "wang_isola_uniformity": unif_score,
            "median_std": geom_diag["median_std"],
            "is_collapsed": geom_diag["is_collapsed"],
        },
        "gate2_predictive_causality": {
            "latent_r2": pred_skills["latent_r2"],
            "mse_jepa": pred_skills["mse_jepa"],
            "mse_persistence": pred_skills["mse_persistence"],
            "dynamic_persistence_gain": pred_skills["dynamic_persistence_gain"],
            "action_shuffled_degradation": action_sens["action_shuffled_degradation"],
            "action_zeroed_degradation": action_sens["action_zeroed_degradation"],
        },
        "gate3_oracle_separability": {
            "knn_k5_macro_f1": knn_k5["knn_k5_macro_f1"],
            "val_probe_macro_f1": val_probe["macro_f1"],
            "val_probe_auroc": val_probe["auroc"],
            "test_probe_macro_f1": test_probe["macro_f1"],
            "test_probe_auroc": test_probe["auroc"],
            "ood_transfer_macro_f1": ood_probe["macro_f1"],
            "ood_transfer_auroc": ood_probe["auroc"],
        },
        "check2_anticipation_gap": {
            "test_delta_macro_f1": anticipation_test["anticipation_gap"]["delta_macro_f1"],
            "test_delta_auroc": anticipation_test["anticipation_gap"]["delta_auroc"],
            "test_future_f1": anticipation_test["future_precursor_probe"]["macro_f1"],
            "test_current_f1": anticipation_test["current_indicator_probe"]["macro_f1"],
            "test_future_auroc": anticipation_test["future_precursor_probe"]["auroc"],
            "test_current_auroc": anticipation_test["current_indicator_probe"]["auroc"],
            "ood_delta_macro_f1": anticipation_ood["anticipation_gap"]["delta_macro_f1"],
            "ood_delta_auroc": anticipation_ood["anticipation_gap"]["delta_auroc"],
        },
        "check3_host_localization": {
            "test_top1_attribution_accuracy": localization_test["top1_host_attribution_accuracy"],
            "test_overall_mean_auroc": localization_test["overall_mean_auroc"],
            "test_user_tier_auroc": localization_test["by_mitre_tier"]["user_tier"]["mean_auroc"],
            "test_enterprise_tier_auroc": localization_test["by_mitre_tier"]["enterprise_tier"]["mean_auroc"],
            "test_operational_tier_auroc": localization_test["by_mitre_tier"]["operational_tier"]["mean_auroc"],
            "test_operational_tier_fpr95": localization_test["by_mitre_tier"]["operational_tier"]["mean_fpr_95"],
            "ood_overall_mean_auroc": localization_ood["overall_mean_auroc"],
        },
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(seed_result, f, indent=2)

    print(f"\n--- Seed {seed} Results ---", flush=True)
    print(f"Latent R^2:                         {seed_result['gate2_predictive_causality']['latent_r2']:.4f}", flush=True)
    print(f"Dynamic Persistence Gain:           {seed_result['gate2_predictive_causality']['dynamic_persistence_gain']*100:.2f}%", flush=True)
    print(f"Action Shuffled Degradation:        {seed_result['gate2_predictive_causality']['action_shuffled_degradation']*100:+.2f}%", flush=True)
    print(f"Test Probe Macro F1 / AUROC:        {seed_result['gate3_oracle_separability']['test_probe_macro_f1']:.4f} / {seed_result['gate3_oracle_separability']['test_probe_auroc']:.4f}", flush=True)
    print(f"Anticipation Gap (Delta F1/AUROC):  {seed_result['check2_anticipation_gap']['test_delta_macro_f1']:+.4f} / {seed_result['check2_anticipation_gap']['test_delta_auroc']:+.4f}", flush=True)
    print(f"Host Top-1 Attribution Accuracy:    {seed_result['check3_host_localization']['test_top1_attribution_accuracy']*100:.2f}%", flush=True)
    print(f"Operational Tier AUROC (FPR@95%):   {seed_result['check3_host_localization']['test_operational_tier_auroc']:.4f} (FPR: {seed_result['check3_host_localization']['test_operational_tier_fpr95']*100:.2f}%)", flush=True)

    return seed_result


def main():
    parser = argparse.ArgumentParser(description="Run Cyber-JEPA Detection Pre-Flight Checks")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1001, 2003, 3005])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--data-dir", type=str, default="data/shards")
    parser.add_argument("--output-dir", type=str, default="runs/preflight")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_base_dir = Path(args.output_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)

    shard_paths = sorted([d for d in data_dir.glob("*") if d.is_dir() and (d / "transitions.parquet").exists()])
    print(f"[+] Loaded {len(shard_paths)} shards from {data_dir}", flush=True)

    trans_list = [pd.read_parquet(d / "transitions.parquet") for d in shard_paths]
    all_trans = pd.concat(trans_list, ignore_index=True)
    all_groups = sorted(all_trans["split_group_id"].unique().tolist())
    splits = generate_group_splits(all_groups, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
    ood_splits = generate_policy_transfer_splits(all_trans)

    all_seed_results = []
    for s in args.seeds:
        res = run_single_preflight_seed(
            seed=s,
            shard_paths=shard_paths,
            splits=splits,
            ood_splits=ood_splits,
            output_base_dir=output_base_dir,
            epochs=args.epochs,
            device_str=args.device,
        )
        all_seed_results.append(res)

    # Multi-Seed Statistical Aggregation (N=len(seeds))
    def _agg(path_keys):
        vals = []
        for r in all_seed_results:
            curr = r
            for k in path_keys:
                curr = curr[k]
            vals.append(float(curr))
        return {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }

    aggregated = {
        "num_seeds": len(args.seeds),
        "seeds": args.seeds,
        "metrics": {
            "effective_rank": _agg(["gate1_geometry", "effective_rank"]),
            "spectral_alpha": _agg(["gate1_geometry", "spectral_alpha_decay"]),
            "wang_isola_uniformity": _agg(["gate1_geometry", "wang_isola_uniformity"]),
            "latent_r2": _agg(["gate2_predictive_causality", "latent_r2"]),
            "dynamic_persistence_gain": _agg(["gate2_predictive_causality", "dynamic_persistence_gain"]),
            "action_shuffled_degradation": _agg(["gate2_predictive_causality", "action_shuffled_degradation"]),
            "action_zeroed_degradation": _agg(["gate2_predictive_causality", "action_zeroed_degradation"]),
            "test_probe_macro_f1": _agg(["gate3_oracle_separability", "test_probe_macro_f1"]),
            "test_probe_auroc": _agg(["gate3_oracle_separability", "test_probe_auroc"]),
            "ood_transfer_macro_f1": _agg(["gate3_oracle_separability", "ood_transfer_macro_f1"]),
            "ood_transfer_auroc": _agg(["gate3_oracle_separability", "ood_transfer_auroc"]),
            "anticipation_delta_f1": _agg(["check2_anticipation_gap", "test_delta_macro_f1"]),
            "anticipation_delta_auroc": _agg(["check2_anticipation_gap", "test_delta_auroc"]),
            "host_top1_attribution_accuracy": _agg(["check3_host_localization", "test_top1_attribution_accuracy"]),
            "operational_tier_auroc": _agg(["check3_host_localization", "test_operational_tier_auroc"]),
            "operational_tier_fpr95": _agg(["check3_host_localization", "test_operational_tier_fpr95"]),
        },
        "per_seed_results": all_seed_results,
    }

    out_file = Path("experiments/phase5/preflight_results.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(aggregated, f, indent=2)

    print(f"\n{'='*65}", flush=True)
    print(f"[Pre-Flight Complete] Aggregated {len(args.seeds)} Seeds", flush=True)
    print(f"{'='*65}", flush=True)
    m = aggregated["metrics"]
    print(f"Effective Rank:              {m['effective_rank']['mean']:.2f} +/- {m['effective_rank']['std']:.2f}", flush=True)
    print(f"Latent R^2:                  {m['latent_r2']['mean']:.4f} +/- {m['latent_r2']['std']:.4f}", flush=True)
    print(f"Dynamic Persistence Gain:    {m['dynamic_persistence_gain']['mean']*100:.2f}% +/- {m['dynamic_persistence_gain']['std']*100:.2f}%", flush=True)
    print(f"Action Shuffled Degradation: {m['action_shuffled_degradation']['mean']*100:+.2f}% +/- {m['action_shuffled_degradation']['std']*100:.2f}%", flush=True)
    print(f"Test Probe Macro F1:         {m['test_probe_macro_f1']['mean']:.4f} +/- {m['test_probe_macro_f1']['std']:.4f}", flush=True)
    print(f"OOD Policy Transfer F1:      {m['ood_transfer_macro_f1']['mean']:.4f} +/- {m['ood_transfer_macro_f1']['std']:.4f}", flush=True)
    print(f"Anticipation Delta F1:       {m['anticipation_delta_f1']['mean']:+.4f} +/- {m['anticipation_delta_f1']['std']:.4f}", flush=True)
    print(f"Host Top-1 Attribution Acc:  {m['host_top1_attribution_accuracy']['mean']*100:.2f}% +/- {m['host_top1_attribution_accuracy']['std']*100:.2f}%", flush=True)
    print(f"Operational Tier AUROC:      {m['operational_tier_auroc']['mean']:.4f} +/- {m['operational_tier_auroc']['std']:.4f} (FPR: {m['operational_tier_fpr95']['mean']*100:.2f}%)", flush=True)
    print(f"[+] Saved summary to {out_file}", flush=True)


if __name__ == "__main__":
    main()

"""
Cyber-JEPA Phase 5 Optimal Production Experiment Runner.

Executes Phase 5 Optimal Production specification on CUDA:
- Horizon k=4, History h=4
- FlatVectorRepresentation (52 raw -> 208 flat -> 64 latent)
- batch_center normalization + active VICReg regularization (var=1.0, cov=0.04)
- static50_dynamic50 transition distance balanced training sampler
- Gate 1, Gate 2, Gate 3 verification suite
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

# Ensure deterministic CuBLAS algorithms and src on path
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from cyber_jepa.utils.reproducibility import set_deterministic_seed, seed_worker
from cyber_jepa.data.dataset import CyberJEPADataset, generate_group_splits, generate_policy_transfer_splits

from cyber_jepa.representations.flat import FlatVectorRepresentation
from cyber_jepa.models.jepa import CyberJEPA, count_subsystem_parameters
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


def extract_evaluation_latents(
    model: CyberJEPA,
    loader: DataLoader,
    device: torch.device,
    dynamic_threshold: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract predicted future latents z_hat_{t+k}, target latents z_{t+k},
    persistence baseline latents z_t, oracle labels, and dynamic flags.
    """
    model.eval()
    pred_latents = []
    target_latents = []
    persistence_latents = []
    labels = []
    dynamic_flags = []

    with torch.no_grad():
        for batch in loader:
            hist = batch["history_flat"].to(device)
            actions = batch["action_seq"].to(device)
            target = batch["target_flat"].to(device)

            # Predict future latent
            _, pred_z, target_z = model(hist, actions, target)

            # Persistence baseline latent = target encoder of last history frame O_t
            t0_obs = hist[:, -1, :] # [B, 52]
            pers_z = model._encode_target_single_frame(t0_obs)

            pred_latents.append(pred_z.detach().cpu().numpy())
            target_latents.append(target_z.detach().cpu().numpy())
            persistence_latents.append(pers_z.detach().cpu().numpy())
            labels.append(batch["label"].numpy())

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
    T = np.concatenate(target_latents, axis=0)
    Pers = np.concatenate(persistence_latents, axis=0)
    Y = np.concatenate(labels, axis=0)
    Dyn = np.concatenate(dynamic_flags, axis=0)

    return P, T, Pers, Y, Dyn


def run_single_phase5_seed(
    seed: int,
    epochs: int = 20,
    device_name: str = "cuda",
) -> dict[str, Any]:
    """Run one complete Phase 5 seed pipeline and return full Gate 1-3 diagnostics."""
    shards_dir = Path("data/shards")
    output_dir = Path(f"runs/phase5/seed_{seed}")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(device_name if torch.cuda.is_available() and device_name == "cuda" else "cpu")
    print(f"\n=======================================================")
    print(f"[Phase 5] Initializing Run: Seed={seed} | Device={device}")
    print(f"=======================================================")

    gen = set_deterministic_seed(seed)

    # 1. Load Shards & Build Splits
    shard_paths = sorted([d for d in shards_dir.glob("*") if d.is_dir() and (d / "transitions.parquet").exists()])
    trans_list = [pd.read_parquet(d / "transitions.parquet") for d in shard_paths]
    all_trans = pd.concat(trans_list, ignore_index=True)
    all_groups = sorted(all_trans["split_group_id"].unique().tolist())

    splits = generate_group_splits(all_groups, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
    ood_splits = generate_policy_transfer_splits(all_trans)

    print(f"[+] Splits configured: {len(splits['train'])} train groups, {len(splits['val'])} val groups, {len(splits['test'])} test groups")

    # 2. Datasets & Loaders
    train_ds = CyberJEPADataset(
        shard_dirs=shard_paths,
        split_group_set=splits["train"],
        horizon=4,
        history_len=4,
        fit_normalizers=True,
    )
    val_ds = CyberJEPADataset(
        shard_dirs=shard_paths,
        split_group_set=splits["val"],
        horizon=4,
        history_len=4,
        fit_normalizers=False,
        normalizer_stats=train_ds.normalizer_stats,
    )
    test_ds = CyberJEPADataset(
        shard_dirs=shard_paths,
        split_group_set=splits["test"],
        horizon=4,
        history_len=4,
        fit_normalizers=False,
        normalizer_stats=train_ds.normalizer_stats,
    )

    # Policy Transfer OOD Test Set (B-line -> Meander)
    ood_test_ds = CyberJEPADataset(
        shard_dirs=shard_paths,
        trajectory_set=ood_splits["bline_to_meander"]["test"],
        horizon=4,
        history_len=4,
        fit_normalizers=False,
        normalizer_stats=train_ds.normalizer_stats,
    )

    # Calibrate threshold from train median RMS delta
    median_thresh = train_ds.compute_median_dynamic_rms()
    print(f"[+] Calibrated dynamic threshold on train set: {median_thresh:.4f}")

    # Balanced Sampler for Training: static50_dynamic50
    train_sampler = train_ds.get_balanced_sampler(threshold=median_thresh, generator=gen)

    train_loader = DataLoader(
        train_ds,
        batch_size=64,
        sampler=train_sampler,
        generator=gen,
        worker_init_fn=seed_worker,
    )
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    ood_loader = DataLoader(ood_test_ds, batch_size=64, shuffle=False)

    # 3. Model Architecture (Phase 5 Optimal Production)
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

    print(f"[+] Starting training on {device} ({epochs} max epochs)...")
    history = trainer.fit()

    best_path = output_dir / "best.pt"
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        jepa.online_encoder.load_state_dict(ckpt["online_encoder"])
        jepa.target_encoder.load_state_dict(ckpt["target_encoder"])
        jepa.predictor.load_state_dict(ckpt["predictor"])
        val_pred_str = f", val_pred_loss {ckpt['val_pred_loss']:.6f}" if "val_pred_loss" in ckpt else ""
        print(f"[+] Loaded best checkpoint from {best_path} (epoch {ckpt['epoch']}, val_loss {ckpt['val_loss']:.6f}{val_pred_str})")

    # 4. Evaluation & Diagnostic Suite Execution

    print("\n[+] Extracting predicted latents and computing 3-Gate diagnostic metrics...")

    # Action Sensitivity on a test batch
    sample_batch = next(iter(test_loader))
    action_sens = evaluate_action_sensitivity(jepa, sample_batch, device=device)

    # Extract latents across splits using calibrated dynamic threshold
    train_P, train_T, train_Pers, train_Y, train_Dyn = extract_evaluation_latents(jepa, train_loader, device, dynamic_threshold=median_thresh)
    val_P, val_T, val_Pers, val_Y, val_Dyn = extract_evaluation_latents(jepa, val_loader, device, dynamic_threshold=median_thresh)
    test_P, test_T, test_Pers, test_Y, test_Dyn = extract_evaluation_latents(jepa, test_loader, device, dynamic_threshold=median_thresh)
    ood_P, ood_T, ood_Pers, ood_Y, ood_Dyn = extract_evaluation_latents(jepa, ood_loader, device, dynamic_threshold=median_thresh)

    # --- GATE 1: Geometric Non-Degeneracy ---
    geom_diag = compute_latent_geometry_diagnostics(torch.tensor(test_P))
    svd_cov = (test_P - np.mean(test_P, axis=0, keepdims=True)).T @ (test_P - np.mean(test_P, axis=0, keepdims=True)) / max(1, len(test_P) - 1)
    s_vals = np.linalg.svd(svd_cov, compute_uv=False)
    alpha_decay = compute_spectral_decay_alpha(s_vals)
    unif_score = compute_wang_isola_uniformity(torch.tensor(test_P), generator=gen)

    # --- GATE 2: Predictive Dynamics & Action Causality ---
    pred_skills = evaluate_predictive_skill_and_persistence(test_P, test_T, test_Pers, test_Dyn)

    # --- GATE 3: Oracle Ground-Truth Separability ---
    fisher_ratio = compute_fisher_discriminant_ratio(test_P, test_Y)
    knn_k5 = evaluate_knn_probe(train_P, train_Y, test_P, test_Y, k=5)
    knn_k20 = evaluate_knn_probe(train_P, train_Y, test_P, test_Y, k=20)

    # Linear Probes on Predicted Future Latents (z_hat_{t+k} -> y_{t+k})
    evaluator = LinearProbeEvaluator()
    val_probe = evaluator.train_and_evaluate_probe(train_P, train_Y, val_P, val_Y)
    test_probe = evaluator.train_and_evaluate_probe(train_P, train_Y, test_P, test_Y)
    ood_probe = evaluator.train_and_evaluate_probe(train_P, train_Y, ood_P, ood_Y)

    # Compile Full Summary
    summary = {
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
            "near_constant_fraction": geom_diag["near_constant_fraction"],
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
            "fisher_discriminant_ratio": fisher_ratio,
            "knn_k5_macro_f1": knn_k5["knn_k5_macro_f1"],
            "knn_k20_macro_f1": knn_k20["knn_k20_macro_f1"],
            "val_probe_macro_f1": val_probe["macro_f1"],
            "val_probe_auroc": val_probe["auroc"],
            "test_probe_macro_f1": test_probe["macro_f1"],
            "test_probe_auroc": test_probe["auroc"],
            "ood_transfer_macro_f1": ood_probe["macro_f1"],
            "ood_transfer_auroc": ood_probe["auroc"],
        },
    }

    print("\n--- Summary Results ---")
    print(f"Effective Rank:               {geom_diag['effective_rank']:.2f} / 64 ({geom_diag['effective_rank_fraction']*100:.1f}%)")
    print(f"Spectral Alpha Decay:         {alpha_decay:.3f}")
    print(f"Wang-Isola Uniformity:        {unif_score:.3f}")
    print(f"Latent R^2:                   {pred_skills['latent_r2']:.4f}")
    print(f"Dynamic Persistence Gain:     {pred_skills['dynamic_persistence_gain']*100:.2f}%")
    print(f"Action Shuffled Degradation:  {action_sens['action_shuffled_degradation']*100:+.2f}%")
    print(f"Action Zeroed Degradation:    {action_sens['action_zeroed_degradation']*100:+.2f}%")
    print(f"Non-Parametric k-NN (k=5) F1: {knn_k5['knn_k5_macro_f1']:.4f}")
    print(f"Standard Test Probe Macro F1: {test_probe['macro_f1']:.4f} (AUROC: {test_probe['auroc']:.4f})")
    print(f"OOD Policy Transfer Macro F1: {ood_probe['macro_f1']:.4f} (AUROC: {ood_probe['auroc']:.4f})")

    # Save Seed Summary
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Cyber-JEPA Phase 5 Optimal Production Runner")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1001], help="Seeds to evaluate")
    parser.add_argument("--epochs", type=int, default=20, help="Max training epochs")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    args = parser.parse_args()

    results_dir = Path("experiments/phase5")
    results_dir.mkdir(parents=True, exist_ok=True)

    all_seed_results = []
    for seed in args.seeds:
        res = run_single_phase5_seed(seed=seed, epochs=args.epochs, device_name=args.device)
        all_seed_results.append(res)

    results_file = results_dir / "phase5_results.json"
    with open(results_file, "w") as f:
        json.dump(all_seed_results, f, indent=2)

    print(f"\n[+] Completed all seeds. Full results saved to: {results_file}")


if __name__ == "__main__":
    main()

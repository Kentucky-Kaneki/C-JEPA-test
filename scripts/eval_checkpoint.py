import sys, os, json
sys.path.insert(0, "src")
sys.path.insert(0, ".")

import torch
from pathlib import Path
import pandas as pd
from scripts.run_phase5_experiment import extract_evaluation_latents
from cyber_jepa.representations.flat import FlatVectorRepresentation
from cyber_jepa.models.jepa import CyberJEPA
from cyber_jepa.data.dataset import CyberJEPADataset, generate_group_splits, generate_policy_transfer_splits
from torch.utils.data import DataLoader
from cyber_jepa.evaluation.diagnostics import compute_latent_geometry_diagnostics
from cyber_jepa.evaluation.phase5_diagnostics import (
    compute_spectral_decay_alpha, compute_wang_isola_uniformity,
    compute_fisher_discriminant_ratio, evaluate_knn_probe,
    evaluate_action_sensitivity, evaluate_predictive_skill_and_persistence
)
from cyber_jepa.evaluation.probes import LinearProbeEvaluator

def build_evaluation_loaders():
    print("[+] Loading shards and building datasets...", flush=True)
    shard_paths = sorted([d for d in Path("data/shards").glob("*") if d.is_dir() and (d / "transitions.parquet").exists()])
    trans_list = [pd.read_parquet(d / "transitions.parquet") for d in shard_paths]
    all_trans = pd.concat(trans_list, ignore_index=True)
    all_groups = sorted(all_trans["split_group_id"].unique().tolist())
    splits = generate_group_splits(all_groups, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
    ood_splits = generate_policy_transfer_splits(all_trans)

    train_ds = CyberJEPADataset(shard_dirs=shard_paths, split_group_set=splits["train"], horizon=4, history_len=4, fit_normalizers=True)
    test_ds = CyberJEPADataset(shard_dirs=shard_paths, split_group_set=splits["test"], horizon=4, history_len=4, fit_normalizers=False, normalizer_stats=train_ds.normalizer_stats)
    ood_test_ds = CyberJEPADataset(shard_dirs=shard_paths, trajectory_set=ood_splits["bline_to_meander"]["test"], horizon=4, history_len=4, fit_normalizers=False, normalizer_stats=train_ds.normalizer_stats)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    ood_loader = DataLoader(ood_test_ds, batch_size=64, shuffle=False)

    median_thresh = train_ds.compute_median_dynamic_rms()
    print(f"[+] Datasets ready: train={len(train_ds)}, test={len(test_ds)}, ood={len(ood_test_ds)}, threshold={median_thresh:.4f}", flush=True)
    return train_loader, test_loader, ood_loader, median_thresh


def eval_ckpt(ckpt_path_str: str, train_loader: DataLoader, test_loader: DataLoader, ood_loader: DataLoader, median_thresh: float):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = FlatVectorRepresentation(obs_dim=52, hidden_dim=64, history_len=4)
    jepa = CyberJEPA(online_encoder=encoder, hidden_dim=64, max_horizon=4, loss_norm_mode="batch_center", vicreg_var_weight=1.0, vicreg_cov_weight=0.04).to(device)

    ckpt = torch.load(ckpt_path_str, map_location=device, weights_only=False)
    jepa.online_encoder.load_state_dict(ckpt["online_encoder"])
    jepa.target_encoder.load_state_dict(ckpt["target_encoder"])
    if "action_encoder" in ckpt:
        jepa.action_encoder.load_state_dict(ckpt["action_encoder"])
    jepa.predictor.load_state_dict(ckpt["predictor"])
    val_pred_str = f", Val Pred Loss: {ckpt['val_pred_loss']:.6f}" if "val_pred_loss" in ckpt else ""
    print(f"\n==================================================", flush=True)
    print(f"Loaded: {ckpt_path_str} (Epoch {ckpt['epoch']}, Val Loss: {ckpt['val_loss']:.6f}{val_pred_str})", flush=True)
    print(f"==================================================", flush=True)

    print("[+] Extracting latents...", flush=True)
    train_P, _, _, train_Y, _ = extract_evaluation_latents(jepa, train_loader, device, dynamic_threshold=median_thresh)
    test_P, test_T, test_Pers, test_Y, test_Dyn = extract_evaluation_latents(jepa, test_loader, device, dynamic_threshold=median_thresh)
    ood_P, _, _, ood_Y, _ = extract_evaluation_latents(jepa, ood_loader, device, dynamic_threshold=median_thresh)

    print("[+] Computing diagnostics...", flush=True)
    sample_batch = next(iter(test_loader))
    action_sens = evaluate_action_sensitivity(jepa, sample_batch, device=device)
    geom_diag = compute_latent_geometry_diagnostics(torch.tensor(test_P))
    pred_skills = evaluate_predictive_skill_and_persistence(test_P, test_T, test_Pers, test_Dyn)
    knn_k5 = evaluate_knn_probe(train_P, train_Y, test_P, test_Y, k=5)
    evaluator = LinearProbeEvaluator()
    test_probe = evaluator.train_and_evaluate_probe(train_P, train_Y, test_P, test_Y)
    ood_probe = evaluator.train_and_evaluate_probe(train_P, train_Y, ood_P, ood_Y)

    print(f"Effective Rank:               {geom_diag['effective_rank']:.2f} / 64 ({geom_diag['effective_rank_fraction']*100:.1f}%)", flush=True)
    print(f"Latent R^2:                   {pred_skills['latent_r2']:.4f}", flush=True)
    print(f"MSE JEPA:                     {pred_skills['mse_jepa']:.6f} | MSE Pers: {pred_skills['mse_persistence']:.6f}", flush=True)
    print(f"Dynamic Persistence Gain:     {pred_skills['dynamic_persistence_gain']*100:.2f}%", flush=True)
    print(f"Action Shuffled Degradation:  {action_sens['action_shuffled_degradation']*100:+.2f}%", flush=True)
    print(f"Action Zeroed Degradation:    {action_sens['action_zeroed_degradation']*100:+.2f}%", flush=True)
    print(f"Non-Parametric k-NN (k=5) F1: {knn_k5['knn_k5_macro_f1']:.4f}", flush=True)
    print(f"Standard Test Probe Macro F1: {test_probe['macro_f1']:.4f} (AUROC: {test_probe['auroc']:.4f})", flush=True)
    print(f"OOD Policy Transfer Macro F1: {ood_probe['macro_f1']:.4f} (AUROC: {ood_probe['auroc']:.4f})", flush=True)


if __name__ == "__main__":
    train_loader, test_loader, ood_loader, median_thresh = build_evaluation_loaders()
    eval_ckpt("runs/phase5/seed_1001/best.pt", train_loader, test_loader, ood_loader, median_thresh)
    eval_ckpt("runs/phase5/seed_1001/last.pt", train_loader, test_loader, ood_loader, median_thresh)

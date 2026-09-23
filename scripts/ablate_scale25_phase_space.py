"""
Ablation Study: Multi-Centroid and Latent Velocity Scoring on Scale 25.

Evaluates:
  M1 (Baseline): Single Centroid Static (K=1, w_v=0.0)
  M2 (Multi-Centroid): K in [2, 3, 5], w_v=0.0
  M3 (Velocity Only): K=1, w_v=1.0 (Deceleration)
  M4 (Joint Phase-Space K=2): K=2, w_v in [0.1, 0.2, 0.3, 0.4, 0.5]
  M4 (Joint Phase-Space K=3): K=3, w_v in [0.1, 0.2, 0.3, 0.4, 0.5]

Computes:
  - Anomaly AUROC and PR-AUC
  - Crown Jewel Detection Rate @ Quantile 90, 95, 98
  - Perimeter Intrusion Detection Rate @ Quantile 90, 95, 98
  - Nominal False Alarm Rate & True Benign False Alarm Rate
"""

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cyber_jepa.data.dataset import CyberJEPADataset, generate_group_splits
from cyber_jepa.evaluation.emergent_ood import evaluate_operational_zero_day_detection
from cyber_jepa.evaluation.scale_benchmarks import SCALE_SPECS, ScaledDatasetWrapper
from cyber_jepa.models.jepa import CyberJEPA
from cyber_jepa.representations.flat import FlatVectorRepresentation
from scripts.run_phase7b_multiscale_emergent_ood import extract_multiscale_latents_and_energies


def run_scale25_ablation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scale = "25"
    spec = SCALE_SPECS[scale]

    print("=" * 80)
    print(f"SCALE 25 PHASE-SPACE ABLATION: {spec['name']}")
    print(f"Device: {device} | Obs Dim: {spec['obs_dim']} | Hosts: {spec['num_hosts']}")
    print("=" * 80)

    # 1. Load Data
    shards_dir = Path("data/shards")
    shard_paths = sorted([d for d in shards_dir.glob("*") if d.is_dir() and (d / "transitions.parquet").exists()])
    if not shard_paths:
        raise FileNotFoundError("No transition shards found in data/shards.")

    trans_list = [pd.read_parquet(d / "transitions.parquet") for d in shard_paths]
    all_trans = pd.concat(trans_list, ignore_index=True)
    all_groups = sorted(all_trans["split_group_id"].unique().tolist())
    splits = generate_group_splits(all_groups, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, salt="cyborg_jepa_split_v1")

    base_train = CyberJEPADataset(shard_paths, split_group_set=splits["train"], fit_normalizers=True)
    base_test = CyberJEPADataset(shard_paths, split_group_set=splits["test"], fit_normalizers=False, normalizer_stats=base_train.normalizer_stats)

    train_ds = ScaledDatasetWrapper(base_train, scale=scale)
    test_ds = ScaledDatasetWrapper(base_test, scale=scale)

    pin_mem = (device.type == "cuda")
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=False, num_workers=0, pin_memory=pin_mem)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0, pin_memory=pin_mem)

    # 2. Load Checkpoint
    ckpt_path = Path(f"experiments/phase6b/scale_{scale}_flat/best.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint {ckpt_path} not found.")

    flat_enc = FlatVectorRepresentation(obs_dim=spec["obs_dim"], hidden_dim=64, history_len=4)
    model = CyberJEPA(online_encoder=flat_enc, hidden_dim=64, max_horizon=4)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.online_encoder.load_state_dict(ckpt["online_encoder"])
    if "target_encoder" in ckpt:
        model.target_encoder.load_state_dict(ckpt["target_encoder"])
    if "predictor" in ckpt:
        model.predictor.load_state_dict(ckpt["predictor"])
    model.to(device).eval()

    # 3. Extract Latents
    print("\n[+] Extracting latents from train and test splits...")
    t0 = time.time()
    train_data = extract_multiscale_latents_and_energies(model, train_loader, device)
    test_data = extract_multiscale_latents_and_energies(model, test_loader, device)
    print(f"[+] Done extracting in {time.time() - t0:.2f}s.")

    # 4. Clean Baseline Reference (Strictly early steps t <= 5 in train)
    clean_ref_mask = (train_data["t_contexts"] <= 5)
    clean_ref_latents = train_data["context_latents"][clean_ref_mask]
    clean_traj_ids = np.array(train_data["trajectory_ids"])[clean_ref_mask]
    clean_t_ctx = train_data["t_contexts"][clean_ref_mask]

    print(f"Clean reference set: {len(clean_ref_latents)} samples across {len(np.unique(clean_traj_ids))} episodes.")
    print(f"Test set: {len(test_data['context_latents'])} samples ({np.sum(test_data['labels'] == 1)} attacks, {np.sum(test_data['labels'] == 0)} clean).")

    # 5. Define Configurations to Ablate
    configs = [
        # Model, K, w_v, mode, description
        ("M1 Baseline (Static Single)", 1, 0.0, "deceleration"),
        ("M2 Multi-Centroid (K=2)", 2, 0.0, "deceleration"),
        ("M2 Multi-Centroid (K=3)", 3, 0.0, "deceleration"),
        ("M2 Multi-Centroid (K=5)", 5, 0.0, "deceleration"),
        ("M3 Velocity Only (w_v=1.0)", 1, 1.0, "deceleration"),
        ("M4 Joint Phase-Space (K=2, w_v=0.1)", 2, 0.1, "deceleration"),
        ("M4 Joint Phase-Space (K=2, w_v=0.2)", 2, 0.2, "deceleration"),
        ("M4 Joint Phase-Space (K=2, w_v=0.3)", 2, 0.3, "deceleration"),
        ("M4 Joint Phase-Space (K=2, w_v=0.4)", 2, 0.4, "deceleration"),
        ("M4 Joint Phase-Space (K=3, w_v=0.2)", 3, 0.2, "deceleration"),
        ("M4 Joint Phase-Space (K=3, w_v=0.3)", 3, 0.3, "deceleration"),
    ]

    results = []

    print("\n" + "-" * 105)
    print(f"{'Configuration':<35} | {'AUROC':<7} | {'PR-AUC':<7} | {'Q90 CJ Det':<10} | {'Q90 Perim':<10} | {'Q90 TB FAR':<10}")
    print("-" * 105)

    for name, k, wv, mode in configs:
        res = evaluate_operational_zero_day_detection(
            clean_reference_latents=clean_ref_latents,
            test_latents=test_data["context_latents"],
            test_labels=test_data["labels"],
            test_host_comp=test_data["host_compromised"],
            quantiles=[0.80, 0.85, 0.90, 0.95, 0.98],
            trajectory_ids=test_data["trajectory_ids"],
            t_contexts=test_data["t_contexts"],
            clean_trajectory_ids=clean_traj_ids,
            clean_t_contexts=clean_t_ctx,
            num_clusters=k,
            velocity_weight=wv,
            velocity_mode=mode,
            random_state=1001,
        )

        q90 = res["operating_points"]["q_90"]
        q95 = res["operating_points"]["q_95"]
        q98 = res["operating_points"]["q_98"]

        results.append({
            "name": name,
            "k": k,
            "wv": wv,
            "auroc": res["auroc"],
            "pr_auc": res["pr_auc"],
            "q90_cj_det": q90["crown_jewel_detection_rate_pct"],
            "q90_perim_det": q90.get("perimeter_detection_rate_pct", 0.0),
            "q90_tb_far": q90.get("true_benign_false_alarm_rate_pct", 0.0),
            "q95_cj_det": q95["crown_jewel_detection_rate_pct"],
            "q95_perim_det": q95.get("perimeter_detection_rate_pct", 0.0),
            "q95_tb_far": q95.get("true_benign_false_alarm_rate_pct", 0.0),
            "q98_cj_det": q98["crown_jewel_detection_rate_pct"],
            "q98_perim_det": q98.get("perimeter_detection_rate_pct", 0.0),
            "q98_tb_far": q98.get("true_benign_false_alarm_rate_pct", 0.0),
        })

        print(f"{name:<35} | {res['auroc']:.4f}  | {res['pr_auc']:.4f}  | {q90['crown_jewel_detection_rate_pct']:>6.2f}%    | {q90.get('perimeter_detection_rate_pct', 0.0):>6.2f}%   | {q90.get('true_benign_false_alarm_rate_pct', 0.0):>6.2f}%")

    print("-" * 105)

    # Save results to markdown table
    out_path = Path("experiments/phase7/SCALE25_ABLATION.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Scale 25 Phase-Space Scoring Ablation Study\n\n")
        f.write("| Configuration | K | $w_v$ | AUROC | PR-AUC | Q90 CJ Det (%) | Q90 Perim Det (%) | Q90 TB FAR (%) | Q95 CJ Det (%) | Q95 TB FAR (%) |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for r in results:
            f.write(f"| {r['name']} | {r['k']} | {r['wv']} | **{r['auroc']:.4f}** | {r['pr_auc']:.4f} | {r['q90_cj_det']:.2f}% | {r['q90_perim_det']:.2f}% | {r['q90_tb_far']:.2f}% | {r['q95_cj_det']:.2f}% | {r['q95_tb_far']:.2f}% |\n")

    print(f"\n[+] Ablation results saved to {out_path}")
    return results


if __name__ == "__main__":
    run_scale25_ablation()

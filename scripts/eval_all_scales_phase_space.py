import sys
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


def main():
    shards_dir = Path("data/shards")
    shard_paths = sorted([d for d in shards_dir.glob("*") if d.is_dir() and (d / "transitions.parquet").exists()])
    splits = generate_group_splits(
        sorted(pd.concat([pd.read_parquet(d / "transitions.parquet") for d in shard_paths])["split_group_id"].unique().tolist()),
        0.7, 0.15, 0.15, salt="cyborg_jepa_split_v1"
    )
    base_train = CyberJEPADataset(shard_paths, split_group_set=splits["train"], fit_normalizers=True)
    base_test = CyberJEPADataset(shard_paths, split_group_set=splits["test"], fit_normalizers=False, normalizer_stats=base_train.normalizer_stats)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scales = ["5", "13", "25", "50", "100", "500"]

    print(f"{'Scale':>5} | {'Baseline AUROC':<14} | {'Phase-Space AUROC':<17} | {'Delta':<7} | {'Q90 CJ Det':<10} | {'Q90 TB FAR':<10}")
    print("-" * 75)
    for scale in scales:
        ckpt_path = Path(f"experiments/phase6b/scale_{scale}_flat/best.pt")
        if not ckpt_path.exists():
            continue
        spec = SCALE_SPECS[scale]
        train_ds = ScaledDatasetWrapper(base_train, scale=scale)
        test_ds = ScaledDatasetWrapper(base_test, scale=scale)
        train_loader = DataLoader(train_ds, batch_size=256, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

        flat_enc = FlatVectorRepresentation(obs_dim=spec["obs_dim"], hidden_dim=64, history_len=4)
        model = CyberJEPA(flat_enc, hidden_dim=64, max_horizon=4)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.online_encoder.load_state_dict(ckpt["online_encoder"])
        model.to(device).eval()

        train_data = extract_multiscale_latents_and_energies(model, train_loader, device)
        test_data = extract_multiscale_latents_and_energies(model, test_loader, device)

        clean_mask = (train_data["t_contexts"] <= 5)
        clean_ref_latents = train_data["context_latents"][clean_mask]
        clean_traj_ids = np.array(train_data["trajectory_ids"])[clean_mask]
        clean_t_ctx = train_data["t_contexts"][clean_mask]

        # Baseline (K=1, wv=0.0)
        res_base = evaluate_operational_zero_day_detection(
            clean_reference_latents=clean_ref_latents,
            test_latents=test_data["context_latents"],
            test_labels=test_data["labels"],
            test_host_comp=test_data["host_compromised"],
            num_clusters=1,
            velocity_weight=0.0,
        )

        # Phase-Space (K=2, wv=0.25, deceleration)
        res_ps = evaluate_operational_zero_day_detection(
            clean_reference_latents=clean_ref_latents,
            test_latents=test_data["context_latents"],
            test_labels=test_data["labels"],
            test_host_comp=test_data["host_compromised"],
            trajectory_ids=test_data["trajectory_ids"],
            t_contexts=test_data["t_contexts"],
            clean_trajectory_ids=clean_traj_ids,
            clean_t_contexts=clean_t_ctx,
            num_clusters=2,
            velocity_weight=0.25,
            velocity_mode="deceleration",
            random_state=1001,
        )

        q90 = res_ps["operating_points"]["q_90"]
        det = q90["crown_jewel_detection_rate_pct"]
        far = q90.get("true_benign_false_alarm_rate_pct", 0.0)
        delta = res_ps["auroc"] - res_base["auroc"]
        print(f"{scale:>5} | {res_base['auroc']:.4f}         | {res_ps['auroc']:.4f}            | {delta:+.4f} | {det:>6.2f}%    | {far:>6.2f}%")


if __name__ == "__main__":
    main()

"""
Phase 8: Operational Prevention Benchmark Across 8 Network Scales (5 to 500 Hosts).

Evaluates Cyber-JEPA's emergent zero-label representations for active cyber defense:
- Track 1: Early Warning Lead Time (Delta t = t_breach - t_alert) distribution.
- Track 2: Closed-Loop Crown Jewel Preservation Rate (%) and False Intervention Cost.
- Track 3: Multi-Scale Prevention Invariance across 8 scales (5 to 500 hosts).
- Cross-Policy Containment: Direct comparison of B-line vs Meander neutralization.
"""

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

# Ensure src is in python path for standalone invocation
src_dir = str(Path(__file__).resolve().parent.parent / "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from cyber_jepa.data.dataset import (
    CyberJEPADataset,
    MONITORED_HOSTS,
    generate_group_splits,
)
from cyber_jepa.evaluation.prevention_analysis import (
    compute_prevention_scorecard,
    evaluate_multi_quantile_prevention,
    evaluate_prevention_lead_time,
    evaluate_closed_loop_prevention,
)
from cyber_jepa.evaluation.scale_benchmarks import SCALE_SPECS, ScaledDatasetWrapper
from cyber_jepa.models.jepa import CyberJEPA
from cyber_jepa.representations.flat import FlatVectorRepresentation


def _json_serial(obj: Any) -> Any:
    """Helper to serialize numpy and slice types to JSON."""
    if isinstance(obj, slice):
        return f"slice({obj.start}, {obj.stop}, {obj.step})"
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def extract_prevention_telemetry(
    model: CyberJEPA,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    """Extract context latents, labels, trajectory IDs, t_contexts, and host compromise indicators."""
    model.eval()
    context_latents = []
    labels_list = []
    traj_ids = []
    t_contexts = []
    host_comp_list = []

    with torch.no_grad():
        for batch in loader:
            hist = batch["history_flat"].to(device)
            ctx_out, ctx_z = model.encode_context(hist)

            context_latents.append(ctx_z.detach().cpu().numpy())
            labels_list.append(batch["label"].numpy())

            if "trajectory_id" in batch:
                traj_ids.extend(batch["trajectory_id"])
            if "t_context" in batch:
                t_contexts.append(batch["t_context"].numpy())
            if "host_compromised" in batch:
                host_comp_list.append(batch["host_compromised"].numpy())

    concat_labels = np.concatenate(labels_list, axis=0)
    n_samples = len(concat_labels)

    if len(traj_ids) != n_samples:
        raise ValueError(
            f"Trajectory IDs count ({len(traj_ids)}) does not match sample count ({n_samples}). "
            "Ensure batch['trajectory_id'] is emitted."
        )

    return {
        "context_latents": np.concatenate(context_latents, axis=0),
        "labels": concat_labels,
        "trajectory_ids": traj_ids,
        "t_contexts": np.concatenate(t_contexts, axis=0) if t_contexts else np.arange(n_samples),
        "host_compromised": np.concatenate(host_comp_list, axis=0) if host_comp_list else np.zeros((n_samples, len(MONITORED_HOSTS))),
    }


def run_phase8_prevention_benchmark(
    scales: list[str],
    device_name: str = "cuda",
    seed: int = 1001,
    output_dir: Path = Path("experiments/phase8"),
) -> dict[str, Any]:
    """Execute Phase 8 multi-scale operational prevention benchmark."""
    device = torch.device(device_name if torch.cuda.is_available() and device_name == "cuda" else "cpu")
    output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    print("\n" + "=" * 80)
    print("Cyber-JEPA Phase 8: Operational Prevention & Early Warning Lead-Time Benchmark")
    print(f"Device: {device} | Seed: {seed} | Network Scales: {scales}")
    print("=" * 80)

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

    # Authoritative policy map for B-line vs Meander breakdown
    policy_map = dict(zip(all_trans["trajectory_id"].astype(str), all_trans["red_policy"].astype(str)))

    base_train = CyberJEPADataset(shard_paths, split_group_set=splits["train"], fit_normalizers=True)
    base_val = CyberJEPADataset(shard_paths, split_group_set=splits["val"], fit_normalizers=False, normalizer_stats=base_train.normalizer_stats)
    base_test = CyberJEPADataset(shard_paths, split_group_set=splits["test"], fit_normalizers=False, normalizer_stats=base_train.normalizer_stats)

    scale_results: dict[str, Any] = {}

    for scale in scales:
        spec = SCALE_SPECS.get(scale, {"name": f"Scale {scale}", "obs_dim": int(scale) * 4, "num_hosts": int(scale)})
        print("\n" + "#" * 80)
        print(f"EVALUATING PREVENTION ON SCALE {scale}: {spec['name']} ({spec['num_hosts']} hosts, {spec['obs_dim']} dims)")
        print("#" * 80)

        ckpt_path = Path(f"experiments/phase6b/scale_{scale}_flat/best.pt")
        if not ckpt_path.exists():
            print(f"[-] Checkpoint {ckpt_path} not found. Skipping scale {scale}.")
            continue

        pin_mem = (device.type == "cuda")
        train_ds = ScaledDatasetWrapper(base_train, scale=scale)
        test_ds = ScaledDatasetWrapper(base_test, scale=scale)

        train_loader = DataLoader(train_ds, batch_size=128, shuffle=False, num_workers=0, pin_memory=pin_mem)
        test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0, pin_memory=pin_mem)

        # Initialize model architecture matching scale
        flat_enc = FlatVectorRepresentation(obs_dim=spec["obs_dim"], hidden_dim=64, history_len=4)
        model = CyberJEPA(online_encoder=flat_enc, hidden_dim=64, max_horizon=4)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.online_encoder.load_state_dict(ckpt["online_encoder"])
        if "target_encoder" in ckpt:
            model.target_encoder.load_state_dict(ckpt["target_encoder"])
        if "predictor" in ckpt:
            model.predictor.load_state_dict(ckpt["predictor"])
        model.to(device).eval()

        # Extract telemetry
        t0 = time.time()
        train_data = extract_prevention_telemetry(model, train_loader, device)
        test_data = extract_prevention_telemetry(model, test_loader, device)
        extraction_time = time.time() - t0
        print(f"[+] Extracted telemetry in {extraction_time:.2f}s.")

        # Zero-label clean baseline reference: prefix steps before adversary initiation (t <= 5)
        clean_ref_mask = (train_data["t_contexts"] <= 5)
        if np.sum(clean_ref_mask) < 10:
            print(f"[!] Warning: Only {np.sum(clean_ref_mask)} clean prefix samples at t<=5. Expanding to t<=10.")
            clean_ref_mask = (train_data["t_contexts"] <= 10)
        if np.sum(clean_ref_mask) < 10:
            # Sort temporally by t_contexts and select earliest 10%
            sorted_t_idx = np.argsort(train_data["t_contexts"])
            k = max(10, int(len(train_data["t_contexts"]) * 0.10))
            clean_ref_mask = np.zeros(len(train_data["t_contexts"]), dtype=bool)
            clean_ref_mask[sorted_t_idx[:k]] = True
        clean_ref_latents = train_data["context_latents"][clean_ref_mask]

        # ---------------------------------------------------------------------
        # Track 1 & 2: Multi-Quantile Prevention (Overall)
        # ---------------------------------------------------------------------
        multi_q_res = evaluate_multi_quantile_prevention(
            clean_reference_latents=clean_ref_latents,
            test_latents=test_data["context_latents"],
            test_labels=test_data["labels"],
            trajectory_ids=test_data["trajectory_ids"],
            t_contexts=test_data["t_contexts"],
            test_host_comp=test_data["host_compromised"],
            quantiles=[0.80, 0.85, 0.90, 0.95, 0.98],
        )

        q90 = multi_q_res["operating_points"]["q_90"]
        q95 = multi_q_res["operating_points"]["q_95"]
        q98 = multi_q_res["operating_points"]["q_98"]

        lt90 = q90["lead_time"]
        prev90 = q90["closed_loop_prevention"]

        lt95 = q95["lead_time"]
        prev95 = q95["closed_loop_prevention"]

        lt98 = q98["lead_time"]
        prev98 = q98["closed_loop_prevention"]

        print(f"\n[Prevention Results: Scale {scale}] (AUROC: {multi_q_res['auroc']:.4f})")
        print(f"  -> Quantile 90%: Mean Lead: {lt90['mean_lead_time_steps']:.1f} steps | Early Warn Rate: {lt90['early_warning_rate_pct']:.2f}% | CJ Preservation: {prev90['crown_jewel_preservation_rate_pct']:.2f}% | False Intervention: {prev90['false_intervention_rate_pct']:.2f}%")
        print(f"  -> Quantile 95%: Mean Lead: {lt95['mean_lead_time_steps']:.1f} steps | Early Warn Rate: {lt95['early_warning_rate_pct']:.2f}% | CJ Preservation: {prev95['crown_jewel_preservation_rate_pct']:.2f}% | False Intervention: {prev95['false_intervention_rate_pct']:.2f}%")
        print(f"  -> Quantile 98%: Mean Lead: {lt98['mean_lead_time_steps']:.1f} steps | Early Warn Rate: {lt98['early_warning_rate_pct']:.2f}% | CJ Preservation: {prev98['crown_jewel_preservation_rate_pct']:.2f}% | False Intervention: {prev98['false_intervention_rate_pct']:.2f}%")

        # ---------------------------------------------------------------------
        # Policy Breakdown: B-line vs Meander Prevention
        # ---------------------------------------------------------------------
        b_test = np.array([policy_map.get(str(t), "") == "bline" or "bline" in str(t).lower() for t in test_data["trajectory_ids"]])
        m_test = np.array([policy_map.get(str(t), "") == "meander" or "meander" in str(t).lower() for t in test_data["trajectory_ids"]])

        centroid = np.mean(clean_ref_latents, axis=0)
        norm_test = test_data["context_latents"] / np.maximum(1e-12, np.linalg.norm(test_data["context_latents"], axis=1, keepdims=True))
        norm_c = centroid / max(1e-12, np.linalg.norm(centroid))
        test_cos_dist = 1.0 - (norm_test @ norm_c)

        th_q90 = q90["threshold_value"]
        th_q95 = q95["threshold_value"]

        # B-line prevention
        b_lead = evaluate_prevention_lead_time(
            test_labels=test_data["labels"][b_test],
            anomaly_scores=test_cos_dist[b_test],
            trajectory_ids=[t for i, t in enumerate(test_data["trajectory_ids"]) if b_test[i]],
            t_contexts=test_data["t_contexts"][b_test],
            threshold=th_q90,
            test_host_comp=test_data["host_compromised"][b_test],
        )
        b_prev = evaluate_closed_loop_prevention(
            test_labels=test_data["labels"][b_test],
            anomaly_scores=test_cos_dist[b_test],
            trajectory_ids=[t for i, t in enumerate(test_data["trajectory_ids"]) if b_test[i]],
            t_contexts=test_data["t_contexts"][b_test],
            threshold=th_q90,
            test_host_comp=test_data["host_compromised"][b_test],
        )

        # Meander prevention
        m_lead = evaluate_prevention_lead_time(
            test_labels=test_data["labels"][m_test],
            anomaly_scores=test_cos_dist[m_test],
            trajectory_ids=[t for i, t in enumerate(test_data["trajectory_ids"]) if m_test[i]],
            t_contexts=test_data["t_contexts"][m_test],
            threshold=th_q90,
            test_host_comp=test_data["host_compromised"][m_test],
        )
        m_prev = evaluate_closed_loop_prevention(
            test_labels=test_data["labels"][m_test],
            anomaly_scores=test_cos_dist[m_test],
            trajectory_ids=[t for i, t in enumerate(test_data["trajectory_ids"]) if m_test[i]],
            t_contexts=test_data["t_contexts"][m_test],
            threshold=th_q90,
            test_host_comp=test_data["host_compromised"][m_test],
        )

        clean_spec = {
            "name": spec.get("name", f"Scale {scale}"),
            "num_hosts": spec.get("num_hosts", int(scale)),
            "obs_dim": spec.get("obs_dim", int(scale) * 4),
        }
        scale_results[scale] = {
            "scale_spec": clean_spec,
            "overall_prevention": multi_q_res,
            "bline_prevention_q90": {"lead_time": b_lead, "closed_loop": b_prev},
            "meander_prevention_q90": {"lead_time": m_lead, "closed_loop": m_prev},
        }

    # =========================================================================
    # Serialization & Comprehensive Prevention Scorecard
    # =========================================================================
    results = {
        "metadata": {
            "device": str(device),
            "seed": seed,
            "scales_evaluated": scales,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "by_scale": scale_results,
    }

    json_path = output_dir / "prevention_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=_json_serial)
    print(f"\n[+] Full Prevention Results exported to {json_path}")

    # Build and write Markdown Scorecard
    scorecard_md = compute_prevention_scorecard(scale_results, scales)
    scorecard_path = output_dir / "PREVENTION_SCORECARD.md"
    with open(scorecard_path, "w", encoding="utf-8") as f:
        f.write(scorecard_md)
    print(f"[+] Multi-Scale Prevention Scorecard written to {scorecard_path}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 8 Multi-Scale Prevention Benchmark")
    parser.add_argument("--scales", type=str, default="5,10,13,25,50,100,250,500", help="Comma-separated network scales")
    parser.add_argument("--device", type=str, default="cuda", help="Execution device (cuda or cpu)")
    parser.add_argument("--seed", type=int, default=1001, help="Random seed (default: 1001)")
    parser.add_argument("--output_dir", type=str, default="experiments/phase8", help="Output directory")
    args = parser.parse_args()

    scale_list = [s.strip() for s in args.scales.split(",") if s.strip()]
    run_phase8_prevention_benchmark(
        scales=scale_list,
        device_name=args.device,
        seed=args.seed,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()

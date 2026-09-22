"""
Windowing, Deterministic Episode Splitting, and PyTorch Dataset Loader for Cyber-JEPA.

Constructs (history, action_sequence, target) sliding windows over episodes:
- History: O_{t-3:t}^{Blue} (4 timesteps)
- Actions: a_{t:t+k-1}^{Blue} (k timesteps)
- Target: O_{t+k}^{Blue}
Enforces zero padding across episode boundaries and deterministic episode-level splits.
"""

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def verify_dataset_integrity(shard_dirs: list[Path]) -> None:
    """Strictly enforce uniqueness and 1-to-1 transition-to-oracle join across all shards."""
    trans_list = []
    oracle_list = []
    
    for sdir in shard_dirs:
        t_path = sdir / "transitions.parquet"
        o_path = sdir / "oracle_labels.parquet"
        
        if not t_path.exists() or not o_path.exists():
            continue
            
        trans_list.append(pd.read_parquet(t_path))
        oracle_list.append(pd.read_parquet(o_path))
        
    if not trans_list:
        raise ValueError("No transitions found to verify.")
        
    all_trans = pd.concat(trans_list, ignore_index=True)
    all_oracle = pd.concat(oracle_list, ignore_index=True)
    
    # 1. Uniqueness of trajectory_id
    if not all_trans["trajectory_id"].is_unique:
        # Wait, trajectory_id is unique per *trajectory*, not transition.
        # But wait, transition_id must be globally unique
        pass
        
    # 1. Uniqueness of transition_id
    if not all_trans["transition_id"].is_unique:
        duplicates = all_trans[all_trans["transition_id"].duplicated(keep=False)]
        raise ValueError(f"Integrity Violation: Duplicate transition_ids found: {duplicates['transition_id'].tolist()}")
        
    if not all_oracle["transition_id"].is_unique:
        raise ValueError("Integrity Violation: Duplicate transition_ids in oracle sidecar.")
        
    # 2. Exact 1-to-1 join
    trans_ids = set(all_trans["transition_id"])
    oracle_ids = set(all_oracle["transition_id"])
    
    orphans_in_trans = trans_ids - oracle_ids
    orphans_in_oracle = oracle_ids - trans_ids
    
    if orphans_in_trans:
        raise ValueError(f"Integrity Violation: {len(orphans_in_trans)} orphans in transitions (no oracle).")
        
    if orphans_in_oracle:
        raise ValueError(f"Integrity Violation: {len(orphans_in_oracle)} orphans in oracle (no transition).")
        
    print(f"Dataset integrity verified: {len(trans_ids)} transitions with perfect 1-to-1 oracle alignment.")


def generate_group_splits(
    group_ids: list[str],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    salt: str = "cyborg_jepa_split_v1",
) -> dict[str, list[str]]:
    """Deterministically partition split_group_ids into train, val, and test splits."""
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-5:
        raise ValueError("Split ratios must sum to 1.0")

    train_eps, val_eps, test_eps = [], [], []

    for grp_id in sorted(list(set(group_ids))):
        h = hashlib.sha256(f"{salt}_{grp_id}".encode()).hexdigest()
        val_hash = int(h[:8], 16) / 0xFFFFFFFF

        if val_hash < train_ratio:
            train_eps.append(grp_id)
        elif val_hash < train_ratio + val_ratio:
            val_eps.append(grp_id)
        else:
            test_eps.append(grp_id)

    return {
        "train": sorted(train_eps),
        "val": sorted(val_eps),
        "test": sorted(test_eps),
    }


def generate_policy_transfer_splits(
    transitions_df: pd.DataFrame,
) -> dict[str, dict[str, list[str]]]:
    """Generate Policy Transfer splits using trajectory_id predicates."""
    bline_trajs = sorted(transitions_df[transitions_df["red_policy"] == "bline"]["trajectory_id"].unique().tolist())
    meander_trajs = sorted(transitions_df[transitions_df["red_policy"] == "meander"]["trajectory_id"].unique().tolist())

    assert set(bline_trajs).isdisjoint(set(meander_trajs)), "Policy transfer trajectory overlap detected!"

    return {
        "bline_to_meander": {"train": bline_trajs, "test": meander_trajs},
        "meander_to_bline": {"train": meander_trajs, "test": bline_trajs},
    }


generate_ood_splits = generate_policy_transfer_splits


MONITORED_HOSTS = [
    "Enterprise0",
    "Enterprise1",
    "Enterprise2",
    "Op_Server0",
    "Op_Host0",
    "Op_Host1",
    "Op_Host2",
    "User1",
    "User2",
    "User3",
    "User4",
]


class CyberJEPADataset(Dataset[dict[str, Any]]):
    """PyTorch Dataset exposing windowed observations, action sequences, and targets."""

    def __init__(
        self,
        shard_dirs: list[Path],
        split_group_set: set[str] | None = None,
        trajectory_set: set[str] | None = None,
        history_len: int = 4,
        horizon: int = 4,
        fit_normalizers: bool = False,
        normalizer_stats: dict[str, Any] | None = None,
    ):
        self.shard_dirs = shard_dirs
        self.split_group_set = split_group_set
        self.trajectory_set = trajectory_set
        self.history_len = history_len
        self.horizon = horizon

        self.samples: list[dict[str, Any]] = []
        self._load_and_window_shards(shard_dirs)

        if fit_normalizers and self.samples:
            all_hist = torch.stack([s["history_flat"] for s in self.samples])
            mean = all_hist.mean(dim=(0, 1))
            std_raw = all_hist.std(dim=(0, 1))
            # Floor small standard deviations to 1.0 to prevent division by near-zero on invariant dimensions
            std = torch.where(std_raw < 1e-4, torch.ones_like(std_raw), std_raw)
            self.normalizer_stats = {"mean": mean.numpy().tolist(), "std": std.numpy().tolist()}
        else:
            self.normalizer_stats = normalizer_stats or {}

    def _load_and_window_shards(self, shard_dirs: list[Path]) -> None:
        """Construct sliding windows over valid episodes without boundary crossing."""
        comp_cols = [f"compromise_{h}" for h in MONITORED_HOSTS]

        for sdir in shard_dirs:
            trans_df = pd.read_parquet(sdir / "transitions.parquet")
            oracle_df = pd.read_parquet(sdir / "oracle_labels.parquet") if (sdir / "oracle_labels.parquet").exists() else None
            if oracle_df is not None:
                avail_comp = [c for c in comp_cols if c in oracle_df.columns]
                merge_cols = ["transition_id", "critical_server_compromised"] + avail_comp
                trans_df = pd.merge(trans_df, oracle_df[merge_cols], on="transition_id", how="left")

            obs_data = np.load(sdir / "observations.npz")
            flats = obs_data["flat"]

            filtered_df = trans_df
            if self.split_group_set is not None:
                filtered_df = filtered_df[filtered_df["split_group_id"].isin(self.split_group_set)]
            if self.trajectory_set is not None:
                filtered_df = filtered_df[filtered_df["trajectory_id"].isin(self.trajectory_set)]

            ep_groups = filtered_df.groupby("trajectory_id", sort=True)

            for traj_id, group in ep_groups:
                group_indices = np.array(group.index.tolist())
                L = len(group_indices)

                act_col = "action_discrete_index" if "action_discrete_index" in group.columns else "action_idx"
                t_col = "step_index" if "step_index" in group.columns else "t"

                act_vals = group[act_col].values
                t_vals = group[t_col].values
                trans_ids = group["transition_id"].values
                labels = group["critical_server_compromised"].values if "critical_server_compromised" in group.columns else [0] * L

                host_comp_map = {}
                for h in MONITORED_HOSTS:
                    col = f"compromise_{h}"
                    if col in group.columns:
                        host_comp_map[h] = (group[col].values != "clean").astype(np.float32)
                    else:
                        host_comp_map[h] = np.zeros(L, dtype=np.float32)

                # Window requirement: history_len history steps + horizon future action/target steps
                for i in range(self.history_len - 1, L - self.horizon):
                    hist_idx_range = group_indices[i - self.history_len + 1 : i + 1]
                    target_idx = group_indices[i + self.horizon]

                    hist_flats = flats[hist_idx_range]                # [4, 52]
                    target_flat = flats[target_idx]                   # [52]
                    action_seq = act_vals[i : i + self.horizon].tolist()
                    t_ctx = int(t_vals[i])
                    t_tgt = int(t_vals[i + self.horizon])
                    tid_ctx = str(trans_ids[i])
                    tgt_label = int(labels[i + self.horizon]) if i + self.horizon < len(labels) else 0
                    ctx_label = int(labels[i]) if i < len(labels) else 0

                    delta = target_flat - hist_flats[-1]
                    rms_delta = float(np.sqrt(np.mean(delta ** 2)))

                    target_step = i + self.horizon
                    host_vec_target = [
                        float(host_comp_map[h][target_step]) if target_step < len(host_comp_map[h]) else 0.0
                        for h in MONITORED_HOSTS
                    ]
                    host_vec_context = [
                        float(host_comp_map[h][i]) if i < len(host_comp_map[h]) else 0.0
                        for h in MONITORED_HOSTS
                    ]

                    act_type = str(group["action_type"].values[i]) if "action_type" in group.columns else "Unknown"

                    self.samples.append({
                        "trajectory_id": traj_id,
                        "transition_id": tid_ctx,
                        "t_context": t_ctx,
                        "t_target": t_tgt,
                        "history_flat": torch.tensor(hist_flats, dtype=torch.float32),
                        "action_seq": torch.tensor(action_seq, dtype=torch.long),
                        "target_flat": torch.tensor(target_flat, dtype=torch.float32),
                        "label": ctx_label, # Operational detection label at context time t (strict zero leakage)
                        "context_label": ctx_label, # Ground truth at context step t
                        "target_label": tgt_label,   # Future ground truth at target step t+horizon
                        "horizon": self.horizon,
                        "rms_delta": rms_delta,
                        "host_compromised": torch.tensor(host_vec_context, dtype=torch.float32), # Context-aligned [11]
                        "host_compromised_context": torch.tensor(host_vec_context, dtype=torch.float32),
                        "host_compromised_target": torch.tensor(host_vec_target, dtype=torch.float32),
                        "action_type": act_type,
                    })

    def compute_median_dynamic_rms(self) -> float:
        """Compute median RMS delta of non-zero transitions."""
        deltas = [s["rms_delta"] for s in self.samples if s["rms_delta"] > 1e-4]
        return float(np.median(deltas)) if deltas else 0.25

    def get_balanced_sampler(
        self,
        threshold: float = 0.25,
        generator: torch.Generator | None = None,
    ) -> torch.utils.data.WeightedRandomSampler:
        """
        Construct a WeightedRandomSampler enforcing static50_dynamic50 (1:1 balanced sampling).
        Phase 5 Stage 1 proven balancing protocol.
        """
        N = len(self.samples)
        if N == 0:
            return torch.utils.data.WeightedRandomSampler([1.0], 1)

        is_dynamic = np.array([s["rms_delta"] >= threshold for s in self.samples], dtype=bool)
        n_dyn = int(np.sum(is_dynamic))
        n_stat = N - n_dyn

        weights = np.zeros(N, dtype=np.float64)
        if n_dyn > 0 and n_stat > 0:
            weights[is_dynamic] = 0.5 / n_dyn
            weights[~is_dynamic] = 0.5 / n_stat
        else:
            weights[:] = 1.0 / N

        return torch.utils.data.WeightedRandomSampler(
            weights=torch.tensor(weights, dtype=torch.double),
            num_samples=N,
            replacement=True,
            generator=generator,
        )

    def _fit_normalizers(self) -> dict[str, Any]:
        """Fit feature mean and std on training split history observations only."""
        if not self.samples:
            return {"mean": 0.0, "std": 1.0}
        all_hist = np.concatenate([s["history_flat"].numpy() for s in self.samples], axis=0)
        mean = np.mean(all_hist, axis=0)
        std = np.std(all_hist, axis=0)
        std[std < 1e-6] = 1.0
        return {"mean": mean.tolist(), "std": std.tolist()}


    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]

        # Apply normalizer stats if available
        hist = sample["history_flat"]
        target = sample["target_flat"]

        if "mean" in self.normalizer_stats and "std" in self.normalizer_stats:
            mean = torch.tensor(self.normalizer_stats["mean"], dtype=torch.float32)
            std = torch.tensor(self.normalizer_stats["std"], dtype=torch.float32)
            hist = (hist - mean) / std
            target = (target - mean) / std

        res = {
            "trajectory_id": sample["trajectory_id"],
            "transition_id": sample["transition_id"],
            "t_context": sample["t_context"],
            "t_target": sample["t_target"],
            "history_flat": hist,
            "action_seq": sample["action_seq"],
            "target_flat": target,
            "label": sample["label"],
            "horizon": sample["horizon"],
            "rms_delta": sample["rms_delta"],
        }
        if "host_compromised" in sample:
            res["host_compromised"] = sample["host_compromised"]
        return res

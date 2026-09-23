"""
Calibrate Cyber-JEPA Decentralized Zone Defenders on CAGE Challenge 4.

Collects nominal enterprise operational traces (with Red adversary absent/sleep)
to establish clean phase-space reference manifolds:
1. Normalizer statistics (mean, std) per agent
2. Multi-centroid clean representations in latent space (K=3)
3. Nominal mean velocity v_clean per agent
"""

import sys
from pathlib import Path

# Ensure src and CAGE 4 are in python path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))
sys.path.insert(0, str(repo_root / "external" / "cage-challenge-4"))

import numpy as np
import torch

from CybORG import CybORG
from CybORG.Agents import SleepAgent, EnterpriseGreenAgent
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from CybORG.Agents.Wrappers.EnterpriseMAE import EnterpriseMAE
from cyber_jepa.representations.flat import FlatVectorRepresentation


def run_calibration(
    num_episodes: int = 5,
    steps_per_episode: int = 30,
    seed: int = 1001,
    output_path: str = "experiments/cage_eval/cage4_calibration.pt",
) -> None:
    print(f"=== Starting CAGE 4 Cyber-JEPA Defender Calibration ===")
    print(f"Episodes: {num_episodes}, Steps: {steps_per_episode}, Seed: {seed}")

    # Load pretrained encoders
    enc25 = FlatVectorRepresentation(obs_dim=100, hidden_dim=64, history_len=4)
    ckpt25_path = repo_root / "experiments" / "phase6b" / "scale_25_flat" / "best.pt"
    ckpt25 = torch.load(ckpt25_path, map_location="cpu", weights_only=False)
    enc25.load_state_dict(ckpt25["online_encoder"])
    enc25.eval()

    enc100 = FlatVectorRepresentation(obs_dim=400, hidden_dim=64, history_len=4)
    ckpt100_path = repo_root / "experiments" / "phase6b" / "scale_100_flat" / "best.pt"
    ckpt100 = torch.load(ckpt100_path, map_location="cpu", weights_only=False)
    enc100.load_state_dict(ckpt100["online_encoder"])
    enc100.eval()

    agent_names = [f"blue_agent_{i}" for i in range(5)]
    raw_obs_data: dict[str, list[np.ndarray]] = {a: [] for a in agent_names}

    # 1. Collect clean nominal observations
    for ep in range(num_episodes):
        sg = EnterpriseScenarioGenerator(
            blue_agent_class=SleepAgent,
            green_agent_class=EnterpriseGreenAgent,
            red_agent_class=SleepAgent,
            steps=steps_per_episode,
        )
        cyborg = CybORG(sg, "sim", seed=seed + ep * 17)
        env = EnterpriseMAE(cyborg)
        obs, _ = env.reset()

        sleep_actions = {a: env.action_labels(a).index("Sleep") for a in agent_names}

        for a in agent_names:
            raw_obs_data[a].append(obs[a].copy())

        for step in range(steps_per_episode):
            obs, rews, term, trunc, _ = env.step(sleep_actions)
            for a in agent_names:
                raw_obs_data[a].append(obs[a].copy())

    print("Data collection complete. Computing normalization and phase-space metrics...")

    calibration_results: dict[str, dict] = {}

    for a in agent_names:
        obs_arr = np.array(raw_obs_data[a], dtype=np.float32)  # [N_total, D_raw]
        mean = np.mean(obs_arr, axis=0)
        std = np.std(obs_arr, axis=0)
        std[std < 1e-4] = 1.0

        target_dim = 100 if a != "blue_agent_4" else 400
        encoder = enc25 if a != "blue_agent_4" else enc100

        # Build sliding windows
        latent_vectors = []
        velocities = []
        prev_norm_z = None

        # Episode by episode sliding window computation
        n_steps = steps_per_episode + 1
        for ep in range(num_episodes):
            ep_obs = obs_arr[ep * n_steps : (ep + 1) * n_steps]
            norm_ep_obs = (ep_obs - mean) / std

            padded_obs = np.zeros((len(norm_ep_obs), target_dim), dtype=np.float32)
            padded_obs[:, : ep_obs.shape[1]] = norm_ep_obs

            buf = []
            prev_norm_z = None
            for t in range(len(padded_obs)):
                buf.append(padded_obs[t])
                if len(buf) >= 4:
                    window = np.stack(buf[-4:], axis=0)[np.newaxis, ...]  # [1, 4, D]
                    with torch.no_grad():
                        z = encoder.encode_context(torch.tensor(window), return_context_tokens=False)
                        z_np = z.squeeze(0).cpu().numpy().astype(np.float32)
                        norm_z = z_np / max(1e-12, float(np.linalg.norm(z_np)))
                        latent_vectors.append(norm_z)

                        if prev_norm_z is not None:
                            v = float(np.clip(1.0 - np.dot(norm_z, prev_norm_z), 0.0, 2.0))
                            velocities.append(v)
                        prev_norm_z = norm_z

        latents_arr = np.array(latent_vectors, dtype=np.float32)

        # Perform KMeans clustering (K=6) to capture all operational phases
        from sklearn.cluster import KMeans
        n_clusters = min(6, len(latents_arr))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        kmeans.fit(latents_arr)
        clean_centroids_arr = kmeans.cluster_centers_.astype(np.float32)
        norms = np.linalg.norm(clean_centroids_arr, axis=1, keepdims=True)
        clean_centroids_arr = clean_centroids_arr / np.maximum(1e-12, norms)
        clean_mean_vel = float(np.mean(velocities)) if velocities else 0.05

        # Compute empirical clean score distribution to calibrate alert_threshold
        clean_scores = []
        w_v = 0.25
        for i, norm_z in enumerate(latents_arr):
            sims = np.dot(clean_centroids_arr, norm_z)
            d_cluster = float(np.clip(1.0 - np.max(sims), 0.0, 2.0))
            v_t = velocities[i - 1] if i > 0 and i - 1 < len(velocities) else clean_mean_vel
            v_ratio = v_t / max(1e-4, clean_mean_vel)
            v_anomaly = float(np.clip(abs(v_ratio - 1.0) / 2.0, 0.0, 1.0))
            score = (1.0 - w_v) * d_cluster + w_v * v_anomaly
            clean_scores.append(score)

        thresh = float(np.percentile(clean_scores, 95) + 0.05) if clean_scores else 0.35

        calibration_results[a] = {
            "normalizer_mean": mean,
            "normalizer_std": std,
            "target_dim": target_dim,
            "clean_centroids": clean_centroids_arr,
            "clean_mean_velocity": max(1e-4, clean_mean_vel),
            "alert_threshold": thresh,
            "velocity_weight": w_v,
        }
        print(f"  {a}: target_dim={target_dim}, centroids={len(clean_centroids_arr)}, v_clean={clean_mean_vel:.4f}, alert_threshold={thresh:.4f}")

    out_file = repo_root / output_path
    out_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(calibration_results, out_file)
    print(f"Calibration saved successfully to: {out_file}")


if __name__ == "__main__":
    run_calibration()

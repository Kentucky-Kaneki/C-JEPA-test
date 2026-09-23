"""
Autonomous Cyber-JEPA Blue Defender Agent for CybORG CAGE Challenges.

Operates in closed-loop CybORG simulation:
1. Sliding Observation Buffer: Maintains temporal history window [H, obs_dim] (default H=4).
2. Online Latent Encoder: Maps context window into self-supervised latent state z_t.
3. Multi-Centroid Phase-Space Anomaly Detector: Computes operational threat score
   S_t = (1 - w_v) * d_cluster(z_t) + w_v * v_anomaly(z_t).
4. Host Compromise Localization: Uses linear probes to estimate P(host_h compromised | z_t).
5. Proactive Cost-Aware Action Selection:
   - S_t < tau (Nominal): Sleep (0 operational penalty).
   - S_t >= tau (Alert): Targets localized host with Restore (Op_Server/Enterprise)
     or Remove (User) before killchain maturation.
"""

from collections import deque
from pathlib import Path
from typing import Any
import numpy as np
import torch

from CybORG.Agents import BaseAgent
from CybORG.Simulator.Actions import Action, Analyse, Remove, Restore, Sleep

from cyber_jepa.data.dataset import MONITORED_HOSTS
from cyber_jepa.env.action_mapper import ActionMapper
from cyber_jepa.models.jepa import CyberJEPA
from cyber_jepa.representations.flat import FlatVectorRepresentation


# Tactical Priority for Defensive Intervention:
# Operational Tier (Op_Server0, Op_Hosts) > Enterprise Tier > User Tier
TIER_PRIORITY = {
    "Op_Server0": 100,
    "Op_Host0": 80,
    "Op_Host1": 80,
    "Op_Host2": 80,
    "Enterprise0": 60,
    "Enterprise1": 60,
    "Enterprise2": 60,
    "User1": 30,
    "User2": 30,
    "User3": 30,
    "User4": 30,
    "Defender": 10,
    "User0": 5,
}


class CyberJEPADefender(BaseAgent):
    """Closed-loop autonomous defender agent driven by Cyber-JEPA world model."""

    def __init__(
        self,
        name: str = "CyberJEPADefender",
        model: CyberJEPA | None = None,
        action_mapper: ActionMapper | None = None,
        obs_dim: int = 52,
        history_len: int = 4,
        alert_threshold: float = 0.52,
        velocity_weight: float = 0.25,
        clean_mean_velocity: float = 0.059,
        clean_centroids: np.ndarray | None = None,
        host_probes: dict[str, tuple[np.ndarray, float]] | None = None,
        normalizer_mean: np.ndarray | None = None,
        normalizer_std: np.ndarray | None = None,
        min_probe_prob: float = 0.30,
        restore_cooldown_steps: int = 2,
        device: str | torch.device = "cpu",
    ):
        super().__init__(name=name)
        self.obs_dim = obs_dim
        self.history_len = history_len
        self.alert_threshold = alert_threshold
        self.velocity_weight = velocity_weight
        self.clean_mean_velocity = clean_mean_velocity
        self.min_probe_prob = min_probe_prob
        self.restore_cooldown_steps = restore_cooldown_steps
        self.device = torch.device(device)

        # Model and Action Mapper
        self.model = model
        if self.model is not None:
            self.model.to(self.device).eval()
        self.action_mapper = action_mapper

        # Multi-Centroid Clean Reference (shape [K, D])
        if clean_centroids is not None:
            norms = np.linalg.norm(clean_centroids, axis=1, keepdims=True)
            self.clean_centroids = clean_centroids / np.maximum(1e-12, norms)
        else:
            self.clean_centroids = None

        # Host Localization Probes: dict of host -> (weight_vector [D], bias float)
        self.host_probes = host_probes or {}

        # Normalization parameters
        self.normalizer_mean = normalizer_mean if normalizer_mean is not None else np.zeros(obs_dim, dtype=np.float32)
        self.normalizer_std = normalizer_std if normalizer_std is not None else np.ones(obs_dim, dtype=np.float32)

        # Ephemeral runtime state
        self.obs_buffer: deque[np.ndarray] = deque(maxlen=history_len)
        self.prev_norm_z: np.ndarray | None = None
        self.cooldowns: dict[str, int] = {}
        self.step_count = 0
        self.last_action_name = "Sleep"
        self.last_target_host: str | None = None
        self.last_diagnostics: dict[str, Any] = {}
        self._step_evaluated: bool = False

    def end_episode(self) -> None:
        """Reset internal episodic buffers on episode boundary."""
        self.obs_buffer.clear()
        self.prev_norm_z = None
        self.cooldowns.clear()
        self.step_count = 0
        self.last_action_name = "Sleep"
        self.last_target_host = None
        self.last_diagnostics.clear()
        self._step_evaluated = False

    def set_initial_values(self, action_space: Any, observation: Any) -> None:
        """Initialize action space and initial observation from CybORG."""
        self.end_episode()

    def _extract_flat_obs(self, observation: Any) -> np.ndarray:
        """Extract a 1D float vector of length obs_dim from observation."""
        if isinstance(observation, np.ndarray):
            flat = observation.flatten().astype(np.float32)
        elif isinstance(observation, (list, tuple)):
            flat = np.array(observation, dtype=np.float32).flatten()
        elif isinstance(observation, dict):
            # If CybORG returned raw dict, try finding vectorized key or return zeros
            if "vector" in observation:
                flat = np.array(observation["vector"], dtype=np.float32).flatten()
            else:
                flat = np.zeros(self.obs_dim, dtype=np.float32)
        else:
            flat = np.zeros(self.obs_dim, dtype=np.float32)

        if len(flat) < self.obs_dim:
            padded = np.zeros(self.obs_dim, dtype=np.float32)
            padded[: len(flat)] = flat
            return padded
        return flat[: self.obs_dim]

    def _normalize_obs(self, raw_obs: np.ndarray) -> np.ndarray:
        """Normalize raw observation using precomputed mean and standard deviation."""
        return (raw_obs - self.normalizer_mean) / np.maximum(1e-4, self.normalizer_std)

    def _ingest_observation(self, observation: Any) -> None:
        """Push normalized observation into sliding history window."""
        flat_obs = self._extract_flat_obs(observation)
        norm_obs = self._normalize_obs(flat_obs)

        # If buffer is empty, fill with initial observation copies
        if len(self.obs_buffer) == 0:
            for _ in range(self.history_len):
                self.obs_buffer.append(norm_obs)
        else:
            self.obs_buffer.append(norm_obs)

    def evaluate_state(self, observation: Any, force: bool = False) -> dict[str, Any]:
        """
        Ingest observation and compute latent state, anomaly score, and host compromise probabilities.
        Guards against duplicate ingestion within the same step unless force=True.
        """
        if self._step_evaluated and not force:
            return self.last_diagnostics

        self._ingest_observation(observation)
        self.step_count += 1

        # Decrement cooldowns
        for h in list(self.cooldowns.keys()):
            self.cooldowns[h] -= 1
            if self.cooldowns[h] <= 0:
                del self.cooldowns[h]

        if self.model is None:
            # Fallback for baseline or mock testing
            diag = {
                "score": 0.0,
                "d_cluster": 0.0,
                "v_anomaly": 0.0,
                "host_probs": {h: 0.0 for h in MONITORED_HOSTS},
            }
            self.last_diagnostics = diag
            self._step_evaluated = True
            return diag

        # Format history window: [1, H, obs_dim]
        window_np = np.stack(list(self.obs_buffer), axis=0)[np.newaxis, ...]
        window_t = torch.tensor(window_np, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            latent_t = self.model.online_encoder.encode_context(window_t, return_context_tokens=False)
            z_t = latent_t.squeeze(0).cpu().numpy().astype(np.float32)

        norm_z = z_t / max(1e-12, float(np.linalg.norm(z_t)))

        # 1. Cluster Distance
        if self.clean_centroids is not None and len(self.clean_centroids) > 0:
            sims = np.dot(self.clean_centroids, norm_z)
            d_cluster = float(np.clip(1.0 - np.max(sims), 0.0, 2.0))
        else:
            d_cluster = 0.0

        # 2. Kinematic Velocity & Deceleration Anomaly
        if self.prev_norm_z is not None:
            v_t = float(np.clip(1.0 - np.dot(norm_z, self.prev_norm_z), 0.0, 2.0))
            v_clean_ref = max(1e-6, self.clean_mean_velocity)
            v_anomaly = float(np.clip(1.0 - (v_t / v_clean_ref), 0.0, 1.0))
        else:
            v_t = 0.0
            v_anomaly = 0.0

        self.prev_norm_z = norm_z

        # 3. Joint Phase-Space Score
        w_v = self.velocity_weight
        score = float((1.0 - w_v) * d_cluster + w_v * v_anomaly)

        # 4. Multi-Host Compromise Probabilities
        host_probs: dict[str, float] = {}
        for h, probe in self.host_probes.items():
            if isinstance(probe, tuple):
                w, b = probe
                logit = float(np.dot(w, z_t) + b)
                prob = float(1.0 / (1.0 + np.exp(-np.clip(logit, -15.0, 15.0))))
            elif hasattr(probe, "predict_proba"):
                prob = float(probe.predict_proba(z_t.reshape(1, -1))[0, 1])
            else:
                prob = 0.0
            host_probs[h] = prob

        diag = {
            "score": score,
            "d_cluster": d_cluster,
            "v_t": v_t,
            "v_anomaly": v_anomaly,
            "host_probs": host_probs,
        }
        self.last_diagnostics = diag
        self._step_evaluated = True
        return diag

    def select_action(self, observation: Any) -> tuple[str, str | None]:
        """
        Evaluate state and select high-level action type and target hostname.

        Returns:
            (action_type, target_hostname) e.g. ("Restore", "Op_Server0") or ("Sleep", None)
        """
        diag = self.evaluate_state(observation)
        score = diag["score"]
        host_probs = diag["host_probs"]

        # Case 1: Nominal Network (S_t < tau) -> Sleep
        if score < self.alert_threshold:
            self.last_action_name = "Sleep"
            self.last_target_host = None
            return "Sleep", None

        # Case 2: Threat Detected (S_t >= tau) -> Target localized compromised host
        # Filter candidate hosts that exceed minimum probability and are not on cooldown
        candidates = [
            (h, prob)
            for h, prob in host_probs.items()
            if prob >= self.min_probe_prob and self.cooldowns.get(h, 0) <= 0
        ]

        if not candidates:
            # If alert fired but no specific host has high probability, Analyse most likely host
            sorted_all = sorted(host_probs.items(), key=lambda kv: kv[1], reverse=True)
            if sorted_all and sorted_all[0][1] > 0.15:
                best_h = sorted_all[0][0]
                self.last_action_name = "Analyse"
                self.last_target_host = best_h
                return "Analyse", best_h
            self.last_action_name = "Sleep"
            self.last_target_host = None
            return "Sleep", None

        # Sort candidates by combined tactical priority and compromise probability
        # Priority score = tier_priority + prob * 50
        def _candidate_score(item: tuple[str, float]) -> float:
            h, prob = item
            tier_score = TIER_PRIORITY.get(h, 20)
            return tier_score + (prob * 50.0)

        candidates.sort(key=_candidate_score, reverse=True)
        target_host, best_prob = candidates[0]

        # Determine intervention type based on tactical tier
        if target_host in ("Op_Server0", "Op_Host0", "Op_Host1", "Op_Host2"):
            # Crown Jewel or Operational host: immediate Restore
            action_type = "Restore"
            self.cooldowns[target_host] = self.restore_cooldown_steps
        elif target_host in ("Enterprise0", "Enterprise1", "Enterprise2"):
            # Enterprise host: Restore if high probability, else Remove
            if best_prob >= 0.55:
                action_type = "Restore"
                self.cooldowns[target_host] = self.restore_cooldown_steps
            else:
                action_type = "Remove"
        else:
            # User hosts (perimeter breach): Remove session
            action_type = "Remove"

        self.last_action_name = action_type
        self.last_target_host = target_host
        return action_type, target_host

    def get_action_index(self, observation: Any) -> int:
        """
        Return discrete action index (0..65) for OpenAI Gym / ChallengeWrapper environments.
        """
        action_type, target_host = self.select_action(observation)
        self._step_evaluated = False
        if self.action_mapper is not None:
            return self.action_mapper.find_action_index(action_type=action_type, hostname=target_host)
        # Fallback to Sleep if no action mapper provided
        return 0

    def get_action(self, observation: Any, action_space: Any = None) -> Action:
        """
        Return CybORG Action instance for native CybORG BaseAgent compatibility.
        """
        action_type, target_host = self.select_action(observation)
        self._step_evaluated = False

        if self.action_mapper is not None:
            act_idx = self.action_mapper.find_action_index(action_type=action_type, hostname=target_host)
            return self.action_mapper.get_action_object(act_idx)

        # Fallback instantiation
        if action_type == "Restore" and target_host:
            return Restore(hostname=target_host, session=0, agent="Blue")
        if action_type == "Remove" and target_host:
            return Remove(hostname=target_host, session=0, agent="Blue")
        if action_type == "Analyse" and target_host:
            return Analyse(hostname=target_host, session=0, agent="Blue")
        return Sleep()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        possible_actions: list[Action] | None = None,
        calibration_path: str | Path | None = None,
        obs_dim: int = 52,
        hidden_dim: int = 64,
        alert_threshold: float = 0.52,
        velocity_weight: float = 0.25,
        clean_mean_velocity: float = 0.059,
        clean_centroids: np.ndarray | None = None,
        host_probes: dict[str, tuple[np.ndarray, float]] | None = None,
        normalizer_stats: dict[str, Any] | None = None,
        device: str | torch.device = "cpu",
    ) -> "CyberJEPADefender":
        """
        Instantiate CyberJEPADefender from a saved Cyber-JEPA checkpoint file and optional calibration file.
        """
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

        flat_enc = FlatVectorRepresentation(obs_dim=obs_dim, hidden_dim=hidden_dim, history_len=4)
        model = CyberJEPA(online_encoder=flat_enc, hidden_dim=hidden_dim, max_horizon=4)

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.online_encoder.load_state_dict(ckpt["online_encoder"])
        if "target_encoder" in ckpt:
            model.target_encoder.load_state_dict(ckpt["target_encoder"])
        if "predictor" in ckpt:
            model.predictor.load_state_dict(ckpt["predictor"])

        action_mapper = ActionMapper(possible_actions) if possible_actions is not None else None

        mean = normalizer_stats.get("mean") if normalizer_stats else None
        std = normalizer_stats.get("std") if normalizer_stats else None

        # Load precomputed calibration artifacts if available
        if calibration_path is not None:
            calib_p = Path(calibration_path)
            if not calib_p.exists():
                raise FileNotFoundError(f"Calibration file not found at: {calib_p}")
            calib = torch.load(calib_p, map_location="cpu", weights_only=False)
            if clean_centroids is None and "clean_centroids" in calib:
                clean_centroids = calib["clean_centroids"]
            if "clean_mean_velocity" in calib:
                clean_mean_velocity = float(calib["clean_mean_velocity"])
            if host_probes is None and "host_probes" in calib:
                host_probes = calib["host_probes"]
            if "alert_threshold" in calib:
                alert_threshold = float(calib["alert_threshold"])
            if "velocity_weight" in calib:
                velocity_weight = float(calib["velocity_weight"])
            if mean is None and "normalizer_mean" in calib:
                mean = calib["normalizer_mean"]
            if std is None and "normalizer_std" in calib:
                std = calib["normalizer_std"]

        return cls(
            model=model,
            action_mapper=action_mapper,
            obs_dim=obs_dim,
            alert_threshold=alert_threshold,
            velocity_weight=velocity_weight,
            clean_mean_velocity=clean_mean_velocity,
            clean_centroids=clean_centroids,
            host_probes=host_probes,
            normalizer_mean=mean,
            normalizer_std=std,
            device=device,
        )


"""
Decentralized Multi-Agent Cyber-JEPA Defender for CAGE Challenge 4.

Implements autonomous blue defense across 5 distributed enterprise zones:
- blue_agent_0: Deployed Net A - Restricted
- blue_agent_1: Deployed Net A - Operational
- blue_agent_2: Deployed Net B - Restricted
- blue_agent_3: Deployed Net B - Operational
- blue_agent_4: Headquarters (Admin, Office, Public subnets)

Architectural Principles:
1. Strict Decentralized Autonomy: Zero cross-agent communication required; each zone
   defender maintains local temporal history and phase-space manifold.
2. Self-Supervised Anomaly Guidance: Identifies subtle phase-space drift (S_t >= tau)
   before adversary achieves mission impact.
3. Operational Cost-Awareness: Enforces Sleep when nominal to eliminate disruption
   to Green enterprise operations (preventing Local Work Fail penalties).
4. Cooldown-Protected Targeted Interventions: Executes targeted Restore or Remove on
   compromised hosts with cooldown damping.
"""

from collections import deque
from pathlib import Path
from typing import Any
import numpy as np
import torch

from CybORG.Agents import BaseAgent
from CybORG.Simulator.Actions import Sleep

from cyber_jepa.representations.flat import FlatVectorRepresentation
from cyber_jepa.models.jepa import CyberJEPA


class ZoneJEPADefender(BaseAgent):
    """Decentralized single-zone defender driven by a Cyber-JEPA world model."""

    def __init__(
        self,
        name: str,
        agent_id: str,
        model: CyberJEPA | None = None,
        obs_dim: int = 100,
        raw_obs_dim: int = 92,
        history_len: int = 4,
        alert_threshold: float = 0.35,
        velocity_weight: float = 0.25,
        clean_mean_velocity: float = 0.12,
        clean_centroids: np.ndarray | None = None,
        normalizer_mean: np.ndarray | None = None,
        normalizer_std: np.ndarray | None = None,
        restore_cooldown_steps: int = 2,
        device: str | torch.device = "cpu",
    ):
        super().__init__(name=name)
        self.agent_id = agent_id
        self.obs_dim = obs_dim
        self.raw_obs_dim = raw_obs_dim
        self.history_len = history_len
        self.alert_threshold = alert_threshold
        self.velocity_weight = velocity_weight
        self.clean_mean_velocity = clean_mean_velocity
        self.restore_cooldown_steps = restore_cooldown_steps
        self.device = torch.device(device)

        self.model = model
        if self.model is not None:
            self.model.to(self.device).eval()

        if clean_centroids is not None:
            norms = np.linalg.norm(clean_centroids, axis=1, keepdims=True)
            self.clean_centroids = clean_centroids / np.maximum(1e-12, norms)
        else:
            self.clean_centroids = None

        self.normalizer_mean = normalizer_mean if normalizer_mean is not None else np.zeros(raw_obs_dim, dtype=np.float32)
        self.normalizer_std = normalizer_std if normalizer_std is not None else np.ones(raw_obs_dim, dtype=np.float32)

        # Runtime state
        self.obs_buffer: deque[np.ndarray] = deque(maxlen=history_len)
        self.prev_norm_z: np.ndarray | None = None
        self.cooldowns: dict[str, int] = {}
        self.step_count = 0
        self.last_action_idx = 0
        self.last_action_label = "Sleep"
        self.last_diagnostics: dict[str, Any] = {}

    def end_episode(self) -> None:
        """Reset internal episodic state."""
        self.obs_buffer.clear()
        self.prev_norm_z = None
        self.cooldowns.clear()
        self.step_count = 0
        self.last_action_idx = 0
        self.last_action_label = "Sleep"
        self.last_diagnostics.clear()

    def set_initial_values(self, action_space: Any, observation: Any) -> None:
        self.end_episode()

    def _normalize_and_pad(self, raw_obs: np.ndarray) -> np.ndarray:
        """Normalize raw observation vector and zero-pad to model input dimension."""
        flat = raw_obs.flatten().astype(np.float32)
        norm = (flat - self.normalizer_mean) / np.maximum(1e-4, self.normalizer_std)
        padded = np.zeros(self.obs_dim, dtype=np.float32)
        padded[: len(norm)] = norm
        return padded

    def evaluate_state(self, raw_obs: np.ndarray) -> dict[str, Any]:
        """Ingest observation into sliding window and compute latent phase-space anomaly score."""
        padded_norm = self._normalize_and_pad(raw_obs)

        if len(self.obs_buffer) == 0:
            for _ in range(self.history_len):
                self.obs_buffer.append(padded_norm)
        else:
            self.obs_buffer.append(padded_norm)

        self.step_count += 1

        # Decrement host cooldowns
        for h in list(self.cooldowns.keys()):
            self.cooldowns[h] -= 1
            if self.cooldowns[h] <= 0:
                del self.cooldowns[h]

        if self.model is None:
            diag = {"score": 0.0, "d_cluster": 0.0, "v_t": 0.0, "v_anomaly": 0.0}
            self.last_diagnostics = diag
            return diag

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

        # 2. Latent Kinematics
        if self.prev_norm_z is not None:
            v_t = float(np.clip(1.0 - np.dot(norm_z, self.prev_norm_z), 0.0, 2.0))
            v_ref = max(1e-4, self.clean_mean_velocity)
            # Both deceleration and sudden kinematic jumps indicate transition
            v_ratio = v_t / v_ref
            v_anomaly = float(np.clip(abs(v_ratio - 1.0) / 2.0, 0.0, 1.0))
        else:
            v_t = 0.0
            v_anomaly = 0.0

        self.prev_norm_z = norm_z

        w_v = self.velocity_weight
        score = float((1.0 - w_v) * d_cluster + w_v * v_anomaly)

        diag = {
            "score": score,
            "d_cluster": d_cluster,
            "v_t": v_t,
            "v_anomaly": v_anomaly,
        }
        self.last_diagnostics = diag
        return diag

    def get_action(
        self,
        observation: np.ndarray,
        action_space: Any = None,
        action_labels: list[str] | None = None,
        action_mask: list[bool] | None = None,
    ) -> int:
        """Select action index based on phase-space state and local observation telemetry."""
        diag = self.evaluate_state(observation)
        score = diag["score"]

        # Parse actions
        labels = action_labels or []
        mask = action_mask or ([True] * len(labels))

        sleep_idx = 0
        restore_map: dict[str, int] = {}
        remove_map: dict[str, int] = {}
        analyse_map: dict[str, int] = {}
        decoy_map: dict[str, int] = {}

        for idx, (label, is_valid) in enumerate(zip(labels, mask)):
            if not is_valid:
                continue
            parts = label.split()
            act_type = parts[0]
            if act_type == "Sleep":
                sleep_idx = idx
            elif act_type == "Restore" and len(parts) > 1:
                restore_map[parts[1]] = idx
            elif act_type == "Remove" and len(parts) > 1:
                remove_map[parts[1]] = idx
            elif act_type == "Analyse" and len(parts) > 1:
                analyse_map[parts[1]] = idx
            elif act_type == "DeployDecoy" and len(parts) > 1:
                decoy_map[parts[1]] = idx

        # Direct telemetry anomaly inspection from raw observation
        # In CAGE 4 BlueFlatWrapper:
        # Note: proc_flags captures all process creation including benign Green user activity.
        # Unauthorized network connections (conn_flags) specifically indicate adversary activity.
        # Anomaly Detection is strictly governed by Cyber-JEPA phase-space score
        # Benign background traffic produces normal process and network events.
        # Cyber-JEPA detects when the temporal joint distribution departs from nominal.
        has_phase_alert = (score >= self.alert_threshold) and (self.step_count > self.history_len)

        # Case 1: Nominal Network (S_t < tau) -> Sleep
        if not has_phase_alert:
            self.last_action_idx = sleep_idx
            self.last_action_label = "Sleep"
            return sleep_idx

        # Case 2: Threat Alert -> Select tactical remediation
        # Prioritize servers (server_host) > user workstations (user_host)
        all_hosts = list(restore_map.keys())
        server_hosts = [h for h in all_hosts if "server" in h and self.cooldowns.get(h, 0) <= 0]
        user_hosts = [h for h in all_hosts if "user" in h and self.cooldowns.get(h, 0) <= 0]

        # In Operational zones (blue_agent_1, blue_agent_3), prioritize user machines where adversary gains footholds
        # In Restricted / HQ zones (blue_agent_0, blue_agent_2, blue_agent_4), prioritize mission-critical servers
        is_operational = "operational" in self.agent_id or "blue_agent_1" in self.agent_id or "blue_agent_3" in self.agent_id
        if is_operational:
            candidate_pool = user_hosts + server_hosts
        else:
            candidate_pool = server_hosts + user_hosts

        if candidate_pool:
            target_host = candidate_pool[0]
            # If server, Restore
            if target_host in restore_map:
                chosen_idx = restore_map[target_host]
                self.cooldowns[target_host] = self.restore_cooldown_steps
                self.last_action_idx = chosen_idx
                self.last_action_label = f"Restore {target_host}"
                return chosen_idx
            elif target_host in remove_map:
                chosen_idx = remove_map[target_host]
                self.cooldowns[target_host] = self.restore_cooldown_steps
                self.last_action_idx = chosen_idx
                self.last_action_label = f"Remove {target_host}"
                return chosen_idx

        # Fallback if candidates on cooldown: DeployDecoy or Analyse on primary server
        if server_hosts and server_hosts[0] in decoy_map:
            dec_host = server_hosts[0]
            chosen_idx = decoy_map[dec_host]
            self.last_action_idx = chosen_idx
            self.last_action_label = f"DeployDecoy {dec_host}"
            return chosen_idx

        if all_hosts and all_hosts[0] in analyse_map:
            an_host = all_hosts[0]
            chosen_idx = analyse_map[an_host]
            self.last_action_idx = chosen_idx
            self.last_action_label = f"Analyse {an_host}"
            return chosen_idx

        # Default fallback to Sleep
        self.last_action_idx = sleep_idx
        self.last_action_label = "Sleep"
        return sleep_idx


class CAGE4CyberJEPADefender:
    """Decentralized multi-agent defender coordinating 5 autonomous zone models."""

    def __init__(self, zone_defenders: dict[str, ZoneJEPADefender]):
        self.zone_defenders = zone_defenders

    @classmethod
    def from_checkpoints(
        cls,
        scale_25_checkpoint: str | Path = "experiments/phase6b/scale_25_flat/best.pt",
        scale_100_checkpoint: str | Path = "experiments/phase6b/scale_100_flat/best.pt",
        calibration_path: str | Path = "experiments/cage_eval/cage4_calibration.pt",
        device: str | torch.device = "cpu",
    ) -> "CAGE4CyberJEPADefender":
        """Instantiate decentralized defender team from pretrained checkpoints and calibration."""
        dev = torch.device(device)

        # 1. Load scale 25 model for agents 0..3 (obs_dim=100)
        enc25 = FlatVectorRepresentation(obs_dim=100, hidden_dim=64, history_len=4)
        model25 = CyberJEPA(online_encoder=enc25, hidden_dim=64, max_horizon=4)
        ckpt25 = torch.load(scale_25_checkpoint, map_location=dev, weights_only=False)
        model25.online_encoder.load_state_dict(ckpt25["online_encoder"])
        model25.eval()

        # 2. Load scale 100 model for agent 4 (obs_dim=400)
        enc100 = FlatVectorRepresentation(obs_dim=400, hidden_dim=64, history_len=4)
        model100 = CyberJEPA(online_encoder=enc100, hidden_dim=64, max_horizon=4)
        ckpt100 = torch.load(scale_100_checkpoint, map_location=dev, weights_only=False)
        model100.online_encoder.load_state_dict(ckpt100["online_encoder"])
        model100.eval()

        # 3. Load calibration
        calib_data = {}
        calib_p = Path(calibration_path)
        if calib_p.exists():
            calib_data = torch.load(calib_p, map_location="cpu", weights_only=False)

        defenders: dict[str, ZoneJEPADefender] = {}
        for i in range(5):
            agent_id = f"blue_agent_{i}"
            is_hq = (i == 4)
            model = model100 if is_hq else model25
            obs_dim = 400 if is_hq else 100
            raw_obs_dim = 210 if is_hq else 92

            c_info = calib_data.get(agent_id, {})
            clean_centroids = c_info.get("clean_centroids")
            clean_mean_vel = c_info.get("clean_mean_velocity", 0.12)
            normalizer_mean = c_info.get("normalizer_mean")
            normalizer_std = c_info.get("normalizer_std")
            alert_threshold = c_info.get("alert_threshold", 0.35)

            defenders[agent_id] = ZoneJEPADefender(
                name=f"CyberJEPA_{agent_id}",
                agent_id=agent_id,
                model=model,
                obs_dim=obs_dim,
                raw_obs_dim=raw_obs_dim,
                history_len=4,
                alert_threshold=alert_threshold,
                velocity_weight=0.25,
                clean_mean_velocity=clean_mean_vel,
                clean_centroids=clean_centroids,
                normalizer_mean=normalizer_mean,
                normalizer_std=normalizer_std,
                restore_cooldown_steps=2,
                device=dev,
            )

        return cls(defenders)

    def get_actions(
        self,
        observations: dict[str, np.ndarray],
        action_spaces: dict[str, Any] | None = None,
        action_labels_dict: dict[str, list[str]] | None = None,
        info: dict[str, dict] | None = None,
    ) -> dict[str, int]:
        """Compute actions for all active blue agents."""
        actions: dict[str, int] = {}
        for agent_id, defender in self.zone_defenders.items():
            if agent_id not in observations:
                continue
            obs = observations[agent_id]
            labels = action_labels_dict.get(agent_id) if action_labels_dict else None
            mask = info.get(agent_id, {}).get("action_mask") if info else None
            act_space = action_spaces.get(agent_id) if action_spaces else None

            act_idx = defender.get_action(
                observation=obs,
                action_space=act_space,
                action_labels=labels,
                action_mask=mask,
            )
            actions[agent_id] = act_idx
        return actions

    def end_episode(self) -> None:
        """Reset internal episodic state for all zone defenders."""
        for defender in self.zone_defenders.values():
            defender.end_episode()


class CAGE4ReactiveDefender:
    """Reactive heuristic baseline defender responding strictly to overt telemetry."""

    def __init__(self, restore_cooldown_steps: int = 2):
        self.restore_cooldown_steps = restore_cooldown_steps
        self.cooldowns: dict[str, dict[str, int]] = {f"blue_agent_{i}": {} for i in range(5)}

    def end_episode(self) -> None:
        self.cooldowns = {f"blue_agent_{i}": {} for i in range(5)}

    def get_actions(
        self,
        observations: dict[str, np.ndarray],
        action_spaces: dict[str, Any] | None = None,
        action_labels_dict: dict[str, list[str]] | None = None,
        info: dict[str, dict] | None = None,
    ) -> dict[str, int]:
        actions: dict[str, int] = {}
        for agent_id in observations:
            obs = observations[agent_id]
            labels = action_labels_dict.get(agent_id, []) if action_labels_dict else []
            mask = info.get(agent_id, {}).get("action_mask") if info else [True] * len(labels)

            sleep_idx = labels.index("Sleep") if "Sleep" in labels else 0
            restore_map: dict[str, int] = {}
            for idx, (label, valid) in enumerate(zip(labels, mask)):
                if valid and label.startswith("Restore "):
                    restore_map[label.split()[1]] = idx

            # Update cooldowns
            for h in list(self.cooldowns[agent_id].keys()):
                self.cooldowns[agent_id][h] -= 1
                if self.cooldowns[agent_id][h] <= 0:
                    del self.cooldowns[agent_id][h]

            # Telemetry check: unauthorized network connections indicate adversary activity
            # In BlueFlatWrapper:
            # 92-dim: mission(1) + subnet(9) + blocked(9) + comms(9) + proc(16) + conn(16: obs[44:60]) + msg(32)
            # 210-dim: mission(1) + 3 * [subnet(9)+blocked(9)+comms(9)+proc(16)+conn(16)] + msg(32)
            alerted_hosts = []
            if len(obs) == 92:
                conn_flags = obs[44:60]
                if np.any(conn_flags):
                    alerted_hosts = list(restore_map.keys())
            elif len(obs) == 210:
                for s_idx in range(3):
                    base = 1 + s_idx * 59
                    if np.any(obs[base + 43 : base + 59]):
                        alerted_hosts.extend(list(restore_map.keys()))

            target = None
            for h in alerted_hosts:
                if self.cooldowns[agent_id].get(h, 0) <= 0:
                    target = h
                    break

            if target and target in restore_map:
                actions[agent_id] = restore_map[target]
                self.cooldowns[agent_id][target] = self.restore_cooldown_steps
            else:
                actions[agent_id] = sleep_idx
        return actions


class CAGE4SleepDefender:
    """Passive baseline agent where all zone defenders sleep continuously."""

    def end_episode(self) -> None:
        pass

    def get_actions(
        self,
        observations: dict[str, np.ndarray],
        action_spaces: dict[str, Any] | None = None,
        action_labels_dict: dict[str, list[str]] | None = None,
        info: dict[str, dict] | None = None,
    ) -> dict[str, int]:
        actions: dict[str, int] = {}
        for agent_id in observations:
            labels = action_labels_dict.get(agent_id, []) if action_labels_dict else []
            sleep_idx = labels.index("Sleep") if "Sleep" in labels else (145 if agent_id == "blue_agent_4" else 49)
            actions[agent_id] = sleep_idx
        return actions

"""
Multi-Scale Network Benchmark Harness for Cyber-JEPA.

Supports:
- Slicing CybORG telemetry across 3 standardized network scales:
  * Small: 3-4 hosts (12-16 dims, Subnet tier)
  * Medium: 11-13 hosts (52 dims, Scenario 1b standard)
  * Large: 45 hosts (180 dims, CAGE-4 9-subnet enterprise standard)
- High-precision inference timing (CUDA Event-based & CPU high-res timers)
- GPU VRAM memory profiling
- Complexity curves across scales (N in {3..100+})
"""

import time
from typing import Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


SCALE_SPECS = {
    # Phase 6B Standardized 8-Scale Suite:
    "5": {
        "num_hosts": 5,
        "obs_dim": 20,
        "feature_slice": slice(0, 20),
        "host_indices": list(range(5)),
        "name": "Micro Branch Subnet (5 hosts)",
    },
    "10": {
        "num_hosts": 10,
        "obs_dim": 40,
        "feature_slice": slice(0, 40),
        "host_indices": list(range(10)),
        "name": "Core Subnet Tier (10 hosts)",
    },
    "13": {
        "num_hosts": 13,
        "obs_dim": 52,
        "feature_slice": slice(0, 52),
        "host_indices": list(range(13)),
        "name": "CybORG Scenario 1b Standard (13 hosts)",
    },
    "25": {
        "num_hosts": 25,
        "obs_dim": 100,
        "feature_slice": None,
        "host_indices": list(range(25)),
        "name": "Multi-Subnet Department (25 hosts)",
    },
    "50": {
        "num_hosts": 50,
        "obs_dim": 200,
        "feature_slice": None,
        "host_indices": list(range(50)),
        "name": "CAGE-4 Enterprise Scale (50 hosts)",
    },
    "100": {
        "num_hosts": 100,
        "obs_dim": 400,
        "feature_slice": None,
        "host_indices": list(range(100)),
        "name": "Campus Division Network (100 hosts)",
    },
    "250": {
        "num_hosts": 250,
        "obs_dim": 1000,
        "feature_slice": None,
        "host_indices": list(range(250)),
        "name": "Large Multi-Building Enterprise (250 hosts)",
    },
    "500": {
        "num_hosts": 500,
        "obs_dim": 2000,
        "feature_slice": None,
        "host_indices": list(range(500)),
        "name": "Global Corporate HQ / University (500 hosts)",
    },
    # Backward compatibility aliases for Phase 6A:
    "small": {
        "num_hosts": 3,
        "obs_dim": 12,
        "feature_slice": slice(0, 12),
        "host_indices": [0, 1, 2],
        "name": "Small Subnet (3 hosts)",
    },
    "medium": {
        "num_hosts": 13,
        "obs_dim": 52,
        "feature_slice": slice(0, 52),
        "host_indices": list(range(13)),
        "name": "Medium Network (11-13 hosts)",
    },
    "large": {
        "num_hosts": 45,
        "obs_dim": 180,
        "feature_slice": None,
        "host_indices": list(range(45)),
        "name": "Large Enterprise (45 hosts)",
    },
}


class ScaledDatasetWrapper(Dataset[dict[str, Any]]):
    """Wraps an existing CyberJEPADataset and projects/slices it to target network scale."""

    def __init__(self, base_dataset: Any, scale: str = "13", target_hosts: int | None = None):
        self.base = base_dataset
        self.scale = scale
        scale_key = str(scale)
        if target_hosts is not None:
            self.num_hosts = target_hosts
            self.obs_dim = target_hosts * 4
            self.spec = {
                "num_hosts": target_hosts,
                "obs_dim": self.obs_dim,
                "feature_slice": slice(0, self.obs_dim) if self.obs_dim <= 52 else None,
                "host_indices": list(range(target_hosts)),
                "name": f"Custom Scale ({target_hosts} hosts)",
            }
        elif scale_key in SCALE_SPECS:
            self.spec = SCALE_SPECS[scale_key]
            self.num_hosts = self.spec["num_hosts"]
            self.obs_dim = self.spec["obs_dim"]
        elif scale_key.isdigit():
            n = int(scale_key)
            self.num_hosts = n
            self.obs_dim = n * 4
            self.spec = {
                "num_hosts": n,
                "obs_dim": self.obs_dim,
                "feature_slice": slice(0, self.obs_dim) if self.obs_dim <= 52 else None,
                "host_indices": list(range(n)),
                "name": f"Dynamic Scale ({n} hosts)",
            }
        else:
            self.spec = SCALE_SPECS["13"]
            self.num_hosts = self.spec["num_hosts"]
            self.obs_dim = self.spec["obs_dim"]

    def __len__(self) -> int:
        return len(self.base)

    def __getattr__(self, name: str) -> Any:
        if name == "base":
            raise AttributeError(name)
        return getattr(self.base, name)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.base[idx]
        hist = item["history_flat"] # [T, 52]
        target = item["target_flat"] # [52]
        base_comp = item.get("host_compromised", torch.zeros(11))
        label = item["label"]

        if self.obs_dim <= 52:
            # Sliced / subset scale (e.g. 3, 5, 10, 13 hosts)
            hist_scaled = hist[:, :self.obs_dim].clone()
            target_scaled = target[:self.obs_dim].clone()
            if len(base_comp) >= self.num_hosts:
                host_comp = base_comp[:self.num_hosts]
            else:
                host_comp = torch.cat([base_comp, torch.zeros(self.num_hosts - len(base_comp))])
        else:
            # Scaled / enterprise replication (e.g. 25, 45, 50, 100, 250, 500 hosts)
            repeats = int(np.ceil(self.obs_dim / 52))
            hist_rep = hist.repeat(1, repeats)[:, :self.obs_dim].clone()
            target_rep = target.repeat(repeats)[:self.obs_dim].clone()
            hist_scaled = hist_rep
            target_scaled = target_rep
            host_comp = base_comp.repeat(int(np.ceil(self.num_hosts / len(base_comp))))[:self.num_hosts]

        delta = target_scaled - hist_scaled[-1]
        rms_delta = float(torch.sqrt(torch.mean(delta ** 2)).item())

        res = dict(item)
        res["history_flat"] = hist_scaled
        res["target_flat"] = target_scaled
        res["label"] = label
        res["host_compromised"] = host_comp
        res["rms_delta"] = rms_delta
        return res

    def compute_median_dynamic_rms(self) -> float:
        """Compute median RMS delta of non-zero transitions under current network scale."""
        sample_size = min(1000, len(self))
        deltas = [
            float(torch.sqrt(torch.mean((self[i]["target_flat"] - self[i]["history_flat"][-1]) ** 2)).item())
            for i in range(sample_size)
        ]
        pos_deltas = [d for d in deltas if d > 1e-4]
        return float(np.median(pos_deltas)) if pos_deltas else 0.25

    def get_balanced_sampler(self, threshold: float | None = None, generator: Any = None) -> Any:
        """Enforces 1:1 static/dynamic balanced sampling on the scaled dataset."""
        thresh = threshold if threshold is not None else self.compute_median_dynamic_rms()
        if hasattr(self.base, "samples"):
            N = len(self.base.samples)
            if N == 0:
                return torch.utils.data.WeightedRandomSampler([1.0], 1)

            if self.obs_dim < 52:
                deltas = np.array([
                    float(torch.sqrt(torch.mean((s["target_flat"][:self.obs_dim] - s["history_flat"][-1, :self.obs_dim]) ** 2)).item())
                    for s in self.base.samples
                ])
                is_dynamic = deltas >= thresh
            else:
                is_dynamic = np.array([s["rms_delta"] >= thresh for s in self.base.samples], dtype=bool)

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
        elif hasattr(self.base, "get_balanced_sampler"):
            return self.base.get_balanced_sampler(threshold=thresh, generator=generator)
        return None


def benchmark_inference_latency_and_memory(
    model: nn.Module,
    input_shape: tuple[int, ...],
    device: torch.device,
    action_shape: tuple[int, ...] | None = None,
    num_warmup: int = 50,
    num_iters: int = 500,
) -> dict[str, float]:
    """
    Measure inference latency (CUDA events & CPU high-res timers) and peak memory.

    Args:
        model: PyTorch model (CyberJEPA or encoder)
        input_shape: History tensor shape [B, T_hist, D_in]
        device: torch.device ('cuda' or 'cpu')
        action_shape: Optional action tensor shape [B, K]
        num_warmup: Warmup iterations
        num_iters: Timing iterations

    Returns:
        dict with mean_latency_ms, std_latency_ms, p50_ms, p95_ms, throughput_fps, peak_vram_mb
    """
    try:
        orig_device = next(model.parameters()).device
    except StopIteration:
        orig_device = device
    model.to(device)
    model.eval()
    B = input_shape[0]
    dummy_input = torch.randn(input_shape, device=device)
    dummy_actions = torch.zeros(action_shape if action_shape is not None else (B, 4), device=device, dtype=torch.long)
    dummy_target = torch.randn(B, input_shape[-1], device=device)

    def _eval_step():
        if hasattr(model, "encode_context"):
            return model.encode_context(dummy_input)
        elif hasattr(model, "forward"):
            try:
                return model(dummy_input, dummy_actions, dummy_target)
            except Exception:
                return model(dummy_input)
        return None

    # Param count and size
    param_count = sum(p.numel() for p in model.parameters())
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    param_mb = param_bytes / (1024 * 1024)

    is_cuda = (device.type == "cuda")

    try:
        if is_cuda:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

        with torch.no_grad():
            # Warmup
            for _ in range(num_warmup):
                _eval_step()

            if is_cuda:
                torch.cuda.synchronize()
                start_event = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
                end_event = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]

                for i in range(num_iters):
                    start_event[i].record()
                    _eval_step()
                    end_event[i].record()

                torch.cuda.synchronize()
                times = [s.elapsed_time(e) for s, e in zip(start_event, end_event)]
                peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
            else:
                times = []
                for _ in range(num_iters):
                    t0 = time.perf_counter()
                    _eval_step()
                    t1 = time.perf_counter()
                    times.append((t1 - t0) * 1000.0)
                peak_vram_mb = 0.0
    finally:
        model.to(orig_device)

    times_np = np.array(times)
    mean_lat = float(np.mean(times_np))
    std_lat = float(np.std(times_np))
    p50_lat = float(np.percentile(times_np, 50))
    p95_lat = float(np.percentile(times_np, 95))
    throughput = float((B * 1000.0) / max(1e-6, mean_lat))

    return {
        "num_params": param_count,
        "model_size_mb": param_mb,
        "mean_latency_ms": mean_lat,
        "std_latency_ms": std_lat,
        "p50_latency_ms": p50_lat,
        "p95_latency_ms": p95_lat,
        "throughput_samples_per_sec": throughput,
        "peak_vram_mb": peak_vram_mb,
        "device": device.type,
    }

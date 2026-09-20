"""
Unit tests for Phase 6A: Modernized Host Token JEPA, Multi-Scale Benchmarks, and Attention Analysis.
"""

import pytest
import numpy as np
import torch

from cyber_jepa.representations.host import (
    ModernHostTokenRepresentation,
    InterpretableTransformerLayer,
)
from cyber_jepa.models.jepa import CyberJEPA
from cyber_jepa.evaluation.attention_analysis import (
    extract_host_attention_distribution,
    compute_attention_entropy,
    analyze_attack_attention_focus,
)
from cyber_jepa.evaluation.scale_benchmarks import (
    benchmark_inference_latency_and_memory,
    SCALE_SPECS,
)


def test_interpretable_transformer_layer():
    """Verify InterpretableTransformerLayer computes forward and extracts attention weights."""
    layer = InterpretableTransformerLayer(d_model=64, nhead=4, dim_feedforward=256)
    x = torch.randn(4, 10, 64)

    out_no_weights = layer(x, need_weights=False)
    assert out_no_weights.shape == (4, 10, 64)
    assert layer.last_attn_weights is None

    out_with_weights = layer(x, need_weights=True)
    assert out_with_weights.shape == (4, 10, 64)
    assert layer.last_attn_weights is not None
    assert layer.last_attn_weights.shape == (4, 10, 10)


def test_modern_host_token_representation_variable_scales():
    """Verify ModernHostTokenRepresentation operates across 3, 11, 45 hosts."""
    scales = [3, 11, 45]
    for num_hosts in scales:
        enc = ModernHostTokenRepresentation(
            num_hosts=num_hosts,
            features_per_host=4,
            hidden_dim=64,
            history_len=4,
        )
        obs_dim = num_hosts * 4
        x = torch.randn(2, 4, obs_dim)

        ctx = enc(x, need_weights=True)
        assert ctx.tokens.shape == (2, 4 * num_hosts, 64)
        assert ctx.global_token is not None
        assert ctx.global_token.shape == (2, 64)

        attn_weights = enc.get_attention_weights()
        assert len(attn_weights) == 3 # 3 layers
        total_L = 4 * num_hosts + 1 # history * hosts + global reg token
        assert attn_weights[-1].shape == (2, total_L, total_L)


def test_attention_analysis_entropy_and_focus():
    """Verify attention distribution, entropy computation, and attack focus metrics."""
    B, N = 4, 11
    # Create synthetic attention probabilities
    host_probs = np.ones((B, N)) / N
    entropies = compute_attention_entropy(host_probs)
    expected_entropy = np.log2(N)
    assert np.allclose(entropies, expected_entropy, atol=1e-5)

    # Attack focus analysis
    host_comp = np.zeros((B, N))
    host_comp[0, 3] = 1.0 # Step 0 has host 3 compromised
    host_comp[1, 5] = 1.0 # Step 1 has host 5 compromised

    # Skew attention on step 0 toward host 3
    host_probs[0, :] = 0.01
    host_probs[0, 3] = 0.90
    host_probs[0] /= np.sum(host_probs[0])

    # Skew attention on step 1 toward host 5
    host_probs[1, :] = 0.01
    host_probs[1, 5] = 0.90
    host_probs[1] /= np.sum(host_probs[1])

    analysis = analyze_attack_attention_focus(host_probs, host_comp)
    assert analysis["num_hosts"] == 11
    assert analysis["attention_focus_ratio"] > 1.0
    assert analysis["clean_steps_entropy"] > analysis["compromised_steps_entropy"]


def test_extract_host_attention_distribution_end_to_end():
    """Verify extract_host_attention_distribution works end-to-end with CyberJEPA."""
    enc = ModernHostTokenRepresentation(num_hosts=3, features_per_host=4, hidden_dim=64, history_len=4)
    model = CyberJEPA(online_encoder=enc, hidden_dim=64, max_horizon=4)
    hist = torch.randn(8, 4, 12)
    device = torch.device("cpu")

    probs, raw_attn = extract_host_attention_distribution(model, hist, device, batch_size=4)
    assert probs.shape == (8, 3)
    # Check that each row sums to 1.0
    row_sums = np.sum(probs, axis=-1)
    assert np.allclose(row_sums, 1.0, atol=1e-5)
    assert raw_attn.shape[-1] == 4 * 3 + 1 # history * hosts + global reg token


def test_scale_benchmark_latency_harness():
    """Verify benchmark_inference_latency_and_memory executes without errors on CPU."""
    enc = ModernHostTokenRepresentation(num_hosts=3, features_per_host=4, hidden_dim=64)
    res = benchmark_inference_latency_and_memory(
        model=enc,
        input_shape=(2, 4, 12),
        device=torch.device("cpu"),
        num_warmup=5,
        num_iters=10,
    )
    assert "mean_latency_ms" in res
    assert res["mean_latency_ms"] > 0.0
    assert res["num_params"] > 0
    assert res["device"] == "cpu"


def test_scaled_dataset_wrapper():
    """Verify ScaledDatasetWrapper shapes for Small, Medium, and Large."""
    from cyber_jepa.evaluation.scale_benchmarks import ScaledDatasetWrapper

    class DummyBaseDataset:
        def __len__(self):
            return 5

        def __getitem__(self, idx):
            return {
                "history_flat": torch.randn(4, 52),
                "target_flat": torch.randn(52),
                "label": 1,
                "host_compromised": torch.zeros(11),
                "action_seq": torch.zeros(4, dtype=torch.long),
            }

    dummy = DummyBaseDataset()
    for scale, exp_dim, exp_hosts in [("small", 12, 3), ("medium", 52, 13), ("large", 180, 45)]:
        wrapper = ScaledDatasetWrapper(dummy, scale=scale)
        item = wrapper[0]
        assert item["history_flat"].shape == (4, exp_dim)
        assert item["target_flat"].shape == (exp_dim,)
        assert item["host_compromised"].shape == (exp_hosts,)


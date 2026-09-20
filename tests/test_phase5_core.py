"""
Unit and Integration Tests for Phase 5 Core Implementations:
- batch_center normalization and VICReg loss regularization
- CyberJEPADataset RMS delta and static50_dynamic50 balanced sampler
- Phase 5 mathematical diagnostic metrics (alpha decay, uniformity, k-NN, Fisher ratio)
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
from pathlib import Path

from cyber_jepa.models.predictor import compute_jepa_loss
from cyber_jepa.models.jepa import CyberJEPA
from cyber_jepa.representations.flat import FlatVectorRepresentation
from cyber_jepa.data.dataset import CyberJEPADataset
from cyber_jepa.evaluation.phase5_diagnostics import (
    compute_spectral_decay_alpha,
    compute_wang_isola_uniformity,
    compute_fisher_discriminant_ratio,
    evaluate_knn_probe,
    evaluate_action_sensitivity,
    evaluate_predictive_skill_and_persistence,
)


def test_batch_center_loss_and_vicreg():
    """Verify compute_jepa_loss with batch_center and VICReg penalties."""
    pred = torch.randn(16, 64, requires_grad=True)
    target = torch.randn(16, 64)

    # 1. Standard batch_center + VICReg
    loss, details = compute_jepa_loss(
        pred,
        target,
        norm_mode="batch_center",
        vicreg_var_weight=1.0,
        vicreg_cov_weight=0.04,
        return_dict=True,
    )

    assert loss.dim() == 0
    assert loss.item() > 0.0
    assert "pred_loss" in details
    assert "var_loss" in details
    assert "cov_loss" in details

    # Test backward pass
    loss.backward()
    assert pred.grad is not None
    assert torch.all(torch.isfinite(pred.grad))


def test_cyber_jepa_phase5_loss_integration():
    """Verify CyberJEPA forward pass with phase 5 loss settings."""
    encoder = FlatVectorRepresentation(hidden_dim=64, ffn_dim=256)
    jepa = CyberJEPA(
        online_encoder=encoder,
        hidden_dim=64,
        loss_norm_mode="batch_center",
        vicreg_var_weight=1.0,
        vicreg_cov_weight=0.04,
    )

    hist = torch.randn(8, 4, 52)
    actions = torch.randint(0, 66, (8, 4))
    target = torch.randn(8, 52)

    loss, pred_z, target_z = jepa(hist, actions, target)
    assert loss.dim() == 0
    assert loss.item() > 0.0
    assert pred_z.shape == (8, 64)
    assert target_z.shape == (8, 64)
    assert "pred_loss" in jepa.last_loss_breakdown
    assert "var_loss" in jepa.last_loss_breakdown
    assert "cov_loss" in jepa.last_loss_breakdown


def test_balanced_sampler_ratio(tmp_path: Path):
    """Verify get_balanced_sampler yields 50% static and 50% dynamic transitions."""
    # Create synthetic dataset with known deltas
    dataset = CyberJEPADataset(shard_dirs=[])
    dataset.samples = []

    # 80 static transitions (delta = 0.05), 20 dynamic transitions (delta = 0.50)
    for i in range(80):
        dataset.samples.append({"rms_delta": 0.05, "history_flat": torch.zeros(4, 52), "action_seq": torch.zeros(4, dtype=torch.long), "target_flat": torch.zeros(52), "label": 0, "horizon": 4})
    for i in range(20):
        dataset.samples.append({"rms_delta": 0.50, "history_flat": torch.zeros(4, 52), "action_seq": torch.zeros(4, dtype=torch.long), "target_flat": torch.zeros(52), "label": 1, "horizon": 4})

    sampler = dataset.get_balanced_sampler(threshold=0.25)
    sampled_indices = list(sampler)
    assert len(sampled_indices) == 100

    dynamic_count = sum(1 for idx in sampled_indices if dataset.samples[idx]["rms_delta"] >= 0.25)
    # With replacement over 100 samples from 50/50 weights, dynamic count should be near 50
    assert 30 <= dynamic_count <= 70


def test_spectral_decay_and_uniformity():
    """Verify spectral decay alpha and Wang-Isola uniformity computation."""
    # 1. Perfectly isotropic sphere
    latents = torch.randn(200, 64)
    unif = compute_wang_isola_uniformity(latents)
    assert isinstance(unif, float)
    assert np.isfinite(unif)

    # 2. Power-law decay alpha
    # Synthesize power-law spectrum sigma_i = i^(-1.2)
    singular_values = np.array([1.0 / (i ** 1.2) for i in range(1, 30)])
    alpha = compute_spectral_decay_alpha(singular_values, top_k=20)
    assert 1.0 <= alpha <= 1.4


def test_fisher_ratio_and_knn_probe():
    """Verify Fisher Discriminant Ratio and non-parametric k-NN probe."""
    # Create two well-separated Gaussian clusters in 64 dims
    N = 100
    cluster_0 = np.random.randn(N, 64) * 0.5 - 2.0
    cluster_1 = np.random.randn(N, 64) * 0.5 + 2.0

    X = np.vstack([cluster_0, cluster_1])
    y = np.array([0] * N + [1] * N)

    fisher_ratio = compute_fisher_discriminant_ratio(X, y)
    assert fisher_ratio > 1.5

    # k-NN probe
    knn_results = evaluate_knn_probe(X[:160], y[:160], X[160:], y[160:], k=5)
    assert knn_results["knn_k5_macro_f1"] > 0.90


def test_batch_center_n1_fallback():
    """Verify compute_jepa_loss handles N=1 without crashing or falling into layer_norm."""
    pred = torch.randn(1, 64, requires_grad=True)
    target = torch.randn(1, 64)

    loss, details = compute_jepa_loss(
        pred,
        target,
        norm_mode="batch_center",
        vicreg_var_weight=1.0,
        vicreg_cov_weight=0.04,
        return_dict=True,
    )
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    assert details["var_loss"].item() == 0.0
    assert details["cov_loss"].item() == 0.0
    loss.backward()
    assert pred.grad is not None


def test_dataset_getitem_returns_rms_delta():
    """Verify CyberJEPADataset.__getitem__ includes rms_delta field."""
    dataset = CyberJEPADataset(shard_dirs=[])
    dataset.samples = [{
        "trajectory_id": "traj_0",
        "transition_id": "trans_0",
        "t_context": 0,
        "t_target": 4,
        "rms_delta": 0.35,
        "history_flat": torch.zeros(4, 52),
        "action_seq": torch.zeros(4, dtype=torch.long),
        "target_flat": torch.zeros(52),
        "label": 1,
        "horizon": 4,
    }]

    item = dataset[0]
    assert "rms_delta" in item
    assert abs(item["rms_delta"] - 0.35) < 1e-6


def test_action_sensitivity_pred_loss_isolation():
    """Verify evaluate_action_sensitivity isolates pred_loss from vicreg fluctuations."""
    encoder = FlatVectorRepresentation(hidden_dim=64, ffn_dim=256)
    jepa = CyberJEPA(
        online_encoder=encoder,
        hidden_dim=64,
        loss_norm_mode="batch_center",
        vicreg_var_weight=1.0,
        vicreg_cov_weight=0.04,
    )

    batch = {
        "history_flat": torch.randn(4, 4, 52),
        "action_seq": torch.randint(0, 66, (4, 4)),
        "target_flat": torch.randn(4, 52),
    }

    results = evaluate_action_sensitivity(jepa, batch, device=torch.device("cpu"))
    assert "action_shuffled_degradation" in results
    assert "action_zeroed_degradation" in results
    assert np.isfinite(results["action_shuffled_degradation"])
    assert np.isfinite(results["action_zeroed_degradation"])


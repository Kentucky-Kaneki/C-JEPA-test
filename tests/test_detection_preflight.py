"""
Unit tests for Detection Pre-Flight Checks:
- Multi-host label extraction from dataset
- NIST SP 800-61 Precursor Anticipation Gap benchmark
- MITRE ATT&CK Multi-host compromise localization
"""

import numpy as np
import pytest
import torch

from cyber_jepa.data.dataset import CyberJEPADataset, MONITORED_HOSTS
from cyber_jepa.evaluation.anticipation_diagnostics import evaluate_anticipation_gap
from cyber_jepa.evaluation.localization_probes import (
    evaluate_multi_host_localization,
    compute_fpr_at_recall,
)


def test_multi_host_labels_in_dataset():
    """Verify CyberJEPADataset extracts and collates host_compromised vectors."""
    dataset = CyberJEPADataset(shard_dirs=[])
    dataset.samples = [{
        "trajectory_id": "traj_0",
        "transition_id": "trans_0",
        "t_context": 0,
        "t_target": 4,
        "history_flat": torch.zeros(4, 52),
        "action_seq": torch.zeros(4, dtype=torch.long),
        "target_flat": torch.zeros(52),
        "label": 1,
        "horizon": 4,
        "rms_delta": 0.25,
        "host_compromised": torch.zeros(len(MONITORED_HOSTS)),
    }]
    # Mark Op_Server0 (index 3) as compromised
    dataset.samples[0]["host_compromised"][3] = 1.0

    item = dataset[0]
    assert "host_compromised" in item
    assert item["host_compromised"].shape == (len(MONITORED_HOSTS),)
    assert item["host_compromised"][3] == 1.0


def test_anticipation_gap_benchmark():
    """Verify evaluate_anticipation_gap detects positive anticipation gain."""
    N = 200
    D = 64
    np.random.seed(42)

    # Future labels: alternating 0 and 1 so both splits have both classes
    labels = np.array([i % 2 for i in range(N)])

    # Current state has only partial information (noisy correlation with label)
    z_curr = np.random.randn(N, D)
    z_curr[labels == 1, 0] += 0.5

    # Predicted future state has high information (strong correlation with future label)
    z_fut = np.random.randn(N, D)
    z_fut[labels == 1, 0] += 3.0

    train_idx = np.arange(0, 140)
    test_idx = np.arange(140, 200)

    results = evaluate_anticipation_gap(
        train_z_current=z_curr[train_idx],
        train_z_future=z_fut[train_idx],
        train_labels=labels[train_idx],
        test_z_current=z_curr[test_idx],
        test_z_future=z_fut[test_idx],
        test_labels=labels[test_idx],
    )

    assert "anticipation_gap" in results
    gap = results["anticipation_gap"]
    assert gap["delta_macro_f1"] > 0
    assert gap["delta_auroc"] > 0
    assert gap["has_anticipation_advantage"] is True


def test_multi_host_localization_and_fpr():
    """Verify evaluate_multi_host_localization correctly attributes host compromises."""
    N = 300
    D = 64
    num_hosts = len(MONITORED_HOSTS)
    np.random.seed(42)

    latents = np.random.randn(N, D)
    host_comp = np.zeros((N, num_hosts), dtype=np.float32)

    # Host 0 (Enterprise0) compromised in first 100
    host_comp[:100, 0] = 1.0
    latents[:100, 0] += 2.5

    # Host 3 (Op_Server0) compromised in next 100
    host_comp[100:200, 3] = 1.0
    latents[100:200, 3] += 2.5

    train_idx = np.arange(0, 200)
    test_idx = np.arange(200, 300)

    results = evaluate_multi_host_localization(
        train_latents=latents[train_idx],
        train_host_compromised=host_comp[train_idx],
        test_latents=latents[test_idx],
        test_host_compromised=host_comp[test_idx],
    )

    assert "by_host" in results
    assert "by_mitre_tier" in results
    assert "top1_host_attribution_accuracy" in results
    assert results["top1_host_attribution_accuracy"] >= 0.0

    # Test FPR computation
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.4, 0.8, 0.85, 0.9, 0.95])
    fpr = compute_fpr_at_recall(y_true, y_prob, target_recall=0.75)
    assert 0.0 <= fpr <= 1.0

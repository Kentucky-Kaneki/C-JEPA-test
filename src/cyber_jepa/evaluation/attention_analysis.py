"""
Self-Attention Interpretability & Host Attribution Analyzer for Cyber-JEPA.

Quantifies how self-attention routes information between hosts across network scales,
measuring attention entropy and focus on compromised hosts during cyber attacks.
"""

from typing import Any
import numpy as np
import torch
import torch.nn as nn

from cyber_jepa.models.jepa import CyberJEPA


def extract_host_attention_distribution(
    model: CyberJEPA,
    history_obs: torch.Tensor,
    device: torch.device,
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract attention distribution from the global readout token to each host.
    Batches evaluation across chunks to guarantee zero OOM risk on large evaluation sets.

    Args:
        model: CyberJEPA instance with ModernHostTokenRepresentation online_encoder.
        history_obs: Context history observations [B, T_hist, N * 4]
        device: torch device
        batch_size: Chunk size for GPU processing

    Returns:
        host_probs: Normalized attention probability per host [B, N]
        raw_attn: Full attention matrix from last layer for the first batch [B_sample, L, L]
    """
    model.eval()
    enc = model.online_encoder
    if not hasattr(enc, "get_attention_weights"):
        raise ValueError("online_encoder must support get_attention_weights (e.g. ModernHostTokenRepresentation)")

    num_samples = history_obs.shape[0]
    T_hist = history_obs.shape[1]
    assert T_hist > 0, f"T_hist must be > 0, got {T_hist}"

    all_host_probs = []
    sample_top_attn = None

    with torch.no_grad():
        for start_idx in range(0, num_samples, batch_size):
            chunk = history_obs[start_idx : start_idx + batch_size].to(device)
            B_chunk = chunk.shape[0]

            if hasattr(enc, "encode_context"):
                _ = enc.encode_context(chunk, need_weights=True)
            else:
                _ = enc(chunk, need_weights=True)

            attn_layers = enc.get_attention_weights()
            if not attn_layers:
                raise RuntimeError("No attention weights captured from encoder layers.")

            # Top layer attention weights for this chunk: [B_chunk, L, L]
            top_attn = attn_layers[-1]
            L = top_attn.shape[-1]
            N = (L - 1) // T_hist

            # Global token is at index -1. Extract attention from global token to all tokens [B_chunk, L-1]
            global_attn = top_attn[:, -1, :-1].view(B_chunk, T_hist, N)
            host_attn = global_attn.sum(dim=1) # [B_chunk, N]
            chunk_probs = host_attn / torch.clamp(host_attn.sum(dim=-1, keepdim=True), min=1e-8)
            all_host_probs.append(chunk_probs.cpu().numpy())

            if sample_top_attn is None:
                sample_top_attn = top_attn[:min(64, B_chunk)].cpu().numpy()

    if all_host_probs:
        full_host_probs = np.concatenate(all_host_probs, axis=0)
    else:
        full_host_probs = np.zeros((0, (history_obs.shape[-1] // 4)))

    raw_attn_ret = sample_top_attn if sample_top_attn is not None else np.zeros((0, 1, 1))
    return full_host_probs, raw_attn_ret


def compute_attention_entropy(host_probs: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    """
    Compute Shannon entropy (in bits) of host attention distribution.

    Args:
        host_probs: [B, N] probability matrix

    Returns:
        entropies: [B] entropy values
    """
    p = np.clip(host_probs, epsilon, 1.0)
    p = p / np.sum(p, axis=-1, keepdims=True)
    entropy = -np.sum(p * np.log2(p), axis=-1)
    return entropy


def analyze_attack_attention_focus(
    host_probs: np.ndarray,
    host_compromised_matrix: np.ndarray,
) -> dict[str, float]:
    """
    Analyze whether attention concentrates on compromised hosts during attack steps.

    Args:
        host_probs: [B, N] attention distribution
        host_compromised_matrix: [B, N] binary compromise indicators

    Returns:
        dict with attention focus metrics
    """
    B, N = host_probs.shape
    if host_compromised_matrix.shape[1] != N:
        if host_compromised_matrix.shape[1] < N:
            pad = np.zeros((B, N - host_compromised_matrix.shape[1]))
            host_compromised_matrix = np.concatenate([host_compromised_matrix, pad], axis=1)
        else:
            host_compromised_matrix = host_compromised_matrix[:, :N]

    any_comp = np.any(host_compromised_matrix > 0, axis=-1)

    entropies = compute_attention_entropy(host_probs)
    max_entropy = float(np.log2(N))

    clean_entropy = float(np.mean(entropies[~any_comp])) if np.sum(~any_comp) > 0 else max_entropy
    comp_entropy = float(np.mean(entropies[any_comp])) if np.sum(any_comp) > 0 else clean_entropy

    # Fraction of attention allocated to compromised hosts
    if np.sum(any_comp) > 0:
        comp_probs = host_probs[any_comp]
        comp_truth = host_compromised_matrix[any_comp] > 0

        # Sum attention on actually compromised hosts
        mass_on_compromised = np.array([
            float(np.sum(comp_probs[i, comp_truth[i]]))
            for i in range(len(comp_probs))
        ])
        # Expected mass under uniform random attention
        comp_counts = np.sum(comp_truth, axis=-1)
        expected_uniform_mass = comp_counts / N

        mean_mass = float(np.mean(mass_on_compromised))
        mean_uniform = float(np.mean(expected_uniform_mass))
        focus_ratio = mean_mass / max(1e-6, mean_uniform)
    else:
        mean_mass = 0.0
        mean_uniform = 0.0
        focus_ratio = 1.0

    return {
        "num_hosts": N,
        "max_theoretical_entropy": max_entropy,
        "clean_steps_entropy": clean_entropy,
        "compromised_steps_entropy": comp_entropy,
        "entropy_reduction_under_attack": float(clean_entropy - comp_entropy),
        "compromised_attention_mass": mean_mass,
        "expected_uniform_mass": mean_uniform,
        "attention_focus_ratio": focus_ratio,
    }

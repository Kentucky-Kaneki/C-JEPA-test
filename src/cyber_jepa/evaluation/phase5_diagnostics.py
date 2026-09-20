"""
Phase 5 Comprehensive Mathematical Diagnostic Suite for Cyber-JEPA.

Implements rigorous, literature-grounded evaluation metrics:
1. Spectral Decay (alpha-power law, Stringer et al., Nature 2019)
2. Wang & Isola Uniformity (ICML 2020)
3. Non-Parametric k-NN Classification (Wu et al. 2018; Caron et al. DINO 2021)
4. Fisher Discriminant Ratio (Between vs Within Class Scatter)
5. Causal Action Sensitivity (Permutation & Zeroing, Bardes et al. V-JEPA 2 2024)
6. Dynamic-Transition Persistence Gain & Latent R^2 (Boardman et al. T-JEPA 2025)
"""

from typing import Any
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score

from cyber_jepa.evaluation.diagnostics import compute_latent_geometry_diagnostics


def compute_spectral_decay_alpha(singular_values: np.ndarray, top_k: int = 20) -> float:
    """
    Fit power-law decay exponent alpha: log(sigma_i) = -alpha * log(i) + c.
    (Stringer et al., Nature 2019; Balestriero & LeCun, NeurIPS 2022).
    A healthy representation exhibits alpha in [0.8, 1.4].
    alpha > 2.0 indicates severe dimensional collapse.
    """
    s = np.asarray(singular_values, dtype=np.float64)
    s = s[s > 1e-10]
    K = min(len(s), top_k)
    if K < 3:
        return 0.0

    ranks = np.arange(1, K + 1, dtype=np.float64)
    log_ranks = np.log(ranks)
    log_s = np.log(s[:K])

    # Linear regression slope: log_s = -alpha * log_ranks + c
    cov = np.cov(log_ranks, log_s)
    var_x = cov[0, 0]
    if var_x < 1e-12:
        return 0.0
    slope = cov[0, 1] / var_x
    alpha = float(-slope)
    return max(0.0, alpha)


def compute_wang_isola_uniformity(
    latents: torch.Tensor,
    t: float = 2.0,
    subsample: int = 1000,
    generator: torch.Generator | None = None,
) -> float:
    """
    Uniformity on the unit hypersphere (Wang & Isola, ICML 2020).
    L_unif = log E_{u, v ~ Z} [ exp(-t * ||u - v||^2) ]
    Lower values indicate more uniform spherical distribution (non-collapsed).
    """
    if latents.dim() > 2:
        latents = latents.reshape(-1, latents.shape[-1])
    N = latents.shape[0]
    if N < 2:
        return 0.0

    u = F.normalize(latents.detach(), dim=-1)
    if N > subsample:
        perm = torch.randperm(N, generator=generator)[:subsample]
        u = u[perm]

    # Pairwise squared Euclidean distance: ||u_i - u_j||^2 = 2 - 2 * (u_i . u_j)
    sim = u @ u.T
    dist_sq = 2.0 - 2.0 * torch.clamp(sim, -1.0, 1.0)
    # Mask out diagonal (self-distance)
    diag_mask = torch.eye(dist_sq.shape[0], device=dist_sq.device, dtype=torch.bool)
    off_diag_dist_sq = dist_sq[~diag_mask]

    exp_term = torch.exp(-t * off_diag_dist_sq)
    unif_score = float(torch.log(torch.mean(exp_term) + 1e-12).item())
    return unif_score


def compute_fisher_discriminant_ratio(latents: np.ndarray, labels: np.ndarray) -> float:
    """
    Fisher's Discriminant Ratio: F = Tr(S_B) / Tr(S_W)
    where S_B is between-class scatter and S_W is within-class scatter.
    High F (> 1.5) indicates strong linear/topological class separability.
    """
    X = np.asarray(latents, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)

    classes = np.unique(y)
    if len(classes) < 2:
        return 0.0

    global_mean = np.mean(X, axis=0)
    tr_sb = 0.0
    tr_sw = 0.0

    for c in classes:
        Xc = X[y == c]
        Nc = len(Xc)
        if Nc == 0:
            continue
        mean_c = np.mean(Xc, axis=0)
        # Between-class scatter trace
        tr_sb += Nc * float(np.sum((mean_c - global_mean) ** 2))
        # Within-class scatter trace
        tr_sw += float(np.sum((Xc - mean_c) ** 2))

    if tr_sw < 1e-12:
        return 100.0 if tr_sb > 0 else 0.0
    return float(tr_sb / tr_sw)


def evaluate_knn_probe(
    train_latents: np.ndarray,
    train_labels: np.ndarray,
    test_latents: np.ndarray,
    test_labels: np.ndarray,
    k: int = 5,
) -> dict[str, float]:
    """
    Non-parametric k-NN evaluation of oracle state separability (Wu et al. 2018; Caron et al. 2021).
    Zero trainable weights: directly measures neighborhood topological clustering.
    """
    knn = KNeighborsClassifier(n_neighbors=k, metric="cosine", weights="uniform")
    knn.fit(train_latents, train_labels)
    preds = knn.predict(test_latents)

    macro_f1 = float(f1_score(test_labels, preds, average="macro", zero_division=0))
    acc = float(accuracy_score(test_labels, preds))
    bacc = float(balanced_accuracy_score(test_labels, preds))

    return {
        f"knn_k{k}_macro_f1": macro_f1,
        f"knn_k{k}_accuracy": acc,
        f"knn_k{k}_balanced_acc": bacc,
    }


def evaluate_action_sensitivity(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, float]:
    """
    Measure action causality via permutation and zeroing tests (V-JEPA 2-AC, Bardes et al. 2024).
    Target: Shuffled actions should degrade prediction loss by >= +15%.
    """
    model.eval()
    hist = batch["history_flat"].to(device)
    actions = batch["action_seq"].to(device)
    target = batch["target_flat"].to(device)
    B = actions.shape[0]

    with torch.no_grad():
        # 1. True actions
        loss_true, _, _ = model(hist, actions, target)
        loss_true_val = float(
            getattr(model, "last_loss_breakdown", {}).get("pred_loss", loss_true.item())
        )

        # 2. Shuffled actions (permuted across batch)
        if B > 1:
            perm = torch.randperm(B, device=device)
            actions_shuffled = actions[perm]
            loss_shuffled, _, _ = model(hist, actions_shuffled, target)
            loss_shuffled_val = float(
                getattr(model, "last_loss_breakdown", {}).get("pred_loss", loss_shuffled.item())
            )
            shuffled_degradation = (loss_shuffled_val - loss_true_val) / max(1e-6, loss_true_val)
        else:
            loss_shuffled_val = loss_true_val
            shuffled_degradation = 0.0

        # 3. Zeroed actions (all actions set to 0 = Sleep)
        actions_zero = torch.zeros_like(actions)
        loss_zero, pred_zero, _ = model(hist, actions_zero, target)
        loss_zero_val = float(
            getattr(model, "last_loss_breakdown", {}).get("pred_loss", loss_zero.item())
        )
        zeroed_degradation = (loss_zero_val - loss_true_val) / max(1e-6, loss_true_val)

    return {
        "loss_true": loss_true_val,
        "loss_shuffled": loss_shuffled_val,
        "loss_zero": loss_zero_val,
        "action_shuffled_degradation": float(shuffled_degradation),
        "action_zeroed_degradation": float(zeroed_degradation),
    }


def evaluate_predictive_skill_and_persistence(
    pred_latents: np.ndarray,
    target_latents: np.ndarray,
    persistence_latents: np.ndarray,
    is_dynamic: np.ndarray,
) -> dict[str, float]:
    """
    Compute Latent R^2, overall Persistence MSE, and Dynamic Persistence Gain (T-JEPA 2025).
    """
    P = np.asarray(pred_latents, dtype=np.float64)
    T = np.asarray(target_latents, dtype=np.float64)
    Pers = np.asarray(persistence_latents, dtype=np.float64)
    dyn = np.asarray(is_dynamic, dtype=bool)

    # 1. Overall MSEs
    mse_jepa = float(np.mean((P - T) ** 2))
    mse_pers = float(np.mean((Pers - T) ** 2))
    var_target = float(np.var(T))

    latent_r2 = 1.0 - (mse_jepa / max(1e-6, var_target))

    # 2. Dynamic-transition subsets (where attack occurred)
    if np.sum(dyn) > 0:
        dyn_mse_jepa = float(np.mean((P[dyn] - T[dyn]) ** 2))
        dyn_mse_pers = float(np.mean((Pers[dyn] - T[dyn]) ** 2))
        dyn_gain = (dyn_mse_pers - dyn_mse_jepa) / max(1e-6, dyn_mse_pers)
    else:
        dyn_mse_jepa = mse_jepa
        dyn_mse_pers = mse_pers
        dyn_gain = 0.0

    return {
        "mse_jepa": mse_jepa,
        "mse_persistence": mse_pers,
        "latent_r2": float(latent_r2),
        "dynamic_mse_jepa": dyn_mse_jepa,
        "dynamic_mse_persistence": dyn_mse_pers,
        "dynamic_persistence_gain": float(dyn_gain),
    }

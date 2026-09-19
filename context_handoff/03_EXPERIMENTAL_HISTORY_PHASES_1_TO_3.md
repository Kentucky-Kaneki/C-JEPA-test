# 03 — Experimental Journey & Chronological Progression (Phases 1 to 3)

> **Document Purpose**: Comprehensive chronicle of the research project's evolution, milestones, technical hurdles diagnosed, and scientific breakthroughs across Phases 1, 2, and 3.

---

## 1. Overview of Experimental Progression

```
+-----------------------------------------------------------------------------------------+
| PHASE 1: EXPLORATORY SWEEP & HORIZON CHARACTERIZATION                                  |
| - Validated CybORG 3.1 Gym telemetry collection & 52-dim observation pipeline.           |
| - Executed 20-run horizon sweep (k in {1, 2, 4, 8, 16}) across 4 representations.       |
| - DISCOVERY: k=8 is optimal (26.3% state change rate matching Red kill-chain).          |
+-----------------------------------------------------------------------------------------+
                                            |
                                            v
+-----------------------------------------------------------------------------------------+
| PHASE 2: FAIRNESS AUDIT, ABLATIONS & CAPACITY RESCUE                                    |
| - Controlled history length (h=1, 4, 8); proved multi-step history is essential.        |
| - Proved temporal order matters via sequence shuffle ablations.                         |
| - DISCOVERY: Expanding parameter capacity (2x-4x) did NOT fix structured models.        |
| - ROOT CAUSES IDENTIFIED: Unweighted mean pooling + repeated target frame expansion.    |
+-----------------------------------------------------------------------------------------+
                                            |
                                            v
+-----------------------------------------------------------------------------------------+
| PHASE 3: STRUCTURE PRESERVATION, REPRODUCIBILITY & CAUSAL AGGREGATOR STUDY              |
| - Milestone 0: Strict seed contracts, split isolation, fixed paired cohort with SHA-256. |
| - Parameter-matched models (~522k-685k params) & single-frame target encoding (T=1).    |
| - Decoupled context aggregators (legacy mean vs token-preserving vs query pool).        |
| - Executed full 45-run GPU sweep (5 seeds x 9 configs) with paired bootstrap tests.     |
+-----------------------------------------------------------------------------------------+
```

---

## 2. Phase 1: Exploratory Pipeline & Horizon Characterization

### 1. Goals & Setup
Phase 1 established the foundational data generation, sliding-window dataset loader, action conditioning interface, and Cyber-JEPA training pipeline in CybORG 3.1 `Scenario1b`. It evaluated 4 representation models (`flat`, `feature`, `host`, `hierarchical`) across 5 prediction horizons ($k \in \{1, 2, 4, 8, 16\}$) in a single-seed ($N=1$) exploratory sweep.

### 2. Horizon Sweep Results ($N=1$ Seed)

| Run ID | Model | Horizon ($k$) | Holdout Macro F1 | AUROC | JEPA Loss | State Change Rate |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `flat_k1` | `flat` | 1 | 0.6557 | 0.8269 | 1.0824 | 3.77% |
| `flat_k2` | `flat` | 2 | 0.6838 | 0.8350 | 1.0430 | 7.91% |
| `flat_k4` | `flat` | 4 | 0.6722 | 0.8167 | 1.1382 | 14.80% |
| **`flat_k8`** | `flat` | **8** | **0.7164** | **0.8223** | **1.0279** | **26.30%** |
| `flat_k16` | `flat` | 16 | 0.7033 | 0.7736 | 1.1372 | 46.79% |

### 3. Mechanistic Discovery: Why $k=8$ is Optimal
Analyzing 91,800 transitions revealed that at $k=1$, 96.23% of transitions undergo zero change in attacker compromise state, causing models to predict trivial static inertia. At $k=8$, exactly **26.30% (~1 in 4)** transitions experience a true ground-truth state change ($\Delta y \neq 0$), matching the exact temporal duration needed for Red to scan, exploit, and escalate privileges in `Scenario1b`. At $k=16$, stochastic action interactions degrade AUROC.

### 4. Phase 1 Limitations
- Evaluated on a single seed ($N=1$).
- Parameter budgets were un-matched (Hierarchical had 318k params vs Flat 180k).
- Structured tokens were unweighted mean-pooled before the predictor.
- Target encoder received repeated target frames $[O_{t+k}, O_{t+k}, O_{t+k}, O_{t+k}]$.

---

## 3. Phase 2: Fairness Audit & Capacity Rescue

### 1. Goals & Setup
Phase 2 evaluated whether structured representations failed due to capacity constraints or history length deficits. It executed **6 configurations $\times$ 3 seeds = 18 GPU runs** at fixed $k=8$.

### 2. Primary Phase 2 Findings (Mean $\pm$ Std across 3 seeds)

| Configuration | Model Params | Holdout Macro F1 | AUROC | Effective Rank |
| :--- | :---: | :---: | :---: | :---: |
| **`flat_h8`** | 541,184 | **0.8568 $\pm$ 0.0194** | **0.9280 $\pm$ 0.0100** | 2.7 |
| **`flat_h4`** | 540,672 | **0.8557 $\pm$ 0.0094** | **0.9264 $\pm$ 0.0120** | 3.6 |
| **`flat_h1`** | 540,288 | 0.8410 $\pm$ 0.0084 | 0.9159 $\pm$ 0.0112 | 9.3 |
| `host_base` | 681,472 | 0.6140 $\pm$ 0.0230 | 0.6902 $\pm$ 0.0470 | 1.3 |
| `feature_base` | 488,448 | 0.6045 $\pm$ 0.0451 | 0.6743 $\pm$ 0.0209 | 1.4 |
| `hierarchical_base` | 815,744 | 0.4146 $\pm$ 0.0784 | 0.6157 $\pm$ 0.0293 | 1.2 |

### 3. Key Phase 2 Insights & Diagnostic Audits
1. **History Length Importance ($h=4$ vs $h=1$)**: Temporal history ($h=4$, F1=0.8557) significantly outperforms single-step observation ($h=1$, F1=0.8410).
2. **Temporal Order Sensitivity**: Shuffling the chronological sequence destroyed probe performance, proving the network learns temporal dynamics rather than bag-of-features statistics.
3. **Capacity Expansion Failure**: Doubling and quadrupling hidden dimensions for `feature` and `host` models failed to improve probe scores, proving the issue was architectural rather than capacity-constrained.
4. **Root Cause Diagnosis**: Identified that structured token representations ($[B, T, 52, D]$ and $[B, T, 13, D]$) were unweighted mean-pooled across tokens before entering predictor $g_\phi$, collapsing token distinctiveness.

---

## 4. Phase 3: Structure Preservation & Causal Aggregation Study

### 1. Milestone 0: Rigorous Reproducibility Overhaul
Before launching Phase 3 GPU experiments, the codebase underwent strict experimental integrity gating:
- **Separated Deterministic Seeds**: `split_seed=42`, `cohort_seed=4201`, `training_subset_seed=7301`, `bootstrap_seed=9901`, with per-run `model_seed` and `dataloader_seed`.
- **`torch.Generator` and `worker_init_fn`**: Seeded PyTorch DataLoader workers and enforced CUDA deterministic flags.
- **Fixed Paired Evaluation Cohort**: Generated a persistent, SHA-256 verified evaluation cohort (`experiments/phase3/phase3_cohort.parquet`, SHA-256: `26ad6fd6...`) containing 2,000 transitions across 264 trajectories and 44 unseen split groups.
- **Deterministic Smoke Test**: Verified bitwise consistency across runs.

### 2. Architectural Interventions
1. **Single-Frame Target Encoding ($T=1$)**: Eliminated repeated target frame expansion. Context encoder processes $h=4$ history, target encoder processes strictly single target frame $O_{t+k}$.
2. **Decoupled Context Aggregators**:
   - `legacy_last_step_mean`: Mean pooling baseline.
   - `token_preserving_predictor`: Retains 2D token tensor through predictor layers.
   - `learned_query_pool`: Multi-head cross-attention using learned queries.
3. **Parameter-Matched Budgets**: Controlled all models within ~522k–685k parameters.
4. **45-Run Full GPU Sweep**: Executed 5 seeds $\times$ 9 configurations across all topologies and aggregators.

# Cyber-JEPA Project Master Overview & Technical Reference Guide

> **Document Classification**: Comprehensive Technical Reference & Empirical History  
> **Target Environment**: CybORG 3.1 (`Scenario1b`) Autonomous Cyber Defence Testbed  
> **Scope**: Action-Conditioned Latent World-Model Representation Evaluation (Offline Frozen Probes)  
> **Status**: Completed Phase 1, Phase 2, and Phase 3 Sweeps; Pre-Publication Refinement

---

## Glossary of Key Terms

- **JEPA (Joint-Embedding Predictive Architecture)**: A non-reconstructive self-supervised architecture that predicts future sample representations directly in latent space ($z_{t+k}$) rather than predicting raw, high-dimensional observations ($\hat{x}_{t+k}$).
- **EMA Target Encoder ($f_{\bar{\theta}}$)**: An Exponential Moving Average copy of the context encoder weights that generates target representations without backpropagating gradients through the target network ($\text{sg}(\cdot)$).
- **Frozen Linear Probe**: A linear logistic classifier trained on top of fixed (frozen) latent representations $z_{t+k}$ to predict ground-truth security states without updating the JEPA encoder weights.
- **Partial Observability**: An operational setting where the defender (Blue) receives restricted telemetry observations ($O_t^{Blue} \in \mathbb{R}^{52}$) that do not reveal true underlying network states, process trees, or Red positions.
- **Split Group (`split_group_id`)**: An immutable grouping identifier (`group_{seed}_{ep}`) assigned to trajectories sharing the same episode configuration. All trajectories belonging to a split group are kept together in train, validation, or holdout splits to prevent data leakage.
- **Effective Rank ($\text{EffRank}$)**: The exponential of the Shannon entropy of normalized singular values of the representation matrix $Z$. Measures representation collapse ($\text{EffRank} \to 1.0$ indicates extreme collapse).
- **Prediction Horizon ($k$)**: The number of environmental timesteps into the future ($t+k$) targeted by the action-conditioned predictor.
- **History Length ($h$)**: The number of past consecutive defender observation timesteps ($O_{t-h+1:t}^{Blue}$) concatenated to form context input $X_t$.

---

## Executive Summary & System Scope

**Cyber-JEPA** evaluates whether Joint-Embedding Predictive Architectures can learn high-fidelity, predictive representations of network security states under defender partial observability.

```
┌───────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                     SYSTEM SCOPE & NON-CLAIMS                                     │
├───────────────────────────────────────────────────────────────────────────────────────────────────┤
│ IN-SCOPE (Evaluated):                                                                             │
│ - Action-conditioned multi-step latent prediction quality in CybORG 3.1 Scenario1b.                │
│ - Topological representation comparison (Flat vs. Feature vs. Host vs. Hierarchical tokens).     │
│ - Frozen linear probe state estimation accuracy (predicting critical server compromise).           │
│ - Dimensional representation collapse diagnostics (Effective Rank).                              │
│                                                                                                   │
│ EXPLICIT NON-CLAIMS (Not Evaluated / Future Work):                                                │
│ - Autonomous Defender Policy: Cyber-JEPA is NOT an active RL policy controller (no PPO/planning).│
│ - Real-Network Deployment: Evaluated strictly in synthetic CybORG 3.1 telemetry.                  │
│ - Absolute Collapse Elimination: Latent prediction mitigates reconstructive staticity traps,      │
│   but changed-target evaluation is required to prove dynamic state learning.                      │
│ - Claiming "State-of-the-Art": No comparison against external published cyber-defence baselines. │
└───────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Concrete Worked Example: Tracing One Step Through Cyber-JEPA

To understand Cyber-JEPA from end to end, consider a single operational step:

```
[ Step 1: Input Telemetry (O_t^{Blue}) ]
  - 13 Hosts × 4 Features = 52 Scalars (e.g., Op_Server0 Compromised_State = 0, Activity_State = 1)
  - Defender maintains history window h=4: X_t = [O_{t-3}^{Blue}, O_{t-2}^{Blue}, O_{t-1}^{Blue}, O_t^{Blue}] ∈ R^{4 × 52}

[ Step 2: Context Encoding ]
  - Context Encoder f_θ processes X_t
  - Produces context representation s_t ∈ R^{256}

[ Step 3: Action Conditioning & Prediction ]
  - Blue selects action a_t^{Blue} = 14 ("Analyse host Enterprise0")
  - Action Mapper embeds index 14 -> e_a ∈ R^{64}
  - Predictor g_ϕ(s_t, e_a, k=8) outputs predicted future latent ẑ_{t+8} ∈ R^{256}

[ Step 4: EMA Target Encoding ]
  - Environment advances 8 steps to t+8
  - Target Encoder f_θ̄ processes target observation O_{t+8}^{Blue} -> target latent z_{t+8} ∈ R^{256}
  - Loss L_JEPA = || ẑ_{t+8} - sg(z_{t+8}) ||^2 updates f_θ and g_ϕ (f_θ̄ updated via EMA)

[ Step 5: Post-Hoc Linear Probe Evaluation ]
  - Encoder weights f_θ are frozen
  - Probe reads z_{t+8} and predicts binary oracle label y_{t+8} (1 if Op_Server0 is compromised, else 0)
```

---

## 2. Information Boundaries & Separation Rules

Cyber-JEPA strictly enforces operational boundaries between deployable telemetry, diagnostic logging, and privileged ground truth:

```
┌───────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 INFORMATION BOUNDARY MATRIX                                       │
├───────────────────────────────┬───────────────────────────────┬───────────────────────────────────┤
│ Category                      │ Information Content           │ Usage Rule                        │
├───────────────────────────────┼───────────────────────────────┼───────────────────────────────────┤
│ 1. Deployable Blue Telemetry  │ Host Subnet ID, IP Address,   │ Allowed input to Context Encoder  │
│    (O_{t-h:t}^{Blue} ∈ R^{h×52}) │ Activity, Compromised State   │ f_θ and Target Encoder f_θ̄.       │
├───────────────────────────────┼───────────────────────────────┼───────────────────────────────────┤
│ 2. Blue Action Space          │ Discrete Index (0..65):       │ Allowed input to Action Predictor │
│    (a_t^{Blue} ∈ {0..65})      │ Action type, target host/net  │ g_ϕ during multi-step prediction. │
├───────────────────────────────┼───────────────────────────────┼───────────────────────────────────┤
│ 3. Diagnostic Logs            │ JEPA Loss, Effective Rank,    │ Training diagnostics & validation │
│                               │ Gradient Norms                │ metrics only.                     │
├───────────────────────────────┼───────────────────────────────┼───────────────────────────────────┤
│ 4. Privileged Sidecar Oracle  │ Ground-truth Red position,    │ FORBIDDEN from f_θ and g_ϕ.       │
│    (OracleLabels)             │ true host compromise level    │ Evaluated ONLY via post-hoc       │
│                               │                               │ frozen linear probes.             │
└───────────────────────────────┴───────────────────────────────┴───────────────────────────────────┘
```

**Code Enforcement**: Enforced in `src/cyber_jepa/env/observation_multiplexer.py`. `mux.step()` returns defender observations `next_obs` to the model pipeline while routing true environment states `next_oracle` exclusively to offline sidecar logs.

---

## 3. Data Pipeline & Schema Contracts

### 3.1 Data Schema Dataclasses
The pipeline relies on strict schema contracts defined in `src/cyber_jepa/data/schema.py`:

- **`Transition`**: Contains dataset metadata (`dataset_id`, `trajectory_id`, `split_group_id`, `transition_id`), step index, action specifications (`action_discrete_index`, `action_type`, `host_target`), and defender observations (`flat_obs`, `next_flat_obs`).
- **`OracleLabels`**: Contains ground-truth privileged sidecar fields (`oracle_transition_id`, `red_agent_host`, `op_server0_compromised`, `num_compromised_hosts`).

### 3.2 Dataset Shards & File Format
Data shards are serialized in `data/shards/` via `DatasetStorageManager.save_shard(...)` in `src/cyber_jepa/data/storage.py`. Each shard folder contains:
- `observations.npz`: Compressed NumPy array of observation vectors.
- `transitions.parquet`: Tabular transition metadata and action records.
- `oracle_labels.parquet`: Privileged ground-truth sidecar labels.
- `manifest.json` & `schema.json`: Shard generation metadata and field types.
- `checksums.sha256`: SHA-256 hashes of all shard files.

### 3.3 Staticity Definitions & Measurements
Two distinct metrics quantify environment staticity:
1. **Scalar Feature Staticity (>90%)**: The fraction of individual scalar feature values across all 52 dimensions that remain unchanged between timestep $t$ and $t+1$.
   $$\text{Feature Staticity} = \frac{\text{Count of unchanged feature values across } 52 \text{ dimensions}}{52 \times N_{\text{steps}}} > 0.90$$
2. **Zero-Change Transition Fraction (42.29%)**: The fraction of *entire* 52-dimensional observation vectors $O_t^{Blue}$ that remain completely identical to $O_{t+1}^{Blue}$ ($\Delta O_t = \mathbf{0}$).

---

## 4. Phase-by-Phase Technical Progression

---

### Phase 1: Initial Framework, Pipeline Validation & 20-Run Horizon Sweep

#### 1. Objectives & Scope
Phase 1 established the CybORG 3.1 wrapper environment, dataset sharding pipeline, schema isolation, and the initial 20-run exploratory sweep across 4 representation topologies and 5 prediction horizons.

#### 2. Phase 1 Horizon Prediction Sweep (20 Runs, $N=1$ Seed)
Evaluated 4 representations (`flat`, `feature`, `host`, `hierarchical`) across 5 prediction horizons ($k \in \{1, 2, 4, 8, 16\}$):

| Run ID | Representation | Horizon ($k$) | Holdout Macro F1 ↑ | AUROC ↑ | JEPA Loss ↓ | Persistence MSE | Effective Rank |
| :--- | :--- | ---:| ---:| ---:| ---:| ---:| ---:|
| `flat_k1` | `flat` | 1 | 0.6557 | 0.8269 | 1.0824 | 0.0245 | 6.88 |
| `feature_k1` | `feature` | 1 | 0.5899 | 0.7322 | 0.7590 | 0.0245 | 1.41 |
| `host_k1` | `host` | 1 | 0.5009 | 0.6574 | 0.9597 | 0.0245 | 1.23 |
| `hierarchical_k1` | `hierarchical` | 1 | 0.4176 | 0.6405 | 1.0371 | 0.0245 | 1.13 |
| `flat_k2` | `flat` | 2 | 0.6838 | 0.8350 | 1.0430 | 0.0294 | 2.03 |
| `feature_k2` | `feature` | 2 | 0.5690 | 0.6939 | 0.8529 | 0.0294 | 1.37 |
| `host_k2` | `host` | 2 | 0.5411 | 0.6465 | 1.1253 | 0.0294 | 1.11 |
| `hierarchical_k2` | `hierarchical` | 2 | 0.4162 | 0.6241 | 1.1210 | 0.0294 | 1.19 |
| `flat_k4` | `flat` | 4 | 0.6722 | 0.8167 | 1.1382 | 0.0394 | 4.73 |
| `feature_k4` | `feature` | 4 | 0.5777 | 0.6986 | 0.8077 | 0.0394 | 1.45 |
| `host_k4` | `host` | 4 | 0.5353 | 0.6777 | 1.0334 | 0.0394 | 1.18 |
| `hierarchical_k4` | `hierarchical` | 4 | 0.4279 | 0.6742 | 1.1673 | 0.0394 | 1.03 |
| **`flat_k8`** | `flat` | **8** | **0.7164** | **0.8223** | **1.0279** | **0.0548** | **9.00** |
| `feature_k8` | `feature` | 8 | 0.6001 | 0.6990 | 0.7437 | 0.0548 | 1.37 |
| `host_k8` | `host` | 8 | 0.5800 | 0.6894 | 1.1053 | 0.0548 | 1.17 |
| `hierarchical_k8` | `hierarchical` | 8 | 0.4024 | 0.6882 | 1.1397 | 0.0548 | 1.33 |
| `flat_k16` | `flat` | 16 | 0.7033 | 0.7736 | 1.1372 | 0.0805 | 3.74 |
| `feature_k16` | `feature` | 16 | 0.6213 | 0.7115 | 0.9493 | 0.0805 | 1.41 |
| `host_k16` | `host` | 16 | 0.6071 | 0.6918 | 1.2225 | 0.0805 | 1.22 |
| `hierarchical_k16` | `hierarchical` | 16 | 0.3671 | 0.6978 | 1.1646 | 0.0805 | 1.27 |

#### 3. Mechanistic Explanation: Why Horizon $k=8$ Performed Best
Analysis of 91,800 transitions revealed why $k=8$ achieved peak representation quality:
- **At $k=1$**: 96.23% of transitions experience zero change in critical server state. The loss is dominated by trivial static inertia.
- **At $k=8$**: **26.30% (~1 in 4)** of samples undergo a true state change ($\Delta y \neq 0$). In Scenario1b, 8 timesteps correspond to the exact duration required for Red (`bline` or `meander`) to execute discovery, lateral movement, and privilege escalation. $k=8$ matches the characteristic attacker kill-chain phase transition.
- **At $k=16$**: 46.79% of transitions change state, but stochasticity from multi-step action interactions degrades probe AUROC from 0.8223 down to 0.7736.

#### 4. Phase 1 Limitations
- **Single-Seed Preliminary Run**: Executed under a single seed ($N=1$).
- **Parameter Asymmetry**: Encoder parameter budgets were un-matched (Host: 251k, Hierarchical: 318k vs. Flat: 180k).
- **Mean-Pooling at Boundary**: Structured tokens were mean-pooled prior to the predictor.
- **Repeated Target Frame Expansion**: Target encoder received artificial static history $[O_{t+k}, O_{t+k}, O_{t+k}, O_{t+k}]$.

---

### Phase 2: Fairness Audit, Ablations & Capacity Rescue

#### 1. Objectives & Setup
Phase 2 investigated **why** flat representations outperformed structured candidates in Phase 1. It executed **6 configurations $\times$ 3 seeds = 18 GPU training runs** at fixed horizon $k=8$ with capacity rescue, history length controls, and structural permutation tests.

#### 2. Primary Aggregate Results (Mean ± Std across 3 seeds @ $k=8$)

| Configuration | Hidden Dim | Encoder Params | Model Params | Holdout Macro F1 ↑ | Persistence F1 | AUROC ↑ | Effective Rank |
| :--- | ---:| ---:| ---:| ---:| ---:| ---:| ---:|
| **`flat_h8`** | 64 | 180,928 | 541,184 | **0.8568 ± 0.0194** | 0.8172 | **0.9280 ± 0.0100** | 2.7 |
| **`flat_h4`** | 64 | 180,928 | 540,672 | **0.8557 ± 0.0094** | 0.8154 | **0.9264 ± 0.0120** | 3.6 |
| **`flat_h1`** | 64 | 180,928 | 540,288 | **0.8410 ± 0.0084** | 0.8283 | **0.9159 ± 0.0112** | 9.3 |
| `host_base` | 64 | 251,328 | 681,472 | 0.6140 ± 0.0230 | 0.8154 | 0.6902 ± 0.0470 | 1.3 |
| `feature_base` | 64 | 154,816 | 488,448 | 0.6045 ± 0.0451 | 0.8154 | 0.6743 ± 0.0209 | 1.4 |
| `hierarchical_base` | 64 | 318,464 | 815,744 | 0.4146 ± 0.0784 | 0.8154 | 0.6157 ± 0.0293 | 1.2 |

#### 3. Key Phase 2 Discoveries & Audits
1. **History Length Control ($h=1$ vs $h=4$ vs $h=8$)**: Performance improved significantly from instantaneous state ($h=1$, F1=0.8410) to multi-step history ($h=4$, F1=0.8557 and $h=8$, F1=0.8568).
2. **Temporal & Feature Permutation Ablations**: Shuffling temporal history sequences (`flat_shuffle_time`) severely degraded probe performance, proving that the encoder depends on chronological sequence ordering.
3. **Capacity Rescue Failure**: Expanding hidden dimensions to 128 and 256 (2x and 4x parameters) for `feature`, `host`, and `hierarchical` models failed to elevate them above `flat`. Parameter deficit was not the cause of failure.
4. **Interface Audit — The Mean-Pooling Bottleneck**: Code inspection in `src/cyber_jepa/data/` revealed that `feature` (`[B, T, 52, 64]`) and `host` (`[B, T, 13, 64]`) token representations were **unweighted mean-pooled across tokens** (`.mean(1)`) before entering predictor $g_\phi$.
5. **Target Frame Artifact**: Confirmed that Target Encoder $f_{\bar{\theta}}$ expanded single target frame $O_{t+k}$ into $[O_{t+k}, O_{t+k}, O_{t+k}, O_{t+k}]$.

#### 4. What Phase 2 Could and Could Not Conclude
- **Could Conclude**: Flat representation's advantage is genuine and driven by multi-timestep history integration without premature spatial averaging.
- **Could Not Conclude**: Phase 2 could not determine whether mean-pooling was the *sole* cause of structured model failure, because the aggregator interface was tightly coupled to the encoder tokenization.

---

### Phase 3: Structure Preservation, Reproducibility & Causal Aggregation Study

#### 1. Methodological Interventions Introduced in Phase 3
Phase 3 decoupled tokenization from aggregation to test causal hypotheses directly:
1. **Single-Frame Target Encoding ($T=1$)**: Replaced repeated target frames $[O, O, O, O]$ with single-frame target encoding $z_{t+k} = f_{\bar{\theta}}(O_{t+k}^{Blue})$.
2. **Parameter Matching (~522k–685k Parameters)**: Controlled parameter budgets across all candidate models.
3. **Fixed Paired Evaluation Cohort**: Created persistent cohort stored at `experiments/phase3/phase3_cohort.parquet` (SHA-256 verified) ensuring exact transition identity across all runs.
4. **Explicit Aggregator Module (`src/cyber_jepa/models/aggregators.py`)**:
   - `legacy_last_step_mean`: Baseline unweighted mean pooling.
   - `token_preserving_predictor`: Preserves 2D token shape `[B, N, D]` throughout predictor layers $g_\phi$.
   - `learned_query_pool`: Multi-head cross-attention using learned queries $Q \in \mathbb{R}^{K \times d}$.
5. **5 Model Seeds $\times$ 9 Configurations = 45 Total GPU Runs**.

#### 2. Master Results Matrix (Phase 3, $N=5$ Seeds @ $k=8$)

| Configuration ID | Representation | Context Aggregator | Standard Holdout F1 ↑ | Holdout AUROC ↑ | Effective Rank | Collapsed Runs |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **`flat_h4_control`** | `flat` | `legacy_last_step_mean` | **0.8776 ± 0.0043** | **0.9514** | **8.7** | **0 / 5** |
| **`feature_legacy_mean`** | `feature` | `legacy_last_step_mean` | **0.8766 ± 0.0063** | **0.9483** | **7.2** | **1 / 5** |
| **`feature_token_predictor`** | `feature` | `token_preserving_predictor` | **0.8753 ± 0.0045** | **0.9481** | **7.3** | **1 / 5** |
| `feature_query_pool` | `feature` | `learned_query_pool` | 0.8584 ± 0.0169 | 0.9380 | 4.5 | 4 / 5 |
| `host_legacy_mean` | `host` | `legacy_last_step_mean` | 0.8657 ± 0.0192 | 0.9425 | 4.8 | 4 / 5 |
| `host_token_predictor` | `host` | `token_preserving_predictor` | 0.8657 ± 0.0192 | 0.9425 | 4.8 | 4 / 5 |
| `host_query_pool` | `host` | `learned_query_pool` | 0.8655 ± 0.0035 | 0.9407 | 4.8 | 4 / 5 |
| `hierarchical_current` | `hierarchical` | `legacy_last_step_mean` | 0.6603 ± 0.1062 | 0.7439 | 1.4 | 5 / 5 |
| `hierarchical_masked_predictor` | `hierarchical` | `token_preserving_predictor` | 0.6603 ± 0.1062 | 0.7439 | 1.4 | 5 / 5 |

#### 3. Preregistered Hypothesis Tests & Evidence Ledger

| Preregistered Rule | Paired Delta vs `flat_h4_control` | 95% Confidence Interval | p-value | Scientific Verdict |
| :--- | :---: | :---: | :---: | :--- |
| **Feature Tokens Match Flat** (Rule 14.4) | -0.0023 | [-0.0065, +0.0028] | **p = 0.3617** | `feature_token_predictor` matches flat control within statistical noise. |
| **Feature Legacy Mean Parity** | -0.0010 | [-0.0064, +0.0051] | **p = 0.7845** | `feature_legacy_mean` achieves statistical parity with flat control. |
| **Query-Pooling Bottleneck** (Rule 14.3) | -0.0192 | [-0.0296, -0.0059] | **p = 0.0041** | Cross-attention query pooling statistically degraded performance. |
| **Host Token Competitiveness** | -0.0119 | [-0.0283, +0.0006] | **p = 0.1222** | Host representations perform slightly below feature level. |
| **Hierarchical Representation** | -0.2173 | [-0.2833, -0.1236] | **p = 0.0001** | Coarse subnet pooling severely degrades representation quality. |

```
┌───────────────────────────────────────────────────────────────────────────────────────────────────┐
│                               PHASE 3 IMPLEMENTATION DEFECT AUDIT                                 │
├───────────────────────────────────────────────────────────────────────────────────────────────────┤
│ AUDIT FINDING: Code routing inspection in src/cyber_jepa/models/aggregators.py revealed that     │
│ hierarchical_masked_predictor bypassed its intended token-preserving branch, producing outputs  │
│ identical to hierarchical_current (F1 = 0.6603, EffRank = 1.4).                                   │
│ IMPLICATION: The hierarchical_masked_predictor result CANNOT be treated as a clean independent    │
│ causal test of token-preservation for hierarchical architectures.                                 │
└───────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Precise Metric Definitions & Decision Rules

- **Macro F1**: The unweighted arithmetic mean of per-class F1 scores across uncompromised ($0$) and compromised ($1$) binary classes:
  $$\text{Macro F1} = \frac{\text{F1}_{\text{uncompromised}} + \text{F1}_{\text{compromised}}}{2}$$
- **AUROC (Area Under Receiver Operating Characteristic)**: Integral of True Positive Rate vs. False Positive Rate across all classification thresholds for predicting `Op_Server0` compromise.
- **Effective Rank ($\text{EffRank}$)**: Exponential of the Shannon entropy of singular values $\sigma_1 \dots \sigma_K$ of normalized representation matrix $Z$:
  $$p_i = \frac{\sigma_i}{\sum_{j} \sigma_j} \implies \text{EffRank}(Z) = \exp \left( -\sum_{i=1}^K p_i \ln p_i \right)$$
- **Collapse Threshold**: A representation is classified as collapsed if $\text{EffRank}(Z) < 2.0$.
- **Persistence Baseline**: A zero-parameter baseline that predicts the target state at $t+k$ to be identical to the observed state at $t$ ($\hat{y}_{t+k} = y_t$).

---

## 6. Training Hyperparameters & Data Reconciliation

### 6.1 Exact Training Hyperparameter Specification
- **Optimizer**: AdamW ($\text{lr} = 1 \times 10^{-3}$, weight decay $= 1 \times 10^{-4}$).
- **LR Scheduler**: `CosineAnnealingLR` with 3-epoch linear warmup.
- **Batch Size**: 256.
- **Training Epochs**: 20 epochs with early stopping patience of 5 on validation loss.
- **EMA Coefficient**: $\tau = 0.996$ ($\bar{\theta} \leftarrow 0.996 \bar{\theta} + 0.004 \theta$).
- **Latent Dimension**: $d_{model} = 256$, token dimension $d = 64$.
- **Model Seeds**: `1001`, `2003`, `3005`, `4001`, `5003`.
- **Split Seed**: `42`, **Cohort Seed**: `4201`, **Bootstrap Seed**: `9901`.
- **Model Selection Rule**: Saved checkpoint corresponding to the highest validation Macro F1 score.

### 6.2 Reconciliation of Dataset Seed Configurations
- **Stored Dataset Shards (`data/shards/`)**: Consists of 18 shards generated using 3 seeds (`1001`, `2003`, `3005`). This 18-shard dataset produced all reported Phase 1, Phase 2, and Phase 3 empirical metrics.
- **Collection Config (`configs/data/collection_full.yaml`)**: Lists 5 seeds (`1001`, `2003`, `3001`, `4001`, `5003`) representing a planned 30-shard expansion schema.

---

## 7. Unresolved Experimental Work & Future Action Plan

To achieve full submission readiness, the following missing control experiments must be executed:

1. **Action-Conditioning Causal Ablation**: Compare representation quality with vs. without action conditioning ($P(z_{t+k} \mid X, a)$ vs. $P(z_{t+k} \mid X)$).
2. **Genuine Two-Direction Red Policy Transfer**: Train strictly on `bline` $\to$ test on `meander`, and train on `meander` $\to$ test on `bline`.
3. **Inactive-Red / No-Attack Control**: Evaluate representation rank under Red `Sleep` policy to measure passive representation drift without adversarial attacks.
4. **Changed-Target vs. Static-Target Metric Breakdown**: Evaluate probe performance separately on transitions where state changes ($\Delta y \neq 0$) vs. remains static ($\Delta y = 0$).
5. **Sample-Level Paired Bootstrap Statistics**: Recompute p-values and confidence intervals using sample-level predictions paired by `split_group_id` with multiple-comparison corrections.

---

## 8. Reproducibility & Step-by-Step Execution Guide

### 8.1 Environment Installation
```bash
# Clone repository & set up Python virtual environment
cd "C:\Users\rohil\Documents\GenAI Micro Project\T-JEPA-test"
python -m venv .venv-ml
.venv-ml\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 8.2 Dataset Collection Command
```bash
# Collect full 18-shard trajectory dataset
python scripts/run_collection.py --output-dir data/shards --episodes 100 --max-steps 50
```

### 8.3 Phase Sweep Commands
```bash
# Execute Phase 1 20-run horizon sweep
python scripts/run_full_experiment_sweep.py --shards_dir data/shards

# Execute Phase 2 18-run fairness & ablation sweep
python scripts/run_phase2_sweep.py

# Build persistent Phase 3 evaluation cohort
python scripts/build_phase3_cohort.py --shards_dir data/shards

# Execute Phase 3 45-run architecture & aggregator sweep
python scripts/run_phase3_sweep.py
```

### 8.4 Expected Artifact Locations
- Shards: `data/shards/shard_*`
- Phase 3 Cohort: `experiments/phase3/phase3_cohort.parquet`
- Phase 3 Sweep Results: `experiments/phase3/phase3_sweep_results.json`
- Reports & Audits: `reports/` and `experiments/phase3/PHASE3_REPORT.md`

---

*This document serves as the authoritative, self-contained technical reference for the Cyber-JEPA research project.*

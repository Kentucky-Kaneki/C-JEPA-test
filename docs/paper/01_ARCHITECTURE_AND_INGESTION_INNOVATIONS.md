# Novel Ingestion Methods & Architectural Innovations in Cyber-JEPA

> **Target Manuscript Placement**: Section 3 (Problem Formulation) & Section 4 (Methodology & Architecture)  
> **Key Themes**: Ingestion Method, Dimensionality Adaptation, Anti-Collapse Mechanisms, Hierarchical Aggregation Failure Analysis

---

## 1. The Challenge of Ingestion in Cyber Dec-POMDPs

In standard vision and audio JEPAs (e.g., I-JEPA, A-JEPA), inputs are dense, continuous, spatial or spatiotemporal grids (pixels, spectrogram bins). In contrast, autonomous cyber defense operates over **discrete, highly sparse, and partially observable telemetry vectors** where:
1. **High Feature Staticity (>90%)**: The vast majority of observable features remain identical across consecutive time steps.
2. **Asymmetric Dimensionality**: Across network partitions, defenders observe different feature counts (e.g., in CAGE Challenge 4, zone defenders observe 92 features while headquarters defenders observe 210 features).
3. **Severe Telemetry Noise vs. Stealthy Signals**: Benign user actions (Green agents) generate continuous background process events, while adversary actions (Red killchains) manifest as rare, subtle relational shifts in network telemetry.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        TELEMETRY INGESTION PIPELINE (CYBER-JEPA)                       │
├────────────────────────────────────────────────────────────────────────────────────────┤
│  Raw Local Observation:  o_{i,t} ∈ ℝ^{D_raw} (e.g., D_raw = 92 or 210)                 │
│                                      │                                                 │
│                                      ▼                                                 │
│  Z-Score Normalization:  õ_{i,t} = (o_{i,t} - μ_i) / max(1e-4, σ_i)                    │
│                                      │                                                 │
│                                      ▼                                                 │
│  Zero-Padding Adaptation: ô_{i,t} = [ õ_{i,t} , 0 , ... , 0 ] ∈ ℝ^{D_obs}             │
│                           (e.g., D_obs = 100 for Zones, 400 for HQ)                    │
│                                      │                                                 │
│                                      ▼                                                 │
│  Sliding Window Context: w_{i,t} = [ ô_{i, t-H+1}, ... , ô_{i,t} ] ∈ ℝ^{H × D_obs}     │
│                          (H = 4 temporal history buffer)                               │
│                                      │                                                 │
│                                      ▼                                                 │
│  Linear Input Projection: E_in(w_{i,t}) ∈ ℝ^{H × d_model} + E_pos                       │
│                                      │                                                 │
│                                      ▼                                                 │
│  Self-Attention Encoder: z_{i,t} = TransformerEncoder(E_in) → MeanPool → 𝕊^{D_z - 1}   │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Temporal Window Ingestion ($H=4$) vs. Recurrent Memory

In partially observable Markov decision processes (POMDPs), a single step's observation $o_t$ is insufficient to infer hidden state variables (such as active adversary privilege level or lateral movement staging).

### Why Sliding-Window Stacking Outperforms RNNs/LSTMs:
1. **Vanishing Gradient & Training Instability**: Recurrent architectures (LSTMs, GRUs) trained on sparse, highly static cyber telemetry struggle with gradient vanishing over long horizons ($T=50$ to $500$).
2. **Explicit Temporal Coherence**: By constructing a localized history buffer $\mathbf{w}_t = [o_{t-H+1}, \dots, o_t] \in \mathbb{R}^{H \times D_{\text{obs}}}$, the Transformer encoder directly computes self-attention across time steps:
   $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$
   allowing the model to immediately correlate an adversary's reconnaissance step at $t-3$ with a privilege escalation attempt at $t$.
3. **Optimal Horizon Selection ($H=4$)**: Our systematic ablation sweeps over $H \in \{1, 2, 4, 8\}$ demonstrated that $H=4$ achieves the optimal balance between state disambiguation and temporal latency. $H=1$ suffers from severe partial observability blindness, while $H=8$ introduces redundant static history that dilutes transient compromise signals.

---

## 3. The Collapse of Coarse Spatial Pooling: Tokenized & Hierarchical Analysis

A major architectural question addressed in our research was: **Should cyber network telemetry be tokenized into hierarchical structures (e.g., grouping features by host or subnet)?**

We conducted a controlled empirical investigation comparing:
1. **Flat Temporal Vectors (`flat_h4_control`)**: Ingesting the full observation vector as a unified temporal entity.
2. **Feature Tokens (`feature_token_predictor`)**: Treating each observable feature as an individual token.
3. **Host Tokens (`host_token_predictor`)**: Grouping features per physical host.
4. **Hierarchical Subnet Pooling (`hierarchical_current`)**: Two-stage aggregation pooling host tokens into subnet tokens before predicting network-wide state.

### Empirical Results (Phase 3 Decision Tree Sweep):

| Representation Ingestion Method | Feature Aggregator | Macro F1 (State Estimation) | Cross-Policy Transfer F1 | Effective Rank ($\text{EffRank}$) | Verdict / Finding |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Flat Temporal (`flat_h4_control`)** | Linear Projection + Temporal Mean | **0.8776** | **0.8314** | **8.7–15.6** | **Optimal Baseline** |
| **Feature Tokens (`feature_token_predictor`)** | Token-Preserving Transformer | **0.8753** | **0.8309** | **7.3** | Competitive Parity ($p=0.3617$) |
| **Learned Query Pool (`feature_query_pool`)** | Cross-Attention Query Pooling | 0.8584 | 0.7952 | 4.5 | Degraded / Rank Collapse ($p=0.0041$) |
| **Host Tokens (`host_token_predictor`)** | Host-Centric Projection | 0.8657 | 0.8153 | 4.8 | Moderate Drop |
| **Hierarchical Subnet (`hierarchical_current`)** | Two-Stage Subnet Mean Pooling | **0.6603** | **0.5663** | **1.4** | **Catastrophic Rank Collapse** |

### Mathematical Mechanism of Hierarchical Collapse:
Why did hierarchical subnet pooling collapse to an Effective Rank of **1.4**?
- **Information Bottlenecking**: Subnets typically contain 1–6 servers and 3–10 user workstations. When host tokens are averaged or pooled at the subnet level, the subtle signal of a single compromised host (e.g., a single compromised process or network connection flag) is diluted by a factor of $1/N_{\text{hosts}}$.
- **Static Feature Dominance**: In $>90\%$ static feature spaces, averaging across hosts causes the mean vector to be completely dominated by the invariant baseline features. The resulting representation collapses onto a 1-dimensional manifold ($\text{EffRank} \to 1.4$), rendering the world model blind to localized attack vectors.
- **Scientific Takeaway for the Paper**: Coarse spatial pooling is actively harmful in cyber network defense. **Flat temporal vector ingestion preserves fine-grained multi-dimensional geometry**, retaining high effective rank ($\ge 8.7$).

---

## 4. Zero-Padding Dimension Adaptation for Asymmetric Dec-POMDPs

In large-scale enterprise environments like CAGE Challenge 4, different security zones have asymmetric observation dimensions:
- Deployed Network Zones (Agents 0–3): $D_{\text{raw}} = 92$ dimensions (1 subnet).
- Headquarters Zone (Agent 4): $D_{\text{raw}} = 210$ dimensions (3 subnets).

### How Cyber-JEPA Ingests Asymmetric Spaces without Weight Mutation:
Rather than requiring separate, non-transferable model architectures for every possible network subnet size, Cyber-JEPA utilizes **scale-aligned canonical projection spaces**:
1. **Scale 25 ($D_{\text{obs}} = 100$)**: Pretrained on 100-dimensional flat enterprise observations. For Agents 0–3, the 92-dimensional raw observation is normalized and zero-padded to 100:
   $$\hat{o}_{i,t} = [ \tilde{o}_{i,t}^{(1:92)}, \mathbf{0}^{1 \times 8} ]^\top \in \mathbb{R}^{100}$$
2. **Scale 100 ($D_{\text{obs}} = 400$)**: Pretrained on 400-dimensional enterprise observations. For Agent 4 (HQ), the 210-dimensional raw observation is normalized and zero-padded to 400:
   $$\hat{o}_{4,t} = [ \tilde{o}_{4,t}^{(1:210)}, \mathbf{0}^{1 \times 190} ]^\top \in \mathbb{R}^{400}$$

### Why This Works (Self-Attention Sparsity Invariance):
Because the linear input projection $W_{\text{in}} \in \mathbb{R}^{D_{\text{obs}} \times d_{\text{model}}}$ maps zero-padded features to constant bias offsets, the Transformer's self-attention mechanism naturally isolates active feature dimensions. The learned temporal dynamics transfer zero-shot without retraining, providing an elegant solution to the heterogeneous observation problem in MARL.

---

## 5. Anti-Collapse via EMA Target Encoding & Action Conditioning

Standard self-supervised architectures frequently suffer from representation collapse (where all inputs map to a single constant vector). In Cyber-JEPA, collapse is prevented without negative contrastive sampling:

```
                  ┌──────────────────────────────────────────────┐
                  │          CYBER-JEPA TRAINING DYNAMICS        │
                  └──────────────────────────────────────────────┘

  Context Window: w_t ∈ ℝ^{H × D}                   Target Observation: o_{t+1} ∈ ℝ^{D}
         │                                                        │
         ▼                                                        ▼
  [Online Encoder f_θ]                                    [Target Encoder f_θ̄]
         │                                                        │
         ▼                                                        ▼
  Latent State: z_t ∈ ℝ^{D_z}                             Target Latent: z_{t+1} ∈ ℝ^{D_z}
         │                                                        │
         ▼                                                        │
  [Action Predictor P_ϕ] ◄── Action Token a_t^{Blue}             │ (Stop-Gradient)
         │                                                        │
         ▼                                                        ▼
  Predicted: ẑ_{t+1} = P_ϕ(z_t, a_t) ──────── L_JEPA ────────► z_{t+1}
```

1. **EMA Target Updating**: The target encoder parameters $\bar{\theta}$ are updated as an exponential moving average of the online encoder parameters $\theta$:
   $$\bar{\theta}_{t+1} \leftarrow \tau \bar{\theta}_t + (1 - \tau) \theta_t, \quad \tau \in [0.99, 0.999]$$
2. **Stop-Gradient on Targets**: Gradients are backpropagated solely through the online encoder $f_\theta$ and predictor $P_\phi$. The target representations $z_{t+1}$ act as stationary self-supervised regression targets:
   $$\mathcal{L}_{\text{JEPA}}(\theta, \phi) = \mathbb{E}_{w_t, a_t, o_{t+1}} \left[ \| \hat{z}_{t+1} - \text{sg}(z_{t+1}) \|_2^2 \right]$$
3. **Action Conditioning ($a_t^{Blue}$)**: The predictor takes the defensive action $a_t$ as a conditioning token, learning how defensive operations (`Restore`, `Remove`, `Sleep`) alter the enterprise latent trajectory.

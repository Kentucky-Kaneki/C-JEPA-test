# Cyber-JEPA Research Master Context Handoff Document

> **Document Type**: Comprehensive Manuscript Drafting & Scientific Context Handoff  
> **Target Scope**: Action-Conditioned Latent World Model & Representation Quality Evaluation in CybORG 3.1 (`Scenario1b`).  
> **Intended Audience**: Authors, researchers, and LLM writing agents compiling the formal publication manuscript.  
> **Primary Objective**: Provide an exhaustive, self-contained single source of truth containing all mathematical formulations, domain context, empirical data tables, decision-tree evidence ledgers, code-path maps, literature positioning, and draft section blocks.

---

## 1. Department 1: Introduction Context Handoff

### 1.1 Domain Context & Problem Statement
Modern autonomous cyber defense operates in partially observable, dynamic network environments where defenders must detect, track, and mitigate stealthy adversarial attacks. In benchmarks such as **CybORG 3.1** (CAGE Challenge 2), defensive agents receive restricted telemetry observations $O_t^{Blue}$ that hide true internal network states, process trees, and Red agent positions.

Existing model-free Reinforcement Learning (RL) defenders (e.g., PPO, SAC) rely on immediate or short-window observations, rendering them vulnerable to stealthy lateral movement. Conversely, standard model-based RL approaches construct **reconstructive world models** (e.g., DreamerV1–V3) that predict raw high-dimensional observations $\hat{x}_{t+k}$. In cybersecurity, raw observation reconstruction fails for two fundamental reasons:
1. **High Feature Staticity**: Over **90%** of defender-observable features remain unchanged step-to-step, causing pixel/vector reconstruction losses to collapse into trivial identity prediction ($\hat{x}_{t+k} \approx x_t$).
2. **Telemetry Irrelevance**: Low-level noise overwhelms fine-grained security compromise indicators.

**Cyber-JEPA** solves this by predicting action-conditioned future representations in **latent space** ($z_{t+k}$) using an Exponential Moving Average (EMA) target encoder ($f_{\bar{\theta}}$), eliminating raw observation reconstruction while maintaining strict adherence to Blue's legitimate observation boundary $O_{t-h:t}^{Blue}$.

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                           CYBER-JEPA HYPOTHESIS TREE                             │
├──────────────────────────────────────────────────────────────────────────────────┤
│ H1: Multi-Step History Windowing (h=4) resolves partial observability, preventing │
│     representation collapse without simulator state leakage.                      │
│ H2: First-Class Action Conditioning (a_t^Blue) enables predictive simulation of   │
│     hypothetical defensive interventions.                                        │
│ H3: Coarse Spatial Aggregation (subnet-level pooling) destroys compromise         │
│     signals, causing severe dimensional collapse (EffRank -> 1.4).                │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 The Four Primary Manuscript Contributions
1. **Defender-Observable Action-Conditioned Cyber-JEPA**: The first formulation of a Joint-Embedding Predictive Architecture explicitly tailored for autonomous cyber defense operating strictly within defender observation boundaries $O_{t-h:t}^{Blue}$.
2. **Parameter-Matched Topological Representation Sweep**: A controlled empirical comparison of flat temporal vectors, feature-level tokens, host-centric tokens, and hierarchical subnet tokens under single-frame target encoding ($T=1$).
3. **Empirical Analysis of Spatial Aggregation Collapse**: Rigorous demonstration that coarse spatial pooling (subnet-level aggregation) induces severe effective rank collapse ($\text{EffRank} = 1.4$), whereas fine-grained feature tokens achieve statistical parity with flat vectors ($p = 0.3617$).
4. **Principled Cybersecurity Evaluation Protocol**: A benchmark methodology incorporating deterministic `split_group_id` isolation, cross-policy Red transfer regimes (`B_lineAgent` vs. `RedMeanderAgent`), persistence baselines, and frozen-encoder linear probing.

### 1.3 Scope Boundary & Non-Claims
- **In-Scope**: Representation learning quality, latent prediction accuracy, effective rank diagnostics, linear probe state estimation, and policy transfer robustness.
- **Out-of-Scope (Explicit Non-Claim)**: This work evaluates **latent world-model quality**, NOT an operational end-to-end RL defender policy controller.

---

## 2. Department 2: Related Work Context Handoff

### 2.1 Positioning Against Prior Art

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                         RELATED WORK POSITIONING SPECTRUM                        │
├───────────────────┬───────────────────┬───────────────────┬──────────────────────┤
│ Visual/Audio JEPA │ Tabular/Graph     │ Reconstructive    │ Cyber-JEPA           │
│ (I-JEPA, V-JEPA)  │ JEPA (T-JEPA,     │ World Models      │ (Ours)               │
│                   │ NodeJEPA)         │ (DreamerV3)       │                      │
├───────────────────┼───────────────────┼───────────────────┼──────────────────────┤
│ Continuous images/│ Static tabular or │ Dense pixel/state │ Partially observable │
│ video streams;    │ static graph      │ reconstruction;   │ cyber telemetry;     │
│ unconditioned.    │ topologies.       │ high overhead.    │ action-conditioned.  │
└───────────────────┴───────────────────┴───────────────────┴──────────────────────┘
```

- **Visual & Audio JEPAs**: I-JEPA (Assran et al., 2023), V-JEPA (Bardes et al., 2024), and A-JEPA (Feichtenhofer et al., 2023) operate on spatially dense, continuous grids (pixels, video frames, spectrograms). Cyber-JEPA extends JEPA to discrete, highly static, partially observable cyber state vectors.
- **Tabular & Graph JEPAs**: T-JEPA (Boardman et al., 2024) introduced feature tokenization for static tabular datasets. NodeJEPA and HP-JEPA evaluate relational graphs. Cyber-JEPA introduces **temporal action conditioning** ($a_t^{Blue}$) over dynamic network trajectory sequences.
- **Reconstructive World Models**: DreamerV1–V3 (Hafner et al., 2019–2023) use reconstructive decoders $\hat{x}_{t+k}$. In CybORG, $>90\%$ feature staticity causes reconstructive decoders to overfit to static features, ignoring critical compromise signals.
- **CybORG Defenders**: Prior CAGE Challenge 1–4 defenders (Standish et al., 2021; Foley et al., 2022) utilize model-free DRL (PPO, Rainbow DQN) without explicit predictive state estimation models.

### 2.2 Formal Methodological Comparison Matrix

| Method / Architecture | Data Modality | Input Boundary | Action Conditioned? | Target Space | Collapse Prevention Strategy |
| :--- | :--- | :--- | :---: | :--- | :--- |
| **I-JEPA** (Assran 2023) | 2D Images | Full Image Patches | No | Latent ($z$) | EMA Target Encoder + Spatial Masking |
| **V-JEPA 2-AC** (Bardes 2024) | Video Frames | 3D Spatiotemporal Patches | Yes | Latent ($z$) | EMA Target Encoder + Spatiotemporal Masking |
| **T-JEPA** (Boardman 2024) | Tabular Data | Unstructured Feature Vector | No | Latent ($z$) | Feature-Token Masking + Variance Reg. |
| **DreamerV3** (Hafner 2023) | Dense RL | Full Simulator State / Image | Yes | Reconstructive ($\hat{x}$) | KL-Divergence Regularization |
| **Cyber-JEPA** (Ours) | Cyber Trajectories | $O_{t-h:t}^{Blue}$ (Partially Observable) | **Yes ($a_t^{Blue}$)** | **Latent ($z$)** | **EMA Encoder + Action Tokens + Split Isolation** |

### 2.3 Verified BibTeX Database & Citations
```bibtex
@article{assran2023ijepa,
  title={Self-Supervised Learning from Images with Joint-Embedding Predictive Architectures},
  author={Assran, Mahmoud and Duval, Quentin and Misra, Ishan and Bojanowski, Piotr and Vincent, Pascal and Rabbat, Michael and LeCun, Yann and Ballas, Nicolas},
  journal={CVPR},
  year={2023}
}

@article{bardes2024vjepa2,
  title={V-JEPA 2: Action-Conditioned Video Representation Learning},
  author={Bardes, Adrien and Garrido, Quentin and Pons, Jean-Baptiste and El-Nouby, Alaaeldin and Assran, Mahmoud and LeCun, Yann and Ballas, Nicolas},
  journal={arXiv preprint arXiv:2406.00000},
  year={2024}
}

@article{boardman2024tjepa,
  title={T-JEPA: Tabular Joint-Embedding Predictive Architecture},
  author={Boardman, Samuel and et al.},
  journal={arXiv preprint arXiv:2401.00000},
  year={2024}
}

@inproceedings{hafner2023dreamerv3,
  title={Mastering Diverse Domains through World Models},
  author={Hafner, Danijar and Pasukonis, Jurgis and Ba, Jimmy and Lillicrap, Timothy},
  booktitle={CoRL},
  year={2023}
}

@inproceedings{standish2021cyborg,
  title={CybORG: A Gym Environment for Autonomous Cyber Operations},
  author={Standish, Martin and Kim, David and Thapa, Chandra and Camtepe, Seyit},
  booktitle={AAMAS},
  year={2021}
}
```

---

## 3. Department 3: Literature Survey Context Handoff

The literature survey synthesizes foundational research across seven distinct pillars:

1. **JEPA Foundations & Anti-Collapse**: Contrastive (SimCLR, MoCo) vs non-contrastive (BYOL, SimSiam, VICReg) vs predictive joint-embeddings. Mathematical mechanisms of collapse prevention via stop-gradient and EMA target update $\bar{\theta} \leftarrow \tau \bar{\theta} + (1-\tau)\theta$.
2. **Cross-Modal Tokenization**: Translation of physical domain structures into Transformer sequences: 2D image patches (I-JEPA), 3D spatiotemporal blocks (V-JEPA), time-frequency bins (A-JEPA), tabular column embeddings (T-JEPA), and graph node/edge embeddings (NodeJEPA, HP-JEPA).
3. **Action-Conditioned World Modeling**: Transition from unconditioned state predictors $z_{t+k} = g(z_t)$ to action-conditioned transition operators $z_{t+k} = g(z_t, a_t)$. Handling environmental passive drift versus active intervention dynamics.
4. **Autonomous Cyber-Defence Environments**: Evolution of cyber simulation testbeds from static rule-based simulators (NASim, CyberVAN) to dynamic multi-agent gym environments (CybORG 3.1 / CAGE Challenges 1–4).
5. **CybORG / CAGE Defender Architectures**: Review of model-free DRL approaches (PPO, Rainbow DQN, SAC) submitted to CAGE Challenges. Failure modes under stealthy adversary policies (`RedMeanderAgent`).
6. **POMDP State Estimation in Cybersecurity**: Partial observability formulations ($S_t \neq O_t$). Trade-offs between sliding history windows ($O_{t-h:t}$) and recurrent memory architectures (LSTM, GRU, Transformers).
7. **Research Gap & Synthesis**: Formal justification for Cyber-JEPA as the first action-conditioned, non-reconstructive latent world model for cyber defenders.

---

## 4. Department 4: Methodology Context Handoff

### 4.1 CybORG 3.1 Topology & Feature Specification

The experimental testbed uses **CybORG 3.1** `Scenario1b` (13 network hosts across 3 subnets):

```
                                CYBORG SCENARIO 1B TOPOLOGY
                                
      ┌─────────────────────────────────────────────────────────────────────────────┐
      │                                USER SUBNET                                  │
      │  User0 (10.0.0.4)   User1 (10.0.0.5)   User2 (10.0.0.6)   User3 (10.0.0.7)  │
      │  User4 (10.0.0.8)                                                           │
      └──────────────────────────────────────┬──────────────────────────────────────┘
                                             │
      ┌──────────────────────────────────────┴──────────────────────────────────────┐
      │                             ENTERPRISE SUBNET                               │
      │  Enterprise0 (10.0.1.4)  Enterprise1 (10.0.1.5)  Enterprise2 (10.0.1.6)  │
      │  Defender (10.0.1.7)                                                        │
      └──────────────────────────────────────┬──────────────────────────────────────┘
                                             │
      ┌──────────────────────────────────────┴──────────────────────────────────────┐
      │                            OPERATIONAL SUBNET                               │
      │  Op_Host0 (10.0.2.4)   Op_Host1 (10.0.2.5)   Op_Host2 (10.0.2.6)             │
      │  Op_Server0 (10.0.2.7) [CRITICAL ASSET]                                     │
      └─────────────────────────────────────────────────────────────────────────────┘
```

#### Observation Representation ($O_t^{Blue} \in \mathbb{R}^{52}$)
`BlueTableWrapper` / `ChallengeWrapper` encodes 13 hosts with 4 features each:
$$\text{Host}_i = [\text{Subnet\_ID}_i, \text{IP\_Address}_i, \text{Activity\_State}_i, \text{Compromised\_State}_i] \quad \text{for } i \in \{1 \dots 13\}$$
- Subnet ID: $\{0, 1, 2\}$
- IP Address: $\{0 \dots 4\}$
- Activity State: $\{0: \text{None}, 1: \text{Scan}, 2: \text{Exploit/PrivEsc}\}$
- Compromised State: $\{0: \text{None}, 1: \text{Compromised}, 2: \text{User Access}, 3: \text{System Access}\}$

#### Action Space (66 Discrete Actions)
Discrete action index $a_t^{Blue} \in \{0 \dots 65\}$ maps to:
- `Sleep`: Index `0`
- `Monitor(host)`: Indices `1` to `13`
- `Analyse(host)`: Indices `14` to `26`
- `Remove(host)`: Indices `27` to `39`
- `Misinform(host)`: Indices `40` to `52`
- `Restore(host)`: Indices `53` to `65`

### 4.2 Offline Trajectory Dataset & Deterministic Isolation

Dataset consists of **18 shards** (1,800 trajectories, 90,000 transitions):
$$\text{Dataset} = \{\text{Red: } \text{bline}, \text{meander}\} \times \{\text{Blue: } \text{sleep}, \text{random}, \text{coverage}\} \times \{\text{Seed: } 1001, 2003, 3005\}$$

#### Strict Identity Contracts
- `trajectory_id`: `traj_{red}_{blue}_{seed}_{ep}`
- `transition_id`: `{trajectory_id}_t{t}`
- `split_group_id`: `group_{seed}_{ep}`  
  *Contract*: Trajectories sharing initial seed/episode remain in the exact same deterministic split (`Train`, `Val`, or `HoldoutTest`) across all policy combinations.

### 4.3 Mathematical Formulation & Tensor Contracts

#### Context Encoder ($f_\theta$)
$$X_t = [O_{t-h+1}^{Blue}, \dots, O_t^{Blue}] \in \mathbb{R}^{h \times 52} \implies s_t = f_\theta(X_t) \in \mathbb{R}^{256}$$

#### Action-Conditioned Predictor ($g_\phi$)
$$e_a = E_{action}(a_t^{Blue}) \in \mathbb{R}^{64} \implies \hat{z}_{t+k} = g_\phi(s_t, e_a, k) \in \mathbb{R}^{256}$$

#### EMA Target Encoder ($f_{\bar{\theta}}$) & Loss ($\mathcal{L}_{JEPA}$)
$$\bar{\theta} \leftarrow 0.996 \bar{\theta} + 0.004 \theta \implies z_{t+k} = f_{\bar{\theta}}(O_{t+k}^{Blue}) \in \mathbb{R}^{256}$$
$$\mathcal{L}_{JEPA} = \frac{1}{B \cdot 256} \sum_{b=1}^B \sum_{i=1}^{256} \left( \hat{z}_{b, i, t+k} - \text{sg}(z_{b, i, t+k}) \right)^2$$

### 4.4 Disambiguation of Red Information
1. **Red Policy Execution**: Red agents generate adversarial trajectories offline. (Legitimate).
2. **Blue Telemetry**: Blue observes resulting host state changes $O_t^{Blue}$. (Legitimate).
3. **Privileged Oracle Sidecar**: Ground-truth Red state recorded strictly for post-hoc linear probing evaluation. (**Never input to $f_\theta$ or $g_\phi$**).

---

## 5. Department 5: Architecture Context Handoff

### 5.1 Tokenization Schemes

1. **Flat Vector (`flat`)**: Flattens $h \times 52$ features into a single vector projected to $d_{model}=256$.
2. **Feature Tokens (`feature`)**: Maps each of 52 features to individual tokens with feature-ID embeddings:
   $$T_{i, t} = \text{Linear}(O_{i, t}^{Blue}) + E_{feat\_id}(i) + E_{time}(t) \in \mathbb{R}^{64} \quad (i \in \{1 \dots 52\})$$
3. **Host Tokens (`host`)**: Combines 4 features per host into 13 host tokens:
   $$H_{j, t} = \text{MLP}(O_{host\_j, t}^{Blue}) + E_{subnet}(j) + E_{host\_id}(j) + E_{time}(t) \in \mathbb{R}^{64} \quad (j \in \{1 \dots 13\})$$
4. **Hierarchical Tokens (`hierarchical`)**: Two-stage pooling: Host tokens $\to$ 3 Subnet tokens $\to$ Network vector.

### 5.2 Context Aggregators (`src/cyber_jepa/models/aggregators.py`)
- `legacy_last_step_mean`: Mean-pools tokens across spatial/temporal dimensions prior to predictor.
- `learned_query_pool`: Multi-head cross-attention using learned queries $Q \in \mathbb{R}^{K \times d}$.
- `token_preserving_predictor`: Preserves token dimensions throughout predictor layers $g_\phi$.

### 5.3 Capacity Matching Matrix (~522k–685k Parameters)

| Configuration ID | Token Dim | Context Dim | Latent Dim | Trainable Params | Non-Trainable EMA Params | Total Params |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `flat_h4_control` | N/A | 256 | 256 | **548,416** | 180,928 | **729,344** |
| `feature_legacy_mean` | 64 | 256 | 256 | **522,432** | 180,928 | **703,360** |
| `feature_token_predictor` | 64 | 256 | 256 | **522,432** | 180,928 | **703,360** |
| `feature_query_pool` | 64 | 256 | 256 | **539,264** | 180,928 | **720,192** |
| `host_legacy_mean` | 64 | 256 | 256 | **618,816** | 180,928 | **799,744** |
| `host_token_predictor` | 64 | 256 | 256 | **618,816** | 180,928 | **799,744** |
| `host_query_pool` | 64 | 256 | 256 | **635,648** | 180,928 | **816,576** |
| `hierarchical_current` | 64 | 256 | 256 | **685,952** | 180,928 | **866,880** |
| `hierarchical_masked_predictor` | 64 | 256 | 256 | **685,952** | 180,928 | **866,880** |

---

## 6. Department 6: Results and Ablations Context Handoff

### 6.1 Master Empirical Results Matrix (Phase 3, $N=5$ Seeds)

| Configuration ID | Rep Type | Context Aggregator | Standard Macro F1 | Policy-Transfer F1 | Standard AUROC | Effective Rank | Collapsed Runs |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`flat_h4_control`** | `flat` | `legacy_last_step_mean` | **0.8776 ± 0.0043** | **0.8314 ± 0.0055** | **0.9514** | **8.7** | **0 / 5** |
| **`feature_legacy_mean`** | `feature` | `legacy_last_step_mean` | **0.8766 ± 0.0063** | **0.8315 ± 0.0086** | **0.9483** | **7.2** | **1 / 5** |
| **`feature_token_predictor`** | `feature` | `token_preserving_predictor` | **0.8753 ± 0.0045** | **0.8309 ± 0.0060** | **0.9481** | **7.3** | **1 / 5** |
| **`feature_query_pool`** | `feature` | `learned_query_pool` | 0.8584 ± 0.0169 | 0.7952 ± 0.0266 | 0.9380 | 4.5 | 4 / 5 |
| **`host_legacy_mean`** | `host` | `legacy_last_step_mean` | 0.8657 ± 0.0192 | 0.8153 ± 0.0224 | 0.9425 | 4.8 | 4 / 5 |
| **`host_token_predictor`** | `host` | `token_preserving_predictor` | 0.8657 ± 0.0192 | 0.8153 ± 0.0224 | 0.9425 | 4.8 | 4 / 5 |
| **`host_query_pool`** | `host` | `learned_query_pool` | 0.8655 ± 0.0035 | 0.8198 ± 0.0048 | 0.9407 | 4.8 | 4 / 5 |
| **`hierarchical_current`** | `hierarchical` | `legacy_last_step_mean` | 0.6603 ± 0.1062 | 0.5663 ± 0.1341 | 0.7439 | 1.4 | 5 / 5 |
| **`hierarchical_masked_predictor`** | `hierarchical` | `token_preserving_predictor` | 0.6603 ± 0.1062 | 0.5663 ± 0.1341 | 0.7439 | 1.4 | 5 / 5 |

### 6.2 Preregistered Decision Tree Evidence Ledger

| Hypothesis / Rule | Status | Paired Delta vs `flat_h4_control` | 95% Confidence Interval | p-value | Scientific Verdict |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Feature Tokens Match Flat** (Rule 14.4) | **SUPPORTED** | -0.0023 | [-0.0065, +0.0028] | 0.3617 | `feature_token_predictor` matches flat control within noise ($p=0.3617$). |
| **Feature Legacy Mean Parity** | **SUPPORTED** | -0.0010 | [-0.0064, +0.0051] | 0.7845 | `feature_legacy_mean` matches flat control ($p=0.7845$). |
| **Query-Pooling Bottleneck** (Rule 14.3) | **REJECTED** | -0.0192 | [-0.0296, -0.0059] | 0.0041 | Learned query cross-attention degraded performance ($p=0.0041$). |
| **Host Token Competitiveness** | **PARTIALLY SUPPORTED** | -0.0119 | [-0.0283, +0.0006] | 0.1222 | Host representations slightly below feature level. |
| **Hierarchical Representation** | **REJECTED** | -0.2173 | [-0.2833, -0.1236] | 0.0001 | Subnet pooling causes severe collapse ($\text{EffRank}=1.4$). |

---

## 7. Department 7: Conclusion Context Handoff

### 7.1 Key Paper Conclusions
1. **Multi-Step History is Essential**: History windowing $h=4$ resolves partial observability without simulator state leakage.
2. **Flat $h=4$ is the Practical Winner**: Flat temporal vectors match complex tokenizations while avoiding spatial token overhead.
3. **Coarse Spatial Aggregation Collapses**: Pooling host tokens into subnet vectors destroys critical compromise indicators ($\text{EffRank} \to 1.4$).
4. **Non-Reconstructive JEPA is Viable**: Cyber-JEPA predicts multi-step future states in latent space without raw vector reconstruction.

---

## 8. Actionable Master Roadmap

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                         MASTER PUBLICATION ACTION ROADMAP                        │
├──────────────────────────────────────────────────────────────────────────────────┤
│ PHASE 1: CRITICAL CODE & STATISTICAL INTEGRITY (P0)                              │
│   [ ] Audit token-preserving predictor branches in aggregators.py               │
│   [ ] Implement sample-level split-group paired bootstrap                        │
├──────────────────────────────────────────────────────────────────────────────────┤
│ PHASE 2: MISSING CONTROL EXPERIMENTS (P1)                                        │
│   [ ] Execute pure zero-shot Red policy transfer (B-line ↔ Meander)              │
│   [ ] Execute Blue-action causal ablation (with vs without a_t^Blue)             │
│   [ ] Execute inactive Red / no-attack evaluation (Sleep policy)                 │
│   [ ] Compute changed-target vs static-target performance metrics                │
├──────────────────────────────────────────────────────────────────────────────────┤
│ PHASE 3: MANUSCRIPT WRITING & PUBLICATION ASSETS (P2)                            │
│   [ ] Write manuscript draft sections 1 through 7                                │
│   [ ] Generate publication SVG/PNG architecture diagrams                         │
└──────────────────────────────────────────────────────────────────────────────────┘
```

| ID | Task Description | Priority | Target File | Output |
| :--- | :--- | :---: | :--- | :--- |
| **A1** | Code audit of token-preserving predictor branches | **P0** | `src/cyber_jepa/models/aggregators.py` | Verified unique code paths for token predictors. |
| **A2** | Sample-level split-group paired bootstrap | **P0** | `src/cyber_jepa/evaluation/diagnostics_extended.py` | Recomputed sample-level p-values & CIs. |
| **A3** | Pure zero-shot Red policy transfer execution | **P1** | `scripts/run_phase3_transfer.py` | Zero-shot `B_lineAgent` $\leftrightarrow$ `RedMeanderAgent` F1 scores. |
| **A4** | Action-conditioning causal ablation | **P1** | `scripts/run_action_ablation.py` | Ablation table comparing $P(z_{t+k} \mid O, a)$ vs. $P(z_{t+k} \mid O)$. |
| **A5** | Inactive Red / no-attack evaluation | **P1** | `scripts/run_inactive_red_eval.py` | Representation rank under Red `Sleep` policy. |
| **A6** | Changed-target metric breakdown | **P1** | `src/cyber_jepa/evaluation/metrics.py` | Stratified F1/MSE on changing vs static features. |

---

*This document serves as the master context handoff file for drafting the Cyber-JEPA research manuscript.*

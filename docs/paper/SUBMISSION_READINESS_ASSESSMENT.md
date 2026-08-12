# Cyber-JEPA Project Master Assessment: Submission Readiness & Departmental Single Source of Truth

> **Current Submission Readiness: 48%**  
> **Research & Implementation Readiness: 65–70%**  
> **Target Scope**: Autonomous Cyber-Defence Latent World-Model & Representation Evaluation in CybORG 3.1 (`Scenario1b`).  
> **Primary Document Purpose**: Comprehensive, single-source-of-truth master document detailing current progress, empirical findings, architectural contracts, mathematical derivations, code-path integrity audits, literature positioning, statistical protocols, identified gaps, and the step-by-step roadmap to publication readiness.

---

## 1. Executive Summary & Readiness Matrix

The Cyber-JEPA research project implements a Joint-Embedding Predictive Architecture (JEPA) for autonomous defender world-modeling in the CybORG 3.1 environment under strict Blue partial observability constraints. While the core codebase, experimental execution pipeline, and empirical sweeps (Phase 1, Phase 2, and Phase 3) are technically advanced (**65–70% complete**), the submission manuscript itself remains at **48% readiness**.

The primary gap is the transition from **exploratory empirical findings** to a **statistically defensible, published scientific manuscript**. This document consolidates all project knowledge, metric tables, architectural formulas, decision-tree outcomes, literature positioning, identified gaps, and actionable remediation steps.

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                             PROJECT READINESS OVERVIEW                           │
├──────────────────────────────────────────────────────┬───────────────────────────┤
│ Implementation & Code Base Readiness                 │ ██████████████░░░░  68%   │
│ Manuscript & Scientific Defensibility Readiness      │ █████████░░░░░░░░░  48%   │
│ Scope-Adjusted (with Policy Control)                 │ ███████░░░░░░░░░░░  37%   │
└──────────────────────────────────────────────────────┴───────────────────────────┘
```

### Departmental Readiness Breakdown

| Paper Department | Completion | Status & Primary Assessment | Primary Blocker / Next Step |
| :--- | :---: | :--- | :--- |
| **1. Introduction** | **35%** | Problem motivation clear; hypotheses & formal contribution lists unwritten. | Draft polished problem statement, 4 contribution points, and scope boundaries. |
| **2. Related Work** | **30%** | Key JEPA papers identified; lacks cyber-RL defender positioning & BibTeX audit. | Synthesize CybORG RL baseline comparisons and construct verified BibTeX database. |
| **3. Literature Survey** | **45%** | ~2.7k words of exploratory notes exist; lacks thematic scholarly organization. | Structure into 7 thematic sub-domains with a formal comparison matrix. |
| **4. Methodology** | **68%** | Data taxonomy, split-group isolation, and contracts mature; split across docs. | Consolidate math formulations, tensor shapes, and Red-policy transfer definitions. |
| **5. Architecture** | **72%** | Encoder/predictor/EMA modules fully functional; integrity issues in token paths. | Audit token-preserving code paths; generate vector SVG/Mermaid architecture diagrams. |
| **6. Results & Ablations** | **58%** | 20-run (Phase 2) & 45-run (Phase 3) sweeps complete; statistics operate on seed aggregates. | Re-run paired bootstrap on sample-level split groups; complete B-line $\leftrightarrow$ Meander transfer. |
| **7. Conclusion** | **25%** | Core empirical findings established; relies on final ablation & control freezes. | Write final conclusions, explicit scope boundaries, and honest failure mode analysis. |

---

## 2. Department 1: Introduction (Completion: 35%)

### 2.1 Established Foundations
1. **Problem Definition**: Learning a defender-oriented, latent world-model in complex, partially observable network environments (CybORG 3.1 / CAGE Challenge 2).
2. **Information Boundary**: Enforcing zero simulator-state leakage by restricting input strictly to Blue's legitimate observation history $O_{t-h:t}^{Blue} \in \mathbb{R}^{h \times 52}$.
3. **Core Motivation**: Utilizing Joint-Embedding Predictive Architectures (JEPAs) to capture temporal cyber-state evolution without reconstructive pixel/vector loss (which collapses under sparse cyber state changes).
4. **Empirical Question**: Determining whether flat observation vectors or structured topological entities (feature, host, hierarchical subnets) form superior latent representations under action-conditioned multi-step prediction.
5. **Action Conditioning**: Conditioning the predictor on Blue defensive actions $a_t^{Blue}$ to enable hypothetical outcome simulation for downstream planning.

### 2.2 Critical Gaps & Unwritten Elements
- **Polished Problem Statement**: Needs formal articulation of why partial observability and high feature staticity ($>90\%$ unchanged step-to-step) break standard world models.
- **Explicit Research Hypotheses**: Formally stating $H_1$ (temporal history prevents representation collapse), $H_2$ (action conditioning improves future representation fidelity), and $H_3$ (coarse spatial pooling induces information bottlenecks).
- **Concise Contribution List**: Drafting the 4 core paper contributions.
- **Critique of Existing CybORG Defenders**: Explaining why standard model-free RL (PPO, SAC) and static observations fail to maintain state estimates under stealthy Red lateral movement.
- **Explicit Scope Boundary**: Clarifying that this paper evaluates **latent world-model representation quality and predictive accuracy**, NOT an end-to-end policy controller (which would lower total completion to ~35–40%).

### 2.3 Publication Contribution Structure

```
                             PAPERS FOUR CORE CONTRIBUTIONS
                                          │
    ┌─────────────────────────┬───────────┴─────────────┬─────────────────────────┐
    ▼                         ▼                         ▼                         ▼
1. Action-Conditioned   2. Controlled Token       3. Failure Mode Analysis  4. Rigorous Evaluation
   Cyber-JEPA              Topology Comparison       of Hierarchical Pooling    Framework
   Formulation for CybORG  (Flat vs Host vs Subnet)  & Representation Collapse  (Split Groups, Holdouts,
   Blue Observations                                                            Persistence Baselines)
```

1. **Defender-Observable Action-Conditioned Cyber-JEPA**: The first adaptation of JEPA to autonomous cyber defense operating strictly within the $O_{t-h:t}^{Blue}$ observation boundary.
2. **Controlled Topological Representation Sweep**: A parameter-matched, single-frame target ($T=1$) empirical comparison of flat, feature-token, host-token, and hierarchical subnet-token architectures.
3. **Empirical Identification of Aggregation Collapse**: Rigorous demonstration that coarse spatial pooling (subnet-level hierarchical aggregation) destroys fine-grained compromise indicators, resulting in severe effective-rank collapse ($\text{EffRank} = 1.4$).
4. **Principled Cybersecurity Evaluation Protocol**: An evaluation benchmark incorporating deterministic `split_group_id` isolation, Red-policy transfer regimes (`B_lineAgent` vs `RedMeanderAgent`), persistence baselines, and frozen-encoder linear probing.

---

## 3. Department 2: Related Work (Completion: 30%)

### 3.1 Primary Literature & Source Assets
Primary design discussions reside in [Initial_testing_methods.md](file:///c:/Users/rohil/Documents/GenAI%20Micro%20Project/T-JEPA-test/docs/Initial_testing_methods.md). Key frameworks under positioning include:

- **I-JEPA** (Assran et al., 2023): Joint-embedding predictive architecture for images using spatial masking and EMA target encoders.
- **T-JEPA** (Boardman et al., 2024): Tabular JEPA introducing feature-token embeddings and masking strategies for tabular datasets.
- **V-JEPA & V-JEPA 2 / 2-AC** (Bardes et al., 2024): Spatiotemporal video world models and action-conditioned predictive world models.
- **A-JEPA** (Feichtenhofer et al., 2023): Audio JEPA using time-frequency spectrogram masking.
- **Graph & Hierarchical JEPAs**: NodeJEPA and HP-JEPA (Hierarchical Partition JEPA) for relational graph dynamics.
- **CybORG / CAGE Benchmark**: Standish et al. (2021), Foley et al. (2022) defining Cyber Autonomy Gym for Experimentation.

### 3.2 Required Enhancements & Unwritten Positioning
- **Model-Free CybORG Defenders**: Positioning against PPO, A3C, and SAC implementations in CAGE Challenges 1–4.
- **Model-Based RL in Cyber Security**: Comparing Cyber-JEPA against reconstructive world models (World Models, DreamerV1-V3) that fail in cyber due to raw observation reconstruction overhead and state sparsity.
- **Predictive State & Recurrent Models**: Positioning against standard LSTM/GRU state trackers and predictive contrastive models.
- **BibTeX Verification & Clean-up**: Removing unverified preprints and establishing primary paper citations.

### 3.3 Core Research Gap Statement

> **Core Research Gap**: *Existing JEPA literature focuses on semantic prediction in visual, audio, tabular, and graph domains characterized by dense spatial or temporal continuity. In contrast, autonomous cyber-defence requires action-conditioned temporal prediction over discrete, highly static ($>90\%$ static steps), and partially observable observation spaces under strict operational boundaries. Cyber-JEPA addresses this gap by formalizing action-conditioned latent prediction for cyber defender observation streams.*

### 3.4 Methodological Positioning Matrix

| Method / Architecture | Data Modality | Input Boundary | Action Conditioned? | Target Space | Collapse Prevention Strategy |
| :--- | :--- | :--- | :---: | :--- | :--- |
| **I-JEPA** (Assran 2023) | 2D Images | Full Image Patches | No | Latent ($z_{target}$) | EMA Target Encoder + Spatial Masking |
| **V-JEPA 2-AC** (Bardes 2024) | Video Frames | 3D Spatiotemporal Patches | Yes | Latent ($z_{target}$) | EMA Target Encoder + Spatiotemporal Masking |
| **T-JEPA** (Boardman 2024) | Tabular Data | Unstructured Feature Vector | No | Latent ($z_{target}$) | Feature-Token Masking + Variance Regularization |
| **DreamerV3** (Hafner 2023) | Dense RL | Full Simulator State / Image | Yes | Reconstructive ($\hat{x}_{t+k}$) | KL-Divergence Regularization |
| **Cyber-JEPA** (Ours) | Cyber Trajectories | $O_{t-h:t}^{Blue}$ (Partially Observable) | **Yes ($a_t^{Blue}$)** | **Latent ($z_{target}$)** | **EMA Encoder + Action Tokens + Split Group Isolation** |

---

## 4. Department 3: Literature Survey (Completion: 45%)

The repository contains ~2,745 words of literature and design notes in [Initial_testing_methods.md](file:///c:/Users/rohil/Documents/GenAI%20Micro%20Project/T-JEPA-test/docs/Initial_testing_methods.md). To transform this into a scholarly literature survey, the content must be organized into seven core thematic pillars.

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                        SEVEN PILLARS OF THE LITERATURE SURVEY                    │
├───────────────────┬───────────────────┬───────────────────┬──────────────────────┤
│ 1. JEPA           │ 2. Cross-Modal    │ 3. Latent World   │ 4. Autonomous Cyber  │
│    Foundations &  │    JEPA           │    Models & Action│    Defence           │
│    Anti-Collapse  │    Adaptations    │    Conditioning   │    Environments      │
├───────────────────┴───────────────────┴───────────────────┴──────────────────────┤
│ 5. CybORG & CAGE Defender Architectures                                          │
│ 6. Representation Learning Under Severe Partial Observability                    │
│ 7. Identified Research Gaps & Resulting Hypotheses                               │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### 4.1 Reorganization into Seven Thematic Pillars

1. **JEPA Foundations & Anti-Collapse Mechanisms**:
   - Evolution from contrastive learning (SimCLR, MoCo) and non-contrastive methods (BYOL, SimSiam, VICReg) to joint-embedding predictive architectures.
   - Analysis of collapse modes: complete collapse ($z_t = \text{const}$) vs dimensional/rank collapse ($\text{EffRank} \to 1$).
   - Role of Exponential Moving Average (EMA) target encoders ($f_{\bar{\theta}}$), stop-gradient operators, and predictor architectures ($g_\phi$).

2. **Cross-Modal JEPA Adaptations**:
   - Visual (I-JEPA, V-JEPA), Audio (A-JEPA), Graph (NodeJEPA, HP-JEPA), and Tabular (T-JEPA).
   - Domain-specific tokenization: How spatial patches, spectrogram bins, graph nodes, and tabular columns translate into sequence tokens $E \in \mathbb{R}^{N \times d}$.

3. **Latent World Models & Action Conditioning**:
   - Transition from reconstructive world models (World Models, DreamerV1-V3) to latent predictive models.
   - Action-conditioned transition functions $z_{t+k} = g_\phi(z_t, a_t)$. Handling passive state drift versus action-driven state perturbations.

4. **Autonomous Cyber-Defence Environments**:
   - Survey of simulation environments: CybORG, FARLAND, CyberVAN, PRIMA, NASim.
   - Standardized benchmarks: CAGE Challenges 1, 2, 3, and 4 (`Scenario1b`, `Scenario2`).

5. **CybORG & CAGE Defender Architectures**:
   - Prior defender implementations: Heuristic agents, model-free DRL (PPO, Rainbow DQN), hierarchical RL.
   - Limitations of existing defenders: Vulnerability to unseen Red tactics, lack of internal state estimation, inability to predict multi-step attack escalation.

6. **Representation Learning Under Severe Partial Observability**:
   - Partially Observable Markov Decision Processes (POMDPs) in cybersecurity.
   - History windowing ($O_{t-h:t}$) versus recurrent memory units (LSTMs, Transformers) for state estimation under incomplete telemetry.

7. **Synthesized Research Gap & Hypotheses**:
   - Formal synthesis establishing why existing JEPA and RL models fail in CybORG, leading directly to the experimental design of Cyber-JEPA.

---

## 5. Department 4: Methodology (Completion: 68%)

### 5.1 Environment Specification & CybORG 3.1 Topology

Evaluations operate on **CybORG 3.1** `Scenario1b` (CAGE Challenge 2), comprising a 13-host enterprise network split across 3 subnets:

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

#### Observation Encoding Details (52 Features)
Each of the 13 hosts is represented by 4 discrete/discretized categorical fields in `BlueTableWrapper` / `ChallengeWrapper`:
$$\text{Host}_i = [\text{Subnet\_ID}_i, \text{IP\_Address}_i, \text{Activity\_State}_i, \text{Compromised\_State}_i]$$
- **Subnet ID**: $\{0, 1, 2\}$
- **IP Address**: Discretized host index within subnet $\{0 \dots 4\}$
- **Activity State**: $\{0: \text{None}, 1: \text{Scanning}, 2: \text{Exploitation/PrivEsc}\}$
- **Compromised State**: $\{0: \text{Unknown/None}, 1: \text{Compromised}, 2: \text{User Access}, 3: \text{System Access}\}$

#### Discrete Action Space (66 Actions)
Blue defensive actions $a_t^{Blue} \in \{0 \dots 65\}$ map to abstract action types parameterized by host target:
- **`Sleep`**: Index `0`
- **`Monitor(host)`**: Indices `1` to `13`
- **`Analyse(host)`**: Indices `14` to `26`
- **`Remove(host)`**: Indices `27` to `39`
- **`Misinform(host)`**: Indices `40` to `52`
- **`Restore(host)`**: Indices `53` to `65`

### 5.2 Offline Trajectory Dataset & Deterministic Identity Contracts

The dataset consists of **18 shards** (1,800 trajectories, 90,000 transitions) generated via:
$$\text{Dataset} = \{\text{Red: } \text{bline}, \text{meander}\} \times \{\text{Blue: } \text{sleep}, \text{random}, \text{coverage}\} \times \{\text{Seed: } 1001, 2003, 3005\}$$

#### Strict Identity Contracts
To prevent split leakage and guarantee deterministic evaluation pairing:
- **`trajectory_id`**: `traj_{red}_{blue}_{seed}_{ep}`
- **`transition_id`**: `{trajectory_id}_t{t}`
- **`split_group_id`**: `group_{seed}_{ep}`  
  *Contract*: All trajectories originating from the same environment random seed and episode index belong to the exact same `split_group_id` and are assigned atomically to either `Train`, `Val`, or `HoldoutTest`.

### 5.3 Formal Input, Target, and Loss Formulations

#### Context Encoder ($f_\theta$)
Given observation history $X_t = [O_{t-h+1}^{Blue}, \dots, O_t^{Blue}] \in \mathbb{R}^{h \times 52}$:
$$s_t = f_\theta(X_t) \in \mathbb{R}^{d_{context}}$$

#### Action Conditioning & Predictor ($g_\phi$)
Given Blue action $a_t^{Blue} \in \{0 \dots 65\}$ and horizon step $k \in \{1 \dots 16\}$:
$$e_a = E_{action}(a_t^{Blue}) \in \mathbb{R}^{d_{action}}$$
$$\hat{z}_{t+k} = g_\phi(s_t, e_a, k) \in \mathbb{R}^{d_{latent}}$$

#### Target Encoder ($f_{\bar{\theta}}$) & Single-Frame Loss ($T=1$)
The target is computed by the EMA target encoder $f_{\bar{\theta}}$ ($\bar{\theta} \leftarrow \tau \bar{\theta} + (1-\tau)\theta$, $\tau=0.996$):
$$z_{t+k} = f_{\bar{\theta}}(O_{t+k}^{Blue}) \in \mathbb{R}^{d_{latent}}$$
$$\mathcal{L}_{JEPA} = \frac{1}{B \cdot d_{latent}} \sum_{b=1}^B \sum_{i=1}^{d_{latent}} \left( \hat{z}_{b, i, t+k} - \text{sg}(z_{b, i, t+k}) \right)^2$$

```
                           CYBER-JEPA TENSOR FLOW ARCHITECTURE
                           
  Observation History                                              EMA Target Encoder
  [O_{t-3}, ..., O_t]                                                  O_{t+k}
     (4 x 52)                                                          (1 x 52)
        │                                                                 │
        ▼                                                                 ▼
  Context Encoder f_θ                                           Target Encoder f_θ̄ (EMA)
        │                                                                 │
        ▼                                                                 ▼
   Context s_t                                                     Target z_{t+k}
   (d_context)                                                       (d_latent)
        │                                                                 │
        ├──────────────────────┐                                          │
        ▼                      ▼                                          │
  Blue Action a_t      Predictor g_φ                                      │
   (d_action)                  │                                          │
        │                      ▼                                          │
        └──────────────► Latent Pred ẑ_{t+k}                              │
                           (d_latent)                                     │
                               │                                          │
                               └──────────────► MSE Loss ◄────────────────┘
                                            ||ẑ_{t+k} - sg(z_{t+k})||²
```

### 5.4 Primary & Diagnostic Metric Definitions

#### 1. Linear Probe Macro F1 & AUROC
A frozen linear probe $h_\omega: z_t \to \{0, 1\}^{13}$ is trained on $z_t$ to predict host compromise state across all 13 hosts. Macro F1 and AUROC are calculated on holdout test split:
$$\text{Macro F1} = \frac{1}{13} \sum_{i=1}^{13} \text{F1}_i$$

#### 2. Effective Rank ($\text{EffRank}$)
Measures dimensional collapse of representation matrix $Z \in \mathbb{R}^{N \times d_{latent}}$. Given singular values $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_d$:
$$p_i = \frac{\sigma_i}{\sum_{j=1}^d \sigma_j}, \quad H(p) = -\sum_{i=1}^d p_i \ln p_i, \quad \text{EffRank}(Z) = \exp(H(p))$$
- **Healthy Representation**: $\text{EffRank} \ge 6.0$
- **Dimensional Collapse**: $\text{EffRank} < 2.0$

#### 3. Persistence Baseline MSE
Evaluates static identity prediction:
$$\mathcal{L}_{pers}(k) = \frac{1}{B \cdot 52} \sum_{b=1}^B \| O_{t+k, b}^{Blue} - O_{t, b}^{Blue} \|_2^2$$

---

## 6. Department 5: Architecture (Completion: 72%)

### 6.1 Code Package Structure (`src/cyber_jepa/`)
```
src/cyber_jepa/
├── data/
│   ├── characterize.py    # Dataset characterization & staticity metrics
│   ├── collector.py       # CybORG trajectory collector
│   ├── dataset.py         # PyTorch Dataset wrappers & split group handlers
│   ├── schema.py          # Data taxonomy & typing contracts
│   └── storage.py         # Parquet shard I/O & integrity checkers
├── env/
│   ├── action_mapper.py   # Discrete action 0-65 to CybORG action mapping
│   └── observation_multiplexer.py # ChallengeWrapper 52D vector extractor
├── evaluation/
│   ├── diagnostics.py     # Effective rank & collapse detectors
│   ├── diagnostics_extended.py # Paired bootstrap analysis scripts
│   ├── metrics.py         # F1, AUROC, MSE calculation functions
│   ├── probes.py          # Linear probe trainer & evaluator
│   └── selection.py       # Preregistered decision tree evaluator
├── models/
│   ├── aggregators.py     # LegacyMean, QueryPool, TokenPreservingPredictor
│   ├── baselines.py       # Persistence & Random baselines
│   ├── context.py         # TemporalContextEncoder
│   ├── interface.py       # Model abstract base classes
│   ├── jepa.py            # CyberJEPA main wrapper & EMA logic
│   └── predictor.py       # ActionConditionedPredictor
├── representations/
│   ├── flat.py            # Flat vector representation module
│   ├── feature.py         # Feature-token representation module
│   ├── host.py            # Host-centric token representation module
│   └── hierarchical.py    # Subnet-level hierarchical representation module
├── training/
│   ├── orchestrator.py    # Multi-run sweep orchestrator
│   └── trainer.py         # Single-run PyTorch training loop
└── utils/
    └── reproducibility.py # Global seed setting & deterministic PyTorch flags
```

### 6.2 Model Capacity & Parameter Accounting Table

To ensure fair comparisons in Phase 3, representation architectures were capacity-matched (~522k–685k trainable parameters):

| Configuration ID | Token Dimension ($d_{token}$) | Context Dim ($d_{ctx}$) | Latent Dim ($d_{lat}$) | Trainable Params | Non-Trainable EMA Params | Total Params |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `flat_h4_control` | N/A | 256 | 256 | 548,416 | 180,928 | 729,344 |
| `feature_legacy_mean` | 64 | 256 | 256 | 522,432 | 180,928 | 703,360 |
| `feature_token_predictor` | 64 | 256 | 256 | 522,432 | 180,928 | 703,360 |
| `feature_query_pool` | 64 | 256 | 256 | 539,264 | 180,928 | 720,192 |
| `host_legacy_mean` | 64 | 256 | 256 | 618,816 | 180,928 | 799,744 |
| `host_token_predictor` | 64 | 256 | 256 | 618,816 | 180,928 | 799,744 |
| `host_query_pool` | 64 | 256 | 256 | 635,648 | 180,928 | 816,576 |
| `hierarchical_current` | 64 | 256 | 256 | 685,952 | 180,928 | 866,880 |
| `hierarchical_masked_predictor` | 64 | 256 | 256 | 685,952 | 180,928 | 866,880 |

### 6.3 Critical Code-Path Integrity Audit & Identified Bypasses

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                    CRITICAL ARCHITECTURAL INTEGRITY AUDIT                        │
├──────────────────────────────────────────────────────────────────────────────────┤
│ ISSUE 1: Host Token Path Identity                                               │
│   `host_legacy_mean` F1 = 0.8657  ≡  `host_token_predictor` F1 = 0.8657         │
│   Diagnosis: High probability that `host_token_predictor` fell back to mean     │
│              pooling in execution due to structural branch bypass.               │
├──────────────────────────────────────────────────────────────────────────────────┤
│ ISSUE 2: Hierarchical Path Identity                                             │
│   `hierarchical_current` F1 = 0.6603  ≡  `hierarchical_masked_predictor` F1 = 0.6603│
│   Diagnosis: `hierarchical_masked_predictor` bypassed token-preserving path,     │
│              executing identical code path as baseline hierarchical mean pool.   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

#### Affected Files & Inspection Mandates
- **[src/cyber_jepa/models/aggregators.py](file:///c:/Users/rohil/Documents/GenAI%20Micro%20Project/T-JEPA-test/src/cyber_jepa/models/aggregators.py)**: Audit `TokenPreservingPredictor` class implementation to ensure it does not invoke `torch.mean()` prior to transformer layer entry.
- **[src/cyber_jepa/models/jepa.py](file:///c:/Users/rohil/Documents/GenAI%20Micro%20Project/T-JEPA-test/src/cyber_jepa/models/jepa.py)**: Audit `CyberJEPA.forward()` method to verify that `aggregator_mode` flags correctly route token tensors.

---

## 7. Department 6: Results and Ablations (Completion: 58%)

### 7.1 Master Results Matrix (Phase 2 & Phase 3 Combined)

#### Phase 2 Horizon & Representation Sweep ($N=1$ Seed, 20 Runs)
Phase 2 established that 4-step temporal history ($h=4$) provided optimal performance. `flat_k4` achieved top performance (OOD F1 = 0.6722, AUROC = 0.8167, EffRank = 4.7).

#### Phase 3 Matrix (Capacity-Matched, $N=5$ Seeds, 45 Runs Completed)
Phase 3 enforced parameter matching (~522k–685k), 52-feature input exactness, and single-frame target encoding ($T=1$).

| Configuration ID | Rep Type | Context Aggregator Mode | Standard Macro F1 | Policy-Transfer F1 | Standard AUROC | Effective Rank | Collapsed Runs | Trainable Parameters |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`flat_h4_control`** | `flat` | `legacy_last_step_mean` | **0.8776 ± 0.0043** | **0.8314 ± 0.0055** | **0.9514** | **8.7** | **0 / 5** | **548,416** |
| **`feature_legacy_mean`** | `feature` | `legacy_last_step_mean` | **0.8766 ± 0.0063** | **0.8315 ± 0.0086** | **0.9483** | **7.2** | **1 / 5** | **522,432** |
| **`feature_token_predictor`** | `feature` | `token_preserving_predictor` | **0.8753 ± 0.0045** | **0.8309 ± 0.0060** | **0.9481** | **7.3** | **1 / 5** | **522,432** |
| **`feature_query_pool`** | `feature` | `learned_query_pool` | 0.8584 ± 0.0169 | 0.7952 ± 0.0266 | 0.9380 | 4.5 | 4 / 5 | 539,264 |
| **`host_legacy_mean`** | `host` | `legacy_last_step_mean` | 0.8657 ± 0.0192 | 0.8153 ± 0.0224 | 0.9425 | 4.8 | 4 / 5 | 618,816 |
| **`host_token_predictor`** | `host` | `token_preserving_predictor` | 0.8657 ± 0.0192 | 0.8153 ± 0.0224 | 0.9425 | 4.8 | 4 / 5 | 618,816 |
| **`host_query_pool`** | `host` | `learned_query_pool` | 0.8655 ± 0.0035 | 0.8198 ± 0.0048 | 0.9407 | 4.8 | 4 / 5 | 635,648 |
| **`hierarchical_current`** | `hierarchical` | `legacy_last_step_mean` | 0.6603 ± 0.1062 | 0.5663 ± 0.1341 | 0.7439 | 1.4 | 5 / 5 | 685,952 |
| **`hierarchical_masked_predictor`** | `hierarchical` | `token_preserving_predictor` | 0.6603 ± 0.1062 | 0.5663 ± 0.1341 | 0.7439 | 1.4 | 5 / 5 | 685,952 |

### 7.2 Preregistered Decision Tree Evidence Ledger

| Hypothesis / Claim | Status | Paired Delta vs `flat_h4_control` | 95% Confidence Interval | p-value | Primary Scientific Conclusion |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Feature Tokens Match Flat** (Decision Rule 14.4) | **SUPPORTED (Parity)** | -0.0023 | [-0.0065, +0.0028] | 0.3617 | `feature_token_predictor` (0.8753) matches `flat_h4_control` (0.8776) within statistical noise ($p=0.3617$). |
| **Feature Legacy Mean Competitiveness** | **SUPPORTED (Parity)** | -0.0010 | [-0.0064, +0.0051] | 0.7845 | `feature_legacy_mean` (0.8766) achieves statistical parity with flat control ($p=0.7845$). |
| **Mean-Pooling Bottleneck Hypothesis** (Decision Rule 14.3) | **REJECTED** | -0.0192 | [-0.0296, -0.0059] | 0.0041 | Learned query cross-attention degraded performance (F1=0.8584) and increased collapse (4/5 runs collapsed). |
| **Host Token Competitiveness** | **PARTIALLY SUPPORTED** | -0.0119 | [-0.0283, +0.0006] | 0.1222 | Host representations remain competitive (0.8657), though slightly below feature-level representations. |
| **Hierarchical Representation** | **REJECTED (Rank Collapsed)** | -0.2173 | [-0.2833, -0.1236] | 0.0001 | Two-stage subnet pooling causes severe representation collapse ($\text{EffRank}=1.4$, 5/5 runs collapsed). |

### 7.3 Core Empirical Findings
1. **Input Exactness & Single-Frame Target Correction**: Restoring 52-feature input exactness and correcting target encoding to single-frame $T=1$ brought feature-structured representations to statistical parity with flat temporal representations ($p=0.3617$).
2. **Rejection of Learned Query Pooling**: Cross-attention query pooling induced rank collapse ($\text{EffRank} = 4.5$) and significantly degraded performance ($p=0.0041$). Simple mean pooling is superior.
3. **Hierarchical Topology Bottlenecking**: Coarse spatial pooling at the subnet level destroys fine-grained compromise indicators, causing 100% representation collapse ($\text{EffRank} = 1.4$).
4. **Optimal Practical Input**: Flat 4-step temporal history ($h=4$) provides the top performance-to-complexity ratio (F1 = 0.8776, zero collapsed runs, highest effective rank 8.7).

### 7.4 Outstanding Experimental Deficiencies
- **Sample-Level Paired Statistics**: Current p-values bootstrap seed aggregates ($N=5$). Must be computed on sample-level evaluation predictions paired by `split_group_id`.
- **Pure Zero-Shot Red Policy Transfer**: Must evaluate models trained exclusively on `bline` against `meander` evaluation trajectories (and vice versa).
- **Causal Blue Action Ablation**: Train models with $a_t^{Blue}$ zeroed out to measure exact performance gain attributed to action-conditioning.
- **Changed-Target Metric Emphasis**: Calculate F1/MSE specifically on features that change state between $t$ and $t+k$ to verify the model avoids trivial static-state identity prediction.

---

## 8. Department 7: Conclusion (Completion: 25%)

### 8.1 Preliminary Scientific Conclusions
1. **Temporal History is Vital**: Multi-step observation history ($h=4$) is essential for state estimation under partial observability.
2. **Flat $h=4$ is the Practical Winner**: Simple flat temporal vectors match or outperform complex entity tokenizations while eliminating architectural complexity.
3. **Feature-Token Parity**: Feature-level tokenization matches flat performance but fails to justify its added computational overhead.
4. **Coarse Aggregation Destroys Information**: Subnet-level pooling causes severe representation collapse, proving that cyber world models require fine-grained feature access.
5. **Action-Conditioned Latent World Models are Viable**: Cyber-JEPA successfully predicts multi-step future latent states without raw observation reconstruction.

### 8.2 Scope Limitations & Honest Failure Modes (Must Include in Paper)
- **Latent Space vs Policy Control**: Cyber-JEPA is evaluated as a latent world-model state estimator, not an active defender policy controller.
- **Synthetic Simulator Dynamics**: CybORG 3.1 `Scenario1b` represents abstraction of real-world networks; transfer to real telemetry (e.g., Sysflow, Zeek logs) remains unverified.
- **Fixed Network Topology**: Evaluations operate on fixed 13-host network graphs; dynamic host joining/leaving requires extended graph tokenizers.

---

## 9. Master Action Plan & Remediation Roadmap

To bring submission readiness from **48% to 100%**, the following prioritized action items must be executed:

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                         MASTER PUBLICATION REMEDIATION ROADMAP                   │
├──────────────────────────────────────────────────────────────────────────────────┤
│ PHASE 1: CRITICAL CODE & STATISTICAL INTEGRITY (P0)                              │
│   [ ] Audit & fix token-preserving path identity bugs in aggregators.py          │
│   [ ] Upgrade statistical protocol to sample-level split-group paired bootstrap  │
│   [ ] Retain per-sample prediction artifacts during evaluation                   │
├──────────────────────────────────────────────────────────────────────────────────┤
│ PHASE 2: MISSING CONTROL EXPERIMENTS (P1)                                        │
│   [ ] Execute pure zero-shot Red policy transfer (B-line ↔ Meander)              │
│   [ ] Execute Blue-action causal ablation (with vs without a_t^Blue)             │
│   [ ] Execute inactive Red / no-attack evaluation (Sleep policy)                 │
│   [ ] Compute changed-target vs static-target performance metrics                │
├──────────────────────────────────────────────────────────────────────────────────┤
│ PHASE 3: MANUSCRIPT WRITING & PUBLICATION ASSETS (P2)                            │
│   [ ] Draft Department 1 (Introduction) with 4 explicit contribution points       │
│   [ ] Complete Department 2 (Related Work) with BibTeX audit & delta matrix      │
│   [ ] Consolidate Department 3 (Literature Survey) into 7 thematic pillars        │
│   [ ] Unify Department 4 & 5 (Methodology & Architecture) equations & math      │
│   [ ] Generate vector SVG / Mermaid publication-quality architecture diagrams    │
│   [ ] Write Department 7 (Conclusion & Limitations)                              │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### Action Item Matrix

| ID | Task Description | Priority | Affected File / Target | Target Completion Output |
| :--- | :--- | :---: | :--- | :--- |
| **A1** | Code audit of token-preserving predictor branches in host & hierarchical aggregators | **P0** | `src/cyber_jepa/models/aggregators.py` | Verified unique code paths for `host_token_predictor` and `hierarchical_masked_predictor`. |
| **A2** | Sample-level split-group paired bootstrap implementation | **P0** | `src/cyber_jepa/evaluation/diagnostics_extended.py` | Corrected 95% CIs and p-values operating on paired evaluation samples. |
| **A3** | Pure zero-shot Red policy transfer execution | **P1** | `scripts/run_phase3_transfer.py` | Pure B-line $\to$ Meander and Meander $\to$ B-line F1 transfer scores. |
| **A4** | Action-conditioning causal ablation | **P1** | `scripts/run_action_ablation.py` | Performance comparison table of $P(z_{t+k} \mid O, a)$ vs $P(z_{t+k} \mid O)$. |
| **A5** | Inactive Red / no-attack baseline evaluation | **P1** | `scripts/run_inactive_red_eval.py` | Representation rank and drift metrics under Red `Sleep` policy. |
| **A6** | Changed-target metric breakdown | **P1** | `src/cyber_jepa/evaluation/metrics.py` | Stratified F1/MSE on changing vs static feature subsets across horizon $k$. |
| **A7** | Draft Introduction & Contributions | **P2** | `docs/paper/01_introduction.md` | Complete Introduction section with problem statement and 4 contributions. |
| **A8** | Related Work BibTeX & Positioning Matrix | **P2** | `docs/paper/02_related_work.md` | Verified BibTeX database and comparative positioning table. |
| **A9** | Generate Publication Architecture Figures | **P2** | `reports/figures/architecture_diagram.svg` | High-resolution SVG/PNG diagrams of Cyber-JEPA tensor pipeline. |

---

*This document serves as the master single source of truth for the Cyber-JEPA submission readiness assessment, empirical ledger, architectural contracts, and publication roadmap.*

# 02 — Architecture & Codebase Deep Dive

> **Document Purpose**: Exhaustive engineering guide to the Cyber-JEPA codebase, module abstractions, tensor shapes, and component pipelines for developers and research engineers.

---

## 1. System Architecture & End-to-End Data Pipeline

```
                                CYBER-JEPA DATA & TENSOR FLOW
                                
   +-----------------------------------------------------------------------------------+
   | 1. DATA COLLECTION & STORAGE (scripts/run_collection.py, src/cyber_jepa/data/)   |
   |                                                                                   |
   |   CybORG 3.1 Gym -> ObservationMultiplexer -> Raw Shards (JSON Lines / Parquet)   |
   |   1,800 Trajectories / 90,000 Transitions / 300 Split Groups                      |
   +-----------------------------------------------------------------------------------+
                                            |
                                            v
   +-----------------------------------------------------------------------------------+
   | 2. DATASET & COHORT LOADER (src/cyber_jepa/data/dataset.py)                       |
   |                                                                                   |
   |   Batch: {                                                                        |
   |     'context_obs': [B, h=4, 52]        (Deployable History Window)                |
   |     'action':      [B]                  (Blue Action Index 0..65)                 |
   |     'target_obs':  [B, 52]              (Single-Frame Target T=1)                 |
   |     'oracle_label':[B]                  (Privileged Op_Server0 Compromise)        |
   |   }                                                                               |
   +-----------------------------------------------------------------------------------+
                                            |
                                            v
   +-----------------------------------------------------------------------------------+
   | 3. REPRESENTATION & CONTEXT ENCODING (src/cyber_jepa/representations/)           |
   |                                                                                   |
   |   Input: [B, 4, 52]                                                               |
   |   - Flat:         [B, 4*52] -> Linear -> [B, 256]                                 |
   |   - FeatureToken: [B, 4, 52, 1] -> Linear -> [B, 4, 52, D] -> Context -> [B, N, D]|
   |   - HostToken:    [B, 4, 13, 4] -> Linear -> [B, 4, 13, D] -> Context -> [B, 13, D]|
   |   - Hierarchical: [B, 4, 13, 4] -> Host MLP -> Subnet Pool -> Context -> [B, 3, D] |
   +-----------------------------------------------------------------------------------+
                                            |
                                            v
   +-----------------------------------------------------------------------------------+
   | 4. CONTEXT AGGREGATION & LATENT PREDICTION (src/cyber_jepa/models/)               |
   |                                                                                   |
   |   Context Latent s_t [B, N, D] + Action Embedding e_a [B, D_a]                   |
   |   - Mode 1: Legacy Mean -> Pool [B, D] -> MLP Predictor -> \hat{z}_{t+k} [B, D]   |
   |   - Mode 2: Token-Preserving -> Transformer Predictor [B, N, D] -> \hat{z} [B, D]  |
   |   - Mode 3: Learned Query Pool -> Cross-Attn [B, K, D] -> Predictor -> \hat{z}    |
   +-----------------------------------------------------------------------------------+
                                            |
                                            v
   +-----------------------------------------------------------------------------------+
   | 5. EMA TARGET ENCODING (Single-Frame Target T=1)                                  |
   |                                                                                   |
   |   Target Input: [B, 52] -> f_θ̄(Target) -> z_{t+k} [B, D]                          |
   |   Loss: L_JEPA = || \hat{z}_{t+k} - stop_gradient(z_{t+k}) ||^2                   |
   |   EMA Update: θ̄ <- 0.996 θ̄ + 0.004 θ                                             |
   +-----------------------------------------------------------------------------------+
                                            |
                                            v
   +-----------------------------------------------------------------------------------+
   | 6. FROZEN LINEAR PROBE EVALUATION (src/cyber_jepa/evaluation/probes.py)           |
   |                                                                                   |
   |   z_eval -> Freeze -> Logistic Regression -> \hat{y} in {0, 1} vs Oracle y        |
   |   Metrics: Macro F1, Policy-Transfer F1, AUROC, Effective Rank                    |
   +-----------------------------------------------------------------------------------+
```

---

## 2. Directory Tree & Codebase Layout

```
c:\Users\rohil\Documents\GenAI Micro Project\T-JEPA-test\
├── src/cyber_jepa/                     # Primary Core Library
│   ├── env/                            # CybORG environment adapters & action wrappers
│   │   ├── observation_multiplexer.py  # 52-dim canonical vector extractor
│   │   └── action_mapper.py            # 66 discrete defensive action indexing
│   ├── data/                           # Dataset schemas, collectors, and storage
│   │   ├── schema.py                   # Strict dataclasses for transitions & oracle
│   │   ├── dataset.py                  # PyTorch DataLoader with sliding window h=4
│   │   ├── collector.py                # Telemetry collector with seed isolation
│   │   ├── storage.py                  # Parquet/JSONL shard reader/writer
│   │   └── characterize.py             # Feature staticity & distribution profiler
│   ├── representations/                # Representation topology implementations
│   │   ├── flat.py                     # Baseline flat temporal MLP/Linear encoder
│   │   ├── feature.py                  # 52 individual feature token embeddings
│   │   ├── host.py                     # 13 host entity token embeddings
│   │   ├── hierarchical.py             # Subnet-to-host multi-stage aggregation
│   │   ├── canonical.py                # Canonical representation registry & factory
│   │   ├── masking.py                  # Spatio-temporal token masking strategies
│   │   ├── budget.py                   # Parameter budget scaling & matching logic
│   │   └── flat_ablations.py           # Shuffled time & zeroed feature controls
│   ├── models/                         # JEPA model architectures
│   │   ├── jepa.py                     # Master Cyber-JEPA model with EMA target encoder
│   │   ├── predictor.py                # Action-conditioned latent transition predictors
│   │   ├── aggregators.py              # Context aggregators (mean, query, token-preserving)
│   │   ├── context.py                  # Temporal context encoders (Transformer/GRU)
│   │   ├── baselines.py                # Reconstructive & persistence baseline models
│   │   └── interface.py                # Abstract base class definitions
│   ├── evaluation/                     # Evaluation probes, metrics, and diagnostics
│   │   ├── probes.py                   # Frozen linear probe classifiers
│   │   ├── metrics.py                  # Macro F1, AUROC, Precision, Recall
│   │   ├── diagnostics.py              # Effective Rank computation & singular values
│   │   ├── diagnostics_extended.py     # Cosine similarity & variance monitoring
│   │   └── selection.py                # Model selection & validation criteria
│   ├── training/                       # Training engine & orchestration
│   │   ├── trainer.py                  # Core training loop with PyTorch optimizations
│   │   └── orchestrator.py             # Matrix sweep executor and checkpoint manager
│   └── utils/                          # Cross-cutting utilities
│       └── reproducibility.py          # Deterministic seeding contracts
├── scripts/                            # Execution scripts & CLI entrypoints
│   ├── run_collection.py               # Generates 90k transition CybORG dataset
│   ├── build_phase3_cohort.py          # Constructs fixed paired evaluation cohort
│   ├── run_phase3_pilot.py             # 1-seed architecture diagnostic pilot
│   ├── run_phase3_sweep.py             # 45-run parameter-matched Phase 3 sweep
│   └── analyze_phase3_results.py       # Bootstrap statistics & report generator
├── tests/                              # Automated pytest suite (12 test modules)
├── configs/                            # YAML experiment configuration files
├── experiments/phase3/                 # Phase 3 cohort, results JSON, and markdown report
└── context_handoff/                    # Comprehensive handoff documentation
```

---

## 3. Detailed Component Breakdown

### 3.1 Observation Multiplexer (`src/cyber_jepa/env/observation_multiplexer.py`)
- **Input**: Raw CybORG 3.1 dictionary observation emitted by `CybORG.step()`.
- **Output**: Fixed 52-dimensional float tensor representing 13 network hosts $\times$ 4 canonical properties:
  1. `Subnet_ID` (User=0, Enterprise=1, Operational=2)
  2. `IP_Address` (Normalized host octet)
  3. `Activity_State` (Active network connections or scans)
  4. `Compromised_State` (Defender-observable anomalous behavior or scan alerts)

### 3.2 Representation Encoders (`src/cyber_jepa/representations/`)
- **`FlatRepresentation`**: Flattens $[B, T=4, 52] \to [B, 208]$, passes through a 2-layer MLP with LayerNorm to output a compact latent $s_t \in \mathbb{R}^{256}$.
- **`FeatureTokenRepresentation`**: Treats every individual feature as a token. Passes 52 scalars through learned linear projections to create $[B, T, 52, D]$, followed by temporal self-attention.
- **`HostTokenRepresentation`**: Groups 4 features per host into 13 host tokens $[B, T, 13, D]$, capturing entity-level locality.
- **`HierarchicalTokenRepresentation`**: Two-stage aggregation: 13 hosts are aggregated into 3 subnet tokens (`User`, `Enterprise`, `Operational`), then aggregated into a global network latent.

### 3.3 Context Aggregators (`src/cyber_jepa/models/aggregators.py`)
Decoupled in Phase 3 to test why structured tokens previously underperformed:
1. `legacy_last_step_mean`: Averages token embeddings across the token dimension at the final timestep $t$.
2. `token_preserving_predictor`: Retains the full 2D token tensor $[B, N, D]$ into the predictor, allowing multi-head cross-attention across tokens during action conditioning.
3. `learned_query_pool`: Uses $K$ learned query vectors $Q \in \mathbb{R}^{K \times D}$ to pool tokens via cross-attention.

### 3.4 Action Predictor (`src/cyber_jepa/models/predictor.py`)
- Embeds discrete action index $a_t^{Blue} \in \{0\dots 65\}$ into action vector $e_a \in \mathbb{R}^{64}$.
- Predictor layers $g_\phi$ process $(s_t, e_a, k=8)$ using residual MLP blocks or Transformer cross-attention layers.
- Outputs predicted target latent $\hat{z}_{t+8} \in \mathbb{R}^{256}$.

### 3.5 Target Encoder & Single-Frame Correction
- **Old Approach (Phase 1–2 Bug)**: Expanded single target frame $O_{t+k}$ into artificial history $[O_{t+k}, O_{t+k}, O_{t+k}, O_{t+k}]$, forcing target encoder to process fake temporal context.
- **Phase 3 Correction ($T=1$)**: Context Encoder $f_\theta$ processes history $X_t \in \mathbb{R}^{4 \times 52}$, while Target Encoder $f_{\bar{\theta}}$ processes strictly single target frame $O_{t+k} \in \mathbb{R}^{52}$.

### 3.6 Evaluation Probes & Diagnostics (`src/cyber_jepa/evaluation/`)
- **`CriticalServerProbe`**: Frozen linear classifier reading $z_{t+k}$ to predict binary oracle label $y_{t+k} \in \{0, 1\}$ (whether `Op_Server0` is compromised).
- **`EffectiveRank`**: Measures dimensional collapse of representation matrix $Z \in \mathbb{R}^{M \times D}$:
  $$\text{EffRank}(Z) = \exp \left( - \sum_{i=1}^D p_i \ln p_i \right), \quad p_i = \frac{\sigma_i}{\sum_j \sigma_j}$$
  where $\sigma_i$ are singular values of $Z$.
  - $\text{EffRank} \approx 1.0$: Severe dimensional collapse (all vectors lie on a 1D line).
  - $\text{EffRank} \ge 7.0$: Healthy, diverse representation utilization.

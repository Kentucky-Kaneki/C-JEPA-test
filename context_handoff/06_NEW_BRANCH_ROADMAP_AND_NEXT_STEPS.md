# 06 — New Branch Roadmap & Next Steps Guide

> **Document Purpose**: Actionable blueprint, research priorities, and execution checklist for developers and researchers moving to the new experimental branch.

---

## 1. Context & Branch Transition Strategy

### Why Transition to a New Branch?
Phase 1, Phase 2, and Phase 3 empirical investigations on `cyborg-jepa-research` are **100% complete, statistically audited, and frozen**:
- Complete 45-run parameter-matched matrix executed.
- Paired bootstrap statistics and 95% confidence intervals generated.
- All 12 unit and integration test suites passing.
- Evidence ledger and decision-tree outcomes finalized.

The `cyborg-jepa-research` branch represents the canonical empirical foundation. All forward-looking experimental tracks belong on dedicated feature/research branches.

```
                           BRANCH TRANSITION ARCHITECTURE
                           
                                     [ cyborg-jepa-research ]
                                 (Phase 1-3 Empirical Baseline)
                                                |
        +-----------------------+---------------+-----------------------+
        |                       |                                       |
        v                       v                                       v
 [ feat/latent-mpc-planner ] [ feat/nasim-generalization ] [ feat/gnn-jepa-architecture ]
   Track 1: Active Defense     Track 2: Multi-Environment    Track 3: Anti-Collapse GNNs
```

---

## 2. Four Active Research Tracks for the New Branch

---

### Track 1: Downstream Action-Conditioned Planning (Active Latent World-Model / MPC)

#### Objective
Transform Cyber-JEPA from a passive representation learning model into an **active autonomous defensive planner** using Latent Model Predictive Control (MPC).

#### Theoretical Formulation
At timestep $t$, given history $X_t \in \mathbb{R}^{h \times 52}$, the defender:
1. Encodes current state: $s_t = f_\theta(X_t) \in \mathbb{R}^{256}$.
2. Samples $M=64$ candidate action trajectories of horizon $H=8$: $\mathbf{a}^{(m)} = [a_t^{(m)}, a_{t+1}^{(m)}, \dots, a_{t+H-1}^{(m)}]$.
3. Rolls out latent predictions autoregressively:
   $$\hat{z}_{t+1}^{(m)} = g_\phi(s_t, e(a_t^{(m)})), \quad \dots \quad \hat{z}_{t+H}^{(m)} = g_\phi(\hat{z}_{t+H-1}^{(m)}, e(a_{t+H-1}^{(m)}))$$
4. Evaluates trajectory cost using the frozen critical-server probe $h_\psi$:
   $$\mathcal{J}(\mathbf{a}^{(m)}) = \sum_{k=1}^H \gamma^k \cdot \text{Pr}\left( y_{t+k} = 1 \mid \hat{z}_{t+k}^{(m)} \right)$$
5. Selects the action sequence minimizing compromise probability and executes the first action $a_t^*$.

#### Deliverables & Implementation Plan
- [ ] Create `src/cyber_jepa/planning/` module.
- [ ] Implement `LatentMPCPlanner` (Cross-Entropy Method / Random Shooting).
- [ ] Evaluate episode return and defense success rate against CAGE Challenge 1–4 baselines (PPO, Rainbow DQN, Heuristic Sleep/Coverage).

---

### Track 2: Multi-Environment & Benchmark Generalization (NASim & CAGE Challenges)

#### Objective
Evaluate cross-environment transfer by benchmarking Cyber-JEPA on **NASim (Network Attack Simulator)** and larger CybORG scenarios (CAGE Challenge 2/3/4).

#### Key Challenges & Adaptations
1. **Dynamic Topologies**: NASim features variable numbers of hosts and services.
2. **Action Space Mapping**: Adapter must map NASim's multi-discrete action space to tokenized action embeddings.
3. **Existing Branch Reference**: Inspect `remotes/origin/Adding-NASim` for initial environment adapter work.

#### Deliverables & Implementation Plan
- [ ] Merge or rebase `Adding-NASim` onto the new branch.
- [ ] Standardize NASim observation multiplexer into canonical 4-feature host schema.
- [ ] Run zero-shot / few-shot transfer sweeps from CybORG pre-trained encoders to NASim evaluation trajectories.

---

### Track 3: Representation Architecture Innovations (GNN-JEPA to Fix Hierarchical Collapse)

#### Objective
Overcome the severe effective rank collapse ($\text{EffRank} = 1.4$) observed in Phase 3 hierarchical pooling by replacing static subnet averaging with **Graph Neural Networks (GNN-JEPA)**.

#### Proposed Architecture
```
    [ 13 Network Host Tokens ] ---> [ Relational Graph Conv / GAT ] ---> [ Edge-Conditioned Predictor ]
             (Nodes)                             (Topology)                         (Latent Predictions)
```
- Use the network adjacency matrix $\mathbf{A} \in \{0, 1\}^{13 \times 13}$ as relational edge priors.
- Host tokens exchange messages across active routing links rather than unweighted subnet pooling.
- Retain node-level resolution while aggregating global network risk.

#### Deliverables & Implementation Plan
- [ ] Implement `GraphJEPA` in `src/cyber_jepa/representations/graph.py` using PyTorch Geometric (`torch_geometric`).
- [ ] Run 5-seed sweep at $k=8$ and compute Effective Rank. Verify if $\text{EffRank} \ge 7.0$.

---

### Track 4: Paper Manuscript Drafting & Submission Packaging

#### Objective
Finalize the research paper for top-tier submission (NeurIPS, ICLR, or IEEE S&P).

#### Deliverables & Status Matrix

| Paper Department | Target Section | Current Status | Next Action on New Branch |
| :--- | :--- | :---: | :--- |
| **Dept 1: Title & Abstract** | Abstract & Problem Statement | 65% Complete | Finalize formal contribution bullet points. |
| **Dept 2: Related Work** | Methodological Comparison Matrix | 70% Complete | Verify BibTeX keys against official conference publications. |
| **Dept 3: Literature Survey** | 7 Theoretical Pillars | 80% Complete | Format into scholarly 2-column narrative. |
| **Dept 4: Methodology** | POMDP & Loss Formulation | 85% Complete | Export vector SVG/PDF diagrams of architecture. |
| **Dept 5: Experimental Setup** | CybORG 3.1 & Evaluation Protocol | 90% Complete | Finalize parameter budget table. |
| **Dept 6: Results & Analysis** | 45-Run Sweep & Paired Bootstrap | 95% Complete | Insert publication-quality F1 / AUROC bar charts. |
| **Dept 7: Discussion & Limitations** | Scope Boundaries & Collapse | 75% Complete | Refine mechanistic discussion on subnet pooling. |
| **Dept 8: Artifact Packaging** | Code & Data Open-Sourcing | 90% Complete | Create anonymous GitHub repository archive. |

---

## 3. Quickstart: Creating the New Branch & Immediate Commands

```powershell
# 1. Create and switch to the new experimental branch
git checkout -b feat/phase4-active-planning-and-mpc

# 2. Verify all existing tests pass on the clean branch
.\.venv-ml\Scripts\Activate.ps1
pytest -v tests/

# 3. Create planning scaffold
New-Item -ItemType Directory -Path "src/cyber_jepa/planning" -Force
New-Item -ItemType File -Path "src/cyber_jepa/planning/__init__.py" -Force
New-Item -ItemType File -Path "src/cyber_jepa/planning/mpc.py" -Force

# 4. Verify GPU availability
python -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

## 4. Quality Checklist for New Branch Contributors

Before submitting pull requests on the new branch, ensure:
- [ ] No hardcoded random seeds; all randomness must use `set_seed()`.
- [ ] No evaluation on training split groups; all test evaluations must use `split_group_id` holdout isolation.
- [ ] No privileged oracle features leaked to Context Encoder $f_\theta$ or Predictor $g_\phi$.
- [ ] All new modules accompanied by unit tests in `tests/`.
- [ ] All 12 baseline test suites continue to pass (`pytest tests/`).

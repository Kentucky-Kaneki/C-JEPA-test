# 01 — Problem Formulation, Theory & Mathematical Foundations

> **Document Purpose**: Complete mathematical, theoretical, and operational foundation of Cyber-JEPA for researchers and algorithm designers.

---

## 1. Problem Formulation: Autonomous Cyber Defense as a POMDP

Autonomous cyber operations operate under **Partial Observability Markov Decision Processes (POMDP)**. The environment state is defined by the 6-tuple:
$$\mathcal{M} = \langle \mathcal{S}, \mathcal{A}^{Blue}, \mathcal{A}^{Red}, \mathcal{T}, \mathcal{O}^{Blue}, \mathcal{R} \rangle$$

Where:
- $\mathcal{S}$: The true, unobserved state of the network (complete process tables, file system privileges, active attacker C2 sessions across all hosts).
- $\mathcal{A}^{Blue}$: Discrete action space of the defender ($|\mathcal{A}^{Blue}| = 66$ in CybORG 3.1 `Scenario1b`).
- $\mathcal{A}^{Red}$: Adversary policy space executing stealthy lateral movement (`B_lineAgent` direct attack or `RedMeanderAgent` stochastic search).
- $\mathcal{T}(S_{t+1} \mid S_t, a_t^{Blue}, a_t^{Red})$: Environmental state transition dynamics.
- $\mathcal{O}^{Blue}$: Partially observable defensive telemetry ($O_t^{Blue} \in \mathbb{R}^{52}$).
- $\mathcal{R}$: Defender reward function (penalizing host compromise and service interruption).

```
                      POMDP OPERATIONAL TIMELINE
                      
   True State:       S_t ----------------------------------> S_{t+k} (Unobserved)
                      |                                         |
   Emits:             v                                         v
   Observations:  O_t^{Blue} in R^{52}                      O_{t+k}^{Blue} in R^{52}
                      |                                         |
   Sliding Window: X_t = [O_{t-h:t}^{Blue}]                     | (Target T=1)
                      |                                         |
   Encoder:       s_t = f_θ(X_t)                            z_{t+k} = f_θ̄(O_{t+k}^{Blue})
                      |                                         |
   Predictor:     \hat{z}_{t+k} = g_φ(s_t, a_t^{Blue}, k)      |
                      \                                        /
                       \---> L_JEPA = ||\hat{z} - sg(z)||^2 <--/
```

---

## 2. The Observation Staticity Paradox in Cyber Systems

In continuous domains (robotics, autonomous driving, video), world models predict dense, continuous sensor variations. In contrast, cyber telemetry exhibits **extreme staticity**:

### 1. Mathematical Definitions
1. **Scalar Feature Staticity (>90%)**:
   $$\text{Feature Staticity} = \frac{1}{52 \cdot N_{\text{steps}}} \sum_{t=1}^{N_{\text{steps}}} \sum_{i=1}^{52} \mathbb{I}\left( O_{t, i}^{Blue} = O_{t-1, i}^{Blue} \right) > 0.90$$
2. **Zero-Change Transition Fraction (42.29%)**:
   $$\text{Zero-Change Fraction} = \frac{1}{N_{\text{steps}}} \sum_{t=1}^{N_{\text{steps}}} \mathbb{I}\left( O_t^{Blue} = O_{t-1}^{Blue} \right) = 0.4229$$

### 2. Why Reconstructive World Models Fail
Reconstructive world models (e.g., Dreamer, VAE-based transition models) optimize:
$$\mathcal{L}_{\text{Recon}} = \mathbb{E} \left[ \| \hat{O}_{t+k}^{Blue} - O_{t+k}^{Blue} \|_2^2 \right]$$

Because $>90\%$ of features are static across steps, a decoder minimizing $\mathcal{L}_{\text{Recon}}$ minimizes loss almost completely by learning a trivial **identity copy operation**:
$$\hat{O}_{t+k}^{Blue} \approx O_t^{Blue}$$
Under this identity shortcut, the latent state representation discards subtle, sparse compromise signals (such as 1 bit indicating an abnormal session on `Enterprise1`).

### 3. The JEPA Solution
Cyber-JEPA operates exclusively in **latent representation space**:
$$\mathcal{L}_{\text{JEPA}} = \| \hat{z}_{t+k} - \text{sg}(z_{t+k}) \|_2^2$$
Because target embeddings $z_{t+k} = f_{\bar{\theta}}(O_{t+k}^{Blue})$ are produced by an Exponential Moving Average (EMA) encoder and regularized against collapse, the predictor $g_\phi$ is forced to predict high-level semantic transition dynamics rather than raw static features.

---

## 3. The Characteristic Attacker Kill-Chain Horizon ($k=8$)

Empirical horizon sweeps across $k \in \{1, 2, 4, 8, 16\}$ proved that representation quality peaks at **$k=8$**:

```
                       HORIZON STATE CHANGE MECHANISM
                       
     Horizon k=1:  [  3.77% State Changes  ] -> Dominated by trivial static inertia.
     Horizon k=4:  [ 14.80% State Changes  ] -> Partial attacker reconnaissance.
     Horizon k=8:  [ 26.30% State Changes  ] -> OPTIMAL: Matches Red kill-chain lateral transition.
     Horizon k=16: [ 46.79% State Changes  ] -> Stochastic compounding degrades AUROC (0.82 -> 0.77).
```

- In CybORG `Scenario1b`, 8 timesteps correspond exactly to the duration required for Red to scan a subnet, exploit a service (e.g., `SSHBruteForce` or `FTPDirectoryTraversal`), and escalate privileges to root.
- At $k=8$, **26.30% (~1 in 4)** transitions experience a true ground-truth change in critical server compromise state ($\Delta y \neq 0$).

---

## 4. Information Boundaries & Strict Separation Rules

To prevent data leakage and maintain deployability, Cyber-JEPA enforces 4 strict information tiers:

| Information Tier | Content | Permitted Usage | Prohibited Usage |
| :--- | :--- | :--- | :--- |
| **Tier 1: Deployable Telemetry** | $O_t^{Blue} \in \mathbb{R}^{52}$ (Subnet ID, IP, Activity, Compromise flag per host). | Input to Context Encoder $f_\theta$ and Target Encoder $f_{\bar{\theta}}$. | Must never be mixed with ground truth labels. |
| **Tier 2: Blue Action Space** | $a_t^{Blue} \in \{0\dots 65\}$ (Action type, target host/subnet). | Input to Action Predictor $g_\phi$. | Must not condition target encoder. |
| **Tier 3: Diagnostic Logs** | Loss values, Effective Rank, gradient norms. | Offline validation and collapse monitoring. | Cannot be fed into model forward passes. |
| **Tier 4: Privileged Oracle Sidecar** | $y_t \in \{0, 1\}$ (True ground-truth root compromise of `Op_Server0`). | **Post-hoc frozen linear probes ONLY**. | **STRICTLY FORBIDDEN** from $f_\theta, f_{\bar{\theta}}, g_\phi$. |

---

## 5. Mathematical Objective & Loss Formulation

### 1. Forward Loss Objective
Given a batch of transitions $(X_t, a_t^{Blue}, O_{t+k}^{Blue})$:
1. **Context Encoding**:
   $$s_t = f_\theta(X_t), \quad X_t \in \mathbb{R}^{h \times 52}, \quad s_t \in \mathbb{R}^{d}$$
2. **Action Conditioning**:
   $$e_a = W_{\text{action}} \cdot \text{one\_hot}(a_t^{Blue}), \quad e_a \in \mathbb{R}^{d_a}$$
3. **Latent Prediction**:
   $$\hat{z}_{t+k} = g_\phi(s_t, e_a, k), \quad \hat{z}_{t+k} \in \mathbb{R}^{d}$$
4. **Target Encoding (Single Frame $T=1$)**:
   $$z_{t+k} = f_{\bar{\theta}}(O_{t+k}^{Blue}), \quad z_{t+k} \in \mathbb{R}^{d}$$
5. **JEPA Loss**:
   $$\mathcal{L}_{\text{JEPA}}(\theta, \phi) = \frac{1}{B} \sum_{i=1}^B \| \hat{z}_{t+k}^{(i)} - \text{sg}\left(z_{t+k}^{(i)}\right) \|_2^2$$
   where $\text{sg}(\cdot)$ denotes the `stop_gradient` operator.

### 2. EMA Target Encoder Update
The target encoder parameters $\bar{\theta}$ are updated asynchronously via Polyak averaging:
$$\bar{\theta}_{step} \leftarrow \tau \bar{\theta}_{step} + (1 - \tau) \theta_{step}$$
where $\tau = 0.996$ (linearly annealed to $1.0$ during training).

### 3. Post-Hoc Linear Probe Evaluation
To evaluate whether latent vectors capture meaningful cyber security state without fine-tuning encoder weights:
1. Freeze encoder parameters $\theta$.
2. Train a linear classifier $h_\psi(z) = \sigma(W_{\text{probe}} z + b)$.
3. Compute **Standard Macro F1**, **Policy-Transfer F1**, and **AUROC** on unseen test split-groups predicting binary oracle target $y_{t+k} \in \{0, 1\}$.

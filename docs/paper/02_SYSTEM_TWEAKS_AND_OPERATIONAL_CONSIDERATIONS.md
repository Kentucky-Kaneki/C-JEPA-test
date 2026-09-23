# System Tweaks, Engineering Considerations & Operational Design in Cyber-JEPA

> **Target Manuscript Placement**: Section 4 (Methodology & Policy Coupling) & Supplementary Operational Guide  
> **Key Themes**: Phase-Space Kinematic Manifolds, Action Cooldowns, Zone-Specific Remediations, Telemetry Slicing, Enterprise False-Alarm Suppression

---

## 1. The Operational Reality of Enterprise Cyber Defense

In production enterprise networks (and high-fidelity benchmarks like CybORG Enterprise / CAGE Challenge 4), defenders cannot treat cyber defense as an unconstrained video game. Every blue action carries real operational consequences:
- **`Restore <host>`**: Re-images the operating system and reboots services. While it guarantees removal of malware, all active user sessions are severed, incurring severe **Local Work Fail (LWF)** penalties and service downtime.
- **`Remove <host>`**: Terminates suspected malicious processes and sessions. Less disruptive than `Restore`, but risks killing legitimate administrative or background scripts.
- **`Sleep`**: The defender takes zero action, allowing all legitimate business, administrative, and mission services to run unhindered.

The fundamental engineering challenge is **cost-aware gating**: executing `Sleep` for $>95\%$ of nominal operational life, while reacting with surgical precision the instant an adversary killchain departs from nominal dynamics.

---

## 2. Kinematic Phase-Space Anomaly Scoring ($S_t$)

Cyber-JEPA replaces static alert thresholds and unguided RL exploration with **Phase-Space Kinematic Manifold Scoring**.

```
                           THE PHASE-SPACE ANOMALY GEOMETRY
                                          
               Nominal Manifold ℳ ⊂ 𝕊^{D-1}          Adversary Deviation
             ┌──────────────────────────────┐              ┌───────────────┐
             │       c_1         c_2        │              │               │
             │           z_{t-1}            │   Killchain  │      z_t      │
             │              \               │  Trajectory  │       ▲       │
             │               \ v_t          │ ───────────► │       │       │
             │                ▼             │              │  d_cluster(z) │
             │               z_t     c_3    │              │       │       │
             │                              │              │       ▼       │
             │       c_4         c_5        │              │   c_closest   │
             └──────────────────────────────┘              └───────────────┘
```

### Mathematical Formulation:
1. **Hypersphere Projection**:
   Every latent context vector $z_t \in \mathbb{R}^{D_z}$ output by the Transformer encoder is $L_2$-normalized onto the unit hypersphere:
   $$\hat{z}_t = \frac{z_t}{\max(1e-12, \|z_t\|_2)} \in \mathbb{S}^{D_z - 1}$$
   Unit-sphere projection eliminates trivial magnitude scaling and ensures distance metrics operate purely on directional representation alignment.

2. **Multi-Centroid Manifold Distance ($d_{\text{cluster}}$)**:
   During an offline calibration phase on clean nominal operational traces (with Red absent/sleeping), we fit $K=6$ spherical $K$-Means centroids $\mathcal{C} = \{c_1, \dots, c_K\} \subset \mathbb{S}^{D_z - 1}$. Six centroids capture the distinct operational phases (Preplanning, Mission A, Mission B, shifts in Green work schedules). The cluster distance is defined as:
   $$d_{\text{cluster}}(\hat{z}_t) = \text{clip}\left(1.0 - \max_{k \in \{1,\dots,K\}} c_k^\top \hat{z}_t, \, 0.0, \, 2.0\right)$$
   When the network operates nominally, $\hat{z}_t$ aligns closely with at least one centroid, yielding $d_{\text{cluster}} \approx 0$.

3. **Latent Angular Velocity ($v_t$) and Kinematic Anomaly ($v_{\text{anomaly}}$)**:
   Adversarial actions (exploits, lateral pivoting, privilege escalation) alter network relational states much faster or differently than background user drift. We track the instantaneous latent velocity:
   $$v_t = \text{clip}\left(1.0 - \hat{z}_t^\top \hat{z}_{t-1}, \, 0.0, \, 2.0\right)$$
   We compute the relative velocity departure against the empirical nominal mean velocity $v_{\text{clean}}$:
   $$v_{\text{anomaly}}(v_t) = \text{clip}\left(\frac{|v_t / v_{\text{clean}} - 1.0|}{2.0}, \, 0.0, \, 1.0\right)$$
   **Why this matters**: Crucially, this detects **both sudden acceleration** (abrupt lateral pivot spikes) and **abnormal deceleration** (process freezing or stealthy persistent stalls).

4. **Composite Score & Calibrated Threshold**:
   The final continuous threat score blends geometry and kinematics with weight $w_v = 0.25$:
   $$S_t = (1 - w_v) \cdot d_{\text{cluster}}(\hat{z}_t) + w_v \cdot v_{\text{anomaly}}(v_t)$$
   The alert threshold $\tau$ is set empirically to the **95th percentile of clean nominal scores $+ 0.05$ margin** (typically $\tau \in [0.30, 0.48]$ depending on zone scale).

---

## 3. Cooldown Damping Mechanism

A major failure mode in autonomous defense is **action flapping** (repeatedly issuing `Restore` or `Remove` to the same machine step after step). 
- If host $h$ is restored at step $t$, it requires reboot time to re-establish legitimate services.
- If the defender issues another `Restore` at $t+1$, it unnecessarily resets the host again, extending downtime and multiplying LWF penalties.

### Implementation:
Cyber-JEPA maintains an episodic cooldown registry $\mathcal{D}_{\text{cooldown}}: \mathcal{H} \to \mathbb{Z}_{\ge 0}$:
- When an action is taken on host $h$, a cooldown timer is set: $\mathcal{D}_{\text{cooldown}}[h] \leftarrow \Delta_{\text{cooldown}}$ (default $\Delta = 2$ steps).
- On each environment step, all active cooldowns decrement: $\mathcal{D}_{\text{cooldown}}[h] \leftarrow \max(0, \mathcal{D}_{\text{cooldown}}[h] - 1)$.
- Hosts with $\mathcal{D}_{\text{cooldown}}[h] > 0$ are strictly excluded from candidate target pools, allowing services to recover and preventing oscillatory action thrashing.

---

## 4. Zone-Differentiated Tactical Remediation Policy

Enterprise networks are partitioned into zones with distinct operational missions. Cyber-JEPA applies zone-differentiated remediation priority:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        ZONE-SPECIFIC TACTICAL ACTION HIERARCHY                         │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 1. Restricted Zones (Agents 0, 2) & Headquarters (Agent 4):                            │
│    Mission Priority: Core Mission Services & Crown Jewel Integrity                     │
│    Strategy: Prioritize Server Hosts → User Workstations                               │
│    Remediation: Targeted `Restore` on Servers; `Remove` on Secondary Hosts             │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 2. Operational Zones (Agents 1, 3):                                                    │
│    Mission Priority: Containment of Initial Attack Footholds                           │
│    Strategy: Prioritize User Workstations → Server Hosts                               │
│    Remediation: Rapid `Remove` / `Restore` on Compromised User Endpoints to Sever     │
│                 Lateral Killchain Pivots Before Reaching Restricted Enclaves           │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 3. Fallback Hierarchy (When Candidate Hosts are on Cooldown):                          │
│    DeployDecoy on Primary Server → Analyse on Suspicious Host → Fallback to Sleep     │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Telemetry Feature Layout & Flag Isolation in CAGE 4

Through line-by-line inspection of CybORG's `BlueFlatWrapper`, we identified a critical telemetry pitfall that caused prior heuristic baselines to fail:

### The 59-Feature Subnet Block Layout:
In `BlueFlatWrapper`, each protected subnet produces exactly 59 scalar features:
1. `subnet_onehot`: Indices $0 \dots 8$ (9 features: one-hot subnet identity).
2. `blocked_subnets`: Indices $9 \dots 17$ (9 features: subnet communication blocks).
3. `comms_policy`: Indices $18 \dots 26$ (9 features: phase communication permissions).
4. `malicious_processes`: Indices $27 \dots 42$ (16 features: process alerts per host).
5. `network_connections`: Indices $43 \dots 58$ (16 features: network connection flags per host).

### The Discovery:
- **Process Creation Flags (`[27:43]`)**: In CybORG Enterprise, Green background users spawn legitimate processes continuously as part of normal enterprise work. Naive telemetry baselines checking process flags misinterpret benign enterprise software as malware, triggering non-stop false alarms.
- **Connection Flags (`[43:59]`)**: Unauthorized network connections specifically flag external adversary lateral movement.
- **Why Cyber-JEPA Wins**: Cyber-JEPA does not rely on hand-tuned hardcoded flag rules. By embedding the temporal trajectory of all 59 features into latent space, it naturally learns the joint covariance of benign user traffic. It only alerts when the entire temporal manifold departs from nominal dynamics.

---

## 6. The Warmup Guard ($t > H$)

At the start of an episode ($t < H$), the sliding context buffer $\mathbf{w}_t$ is populated by repeating the initial observation frame. 
- While this prevents mathematical `NaN`s, the repeated initial frame creates an artificial zero-velocity kinematic artifact ($v_t = 0$).
- Cyber-JEPA enforces a strict **Warmup Guard**:
  $$\text{has\_phase\_alert} = (S_t \ge \tau) \;\land\; (t > H)$$
- During steps $1 \dots H$, the agent unconditionally executes `Sleep`. This guarantees that the temporal buffer reaches stationary sliding dynamics before any tactical remediation decisions are permitted.

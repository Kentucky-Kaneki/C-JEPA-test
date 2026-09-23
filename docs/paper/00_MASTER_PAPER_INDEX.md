# Cyber-JEPA Research Dossier: Master Paper Index & Organization Blueprint

> **Branch**: `paper-materials`  
> **Status**: Active Working Repository for Manuscript Drafting & Evidence Organization  
> **Scope**: Autonomous Cyber Defense across CAGE Challenge 2, 3, and 4

---

## 1. Executive Organization & Guiding Principles

This directory (`docs/paper/`) serves as the **single source of truth** for all scientific, empirical, architectural, and bibliographic materials for the Cyber-JEPA publication. 

To maintain clarity, rigor, and focus in accordance with top-tier conference standards (e.g., AAAI, NeurIPS, IEEE S&P):
1. **Main Paper vs. Supplementary Partitioning**:
   - **Main Paper**: Focuses on the core conceptual paradigm shift, novel architectural ingestion methods, system design tweaks, and straightforward headline benchmark results that directly prove our hypotheses.
   - **Supplementary Material**: Houses exhaustive ablation sweeps, multi-scale pretraining trajectories (Scale 1–100), full per-episode per-agent logs, complete parameter tables, and extended failure mode proofs.
2. **Three Core Pillars of the Paper**:
   - **Pillar 1: Novel Ingestion & Architectural Foundations**: The Flat Vector sliding-window ($H=4$) representation, zero-padding dimension adaptation for asymmetric Dec-POMDPs, and mathematical proof of why hierarchical/tokenized pooling suffers rank collapse.
   - **Pillar 2: System Considerations & Operational Tweaks**: Kinematic phase-space anomaly detection ($d_{\text{cluster}} + v_{\text{anomaly}}$), cooldown damping, zone-differentiated action mapping, and telemetry separation.
   - **Pillar 3: Empirical Validation across Benchmarks**: Direct comparison against reactive baselines, passive baselines, and CAGE Challenge 4 competition leaders (CardiffUni, Punch Cyber, Cybermonic).

---

## 2. Master Document Map

| Document | Primary Scope | Intended Destination |
| :--- | :--- | :--- |
| [`01_ARCHITECTURE_AND_INGESTION_INNOVATIONS.md`](file:///c:/Users/rohil/Documents/GenAI%20Micro%20Project/T-JEPA-test/docs/paper/01_ARCHITECTURE_AND_INGESTION_INNOVATIONS.md) | Ingestion method, flat representation, history windowing, asymmetric Dec-POMDP zero-padding, rank collapse analysis | **Main Paper (Sec 3 & 4)** |
| [`02_SYSTEM_TWEAKS_AND_OPERATIONAL_CONSIDERATIONS.md`](file:///c:/Users/rohil/Documents/GenAI%20Micro%20Project/T-JEPA-test/docs/paper/02_SYSTEM_TWEAKS_AND_OPERATIONAL_CONSIDERATIONS.md) | Phase-space kinematic manifolds ($S_t$), cooldown timers, zone-differentiated policies, CAGE 4 telemetry layout | **Main Paper (Sec 4) & Supplement** |
| [`03_MAIN_PAPER_RESULTS.md`](file:///c:/Users/rohil/Documents/GenAI%20Micro%20Project/T-JEPA-test/docs/paper/03_MAIN_PAPER_RESULTS.md) | Clean headline tables: CAGE 2/3 Crown Jewel Preservation, CAGE 4 50-ep benchmark, comparison with CC4 leaders | **Main Paper (Sec 5)** |
| [`04_SUPPLEMENTARY_EXHAUSTIVE_RESULTS.md`](file:///c:/Users/rohil/Documents/GenAI%20Micro%20Project/T-JEPA-test/docs/paper/04_SUPPLEMENTARY_EXHAUSTIVE_RESULTS.md) | Exhaustive Scale 1–100 curves, full OOD matrices, 480-ep breakdown, per-agent logs, hyperparameter grids | **Supplementary Appendix** |
| [`05_CITATIONS_AND_RELATED_WORK_POSITIONING.md`](file:///c:/Users/rohil/Documents/GenAI%20Micro%20Project/T-JEPA-test/docs/paper/05_CITATIONS_AND_RELATED_WORK_POSITIONING.md) | Verified BibTeX library, competitive positioning matrices, literature survey across 7 foundational pillars | **Main Paper (Sec 2) & References** |
| [`RESEARCH_FRAMING_AND_MANUSCRIPT_BLUEPRINT.md`](file:///c:/Users/rohil/Documents/GenAI%20Micro%20Project/T-JEPA-test/docs/paper/RESEARCH_FRAMING_AND_MANUSCRIPT_BLUEPRINT.md) | Narrative arc, hypothesis trees, contribution claims, section-by-section outline | **Authors' Drafting Guide** |

---

## 3. High-Level Evidence Summary

```
                                  CYBER-JEPA RESULTS AT A GLANCE
                                                │
         ┌──────────────────────────────────────┼──────────────────────────────────────┐
         ▼                                      ▼                                      ▼
[CAGE 2 & 3 Benchmark]                 [CAGE 4 Enterprise MARL]              [Representation Quality]
• Crown Jewel Pres: 90.0% (B_line)     • Team Reward: -50.00 ± 44.5          • Effective Rank: 15.6 (Flat)
• Crown Jewel Pres: 100.0% (Meander)   • vs Reactive: -78.50 (p < 0.005)     • vs Hierarchical: 1.4 (Collapsed)
• Reward: -13.46 (Best on CybORG 3.1)  • Intrusions Cut: 57.3% (194 vs 454)  • Latency: 12.3ms (5 agents on CPU)
• False Alarms: 0.0 on nominal peace   • Sleep Efficiency: 97.3%             • Zero-day OOD transfer proven
```

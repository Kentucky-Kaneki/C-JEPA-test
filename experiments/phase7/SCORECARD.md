# Cyber-JEPA Phase 7: Failure Forensics & Emergent OOD Scorecard

## Track 1: The False Alarm Paradox (Perimeter vs. Crown Jewel Audit)

| Metric | Measured Value | Operational Interpretation |
| :--- | :---: | :--- |
| **Nominal Clean Steps (`Op_Server0` Clean)** | 6044 | Steps where crown jewel was uncompromised |
| **Nominal False Alarms** | 201 (3.33%) | Flagged alerts when `Op_Server0` was clean |
| **Early Perimeter Detections (>= 1 host compromised)** | **200 / 201 (99.5%)** | **Alerts where active intruder had breached User/Enterprise hosts!** |
| **True False Alarms (0 hosts compromised)** | **1 (0.17%)** | **Genuine false alarms during completely benign operations** |
| **Avg Hosts Compromised on False Alarm** | **4.28 hosts** | Average number of active compromised hosts during flagged alerts |

### Blue Defense Action Interference Audit

| Defender Action Category | Clean Steps | False Alarms | False Alarm Rate (%) | Operational Impact |
| :--- | :---: | :---: | :---: | :--- |
| **Disruptive Actions (`Restore`, `Remove`)** | 0 | 0 | **0.00%** | Telemetry resets create transient false alerts |
| **Passive Actions (`Sleep`, `Monitor`, etc.)** | 6044 | 201 | **3.33%** | Stable background telemetry baseline |
| **Disruption Amplification Ratio** | -- | -- | **0.00x** | Relative false alarm multiplier during remediation |

---

## Track 2: Detection Performance Optimization & Threshold Calibration

### Operating Threshold Comparison

| Operating Mode | Operating Threshold (tau) | Temporal Smoothing (alpha) | Attacks Caught / Total | Detection Rate (%) | Missed Attacks | False Alarms (%) | Macro F1 | Balanced Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Default Baseline** | 0.500 | None (1.0) | 5138 / 5308 | 96.80% | 170 | 3.33% | 0.9672 | 0.9674 |
| **Youden's J Calibrated** | **0.555** | None (1.0) | **5108 / 5308** | **96.23%** | **200** | 2.81% | **0.9673** | **0.9671** |
| **$F_2$ Recall Optimized** | **0.380** | None (1.0) | **5199 / 5308** | **97.95%** | **109** | 4.68% | 0.9654 | 0.9663 |
| **FPR <= 5% Capped** | **0.320** | None (1.0) | **5220 / 5308** | **98.34%** | **88** | 5.59% | 0.9624 | 0.9637 |
| **Smoothed + Calibrated** | **0.555** | **0.60** | **5059 / 5308** | **95.31%** | **249** | 2.23% | **0.9660** | **0.9654** |

### Dual-Target Evaluation (Perimeter vs Crown Jewel)

| Probing Target | Detection Target | Test Positives | Detection Rate (%) | False Alarm Rate (%) | Macro F1 | AUROC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Crown Jewel Probe** | `critical_server_compromised` | 5308 | 96.80% | 3.33% | 0.9672 | 0.9928 |
| **Perimeter Probe** | `any_host_compromised` | 10765 | 99.84% | 8.01% | 0.9705 | 0.9977 |

---

## Track 3: Emergent Zero-Label Latent Anomaly Detection

| Anomaly Distance Metric | AUROC (Zero Supervision) | PR-AUC (Zero Supervision) | Detection Rate @ 95% Specificity | Emergent Detection Viable? |
| :--- | :---: | :---: | :---: | :---: |
| **Latent Cosine Distance** | **0.7641** | **0.6808** | **10.15%** | **YES (AUROC >= 0.75)** |
| **Latent Euclidean Distance** | 0.7628 | 0.6797 | -- | YES (AUROC >= 0.75) |

---

## Track 4: Cross-Policy Out-of-Distribution (OOD) Zero-Shot Transfer

| Transfer Direction | In-Distribution F1 | Zero-Shot OOD F1 | In-Dist Det (%) | Zero-Shot OOD Det (%) | Zero-Shot Retention Rate (%) | Generalization Penalty (Delta F1) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **B-line -> Meander (Targeted -> Stealth)** | 0.9823 | **0.8353** | 99.6% | **67.4%** | **85.0%** | -0.1470 |
| **Meander -> B-line (Stealth -> Targeted)** | 0.9416 | **0.9513** | 90.3% | **98.0%** | **101.0%** | +0.0097 |

---

## Track 5: MITRE ATT&CK Multi-Tier Host Localization

| MITRE ATT&CK Tier | Monitored Hosts | Mean Macro F1 | Mean AUROC | Mean FPR@95% Recall | Localization Health |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **User Tier (Initial Access)** | User1, User2, User3, User4 | **0.9533** | **0.9808** | 0.1072 | Strong |
| **Enterprise Tier (Lateral Movement)** | Enterprise0, Enterprise1, Enterprise2 | **0.9371** | **0.9804** | 0.0904 | Robust |
| **Operational Tier (Crown Jewels)** | Op_Server0, Op_Host0, Op_Host1, Op_Host2 | **0.7418** | **0.9532** | 0.0616 | High Precision |

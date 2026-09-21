# Cyber-JEPA Phase 7: Failure Forensics & Emergent OOD Scorecard

## Track 1: The False Alarm Paradox (Perimeter vs. Crown Jewel Audit)

| Metric | Measured Value | Operational Interpretation |
| :--- | :---: | :--- |
| **Nominal Clean Steps (`Op_Server0` Clean)** | 5155 | Steps where crown jewel was uncompromised |
| **Nominal False Alarms** | 494 (9.58%) | Flagged alerts when `Op_Server0` was clean |
| **Early Perimeter Detections ($\ge 1$ host compromised)** | **488 / 494 (98.8%)** | **Alerts where active intruder had breached User/Enterprise hosts!** |
| **True False Alarms (0 hosts compromised)** | **6 (10.71%)** | **Genuine false alarms during completely benign operations** |
| **Avg Hosts Compromised on False Alarm** | **4.01 hosts** | Average number of active compromised hosts during flagged alerts |

---

## Track 2: Detection Performance Optimization & Threshold Calibration

| Operating Mode | Operating Threshold (tau) | Temporal Smoothing (alpha) | Attacks Caught / Total | Detection Rate (%) | Missed Attacks | False Alarms (%) | Macro F1 | Balanced Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Default Baseline** | 0.500 | None (1.0) | 5842 / 6197 | 94.27% | 355 | 9.58% | 0.9244 | 0.9234 |
| **Youden's J Calibrated** | **0.487** | None (1.0) | **5884 / 6197** | **94.95%** | **313** | 10.05% | **0.9259** | **0.9245** |
| **$F_2$ Recall Optimized** | **0.270** | None (1.0) | **6015 / 6197** | **97.06%** | **182** | 13.71% | 0.9202 | 0.9167 |
| **Smoothed + Calibrated** | **0.487** | **0.60** | **5766 / 6197** | **93.05%** | **431** | 8.63% | **0.9222** | **0.9221** |

---

## Track 3: Emergent Zero-Label Latent Anomaly Detection

| Anomaly Distance Metric | AUROC (Zero Supervision) | PR-AUC (Zero Supervision) | Detection Rate @ 95% Specificity | Emergent Detection Viable? |
| :--- | :---: | :---: | :---: | :---: |
| **Latent Cosine Distance** | **0.7959** | **0.7823** | **16.23%** | **YES (AUROC >= 0.75)** |
| **Latent Euclidean Distance** | 0.7953 | 0.7817 | -- | YES |

---

## Track 4: Cross-Policy Out-of-Distribution (OOD) Zero-Shot Transfer

| Transfer Direction | In-Distribution F1 | Zero-Shot OOD F1 | In-Dist Det (%) | Zero-Shot OOD Det (%) | Zero-Shot Retention Rate (%) | Generalization Penalty (Delta F1) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **B-line -> Meander (Targeted -> Stealth)** | 0.9336 | **0.6832** | 99.0% | **97.7%** | **73.2%** | -0.2504 |
| **Meander -> B-line (Stealth -> Targeted)** | 0.8877 | **0.8385** | 85.0% | **86.7%** | **94.5%** | -0.0492 |

---

## Track 5: MITRE ATT&CK Multi-Tier Host Localization

| MITRE ATT&CK Tier | Monitored Hosts | Mean Macro F1 | Mean AUROC | Mean FPR@95% Recall | Localization Health |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **User Tier (Initial Access)** | User1, User2, User3, User4 | **0.8997** | **0.9566** | 0.2349 | Strong |
| **Enterprise Tier (Lateral Movement)** | Enterprise0, Enterprise1, Enterprise2 | **0.8691** | **0.9409** | 0.2796 | Robust |
| **Operational Tier (Crown Jewels)** | Op_Server0, Op_Host0, Op_Host1, Op_Host2 | **0.7310** | **0.9671** | 0.1137 | High Precision |

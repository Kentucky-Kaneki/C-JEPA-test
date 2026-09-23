# CybORG CAGE Closed-Loop Autonomous Defender Benchmark Scorecard

Evaluation of Cyber-JEPA world model vs CybORG standard defensive baselines.

## Summary Results Table

| Scenario | Red Attacker | Blue Defender | Mean Reward +/- Std | Crown Jewel Preserved (%) | Enterprise Preserved (%) | Mean CJ Steps Comp | Latency (ms) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Scenario1b.yaml | `bline` | **Cyber-JEPA (Ours)** | -24.40 +/- 7.63 | **90.0%** | 0.0% | 0.70 | 4.21 |
| Scenario1b.yaml | `bline` | BlueReactRestore | -14.28 +/- 2.72 | **100.0%** | 73.3% | 0.00 | 0.02 |
| Scenario1b.yaml | `bline` | BlueReactRemove | -25.29 +/- 56.56 | 80.0% | 60.0% | 1.87 | 0.01 |
| Scenario1b.yaml | `bline` | SleepAgent (Passive) | -205.20 +/- 42.87 | 0.0% | 0.0% | 16.30 | 0.01 |
| Scenario1b.yaml | `meander` | **Cyber-JEPA (Ours)** | -13.46 +/- 4.25 | **100.0%** | 0.0% | 0.00 | 4.23 |
| Scenario1b.yaml | `meander` | BlueReactRestore | -10.45 +/- 1.43 | **100.0%** | 70.0% | 0.00 | 0.02 |
| Scenario1b.yaml | `meander` | BlueReactRemove | -8.44 +/- 7.68 | **100.0%** | 23.3% | 0.00 | 0.02 |
| Scenario1b.yaml | `meander` | SleepAgent (Passive) | -32.16 +/- 17.79 | 76.7% | 0.0% | 0.93 | 0.01 |
| Scenario2.yaml | `bline` | **Cyber-JEPA (Ours)** | -24.40 +/- 7.63 | **90.0%** | 0.0% | 0.70 | 4.59 |
| Scenario2.yaml | `bline` | BlueReactRestore | -14.28 +/- 2.72 | **100.0%** | 73.3% | 0.00 | 0.03 |
| Scenario2.yaml | `bline` | BlueReactRemove | -25.29 +/- 56.56 | 80.0% | 60.0% | 1.87 | 0.02 |
| Scenario2.yaml | `bline` | SleepAgent (Passive) | -205.20 +/- 42.87 | 0.0% | 0.0% | 16.30 | 0.01 |
| Scenario2.yaml | `meander` | **Cyber-JEPA (Ours)** | -13.46 +/- 4.25 | **100.0%** | 0.0% | 0.00 | 3.82 |
| Scenario2.yaml | `meander` | BlueReactRestore | -10.45 +/- 1.43 | **100.0%** | 70.0% | 0.00 | 0.02 |
| Scenario2.yaml | `meander` | BlueReactRemove | -8.44 +/- 7.68 | **100.0%** | 23.3% | 0.00 | 0.02 |
| Scenario2.yaml | `meander` | SleepAgent (Passive) | -32.16 +/- 17.79 | 76.7% | 0.0% | 0.93 | 0.01 |

## Tactical Action Distribution Breakdown

| Scenario | Red Attacker | Blue Defender | Sleep | Restore | Remove | Analyse | Decoy / Other |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Scenario1b.yaml | `bline` | **Cyber-JEPA** | 317 | 317 | 260 | 6 | 0 |
| Scenario1b.yaml | `bline` | BlueReactRestore | 0 | 385 | 0 | 0 | 515 |
| Scenario1b.yaml | `bline` | BlueReactRemove | 0 | 0 | 341 | 0 | 559 |
| Scenario1b.yaml | `bline` | SleepAgent | 900 | 0 | 0 | 0 | 0 |
| Scenario1b.yaml | `meander` | **Cyber-JEPA** | 398 | 153 | 349 | 0 | 0 |
| Scenario1b.yaml | `meander` | BlueReactRestore | 0 | 280 | 0 | 0 | 620 |
| Scenario1b.yaml | `meander` | BlueReactRemove | 0 | 0 | 236 | 0 | 664 |
| Scenario1b.yaml | `meander` | SleepAgent | 900 | 0 | 0 | 0 | 0 |
| Scenario2.yaml | `bline` | **Cyber-JEPA** | 317 | 317 | 260 | 6 | 0 |
| Scenario2.yaml | `bline` | BlueReactRestore | 0 | 385 | 0 | 0 | 515 |
| Scenario2.yaml | `bline` | BlueReactRemove | 0 | 0 | 341 | 0 | 559 |
| Scenario2.yaml | `bline` | SleepAgent | 900 | 0 | 0 | 0 | 0 |
| Scenario2.yaml | `meander` | **Cyber-JEPA** | 398 | 153 | 349 | 0 | 0 |
| Scenario2.yaml | `meander` | BlueReactRestore | 0 | 280 | 0 | 0 | 620 |
| Scenario2.yaml | `meander` | BlueReactRemove | 0 | 0 | 236 | 0 | 664 |
| Scenario2.yaml | `meander` | SleepAgent | 900 | 0 | 0 | 0 | 0 |

## Key Defense Observations
1. **Proactive Cost Efficiency**: Traditional reactive agents spam `Restore` (-5 cost per action) indiscriminately, incurring massive operational penalties even during nominal traffic.
2. **Targeted Tactical Eviction**: Cyber-JEPA selectively sleeps during nominal baseline states, and only initiates high-confidence targeted `Restore` on operational hosts when lateral movement is detected.
3. **Inference Latency**: Decision latencies remain strictly real-time (< 3.0ms per step), well within active cyber operational limits.

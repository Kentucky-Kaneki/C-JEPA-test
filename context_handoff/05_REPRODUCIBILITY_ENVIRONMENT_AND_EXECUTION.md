# 05 — Reproducibility, Environment Setup & Execution Guide

> **Document Purpose**: Step-by-step instructions for environment configuration, deterministic data generation, GPU training, and automated verification.

---

## 1. Dual-Environment Architecture

Because CybORG 3.1 simulation dependencies (e.g., PettingZoo, Gym, Ray) have strict legacy dependency bounds while modern JEPA training requires PyTorch 2.x and CUDA 12.1+, the repository uses a **dual virtual-environment design**:

```
+-----------------------------------------------------------------------------------------+
| .venv-ml  (Machine Learning & Deep Learning Runtime)                                    |
| - Python: 3.10 or 3.11                                                                  |
| - PyTorch: 2.1.0+cu121 (CUDA 12.1 support)                                              |
| - Libraries: torchvision, torchaudio, scikit-learn, pandas, pyarrow, fastparquet, scipy |
| - Used For: Training Cyber-JEPA models, linear probing, rank diagnostics, sweeps        |
+-----------------------------------------------------------------------------------------+

+-----------------------------------------------------------------------------------------+
| .venv-sim (CybORG 3.1 Simulation & Data Collection Runtime)                             |
| - Python: 3.10 or 3.11                                                                  |
| - Simulation: CybORG 3.1, gym==0.21.0, pettingzoo, ray                                 |
| - Used For: Running CybORG gym episodes and collecting raw telemetry shards             |
+-----------------------------------------------------------------------------------------+
```

---

## 2. Hardware & OS Prerequisites

- **Operating System**: Windows 10/11 or Ubuntu 20.04/22.04 LTS
- **GPU**: NVIDIA GPU with $\ge 6\text{GB}$ VRAM (e.g., RTX 3060, RTX 4080, A100) with CUDA 12.1+
- **RAM**: Minimum 16GB system memory
- **Disk**: $\sim 10\text{GB}$ free space for 90k transition dataset shards and run checkpoints

---

## 3. Environment Setup Commands

### Setting up `.venv-ml` (PyTorch & Training)
```powershell
# In PowerShell:
python -m venv .venv-ml
.\.venv-ml\Scripts\Activate.ps1

# Install PyTorch with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install dependencies and editable package
pip install -r requirements.txt
pip install -e .
```

### Setting up `.venv-sim` (CybORG Simulation)
```powershell
python -m venv .venv-sim
.\.venv-sim\Scripts\Activate.ps1
pip install -r requirements/sim_requirements.txt
```

---

## 4. Deterministic Seeding & Split Contracts

Reproducibility is strictly enforced in `src/cyber_jepa/utils/reproducibility.py`:

```python
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

### Seed Separation Scheme
```yaml
split_seed: 42             # Partitions 300 split groups into train/val/test
cohort_seed: 4201          # Samples 2,000 holdout transitions
training_subset_seed: 7301 # Shuffles training trajectories
bootstrap_seed: 9901       # Controls 10,000 paired bootstrap resamples
model_seed: [101..105]     # 5 distinct seeds per model configuration
dataloader_seed: [201..205]# Controls batch shuffling order via torch.Generator
```

---

## 5. Step-by-Step Execution Workflow

### Step 1: Run Unit & Integration Test Suite
To run the full 45-test suite (including CybORG simulation and telemetry collection tests), invoke pytest in `.venv-sim`:
```powershell
.\.venv-sim\Scripts\Activate.ps1
pytest -v tests/
# Output: 45 passed across all 12 test modules
```

To run model & training pipeline unit tests only (without CybORG simulator dependencies), invoke pytest in `.venv-ml`:
```powershell
.\.venv-ml\Scripts\Activate.ps1
pytest -v tests/test_models.py tests/test_representations.py tests/test_training_pipeline.py tests/test_evaluation_and_selection.py tests/test_phase2_information_equivalence.py tests/test_legacy.py
```

### Step 2: Telemetry Data Collection (Optional if `data/` already exists)
Collect 90,000 transitions across 1,800 trajectories:
```powershell
.\.venv-sim\Scripts\Activate.ps1
python scripts/run_collection.py --output_dir data/raw_shards --num_trajectories 1800
```

### Step 3: Build Fixed Evaluation Cohort
Generate the persistent SHA-256 verified holdout cohort:
```powershell
.\.venv-ml\Scripts\Activate.ps1
python scripts/build_phase3_cohort.py --data_dir data/raw_shards --output_dir experiments/phase3
```

### Step 4: Run Single-Seed Diagnostic Pilot
Run a fast 1-seed sanity check before launching large sweeps:
```powershell
python scripts/run_phase3_pilot.py --config configs/phase3_pilot.yaml
```

### Step 5: Execute Full 45-Run Phase 3 Sweep
Train all 9 configurations across 5 model seeds:
```powershell
python scripts/run_phase3_sweep.py --output_dir experiments/phase3 --runs_dir runs/phase3
```

### Step 6: Compute Paired Bootstrap Statistics & Generate Report
Compute 95% confidence intervals, bootstrap p-values, and generate `PHASE3_REPORT.md`:
```powershell
python scripts/analyze_phase3_results.py --results_file experiments/phase3/phase3_sweep_results.json
```

---

## 6. Common Gotchas & Troubleshooting

1. **Deterministic Algorithm Exceptions**:
   - Some CUDA operations (like specific scatter/gather kernels in sparse attention) may trigger `RuntimeError: deterministic mode is on`. Cyber-JEPA handles this by configuring numeric tolerances rather than requiring unsafe bitwise CUDA identity.
2. **GPU Out-of-Memory (OOM)**:
   - Batch size 64 fits comfortably in $\le 4\text{GB}$ VRAM. If running on low-memory GPUs ($<4\text{GB}$), reduce `batch_size: 32` in `configs/phase3_sweep.yaml`.
3. **Path Separators on Windows**:
   - Always use `pathlib.Path` or raw string paths (`r"..."`) to avoid Windows backslash escape issues.

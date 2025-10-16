# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Filo-Priori V4** is a machine learning-based test prioritization system that predicts which tests are most likely to fail and ranks them to optimize early fault detection (APFD metric). This is a radical simplification from V3, focusing on pure BCE classification with a minimal feature set after V3's multi-task learning approach proved unstable.

## Core Architecture

### Module Structure

- `filo_priori/data/processor.py` - Feature extraction and data loading pipeline
  - `DataProcessor`: Main orchestrator for loading CSV data and building features
  - `TemporalFeatureBuilder`: Stateful builder that creates 4 temporal features (last_run, fail_count, avg_duration, run_frequency)
  - `FiloDataset`: PyTorch Dataset wrapper

- `filo_priori/models/mlp.py` - Lightweight 2-layer MLP classifier (772→256→128→1)

- `filo_priori/utils/training.py` - Training loop with early stopping
  - Uses pure BCE loss (no ranking loss unlike V3)
  - Monitors `val_auprc` for early stopping (better than accuracy for imbalanced data)
  - Cosine LR schedule with warmup

- `filo_priori/utils/inference.py` - Prioritization strategies and APFD metrics
  - Three strategies: probability-based, diversity-based, hybrid (default)
  - `ProbabilityCalibrator`: Temperature scaling + logistic regression
  - APFD and APFDC (cost-aware) calculation

### Feature Engineering (772 dimensions)

**Semantic (768-dim)**: SBERT embeddings using `all-mpnet-base-v2` on concatenated `TE_Summary` + `TC_Steps`

**Temporal (4-dim)**:
1. `last_run`: Binary flag for previous build presence
2. `fail_count`: Log1p of historical failure count
3. `avg_duration`: Log1p of average execution time
4. `run_frequency`: Recency-weighted frequency (1/(1+days_since_last_run))

**Important**: Temporal features are built statefully in chronological order to ensure causality (past features only use past information).

### Key Design Decisions

1. **Pure BCE Classification**: V3's multi-task learning (BCE + Ranking) caused gradient instability (15-93 vs V4's <5). V4 uses only BCE with conservative pos_weight capping at 5.0x.

2. **Minimal Features**: Removed phylogenetic trees, JDk/SFDk churn, and tree intrinsic features from V3 (785→772 dims).

3. **Class Imbalance Handling**:
   - WeightedRandomSampler targeting 20% positive samples per batch
   - Conservative pos_weight cap (max 5.0x)
   - Monitor AUPRC instead of accuracy for early stopping

4. **Probability Calibration**: Stacked temperature scaling + logistic regression for reliable APFD optimization.

## Development Commands

### Installation
```bash
pip install -r requirements.txt
```

### Running Experiments

**Smoke test** (quick validation with limited data):
```bash
python scripts/core/run_experiment_server.py --smoke-train 20 --smoke-test 10 --smoke-epochs 5
```

**Full test** (production run):
```bash
python scripts/core/run_experiment_server.py --full-test
```

**Custom configuration**:
```bash
python scripts/core/run_experiment_server.py --config custom_config.yaml --device cuda
```

### Configuration

All hyperparameters live in `config.yaml`. Key settings:
- `learning_rate: 5e-4` (conservative)
- `batch_size: 16`
- `early_stopping_patience: 8`
- `monitor_metric: val_auprc` (critical for imbalanced data)
- `label_smoothing: 0.05` (prevents overconfidence)

## Data Pipeline

1. Load CSV from `datasets/train.csv` (~1.7GB) or `datasets/test_full.csv` (~581MB)
2. Remove duplicates on (Build_ID, TC_Key, TE_Key)
3. Normalize labels: "fail"/"failed"/"failure" → 1.0, else → 0.0
4. Build temporal features incrementally per build (stateful, chronological)
5. Encode semantic features via SBERT
6. Split 85/15 train/val (deterministic via seed)
7. Train with WeightedRandomSampler for class balance

## Expected Outputs

**Metrics** (`results/metrics.json`):
- Classification: Accuracy ≥0.90, Precision ≥0.15, Recall ≥0.50, AUPRC ≥0.20
- APFD: Mean ≥0.70, ≥70% builds with APFD ≥0.6

**Files**:
- `models/best_model.pth` - Best checkpoint (monitored on val_auprc)
- `results/training_history.json` - Per-epoch metrics
- `results/prioritized_hybrid.csv` - Ranked test cases with probabilities and scores

## Critical Constraints

1. **Temporal Causality**: When modifying `TemporalFeatureBuilder`, ensure features only use information from previous builds (no future leakage).

2. **Class Imbalance**: Dataset is ~95% negative. Always evaluate using AUPRC, F1, precision/recall, not just accuracy.

3. **No Ranking Loss**: Do not reintroduce ranking losses from V3. They cause gradient explosion with this extreme imbalance.

4. **Reproducibility**: All experiments must set random seeds (random, numpy, torch) for deterministic results.

5. **Calibration Required**: APFD optimization requires calibrated probabilities. Always calibrate on validation set before test inference.

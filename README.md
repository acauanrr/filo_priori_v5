# Filo-Priori V5

**ML-based test prioritization system using BGE embeddings and SAINT transformer for predicting test failures and optimizing test execution order for early fault detection.**

## What It Does

Given a test suite, Filo-Priori V5:
1. Predicts which tests are likely to fail
2. Ranks tests to maximize early fault detection (APFD metric)
3. Outputs a prioritized test execution order

**Target**: APFD ≥ 0.70 (finds 70% of failures in first 30% of tests)

**Latest Results (Execution 006)**:
- Mean APFD: **0.574** (36.1% builds with APFD ≥ 0.7)
- Test AUPRC: **0.048**
- Recall: **0.815** | Precision: **0.038**
- Model: **~69.8M parameters** (4 layers, 8 heads, 1024D embedding)

## What's New in V5?

### Major Upgrades from V4

**Embeddings**: SBERT (768D) → **BGE-large-en-v1.5 (1024D)**
- State-of-the-art embeddings from BAAI
- **No PCA** - uses full 1024 dimensions
- StandardScaler normalization only

**Model**: Simple MLP → **SAINT Transformer**
- **Self-Attention**: Captures feature interactions within samples
- **Intersample Attention**: Novel mechanism that learns from other samples in the batch
- **4 layers, 8 attention heads, ~69.8M parameters**
- **1024D embedding dimension** (uses full BGE dimension)
- Specialized for tabular data classification

**Training**:
- Proper PyTorch training loop with early stopping (patience=3)
- Cosine learning rate schedule with warmup (3 epochs)
- Advanced class imbalance handling (pos_weight=15.0, 40% positive sampling)
- Label smoothing (0.01) for better generalization
- **Probability calibration** (isotonic regression on validation set)

**Features**:
- **Semantic-focused**: Currently uses 1024D BGE embeddings only
- **Additional features**: 7 commit features + 3 categorical features
- **Total**: 1031D continuous + 3D categorical
- **Note**: Temporal features (last_run, fail_count, etc.) are currently DISABLED

## Architecture

### Training Pipeline

1. **Load & Parse Commits**: Load train/test CSV, remove duplicates, normalize labels
2. **Build Text Semantic**: Concatenate TE_Summary + TC_Steps
3. **Generate BGE Embeddings**:
   - Encode with BGE-large-en-v1.5 → 1024D
   - Apply StandardScaler (NO PCA)
4. **Build Tabular Features**:
   - BGE embeddings: 1024D
   - Commit features: 7 numerical (n_msgs, n_apis, n_issues, n_modules, n_packages, n_flags, n_errors)
   - Categorical features: 3 (CR_Resolution, CR_Component_Name, CR_Type)
   - **Total**: 1031D continuous + 3D categorical
   - **Note**: Temporal features currently DISABLED
5. **Split & Prepare Data**:
   - Train/Val split: 85% / 15% (stratified)
   - WeightedRandomSampler (target 40% positive samples per batch)
6. **Train SAINT Transformer**:
   - Embedding: 1031D → 1024D
   - 4 SAINT blocks (self-attention + intersample attention)
   - Classification head: 1024D → 1D (binary)
   - Loss: BCE with pos_weight=15.0
   - Optimizer: AdamW (lr=3e-4, weight_decay=0.1)
   - LR schedule: Cosine with 3-epoch warmup
   - Early stopping: val_auprc (patience=3, min_delta=0.01)
7. **Calibrate Probabilities**: Isotonic regression on validation set
8. **Evaluate & Save**: Calculate metrics, APFD per build, save all artifacts

### Inference
- **Hybrid prioritization strategy** (default)
- Combines failure probability + test diversity
- Calibrated probabilities for reliable ranking
- Three strategies: probability-based, diversity-based, hybrid

## Features

**Semantic (1024-dim)**: BGE-large embeddings from test summaries and steps
- Concatenates `TE_Summary` + `TC_Steps`
- Encoded with `BAAI/bge-large-en-v1.5`
- Normalized with StandardScaler

**Commit Features (7-dim)**: Numerical features from commit metadata
- `commit_n_msgs`: Number of commit messages
- `commit_n_apis`: Number of API changes
- `commit_n_issues`: Number of issues referenced
- `commit_n_modules`: Number of modules changed
- `commit_n_packages`: Number of packages changed
- `commit_n_flags`: Number of flags
- `commit_n_errors`: Number of errors

**Categorical Features (3-dim)**: Encoded with LabelEncoder
- `CR_Resolution`: Change request resolution status
- `CR_Component_Name`: Component name
- `CR_Type`: Change request type

**Temporal Features (DISABLED)**: Currently not in use
- `last_run`: Presence in previous build
- `fail_count`: Historical failure count (log1p)
- `avg_duration`: Average execution time (log1p)
- `run_frequency`: Recency-weighted execution frequency

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Smoke Test (quick validation - 10-20 min)
```bash
./run_smoke_test.sh
```

Or manually:
```bash
cd filo_priori
python scripts/core/run_experiment_server.py --smoke-train 100 --smoke-test 50
```

### Full Test (production run - 45-90 min GPU, 2-4 hours CPU)
```bash
./run_full_test.sh
```

Or manually:
```bash
cd filo_priori
python scripts/core/run_experiment_server.py --full-test --device cuda
```

## Results

### Target Metrics
**Classification**:
- Accuracy: ≥0.90
- Precision: ≥0.15
- Recall: ≥0.50
- AUPRC: ≥0.20

**Prioritization**:
- Mean APFD: ≥0.70
- 70%+ builds with APFD ≥0.6

### Latest Results (Execution 006 - Full Test)

**Dataset**:
- Train: 63,532 samples (2.60% failures)
- Val: 9,530 samples
- Test: 28,859 samples (3.20% failures, 277 builds)

**Training**:
- Best epoch: 2/30
- Best val_auprc: 0.0758
- Training time: ~5 hours (GPU)

**Classification Metrics**:
- AUPRC: **0.048** (❌ below target 0.20)
- Precision: **0.038** (❌ below target 0.15)
- Recall: **0.815** (✅ above target 0.50)
- F1: **0.072**
- Accuracy: **0.330** (❌ below target 0.90)
- AUC: **0.613**

**Prioritization Metrics**:
- Mean APFD: **0.574** (❌ below target 0.70)
- Median APFD: **0.541**
- Builds with APFD ≥ 0.7: **100/277 (36.1%)** (❌ below target 70%)
- Builds with APFD < 0.5: **92/277 (33.2%)**

**Probability Analysis**:
- Failures: mean=0.0343, std=0.0310
- Passes: mean=0.0257, std=0.0275
- Discrimination ratio: **1.34x** (weak separation)

**Status**: 🔴 Below target - model needs improvement

## Project Structure

```
filo_priori_v5/
├── config.yaml                           # Hyperparameters (SAINT + training)
├── filo_priori/
│   ├── data_processing/
│   │   ├── 01_parse_commit.py           # Commit parsing
│   │   ├── 02_build_text_semantic.py    # Text concatenation
│   │   └── 03_embed_sbert.py            # BGE-large embeddings (NO PCA)
│   ├── models/
│   │   ├── __init__.py
│   │   └── saint.py                     # SAINT transformer implementation
│   ├── utils/
│   │   ├── features.py                  # Temporal feature builder
│   │   ├── dataset.py                   # PyTorch datasets
│   │   ├── saint_trainer.py             # Training loop + early stopping
│   │   └── apfd_per_build.py            # APFD calculation
│   └── scripts/core/
│       └── run_experiment_server.py      # Main experiment runner
├── datasets/                             # Train/test CSVs
├── models/                               # Saved checkpoints
├── results/                              # Metrics and rankings
├── run_smoke_test.sh                     # Quick validation script
└── run_full_test.sh                      # Production run script
```

## Key Design Decisions

**Why SAINT?**
- Specialized transformer for tabular data
- **Intersample attention** improves regularization by learning from other samples in batch
- Proven better than MLPs for complex tabular classification
- Handles high-dimensional features (1031D) effectively

**Why BGE over SBERT?**
- State-of-the-art embeddings (1024D vs 768D SBERT)
- Better semantic representation
- No dimensionality reduction needed (removed PCA bottleneck)
- Full 1024D dimension used in model embedding

**Why Full 1024D Embedding in SAINT?**
- Previous versions reduced embeddings to 128D
- V5 uses **full 1024D** to preserve semantic information
- Increases model size (~69.8M params) but preserves representation quality
- Trade-off: Larger model, slower training, but richer features

**Class Imbalance Handling**:
- WeightedRandomSampler (targets 40% positive samples per batch)
- pos_weight=15.0 in BCE loss (aggressive weighting for failures)
- Label smoothing=0.01 for better generalization
- Early stopping on AUPRC (better than accuracy for imbalanced data)

**Probability Calibration**:
- Isotonic regression on validation set
- Improves reliability of probability estimates
- Critical for APFD optimization

**Temporal Features**:
- Currently DISABLED in favor of semantic-only approach
- Historical features (last_run, fail_count, etc.) can be re-enabled in config
- Decision based on experimentation showing semantic features as primary signal

## Configuration

Edit `config.yaml` for hyperparameters:

```yaml
# SAINT Model
saint:
  num_continuous: 1024        # BGE embedding dimension (auto-adjusted at runtime)
  num_categorical: 0          # Currently not using categoricals in SAINT
  embedding_dim: 1024         # Full BGE dimension (no reduction)
  num_layers: 4               # Transformer layers
  num_heads: 8                # Attention heads
  dropout: 0.3                # Regularization
  use_intersample: true       # Key SAINT feature

# Features
features:
  use_semantic: true          # BGE embeddings
  semantic_model: "BAAI/bge-large-en-v1.5"
  semantic_dim: 1024
  use_temporal: false         # Currently DISABLED
  total_dim: 1024            # Semantic only

# Training
training:
  num_epochs: 30
  learning_rate: 0.0003       # Conservative for stability
  weight_decay: 0.1           # Strong regularization
  batch_size: 16
  patience: 3                 # Aggressive early stopping
  min_delta: 0.01            # Minimum improvement threshold
  monitor_metric: val_auprc
  label_smoothing: 0.01
  pos_weight: 15.0           # Aggressive failure weighting
  target_positive_fraction: 0.40  # High positive exposure
  use_calibration: true      # Isotonic regression
```

## Outputs

After running an experiment, results are saved in `filo_priori/results/execution_XXX/`:

- `best_model.pth`: SAINT checkpoint (best val_auprc epoch)
- `training_history.json`: Per-epoch training metrics (loss, accuracy, precision, recall, F1, AUC, AUPRC)
- `metrics.json`: Complete evaluation metrics + dataset stats
- `prioritized_hybrid.csv`: Ranked test cases with probabilities and scores
- `apfd_per_build.csv`: APFD calculated per build (277 builds in full test)
- `summary.txt`: Human-readable experiment summary
- `config.json`: Complete experiment configuration
- `calibrator.pkl`: Isotonic regression calibrator
- `feature_builder.pkl`: Feature engineering artifacts (scalers, encoders)
- `embedder/scaler.pkl`: BGE StandardScaler (1024D normalization)

## Performance Comparison

| Metric | V4 (MLP) | V5 (SAINT) | Change |
|--------|----------|------------|--------|
| Embedding Model | SBERT (all-mpnet-base-v2) | BGE-large-en-v1.5 | Upgraded |
| Embedding Dim | 768D | 1024D | +33% |
| Embedding Reduction | Yes (→128D) | No (full 1024D) | Preserved |
| Total Features | 772D (768 + 4 temporal) | 1031D (1024 + 7 commit + 3 cat) | +33% |
| Model Type | 2-layer MLP | 4-layer SAINT Transformer | Advanced |
| Model Embedding | 128D | 1024D | 8x larger |
| Parameters | ~250K | ~69.8M | **279x larger** |
| Training Time | ~15-30 min | ~5 hours (GPU) | 10x slower |
| Intersample Attention | ❌ | ✅ | Novel |
| Calibration | ❌ | ✅ (Isotonic) | Added |
| Temporal Features | ✅ (4 features) | ❌ (Disabled) | Removed |

## Current Challenges & Future Work

### Current Issues (Execution 006)

**Below-target performance**:
- APFD: 0.574 vs target 0.70 (❌ -18% gap)
- AUPRC: 0.048 vs target 0.20 (❌ -76% gap)
- Precision: 0.038 vs target 0.15 (❌ -75% gap)
- Weak probability discrimination: 1.34x (failures vs passes)

**Possible causes**:
1. **Overfitting**: Best epoch is 2/30, suggests model fits validation set quickly then degrades
2. **Extreme class imbalance**: 2.6% train failures, 3.2% test failures (~38:1 ratio)
3. **Missing temporal signal**: Temporal features disabled, may lose historical patterns
4. **Model capacity**: 69.8M params may be too large for limited failure samples (~1,654 train failures)
5. **Feature quality**: Commit features may not be predictive enough

### Future Work

**Short-term experiments**:
1. **Re-enable temporal features**: Test if historical patterns improve APFD
2. **Reduce model size**: Try 2-layer SAINT or reduce embedding_dim to 512D
3. **Ensemble methods**: Combine multiple models for better generalization
4. **Feature engineering**: Add test execution history, code coverage features
5. **Threshold optimization**: Find optimal classification threshold for F1/precision

**Long-term improvements**:
1. **Data augmentation**: Collect more failure samples or use SMOTE carefully
2. **Multi-task learning**: Add auxiliary tasks (test duration prediction, etc.)
3. **Domain adaptation**: Transfer learning from other test datasets
4. **Explainability**: Add SHAP/attention analysis to understand predictions

## Troubleshooting

### CUDA Out of Memory
```yaml
# Reduce batch size in config.yaml
training:
  batch_size: 8  # Was 16
```

Or reduce embedding dimension:
```yaml
saint:
  embedding_dim: 512  # Was 1024
```

### Slow Training on CPU
- Expected: ~8-12 hours for full test (vs ~5 hours on GPU)
- Use GPU if available (CUDA 11.8+ recommended)
- Or reduce to smoke test for quick validation

### Model Not Improving
- Check `training_history.json` for overfitting (val metrics degrading)
- Try reducing model size or increasing regularization
- Consider re-enabling temporal features

### Missing Dependencies
```bash
pip install -r requirements.txt
```

## Citations

If you use Filo-Priori V5, please cite:

**SAINT Paper**:
```
Somepalli et al., "SAINT: Improved Neural Networks for Tabular Data via
Row Attention and Contrastive Pre-Training", 2021
```

**BGE Embeddings**:
```
BAAI, "BGE: General Embedding Model", 2024
https://huggingface.co/BAAI/bge-large-en-v1.5
```

## License

MIT License - See LICENSE file for details

---

**Version**: 5.0
**Last Updated**: 2025-10-19
**Latest Execution**: 006 (2025-10-18)
**Status**: 🔴 Under Development - Performance Below Target

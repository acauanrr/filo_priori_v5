# Filo-Priori V5 🚀

**ML-based test prioritization system using BGE embeddings and SAINT transformer for predicting test failures and optimizing test execution order for early fault detection.**

## What It Does

Given a test suite, Filo-Priori V5:
1. Predicts which tests are likely to fail
2. Ranks tests to maximize early fault detection (APFD metric)
3. Outputs a prioritized test execution order

**Target**: APFD ≥ 0.70 (finds 70% of failures in first 30% of tests)

## What's New in V5?

### 🎯 Major Upgrades from V4

**Embeddings**: SBERT → **BGE-large-en-v1.5**
- 384D → **1024D** embeddings
- **No PCA** - uses full embedding dimensions
- State-of-the-art semantic representation

**Model**: Simple MLP → **SAINT Transformer**
- **Self-Attention**: Captures feature interactions within samples
- **Intersample Attention**: Novel mechanism that learns from other samples in the batch
- 6 layers, 8 attention heads, ~1.86M parameters
- Specialized for tabular data classification

**Training**:
- Proper PyTorch training loop with early stopping
- Cosine learning rate schedule with warmup
- Advanced class imbalance handling
- Label smoothing for better generalization

## Architecture

### Training Pipeline
1. **Text Embeddings**: BGE-large-en-v1.5 (1024D)
2. **Temporal Features**: 4 stateful features (last_run, fail_count, avg_duration, run_frequency)
3. **Total Features**: 1028D (1024 semantic + 4 temporal)
4. **Model**: SAINT Transformer
   - Embedding layer: 1028D → 128D
   - 6 SAINT blocks (self-attention + intersample attention)
   - Classification head: 128D → 1D (binary)
5. **Loss**: BCE with pos_weight=5.0 for class imbalance
6. **Optimizer**: AdamW with cosine LR schedule
7. **Early Stopping**: Based on val_auprc (patience=8)

### Inference
- **Hybrid prioritization strategy** (default)
- Combines failure probability + test diversity
- Calibrated probabilities for reliable ranking
- Three strategies: probability-based, diversity-based, hybrid

## Features

**Semantic (1024-dim)**: BGE-large embeddings from test summaries and steps

**Temporal (4-dim)**:
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

## Expected Results

**Classification**:
- Accuracy: ≥0.90
- Precision: ≥0.15
- Recall: ≥0.50
- AUPRC: ≥0.20

**Prioritization**:
- Mean APFD: ≥0.70
- 70%+ builds with APFD ≥0.6

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
- Handles high-dimensional features (1028D) effectively

**Why BGE over SBERT?**
- State-of-the-art embeddings (1024D vs 384D)
- Better semantic representation
- No dimensionality reduction needed (removed PCA bottleneck)

**Class Imbalance Handling**:
- WeightedRandomSampler (targets 20% positive samples per batch)
- pos_weight=5.0 in BCE loss
- Label smoothing=0.05 for better generalization
- Early stopping on AUPRC (better than accuracy for imbalanced data)

**Temporal Causality**:
- Features built chronologically per build
- Ensures no future information leakage
- Stateful feature builder maintains historical context

## Configuration

Edit `config.yaml` for hyperparameters:

```yaml
# SAINT Model
saint:
  embedding_dim: 128
  num_layers: 6
  num_heads: 8
  use_intersample: true  # Key SAINT feature

# Training
training:
  num_epochs: 30
  learning_rate: 5e-4
  batch_size: 16
  patience: 8
  monitor_metric: val_auprc
  label_smoothing: 0.05
  pos_weight: 5.0
```

## Outputs

After running an experiment, results are saved in `filo_priori/results/execution_XXX/`:

- `best_model.pth`: SAINT checkpoint (best val_auprc)
- `training_history.json`: Per-epoch training metrics
- `metrics.json`: Complete evaluation metrics
- `prioritized_hybrid.csv`: Ranked test cases with probabilities
- `apfd_per_build.csv`: APFD calculated per build
- `summary.txt`: Human-readable experiment summary
- `config.json`: Experiment configuration
- `feature_builder.pkl`: Feature engineering artifacts
- `embedder/scaler.pkl`: BGE StandardScaler

## Performance Comparison

| Metric | V4 (MLP) | V5 (SAINT) | Improvement |
|--------|----------|------------|-------------|
| Embedding Dim | 768D (SBERT) | 1024D (BGE) | +33% |
| Total Features | 772D | 1028D | +33% |
| Model Type | 2-layer MLP | 6-layer Transformer | Advanced |
| Parameters | ~250K | ~1.86M | 7x larger |
| Training Time | ~15-30 min | ~45-90 min (GPU) | 2-3x slower |
| Intersample Attention | ❌ | ✅ | Novel |

## Troubleshooting

### CUDA Out of Memory
```yaml
# Reduce batch size in config.yaml
training:
  batch_size: 8  # Was 16
```

### Slow Training on CPU
- Expected: ~2-4 hours for full test
- Use GPU if available
- Or reduce to smoke test for validation

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
**Last Updated**: 2025-10-16
**Status**: Production Ready ✅

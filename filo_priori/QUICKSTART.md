# Filo-Priori V5 - Quick Start Guide

## ✅ Implementation Complete

All pipeline components have been implemented and integrated into a single orchestrator script.

## 📁 Project Structure

```
filo_priori_v5/
├── data_processing/
│   ├── __init__.py
│   ├── 01_parse_commit.py          # Structured commit parsing
│   ├── 02_build_text_semantic.py   # Text semantic construction
│   └── 03_embed_sbert.py            # SBERT embeddings with PCA
├── utils/
│   ├── __init__.py
│   ├── features.py                  # Tabular feature engineering
│   ├── dataset.py                   # Balanced sampling
│   └── model.py                     # Deep MLP with BatchNorm
├── scripts/
│   ├── __init__.py
│   └── core/
│       ├── __init__.py
│       └── run_experiment_server.py # Complete orchestrator
├── configs/
│   └── config.yaml                  # Configuration
├── README.md
└── QUICKSTART.md
```

## 🚀 How to Run

### Smoke Test (Quick validation with 100 builds)
```bash
cd /home/acauan/ufam/iats/sprint_07/filo_priori_v5

python scripts/core/run_experiment_server.py \
    --smoke-train 100 \
    --smoke-test 50 \
    --smoke-epochs 20
```

### Full Test (Production run with all data)
```bash
python scripts/core/run_experiment_server.py --full-test
```

## 📊 Pipeline Steps

The orchestrator (`run_experiment_server.py`) executes 7 steps:

1. **Load and Parse Commits**
   - Extract structured fields: APIs, errors, issues, modules, packages, flags
   - Generate commit_text with prioritized content

2. **Build text_semantic**
   - Combine TE_Summary + TC_Steps + commit_text
   - Clean HTML, URLs, normalize whitespace
   - Filter ambiguous labels (Blocked, Pending, etc.)

3. **Generate SBERT Embeddings**
   - Model: `paraphrase-multilingual-MiniLM-L12-v2` (384D)
   - PCA projection: 384D → 128D
   - StandardScaler normalization

4. **Build Tabular Features**
   - Numerical: commit counts (7 features)
   - Categorical: CR_Resolution, CR_Component_Name, CR_Type
   - Concatenate with embeddings → continuous features

5. **Train Deep MLP**
   - Architecture: [512, 256, 128] with BatchNorm + Dropout(0.3)
   - Loss: BCE with pos_weight=5.0 + label_smoothing=0.01
   - Balanced sampler: 30% positive per batch
   - Early stopping on val_auprc (patience=15)

6. **Evaluate**
   - Calculate APFD and APFDc
   - Test metrics: AUPRC, Precision, Recall, F1, Accuracy
   - Probability distribution analysis

7. **Save Results**
   - `results/metrics.json` - Complete metrics and history
   - `results/prioritized_hybrid.csv` - Predictions with ranks

## 🎯 Expected Outputs

After running, a new directory `results/execution_XXX/` will be created with:

### Core Files

- **metrics.json**: Complete experiment metrics
  - `metadata`: Experiment info (timestamp, type, dataset stats)
  - `metrics`: APFD, APFDc, AUPRC, precision, recall, F1, discrimination_ratio
  - `best_epoch`, `best_metrics`: Best model checkpoint info
  - `history`: Epoch-by-epoch training metrics
  - `probability_stats`: Detailed probability distribution analysis

- **config.json**: Experiment configuration
  - All hyperparameters used
  - Dataset paths
  - Model architecture details

- **best_model.pt**: PyTorch model checkpoint
  - Best model state_dict
  - Training metrics
  - Configuration

- **prioritized_hybrid.csv**: Test predictions
  - Build_ID, TC_Key, TE_Test_Result
  - label_binary (ground truth)
  - probability (model prediction)
  - rank (1 = highest priority)
  - priority_score (hybrid: 0.7×prob + 0.3×diversity)

- **training_history.csv**: Epoch-by-epoch metrics
  - Easy to plot with pandas/matplotlib
  - Train/val loss, precision, recall, F1, AUPRC

- **summary.txt**: Human-readable summary
  - Dataset statistics
  - Training details
  - Test results
  - File descriptions

### Artifacts

- **feature_builder.pkl**: Feature engineering artifacts
  - Label encoders for categorical features
  - StandardScaler for numerical features
  - Column mappings

- **embedder/**: SBERT embedder artifacts
  - pca.pkl: PCA projection (384D→128D)
  - scaler.pkl: StandardScaler for embeddings
  - Reusable for inference on new data

## 📈 Success Criteria

| Metric | Target | V4 Baseline |
|--------|--------|-------------|
| APFD | ≥ 0.70 | 0.507 |
| AUPRC | ≥ 0.20 | 0.074 |
| Precision | ≥ 0.15 | 0.032 |
| Recall | ≥ 0.50 | 0.225 |
| Discrimination | ≥ 2.0x | 1.05x |

## 🔧 Configuration

Default hyperparameters (in `run_experiment_server.py`):

```python
DEFAULT_CONFIG = {
    'train_csv': '../filo_priori_v4/datasets/train.csv',
    'test_csv': '../filo_priori_v4/datasets/test_full.csv',
    'output_dir': 'results',
    'sbert_target_dim': 128,
    'sbert_model': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
    'model_hidden_dims': [512, 256, 128],
    'model_dropout': 0.3,
    'lr': 0.001,
    'batch_size': 128,
    'epochs': 30,
    'patience': 15,
    'pos_weight': 5.0,
    'label_smoothing': 0.01,
    'sampler_positive_fraction': 0.3,
    'seed': 42,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}
```

## 📋 Requirements

Ensure the following are installed:
```bash
pip install pandas numpy scikit-learn sentence-transformers torch
```

For CUDA support (recommended):
```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 🐛 Troubleshooting

**Issue**: Import errors
**Solution**: Ensure you're running from the filo_priori_v5 root directory

**Issue**: Dataset not found
**Solution**: Verify filo_priori_v4 datasets exist at `../filo_priori_v4/datasets/`

**Issue**: Out of memory
**Solution**: Reduce `batch_size` in config (try 64 or 32)

**Issue**: CUDA not available
**Solution**: Will automatically fall back to CPU (slower but functional)

## 📊 Comparing Executions

Each experiment creates a separate `execution_XXX` directory, making it easy to compare:

### Quick Comparison (Command Line)
```bash
# Compare APFD scores
jq '.metrics.apfd' results/execution_*/metrics.json

# Compare discrimination ratios
jq '.metrics.discrimination_ratio' results/execution_*/metrics.json

# View all key metrics
for dir in results/execution_*/; do
  echo "=== $(basename $dir) ==="
  jq '.metadata.experiment_type, .metrics | {apfd, auprc, precision, recall, discrimination_ratio}' $dir/metrics.json
done
```

### Python Analysis
```python
import json
import pandas as pd
from pathlib import Path

# Load all executions
results_dir = Path('results')
executions = []

for exec_dir in sorted(results_dir.glob('execution_*')):
    with open(exec_dir / 'metrics.json') as f:
        data = json.load(f)
        executions.append({
            'execution': exec_dir.name,
            'type': data['metadata']['experiment_type'],
            'timestamp': data['metadata']['timestamp'],
            'apfd': data['metrics']['apfd'],
            'apfdc': data['metrics']['apfdc'],
            'auprc': data['metrics']['auprc'],
            'precision': data['metrics']['precision'],
            'recall': data['metrics']['recall'],
            'f1': data['metrics']['f1'],
            'discrimination': data['metrics']['discrimination_ratio']
        })

df = pd.DataFrame(executions)
print(df.to_string(index=False))

# Best APFD
best = df.loc[df['apfd'].idxmax()]
print(f"\nBest APFD: {best['execution']} with {best['apfd']:.4f}")
```

### Files to Compare

- **metrics.json**: All quantitative metrics
- **summary.txt**: Quick human-readable overview
- **training_history.csv**: Plot training curves
- **prioritized_hybrid.csv**: Compare actual rankings

## 📝 Notes

- **Smoke test** runs in ~10-15 minutes on GPU, ~30-40 minutes on CPU
- **Full test** runs in ~2-3 hours on GPU, ~6-8 hours on CPU
- **Each execution** creates a new numbered directory (execution_001, execution_002, etc.)
- Results are saved incrementally (best model checkpoint)
- Early stopping prevents overfitting (monitors val_auprc)
- All experiments are fully reproducible (seed=42)
- **execution_XXX** directories contain everything needed to reproduce/analyze the experiment

## 🎉 Next Steps

After successful smoke test:
1. Analyze `results/metrics.json` to verify improvements
2. Run full test for production evaluation
3. Compare V5 vs V4 results
4. Iterate on hyperparameters if needed

---

**Status**: ✅ Ready to run
**Last updated**: 2025-10-15

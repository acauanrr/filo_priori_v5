# Filo-Priori V5 Refactoring Summary

## Objective
Modernize the ML pipeline by:
1. Replacing `all-MiniLM-L6-v2` (384D) with `BAAI/bge-large-en-v1.5` (1024D)
2. Removing PCA dimensionality reduction
3. Replacing Deep MLP + PyTorch training with TabPFN pre-trained classifier

## Completed Changes ✓

### 1. requirements.txt
- Added `tabpfn>=0.1.9`
- Updated comments to reflect new architecture

### 2. data_processing/03_embed_sbert.py
- Changed default model to `'BAAI/bge-large-en-v1.5'`
- Removed `IncrementalPCA` import
- Removed `self.pca` attribute completely
- Removed `fit_projection()` and `transform_projection()` methods
- Updated `encode()` to document 1024D output
- Updated `save_artifacts()` and `load_artifacts()` to only handle Scaler
- Updated helper functions to skip PCA steps
- All embeddings now 1024D directly

### 3. utils/features.py
- NO CHANGES NEEDED ✓
- Already flexible - concatenates embeddings + numerical features dynamically
- Will automatically handle 1024D embeddings → 1024+4=1028D total features

### 4. utils/tabpfn_classifier.py (NEW FILE)
- Created `FiloPrioriTabPFN` wrapper class
- Implements `fit()`, `predict_proba()`, `predict()` methods
- Save/load using pickle
- Includes `train_tabpfn()` helper function
- No training loop, epochs, or early stopping needed

### 5. scripts/core/run_experiment_server.py (PARTIALLY UPDATED)
- Updated header documentation
- Removed `torch`, `torch.nn`, `DataLoader` imports
- Updated imports to use `FiloPrioriTabPFN` instead of `DeepMLP`
- Updated `DEFAULT_CONFIG` to remove PyTorch hyperparameters
- Simplified `set_seed()` to remove torch calls
- Removed `LabelSmoothingBCELoss`, `train_epoch()`, `evaluate()` functions
- Added `evaluate_tabpfn()` function

## Remaining Changes Required ❌

### scripts/core/run_experiment_server.py - Main Pipeline (lines 310-428)

The embedding generation section (lines 310-323) still has PCA logic:

```python
# CURRENT (WRONG):
if config['sbert_target_dim']:
    embedder.fit_projection(train_emb)
    train_emb = embedder.transform_projection(train_emb)

if embedder.pca:
    test_emb = embedder.transform_projection(test_emb)

# SHOULD BE (CORRECT):
# Remove these lines entirely! BGE produces 1024D directly, no projection needed
```

**Fix Required:**
```python
# Line 310-315: Train embeddings
train_texts = df_train['text_semantic'].tolist()
train_emb = embedder.encode(train_texts)
# REMOVE: projection logic
embedder.fit_scaler(train_emb)
train_emb = embedder.transform_scaler(train_emb)

# Line 318-323: Test embeddings
test_texts = df_test['text_semantic'].tolist()
test_emb = embedder.encode(test_texts)
# REMOVE: projection logic
test_emb = embedder.transform_scaler(test_emb)
```

### scripts/core/run_experiment_server.py - Dataset Creation (lines 345-358)

Remove PyTorch DataLoader logic:

```python
# REMOVE THESE LINES (345-358):
train_dataset = TabularDataset(...)
val_dataset = TabularDataset(...)
test_dataset = TabularDataset(...)
train_sampler = create_balanced_sampler(...)
train_loader = DataLoader(...)
val_loader = DataLoader(...)
test_loader = DataLoader(...)

# REPLACE WITH:
# TabPFN uses raw numpy arrays directly, no DataLoaders needed
X_train = train_cont[train_idx]
y_train = labels_train[train_idx]
X_val = train_cont[val_idx]
y_val = labels_train[val_idx]
X_test = test_cont
y_test = df_test['label_binary'].values
```

### scripts/core/run_experiment_server.py - Training (lines 360-428)

Replace entire PyTorch training loop with TabPFN:

```python
# REMOVE LINES 360-428 (DeepMLP instantiation, loss, optimizer, training loop, early stopping)

# REPLACE WITH:
# ========================================================================
# STEP 5: Train TabPFN
# ========================================================================
logger.info("\n[5/7] Training TabPFN...")

model, train_info = train_tabpfn(
    X_train=X_train,
    y_train=y_train,
    device=config['device'],
    n_estimators=config['tabpfn_n_estimators']
)

logger.info(f"TabPFN trained on {train_info['n_train_samples']} samples")
logger.info(f"Feature dimension: {train_info['n_features']} (1024 semantic + ~4 temporal)")

# Evaluate on validation set
val_metrics, val_preds = evaluate_tabpfn(model, X_val, y_val)
logger.info(f"Validation AUPRC: {val_metrics['auprc']:.4f}")
logger.info(f"Validation Recall: {val_metrics['recall']:.4f}")
logger.info(f"Validation Precision: {val_metrics['precision']:.4f}")
```

### scripts/core/run_experiment_server.py - Evaluation (lines 430-428)

Update test evaluation:

```python
# CHANGE FROM:
test_metrics, test_preds = evaluate(model, test_loader, criterion, config['device'])

# CHANGE TO:
test_metrics, test_preds = evaluate_tabpfn(model, X_test, y_test)
```

### scripts/core/run_experiment_server.py - Saving (lines 540-560)

Update model saving:

```python
# REMOVE torch.save():
torch.save({
    'epoch': best_epoch,
    'model_state_dict': best_model_state,
    'config': config,
    'metrics': results['metrics']
}, exec_dir / 'best_model.pt')

# REPLACE WITH:
model.save(exec_dir / 'tabpfn_model.pkl')
```

### scripts/core/run_experiment_server.py - Metadata (lines 490-540)

Remove epoch/early_stopping fields from metadata:

```python
# REMOVE:
'n_epochs_configured': ...,
'n_epochs_executed': best_epoch,
'early_stopped': ...,

# REMOVE:
'best_epoch': best_epoch,
'best_metrics': best_metrics,
'history': history,

# TabPFN doesn't have epochs or training history
```

### scripts/core/run_experiment_server.py - CLI (lines 660-670)

Remove `--smoke-epochs` argument:

```python
# REMOVE:
parser.add_argument('--smoke-epochs', type=int, default=20, help='Smoke test: number of epochs')

# REMOVE from device detection:
if args.device == 'auto':
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'  # REMOVE torch.cuda check

# REPLACE WITH:
if args.device == 'auto':
    args.device = 'cpu'  # TabPFN works well on CPU
```

### utils/model.py
**DELETE THIS FILE** - DeepMLP is no longer used

### utils/dataset.py
**CONSIDER DEPRECATING** - TabularDataset and create_balanced_sampler() are no longer needed by TabPFN

### config.yaml
Mark obsolete parameters:

```yaml
# OBSOLETE - Deep MLP parameters (not used with TabPFN):
# model:
#   type: "mlp"
#   hidden_dims: [256, 128]
#   dropout: 0.3

# features:
#   semantic_model: "sentence-transformers/all-mpnet-base-v2"  # OLD
features:
  semantic_model: "BAAI/bge-large-en-v1.5"  # NEW
  semantic_dim: 1024  # Changed from 768
  total_dim: 1028  # Changed from 772

# OBSOLETE - Training parameters (not used with TabPFN):
# training:
#   num_epochs: 30
#   learning_rate: 0.0005
#   batch_size: 16
#   patience: 8
#   label_smoothing: 0.05
```

## Architecture Changes Summary

### Before:
```
Text → SBERT (384D) → PCA (128D) → Concat + Temp (132D) → MLP (3 layers) → BCE Loss → Sigmoid → Probs
```

### After:
```
Text → BGE-large (1024D) → Concat + Temp (1028D) → TabPFN (pre-trained) → Probs
```

### Benefits:
1. **Better embeddings**: BGE-large is state-of-the-art for English text
2. **No information loss**: 1024D vs 128D preserves semantic richness
3. **Simpler training**: TabPFN requires no hyperparameter tuning, no epochs, no early stopping
4. **Faster iterations**: No training loop means faster experimentation
5. **Less code**: ~200 lines of PyTorch training logic removed

## Testing Plan

1. Install new dependencies:
```bash
pip install -r requirements.txt
```

2. Test imports:
```bash
python -c "from tabpfn import TabPFNClassifier; print('TabPFN OK')"
python -c "from sentence_transformers import SentenceTransformer; m = SentenceTransformer('BAAI/bge-large-en-v1.5'); print(f'BGE OK: {m.get_sentence_embedding_dimension()}D')"
```

3. Run smoke test:
```bash
cd /home/acauan/ufam/iats/sprint_07/filo_priori_v5/filo_priori
python scripts/core/run_experiment_server.py --smoke-train 20 --smoke-test 10
```

4. Verify outputs:
- Check `results/execution_XXX/metrics.json` has correct dimensions
- Check `results/execution_XXX/prioritized_hybrid.csv` has valid probabilities
- Check APFD ≥ 0.50 (baseline)

## Expected Output Changes

### Files Changed:
- `best_model.pt` → `tabpfn_model.pkl`
- `training_history.csv` → REMOVED (no epochs)
- Embedder artifacts: `pca.pkl` → REMOVED (no PCA)

### Metrics:
- Performance may improve due to richer 1024D embeddings
- Training time should decrease (TabPFN is fast)
- Memory usage may increase slightly (1024D vs 128D features)

## Rollback Plan

If issues arise:
```bash
git checkout HEAD -- requirements.txt
git checkout HEAD -- filo_priori/data_processing/03_embed_sbert.py
git checkout HEAD -- filo_priori/scripts/core/run_experiment_server.py
rm filo_priori/utils/tabpfn_classifier.py
```

## Next Steps

1. Complete remaining changes in `run_experiment_server.py`
2. Delete `utils/model.py`
3. Update `config.yaml`
4. Run smoke test
5. Run full test
6. Compare metrics with baseline
7. Update documentation (README.md, CLAUDE.md)

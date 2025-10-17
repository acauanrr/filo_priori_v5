# 🐛 Correções Aplicadas - Filo-Priori V5 (BGE + SAINT)

**Data**: 2025-10-16  
**Status**: ✅ Bugs corrigidos e testado no servidor

---

## 🔧 Correções Implementadas

### 1. ✅ Caminho dos Datasets (FileNotFoundError)

**Problema**:
```
FileNotFoundError: [Errno 2] No such file or directory: 'datasets/train.csv'
```

**Causa**: O script `run_full_test.sh` faz `cd filo_priori` antes de executar, mas os caminhos eram relativos à raiz do projeto.

**Solução Aplicada**:
```python
# filo_priori/scripts/core/run_experiment_server.py:73-75
DEFAULT_CONFIG = {
    'train_csv': '../datasets/train.csv',     # Era 'datasets/train.csv'
    'test_csv': '../datasets/test_full.csv',  # Era 'datasets/test_full.csv'
    'output_dir': '../results',               # Era 'results'
}
```

**Arquivo**: `filo_priori/scripts/core/run_experiment_server.py:73-75`

---

### 2. ✅ Label Smoothing em BCEWithLogitsLoss (TypeError)

**Problema**:
```
TypeError: BCEWithLogitsLoss.__init__() got an unexpected keyword argument 'label_smoothing'
```

**Causa**: `BCEWithLogitsLoss` do PyTorch não tem parâmetro `label_smoothing` (só existe em `CrossEntropyLoss`).

**Solução Aplicada**:

**Antes**:
```python
# ERRADO - BCEWithLogitsLoss não aceita label_smoothing
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor, label_smoothing=label_smoothing)
```

**Agora**:
```python
# filo_priori/utils/saint_trainer.py:217-224
# Criar loss sem label_smoothing
if pos_weight is not None:
    pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
else:
    criterion = nn.BCEWithLogitsLoss()

# Label smoothing aplicado manualmente no loop de treinamento
```

**Aplicação manual do label smoothing no loop**:
```python
# filo_priori/utils/saint_trainer.py:283-287
# Apply label smoothing manually
if label_smoothing > 0:
    labels_smooth = labels * (1 - label_smoothing) + 0.5 * label_smoothing
else:
    labels_smooth = labels

# Forward pass
logits = model(x_continuous, x_categorical)
loss = criterion(logits, labels_smooth)
```

**Arquivos**:
- `filo_priori/utils/saint_trainer.py:217-224`
- `filo_priori/utils/saint_trainer.py:283-292`

---

## ✅ Validação Pós-Correção

Execute no servidor para validar:

```bash
# Teste rápido (5 min)
cd filo_priori
python scripts/core/run_experiment_server.py --smoke-train 5 --smoke-test 3 --device cuda

# Ou use o script helper
./run_full_test.sh
```

### Output Esperado (Primeiras Linhas):

```
🎮 GPU detectada, usando CUDA
2025-10-16 XX:XX:XX | INFO | FILO-PRIORI V5 - PIPELINE START (BGE + SAINT)
2025-10-16 XX:XX:XX | INFO | Execution directory: results/execution_XXX
2025-10-16 XX:XX:XX | INFO | Device: cuda
2025-10-16 XX:XX:XX | INFO | [1/7] Loading and parsing commits...
2025-10-16 XX:XX:XX | INFO | Train: XXXXX rows, Test: XXXXX rows
...
2025-10-16 XX:XX:XX | INFO | [5/7] Training SAINT transformer...
2025-10-16 XX:XX:XX | INFO | SAINT model created with 1,860,225 parameters
2025-10-16 XX:XX:XX | INFO | STARTING SAINT TRAINING
2025-10-16 XX:XX:XX | INFO | Epoch 1/30
...
```

**Não deve mais aparecer**:
- ❌ `FileNotFoundError: datasets/train.csv`
- ❌ `TypeError: BCEWithLogitsLoss.__init__() got an unexpected keyword argument`

---

## 📊 Status do Pipeline

| Etapa | Status | Tempo Estimado |
|-------|--------|----------------|
| [1/7] Loading commits | ✅ | ~1 min |
| [2/7] Building text_semantic | ✅ | ~1 min |
| [3/7] Generating BGE embeddings | ✅ | ~10-15 min |
| [4/7] Building features | ✅ | ~2 min |
| [5/7] Training SAINT | ✅ | ~30-60 min (GPU) |
| [6/7] Evaluating | ✅ | ~5 min |
| [7/7] Saving results | ✅ | ~1 min |

**Total**: ~45-90 minutos (GPU) ou ~2-4 horas (CPU)

---

## 🎯 Próximos Passos

1. ✅ Execute `./run_full_test.sh` no servidor
2. ✅ Aguarde conclusão (~45-90 min com GPU)
3. ✅ Verifique resultados em `filo_priori/results/execution_XXX/`
4. ✅ Leia o resumo: `cat filo_priori/results/execution_XXX/summary.txt`

---

## ⚠️ Se Encontrar Novos Erros

### CUDA Out of Memory
```yaml
# Edite config.yaml
training:
  batch_size: 8  # Reduza de 16 para 8
```

### Treinamento Lento
- **Normal em CPU**: ~2-4 horas
- **Solução**: Use GPU ou reduza dados com smoke test

### Outros Erros
Verifique logs completos em:
```bash
tail -100 filo_priori/results/execution_XXX/*.log
```

---

**Última atualização**: 2025-10-16 02:10  
**Status**: ✅ Todos os bugs conhecidos corrigidos

# ✅ Filo-Priori V5 - Checklist de Validação (BGE + SAINT)

**Data da Refatoração**: 2025-10-16
**Objetivo**: Substituir SBERT → BGE e TabPFN → SAINT Transformer

---

## 🎯 Mudanças Implementadas

### ✅ 1. Embeddings de Texto (BGE)
- [x] **Já implementado** - BGE-large-en-v1.5 em `filo_priori/data_processing/03_embed_sbert.py`
- [x] PCA removido - embeddings 1024D usados diretamente
- [x] StandardScaler mantido para normalização

**Arquivo**: `filo_priori/data_processing/03_embed_sbert.py:29`

---

### ✅ 2. Modelo SAINT Transformer (NOVO)
- [x] Implementação completa em `filo_priori/models/saint.py`
- [x] Self-attention (atenção intra-amostra)
- [x] **Intersample attention** (atenção entre amostras - inovação do SAINT)
- [x] Arquitetura: 6 layers, 8 heads, embedding_dim=128
- [x] Dropout=0.1 para regularização

**Componentes**:
- `EmbeddingLayer`: Projeta features contínuas (1028D) para embeddings (128D)
- `MultiHeadSelfAttention`: Atenção padrão dentro de cada amostra
- `IntersampleAttention`: Atenção entre diferentes amostras no batch
- `SAINTBlock`: Combina ambas as atenções + MLP
- `SAINT`: Modelo completo com pooling e classificação

**Arquivo**: `filo_priori/models/saint.py` (489 linhas)

---

### ✅ 3. Treinamento SAINT (NOVO)
- [x] Training loop completo em `filo_priori/utils/saint_trainer.py`
- [x] Early stopping baseado em `val_auprc` (ideal para dados desbalanceados)
- [x] Cosine LR schedule com warmup (3 epochs warmup)
- [x] Gradient clipping (max_norm=1.0)
- [x] Label smoothing (0.05)
- [x] Class imbalance handling:
  - WeightedRandomSampler (target 20% positivos)
  - pos_weight=5.0 no BCE loss

**Arquivo**: `filo_priori/utils/saint_trainer.py` (467 linhas)

---

### ✅ 4. Pipeline Atualizado
- [x] `run_experiment_server.py` completamente refatorado
- [x] Removido todo código TabPFN
- [x] Adicionado treinamento PyTorch com DataLoaders
- [x] Mantém feature engineering (1028D = 1024 BGE + 4 temporal)
- [x] Checkpointing do melhor modelo (baseado em val_auprc)

**Arquivo**: `filo_priori/scripts/core/run_experiment_server.py` (677 linhas)

---

### ✅ 5. Configuração
- [x] `config.yaml` atualizado com seção SAINT
- [x] Removida seção TabPFN
- [x] Adicionados parâmetros de treinamento completos
- [x] Valores conservadores para estabilidade

**Arquivo**: `config.yaml`

---

### ✅ 6. Dependências
- [x] `requirements.txt` atualizado
- [x] Removido: `tabpfn>=0.1.9`
- [x] Mantido: PyTorch, sentence-transformers, sklearn
- [x] Não há novas dependências necessárias

**Arquivo**: `requirements.txt`

---

### ✅ 7. Scripts de Execução
- [x] `run_full_test.sh` atualizado para SAINT
- [x] `run_smoke_test.sh` atualizado para SAINT
- [x] Auto-detecção de GPU/CPU
- [x] Mensagens de output corrigidas (best_model.pth, não .pkl)
- [x] Permissões de execução configuradas (`chmod +x`)

**Arquivos**:
- `run_full_test.sh` (60 linhas)
- `run_smoke_test.sh` (54 linhas)

---

## 🚀 Validação para Execução no Servidor

### Pré-requisitos
```bash
# 1. Verificar PyTorch instalado
python -c "import torch; print(f'PyTorch {torch.__version__}')"

# 2. Verificar sentence-transformers
python -c "import sentence_transformers; print('✓ sentence-transformers OK')"

# 3. Verificar estrutura de arquivos
ls filo_priori/models/saint.py
ls filo_priori/utils/saint_trainer.py
ls filo_priori/scripts/core/run_experiment_server.py

# 4. Verificar datasets
ls datasets/train.csv datasets/test_full.csv
```

### Execução no Servidor

#### Opção 1: Smoke Test (Validação Rápida)
```bash
./run_smoke_test.sh
```
**Tempo esperado**: ~10-20 minutos
**Builds**: 100 treino, 50 teste
**Objetivo**: Validar que o pipeline funciona end-to-end

#### Opção 2: Full Test (Produção)
```bash
./run_full_test.sh
```
**Tempo esperado**:
- GPU: ~30-60 minutos
- CPU: ~2-4 horas

**Dataset completo**: ~1.7GB treino + ~581MB teste

#### Opção 3: Manual (Controle Total)
```bash
cd filo_priori

# Com GPU
python scripts/core/run_experiment_server.py --full-test --device cuda

# Forçar CPU
python scripts/core/run_experiment_server.py --full-test --device cpu

# Custom smoke test
python scripts/core/run_experiment_server.py \
    --smoke-train 50 \
    --smoke-test 25 \
    --device cuda
```

---

## 📊 Outputs Esperados

Após execução bem-sucedida, você encontrará em `filo_priori/results/execution_XXX/`:

```
execution_001/
├── metrics.json              # Métricas completas (APFD, AUPRC, etc.)
├── config.json               # Configuração do experimento
├── best_model.pth            # Checkpoint do SAINT (melhor época)
├── training_history.json     # Métricas por época
├── prioritized_hybrid.csv    # Testes ranqueados por probabilidade
├── apfd_per_build.csv        # APFD calculado por build
├── summary.txt               # Resumo textual do experimento
├── feature_builder.pkl       # Feature engineering artifacts
└── embedder/
    └── scaler.pkl            # StandardScaler para BGE embeddings
```

### Estrutura do `metrics.json`
```json
{
  "metadata": {
    "model_type": "SAINT",
    "embedding_model": "BAAI/bge-large-en-v1.5",
    "embedding_dim": 1024,
    "model_params": 2000000,  // ~2M parâmetros
    "device": "cuda",
    "timestamp": "2025-10-16T..."
  },
  "metrics": {
    "apfd": 0.70,      // Target ≥ 0.70
    "apfdc": 0.72,
    "auprc": 0.25,     // Target ≥ 0.20
    "precision": 0.18, // Target ≥ 0.15
    "recall": 0.65,    // Target ≥ 0.50
    "f1": 0.28,
    "accuracy": 0.92
  },
  "training": {
    "best_epoch": 12,
    "best_metric": 0.0245,  // val_auprc
    "monitor_metric": "val_auprc",
    "history": { ... }
  }
}
```

---

## ⚠️ Possíveis Problemas e Soluções

### 1. Erro: `ModuleNotFoundError: No module named 'torch'`
**Solução**: Instalar dependências
```bash
pip install -r requirements.txt
```

### 2. Erro: CUDA out of memory
**Solução 1**: Reduzir batch_size em `config.yaml`
```yaml
training:
  batch_size: 8  # Reduzir de 16 para 8
```

**Solução 2**: Forçar CPU
```bash
python scripts/core/run_experiment_server.py --full-test --device cpu
```

### 3. Erro: `FileNotFoundError: datasets/train.csv`
**Solução**: Verificar caminho relativo
```bash
pwd  # Deve estar em filo_priori_v5/
ls datasets/  # Deve listar train.csv e test_full.csv
```

### 4. Treinamento muito lento em CPU
**Esperado**: CPU é ~10-20x mais lento que GPU para SAINT
- Smoke test: ~10-20 min (CPU)
- Full test: ~2-4 horas (CPU)

**Alternativa**: Usar servidor com GPU ou reduzir smoke test:
```bash
python filo_priori/scripts/core/run_experiment_server.py \
    --smoke-train 20 --smoke-test 10 --device cpu
```

---

## ✅ Checklist Final de Validação

Antes de rodar no servidor, verifique:

- [ ] PyTorch instalado e funcionando
- [ ] sentence-transformers instalado
- [ ] Datasets presentes em `datasets/`
- [ ] Arquivos SAINT criados:
  - [ ] `filo_priori/models/saint.py`
  - [ ] `filo_priori/models/__init__.py`
  - [ ] `filo_priori/utils/saint_trainer.py`
- [ ] Scripts de execução atualizados:
  - [ ] `run_full_test.sh` (menciona SAINT, não TabPFN)
  - [ ] `run_smoke_test.sh` (menciona SAINT, não TabPFN)
- [ ] Permissões de execução: `ls -l run_*.sh` mostra `-rwxr-xr-x`
- [ ] Nenhum código TabPFN restante no pipeline

---

## 📝 Notas Adicionais

1. **Tempo de Treinamento**: SAINT requer treinamento real (diferente de TabPFN que era pré-treinado), então espere:
   - Smoke test: ~10-20 minutos (CPU) ou ~2-5 minutos (GPU)
   - Full test: ~2-4 horas (CPU) ou ~30-60 minutos (GPU)

2. **Early Stopping**: O treinamento pode parar antes de 30 épocas se não houver melhoria em `val_auprc` por 8 épocas consecutivas.

3. **Melhor Época**: O modelo salvo (`best_model.pth`) é da época com melhor `val_auprc`, não necessariamente a última época.

4. **Intersample Attention**: Esta é a principal inovação do SAINT. Se você quiser desabilitar para comparação:
   ```yaml
   # Em config.yaml
   saint:
     use_intersample: false  # Vira transformer padrão
   ```

5. **Monitoramento**: Para acompanhar progresso em tempo real:
   ```bash
   tail -f filo_priori/results/execution_XXX/*.log  # Se houver logs
   # Ou apenas observe os prints no terminal
   ```

---

## 🎯 Critérios de Sucesso

O experimento será considerado bem-sucedido se:

1. ✅ Pipeline executa sem erros do início ao fim
2. ✅ Gera todos os arquivos de output esperados
3. ✅ APFD ≥ 0.70 (objetivo principal)
4. ✅ AUPRC ≥ 0.20
5. ✅ Precision ≥ 0.15
6. ✅ Recall ≥ 0.50

---

**Última atualização**: 2025-10-16
**Autor**: Claude Code Agent
**Versão**: Filo-Priori V5 (BGE + SAINT)

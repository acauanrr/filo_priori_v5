# 🚀 Filo-Priori V5 - Guia Rápido (BGE + TabPFN)

## ✨ Novidades da Refatoração

O pipeline foi modernizado com componentes state-of-the-art:

- **Embeddings**: `BAAI/bge-large-en-v1.5` (1024D) - substitui SBERT (384D)
- **Sem PCA**: Embeddings usados diretamente (1024D → 1028D com features temporais)
- **Classificador**: TabPFN (pré-treinado) - substitui Deep MLP + PyTorch
- **Sem treinamento**: TabPFN não precisa de epochs, early stopping, ou otimizador

### Arquitetura Antes vs Depois

**Antes (V4):**
```
Text → SBERT (384D) → PCA (128D) → Features (132D) → Deep MLP → Probs
                                     ↓
                           Epochs, Early Stopping, AdamW
```

**Depois (V5 Refatorado):**
```
Text → BGE-large (1024D) → Features (1028D) → TabPFN.fit() → Probs
                                                 ↓
                                      Sem treinamento necessário!
```

## 📋 Pré-requisitos

- Python 3.8+
- ~4GB RAM (smoke test) ou ~16GB RAM (full test)
- ~5GB espaço em disco para modelos e datasets

## ⚡ Instalação Rápida

### 1. Instalar Dependências

```bash
cd /home/acauan/ufam/iats/sprint_07/filo_priori_v5
./install_dependencies.sh
```

Ou manualmente:

```bash
pip install -r requirements.txt
```

### 2. Verificar Instalação

```bash
cd filo_priori
python -c "
from tabpfn import TabPFNClassifier
from sentence_transformers import SentenceTransformer
print('✓ TabPFN disponível')
model = SentenceTransformer('BAAI/bge-large-en-v1.5')
print(f'✓ BGE-large carregado ({model.get_sentence_embedding_dimension()}D)')
"
```

## 🧪 Executar Testes

### Smoke Test (Rápido - ~5 minutos)

Testa o pipeline com 100 builds de treino e 50 de teste:

```bash
./run_smoke_test.sh
```

Ou diretamente:

```bash
cd filo_priori
python scripts/core/run_experiment_server.py --smoke-train 100 --smoke-test 50
```

### Full Test (Completo - ~15-30 minutos)

Executa com dataset completo:

```bash
./run_full_test.sh
```

Ou diretamente:

```bash
cd filo_priori
python scripts/core/run_experiment_server.py --full-test
```

## 📊 Resultados

Os resultados são salvos em `filo_priori/results/execution_XXX/`:

```
execution_001/
├── metrics.json              # Métricas completas (APFD, AUPRC, etc.)
├── config.json               # Configuração do experimento
├── tabpfn_model.pkl          # Modelo TabPFN treinado (pickle)
├── prioritized_hybrid.csv    # Testes priorizados com probabilidades
├── apfd_per_build.csv        # APFD calculado por build
├── feature_builder.pkl       # Artefatos de features
├── embedder/                 # Artefatos do BGE (scaler.pkl apenas, sem PCA)
│   └── scaler.pkl
└── summary.txt               # Resumo do experimento
```

### Métricas Esperadas

- **APFD**: ≥ 0.70 (target)
- **AUPRC**: ≥ 0.20
- **Precision**: ≥ 0.15
- **Recall**: ≥ 0.50
- **Accuracy**: ≥ 0.90

## 🔧 Mudanças Importantes

### Arquivos Modificados

1. **requirements.txt**: Adicionado `tabpfn>=0.1.9`
2. **config.yaml**: Parâmetros de MLP/PyTorch marcados como OBSOLETE
3. **filo_priori/data_processing/03_embed_sbert.py**: BGE + sem PCA
4. **filo_priori/utils/tabpfn_classifier.py**: NOVO módulo
5. **filo_priori/scripts/core/run_experiment_server.py**: Refatorado para TabPFN

### Arquivos Removidos

- **filo_priori/utils/model.py**: DeepMLP não é mais usado
- **filo_priori/utils/dataset.py**: TabularDataset/DataLoader não são mais usados

### Outputs Mudaram

- `best_model.pt` → `tabpfn_model.pkl` (pickle ao invés de PyTorch)
- `training_history.csv` → REMOVIDO (TabPFN não tem epochs)
- `embedder/pca.pkl` → REMOVIDO (sem PCA)
- Dimensão de features: 132D → 1028D

## ⚠️ Troubleshooting

### Erro: "TabPFN not installed"

```bash
pip install tabpfn
```

### Erro: "CUDA out of memory"

TabPFN funciona bem em CPU. Use `--device cpu`:

```bash
python scripts/core/run_experiment_server.py --full-test --device cpu
```

### Erro: "Model dimension mismatch"

Certifique-se de que está usando BGE-large (1024D):

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('BAAI/bge-large-en-v1.5')
print(model.get_sentence_embedding_dimension())  # Deve ser 1024
```

### Comparar com V4 Anterior

Se quiser comparar resultados:

```bash
# V5 (novo)
cd /home/acauan/ufam/iats/sprint_07/filo_priori_v5
./run_full_test.sh

# V4 (antigo)
cd /home/acauan/ufam/iats/sprint_07/filo_priori_v4
python run_experiment.py --full-test
```

## 📚 Documentação Adicional

- **REFACTORING_SUMMARY.md**: Detalhes técnicos completos da refatoração
- **CLAUDE.md**: Instruções para Claude Code trabalhar no projeto
- **README.md**: Documentação original do projeto

## 🎯 Próximos Passos

1. ✅ Instalar dependências (`./install_dependencies.sh`)
2. ✅ Executar smoke test (`./run_smoke_test.sh`)
3. ✅ Verificar resultados em `filo_priori/results/execution_001/`
4. ✅ Se tudo funcionar, executar full test (`./run_full_test.sh`)
5. ✅ Comparar métricas com V4 anterior

## 💡 Benefícios da Nova Arquitetura

1. **Melhor qualidade**: BGE-large é state-of-the-art (1024D vs 384D)
2. **Menos perda de informação**: Sem compressão PCA
3. **Mais simples**: TabPFN elimina ~300 linhas de código PyTorch
4. **Mais rápido**: Sem loop de treinamento (epochs, early stopping)
5. **Menos hyperparameters**: TabPFN é pré-treinado, não precisa tuning

---

**Versão**: Filo-Priori V5 (Refatorado - BGE + TabPFN)
**Data**: 2025-10-15
**Autor**: Refatoração automatizada por Claude Code

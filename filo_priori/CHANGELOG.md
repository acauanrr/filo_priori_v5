# Changelog - Filo-Priori V5

## [2025-10-15] - Sistema de Execuções Versionadas

### ✅ Implementado

#### 1. Sistema de Execution Directories
- **Automático**: Cada execução cria `results/execution_XXX/` (001, 002, 003, ...)
- **Sequencial**: Detecta automaticamente o próximo número disponível
- **Isolado**: Cada execução tem seus próprios arquivos, sem sobrescrever

#### 2. Arquivos Salvos por Execução

**Core Files:**
- `metrics.json` - Métricas completas com metadata, history, probability stats
- `config.json` - Configuração do experimento
- `best_model.pt` - Checkpoint do melhor modelo (PyTorch)
- `prioritized_hybrid.csv` - Predições com ranks
- `training_history.csv` - Histórico epoch-by-epoch (fácil plotar)
- `summary.txt` - Resumo legível para humanos

**Artifacts (reutilizáveis):**
- `feature_builder.pkl` - Label encoders e scalers
- `embedder/pca.pkl` - PCA 384D→128D
- `embedder/scaler.pkl` - StandardScaler para embeddings

#### 3. Metadata Enriquecida

Cada `metrics.json` agora contém:

```json
{
  "metadata": {
    "experiment_type": "smoke_test | full_test",
    "timestamp": "2025-10-15T18:56:00",
    "n_train_builds": 100,
    "n_test_builds": 50,
    "n_epochs_configured": 20,
    "n_epochs_executed": 12,
    "early_stopped": true,
    "device": "cuda",
    "dataset_stats": {
      "train_total": 5234,
      "train_failures": 156,
      "train_failure_rate": 0.0298,
      "test_total": 2617,
      "test_failures": 78,
      "test_failure_rate": 0.0298
    }
  },
  "metrics": {
    "apfd": 0.6543,
    "apfdc": 0.6512,
    "auprc": 0.1234,
    "precision": 0.0876,
    "recall": 0.4231,
    "f1": 0.1443,
    "accuracy": 0.9702,
    "discrimination_ratio": 1.87
  },
  "probability_stats": {
    "failures_mean": 0.0356,
    "failures_std": 0.0123,
    "failures_min": 0.0012,
    "failures_max": 0.0987,
    "passes_mean": 0.0190,
    "passes_std": 0.0098,
    "passes_min": 0.0008,
    "passes_max": 0.0654
  },
  "history": [...]
}
```

#### 4. Script de Comparação

Criado `scripts/compare_executions.py` com:

**Features:**
- Carrega todas as execuções automaticamente
- Compara métricas chave (APFD, AUPRC, discrimination, etc.)
- Mostra estatísticas descritivas (mean, std, min, max)
- Identifica best performers para cada métrica
- Calcula diferenças vs baseline (execution_001)
- Exporta comparação para CSV

**Uso:**
```bash
# Ver no terminal
python scripts/compare_executions.py

# Exportar CSV
python scripts/compare_executions.py --export comparison.csv

# Diretório customizado
python scripts/compare_executions.py --results-dir custom_results/
```

**Output Exemplo:**
```
Found 3 execution(s)

==================================================
EXECUTION COMPARISON
==================================================

📊 Key Metrics:
execution      type        apfd    apfdc   auprc   precision recall  f1      discrimination
execution_001  smoke_test  0.5126  0.5098  0.0312  0.0215    0.1053  0.0357  1.0500
execution_002  smoke_test  0.4864  0.4955  0.0399  0.0330    1.0000  0.0639  1.6364
execution_003  full_test   0.5074  0.5123  0.0743  0.0315    0.2251  0.0552  1.0544

🏆 Best Performers:
  Best APFD          : execution_001        = 0.5126
  Best DISCRIMINATION: execution_002        = 1.6364
```

#### 5. Documentação Atualizada

- ✅ `README.md` - Seção "Sistema de Execuções" + comandos de uso
- ✅ `QUICKSTART.md` - Seção "Comparing Executions" com exemplos Python/bash
- ✅ `CHANGELOG.md` - Este arquivo

### 🔧 Mudanças Técnicas

**run_experiment_server.py:**
- Função `get_next_execution_dir()` - Cria próximo diretório automaticamente
- Todas as referências de `output_dir` → `exec_dir`
- Salvamento de artifacts adicionais (model checkpoint, history CSV, etc.)
- Geração de `summary.txt` formatado

**Estrutura de Diretórios:**
```
results/
├── execution_001/
│   ├── metrics.json
│   ├── config.json
│   ├── best_model.pt
│   ├── prioritized_hybrid.csv
│   ├── training_history.csv
│   ├── summary.txt
│   ├── feature_builder.pkl
│   └── embedder/
│       ├── pca.pkl
│       └── scaler.pkl
├── execution_002/
│   └── ... (mesma estrutura)
└── execution_003/
    └── ... (mesma estrutura)
```

### 🎯 Benefícios

1. **Rastreabilidade**: Cada experimento preservado com timestamp
2. **Comparabilidade**: Fácil comparar múltiplas configurações
3. **Reprodutibilidade**: Config + model checkpoint = experimento reproduzível
4. **Organização**: Não sobrescreve resultados, mantém histórico
5. **Análise**: Training history em CSV para plotagem rápida
6. **Legibilidade**: summary.txt para leitura humana rápida

### 🗑️ Limpeza

- Removida pasta duplicada `/home/acauan/ufam/iats/sprint_07/filo_priori_v4/filo_priori_v5`
- Mantida somente `/home/acauan/ufam/iats/sprint_07/filo_priori_v5` (correta)

### 📋 Próximos Passos

1. ✅ Sistema de execuções implementado
2. ✅ Script de comparação criado
3. ✅ Documentação atualizada
4. ⏳ **Aguardando**: Teste smoke test no servidor
5. ⏳ **Aguardando**: Teste full test no servidor

---

**Autor**: Claude Code
**Data**: 2025-10-15
**Status**: ✅ Pronto para uso

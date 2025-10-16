# Filo-Priori V5

Solução melhorada para priorização de testes usando:
- **Extração estruturada** de commits (issues, APIs, erros, etc.)
- **SBERT multilíngue** (384D → 128D) para embeddings semânticos
- **FT-Transformer** ou MLP profundo com balanceamento classe-ciente
- **Estratificação** por Build_ID e otimização de limiar

## 📁 Estrutura

```
filo_priori_v5/
├── data_processing/
│   ├── 01_parse_commit.py       # Extração estruturada de commits
│   ├── 02_build_text_semantic.py # Construção de text_semantic
│   └── 03_embed_sbert.py         # Embeddings SBERT + PCA + Scaler
├── models/                       # (FT-Transformer - implementar se necessário)
├── configs/
│   └── config.yaml               # Configuração centralizada
└── artifacts/                    # Outputs (embeddings, models, reports)
```

## 🚀 Como Usar

### Execução Rápida (Smoke Test)
```bash
cd /home/acauan/ufam/iats/sprint_07/filo_priori_v4/filo_priori

python scripts/core/run_experiment_server.py \
    --smoke-train 100 \
    --smoke-test 50 \
    --smoke-epochs 20
```

### Execução Completa (Full Test)
```bash
cd /home/acauan/ufam/iats/sprint_07/filo_priori_v4/filo_priori
python scripts/core/run_experiment_server.py --full-test
```

### Comparar Experimentos
```bash
# Ver comparação no terminal
python scripts/compare_executions.py

# Exportar para CSV
python scripts/compare_executions.py --export comparison.csv
```

## 📁 Sistema de Execuções

Cada execução cria automaticamente uma pasta `results/execution_XXX/` contendo:

- **metrics.json** - Métricas completas (APFD, AUPRC, discrimination, etc.)
- **config.json** - Configuração usada no experimento
- **best_model.pt** - Checkpoint do melhor modelo
- **prioritized_hybrid.csv** - Predições com ranks
- **training_history.csv** - Histórico de treinamento (fácil plotar)
- **summary.txt** - Resumo legível
- **feature_builder.pkl** - Artefatos de feature engineering
- **embedder/** - Artefatos do SBERT (PCA, scaler)

Isso permite:
- ✅ Comparar experimentos facilmente
- ✅ Reproduzir resultados exatos
- ✅ Versionar automaticamente cada execução
- ✅ Não sobrescrever resultados anteriores

## 🔄 Pipeline Detalhado

### 1. Parse de Commits
```bash
python data_processing/01_parse_commit.py ../datasets/train.csv ../artifacts/train_parsed.csv
```

**Saída**: Adiciona colunas `commit_issue_ids`, `commit_apis`, `commit_errors`, `commit_text`, etc.

### 2. Construção de text_semantic
```bash
python data_processing/02_build_text_semantic.py ../artifacts/train_parsed.csv ../artifacts/train_semantic.csv
```

**Saída**: Adiciona `text_semantic` (TE_Summary + TC_Steps + commit_text) e `label_binary` (0/1).

### 3. Geração de Embeddings
```bash
python data_processing/03_embed_sbert.py ../artifacts/train_semantic.csv 128 ../artifacts/embeddings/
```

**Saída**:
- `embeddings_train.npy` (padronizado)
- `pca.pkl`, `scaler.pkl` (artefatos para validação/teste)

### 4. Treino (implementar depois)
```bash
# Usar embeddings + features tabulares no FT-Transformer
# Com balanced sampler, class weights, e PR-AUC
```

## 📊 Melhorias vs V4

| Aspecto | V4 | V5 |
|---------|----|----|
| Features de commit | ❌ Brutas | ✅ Estruturadas (APIs, erros, módulos) |
| Embeddings | ❌ 768D sem normalização | ✅ 384D→128D + PCA + Scaler |
| Modelo | MLP 2 camadas | MLP profundo ou FT-Transformer |
| Balanceamento | Sampler simples | **Classe-ciente** + Focal Loss opcional |
| Diversidade | Penaliza igualmente | Deduplicação por classe |
| Split | Aleatório | GroupKFold por Build_ID |

## 🎯 Próximos Passos

1. **Implementar módulo de treino** (`04_train_ftt.py` ou usar v4 adaptado)
2. **Testar com smoke test** (100 builds)
3. **Adicionar features temporais enriquecidas** (16D em vez de 4D)
4. **Ajustar pesos hybrid**: `0.9×prob + 0.1×div`

## 📦 Dependências

```bash
pip install pandas numpy scikit-learn sentence-transformers torch
# Para FT-Transformer: pip install rtdl (Research Tabular Deep Learning)
```

## 📝 Notas

- **text_semantic** combina informação de 3 fontes com tags estruturadas
- **commit_text** prioriza: Errors > Actions > APIs > Modules > Flags
- **Labels ambíguos** (Blocked, Pending) são removidos automaticamente
- **Imbalance**: Mantém ~3% taxa de falha, usar sampler 1:3 nos batches
- **PCA 128D** preserva >95% da variância com 1/3 das dimensões

---

**Autor**: Filo-Priori V5 Team
**Data**: 2025-10-15

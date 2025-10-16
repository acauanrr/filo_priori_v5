# Como Executar o Filo-Priori V5

## ⚠️ IMPORTANTE: Localização do Código V5

O código V5 está em: `/filo_priori_v4/filo_priori/`

- ✅ `/filo_priori_v4/filo_priori/scripts/` ← Scripts V5 corretos
- ✅ Pasta antiga `/filo_priori_v4/scripts/` foi **removida** (evitar confusão)

## 🚀 Opção 1: Scripts Helper (Mais Fácil)

Execute da raiz do projeto `/filo_priori_v4/`:

```bash
# Smoke test (10-15 min GPU)
./run_smoke_test.sh

# Full test (2-3h GPU)
./run_full_test.sh
```

## 🚀 Opção 2: Comando Manual

```bash
# Vá para o diretório do código V5
cd /home/acauan/ufam/iats/sprint_07/filo_priori_v4/filo_priori

# Smoke test
python scripts/core/run_experiment_server.py \
    --smoke-train 100 \
    --smoke-test 50 \
    --smoke-epochs 20

# OU Full test
python scripts/core/run_experiment_server.py --full-test
```

## 📊 Ver Resultados

```bash
# Da raiz do projeto (/filo_priori_v4/)
cd filo_priori

# Comparar todas as execuções
python scripts/compare_executions.py

# Exportar para CSV
python scripts/compare_executions.py --export ../results/comparison.csv

# Ver última execução
cat ../results/execution_*/summary.txt | tail -50
```

## 📁 Estrutura de Resultados

Cada execução cria:
```
/filo_priori_v4/results/execution_001/
    ├── metrics.json              ← Métricas completas
    ├── config.json               ← Configuração
    ├── best_model.pt             ← Modelo PyTorch
    ├── prioritized_hybrid.csv    ← Predições ranqueadas
    ├── training_history.csv      ← Histórico de treino
    ├── summary.txt               ← Resumo legível
    ├── feature_builder.pkl       ← Artefatos features
    └── embedder/                 ← PCA + scaler
```

## ✅ Limpeza Concluída

- ✅ Pasta `scripts/` V4 antiga **removida**
- ✅ Apenas `/filo_priori_v4/filo_priori/scripts/` existe agora
- ✅ Sem risco de executar script errado!

## 📝 Documentação Completa

- `/filo_priori_v4/filo_priori/README.md` - Visão geral do V5
- `/filo_priori_v4/filo_priori/QUICKSTART.md` - Guia rápido
- `/filo_priori_v4/filo_priori/CHANGELOG.md` - Mudanças recentes
- `/filo_priori_v4/filo_priori/STRUCTURE_VERIFICATION.md` - Verificação técnica

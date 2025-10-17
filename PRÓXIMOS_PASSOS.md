# Próximos Passos - Full Test

**Data**: 2025-10-16
**Status**: ✅ Correções aplicadas - Pronto para validação

---

## ✅ O Que Foi Feito

### 1. Correção de Paths (Execution_001)
- ✅ Corrigido `output_dir` de `'../results'` → `'results'`
- ✅ Adicionadas validações de salvamento
- ✅ Todos os arquivos agora são salvos corretamente

### 2. Análise do Smoke Test (Execution_002)
- ✅ Identificado overfitting severo (val AUPRC 0.83 → test 0.04)
- ✅ Identificado discrimination ruim (0.97x)
- ✅ Identificado APFD per-build inadequado (apenas 16.7% ≥ 0.7)

### 3. Aplicação de Correções de Hiperparâmetros

| Parâmetro | Antes | Depois | Status |
|-----------|-------|--------|--------|
| `pos_weight` | 5.0 | **10.0** | ✅ |
| `label_smoothing` | 0.05 | **0.01** | ✅ |
| `dropout` | 0.1 | **0.2** | ✅ |
| `patience` | 8 | **5** | ✅ |
| `target_positive_fraction` | 0.20 | **0.30** | ✅ |
| `learning_rate` | 5e-4 | **3e-4** | ✅ |
| `weight_decay` | 0.01 | **0.05** | ✅ |

**Backup criado**: `filo_priori/scripts/core/run_experiment_server.py.backup_pre_hyperparams`

---

## 🚀 Próximos Passos

### Opção 1: Validar com Smoke Test (RECOMENDADO)

**Objetivo**: Verificar se as correções melhoraram as métricas antes de rodar full test (2-6h)

```bash
cd ~/iats/filo_priori_v5

# 1. Rodar smoke test com novos hiperparâmetros (15-30 min)
./run_smoke_test.sh

# 2. Analisar resultados (execution_003)
cat filo_priori/results/execution_003/summary.txt

# 3. Extrair métricas chave
python -c "
import json
with open('filo_priori/results/execution_003/metrics.json') as f:
    m = json.load(f)['metrics']
    p = json.load(f)['probability_stats']

print('🔍 MÉTRICAS CRÍTICAS:')
print(f\"  Discrimination: {m['discrimination_ratio']:.2f}x (antes: 0.97x)\")
print(f\"  Test AUPRC: {m['auprc']:.4f} (antes: 0.0400)\")
print(f\"  Test Precision: {m['precision']:.4f} (antes: 0.0364)\")
print(f\"  Test Recall: {m['recall']:.4f} (antes: 0.0769)\")
print(f\"  Failures mean prob: {p['failures_mean']:.4f} (antes: 0.1801)\")
print(f\"  Passes mean prob: {p['passes_mean']:.4f} (antes: 0.1860)\")
"

# 4. Verificar APFD per-build
cat filo_priori/results/execution_003/apfd_per_build.csv
```

#### ✅ Checklist de Validação

Se as correções funcionaram, você deve ver:

- [ ] **Discrimination > 1.5x** (antes: 0.97x)
  - Failures prob > Passes prob
  - Ideal: 2.0x-4.0x

- [ ] **Test AUPRC > 0.15** (antes: 0.04)
  - Mostra que modelo está aprendendo
  - Ideal: 0.20-0.40

- [ ] **Val/Test gap < 50%** (antes: 95%)
  - Overfitting reduzido
  - Ideal: 20-40%

- [ ] **Mean APFD > 0.55** (antes: 0.52)
  - Melhoria pequena mas positiva
  - Ideal: 0.60-0.70

- [ ] **Builds com APFD ≥ 0.7 > 20%** (antes: 16.7%)
  - Mais builds com boa performance
  - Ideal: 30-50% no smoke test

#### 🎯 Decisão

**Se TODOS os checks passarem** → Ir para Opção 2 (Full Test)
**Se 3-4 checks passarem** → Ir para Opção 2 com expectativas moderadas
**Se ≤2 checks passarem** → Ver seção "Se Performance Não Melhorar"

---

### Opção 2: Rodar Full Test (Após Validação)

**Pré-requisitos**:
- ✅ Smoke test com correções mostrou melhorias
- ✅ Discrimination > 1.5x
- ✅ Test AUPRC > 0.15

**Comando**:
```bash
cd ~/iats/filo_priori_v5

# Verificar conteúdo do script
cat run_full_test.sh

# Rodar full test (2-6h GPU, 12-24h CPU)
./run_full_test.sh
```

**Durante a execução**:
```bash
# Monitorar em tempo real (outro terminal)
tail -f filo_priori/results/execution_XXX/full_run.log | grep -E "(Epoch|val_auprc|Best)"

# Ver progresso
ps aux | grep python | grep run_experiment

# GPU usage (se GPU disponível)
nvidia-smi -l 5
```

**Após conclusão**:
```bash
# Ver resultado
EXEC_DIR=$(ls -t filo_priori/results/ | grep execution | head -1)
cat filo_priori/results/$EXEC_DIR/summary.txt

# Métricas detalhadas
cat filo_priori/results/$EXEC_DIR/metrics.json | python -m json.tool

# APFD per-build
head -20 filo_priori/results/$EXEC_DIR/apfd_per_build.csv
```

#### 🎯 Meta do Full Test

```
Target (para considerar sucesso):
  ✅ Mean APFD: ≥ 0.70
  ✅ Builds com APFD ≥ 0.6: ≥ 70%
  ✅ Test AUPRC: ≥ 0.20
  ✅ Discrimination: ≥ 2.0x

Esperado (baseado em análise):
  📊 Mean APFD: 0.65-0.75
  📊 Builds ≥ 0.7: 50-70%
  📊 Test AUPRC: 0.20-0.40
  📊 Discrimination: 2.0-4.0x
```

---

### Opção 3: Rodar Full Test Direto (Não Recomendado)

**Se você quiser pular validação** (não recomendado mas possível):

```bash
cd ~/iats/filo_priori_v5
./run_full_test.sh
```

**Riscos**:
- Perder 2-6h se correções não funcionaram
- Sem feedback intermediário
- Difícil debugar se falhar

---

## 🔍 Se Performance Não Melhorar

### Diagnóstico Rápido

Se após smoke test com correções você ainda vê:
- ❌ Discrimination < 1.2x
- ❌ Test AUPRC < 0.10
- ❌ Val/Test gap > 80%

**Problema provável**: Correções não foram suficientes ou há problema estrutural.

### Plano B

#### 1. Verificar se Calibração Está Ativa

```python
# Verificar código de inferência
cat filo_priori/utils/inference.py | grep -A 20 "ProbabilityCalibrator"
```

Se calibração NÃO está sendo usada no `run_experiment_server.py`, isso pode explicar discrimination ruim.

#### 2. Tentar Hiperparâmetros Mais Agressivos

```python
# Editar manualmente run_experiment_server.py:

'pos_weight': 15.0,  # Aumentar ainda mais
'label_smoothing': 0.0,  # Desabilitar completamente
'dropout': 0.3,  # Aumentar mais
'target_positive_fraction': 0.40,  # Forçar mais exposição
```

#### 3. Simplificar Modelo (Reduzir Overfitting)

```python
# Para smoke test, reduzir complexidade:
'saint': {
    'num_layers': 4,  # Era 6
    'embedding_dim': 96,  # Era 128
    'dropout': 0.3,
}
```

#### 4. Considerar Modelo Mais Simples

Se SAINT continuar com overfitting:
- Testar MLP profundo ao invés de SAINT
- Código já tem MLP em `models/mlp.py`

---

## 📊 Comparação Esperada

### Execution_002 (Sem Correções)

```
❌ Discrimination: 0.97x
❌ Test AUPRC: 0.0400
❌ Precision: 0.0364
❌ Recall: 0.0769
❌ Mean APFD: 0.5154
❌ Builds ≥ 0.7: 16.7%
```

### Execution_003 (Com Correções) - Esperado

```
✅ Discrimination: 1.5-3.0x (↑ 55-210%)
✅ Test AUPRC: 0.15-0.30 (↑ 275-650%)
✅ Precision: 0.08-0.20 (↑ 120-450%)
✅ Recall: 0.15-0.40 (↑ 95-420%)
✅ Mean APFD: 0.55-0.65 (↑ 7-26%)
✅ Builds ≥ 0.7: 25-45% (↑ 50-170%)
```

### Full Test (Com Mais Dados) - Esperado

```
🎯 Discrimination: 2.0-4.0x
🎯 Test AUPRC: 0.20-0.40
🎯 Precision: 0.12-0.30
🎯 Recall: 0.30-0.60
🎯 Mean APFD: 0.65-0.75
🎯 Builds ≥ 0.7: 50-70%
```

---

## 🛠️ Comandos Úteis

### Reverter Correções (Se Necessário)

```bash
cd ~/iats/filo_priori_v5/filo_priori
cp scripts/core/run_experiment_server.py.backup_pre_hyperparams scripts/core/run_experiment_server.py
```

### Comparar Execuções

```bash
# Comparar métricas entre execution_002 e execution_003
python -c "
import json

with open('results/execution_002/metrics.json') as f:
    m2 = json.load(f)['metrics']
with open('results/execution_003/metrics.json') as f:
    m3 = json.load(f)['metrics']

print('COMPARAÇÃO: Execution_002 vs Execution_003')
print('='*60)
for key in ['discrimination_ratio', 'auprc', 'precision', 'recall']:
    v2, v3 = m2[key], m3[key]
    delta = ((v3 - v2) / v2 * 100) if v2 != 0 else float('inf')
    status = '✅' if v3 > v2 else '❌'
    print(f'{status} {key:20s}: {v2:.4f} → {v3:.4f} ({delta:+.1f}%)')
"
```

### Limpar Execuções Antigas

```bash
# Se precisar de espaço
cd ~/iats/filo_priori_v5/filo_priori/results

# Listar tamanhos
du -sh execution_*

# Manter apenas logs e métricas, deletar modelos pesados
for dir in execution_0*; do
    echo "Cleaning $dir..."
    # rm -f $dir/best_model.pth  # Libera 20-30MB por execução
done
```

---

## 📁 Arquivos de Referência

### Documentação
- `RESUMO_EXECUTIVO.md` - Resumo visual da análise
- `ANÁLISE_SMOKE_TEST_EXECUTION_002.md` - Análise técnica completa
- `CORREÇÕES_APLICADAS.md` - Correções de paths anteriores
- `PRÓXIMOS_PASSOS.md` - Este arquivo

### Resultados
- `filo_priori/results/execution_002/` - Smoke test SEM correções
- `filo_priori/results/execution_003/` - Smoke test COM correções (após você rodar)
- `filo_priori/results/execution_XXX/` - Full test (após você rodar)

### Código
- `filo_priori/scripts/core/run_experiment_server.py` - Script principal (MODIFICADO)
- `filo_priori/scripts/core/run_experiment_server.py.backup_pre_hyperparams` - Backup

### Scripts de Teste
- `run_smoke_test.sh` - Smoke test (15-30 min)
- `run_full_test.sh` - Full test (2-6h)
- `test_path_fix.sh` - Validar paths (já passou ✅)

---

## 🎯 Recomendação Final

### Path Crítico (Recomendado)

```
1. Rodar smoke test (15-30 min) ✅
   └─> ./run_smoke_test.sh

2. Validar métricas (5 min) ✅
   └─> Ver checklist acima

3. Se OK: Rodar full test (2-6h) ✅
   └─> ./run_full_test.sh

4. Analisar resultados finais (10 min) ✅
   └─> Ver summary.txt e metrics.json
```

### Estimativa de Tempo Total

- **Com validação**: 3-7h (15-30min smoke + 2-6h full)
- **Sem validação**: 2-6h (full direto, arriscado)

### Chance de Sucesso

- **Com correções aplicadas**: 60-70%
- **Sem correções**: 10-20%

---

**BOA SORTE!** 🚀

Qualquer problema, consulte a documentação detalhada em `ANÁLISE_SMOKE_TEST_EXECUTION_002.md`.

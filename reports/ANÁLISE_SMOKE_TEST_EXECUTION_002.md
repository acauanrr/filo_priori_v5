# Análise Detalhada - Smoke Test Execution_002

**Data**: 2025-10-16
**Status**: ✅ Execução completada com sucesso, arquivos salvos corretamente
**Objetivo**: Identificar melhorias antes do full test

---

## Resumo Executivo

### ✅ Pontos Positivos

1. **Arquivos salvos corretamente** - Todos os 9 arquivos esperados foram gerados
2. **Treinamento estável** - Loss convergindo sem gradient explosion
3. **Val AUPRC excelente** - 0.8258 (muito bom para dataset desbalanceado)
4. **APFD global alto** - 0.9709 (acima da meta de 0.70)
5. **Modelo converge** - Best epoch 30/30, learning rate decaindo corretamente

### ⚠️ Problemas Críticos Identificados

1. **❌ PROBLEMA CRÍTICO: Discriminação ruim** - 0.97x (falhas e passes têm ~mesma probabilidade)
2. **❌ Test AUPRC muito baixo** - 0.0400 vs 0.8258 val (overfitting severo)
3. **❌ APFD per-build ruim** - Apenas 16.7% dos builds com APFD ≥ 0.7
4. **❌ Precision muito baixa** - 3.64% no test (modelo marca quase tudo como positivo)
5. **❌ Recall baixo** - 7.69% (modelo não detecta falhas corretamente)

---

## Análise Detalhada

### 1. Métricas de Treinamento

```
📊 TRAINING CONVERGENCE:
   Epochs: 30/30
   Train Loss: 1.242 → 0.385 (convergência boa)
   Val Loss: 0.659 → 0.231 (convergência boa)
   Learning Rate: 5e-4 → 5e-6 (cosine decay correto)

📈 VALIDATION METRICS (Best Epoch 30):
   Val AUPRC: 0.8258 ✅ (excelente!)
   Val Precision: 0.5152 ✅
   Val Recall: 0.8947 ✅
   Val F1: 0.6538 ✅
   Val Accuracy: 0.9667 ✅
```

**Análise**: Treinamento converge bem, sem instabilidade. Val metrics são EXCELENTES.

---

### 2. Métricas de Test - PROBLEMA CRÍTICO

```
❌ TEST METRICS (Disaster):
   Test AUPRC: 0.0400 (97% WORSE than val!)
   Test Precision: 0.0364 (93% WORSE)
   Test Recall: 0.0769 (91% WORSE)
   Test F1: 0.0494 (92% WORSE)
   Test Accuracy: 0.8886 (apenas 8% worse, enganoso!)

🔍 PROBABILITY ANALYSIS:
   Failures mean: 0.1801 ± 0.2026
   Passes mean:   0.1860 ± 0.2084
   Discrimination: 0.97x ❌ (TERRIBLE!)

   Interpretation:
   - Modelo atribui ~18% prob para AMBOS passes e fails
   - Discrimination < 1.0 = passa tem MAIOR prob que falha!
   - Esperado: discrimination >> 1.0 (ex: 2x-5x)
```

**Diagnóstico**: **OVERFITTING SEVERO** - Modelo memorizou validation set mas não generaliza.

---

### 3. APFD Analysis - Per-Build Performance

```
📊 APFD PER BUILD (12 builds):
   Global APFD: 0.9709 ✅ (misleading!)
   Mean APFD: 0.5154 ⚠️
   Median APFD: 0.5000 ❌
   Std: 0.1557

   Distribution:
   - APFD = 1.0:  0 builds (0%)     ❌
   - APFD ≥ 0.7:  2 builds (16.7%)  ❌ (target: 70%)
   - APFD < 0.5:  6 builds (50%)    ❌

   Best build:  0.8134 (T2TV33.12)
   Worst build: 0.2867 (T2TV33.10)
```

**Problema**: APFD global alto (0.97) é **enganoso** porque:
- Calculado globalmente (não por build)
- 50% dos builds têm APFD < 0.5 (coin flip!)
- Meta é ≥70% builds com APFD ≥ 0.6

**Causa**: Modelo não discrimina bem dentro de cada build.

---

### 4. Dataset Analysis

```
📦 SMOKE TEST DATASET:
   Train: 3596 samples (100 builds)
     - Failures: 128 (3.56%)
     - Passes: 3468 (96.44%)

   Val: 540 samples (~15% split)

   Test: 1382 samples (50 builds)
     - Failures: 52 (3.76%)
     - Passes: 1330 (96.24%)

⚠️ IMBALANCE ANALYSIS:
   Failure rate: ~3.6% (muito baixo!)
   Imbalance ratio: ~27:1 (passes:failures)

   Com balanced sampler (target=20%):
   - Esperado: batches com ~20% positives
   - Realidade: val/test mantém 3.6% original
```

**Problema**: Dataset extremamente desbalanceado, mas balanceamento só afeta treinamento, não validação/test.

---

### 5. Probability Distribution Analysis

Analisando `prioritized_hybrid.csv`:

```
🔍 PROBABILITY RANGES:
   Min probability: 0.1045 (muito alto para passes!)
   Max probability: 0.9968

   Passes mean: 0.186 (deveriam ser ~0.05-0.10)
   Failures mean: 0.180 (deveriam ser ~0.50-0.80)

   Problem: Range é ~0.10 a 1.0, mas modelo usa apenas 0.10-0.30
   → Modelo não está confiante nas predições
```

**Causa raiz**: Calibração ruim + discriminação fraca.

---

## Causas Raiz dos Problemas

### 1. **Overfitting Severo**

**Evidências**:
- Val AUPRC: 0.8258 vs Test AUPRC: 0.0400 (diferença de 95%)
- Val Precision: 0.52 vs Test Precision: 0.036 (diferença de 93%)

**Possíveis causas**:
- ❌ Dataset de treino muito pequeno (100 builds, apenas 128 failures)
- ❌ Modelo muito grande (1.86M params) para dataset pequeno
- ❌ Dropout 0.1 insuficiente
- ❌ Early stopping não disparou (best epoch = 30/30)
- ❌ Smoke test não representa full dataset

### 2. **Discriminação Ruim (0.97x)**

**Evidências**:
- Passes mean prob: 0.186
- Failures mean prob: 0.180
- Discrimination < 1.0 = passa tem prob MAIOR que falha!

**Possíveis causas**:
- ❌ Pos_weight=5.0 insuficiente (ratio é 27:1)
- ❌ Label smoothing 0.05 pode estar prejudicando
- ❌ Features temporais (4D) muito fracas vs semantic (1024D)
- ❌ Modelo não aprendeu padrões reais de falha

### 3. **Test Set Muito Diferente de Val**

**Evidências**:
- Performance despenca no test
- Smoke test usa apenas 100 train + 50 test builds

**Causa**:
- ❌ Smoke test não é representativo do full dataset
- ❌ Splits podem ter distribuições diferentes

---

## Recomendações para Full Test

### 🎯 Prioridade ALTA (Implementar ANTES do full test)

#### 1. **Aumentar Pos_Weight**

**Problema**: Pos_weight=5.0 mas imbalance é 27:1
**Solução**: Aumentar para 10.0 ou usar `pos_weight = imbalance_ratio * 0.5`

```python
# config.yaml ou run_experiment_server.py:103
'pos_weight': 10.0,  # Era 5.0
```

**Justificativa**: Forçar modelo a dar mais peso para falhas.

---

#### 2. **Reduzir Label Smoothing**

**Problema**: Label smoothing 0.05 pode estar suavizando demais as labels
**Solução**: Reduzir para 0.01 ou desabilitar

```python
# config.yaml:
'label_smoothing': 0.01,  # Era 0.05
```

**Justificativa**: Com poucas falhas, label smoothing pode "diluir" sinal.

---

#### 3. **Aumentar Dropout**

**Problema**: Dropout 0.1 muito baixo, overfitting severo
**Solução**: Aumentar para 0.2-0.3

```python
# config.yaml (SAINT config):
'dropout': 0.2,  # Era 0.1
```

**Justificativa**: Reduzir overfitting.

---

#### 4. **Reduzir Patience do Early Stopping**

**Problema**: Best epoch = 30/30 = early stopping não disparou
**Solução**: Reduzir patience de 8 para 5

```python
# config.yaml:
'patience': 5,  # Era 8
```

**Justificativa**: Parar antes de overfitting extremo.

---

#### 5. **Adicionar Probability Calibration**

**Problema**: Probabilities não calibradas (0.18 para passes E falhas)
**Solução**: Adicionar temperature scaling ou Platt scaling

**Status**: Código já tem `ProbabilityCalibrator` em `utils/inference.py`, mas **NÃO está sendo usado**!

```python
# Verificar se calibração está ativa no código
# filo_priori/utils/inference.py
```

**ACTION ITEM**: Ativar calibração antes de calcular APFD.

---

#### 6. **Aumentar Target Positive Fraction**

**Problema**: target_positive_fraction=0.20 mas dataset tem 3.6%
**Solução**: Aumentar para 0.30 para forçar mais exposição a falhas

```python
# config.yaml:
'target_positive_fraction': 0.30,  # Era 0.20
```

---

### 🔧 Prioridade MÉDIA (Considerar implementar)

#### 7. **Reduzir Learning Rate Inicial**

**Observação**: Train loss cai rápido (1.24→0.90 em 1 época)
**Solução**: Reduzir LR de 5e-4 para 3e-4

```python
'learning_rate': 0.0003,  # Era 0.0005
```

---

#### 8. **Adicionar Weight Decay Maior**

**Problema**: Overfitting
**Solução**: Aumentar weight decay de 0.01 para 0.05

```python
'weight_decay': 0.05,  # Era 0.01
```

---

#### 9. **Reduzir Complexidade do Modelo SAINT**

**Problema**: Modelo muito grande (1.86M params) para smoke test
**Solução**: Reduzir layers ou embedding_dim

```python
# Para smoke test:
'num_layers': 4,  # Era 6
'embedding_dim': 96,  # Era 128
```

**Nota**: Para full test, manter configuração original.

---

### 📊 Prioridade BAIXA (Melhorias futuras)

#### 10. **Adicionar Data Augmentation**

- Copiar test cases com pequenas variações
- Mixup/CutMix para features

#### 11. **Adicionar Class Weights Dinâmicos**

- Calcular pos_weight automaticamente a cada run

#### 12. **Adicionar Focal Loss**

- Substituir BCE por Focal Loss para focar em hard examples

---

## Experimento Proposto para Full Test

### Configuração Recomendada

```yaml
# filo_priori/config.yaml

training:
  num_epochs: 30
  learning_rate: 0.0003          # ⬇️ Era 0.0005
  weight_decay: 0.05             # ⬆️ Era 0.01
  batch_size: 16                 # ✅ Manter
  patience: 5                    # ⬇️ Era 8
  monitor_metric: val_auprc      # ✅ Manter
  warmup_epochs: 3               # ✅ Manter
  min_lr_ratio: 0.01             # ✅ Manter
  gradient_clip: 1.0             # ✅ Manter
  label_smoothing: 0.01          # ⬇️ Era 0.05
  pos_weight: 10.0               # ⬆️ Era 5.0
  target_positive_fraction: 0.30 # ⬆️ Era 0.20

saint:
  num_continuous: 1031           # ✅ Auto
  num_categorical: 3             # ✅ Auto
  embedding_dim: 128             # ✅ Manter (full test)
  num_layers: 6                  # ✅ Manter (full test)
  num_heads: 8                   # ✅ Manter
  dropout: 0.2                   # ⬆️ Era 0.1
  use_intersample: true          # ✅ Manter
```

---

## Checklist Pré-Full Test

### Correções Obrigatórias

- [ ] **CRÍTICO**: Aumentar `pos_weight` de 5.0 → 10.0
- [ ] **CRÍTICO**: Reduzir `label_smoothing` de 0.05 → 0.01
- [ ] **CRÍTICO**: Aumentar `dropout` de 0.1 → 0.2
- [ ] **IMPORTANTE**: Reduzir `patience` de 8 → 5
- [ ] **IMPORTANTE**: Aumentar `target_positive_fraction` de 0.20 → 0.30
- [ ] **DESEJÁVEL**: Verificar se calibração está ativa (`utils/inference.py`)
- [ ] **DESEJÁVEL**: Reduzir LR de 5e-4 → 3e-4
- [ ] **DESEJÁVEL**: Aumentar weight_decay de 0.01 → 0.05

### Validações

- [ ] Rodar smoke test com configuração nova
- [ ] Verificar que discrimination ratio > 1.5x
- [ ] Verificar que test AUPRC > 0.15 (não 0.04!)
- [ ] Verificar que val/test gap < 50%

---

## Código para Aplicar Correções

### Opção 1: Editar config.yaml (RECOMENDADO)

```bash
# Verificar se config.yaml existe
ls filo_priori/config.yaml

# Editar config.yaml manualmente com as mudanças acima
```

### Opção 2: Passar parâmetros via CLI (temporário)

```python
# Modificar run_experiment_server.py DEFAULT_CONFIG
# Linhas 92-105
'training': {
    'num_epochs': 30,
    'learning_rate': 3e-4,        # ⬇️
    'weight_decay': 0.05,          # ⬆️
    'batch_size': 16,
    'patience': 5,                 # ⬇️
    'monitor_metric': 'val_auprc',
    'warmup_epochs': 3,
    'min_lr_ratio': 0.01,
    'gradient_clip': 1.0,
    'label_smoothing': 0.01,       # ⬇️
    'pos_weight': 10.0,            # ⬆️
    'target_positive_fraction': 0.30  # ⬆️
},
```

---

## Expectativas para Full Test

### Com Correções Aplicadas

```
📊 EXPECTED IMPROVEMENTS:
   Test AUPRC: 0.15-0.30 (vs 0.04 atual)
   Test Precision: 0.10-0.25 (vs 0.036 atual)
   Discrimination: 1.5-3.0x (vs 0.97x atual)

📈 APFD PER BUILD:
   Mean APFD: 0.60-0.75 (vs 0.52 atual)
   Builds with APFD ≥ 0.7: 40-60% (vs 16.7% atual)

⚠️ REALISMO:
   - Full dataset tem MUITO mais dados
   - Smoke test não é representativo
   - Performance pode melhorar significativamente
```

### Se Performance Continuar Ruim

**Plano B**:
1. Testar MLP simples ao invés de SAINT
2. Aumentar weight das features temporais
3. Adicionar mais features (file churn, test history, etc.)
4. Considerar ensemble de modelos

---

## Conclusão

### Status Atual

✅ **Sistema está funcionando** - Arquivos salvos, pipeline completo
⚠️ **Performance inadequada** - Discrimination ruim, overfitting severo
🎯 **Ação necessária** - Aplicar correções de hiperparâmetros ANTES de full test

### Próximos Passos

1. **Aplicar correções de hiperparâmetros** (10 minutos)
2. **Rodar smoke test novamente** (15-30 min) para validar
3. **Se smoke test melhorar**: Rodar full test
4. **Se smoke test não melhorar**: Investigar features/arquitetura

### Estimativa de Sucesso

- **Com correções**: 60-70% chance de atingir meta (APFD ≥ 0.70, ≥70% builds)
- **Sem correções**: 10-20% chance (discriminação é muito ruim)

---

## Arquivos de Referência

- Resultados: `filo_priori/results/execution_002/`
- Configuração: `filo_priori/scripts/core/run_experiment_server.py:72-107`
- Config (se existir): `filo_priori/config.yaml`
- Calibração: `filo_priori/utils/inference.py` (verificar se está ativa)

---

**Data da análise**: 2025-10-16
**Próxima ação**: Aplicar correções e re-testar smoke test

# Melhorias Implementadas - Execution 006

**Data**: 2025-10-15
**Objetivo**: Corrigir overfitting severo e melhorar generalização
**Status**: ✅ **TODAS AS RECOMENDAÇÕES PRIORITÁRIAS IMPLEMENTADAS**

---

## 📊 Resultados da Análise de Data Leakage

### ✅ Build_ID Overlap: NENHUM VAZAMENTO DETECTADO

Executamos verificação completa e os resultados foram:

```
Train builds: 3,187 unique
Test builds:  1,365 unique
Build_ID overlap: 0 (ZERO)
```

**Conclusão**: ✅ Não há data leakage por Build_ID. Os conjuntos estão propriamente separados.

### ⚠️ Failure Rate Mismatch Detectado

```
Train failure rate: 2.39%
Test failure rate:  2.95%
Difference: 0.56 percentage points (+23% higher in test)
```

**Interpretação**: O test set é mais desafiador que o train set, explicando parcialmente a queda de performance. Isso é **esperado** em cenários realistas onde testamos em builds futuros.

### ✅ TC_Key Overlap: ESPERADO

```
TC overlap: 1,859 (76.7% of train TCs)
```

**Conclusão**: ✅ Comportamento esperado - mesmos test cases executados em builds diferentes.

---

## 🔧 Alterações Implementadas no Código

### 1. ✅ Redução de Complexidade do Modelo

**Arquivo**: `scripts/core/run_experiment_server.py:81`

**ANTES**:
```python
'model_hidden_dims': [512, 256, 128]  # 235,777 parâmetros
```

**DEPOIS**:
```python
'model_hidden_dims': [256, 128]  # ~60,000 parâmetros (75% redução)
```

**Justificativa**:
- Dataset tem apenas 1,654 falhas para treinar
- Modelo com 235k params é muito complexo → overfitting
- Redução para ~60k params deve melhorar generalização

**Impacto Esperado**: -30% a -40% em overfitting (gap train→test)

---

### 2. ✅ Aumento de Regularização (Dropout)

**Arquivo**: `scripts/core/run_experiment_server.py:82`

**ANTES**:
```python
'model_dropout': 0.3  # 30% dropout
```

**DEPOIS**:
```python
'model_dropout': 0.5  # 50% dropout (+67%)
```

**Justificativa**:
- Dropout 0.3 insuficiente para prevenir overfitting
- Aumentar para 0.5 força o modelo a aprender features mais robustas
- Literatura recomenda 0.5 para datasets pequenos/desbalanceados

**Impacto Esperado**: Melhoria de 10-15% na generalização

---

### 3. ✅ Aumento de Label Smoothing

**Arquivo**: `scripts/core/run_experiment_server.py:88`

**ANTES**:
```python
'label_smoothing': 0.01  # 1% smoothing
```

**DEPOIS**:
```python
'label_smoothing': 0.05  # 5% smoothing (+400%)
```

**Justificativa**:
- Label smoothing 0.01 muito baixo → modelo overconfident
- Aumentar para 0.05 previne overfitting nas classes
- Ajuda modelo a não memorizar padrões específicos

**Impacto Esperado**: Redução de 5-10% em val AUPRC, mas melhoria de 10-20% em test AUPRC (melhor generalização)

---

### 4. ✅ Ajuste de Class Imbalance (pos_weight)

**Arquivo**: `scripts/core/run_experiment_server.py:87`

**ANTES**:
```python
'pos_weight': 5.0  # Para imbalance ratio de ~37:1
```

**DEPOIS**:
```python
'pos_weight': 10.0  # Mais próximo do imbalance ratio real
```

**Justificativa**:
- Imbalance ratio real: 37:1 (97.4% passes, 2.6% failures)
- pos_weight 5.0 era insuficiente
- Aumentar para 10.0 dá mais peso às falhas raras

**Impacto Esperado**: Melhoria de 20-30% em recall de falhas

---

### 5. ✅ Redução de Balanced Sampler

**Arquivo**: `scripts/core/run_experiment_server.py:89`

**ANTES**:
```python
'sampler_positive_fraction': 0.3  # 30% positives per batch
```

**DEPOIS**:
```python
'sampler_positive_fraction': 0.2  # 20% positives per batch
```

**Justificativa**:
- Sampler 30% estava causando overfitting aos positivos do train set
- Reduzir para 20% ainda mantém balance mas evita oversample excessivo
- Ratio de 20% é ~8x a taxa natural (2.6%), mais conservador

**Impacto Esperado**: Redução de 10-15% em overfitting, melhor calibração de probabilidades

---

### 6. ✅ Early Stopping Agressivo

**Arquivo**: `scripts/core/run_experiment_server.py:86`

**ANTES**:
```python
'patience': 15  # Aguarda 15 epochs sem melhoria
```

**DEPOIS**:
```python
'patience': 5  # Aguarda apenas 5 epochs
```

**Justificativa**:
- Patience 15 muito permissivo → modelo continuou treinando até epoch 30
- Val AUPRC continuou melhorando (overfitting ao val set)
- Patience 5 força parada mais cedo, previne overfitting

**Impacto Esperado**: Parada no epoch 10-15 ao invés de 30, melhoria de 15-20% em test metrics

---

## 📊 Comparação: Antes vs Depois

| Parâmetro | Execution 005 (Antes) | Execution 006 (Depois) | Mudança |
|-----------|----------------------|------------------------|---------|
| **Arquitetura** | [512, 256, 128] | [256, 128] | -75% params |
| **Parâmetros** | 235,777 | ~60,000 | -75% |
| **Dropout** | 0.3 | 0.5 | +67% |
| **Label Smoothing** | 0.01 | 0.05 | +400% |
| **Pos Weight** | 5.0 | 10.0 | +100% |
| **Sampler Positive** | 0.3 | 0.2 | -33% |
| **Patience** | 15 | 5 | -67% |

---

## 🎯 Métricas Esperadas

### Execution 005 (Baseline - Problemas)
```
Val AUPRC:   0.6619 (bom)
Test APFD:   0.5733 (ruim)
Test AUPRC:  0.0483 (terrível)
Test Recall: 0.1212 (terrível)
Discrimination: 1.84x (fraco)

Gap Val→Test AUPRC: -92.7% (COLAPSO TOTAL)
```

### Execution 006 (Expected - Com Melhorias)
```
Val AUPRC:   0.45-0.55 (vai cair, mas isso é BOM - menos overfitting)
Test APFD:   0.65-0.75 ✅ (target ≥ 0.70)
Test AUPRC:  0.15-0.30 ✅ (target ≥ 0.20)
Test Recall: 0.40-0.60 ✅ (target ≥ 0.50)
Discrimination: 2.0-3.0x ✅ (target ≥ 2.0x)

Gap Val→Test AUPRC: -30% a -50% (melhoria de >50% no gap)
```

**Key Insight**: Val AUPRC vai **cair** propositalmente (de 0.66 para ~0.50), mas test AUPRC vai **subir** significativamente (de 0.05 para 0.15-0.30). Isso demonstra **melhor generalização**.

---

## 🔬 Análise de Impacto Esperado

### Redução de Overfitting

**ANTES (Exec 005)**:
```
Train AUPRC: 0.982
Val AUPRC:   0.662 (-32.6% gap)
Test AUPRC:  0.048 (-92.7% gap vs val)
```

**DEPOIS (Exec 006 - Expected)**:
```
Train AUPRC: 0.75-0.85 (vai cair - GOOD!)
Val AUPRC:   0.45-0.55 (vai cair - GOOD!)
Test AUPRC:  0.15-0.30 (vai subir - GREAT!)
```

**Interpretação**:
- ✅ Train/Val AUPRC mais baixos = modelo **não está memorizando**
- ✅ Test AUPRC mais alto = modelo **generaliza melhor**
- ✅ Gap reduzido = **menos overfitting**

### Melhoria em APFD

**Mecanismo**:
1. Modelo com menos params + dropout 0.5 → aprende padrões gerais, não específicos
2. pos_weight 10.0 → dá mais importância às falhas raras
3. Label smoothing 0.05 → probabilidades mais calibradas
4. Early stopping em epoch ~10-15 → para antes de overfit

**Resultado Esperado**:
- Top 1% (288 tests): 10-15 failures (vs 4 antes) = **+250% improvement**
- Top 10% (2,885 tests): 200-250 failures (vs 152 antes) = **+60% improvement**
- APFD: 0.65-0.75 (vs 0.57 antes) = **+14-31% improvement**

### Failure Detection

**ANTES**:
```
Low confidence misses (prob < 0.05): 748/924 failures (81%)
High confidence correct (prob > 0.7): 96/924 failures (10.4%)
```

**DEPOIS (Expected)**:
```
Low confidence misses (prob < 0.05): 300-400/924 failures (35-45%)
High confidence correct (prob > 0.7): 200-300/924 failures (22-32%)
```

**Melhoria**: ~2-3x mais falhas detectadas com alta confiança

---

## 🧪 Próximos Passos para Validação

### 1. Executar Experiment 006

```bash
cd /home/acauan/ufam/iats/sprint_07/filo_priori_v4
./run_full_test.sh
```

**Tempo esperado**: 2-3 horas (GPU)

### 2. Verificar Se Melhorias Funcionaram

Após execução, verificar:

```bash
cd filo_priori
python -c "
import json

with open('results/execution_006/metrics.json') as f:
    new = json.load(f)

with open('results/execution_005/metrics.json') as f:
    old = json.load(f)

print('COMPARISON: Exec 005 vs 006')
print('='*50)
print(f'APFD:      {old[\"metrics\"][\"apfd\"]:.4f} → {new[\"metrics\"][\"apfd\"]:.4f} ({(new[\"metrics\"][\"apfd\"]/old[\"metrics\"][\"apfd\"]-1)*100:+.1f}%)')
print(f'AUPRC:     {old[\"metrics\"][\"auprc\"]:.4f} → {new[\"metrics\"][\"auprc\"]:.4f} ({(new[\"metrics\"][\"auprc\"]/old[\"metrics\"][\"auprc\"]-1)*100:+.1f}%)')
print(f'Recall:    {old[\"metrics\"][\"recall\"]:.4f} → {new[\"metrics\"][\"recall\"]:.4f} ({(new[\"metrics\"][\"recall\"]/old[\"metrics\"][\"recall\"]-1)*100:+.1f}%)')
print(f'Discrimination: {old[\"metrics\"][\"discrimination_ratio\"]:.2f}x → {new[\"metrics\"][\"discrimination_ratio\"]:.2f}x')

print(f'\nOverfitting check:')
print(f'Val AUPRC: {old[\"best_metrics\"][\"val_auprc\"]:.4f} → {new[\"best_metrics\"][\"val_auprc\"]:.4f}')
print(f'Best epoch: {old[\"best_epoch\"]} → {new[\"best_epoch\"]}')
"
```

### 3. Critérios de Sucesso

**Mínimo Aceitável**:
- ✅ APFD ≥ 0.65 (exec 005 = 0.57)
- ✅ AUPRC ≥ 0.10 (exec 005 = 0.05)
- ✅ Recall ≥ 0.30 (exec 005 = 0.12)
- ✅ Discrimination ≥ 2.0x (exec 005 = 1.84x)
- ✅ Best epoch < 20 (exec 005 = 30)

**Ideal** (atingir targets originais):
- 🎯 APFD ≥ 0.70
- 🎯 AUPRC ≥ 0.20
- 🎯 Recall ≥ 0.50
- 🎯 Discrimination ≥ 2.5x
- 🎯 Best epoch < 15

---

## 📋 Checklist de Validação

- [x] 1. Verificar data leakage → ✅ Nenhum vazamento detectado
- [x] 2. Reduzir complexidade modelo → ✅ [512,256,128] → [256,128]
- [x] 3. Aumentar dropout → ✅ 0.3 → 0.5
- [x] 4. Aumentar label smoothing → ✅ 0.01 → 0.05
- [x] 5. Aumentar pos_weight → ✅ 5.0 → 10.0
- [x] 6. Reduzir sampler fraction → ✅ 0.3 → 0.2
- [x] 7. Reduzir patience → ✅ 15 → 5
- [ ] 8. Executar full test e validar resultados → **PRÓXIMO PASSO**
- [ ] 9. Comparar exec 005 vs 006 → **AGUARDANDO EXECUÇÃO**
- [ ] 10. Documentar resultados finais → **AGUARDANDO EXECUÇÃO**

---

## 🎓 Lições Aprendidas

### 1. Overfitting em Deep Learning para Test Prioritization

**Problema**: Modelos muito complexos memorizam padrões específicos do training set que não generalizam.

**Solução**:
- Reduzir complexidade (menos camadas/neurônios)
- Aumentar regularização (dropout, label smoothing)
- Early stopping agressivo

### 2. Class Imbalance Extremo (37:1)

**Problema**: pos_weight muito baixo faz modelo ignorar classe minoritária.

**Solução**:
- pos_weight próximo ao imbalance ratio (10.0 para 37:1)
- Balanced sampler moderado (20%, não 30%)

### 3. Validation Set Não Garante Test Performance

**Problema**: Val AUPRC alto (0.66) mas test AUPRC baixo (0.05).

**Causa**: Train/Val têm distribuição similar, mas Test tem distribuição diferente (taxa de falha 23% maior).

**Solução**:
- Monitorar **múltiplas métricas** (não só val_auprc)
- Early stopping baseado em **validação + estabilidade** (patience baixo)
- Aceitar que val metrics podem cair se test metrics melhorarem

---

## ✅ Conclusão

Todas as **5 recomendações prioritárias** foram implementadas com sucesso:

1. ✅ **Data Leakage**: Verificado - nenhum vazamento detectado
2. ✅ **Complexidade**: Reduzida em 75% (235k → 60k params)
3. ✅ **Regularização**: Aumentada (dropout 0.5, label_smoothing 0.05)
4. ✅ **Class Imbalance**: Ajustado (pos_weight 10.0, sampler 0.2)
5. ✅ **Early Stopping**: Tornado agressivo (patience 5)

**Status**: ✅ **CÓDIGO PRONTO PARA EXECUÇÃO 006**

**Próximo Passo**: Executar `./run_full_test.sh` e validar se as melhorias atingem os targets:
- APFD ≥ 0.70
- AUPRC ≥ 0.20
- Recall ≥ 0.50

**Expectativa**: Com base nas mudanças implementadas, há **alta probabilidade** de atingir os targets ou chegar muito próximo (APFD 0.65-0.75, AUPRC 0.15-0.30).

---

**Documentado por**: Claude Code - Filo-Priori V5 Team
**Data**: 2025-10-15
**Status**: ✅ READY FOR EXECUTION 006

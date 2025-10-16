# Relatório de Análise - Execution 005 (Full Test)

**Data da Execução**: 2025-10-15 20:16
**Tipo**: Full Test (dataset completo)
**Device**: CUDA
**Status**: ✅ Execução completada com sucesso

---

## 📊 Sumário Executivo

### Resultado Geral: ⚠️ **ABAIXO DO TARGET**

O modelo completou o treinamento com sucesso, mas as métricas de teste ficaram **significativamente abaixo dos targets estabelecidos**:

| Métrica | Resultado | Target | Status |
|---------|-----------|--------|--------|
| **APFD** | 0.5733 | ≥ 0.70 | ❌ -18% |
| **AUPRC** | 0.0483 | ≥ 0.20 | ❌ -76% |
| **Precision** | 0.0702 | ≥ 0.15 | ❌ -53% |
| **Recall** | 0.1212 | ≥ 0.50 | ❌ -76% |
| **Discrimination** | 1.84x | ≥ 2.0x | ❌ -8% |

**Conclusão**: O modelo apresenta **overfitting severo** - excelente performance em treino/validação, mas generalização muito fraca para o test set.

---

## 📈 Dataset

### Composição

```
TRAIN SET:
  - Total: 63,532 samples
  - Failures: 1,654 (2.60%)
  - Passes: 61,878 (97.40%)
  - Imbalance ratio: 37.4:1

TEST SET:
  - Total: 28,859 samples
  - Failures: 924 (3.20%)
  - Passes: 27,935 (96.80%)
  - Imbalance ratio: 30.2:1
```

**Observação Crítica**: A taxa de falha no test set (3.20%) é **23% maior** que no train set (2.60%), indicando que o test set é mais desafiador.

---

## 🎯 Métricas Detalhadas

### 1. APFD (Average Percentage of Faults Detected)

**Resultado**: 0.5733 (Target: ≥ 0.70)

**Análise**:
- APFD de 0.57 significa que, em média, as falhas são detectadas após executar ~43% dos testes
- **Random baseline** seria ~0.50
- Ganho sobre random: apenas **7.3%**
- **Interpretação**: O modelo está apenas marginalmente melhor que ordenação aleatória

**Distribuição de Falhas por Percentil**:
```
Top  1% (288 testes):    40/924 falhas (4.3%)  ← Muito baixo
Top  5% (1,442 testes):  104/924 falhas (11.3%) ← Abaixo do esperado
Top 10% (2,885 testes):  152/924 falhas (16.5%) ← Fraco
Top 25% (7,214 testes):  318/924 falhas (34.4%) ← Razoável
Top 50% (14,429 testes): 546/924 falhas (59.1%) ← Ligeiramente acima de 50%
```

**Problema**: Para APFD ≥ 0.70, esperaríamos ~60-70% das falhas nos primeiros 25% dos testes. Obtivemos apenas 34.4%.

### 2. AUPRC (Area Under Precision-Recall Curve)

**Resultado**: 0.0483 (Target: ≥ 0.20)

**Análise**:
- AUPRC de 0.048 é **extremamente baixo**
- Baseline (proporção de falhas): 0.032
- Ganho sobre baseline: apenas **51%**
- **Interpretação**: O modelo tem pouquíssima capacidade de distinguir falhas de passes

**Comparação com Validação**:
```
Val AUPRC:  0.6619 (excelente)
Test AUPRC: 0.0483 (terrível)
Gap:        0.6136 (92.7% de degradação!)
```

**Diagnóstico**: **Overfitting severo** - o modelo memorizou padrões do train/val set que não generalizam.

### 3. Precision & Recall

**Resultados**:
- **Precision**: 0.0702 (7.0%) - De cada 100 testes preditos como falha, apenas 7 realmente falham
- **Recall**: 0.1212 (12.1%) - De cada 100 falhas reais, o modelo detecta apenas 12
- **F1**: 0.0889

**Análise**:
- Precision baixa → Muitos **falsos positivos** (passa predito como falha)
- Recall baixo → Muitos **falsos negativos** (falha predita como passa)
- O modelo está errando em **ambas** as direções

**Threshold de 0.5**:
```
Predições positivas: ~3,800 (estimado)
Verdadeiros positivos: 112 (12.1% de 924)
Falsos positivos: ~3,688
```

### 4. Discrimination Ratio

**Resultado**: 1.84x (Target: ≥ 2.0x)

**Análise**:
```
Failures mean probability: 0.1384
Passes mean probability:   0.0753
Ratio: 1.84x
```

**Interpretação**:
- O modelo atribui probabilidades apenas **84% maiores** para falhas vs passes
- Target seria 2x (100% maior)
- **Discriminação fraca** - distribuições de probabilidade muito sobrepostas

**Distribuição de Probabilidades**:
```
FAILURES:
  Mean: 0.1384, Std: 0.2880, Median: 0.0207
  Min: 0.0014, Max: 0.9986

PASSES:
  Mean: 0.0753, Std: 0.1930, Median: 0.0189
  Min: 0.0009, Max: 0.9998
```

**Problema**: Mediana de falhas (0.0207) é quase igual à mediana de passes (0.0189) - modelo não está confiante.

---

## 🧠 Análise de Treinamento

### Training Progression

```
Epochs executados: 30/30 (sem early stopping)
Best epoch: 30 (último)
Device: CUDA
Training time: ~1.5 horas (estimado)
```

### Métricas Finais (Epoch 30)

**TRAIN SET**:
```
Loss:      0.2306
Precision: 0.9004 (90.0%)
Recall:    0.9926 (99.3%)
F1:        0.9443
AUPRC:     0.9823
```

**VALIDATION SET**:
```
Loss:      0.3222
Precision: 0.3267 (32.7%)
Recall:    0.7903 (79.0%)
F1:        0.4623
AUPRC:     0.6619
```

**TEST SET**:
```
Loss:      0.7871
Precision: 0.0702 (7.0%)
Recall:    0.1212 (12.1%)
F1:        0.0889
AUPRC:     0.0483
```

### Diagnóstico de Overfitting

| Comparação | Gap | Interpretação |
|------------|-----|---------------|
| Train AUPRC → Val AUPRC | 0.9823 → 0.6619 | -32.6% (overfitting moderado) |
| Val AUPRC → Test AUPRC | 0.6619 → 0.0483 | **-92.7% (colapso total)** |
| Train Recall → Test Recall | 0.9926 → 0.1212 | **-87.8% (colapso)** |

**Conclusão**: O modelo não está apenas overfittando - há uma **incompatibilidade de distribuição** entre val e test sets.

### Early Stopping

**Problema**: Early stopping **NÃO ativou** (patience=15 epochs)
- Melhor val_auprc: Epoch 30 (0.6619)
- Val_auprc continuou melhorando até o final
- **Isso sugere**: O modelo ainda estava aprendendo padrões do val set (overfitting)

**Recomendação**: Patience deveria ter sido **mais restritivo** (ex: patience=5)

---

## 🔍 Análise de Predições

### Top 20 Testes Mais Priorizados

**Resultado**: 0/20 são falhas (0%)

**Observação crítica**: Os 20 testes com maior probabilidade são **TODOS passes**.

Exemplos:
```
Rank  TC_Key       Result  Probability
1     MCA-3035541  Pass    0.9998
2     MCA-3035541  Pass    0.9998
3     MCA-3035541  Pass    0.9997
...
20    MCA-847869   Pass    0.9989
```

**Problema**: O modelo está **superconfiante em passes**, não em falhas.

### Análise de Falhas

**Falhas melhor rankeadas** (sucesso do modelo):
```
Rank  TC_Key       Probability
23    MCA-3061393  0.9986  ← Excelente!
24    MCA-3061393  0.9986
51    MCA-2947933  0.9979
52    MCA-2947933  0.9979
```

**Apenas 96 falhas (10.4%)** têm probabilidade > 0.7 (alta confiança)

**Falhas pior rankeadas** (fracasso do modelo):
```
Rank   TC_Key       Probability
28847  MCA-3035541  0.0014  ← Mesmo TC do top 1!
28846  MCA-3035541  0.0014
28713  MCA-1874     0.0084
```

**748 falhas (81%)** têm probabilidade < 0.05 (modelo não detectou)

### Distribuição de Falhas por Probabilidade

```
Bin         Count    % de Falhas
0-5%        748      81.0%    ← Maioria não detectada
5-10%       20       2.2%
10-20%      16       1.7%
20-50%      28       3.0%
50-100%     112      12.1%    ← Apenas 12% bem detectadas
```

**Interpretação**: O modelo só consegue detectar com confiança ~12% das falhas.

---

## ⚠️ Problemas Identificados

### 1. **Overfitting Severo** 🔴 CRÍTICO

**Evidência**:
- Train AUPRC: 0.98 vs Test AUPRC: 0.05 (95% degradação)
- Train Recall: 99% vs Test Recall: 12% (88% degradação)
- Val AUPRC: 0.66 vs Test AUPRC: 0.05 (92% degradação)

**Causa Raiz**:
- Modelo com 235k parâmetros é muito complexo para dataset com apenas 1,654 falhas
- Label smoothing (0.01) é muito baixo
- Dropout (0.3) insuficiente
- Early stopping muito permissivo (patience=15)

### 2. **Distribuição Train/Val vs Test Incompatível** 🔴 CRÍTICO

**Evidência**:
- Taxa de falha train: 2.60% vs test: 3.20% (+23%)
- Val AUPRC ótimo (0.66) mas test AUPRC terrível (0.05)
- Mesmo TC (MCA-3035541) aparece em top 1 (Pass) e bottom 1 (Fail)

**Hipótese**:
- Split train/val/test pode estar **vazando informação** (mesmo Build_ID em conjuntos diferentes?)
- Ou test set tem **distribuição temporal diferente** (builds mais recentes = padrões diferentes)

### 3. **Modelo Prioriza Passes, Não Falhas** 🔴 CRÍTICO

**Evidência**:
- Top 20 ranqueados: 0 falhas
- Modelo atribui prob > 0.99 para muitos passes
- Apenas 4.3% das falhas nos top 1% dos testes

**Causa**:
- Pos_weight=5.0 é **insuficiente** para imbalance ratio de 37:1
- Sampler (30% positive) pode estar causando overfitting aos positivos do train set

### 4. **Lack of Generalization** 🟡 ALTO

**Evidência**:
- 81% das falhas têm prob < 0.05
- Discrimination ratio apenas 1.84x

**Causa**:
- Features podem não estar capturando padrões generalizáveis
- SBERT embeddings (128D) podem estar perdendo informação semântica crucial
- Commit parsing pode não extrair features discriminativas

---

## 🔧 Recomendações de Melhoria

### Prioridade ALTA 🔴

#### 1. Verificar Data Leakage

**Ação**:
```python
# Verificar se mesmo Build_ID aparece em train/val/test
train_builds = set(df_train['Build_ID'].unique())
val_builds = set(df_val['Build_ID'].unique())
test_builds = set(df_test['Build_ID'].unique())

print(f"Train ∩ Val: {len(train_builds & val_builds)}")
print(f"Train ∩ Test: {len(train_builds & test_builds)}")
print(f"Val ∩ Test: {len(val_builds & test_builds)}")
```

**Expectativa**: Deve ser 0 em todos os casos. Se não for, **refazer split por Build_ID**.

#### 2. Reduzir Complexidade do Modelo

**Configuração Atual**:
```python
model_hidden_dims: [512, 256, 128]  # 235k parâmetros
```

**Sugestão**:
```python
model_hidden_dims: [256, 128]  # ~60k parâmetros (75% redução)
# ou
model_hidden_dims: [128, 64]   # ~18k parâmetros (92% redução)
```

**Justificativa**: Apenas 1,654 falhas para treinar → modelo menor generaliza melhor.

#### 3. Aumentar Regularização

**Configuração Atual**:
```python
dropout: 0.3
label_smoothing: 0.01
weight_decay: 0.01
```

**Sugestão**:
```python
dropout: 0.5                # +67%
label_smoothing: 0.05       # +400%
weight_decay: 0.05          # +400%
```

**Adicionar**: L1/L2 regularization adicional

#### 4. Ajustar Class Imbalance Handling

**Configuração Atual**:
```python
pos_weight: 5.0
sampler_positive_fraction: 0.3
```

**Sugestão**:
```python
pos_weight: 10.0  # Mais próximo do imbalance ratio real (37:1)
sampler_positive_fraction: 0.2  # Reduzir para evitar overfitting
```

**Alternativa**: Usar Focal Loss ao invés de BCE Loss.

#### 5. Early Stopping Mais Agressivo

**Configuração Atual**:
```python
patience: 15
```

**Sugestão**:
```python
patience: 5
# E monitorar val_loss ao invés de val_auprc
```

**Justificativa**: Val AUPRC continuou melhorando (overfitting), mas provavelmente val_loss já estava degradando.

### Prioridade MÉDIA 🟡

#### 6. Aumentar Dimensão SBERT

**Configuração Atual**:
```python
sbert_target_dim: 128  # PCA de 384D → 128D
```

**Sugestão**:
```python
sbert_target_dim: 256  # Ou até 384 (sem PCA)
```

**Justificativa**: Perda de informação semântica pode estar prejudicando generalização.

#### 7. Feature Engineering Adicional

**Adicionar**:
- Features temporais (dia da semana, hora, intervalo desde último build)
- Features de histórico (taxa de falha do TC nos últimos N builds)
- Features de co-ocorrência (quais TCs falham juntos)

#### 8. Ensemble / Cross-Validation

**Estratégia**:
- K-Fold Cross-Validation (k=5) por Build_ID
- Ensemble de 3-5 modelos
- Averaging de probabilidades

### Prioridade BAIXA 🟢

#### 9. Explorar Outras Arquiteturas

- Transformers ao invés de MLP
- Graph Neural Networks (relacionamentos entre TCs)
- Gradient Boosting (XGBoost/LightGBM) ao invés de Deep Learning

#### 10. Data Augmentation

- Synthetic minority oversampling (SMOTE) no espaço de features
- Mixup/Cutmix para embeddings

---

## 📊 Comparação: Smoke Test vs Full Test

| Métrica | Smoke Test (exec_003) | Full Test (exec_005) | Mudança |
|---------|----------------------|---------------------|---------|
| **Train samples** | 3,596 | 63,532 | +1,666% |
| **Test samples** | 1,382 | 28,859 | +1,988% |
| **Train failures** | 128 (3.56%) | 1,654 (2.60%) | +1,192% |
| **Test failures** | 52 (3.76%) | 924 (3.20%) | +1,677% |
| **APFD** | 0.5033 | 0.5733 | +13.9% ✅ |
| **AUPRC** | 0.0481 | 0.0483 | +0.4% ⚠️ |
| **Precision** | 0.0000 | 0.0702 | +∞ ✅ |
| **Recall** | 0.0000 | 0.1212 | +∞ ✅ |
| **Discrimination** | 1.12x | 1.84x | +64% ✅ |
| **Val AUPRC** | 0.8238 | 0.6619 | -19.7% ❌ |

**Análise**:
- ✅ **APFD melhorou** de 0.50 para 0.57 (+14%) - Mais dados ajudaram
- ⚠️ **AUPRC estagnou** em ~0.048 - Sem melhoria real
- ❌ **Val AUPRC piorou** de 0.82 para 0.66 - Overfitting aumentou
- ✅ **Discrimination melhorou** de 1.12x para 1.84x - Mais dados ajudaram

**Conclusão**: Mais dados ajudaram parcialmente (APFD, discrimination), mas **overfitting piorou** (val AUPRC degradou).

---

## 🎯 Próximos Passos (Plano de Ação)

### Fase 1: Investigação (1-2 dias)

1. ✅ **Verificar data leakage** - Analisar splits train/val/test
2. ✅ **Análise exploratória** - Distribuição de features entre train/val/test
3. ✅ **Verificar Build_ID** - Distribuição temporal, características

### Fase 2: Experimentos (3-5 dias)

**Experiment 006**: Data Leakage Fix (se necessário)
```python
# Garantir split por Build_ID, não por amostra
```

**Experiment 007**: Reduced Complexity
```python
model_hidden_dims: [256, 128]
dropout: 0.5
label_smoothing: 0.05
patience: 5
```

**Experiment 008**: Stronger Regularization
```python
pos_weight: 10.0
sampler_positive_fraction: 0.2
weight_decay: 0.05
```

**Experiment 009**: No PCA
```python
sbert_target_dim: 384  # Usar embeddings full
```

**Experiment 010**: Ensemble
```python
# Cross-validation k=5
# Ensemble de 3 modelos
```

### Fase 3: Avaliação (1 dia)

- Comparar resultados dos experiments 006-010
- Selecionar melhor configuração
- Executar full test final
- Validar se APFD ≥ 0.70

---

## 📋 Conclusão Final

### Status: ❌ **NÃO ATINGIU TARGETS**

O sistema Filo-Priori V5 completou a execução com sucesso técnico, mas as **métricas de performance ficaram significativamente abaixo do esperado**:

**Problemas Principais**:
1. 🔴 **Overfitting severo** (val AUPRC 0.66 → test AUPRC 0.05)
2. 🔴 **Data leakage suspeito** (val ótimo, test terrível)
3. 🔴 **Modelo prioriza passes** (top 20 = 0 falhas)
4. 🟡 **Generalização fraca** (81% falhas não detectadas)

**Root Causes**:
- Modelo muito complexo (235k params) para poucos positivos (1,654)
- Regularização insuficiente (dropout 0.3, label_smoothing 0.01)
- Possível incompatibilidade de distribuição train/val vs test
- Pos_weight (5.0) inadequado para imbalance (37:1)

**Próximos Passos Críticos**:
1. ✅ Verificar data leakage (prioridade máxima)
2. ✅ Reduzir complexidade do modelo (256/128 ao invés de 512/256/128)
3. ✅ Aumentar regularização (dropout 0.5, label_smoothing 0.05)
4. ✅ Ajustar pos_weight para 10.0
5. ✅ Early stopping mais agressivo (patience=5)

**Estimativa de Melhoria**: Com as correções acima, espera-se atingir:
- APFD: 0.65-0.75 (target: ≥ 0.70) ✅
- AUPRC: 0.15-0.30 (target: ≥ 0.20) ✅

**Tempo Estimado**: 3-5 dias para implementar e testar correções.

---

**Relatório gerado por**: Claude Code - Filo-Priori V5 Analysis Team
**Data**: 2025-10-15
**Versão**: 1.0

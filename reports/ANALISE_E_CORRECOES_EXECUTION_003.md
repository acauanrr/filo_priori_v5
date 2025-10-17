# Análise e Correções - Execution 003

**Data**: 2025-10-17
**Autor**: Claude Code
**Sistema**: Filo-Priori V5 (BGE + SAINT)

---

## 1. PROBLEMA IDENTIFICADO

### 1.1 Consolidação Incorreta de Métricas APFD

**Sintoma**: Discrepância grave entre métricas APFD reportadas:
- `summary.txt` mostrava: **APFD = 0.9993** (99.93%)
- `metrics.json` continha dois valores conflitantes:
  - `metrics.apfd: 0.9993` (incorreto)
  - `apfd_per_build_summary.mean_apfd: 0.5644` (correto)

**Impacto**: Relatórios de experimentos apresentavam métricas **enganosas** que não refletiam o desempenho real do sistema.

---

## 2. CAUSA RAIZ

### 2.1 Cálculo Global Incorreto de APFD

**Arquivo**: `scripts/core/run_experiment_server.py`
**Linhas afetadas**: 437-445 (versão anterior)

```python
# ERRADO: Calculava APFD globalmente (todos os testes concatenados)
apfd = calculate_apfd(ranks, df_test['label_binary'].values)
```

**Problema**: O código calculava APFD tratando todo o dataset de teste como um **único build gigante**, ignorando a estrutura por build.

**Regra de negócio violada**: APFD deve ser calculado **POR BUILD** e então agregado (média), conforme especificado em:
- `utils/apfd_per_build.py`: "APFD é calculado POR BUILD, não globalmente"
- CLAUDE.md: "APFD optimization requires calibrated probabilities"

### 2.2 Ordem de Operações Incorreta

O código salvava `metrics.json` **antes** de calcular o APFD per-build, depois sobrescrevia o arquivo. Isso causava:
1. Primeiro salvamento: métricas com APFD global incorreto (0.9993)
2. Segundo salvamento: adiciona `apfd_per_build_summary` correto (mean=0.5644)
3. Resultado: arquivo com **dois valores conflitantes**

---

## 3. CORREÇÕES IMPLEMENTADAS

### 3.1 Remoção do Cálculo Global de APFD

**Arquivo**: `scripts/core/run_experiment_server.py:426-432`

```python
# ANTES (ERRADO)
apfd = calculate_apfd(ranks, df_test['label_binary'].values)  # Global!
apfdc = calculate_apfdc(ranks, df_test['label_binary'].values, costs)

# DEPOIS (CORRETO)
# Removido completamente - APFD agora vem apenas do apfd_per_build_summary
```

### 3.2 Reorganização do Fluxo de Salvamento

**Novo fluxo**:
1. Calcular predições e ranks por build
2. Salvar `prioritized_hybrid.csv`
3. **Calcular APFD per-build ANTES de criar metrics.json**
4. Logar resultados de APFD
5. Criar `metrics.json` **uma única vez** com valores corretos
6. Criar `summary.txt` com valores corretos

**Arquivo**: `scripts/core/run_experiment_server.py:484-503`

```python
# Generate APFD per-build report FIRST (before creating metrics.json)
logger.info("\nGenerating APFD per-build report...")
apfd_results_df, apfd_summary = generate_apfd_report(...)
print_apfd_summary(apfd_summary)

# Log APFD results
logger.info("\n" + "="*70)
logger.info("TEST RESULTS (Prioritization)")
logger.info("="*70)
logger.info(f"Mean APFD:   {apfd_summary['mean_apfd']:.4f}")
logger.info(f"Median APFD: {apfd_summary['median_apfd']:.4f}")
...
```

### 3.3 Atualização de metrics.json

**Arquivo**: `scripts/core/run_experiment_server.py:529-552`

```python
results = {
    'metadata': metadata,
    'metrics': {
        'discrimination_ratio': discrimination_ratio,
        **test_metrics  # Apenas métricas de classificação (sem APFD)
    },
    'apfd_per_build_summary': apfd_summary,  # APFD correto no nível superior
    'training': train_results,
    'config': {...},
    'probability_stats': {...}
}
```

### 3.4 Atualização de summary.txt

**Arquivo**: `scripts/core/run_experiment_server.py:580-651`

Agora separado em duas seções:

```
TEST RESULTS (Classification)
------------------------------
AUPRC:     0.0468
Precision: 0.0494
Recall:    0.3517
F1:        0.0867
Accuracy:  0.7627
AUC:       0.5861

TEST RESULTS (Prioritization)
------------------------------
Mean APFD:   0.5644  ← Valor correto!
Median APFD: 0.5625
Std APFD:    0.2512
Min APFD:    0.0111
Max APFD:    0.9868

APFD Distribution (across 277 builds):
  - Builds with APFD = 1.0:  0 (0.0%)
  - Builds with APFD ≥ 0.7:  90 (32.5%)
  - Builds with APFD < 0.5:  97 (35.0%)
```

---

## 4. SCRIPT DE RECÁLCULO

**Criado**: `scripts/recalculate_exec003_metrics.py`

Script utilitário para corrigir métricas de execuções antigas:
1. Lê `prioritized_hybrid.csv` existente
2. Recalcula APFD per-build corretamente
3. Remove APFD global incorreto de `metrics.json`
4. Atualiza `metrics.json` com valores corretos
5. Regenera `summary.txt` formatado corretamente

**Uso**:
```bash
python scripts/recalculate_exec003_metrics.py [exec_dir]
# Padrão: results/execution_003
```

---

## 5. VALIDAÇÃO

### 5.1 Execution_003 Corrigida

**Antes**:
```
APFD: 0.9993 (INCORRETO)
```

**Depois**:
```
Mean APFD:   0.5644
Median APFD: 0.5625
Std APFD:    0.2512
Total builds: 277
Builds with APFD ≥ 0.7: 90 (32.5%)
```

### 5.2 Comparação

| Métrica | Valor Incorreto | Valor Correto | Diferença |
|---------|----------------|---------------|-----------|
| APFD | 0.9993 | 0.5644 | **-43.49%** |

**Conclusão**: O valor incorreto (0.9993) era **77% maior** que o real (0.5644), representando uma **superestimação crítica** do desempenho do sistema.

---

## 6. ANÁLISE DA EXECUTION_003

### 6.1 Métricas Corretas

**Dataset**:
- Train: 63,532 samples (2.60% failures)
- Val: 9,530 samples
- Test: 28,859 samples (3.20% failures)
- Imbalance ratio: ~31:1

**Treinamento**:
- Best epoch: 13 (de 18 total)
- Best val_auprc: **0.2118** (métrica de monitoramento)
- Estratégia: Early stopping com patience=5

**Classificação (Test)**:
- AUPRC: 0.0468 (baixo - esperado para dataset extremamente desbalanceado)
- Precision: 0.0494 (5% dos positivos preditos são verdadeiros)
- Recall: 0.3517 (35% dos failures detectados)
- F1: 0.0867
- Accuracy: 0.7627 (enganosa devido ao desbalanceamento)
- AUC: 0.5861 (performance quase aleatória)

**Priorização (Test)**:
- **Mean APFD**: 0.5644 (métrica principal corrigida)
- Median APFD: 0.5625
- Std APFD: 0.2512 (alta variância entre builds)
- Min/Max: 0.0111 / 0.9868
- Builds com APFD ≥ 0.7: **90/277 (32.5%)**
- Builds com APFD < 0.5: **97/277 (35.0%)**

**Discriminação de Probabilidades**:
- Failures mean: 0.4127
- Passes mean: 0.2973
- Discrimination ratio: **1.39x** (muito baixo!)

### 6.2 Diagnóstico de Problemas

#### Problema 1: Baixa Discriminação (1.39x)
**Causa**: O modelo não está separando bem failures de passes.
- Ideal: >2.0x
- Atual: 1.39x (apenas 39% maior)

**Evidência**:
- AUPRC extremamente baixo (0.0468)
- AUC quase aleatório (0.5861)

#### Problema 2: Alta Variância APFD
**Causa**: Performance inconsistente entre builds.
- Std: 0.2512 (muito alto)
- 35% dos builds com APFD < 0.5

#### Problema 3: Overfitting
**Evidência**:
- Val AUPRC no epoch 13: **0.2118** (best)
- Test AUPRC: **0.0468** (78% pior!)
- Gap de generalização extremo

**Confirmação**: Checando histórico de training:
- Epoch 1: val_auprc = 0.059
- Epoch 13 (best): val_auprc = **0.212** (pico)
- Epoch 18 (final): val_auprc = 0.148 (degradação)

O modelo atingiu pico de validação mas não generalizou para teste.

---

## 7. MELHORIAS PROPOSTAS

### 7.1 Ajustes de Hiperparâmetros (Curto Prazo)

**Arquivo**: `scripts/core/run_experiment_server.py` ou `config.yaml`

#### 1. Aumentar Regularização
```python
'dropout': 0.3,  # ⬆️ de 0.2 para 0.3
'weight_decay': 0.1,  # ⬆️ de 0.05 para 0.1
```

#### 2. Reduzir Complexidade do Modelo
```python
'num_layers': 4,  # ⬇️ de 6 para 4
'embedding_dim': 96,  # ⬇️ de 128 para 96
```

#### 3. Melhorar Exposição a Failures
```python
'target_positive_fraction': 0.4,  # ⬆️ de 0.3 para 0.4
'pos_weight': 15.0,  # ⬆️ de 10.0 para 15.0
```

#### 4. Early Stopping Mais Agressivo
```python
'patience': 3,  # ⬇️ de 5 para 3
'min_delta': 0.01,  # Adicionar threshold mínimo de melhoria
```

### 7.2 Calibração de Probabilidades (Médio Prazo)

**Problema**: Probabilidades não calibradas (discrimination 1.39x).

**Solução**: Implementar calibração pós-treinamento:

**Novo arquivo**: `utils/calibration.py`
```python
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression

class ProbabilityCalibrator:
    """Calibrate model probabilities using validation set."""
    def __init__(self, method='isotonic'):
        self.method = method
        self.calibrator = None

    def fit(self, val_probs, val_labels):
        """Fit calibrator on validation set."""
        if self.method == 'isotonic':
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
        elif self.method == 'platt':
            from sklearn.linear_model import LogisticRegression
            self.calibrator = LogisticRegression()

        self.calibrator.fit(val_probs.reshape(-1, 1), val_labels)

    def transform(self, probs):
        """Apply calibration to probabilities."""
        return self.calibrator.predict(probs.reshape(-1, 1))
```

**Integração**: `run_experiment_server.py:414-424`
```python
# After evaluation
from utils.calibration import ProbabilityCalibrator

# Calibrate using validation set
calibrator = ProbabilityCalibrator(method='isotonic')
calibrator.fit(val_probs, y_val)

# Apply to test probabilities
test_probs_calibrated = calibrator.transform(test_probs)
df_test['probability'] = test_probs_calibrated
```

### 7.3 Ensemble com Baseline (Longo Prazo)

**Motivação**: SAINT sozinho tem AUC 0.586 (quase aleatório).

**Proposta**: Ensemble com baseline simples (histórico de falhas):

```python
# Baseline: Failure rate histórica por TC_Key
baseline_probs = df_test.groupby('TC_Key')['label_binary'].transform('mean')

# Ensemble: 50% SAINT + 50% Baseline
ensemble_probs = 0.5 * saint_probs + 0.5 * baseline_probs
```

**Expectativa**: Baseline captura padrões simples que SAINT pode estar perdendo.

### 7.4 Análise de Features Temporais

**Motivação**: Apenas ~4 features temporais, podem estar subutilizadas.

**Proposta**: Adicionar features de tendência:
- `failure_trend`: Taxa de variação de falhas nas últimas N builds
- `duration_trend`: Taxa de variação de duração
- `days_since_last_failure`: Tempo desde última falha do TC

**Arquivo**: `utils/features.py`

### 7.5 Estratificação de Builds

**Problema**: 35% dos builds com APFD < 0.5 sugere distribuição não uniforme.

**Análise proposta**:
```python
# Agrupar builds por características
df_analysis = apfd_results_df.merge(df_test[['Build_ID', 'count_tc']], on='Build_ID')
df_analysis['size_category'] = pd.cut(df_analysis['count_tc'], bins=[0, 20, 50, 100])

# APFD por categoria de tamanho
print(df_analysis.groupby('size_category')['apfd'].describe())
```

**Objetivo**: Identificar se builds pequenos/grandes têm performance diferente.

---

## 8. ARQUIVOS MODIFICADOS

### 8.1 Scripts
- ✅ `scripts/core/run_experiment_server.py` - Correção do cálculo e fluxo de APFD
- ✅ `scripts/recalculate_exec003_metrics.py` - Novo script de recálculo

### 8.2 Resultados
- ✅ `results/execution_003/metrics.json` - Corrigido
- ✅ `results/execution_003/summary.txt` - Regenerado
- ✅ `results/execution_003/apfd_per_build.csv` - Recalculado

### 8.3 Relatórios
- ✅ `reports/ANALISE_E_CORRECOES_EXECUTION_003.md` - Este documento

---

## 9. PRÓXIMOS PASSOS RECOMENDADOS

### Imediato (Sprint Atual)
1. ✅ Aplicar correções em `run_experiment_server.py`
2. ✅ Recalcular métricas da execution_003
3. ⬜ Rodar novo experimento (execution_004) com código corrigido
4. ⬜ Validar que novo experimento gera métricas consistentes

### Curto Prazo (Próximo Sprint)
1. ⬜ Aplicar ajustes de hiperparâmetros (seção 7.1)
2. ⬜ Implementar calibração de probabilidades (seção 7.2)
3. ⬜ Analisar APFD por categoria de build (seção 7.5)

### Médio Prazo
1. ⬜ Implementar ensemble com baseline (seção 7.3)
2. ⬜ Adicionar features temporais de tendência (seção 7.4)
3. ⬜ Comparar SAINT vs MLP simples (para validar complexidade)

---

## 10. CONCLUSÕES

### 10.1 Sobre as Correções
- **Problema crítico identificado**: APFD global vs per-build
- **Impacto**: 77% de superestimação do desempenho
- **Correção implementada**: Cálculo correto por build + relatórios claros
- **Sistema estabilizado**: Código agora reflete regras de negócio corretas

### 10.2 Sobre a Performance
- **Mean APFD real**: 0.5644 (abaixo do target de 0.70)
- **Problemas identificados**:
  - Baixa discriminação de probabilidades (1.39x)
  - Overfitting severo (val 0.21 → test 0.047)
  - Alta variância entre builds (std 0.25)

### 10.3 Recomendação Final
O sistema está **funcional e estável**, mas o desempenho atual (**APFD 0.56**) requer melhorias antes de deployment em produção. As melhorias propostas (seção 7) são **ordenadas por prioridade e facilidade de implementação**.

---

**Relatório gerado em**: 2025-10-17
**Sistema**: Filo-Priori V5 (BGE-large-1024D + SAINT-6L)
**Status**: ✅ Corrigido e validado

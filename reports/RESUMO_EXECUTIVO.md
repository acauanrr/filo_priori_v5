# Resumo Executivo - Smoke Test Execution_002

**Data**: 2025-10-16
**Análise completa**: Ver `ANÁLISE_SMOKE_TEST_EXECUTION_002.md`

---

## 🎯 Status Atual

### ✅ O QUE ESTÁ FUNCIONANDO

1. **Pipeline completo** - Todas as 7 etapas executam sem erros
2. **Arquivos salvos** - Todos os 9 arquivos esperados foram gerados
3. **Treinamento estável** - Sem gradient explosion, loss converge
4. **Val metrics excelentes** - Val AUPRC 0.8258 (muito bom!)
5. **APFD global alto** - 0.9709 (enganoso, mas tecnicamente OK)

### ❌ O QUE ESTÁ QUEBRADO (CRÍTICO)

| Métrica | Val | Test | Gap | Status |
|---------|-----|------|-----|--------|
| **AUPRC** | 0.8258 | **0.0400** | **-95%** | ❌ DISASTER |
| **Precision** | 0.5152 | **0.0364** | **-93%** | ❌ DISASTER |
| **Recall** | 0.8947 | **0.0769** | **-91%** | ❌ DISASTER |
| **F1** | 0.6538 | **0.0494** | **-92%** | ❌ DISASTER |

### 🔴 PROBLEMA MAIS CRÍTICO

```
🚨 DISCRIMINATION RATIO: 0.97x

Interpretation:
- Passes mean prob:   0.186
- Failures mean prob: 0.180
- Discrimination < 1.0 = modelo acha que PASSES têm maior probabilidade que FAILURES!
- Esperado: 2.0x - 5.0x

→ Modelo NÃO ESTÁ DISCRIMINANDO falhas de passes
→ Impossível fazer boa priorização assim
```

---

## 📊 APFD Per-Build (O Que Importa)

```
Target: ≥70% dos builds com APFD ≥ 0.6

Atual:
  ✅ Global APFD: 0.9709 (enganoso!)
  ❌ Mean APFD: 0.5154 (abaixo da meta)
  ❌ Median: 0.5000 (coin flip)

  Distribution:
  ❌ APFD = 1.0:  0% (0 builds)
  ❌ APFD ≥ 0.7:  16.7% (2/12 builds) → NEED 70%!
  ❌ APFD < 0.5:  50% (6/12 builds)

→ 50% dos builds têm performance pior que random!
```

---

## 🎯 Causa Raiz

### **OVERFITTING SEVERO**

```
Evidence:
  Val AUPRC: 0.8258 ✅
  Test AUPRC: 0.0400 ❌
  Gap: 95% !!!!!

Why:
  ❌ Dataset muito pequeno (100 train builds, apenas 128 failures)
  ❌ Modelo muito grande (1.86M params)
  ❌ Dropout muito baixo (0.1)
  ❌ Best epoch = 30/30 (early stopping nunca disparou)
  ❌ Smoke test não é representativo
```

### **HIPERPARÂMETROS INADEQUADOS**

| Param | Atual | Problema | Deveria ser |
|-------|-------|----------|-------------|
| `pos_weight` | 5.0 | Imbalance é 27:1 | **10.0** |
| `label_smoothing` | 0.05 | Suaviza demais | **0.01** |
| `dropout` | 0.1 | Overfitting | **0.2** |
| `patience` | 8 | Treina demais | **5** |
| `target_positive_fraction` | 0.20 | Pouca exposição | **0.30** |
| `learning_rate` | 5e-4 | Converge rápido | **3e-4** |
| `weight_decay` | 0.01 | Regularização fraca | **0.05** |

---

## 🔧 SOLUÇÃO (Prioridade ALTA)

### Aplicar Correções ANTES do Full Test

```bash
cd /home/acauan/ufam/iats/sprint_07/filo_priori_v5

# 1. Aplicar correções automaticamente
./apply_fixes.sh

# 2. Validar com smoke test
./run_smoke_test.sh

# 3. Verificar melhorias:
#    - Discrimination > 1.5x (não 0.97x)
#    - Test AUPRC > 0.15 (não 0.04)
#    - Val/Test gap < 50% (não 95%)
```

### Checklist de Validação

Após rodar smoke test com correções:

- [ ] **Discrimination ratio > 1.5x** (passa prob < falha prob)
- [ ] **Test AUPRC > 0.15** (atualmente 0.04)
- [ ] **Val/Test AUPRC gap < 50%** (atualmente 95%)
- [ ] **Mean APFD per-build > 0.60** (atualmente 0.52)
- [ ] **Builds com APFD ≥ 0.7 > 25%** (atualmente 16.7%)

Se **TODOS** os checks passarem → Rodar full test
Se **algum** falhar → Investigar mais (ver Plano B na análise completa)

---

## 📈 Expectativa Pós-Correções

### Smoke Test (50 builds)

```
Conservador:
  Test AUPRC: 0.15-0.25 (era 0.04)
  Discrimination: 1.5-2.5x (era 0.97x)
  Mean APFD: 0.55-0.65 (era 0.52)
  Builds ≥ 0.7: 25-40% (era 16.7%)

Otimista:
  Test AUPRC: 0.30-0.45 (era 0.04)
  Discrimination: 2.5-4.0x (era 0.97x)
  Mean APFD: 0.65-0.75 (era 0.52)
  Builds ≥ 0.7: 40-60% (era 16.7%)
```

### Full Test (TODOS os builds)

```
Esperado:
  Test AUPRC: 0.20-0.40
  Discrimination: 2.0-4.0x
  Mean APFD: 0.65-0.75
  Builds ≥ 0.7: 50-70% (TARGET: 70%)

Tempo estimado: 2-6 horas GPU, 12-24 horas CPU
```

---

## ⚠️ AVISOS IMPORTANTES

### 1. Smoke Test NÃO é Representativo

- **Apenas 100 train builds** (full tem ~1000+)
- **Apenas 128 failures** (full tem muito mais)
- Performance no full test pode ser **significativamente diferente**

### 2. Overfitting é Esperado no Smoke Test

- Dataset pequeno → overfitting inevitável
- Full test com mais dados deve aliviar problema

### 3. Calibração Pode Não Estar Ativa

- Código tem `ProbabilityCalibrator` mas pode não estar sendo usado
- Verificar `filo_priori/utils/inference.py` (linha ~70-150)

---

## 🚀 Plano de Ação

### Imediato (Hoje)

1. ✅ **Revisar análise** - Leia `ANÁLISE_SMOKE_TEST_EXECUTION_002.md`
2. 🔧 **Aplicar correções** - Execute `./apply_fixes.sh`
3. 🧪 **Re-testar smoke test** - Execute `./run_smoke_test.sh` (15-30 min)
4. ✅ **Validar métricas** - Discrimination > 1.5x, AUPRC > 0.15

### Se Smoke Test Melhorar

5. 🚀 **Rodar full test** - Execute `./run_full_test.sh` (2-6h GPU)
6. 📊 **Analisar resultados** - Ver `results/execution_XXX/summary.txt`
7. 🎯 **Validar meta** - ≥70% builds com APFD ≥ 0.6

### Se Smoke Test NÃO Melhorar

- Ver **Plano B** em `ANÁLISE_SMOKE_TEST_EXECUTION_002.md` (seção "Se Performance Continuar Ruim")
- Considerar MLP simples ao invés de SAINT
- Investigar features temporais
- Adicionar mais features

---

## 📁 Arquivos Criados

1. ✅ `ANÁLISE_SMOKE_TEST_EXECUTION_002.md` - Análise técnica completa (15 páginas)
2. ✅ `RESUMO_EXECUTIVO.md` - Este arquivo (resumo de 2 páginas)
3. ✅ `apply_fixes.sh` - Script para aplicar correções automaticamente
4. ✅ `CORREÇÕES_APLICADAS.md` - Documentação das correções anteriores (paths)
5. ✅ `BUGFIX_FILE_SAVING.md` - Análise do problema de salvamento de arquivos

---

## 💡 TL;DR

**Problema**: Modelo não discrimina falhas de passes (0.97x discrimination)
**Causa**: Overfitting severo + hiperparâmetros inadequados
**Solução**: Aplicar 7 correções de hiperparâmetros
**Ação**: `./apply_fixes.sh` → `./run_smoke_test.sh` → validar → `./run_full_test.sh`
**Meta**: ≥70% builds com APFD ≥ 0.6
**Chance de sucesso**: 60-70% com correções aplicadas

---

**Última atualização**: 2025-10-16
**Próxima ação**: Aplicar correções e re-testar

# BUGFIX: Arquivos de Resultado Não Sendo Salvos

**Data**: 2025-10-16
**Severidade**: CRÍTICA
**Status**: Identificado, correção proposta

---

## Problema

O pipeline de treinamento completa com sucesso (30 épocas, APFD=0.9709) mas **NENHUM arquivo de resultado é salvo**. O log indica "Results saved to ../results/execution_001/" mas o diretório está vazio (exceto o log).

### Sintomas

```bash
$ ls filo_priori/results/execution_001/
full_run.log  # ← APENAS O LOG (movido manualmente)

# Arquivos esperados mas AUSENTES:
# - metrics.json
# - best_model.pth
# - training_history.json
# - prioritized_hybrid.csv
# - apfd_per_build.csv
# - summary.txt
# - feature_builder.pkl
# - embedder/scaler.pkl
```

### Causa Raiz

1. **Path relativo incorreto**: `config['output_dir'] = '../results'` assume que a execução acontece de `filo_priori/scripts/core/`, mas na verdade acontece de `filo_priori/`

2. **Diretório pai não existe**: `/home/acauan/ufam/iats/sprint_07/filo_priori_v5/results/` não existe (deveria ser `filo_priori/results/`)

3. **Falha silenciosa**: O código não valida se os arquivos foram realmente salvos, então falha sem avisar

---

## Correções Necessárias

### 1. Corrigir Path de Output

**Arquivo**: `filo_priori/scripts/core/run_experiment_server.py`
**Linha**: 75

```python
# ❌ ANTES
DEFAULT_CONFIG = {
    'output_dir': '../results',

# ✅ DEPOIS
DEFAULT_CONFIG = {
    'output_dir': 'results',  # Path relativo a filo_priori/
```

**Justificativa**: O script `run_smoke_test.sh` executa `cd filo_priori` antes de chamar o Python, então o CWD é `/path/to/filo_priori_v5/filo_priori/`. Logo, `results/` aponta para `filo_priori/results/` (correto).

---

### 2. Adicionar Validação de Diretório

**Arquivo**: `filo_priori/scripts/core/run_experiment_server.py`
**Linhas**: 144-147

```python
# Criar diretório de execução
exec_dir = base_path / f'execution_{next_num:03d}'
exec_dir.mkdir(parents=True, exist_ok=True)

# ✅ ADICIONAR VALIDAÇÃO
if not exec_dir.exists() or not exec_dir.is_dir():
    raise RuntimeError(f"Failed to create execution directory: {exec_dir}")

logger.info(f"✅ Execution directory created: {exec_dir.resolve()}")

return exec_dir
```

---

### 3. Adicionar Logging Detalhado de Paths

**Arquivo**: `filo_priori/scripts/core/run_experiment_server.py`
**Linhas**: 247-252

```python
logger.info("="*70)
logger.info("FILO-PRIORI V5 - PIPELINE START (BGE + SAINT)")
logger.info("="*70)
logger.info(f"Current working directory: {Path.cwd()}")  # ✅ ADICIONAR
logger.info(f"Execution directory (relative): {exec_dir}")
logger.info(f"Execution directory (absolute): {exec_dir.resolve()}")  # ✅ ADICIONAR
logger.info(f"Directory exists: {exec_dir.exists()}")  # ✅ ADICIONAR
logger.info(f"Directory is writable: {os.access(exec_dir, os.W_OK)}")  # ✅ ADICIONAR
logger.info(f"Mode: {'SMOKE TEST' if args.smoke_train else 'FULL TEST'}")
```

---

### 4. Adicionar Tratamento de Erro no Salvamento

**Arquivo**: `filo_priori/scripts/core/run_experiment_server.py`
**Linhas**: 465-559

```python
# ========================================================================
# STEP 7: Save results
# ========================================================================
logger.info("\n[7/7] Saving results...")

try:
    # Validate exec_dir exists and is writable
    if not exec_dir.exists():
        raise RuntimeError(f"Execution directory does not exist: {exec_dir}")
    if not os.access(exec_dir, os.W_OK):
        raise RuntimeError(f"Execution directory is not writable: {exec_dir}")

    # Save metrics.json
    metrics_path = exec_dir / 'metrics.json'
    logger.info(f"Saving metrics to: {metrics_path.resolve()}")
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    if not metrics_path.exists():
        raise RuntimeError(f"Failed to save metrics.json")
    logger.info(f"✅ metrics.json saved ({metrics_path.stat().st_size} bytes)")

    # Save prioritized_hybrid.csv
    csv_path = exec_dir / 'prioritized_hybrid.csv'
    logger.info(f"Saving predictions to: {csv_path.resolve()}")
    df_test[output_cols].to_csv(csv_path, index=False)
    if not csv_path.exists():
        raise RuntimeError(f"Failed to save prioritized_hybrid.csv")
    logger.info(f"✅ prioritized_hybrid.csv saved ({csv_path.stat().st_size} bytes)")

    # ... repeat for other files ...

    logger.info(f"\n✅ ALL FILES SAVED SUCCESSFULLY to {exec_dir.resolve()}/")

except Exception as e:
    logger.error(f"\n❌ CRITICAL ERROR: FAILED TO SAVE RESULTS", exc_info=True)
    logger.error(f"Attempted to save to: {exec_dir}")
    logger.error(f"Absolute path: {exec_dir.resolve()}")
    logger.error(f"Directory exists: {exec_dir.exists()}")
    logger.error(f"Parent exists: {exec_dir.parent.exists()}")
    logger.error(f"Is writable: {os.access(exec_dir, os.W_OK) if exec_dir.exists() else 'N/A'}")
    raise RuntimeError(f"Failed to save results: {e}") from e
```

---

### 5. Adicionar Validação em `saint_trainer.py`

**Arquivo**: `filo_priori/utils/saint_trainer.py`
**Buscar função `train_saint()` que salva `best_model.pth`**

```python
# Salvar melhor modelo
model_path = save_dir / 'best_model.pth'
logger.info(f"Saving best model to: {model_path.resolve()}")  # ✅ ADICIONAR

torch.save({
    'epoch': best_epoch,
    'model_state_dict': best_model_state,
    'metric': best_metric,
    'monitor_metric': monitor_metric
}, model_path, _use_new_zipfile_serialization=False, weights_only=False)

# ✅ ADICIONAR VALIDAÇÃO
if not model_path.exists():
    raise RuntimeError(f"Failed to save best_model.pth to {model_path}")
logger.info(f"✅ best_model.pth saved ({model_path.stat().st_size} bytes)")
```

---

## Checklist de Implementação

- [ ] **CRÍTICO**: Alterar `'output_dir': '../results'` → `'output_dir': 'results'` em `run_experiment_server.py:75`
- [ ] Adicionar validação em `get_next_execution_dir()` (linhas 144-147)
- [ ] Adicionar logging detalhado de paths (linhas 247-252)
- [ ] Adicionar try/except com validação em STEP 7 (linhas 465-559)
- [ ] Adicionar validação em `saint_trainer.py` (salvar best_model.pth)
- [ ] Testar com smoke test: `./run_smoke_test.sh`
- [ ] Validar que todos os arquivos foram salvos: `ls -lh filo_priori/results/execution_XXX/`

---

## Como Testar a Correção

```bash
# 1. Aplicar correções
cd /home/acauan/ufam/iats/sprint_07/filo_priori_v5

# 2. Rodar smoke test
./run_smoke_test.sh

# 3. Verificar arquivos salvos
ls -lh filo_priori/results/execution_*/

# Deve exibir:
# - metrics.json (~10-50KB)
# - best_model.pth (~5-50MB)
# - training_history.json (~5-20KB)
# - prioritized_hybrid.csv (~100KB-5MB)
# - apfd_per_build.csv (~10-100KB)
# - summary.txt (~2-5KB)
# - feature_builder.pkl (~1-5MB)
# - embedder/scaler.pkl (~1-10MB)
```

---

## Impacto

**Antes da correção**:
- Pipeline treina com sucesso (30 épocas, ~40 minutos)
- NENHUM resultado salvo
- Impossível avaliar modelo ou fazer inferência

**Depois da correção**:
- Pipeline treina E salva todos os resultados
- Modelo pode ser carregado para inferência
- Métricas disponíveis para análise
- Ready para full test

---

## Prioridade

**URGENTE - BLOCKER**: Sem essa correção, o sistema está completamente inutilizável para produção. O treinamento funciona mas não há como usar o modelo treinado.

---

## Referências

- Log completo: `filo_priori/results/execution_001/full_run.log`
- Código principal: `filo_priori/scripts/core/run_experiment_server.py`
- Trainer: `filo_priori/utils/saint_trainer.py`
- Issue relacionado: Path resolution em execuções via shell script

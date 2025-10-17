# Correções Aplicadas - Filo-Priori V5

**Data**: 2025-10-16
**Status**: ✅ CORREÇÕES IMPLEMENTADAS - PRONTO PARA TESTE

---

## Problema Identificado

A execução do smoke test no servidor completou o treinamento com sucesso (30 épocas, APFD=0.9709), mas **NENHUM arquivo de resultado foi salvo**. Apenas o log existe no diretório `execution_001/`.

### Causa Raiz

**Path relativo incorreto**: `output_dir = '../results'` assumia que o script rodava de `filo_priori/scripts/core/`, mas na verdade roda de `filo_priori/` (devido ao `cd filo_priori` em `run_smoke_test.sh`).

**Resultado**: O código tentava salvar em `/home/.../filo_priori_v5/results/` (que não existe) ao invés de `filo_priori/results/`.

---

## Correções Aplicadas

### 1. ✅ Correção do Path de Output (CRÍTICO)

**Arquivo**: `filo_priori/scripts/core/run_experiment_server.py:75`

```diff
- 'output_dir': '../results',
+ 'output_dir': 'results',  # Path relativo a filo_priori/
```

**Impacto**: Agora o script salva corretamente em `filo_priori/results/execution_XXX/`

---

### 2. ✅ Validação de Criação de Diretório

**Arquivo**: `filo_priori/scripts/core/run_experiment_server.py:147-149`

```python
# Validate directory was created successfully
if not exec_dir.exists() or not exec_dir.is_dir():
    raise RuntimeError(f"Failed to create execution directory: {exec_dir}")
```

**Impacto**: Pipeline falha IMEDIATAMENTE se não conseguir criar diretório, ao invés de falhar silenciosamente 40 minutos depois.

---

### 3. ✅ Logging Detalhado de Paths

**Arquivo**: `filo_priori/scripts/core/run_experiment_server.py:254-258`

```python
logger.info(f"Current working directory: {Path.cwd()}")
logger.info(f"Execution directory (relative): {exec_dir}")
logger.info(f"Execution directory (absolute): {exec_dir.resolve()}")
logger.info(f"Directory exists: {exec_dir.exists()}")
logger.info(f"Directory is writable: {os.access(exec_dir, os.W_OK)}")
```

**Impacto**: Facilita debugging de problemas de path no futuro.

---

### 4. ✅ Validação Antes de Salvar Resultados

**Arquivo**: `filo_priori/scripts/core/run_experiment_server.py:476-482`

```python
# Validate exec_dir before saving
if not exec_dir.exists():
    raise RuntimeError(f"Execution directory does not exist: {exec_dir.resolve()}")
if not os.access(exec_dir, os.W_OK):
    raise RuntimeError(f"Execution directory is not writable: {exec_dir.resolve()}")

logger.info(f"Saving results to: {exec_dir.resolve()}")
```

**Impacto**: Detecção precoce de problemas de permissão/existência.

---

### 5. ✅ Validação de Salvamento de Arquivos Críticos

**Arquivo**: `filo_priori/scripts/core/run_experiment_server.py:534-540, 549-554`

```python
# Save metrics.json
metrics_path = exec_dir / 'metrics.json'
logger.info(f"Saving metrics.json to {metrics_path.resolve()}")
with open(metrics_path, 'w') as f:
    json.dump(results, f, indent=2, default=float)
if not metrics_path.exists():
    raise RuntimeError(f"Failed to save metrics.json to {metrics_path}")
logger.info(f"✅ metrics.json saved ({metrics_path.stat().st_size} bytes)")
```

**Impacto**: Confirma que cada arquivo foi salvo com sucesso e mostra tamanho.

---

## Arquivos Modificados

1. ✅ `filo_priori/scripts/core/run_experiment_server.py` (5 alterações)
2. ✅ `BUGFIX_FILE_SAVING.md` (documentação completa criada)
3. ✅ `test_path_fix.sh` (script de validação criado)
4. ✅ `CORREÇÕES_APLICADAS.md` (este arquivo)

---

## Como Testar

### Teste Rápido de Validação

```bash
cd /home/acauan/ufam/iats/sprint_07/filo_priori_v5
./test_path_fix.sh
```

**Saída esperada**: Todos os checks devem mostrar ✅

### Smoke Test Completo

```bash
cd /home/acauan/ufam/iats/sprint_07/filo_priori_v5
./run_smoke_test.sh
```

**Duração**: ~15-30 minutos (GPU) ou ~1-2 horas (CPU)

**Validação pós-execução**:
```bash
ls -lh filo_priori/results/execution_002/
```

**Arquivos esperados** (deve haver 8 arquivos + 1 diretório):
```
✅ metrics.json              (~10-50KB)     - Métricas completas
✅ config.json               (~2-5KB)       - Configuração do experimento
✅ best_model.pth            (~5-50MB)      - Modelo SAINT treinado
✅ training_history.json     (~5-20KB)      - Histórico de treinamento
✅ prioritized_hybrid.csv    (~100KB-5MB)   - Predições com ranks
✅ apfd_per_build.csv        (~10-100KB)    - APFD por build
✅ feature_builder.pkl       (~1-5MB)       - Feature engineering artifacts
✅ summary.txt               (~2-5KB)       - Resumo do experimento
✅ embedder/                 (diretório)    - BGE embedder artifacts
    └── scaler.pkl           (~1-10MB)
```

---

## Full Test - Próximos Passos

Após validar que o smoke test salva corretamente:

### 1. Verificar Script de Full Test

```bash
cat run_full_test.sh
```

Confirmar que usa `--full-test` e não tem limitações de builds.

### 2. Executar Full Test

```bash
./run_full_test.sh
```

**Duração estimada**: ~2-6 horas (GPU) ou ~12-24 horas (CPU)

**Dataset completo**:
- Train: ~1.7GB (todos os builds)
- Test: ~581MB (todos os builds)

### 3. Validar Resultados

```bash
# Listar arquivos salvos
ls -lh filo_priori/results/execution_XXX/

# Ver resumo
cat filo_priori/results/execution_XXX/summary.txt

# Ver métricas principais
cat filo_priori/results/execution_XXX/metrics.json | grep -A 5 '"apfd"'

# Ver distribuição de APFD por build
cat filo_priori/results/execution_XXX/apfd_per_build.csv | head -20
```

---

## Métricas Esperadas (Baseline do Smoke Test)

Com base na execução anterior (que falhou ao salvar mas completou o treinamento):

### Treinamento
- **Best epoch**: 30/30
- **Val AUPRC**: 0.8258 (excelente para dataset desbalanceado)
- **Convergência**: Estável (sem gradient explosion)

### Test Set (50 builds)
- **APFD**: 0.9709 ✅ (target ≥0.70)
- **APFDc**: 0.9709 ✅
- **AUPRC**: Esperado ~0.30-0.60
- **Precision**: Esperado ~0.10-0.40
- **Recall**: Esperado ~0.40-0.80

### Distribuição APFD per-build
- **Builds com APFD ≥ 0.7**: 16.7% (2/12 no smoke test anterior)
- **Target para full test**: ≥70% dos builds com APFD ≥ 0.6

**Nota**: Valores exatos dependerão do dataset completo vs smoke test.

---

## Checklist de Validação

Antes de rodar full test, confirme:

- [ ] Smoke test completa SEM ERROS
- [ ] Todos os 8 arquivos + diretório `embedder/` foram salvos
- [ ] `metrics.json` contém métricas válidas (não NaN/Inf)
- [ ] `best_model.pth` tem tamanho razoável (5-50MB)
- [ ] `prioritized_hybrid.csv` tem ranks de 1 a N para cada build
- [ ] `apfd_per_build.csv` mostra APFD calculado por build
- [ ] `summary.txt` mostra configuração e métricas corretas
- [ ] Log não mostra warnings de path ou salvamento

---

## Melhorias Adicionais Recomendadas (Futuro)

Estas não são críticas mas melhorariam a robustez:

### 1. Validação em `saint_trainer.py`

Adicionar log detalhado ao salvar `best_model.pth`:
```python
model_path = save_dir / 'best_model.pth'
logger.info(f"Saving best model to: {model_path.resolve()}")
torch.save(...)
if not model_path.exists():
    raise RuntimeError(f"Failed to save best_model.pth")
logger.info(f"✅ best_model.pth saved ({model_path.stat().st_size} bytes)")
```

### 2. Checksum de Integridade

Salvar MD5 hash dos arquivos críticos para validar integridade:
```python
import hashlib

def compute_file_hash(filepath):
    md5 = hashlib.md5()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            md5.update(chunk)
    return md5.hexdigest()

# Salvar em metadata
metadata['file_checksums'] = {
    'best_model.pth': compute_file_hash(exec_dir / 'best_model.pth'),
    'metrics.json': compute_file_hash(exec_dir / 'metrics.json'),
    # ...
}
```

### 3. Backup Automático de Configuração

Copiar `config.yaml` usado na execução para `execution_XXX/`:
```python
import shutil
shutil.copy('config.yaml', exec_dir / 'config.yaml')
```

### 4. Validação de Formato de Arquivos

Após salvar CSV/JSON, tentar recarregar para validar formato:
```python
# Após salvar CSV
df_test[output_cols].to_csv(csv_path, index=False)
# Validar
df_loaded = pd.read_csv(csv_path)
assert len(df_loaded) == len(df_test), "CSV incomplete"
```

---

## Contato e Suporte

Para questões sobre as correções:
- Ver documentação completa: `BUGFIX_FILE_SAVING.md`
- Ver estrutura do projeto: `CLAUDE.md`
- Ver changelog: `CHANGELOG.md`

---

## Resumo Executivo

✅ **Problema identificado**: Path relativo incorreto impedia salvamento de resultados

✅ **Correção aplicada**: Alterado `'../results'` → `'results'` + validações adicionadas

✅ **Status**: Pronto para smoke test

⏳ **Próximo passo**: Executar `./run_smoke_test.sh` e validar que arquivos são salvos

🎯 **Meta final**: Full test com dataset completo e APFD ≥ 0.70 em ≥70% dos builds

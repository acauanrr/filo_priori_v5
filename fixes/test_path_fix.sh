#!/bin/bash
# Script de teste para validar correção de paths

echo "=============================================="
echo "TESTE DE VALIDAÇÃO DE PATHS - FILO-PRIORI V5"
echo "=============================================="
echo ""

cd filo_priori

echo "1. Diretório atual:"
pwd
echo ""

echo "2. Verificando estrutura de diretórios:"
echo "   results/ existe? $([ -d results ] && echo 'SIM ✅' || echo 'NÃO ❌ (será criado no primeiro run)')"
echo "   ../datasets/ existe? $([ -d ../datasets ] && echo 'SIM ✅' || echo 'NÃO ❌')"
echo ""

echo "3. Testando path resolution em Python:"
python3 -c "
from pathlib import Path
import os

print(f'   CWD: {os.getcwd()}')
print(f'   CWD (Path): {Path.cwd()}')
print(f'   results/ resolve para: {Path(\"results\").resolve()}')
print(f'   ../datasets/ resolve para: {Path(\"../datasets\").resolve()}')
print('')
print(f'   ../datasets/train.csv existe? {Path(\"../datasets/train.csv\").exists()}')
print(f'   ../datasets/test_full.csv existe? {Path(\"../datasets/test_full.csv\").exists()}')
"

echo ""
echo "4. Verificando configuração do script:"
grep -n "output_dir" scripts/core/run_experiment_server.py | head -1
echo ""

echo "5. Testando criação de diretório de execução:"
python3 -c "
from pathlib import Path
import sys
sys.path.insert(0, '.')

# Import get_next_execution_dir logic
def test_get_next_execution_dir(base_dir):
    base_path = Path(base_dir)
    print(f'   Base dir: {base_dir}')
    print(f'   Resolved: {base_path.resolve()}')
    print(f'   Exists before mkdir: {base_path.exists()}')

    base_path.mkdir(parents=True, exist_ok=True)
    print(f'   Exists after mkdir: {base_path.exists()}')

    # Find next execution number
    existing = [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith('execution_')]
    nums = []
    for d in existing:
        try:
            num = int(d.name.split('_')[1])
            nums.append(num)
        except (IndexError, ValueError):
            continue
    next_num = max(nums) + 1 if nums else 1

    print(f'   Existing execution dirs: {len(existing)}')
    print(f'   Next execution number: {next_num:03d}')
    print(f'   Next exec dir will be: {base_path / f\"execution_{next_num:03d}\"}')

test_get_next_execution_dir('results')
"

echo ""
echo "=============================================="
echo "TESTE CONCLUÍDO"
echo "=============================================="
echo ""
echo "Se todos os checks estão ✅, as correções estão funcionando!"
echo "Próximo passo: rodar smoke test completo"
echo ""
echo "Para rodar smoke test:"
echo "  cd .. && ./run_smoke_test.sh"

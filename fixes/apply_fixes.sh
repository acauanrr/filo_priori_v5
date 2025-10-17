#!/bin/bash
# Script para aplicar correções de hiperparâmetros baseadas na análise do smoke test
# Autor: Claude Code
# Data: 2025-10-16

echo "=============================================="
echo "APLICANDO CORREÇÕES DE HIPERPARÂMETROS"
echo "=============================================="
echo ""

cd filo_priori

echo "📋 Correções a serem aplicadas:"
echo "  1. pos_weight: 5.0 → 10.0"
echo "  2. label_smoothing: 0.05 → 0.01"
echo "  3. dropout: 0.1 → 0.2"
echo "  4. patience: 8 → 5"
echo "  5. target_positive_fraction: 0.20 → 0.30"
echo "  6. learning_rate: 0.0005 → 0.0003"
echo "  7. weight_decay: 0.01 → 0.05"
echo ""

read -p "Deseja aplicar as correções? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "❌ Cancelado pelo usuário"
    exit 1
fi

echo ""
echo "🔧 Criando backup do arquivo original..."
cp scripts/core/run_experiment_server.py scripts/core/run_experiment_server.py.backup
echo "   ✅ Backup salvo em scripts/core/run_experiment_server.py.backup"

echo ""
echo "✏️  Aplicando correções via sed..."

# Aplicar correções usando sed
sed -i "s/'pos_weight': 5.0/'pos_weight': 10.0/" scripts/core/run_experiment_server.py
sed -i "s/'label_smoothing': 0.05/'label_smoothing': 0.01/" scripts/core/run_experiment_server.py
sed -i "s/'patience': 8/'patience': 5/" scripts/core/run_experiment_server.py
sed -i "s/'target_positive_fraction': 0.20/'target_positive_fraction': 0.30/" scripts/core/run_experiment_server.py
sed -i "s/'learning_rate': 5e-4/'learning_rate': 3e-4/" scripts/core/run_experiment_server.py
sed -i "s/'learning_rate': 0.0005/'learning_rate': 0.0003/" scripts/core/run_experiment_server.py
sed -i "s/'weight_decay': 0.01/'weight_decay': 0.05/" scripts/core/run_experiment_server.py
sed -i "s/'dropout': 0.1/'dropout': 0.2/" scripts/core/run_experiment_server.py

echo ""
echo "🔍 Verificando mudanças aplicadas..."
echo ""

echo "1. pos_weight:"
grep -n "pos_weight" scripts/core/run_experiment_server.py | grep -v "^#"

echo ""
echo "2. label_smoothing:"
grep -n "label_smoothing" scripts/core/run_experiment_server.py | grep -v "^#"

echo ""
echo "3. dropout:"
grep -n "dropout" scripts/core/run_experiment_server.py | grep -v "^#" | head -2

echo ""
echo "4. patience:"
grep -n "patience" scripts/core/run_experiment_server.py | grep -v "^#"

echo ""
echo "5. target_positive_fraction:"
grep -n "target_positive_fraction" scripts/core/run_experiment_server.py | grep -v "^#"

echo ""
echo "6. learning_rate:"
grep -n "learning_rate" scripts/core/run_experiment_server.py | grep -v "^#" | head -1

echo ""
echo "7. weight_decay:"
grep -n "weight_decay" scripts/core/run_experiment_server.py | grep -v "^#" | head -1

echo ""
echo "=============================================="
echo "✅ CORREÇÕES APLICADAS COM SUCESSO!"
echo "=============================================="
echo ""
echo "📝 Arquivos modificados:"
echo "  - scripts/core/run_experiment_server.py"
echo "  - Backup salvo em scripts/core/run_experiment_server.py.backup"
echo ""
echo "🧪 Próximo passo: Rodar smoke test para validar"
echo "   cd .. && ./run_smoke_test.sh"
echo ""
echo "📊 Para reverter mudanças:"
echo "   cd filo_priori"
echo "   cp scripts/core/run_experiment_server.py.backup scripts/core/run_experiment_server.py"

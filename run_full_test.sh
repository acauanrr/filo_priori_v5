#!/bin/bash
# Script helper para executar full test do Filo-Priori V5 (BGE + SAINT)
# Uso: ./run_full_test.sh

echo "🚀 Executando Filo-Priori V5 - Full Test (BGE + SAINT)"
echo "========================================================"
echo "📌 Modelo: BGE-large-en-v1.5 (1024D) + SAINT Transformer"
echo "📌 Dataset completo: ~1.7GB train + ~581MB test"
echo "📌 SAINT: 6 layers, 8 heads, intersample attention"
echo "📌 Treinamento: ~30 epochs com early stopping"
echo "⏱️  Tempo estimado: ~2-4 horas (CPU) ou ~30-60 min (GPU)"
echo ""

cd filo_priori

# Auto-detect device (prefer CUDA if available)
if command -v nvidia-smi &> /dev/null; then
    echo "🎮 GPU detectada, usando CUDA"
    python scripts/core/run_experiment_server.py --full-test --device cuda
else
    echo "💻 GPU não detectada, usando CPU"
    python scripts/core/run_experiment_server.py --full-test --device cpu
fi

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    # Find last execution directory
    LAST_EXEC=$(ls -t results/ | grep "execution_" | head -1)
    echo "✅ Full test concluído com sucesso!"
    echo "📊 Resultados salvos em: filo_priori/results/$LAST_EXEC/"
    echo ""
    echo "📁 Arquivos principais:"
    echo "   - metrics.json (métricas completas)"
    echo "   - best_model.pth (modelo SAINT treinado - checkpoint)"
    echo "   - training_history.json (histórico de treinamento por época)"
    echo "   - prioritized_hybrid.csv (testes priorizados)"
    echo "   - apfd_per_build.csv (APFD por build)"
    echo "   - summary.txt (resumo do experimento)"
    echo ""
    echo "🎯 Métricas esperadas:"
    echo "   - APFD ≥ 0.70 (target)"
    echo "   - AUPRC ≥ 0.20"
    echo "   - Precision ≥ 0.15"
    echo "   - Recall ≥ 0.50"
    echo ""
    echo "📈 Para ver o resumo completo:"
    echo "   cat filo_priori/results/$LAST_EXEC/summary.txt"
else
    echo "❌ Full test falhou com código de erro: $EXIT_CODE"
    echo "💡 Verifique se as dependências estão instaladas:"
    echo "   pip install -r requirements.txt"
    echo ""
    echo "💡 Se houver erro de GPU, force CPU:"
    echo "   cd filo_priori && python scripts/core/run_experiment_server.py --full-test --device cpu"
fi

exit $EXIT_CODE

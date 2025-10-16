#!/bin/bash
# Script helper para executar smoke test do Filo-Priori V5 (BGE + SAINT)
# Uso: ./run_smoke_test.sh

echo "🚀 Executando Filo-Priori V5 - Smoke Test (BGE + SAINT)"
echo "=========================================================="
echo "📌 Modelo: BGE-large-en-v1.5 (1024D) + SAINT Transformer"
echo "📌 SAINT: 6 layers, 8 heads, intersample attention"
echo "📌 Smoke test: 100 builds treino, 50 builds teste"
echo "⏱️  Tempo estimado: ~10-20 minutos"
echo ""

cd filo_priori

# Auto-detect device
if command -v nvidia-smi &> /dev/null; then
    echo "🎮 GPU detectada, usando CUDA"
    DEVICE="cuda"
else
    echo "💻 GPU não detectada, usando CPU"
    DEVICE="cpu"
fi

python scripts/core/run_experiment_server.py \
    --smoke-train 100 \
    --smoke-test 50 \
    --device $DEVICE

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    # Find last execution directory
    LAST_EXEC=$(ls -t results/ | grep "execution_" | head -1)
    echo "✅ Smoke test concluído com sucesso!"
    echo "📊 Resultados salvos em: filo_priori/results/$LAST_EXEC/"
    echo ""
    echo "📁 Arquivos principais:"
    echo "   - metrics.json (métricas completas)"
    echo "   - best_model.pth (modelo SAINT - checkpoint)"
    echo "   - training_history.json (histórico de treinamento)"
    echo "   - prioritized_hybrid.csv (testes priorizados)"
    echo "   - apfd_per_build.csv (APFD por build)"
    echo "   - summary.txt (resumo do experimento)"
    echo ""
    echo "📈 Para ver o resumo:"
    echo "   cat filo_priori/results/$LAST_EXEC/summary.txt"
else
    echo "❌ Smoke test falhou com código de erro: $EXIT_CODE"
    echo "💡 Verifique logs e tente forçar CPU se necessário"
fi

exit $EXIT_CODE

#!/bin/bash
# Script helper para executar full test do Filo-Priori V5 (BGE + SAINT)
# Uso: ./run_full_test.sh

echo "🚀 Executando Filo-Priori V5 - Full Test (BGE + SAINT)"
echo "========================================================"
echo "📌 Modelo: BGE-large-en-v1.5 (1024D) + SAINT Transformer"
echo "📌 Dataset completo: ~1.7GB train + ~581MB test"
echo "📌 SAINT: 6 layers, 8 heads, intersample attention"
echo "📌 Treinamento: ~30 epochs com early stopping"
echo "📌 Otimizações: batch_size=256 para BGE, weights_only=False para PyTorch 2.6+"
echo "⏱️  Tempo estimado:"
echo "   - Embeddings: ~5-10 min (GPU) ou ~30-60 min (CPU)"
echo "   - Treinamento: ~45-90 min (GPU) ou ~2-4 horas (CPU)"
echo "   - Total: ~1-2 horas (GPU) ou ~3-5 horas (CPU)"
echo ""

cd filo_priori

# Auto-detect device (prefer CUDA if available)
if command -v nvidia-smi &> /dev/null; then
    GPU_AVAILABLE=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
    if [ "$GPU_AVAILABLE" -gt 0 ]; then
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
        GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
        echo "🎮 GPU detectada: $GPU_NAME ($GPU_MEMORY MB)"
        echo "🔧 Usando CUDA para aceleração"
        DEVICE="cuda"
    else
        echo "💻 nvidia-smi encontrado mas sem GPU disponível, usando CPU"
        DEVICE="cpu"
    fi
else
    echo "💻 GPU não detectada, usando CPU"
    DEVICE="cpu"
fi

echo ""
echo "🔧 Iniciando pipeline completo..."
echo "⚠️  ATENÇÃO: Isso pode demorar várias horas. Considere usar 'nohup' ou 'screen' para sessões longas."
echo ""

# Start timestamp
START_TIME=$(date +%s)

python scripts/core/run_experiment_server.py --full-test --device $DEVICE

EXIT_CODE=$?

# End timestamp
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo "⏱️  Tempo total de execução: ${HOURS}h ${MINUTES}m ${SECONDS}s"
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
    echo ""
    echo "📊 Para ver métricas APFD:"
    echo "   cat filo_priori/results/$LAST_EXEC/metrics.json | grep -A 20 apfd_per_build"
else
    echo "❌ Full test falhou com código de erro: $EXIT_CODE"
    echo ""
    echo "💡 Troubleshooting:"
    echo "   1. Verifique se o venv está ativado: source venv/bin/activate"
    echo "   2. Verifique dependências: pip install -r requirements.txt"
    echo "   3. Se erro de GPU/memória, force CPU: cd filo_priori && python scripts/core/run_experiment_server.py --full-test --device cpu"
    echo "   4. Se CUDA OOM, reduza batch_size em filo_priori/scripts/core/run_experiment_server.py (linha 96: batch_size: 16 -> 8)"
    echo "   5. Verifique logs em filo_priori/results/execution_XXX/ (se criado)"
    echo ""
    echo "📋 Para execução longa em servidor, use:"
    echo "   nohup ./run_full_test.sh > full_test.log 2>&1 &"
    echo "   tail -f full_test.log  # para monitorar"
fi

exit $EXIT_CODE

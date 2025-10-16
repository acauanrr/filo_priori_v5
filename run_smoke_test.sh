#!/bin/bash
# Script helper para executar smoke test do Filo-Priori V5 (BGE + SAINT)
# Uso: ./run_smoke_test.sh

echo "🚀 Executando Filo-Priori V5 - Smoke Test (BGE + SAINT)"
echo "=========================================================="
echo "📌 Modelo: BGE-large-en-v1.5 (1024D) + SAINT Transformer"
echo "📌 SAINT: 6 layers, 8 heads, intersample attention"
echo "📌 Smoke test: 100 builds treino, 50 builds teste"
echo "📌 Otimizações: batch_size=256 para BGE, weights_only=False para PyTorch 2.6+"
echo "⏱️  Tempo estimado: ~15-30 minutos (GPU) ou ~1-2 horas (CPU)"
echo ""

cd filo_priori

# Auto-detect device
if command -v nvidia-smi &> /dev/null; then
    GPU_AVAILABLE=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
    if [ "$GPU_AVAILABLE" -gt 0 ]; then
        echo "🎮 GPU detectada: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
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
echo "🔧 Iniciando pipeline..."
echo ""

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
    echo ""
    echo "📊 Para ver métricas APFD:"
    echo "   cat filo_priori/results/$LAST_EXEC/metrics.json | grep -A 20 apfd_per_build"
else
    echo "❌ Smoke test falhou com código de erro: $EXIT_CODE"
    echo ""
    echo "💡 Troubleshooting:"
    echo "   1. Verifique se o venv está ativado: source venv/bin/activate"
    echo "   2. Verifique dependências: pip install -r requirements.txt"
    echo "   3. Se erro de GPU, force CPU: cd filo_priori && python scripts/core/run_experiment_server.py --smoke-train 100 --smoke-test 50 --device cpu"
    echo "   4. Verifique logs em filo_priori/results/execution_XXX/ (se criado)"
fi

exit $EXIT_CODE

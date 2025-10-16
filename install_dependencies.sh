#!/bin/bash
# Script para instalar dependências do Filo-Priori V5 (BGE + TabPFN)
# Uso: ./install_dependencies.sh

echo "📦 Instalando dependências do Filo-Priori V5"
echo "=============================================="
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "🐍 Python version: $PYTHON_VERSION"

if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 não encontrado. Instale Python 3.8+ primeiro."
    exit 1
fi

echo ""
echo "📥 Instalando pacotes do requirements.txt..."
echo ""

pip install -r requirements.txt

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Dependências instaladas com sucesso!"
    echo ""
    echo "🔍 Verificando instalação dos componentes principais..."
    echo ""

    # Test TabPFN
    python3 -c "from tabpfn import TabPFNClassifier; print('✓ TabPFN instalado')" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "⚠️  TabPFN não está disponível. Tentando instalar novamente..."
        pip install tabpfn
    fi

    # Test sentence-transformers
    python3 -c "from sentence_transformers import SentenceTransformer; print('✓ sentence-transformers instalado')" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "⚠️  sentence-transformers não está disponível"
    fi

    # Test BGE model download
    echo ""
    echo "🔽 Baixando modelo BGE-large-en-v1.5 (primeira vez pode demorar)..."
    python3 -c "
from sentence_transformers import SentenceTransformer
import sys
try:
    model = SentenceTransformer('BAAI/bge-large-en-v1.5')
    dim = model.get_sentence_embedding_dimension()
    print(f'✓ BGE-large-en-v1.5 carregado ({dim}D)')
    if dim != 1024:
        print(f'⚠️  Esperado 1024D, obtido {dim}D')
        sys.exit(1)
except Exception as e:
    print(f'❌ Erro ao carregar BGE: {e}')
    sys.exit(1)
" 2>/dev/null

    if [ $? -eq 0 ]; then
        echo ""
        echo "🎉 Instalação completa e verificada!"
        echo ""
        echo "▶️  Próximos passos:"
        echo "   1. Execute smoke test: ./run_smoke_test.sh"
        echo "   2. Se funcionar, execute full test: ./run_full_test.sh"
        echo ""
    else
        echo ""
        echo "⚠️  Alguns componentes podem não estar funcionando corretamente"
    fi
else
    echo "❌ Falha ao instalar dependências"
    echo "💡 Tente manualmente: pip install -r requirements.txt"
fi

exit $EXIT_CODE

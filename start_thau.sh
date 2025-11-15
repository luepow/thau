#!/bin/bash

# Script de inicio rápido para THAU
# Trainable Helpful AI Unit

set -e

echo "=================================================="
echo "🤖 THAU - Sistema de Entrenamiento Autónomo"
echo "=================================================="
echo ""

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Verificar Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 no encontrado${NC}"
    echo "Instala Python 3.10+ desde https://www.python.org/"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo -e "${GREEN}✓${NC} Python detectado: $(python3 --version)"

# Verificar Ollama
if ! command -v ollama &> /dev/null; then
    echo -e "${YELLOW}⚠️  Ollama no encontrado${NC}"
    echo "Instala Ollama desde https://ollama.ai/"
    echo "THAU necesita Ollama para funcionar"
    exit 1
fi

echo -e "${GREEN}✓${NC} Ollama detectado: $(ollama --version 2>/dev/null || echo 'instalado')"

# Verificar que Ollama esté corriendo
if ! curl -s http://localhost:11434/api/version > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠️  Ollama no está corriendo${NC}"
    echo "Inicia Ollama con: ollama serve"
    exit 1
fi

echo -e "${GREEN}✓${NC} Ollama está corriendo"

# Verificar modelo base
if ! ollama list | grep -q "qwen2.5-coder:1.5b-base"; then
    echo -e "${YELLOW}⚠️  Modelo base no encontrado${NC}"
    echo "Descargando qwen2.5-coder:1.5b-base..."
    ollama pull qwen2.5-coder:1.5b-base
fi

echo -e "${GREEN}✓${NC} Modelo base disponible"

# Verificar entorno virtual
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}⚠️  Entorno virtual no encontrado${NC}"
    echo "Creando entorno virtual..."
    python3 -m venv venv
fi

echo -e "${GREEN}✓${NC} Entorno virtual disponible"

# Activar entorno
source venv/bin/activate

# Verificar dependencias
if ! python -c "import fastapi" 2>/dev/null; then
    echo -e "${YELLOW}⚠️  Instalando dependencias...${NC}"
    pip install -r requirements.txt -q
fi

echo -e "${GREEN}✓${NC} Dependencias instaladas"

# Crear directorios necesarios
mkdir -p data/logs
mkdir -p data/memory
mkdir -p data/language/vocabulary
mkdir -p data/language
mkdir -p data/training_queue
mkdir -p data/datasets/auto_generated

echo -e "${GREEN}✓${NC} Directorios creados"

# Modo de operación
MODE="${1:-api}"

echo ""
echo "=================================================="
echo "🚀 Iniciando THAU..."
echo "=================================================="
echo ""

if [ "$MODE" = "api" ]; then
    echo "Modo: API Server"
    echo "URL: http://localhost:8000"
    echo "Docs: http://localhost:8000/docs"
    echo ""
    echo "Presiona Ctrl+C para detener"
    echo ""
    python api/thau_api_integrated.py

elif [ "$MODE" = "test" ]; then
    echo "Modo: Testing"
    echo ""
    python thau_trainer/integrated_trainer.py

elif [ "$MODE" = "mcp" ]; then
    echo "Modo: MCP Server Test"
    echo ""
    python thau_trainer/mcp_server.py

else
    echo "Uso: ./start_thau.sh [api|test|mcp]"
    echo ""
    echo "  api  - Inicia API server (por defecto)"
    echo "  test - Ejecuta pruebas del sistema"
    echo "  mcp  - Prueba servidor MCP"
    exit 1
fi

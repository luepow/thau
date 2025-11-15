#!/bin/bash

# Script para desplegar el modelo entrenado a Ollama

set -e

echo "🚀 Desplegando Arquitecto de Software AI a Ollama"
echo "="

# Verificar que Ollama esté instalado
if ! command -v ollama &> /dev/null; then
    echo "❌ Ollama no está instalado"
    echo "   Instálalo desde: https://ollama.com"
    exit 1
fi

echo "✅ Ollama detectado"

# Verificar que el modelo entrenado exista
if [ ! -d "./data/checkpoints/architecture-expert" ]; then
    echo "❌ Modelo entrenado no encontrado en ./data/checkpoints/architecture-expert"
    echo "   Entrena el modelo primero con:"
    echo "   python scripts/train_architecture.py"
    exit 1
fi

echo "✅ Modelo entrenado encontrado"

# Verificar que el Modelfile exista
if [ ! -f "./Modelfile" ]; then
    echo "❌ Modelfile no encontrado"
    exit 1
fi

echo "✅ Modelfile encontrado"

# Crear el modelo en Ollama
echo ""
echo "📦 Creando modelo en Ollama..."
ollama create architecture-expert -f Modelfile

echo ""
echo "✅ Modelo desplegado exitosamente!"
echo ""
echo "🎯 Para usar el modelo:"
echo "   ollama run architecture-expert"
echo ""
echo "💬 Ejemplo de uso:"
echo '   ollama run architecture-expert "¿Qué es Clean Architecture?"'
echo ""
echo "="

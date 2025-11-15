# THAU - Guía Rápida de Exportación

Pasos rápidos para usar THAU en Ollama o LMStudio.

## Opción 1: Ollama (Recomendado)

### Paso 1: Instalar Ollama

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows: descargar desde https://ollama.com/download
```

### Paso 2: Exportar modelo

```bash
# Activar entorno virtual
cd /Users/lperez/Workspace/Development/fullstack/thau_1_0/my-llm
source venv/bin/activate

# Exportar THAU a GGUF (tarda 5-10 minutos)
python export/export_to_gguf.py
```

Esto creará:
- `export/models/thau-f16.gguf` (~2.2 GB)
- `export/models/thau-f16-Q4_K_M.gguf` (~700 MB) ← Usar este

### Paso 3: Crear modelo en Ollama

```bash
cd export
ollama create thau -f Modelfile
```

### Paso 4: Usar THAU

```bash
# Chat interactivo
ollama run thau

# O via API
curl http://localhost:11434/api/generate -d '{
  "model": "thau",
  "prompt": "Hola, ¿quién eres?",
  "stream": false
}'
```

### Paso 5: Probar funcionamiento

```bash
python export/test_exported_model.py
```

---

## Opción 2: LM Studio

### Paso 1: Instalar LM Studio

Descargar desde: https://lmstudio.ai/

### Paso 2: Exportar modelo

```bash
cd /Users/lperez/Workspace/Development/fullstack/thau_1_0/my-llm
source venv/bin/activate
python export/export_to_gguf.py
```

### Paso 3: Cargar en LM Studio

1. Abrir LM Studio
2. Click en "Load Model" / "Import Model"
3. Navegar a: `export/models/thau-f16-Q4_K_M.gguf`
4. Click "Load"

### Paso 4: Chat

- Ir a pestaña "Chat"
- Escribir mensaje
- THAU responderá

---

## Problemas Comunes

### Error: "llama.cpp not found"

```bash
cd export
git clone https://github.com/ggerganov/llama.cpp.git
```

### Error: "Model not found" en Ollama

```bash
# Verificar modelos instalados
ollama list

# Recrear modelo
ollama rm thau
cd export
ollama create thau -f Modelfile
```

### Error: "Out of memory"

Use cuantización más agresiva:

```bash
python export/export_to_gguf.py --quantization Q4_0
```

---

## Estructura de Archivos

```
export/
├── export_to_gguf.py      # Script de exportación
├── Modelfile               # Configuración Ollama
├── README.md               # Documentación completa
├── QUICKSTART.md           # Esta guía
├── test_exported_model.py  # Testing
├── models/                 # Modelos exportados
│   ├── thau-f16.gguf
│   └── thau-f16-Q4_K_M.gguf
└── llama.cpp/              # Herramientas de conversión (auto)
```

---

## Próximos Pasos

- Lee `export/README.md` para documentación completa
- Prueba con `python export/test_exported_model.py`
- Personaliza el Modelfile para ajustar comportamiento
- Integra con tu aplicación via API

---

## Soporte

Si tienes problemas, consulta:
- `export/README.md` - Documentación completa
- Troubleshooting section - Soluciones a errores comunes

¡Disfruta usando THAU!

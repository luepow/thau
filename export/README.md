# THAU Model Export

Guía completa para exportar THAU y usarlo en Ollama o LMStudio.

## Tabla de Contenidos

1. [Requisitos](#requisitos)
2. [Exportación del Modelo](#exportación-del-modelo)
3. [Uso con Ollama](#uso-con-ollama)
4. [Uso con LMStudio](#uso-con-lmstudio)
5. [Testing](#testing)
6. [Troubleshooting](#troubleshooting)

---

## Requisitos

### Software Necesario

- **Python 3.10+** con pip
- **Git** para clonar repositorios
- **C/C++ compiler** para compilar llama.cpp
  - macOS: Xcode Command Line Tools (`xcode-select --install`)
  - Linux: `build-essential`
  - Windows: Visual Studio Build Tools

### Espacio en Disco

- Modelo base HuggingFace: ~2.2 GB
- Modelo GGUF f16: ~2.2 GB
- Modelo GGUF cuantizado (Q4_K_M): ~700 MB
- Total recomendado: **6-8 GB libres**

### Para Ollama

- **Ollama** instalado
  - macOS: `brew install ollama`
  - Linux: `curl -fsSL https://ollama.com/install.sh | sh`
  - Windows: Descargar desde https://ollama.com/download

### Para LMStudio

- **LM Studio** descargado desde https://lmstudio.ai/

---

## Exportación del Modelo

### Paso 1: Activar entorno virtual

```bash
cd /path/to/my-llm
source venv/bin/activate  # Linux/macOS
# o
venv\Scripts\activate  # Windows
```

### Paso 2: Ejecutar script de exportación

```bash
# Exportación con cuantización (recomendado)
python export/export_to_gguf.py

# O con opciones personalizadas
python export/export_to_gguf.py --quantization Q5_K_M --output-dir ./export/models
```

#### Opciones Disponibles

- `--model`: Modelo de HuggingFace a exportar (default: TinyLlama/TinyLlama-1.1B-Chat-v1.0)
- `--output-dir`: Directorio de salida (default: ./export/models)
- `--no-quantize`: No cuantizar el modelo (mantener f16)
- `--quantization`: Tipo de cuantización (Q4_0, Q4_K_M, Q5_0, Q5_K_M, Q8_0)

#### Tipos de Cuantización

| Tipo | Tamaño | Calidad | Velocidad | Recomendado para |
|------|--------|---------|-----------|------------------|
| Q4_0 | 600 MB | Buena | Rápida | CPU limitadas |
| Q4_K_M | 700 MB | Muy buena | Rápida | **Uso general** |
| Q5_K_M | 850 MB | Excelente | Media | Mejor calidad |
| Q8_0 | 1.2 GB | Casi perfecta | Lenta | GPU potentes |

**Recomendación:** Q4_K_M ofrece el mejor balance calidad/tamaño.

### Paso 3: Verificar archivos exportados

```bash
ls -lh export/models/

# Deberías ver:
# thau-f16.gguf           (~2.2 GB)  - Modelo sin cuantizar
# thau-f16-Q4_K_M.gguf    (~700 MB)  - Modelo cuantizado
```

---

## Uso con Ollama

### Paso 1: Crear el modelo en Ollama

```bash
cd /path/to/my-llm/export
ollama create thau -f Modelfile
```

### Paso 2: Verificar que se creó correctamente

```bash
ollama list

# Deberías ver:
# NAME    ID              SIZE    MODIFIED
# thau    abc123def456    700 MB  5 seconds ago
```

### Paso 3: Ejecutar THAU en Ollama

```bash
# Modo interactivo
ollama run thau

# Responderá con el prompt de THAU
>>> Hola, ¿cómo estás?
Hola! Soy THAU...

# Salir: Ctrl+D o escribir /bye
```

### Paso 4: Usar THAU via API

```bash
# Terminal 1: Iniciar servidor Ollama (si no está corriendo)
ollama serve

# Terminal 2: Hacer peticiones
curl http://localhost:11434/api/generate -d '{
  "model": "thau",
  "prompt": "¿Qué es Python?",
  "stream": false
}'
```

### Paso 5: Integración con tu código

```python
import requests

def chat_with_thau(prompt: str) -> str:
    """Chat con THAU via Ollama API"""
    response = requests.post(
        'http://localhost:11434/api/generate',
        json={
            'model': 'thau',
            'prompt': prompt,
            'stream': False
        }
    )
    return response.json()['response']

# Uso
respuesta = chat_with_thau("Explica qué es THAU")
print(respuesta)
```

### Configuración Avanzada de Ollama

Puedes personalizar el Modelfile:

```dockerfile
# export/Modelfile

FROM ./models/thau-f16-Q4_K_M.gguf

# Ajustar temperatura (0.0-2.0)
PARAMETER temperature 0.8

# Tokens máximos de contexto
PARAMETER num_ctx 4096

# Tokens máximos a generar
PARAMETER num_predict 1024

# Penalización por repetición
PARAMETER repeat_penalty 1.2

# System prompt personalizado
SYSTEM """Eres THAU, un asistente especializado en [TU DOMINIO]..."""
```

Luego recrea el modelo:

```bash
ollama rm thau
ollama create thau -f Modelfile
```

---

## Uso con LMStudio

### Paso 1: Abrir LM Studio

- Descarga LM Studio desde https://lmstudio.ai/
- Instala y abre la aplicación

### Paso 2: Cargar el modelo GGUF

1. En LM Studio, ve a **"Load Model"** o **"Import Model"**
2. Navega a: `my-llm/export/models/`
3. Selecciona: `thau-f16-Q4_K_M.gguf`
4. Click en **"Load"**

### Paso 3: Configurar parámetros

En la pestaña **Settings** o **Parameters**:

```
Context Length: 2048
Temperature: 0.7
Top P: 0.9
Top K: 40
Repeat Penalty: 1.1
Max Tokens: 512
```

### Paso 4: Chat interactivo

- Ve a la pestaña **"Chat"**
- Escribe tu mensaje en el cuadro de texto
- THAU responderá usando el modelo cargado

### Paso 5: Usar la API local de LMStudio

LM Studio expone una API compatible con OpenAI:

```python
import openai

# Configurar cliente para LMStudio
client = openai.OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio"  # API key dummy
)

# Chat con THAU
response = client.chat.completions.create(
    model="thau-f16-Q4_K_M",
    messages=[
        {"role": "system", "content": "Eres THAU, un asistente útil."},
        {"role": "user", "content": "¿Qué puedes hacer?"}
    ],
    temperature=0.7,
    max_tokens=512
)

print(response.choices[0].message.content)
```

### Paso 6: Exportar conversaciones

LM Studio permite exportar conversaciones:

1. Click en el menú de conversación (⋮)
2. **Export** → JSON/Markdown
3. Guarda para análisis posterior

---

## Testing

### Script de testing automático

```bash
python export/test_exported_model.py
```

Este script:
- Verifica que Ollama está corriendo
- Prueba el modelo con varias preguntas
- Mide tiempo de respuesta y tokens por segundo
- Genera un reporte de pruebas

### Testing manual con Ollama

```bash
# Test 1: Presentación
ollama run thau "Preséntate brevemente"

# Test 2: Razonamiento
ollama run thau "¿Cuánto es 15 * 24? Piensa paso a paso"

# Test 3: Programación
ollama run thau "Escribe una función Python para calcular fibonacci"

# Test 4: Español
ollama run thau "Explica qué es la inteligencia artificial"

# Test 5: Seguir instrucciones
ollama run thau "Lista 3 características de un buen código"
```

### Benchmarks de rendimiento

```bash
# Medir tokens por segundo
time ollama run thau "Escribe un ensayo de 500 palabras sobre la IA" > /dev/null

# Comparar diferentes cuantizaciones
# Q4_K_M vs Q5_K_M vs Q8_0
```

---

## Troubleshooting

### Error: "llama.cpp not found"

```bash
# El script debería clonar automáticamente, pero si falla:
cd export
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
make quantize
```

### Error: "Cannot find convert script"

```bash
cd export/llama.cpp
git pull origin master  # Actualizar a última versión
pip install -r requirements.txt
```

### Error: "Model file not found" en Ollama

```bash
# Verifica que el path en Modelfile es correcto
cat export/Modelfile | grep FROM

# Debe ser:
FROM ./models/thau-f16-Q4_K_M.gguf

# O usar path absoluto:
FROM /absolute/path/to/export/models/thau-f16-Q4_K_M.gguf
```

### Error: "Out of memory" durante conversión

Opciones:

1. Cerrar otras aplicaciones
2. Usar cuantización más agresiva (Q4_0 en lugar de Q4_K_M)
3. Convertir sin cuantizar: `--no-quantize`

### Ollama no responde / API timeout

```bash
# Reiniciar Ollama
ollama stop
ollama serve

# O en macOS con Homebrew
brew services restart ollama
```

### LMStudio: "Model failed to load"

- Verifica que el archivo .gguf no está corrupto
- Intenta con una cuantización diferente (Q5_K_M en lugar de Q4_K_M)
- Verifica que tienes suficiente RAM/VRAM

### Respuestas de baja calidad

1. **Ajusta temperature**: Valores más bajos (0.3-0.5) = más determinista
2. **Aumenta context**: Permite más contexto en el prompt
3. **Usa mejor cuantización**: Q5_K_M o Q8_0 en lugar de Q4_K_M
4. **Mejora el system prompt**: Sé más específico en Modelfile

---

## Información Adicional

### Modelos Exportados

- **thau-f16.gguf**: Precisión float16, máxima calidad, 2.2 GB
- **thau-f16-Q4_K_M.gguf**: Cuantizado 4-bit, recomendado, 700 MB
- **thau-f16-Q5_K_M.gguf**: Cuantizado 5-bit, mejor calidad, 850 MB

### Compatibilidad

| Plataforma | Ollama | LMStudio | Notas |
|------------|--------|----------|-------|
| macOS M1/M2/M3 | ✅ | ✅ | Metal acceleration |
| macOS Intel | ✅ | ✅ | CPU only |
| Linux + NVIDIA | ✅ | ✅ | CUDA acceleration |
| Linux + AMD | ✅ | ⚠️ | ROCm support varies |
| Windows + NVIDIA | ✅ | ✅ | CUDA acceleration |
| Windows CPU | ✅ | ✅ | Slower performance |

### Licencia

THAU está basado en TinyLlama-1.1B-Chat-v1.0, licenciado bajo Apache 2.0.

### Soporte

- Issues: https://github.com/tu-repo/my-llm/issues
- Documentación: https://github.com/tu-repo/my-llm/wiki

---

## Próximos Pasos

1. **Fine-tuning personalizado**: Entrena THAU en tu dominio específico
2. **Integración con tu app**: Usa la API de Ollama/LMStudio
3. **Monitoreo**: Implementa logging de uso y calidad de respuestas
4. **Optimización**: Ajusta parámetros según tus necesidades

---

**¡Disfruta usando THAU! 🚀**

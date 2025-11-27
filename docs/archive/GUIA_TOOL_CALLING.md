# Guía: THAU con Tool Calling para Generación de Imágenes

## Resumen

THAU ahora puede **decidir automáticamente cuándo generar imágenes** y ejecutar la generación mediante tool calling. Esta guía cubre todo el flujo:

1. ✅ Entrenamiento de THAU para tool calling
2. ✅ Exportación a GGUF para Ollama
3. ✅ Integración con API de generación de imágenes
4. ✅ Uso interactivo

---

## Arquitectura

```
┌─────────────┐
│   Usuario   │
└──────┬──────┘
       │ "Genera una imagen de un gato espacial"
       ▼
┌────────────────────────────┐
│   thau_chat.py            │  ← Script de integración
│   (Interfaz CLI)          │
└──────┬────────────────────┘
       │
       ▼
┌────────────────────────────┐
│   THAU en Ollama          │  ← Modelo entrenado con tool calling
│   (TinyLlama fine-tuned)  │
└──────┬────────────────────┘
       │
       │ Output: "¡Claro! <TOOL:generate_image>{"prompt": "..."}</TOOL>"
       │
       ▼
┌────────────────────────────┐
│   thau_chat.py            │  ← Parsea tool call
│   (Tool Parser)           │
└──────┬────────────────────┘
       │
       ▼
┌────────────────────────────┐
│   Vision API              │  ← POST /vision/generate
│   (Stable Diffusion)      │
└──────┬────────────────────┘
       │
       │ Response: {"image_url": "/vision/image/..."}
       │
       ▼
┌─────────────┐
│   Usuario   │  ← Ve la imagen generada
└─────────────┘
```

---

## 1. Entrenamiento

### Dataset de Tool Calling

El dataset incluye **30 ejemplos balanceados**:

- **15 ejemplos con tool calling**: Cuando el usuario pide una imagen
- **15 ejemplos sin tool calling**: Conversación normal

**Formato de ejemplo:**

```
### Instrucción:
Eres THAU, un asistente de IA con capacidad de generar imágenes.
Cuando el usuario solicite una imagen o visualización, usa el formato:
<TOOL:generate_image>{"prompt": "descripción en inglés"}</TOOL>
Para conversación normal, responde directamente sin usar herramientas.

### Usuario:
Genera una imagen de un gato espacial

### Asistente:
¡Claro! Voy a generar esa imagen para ti.
<TOOL:generate_image>{"prompt": "a space cat floating in cosmos, astronaut suit, stars and planets, digital art, detailed"}</TOOL>
```

### Ejecutar Entrenamiento

```bash
# Activar entorno
source venv/bin/activate

# Entrenar THAU con tool calling (3 epochs, ~10-15 min)
python train_tool_calling.py --epochs 3 --lr 5e-5 --grad-accum 4

# Ver progreso
# El entrenamiento guardará checkpoints en:
# ./data/checkpoints/incremental/tool_calling_final/
```

### Parámetros de Entrenamiento

- `--epochs`: Número de pasadas completas por el dataset (default: 3)
- `--lr`: Learning rate (default: 5e-5) - Bajo para no olvidar conocimiento previo
- `--grad-accum`: Pasos de acumulación de gradientes (default: 4)
- `--test-only`: Solo probar modelo sin entrenar

### Resultados Esperados

Después del entrenamiento, THAU:

- ✅ Detecta peticiones de imágenes en español
- ✅ Genera el formato correcto de tool calling
- ✅ Traduce descripciones a inglés (mejor calidad en Stable Diffusion)
- ✅ Mantiene conversación normal cuando no se piden imágenes
- ✅ Reconoce contextos donde es apropiado generar visualizaciones

---

## 2. Exportar a GGUF

Una vez entrenado, exportamos THAU para usarlo en Ollama:

```bash
# Método 1: Usando el exportador de GGUF
python export/export_to_gguf.py \
    --model-path ./data/checkpoints/incremental/tool_calling_final \
    --output-name thau-tool-calling

# El script:
# 1. Carga el modelo con adaptadores LoRA
# 2. Fusiona los adaptadores con el modelo base
# 3. Exporta a formato GGUF F16
# 4. Guarda en ./export/gguf/thau-tool-calling-f16.gguf
```

### Importar a Ollama

```bash
# Navegar a carpeta de export
cd export/gguf

# Crear Modelfile
cat > Modelfile-tool-calling <<EOF
FROM ./thau-tool-calling-f16.gguf

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 2048

SYSTEM """Eres THAU, un asistente de IA con capacidades de aprendizaje incremental y generación de imágenes.

Cuando el usuario solicite una imagen o visualización, usa el formato exacto:
<TOOL:generate_image>{"prompt": "descripción detallada en inglés"}</TOOL>

Para conversación normal, responde directamente sin usar herramientas.

Características:
- Aprendes de cada interacción
- Generas imágenes mediante Stable Diffusion
- Respondes en español de forma natural
- Ayudas con programación, conceptos técnicos y más

Sé conversacional, amigable y preciso."""
EOF

# Importar a Ollama
ollama create thau-tool-calling -f Modelfile-tool-calling

# Verificar
ollama list | grep thau-tool-calling
```

---

## 3. Configurar API de Imágenes

La API debe estar corriendo para que THAU pueda generar imágenes:

```bash
# Terminal 1: Iniciar API
source venv/bin/activate
python api/main.py

# Output esperado:
# INFO:     Uvicorn running on http://127.0.0.1:8000
# INFO:     Application startup complete.
```

### Verificar API

```bash
# Verificar salud
curl http://localhost:8000/health

# Probar generación directa
curl -X POST "http://localhost:8000/vision/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a cute robot",
    "num_inference_steps": 30,
    "width": 512,
    "height": 512
  }'

# Ver ejemplos disponibles
curl http://localhost:8000/vision/examples
```

---

## 4. Usar THAU con Tool Calling

### Opción A: Modo Interactivo

```bash
# Activar entorno
source venv/bin/activate

# Iniciar chat interactivo
python thau_chat.py

# Output:
# 🤖 THAU Chat iniciado
#    Modelo: thau-tool-calling:latest
#    API: http://localhost:8000
#
# 💬 THAU Chat Interactivo
# Comandos:
#   - Escribe tu mensaje y presiona Enter
#   - 'salir' o 'exit' para terminar
#   - 'limpiar' para borrar historial
#
# 👤 Tú: _
```

### Ejemplo de Conversación

```
👤 Tú: Hola THAU

🤖 THAU: ¡Hola! Soy THAU, tu asistente de IA. Puedo ayudarte con explicaciones,
programación, y también generar imágenes si lo necesitas. ¿En qué puedo ayudarte hoy?

👤 Tú: Genera una imagen de un perro astronauta

🤔 THAU pensando...
🤖 THAU: ¡Claro! Voy a generar esa imagen para ti.

🎨 Generando imagen: 'an astronaut dog in space suit, floating in cosmos, st...'
✅ Imagen generada: /vision/image/20250114_171500_astronaut_dog.png

🖼️  Imagen disponible en: http://localhost:8000/vision/image/20250114_171500_astronaut_dog.png
📁 Guardada en: ./data/generated_images/20250114_171500_astronaut_dog.png
   (Abriendo imagen...)

👤 Tú: ¿Qué es machine learning?

🤖 THAU: Machine learning es un subcampo de la inteligencia artificial que permite
a las computadoras aprender de datos sin ser programadas explícitamente. Los algoritmos
de ML identifican patrones en datos y hacen predicciones...

👤 Tú: salir

👋 ¡Hasta luego!
```

### Opción B: Mensaje Único

```bash
# Enviar un solo mensaje
python thau_chat.py --message "Genera una imagen de montañas nevadas al atardecer"

# Output:
# 🤖 THAU Chat iniciado
#    Modelo: thau-tool-calling:latest
#    API: http://localhost:8000
#
# 🤖 THAU: Perfecto, te genero esa imagen.
#
# 🎨 Generando imagen: 'snow-capped mountains at sunset, golden hour lighting...'
# ✅ Imagen generada: /vision/image/20250114_171600_mountains_sunset.png
# 🖼️  Imagen disponible en: http://localhost:8000/vision/image/...
```

### Parámetros CLI

```bash
python thau_chat.py [OPTIONS]

Options:
  --model TEXT    Nombre del modelo en Ollama (default: thau-tool-calling:latest)
  --api TEXT      URL base de la API (default: http://localhost:8000)
  --message TEXT  Enviar un mensaje único (modo no interactivo)
```

---

## 5. Cómo Funciona

### Flujo de Detección y Ejecución

1. **Usuario envía mensaje** → `python thau_chat.py`

2. **thau_chat.py llama a Ollama** → `ollama run thau-tool-calling`
   - Pasa el mensaje al modelo entrenado

3. **THAU genera respuesta**:
   - **Con tool**: `"¡Claro! <TOOL:generate_image>{"prompt": "..."}</TOOL>"`
   - **Sin tool**: `"Python es un lenguaje de programación..."`

4. **thau_chat.py parsea respuesta**:
   ```python
   pattern = r'<TOOL:generate_image>(.*?)</TOOL>'
   match = re.search(pattern, response)
   ```

5. **Si detecta tool call**:
   - Extrae el JSON con el prompt
   - Llama a `POST /vision/generate`
   - Muestra la imagen al usuario

6. **Si NO detecta tool call**:
   - Solo muestra la respuesta de texto

### Formato de Tool Call

```
<TOOL:generate_image>{"prompt": "description", "num_inference_steps": 30}</TOOL>
```

**Parámetros soportados:**

- `prompt` (requerido): Descripción de la imagen
- `negative_prompt`: Qué evitar
- `num_inference_steps`: Pasos de generación (10-100)
- `guidance_scale`: Fidelidad al prompt (1.0-20.0)
- `width`: Ancho en píxeles (256-1024)
- `height`: Alto en píxeles (256-1024)
- `seed`: Semilla para reproducibilidad

---

## 6. Patrones de Uso

### Peticiones Directas

THAU detecta estas frases:

- "Genera una imagen de..."
- "Crea una imagen de..."
- "Dibuja..."
- "Muéstrame una imagen de..."
- "Haz una imagen de..."
- "Quiero ver una imagen de..."

### Peticiones Contextuales

THAU también detecta cuando es apropiado generar visualizaciones:

```
👤: Explícame qué es recursión con una imagen visual

🤖: La recursión es cuando una función se llama a sí misma.
Déjame mostrarte una representación visual.
<TOOL:generate_image>{"prompt": "recursion visualized as mirrors within mirrors..."}</TOOL>
```

### Conversación Normal

THAU **NO** usa tool calling para:

- Preguntas generales
- Explicaciones de conceptos
- Programación
- Saludos y conversación casual

---

## 7. Troubleshooting

### Error: "Ollama no encontrado"

```bash
# Verificar instalación
ollama --version

# Si no está instalado:
# macOS: brew install ollama
# Linux: curl -fsSL https://ollama.ai/install.sh | sh
```

### Error: "No se puede conectar a la API"

```bash
# Verificar que la API esté corriendo
curl http://localhost:8000/health

# Si no responde, iniciar API
python api/main.py
```

### Error: "Model 'thau-tool-calling' not found"

```bash
# Listar modelos disponibles
ollama list

# Si no está, crear el modelo:
cd export/gguf
ollama create thau-tool-calling -f Modelfile-tool-calling
```

### THAU no genera tool calls

**Posibles causas:**

1. **Modelo no entrenado**: Ejecutar `train_tool_calling.py` primero
2. **Modelo base sin fine-tune**: Verificar que se exportó el checkpoint correcto
3. **Prompt no claro**: Usar frases más explícitas como "Genera una imagen de..."

**Verificación:**

```bash
# Probar el modelo directamente en Ollama
ollama run thau-tool-calling "Genera una imagen de un robot"

# Debe responder con formato <TOOL:generate_image>...</TOOL>
```

### Imágenes no se generan

**Verificar API:**

```bash
# Test manual
curl -X POST "http://localhost:8000/vision/generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a test image"}'
```

**Verificar logs:**

```bash
# Ver logs de la API
tail -f data/logs/my-llm.log
```

---

## 8. Ejemplos de Prompts Efectivos

### Para THAU

```
✅ "Genera una imagen de un gato astronauta en el espacio"
✅ "Crea una imagen de montañas nevadas al atardecer"
✅ "Dibuja un robot aprendiendo a programar"
✅ "Muéstrame una imagen de un bosque encantado"
✅ "Explícame recursión con una imagen visual"
```

### Prompts que THAU traduce bien

THAU aprende a mejorar las descripciones al traducir:

```
Usuario:  "Genera una imagen de un gato espacial"
THAU:     <TOOL:generate_image>{"prompt": "a space cat floating in cosmos, astronaut suit, stars and planets, digital art, detailed"}</TOOL>
```

---

## 9. Optimizaciones

### Reducir Latencia de Generación

```bash
# En thau_chat.py, modificar parámetros por defecto
payload = {
    "prompt": params.get("prompt", ""),
    "num_inference_steps": 20,  # ← Reducir de 30 a 20
    "width": 384,               # ← Reducir resolución
    "height": 384,
}
```

### Mejorar Calidad de Imágenes

```bash
# Aumentar pasos y resolución
payload = {
    "num_inference_steps": 50,  # ← Más pasos
    "guidance_scale": 8.5,      # ← Mayor fidelidad
    "width": 768,
    "height": 768,
}
```

### Usar GGUF Cuantizado

```bash
# Cuantizar a Q4_K_M (más rápido, menor calidad)
cd ~/.ollama/models/gguf
llama.cpp/quantize thau-tool-calling-f16.gguf thau-tool-calling-q4.gguf Q4_K_M

# Actualizar Modelfile
FROM ./thau-tool-calling-q4.gguf

# Recrear modelo
ollama create thau-tool-calling-q4 -f Modelfile-tool-calling-q4
```

---

## 10. Próximos Pasos

### Mejorar Dataset

Agregar más ejemplos:

```bash
# Editar dataset
vim data/datasets/tool_calling_dataset.json

# Agregar nuevos ejemplos de:
# - Más estilos de arte (anime, sketch, oil painting)
# - Conceptos técnicos (diagramas, arquitectura)
# - Casos edge (peticiones ambiguas)

# Re-entrenar
python train_tool_calling.py --epochs 3
```

### Agregar Más Tools

Extender el sistema para otros capabilities:

```python
# En tool_calling_dataset.json
{
  "user": "Ejecuta este código Python",
  "assistant": "<TOOL:code_execution>{"code": "print('Hello')"}</TOOL>"
}

{
  "user": "Busca información sobre Python",
  "assistant": "<TOOL:web_search>{"query": "Python programming language"}</TOOL>"
}
```

### Integrar con Chat UI

Crear interfaz web:

```bash
# Opción: Usar Streamlit
pip install streamlit

# Crear app.py con UI conversacional
streamlit run app.py
```

---

## Comandos Rápidos

```bash
# Setup completo
source venv/bin/activate
python train_tool_calling.py --epochs 3
python export/export_to_gguf.py --model-path ./data/checkpoints/incremental/tool_calling_final
cd export/gguf && ollama create thau-tool-calling -f Modelfile-tool-calling

# Uso
python api/main.py &                    # Terminal 1: API
python thau_chat.py                     # Terminal 2: Chat

# Verificación
ollama list | grep thau                 # Ver modelos
curl http://localhost:8000/health       # Test API
```

---

## Arquitectura de Archivos

```
my-llm/
├── data/
│   ├── datasets/
│   │   └── tool_calling_dataset.json       # 30 ejemplos de entrenamiento
│   ├── checkpoints/
│   │   └── incremental/
│   │       └── tool_calling_final/         # Modelo entrenado
│   └── generated_images/                   # Imágenes generadas
│       └── 20250114_*.png
├── train_tool_calling.py                   # Script de entrenamiento
├── thau_chat.py                            # Interfaz CLI con tool parsing
├── export/
│   ├── export_to_gguf.py                  # Exportador GGUF
│   └── gguf/
│       ├── thau-tool-calling-f16.gguf     # Modelo exportado
│       └── Modelfile-tool-calling         # Config de Ollama
└── api/
    ├── main.py                             # API FastAPI
    └── routes/
        └── vision.py                       # Endpoints de imágenes
```

---

**¡THAU ahora puede decidir cuándo generar imágenes! 🎨🤖**

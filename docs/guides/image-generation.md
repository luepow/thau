# Guía: Generación de Imágenes con THAU
## THAU ahora puede generar y mostrar imágenes

---

## Resumen

THAU ahora tiene capacidad de **generar imágenes** usando Stable Diffusion. Puede:

- 🎨 Generar imágenes desde descripciones de texto
- 🤖 Detectar automáticamente cuando le pides una imagen
- 💾 Guardar las imágenes generadas
- 🌐 Mostrar las imágenes a través de la API
- 📊 Mantener estadísticas de generaciones

---

## Instalación

### 1. Instalar Dependencias

```bash
# Activar entorno virtual
source venv/bin/activate

# Instalar dependencias de generación de imágenes
pip install -r requirements-image-gen.txt

# O instalar manualmente
pip install diffusers Pillow accelerate transformers
```

### 2. Verificar Instalación

```bash
python -c "from capabilities.vision.image_generator import ThauImageGenerator; print('✅ OK')"
```

---

## Uso Básico

### Opción 1: Desde Python

```python
from capabilities.vision.image_generator import ThauImageGenerator

# Inicializar generador
generator = ThauImageGenerator()

# Generar imagen
result = generator.generate_image(
    prompt="a cute robot learning to code, digital art",
    num_inference_steps=30,
    width=512,
    height=512
)

if result['success']:
    print(f"Imagen guardada: {result['path']}")
    # La imagen se abre automáticamente si es posible
```

### Opción 2: Demo Interactivo

```bash
# Ejecutar demo completo
python demo_image_generation.py

# Ejecutar demo específico
python demo_image_generation.py --demo 1  # Generación directa
python demo_image_generation.py --demo 2  # Con conversación
python demo_image_generation.py --demo 3  # Por lotes
```

### Opción 3: A través de la API

```bash
# 1. Iniciar API
python api/main.py

# 2. En otra terminal, generar imagen
curl -X POST "http://localhost:8000/vision/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a futuristic cityscape at sunset",
    "num_inference_steps": 30,
    "width": 512,
    "height": 512
  }'

# 3. Ver imagen generada
# Abre en navegador: http://localhost:8000/vision/image/[filename].png
```

---

## API Endpoints

### 1. Generar Imagen

**POST** `/vision/generate`

```json
{
  "prompt": "a beautiful sunset over mountains",
  "negative_prompt": "blurry, bad quality",
  "num_inference_steps": 30,
  "guidance_scale": 7.5,
  "width": 512,
  "height": 512,
  "seed": null
}
```

**Respuesta:**
```json
{
  "success": true,
  "image_path": "./data/generated_images/20250114_120000_sunset.png",
  "image_url": "/vision/image/20250114_120000_sunset.png",
  "metadata": {
    "prompt": "a beautiful sunset over mountains",
    "timestamp": "2025-01-14T12:00:00",
    ...
  }
}
```

### 2. Chat con Detección Automática

**POST** `/vision/chat`

```json
{
  "message": "Genera una imagen de un gato astronauta",
  "auto_generate_image": true
}
```

**Respuesta:**
```json
{
  "response": "¡Listo! He generado la imagen. Puedes verla en: /vision/image/...",
  "tool_used": "generate_image",
  "image_generated": true,
  "image_path": "./data/generated_images/...",
  "image_url": "/vision/image/..."
}
```

### 3. Ver Imagen Generada

**GET** `/vision/image/{filename}`

```
http://localhost:8000/vision/image/20250114_120000_gato_astronauta.png
```

### 4. Estadísticas

**GET** `/vision/stats`

```json
{
  "total_images_generated": 15,
  "recent_generations": 5,
  "output_directory": "./data/generated_images",
  "model": "runwayml/stable-diffusion-v1-5",
  "device": "mps"
}
```

### 5. Ejemplos de Prompts

**GET** `/vision/examples`

```json
{
  "examples": [
    "a cute robot learning to code, digital art",
    "a futuristic cityscape at sunset, cyberpunk style",
    ...
  ],
  "tips": [
    "Be specific about the style",
    "Include lighting and mood details",
    ...
  ]
}
```

---

## Sistema de Detección Automática

THAU detecta automáticamente cuando le pides una imagen usando estas frases:

- "Genera una imagen de..."
- "Crea una imagen de..."
- "Dibuja..."
- "Muéstrame una imagen de..."
- "Haz una imagen de..."

**Ejemplos:**

```python
from capabilities.tools.tool_registry import get_tool_registry

registry = get_tool_registry()

# Test detección
messages = [
    "Genera una imagen de un gato espacial",
    "Crea una imagen de montañas nevadas",
    "Dibuja un robot",
]

for msg in messages:
    tool = registry.detect_tool_needed(msg)
    if tool:
        params = registry.extract_parameters(msg, tool)
        print(f"Detectado: {tool.name}")
        print(f"Prompt: {params['prompt']}")
```

---

## Parámetros de Generación

### prompt (requerido)
Descripción de la imagen a generar.

**Ejemplos:**
- ✅ "a photorealistic sunset over mountains with vibrant orange and purple sky"
- ✅ "a cute robot learning to code, digital art, colorful, detailed"
- ❌ "imagen bonita" (muy vago)

### negative_prompt
Qué evitar en la imagen.

**Default:** `"blurry, bad quality, distorted"`

**Ejemplos:**
- "blurry, ugly, bad anatomy, watermark, text"
- "low quality, distorted, deformed"

### num_inference_steps (10-100)
Pasos de generación. Más pasos = mejor calidad pero más lento.

- **20-25**: Rápido, calidad aceptable
- **30-40**: Balance calidad/velocidad (recomendado)
- **50+**: Máxima calidad

### guidance_scale (1.0-20.0)
Qué tan estrictamente seguir el prompt.

- **1-5**: Creativo, puede ignorar detalles
- **7-9**: Balance (recomendado)
- **10-15**: Muy literal
- **15+**: Puede generar artefactos

### width/height (256-1024, múltiplo de 8)
Dimensiones en píxeles.

**Opciones comunes:**
- `512x512`: Cuadrado, rápido
- `768x512`: Panorámico horizontal
- `512x768`: Vertical (retratos)
- `1024x1024`: Alta resolución (más lento)

### seed
Semilla para reproducibilidad.

```python
# Misma imagen cada vez
result = generator.generate_image(
    prompt="a cat",
    seed=42
)
```

---

## Ejemplos de Uso

### 1. Generación Simple

```python
from capabilities.vision.image_generator import ThauImageGenerator

gen = ThauImageGenerator()

result = gen.generate_image("a serene lake at dawn")

if result['success']:
    print(f"Guardada en: {result['path']}")
```

### 2. Con Parámetros Avanzados

```python
result = gen.generate_image(
    prompt="a futuristic city with flying cars, neon lights, rain",
    negative_prompt="blurry, distorted, bad quality, people",
    num_inference_steps=40,
    guidance_scale=8.0,
    width=768,
    height=512,
    seed=12345
)
```

### 3. Batch Generation

```python
prompts = [
    "a robot painting a landscape",
    "an AI brain made of circuits",
    "abstract representation of recursion"
]

results = gen.generate_batch(prompts, num_inference_steps=25)

for i, r in enumerate(results):
    if r['success']:
        print(f"{i+1}. {r['path']}")
```

### 4. Con Conversación

```python
from capabilities.tools.tool_registry import get_tool_registry

registry = get_tool_registry()
gen = ThauImageGenerator()

user_input = "Genera una imagen de un atardecer en la playa"

tool = registry.detect_tool_needed(user_input)
if tool and tool.name == "generate_image":
    params = registry.extract_parameters(user_input, tool)
    result = gen.generate_image(**params)

    if result['success']:
        print(f"THAU: ¡Listo! Aquí está: {result['path']}")
```

---

## Estructura de Archivos

```
my-llm/
├── capabilities/
│   ├── vision/
│   │   ├── __init__.py
│   │   └── image_generator.py      # Generador principal
│   └── tools/
│       ├── __init__.py
│       ├── function_calling.py
│       └── tool_registry.py         # Registro de herramientas
├── api/
│   └── routes/
│       └── vision.py                # API endpoints
├── data/
│   └── generated_images/            # Imágenes guardadas
│       ├── 20250114_120000_cat.png
│       └── 20250114_120000_cat.json # Metadata
└── demo_image_generation.py         # Demo interactivo
```

---

## Troubleshooting

### Error: "CUDA out of memory" / "MPS out of memory"

**Solución:**
```python
# Reducir resolución
generator.generate_image(prompt, width=384, height=384)

# O habilitar atención por capas (ya habilitado para MPS)
```

### Error: "Model not found"

**Solución:**
```bash
# Limpiar caché de HuggingFace
rm -rf ~/.cache/huggingface/

# Volver a ejecutar (descargará automáticamente)
python demo_image_generation.py
```

### Imágenes de Baja Calidad

**Solución:**
```python
# Aumentar steps
result = generator.generate_image(
    prompt="...",
    num_inference_steps=50,  # Aumentar de 30 a 50
    guidance_scale=8.0        # Aumentar de 7.5 a 8.0
)

# Mejorar el prompt
prompt = "a photorealistic sunset, golden hour, vibrant colors, 8k, detailed"
```

### Generación Muy Lenta

**Solución:**
```python
# Reducir pasos
num_inference_steps=20

# Reducir resolución
width=384, height=384
```

---

## Tips para Buenos Prompts

### 1. Sé Específico

❌ "un paisaje"
✅ "a serene mountain landscape with snow-capped peaks, pine trees, and a crystal clear lake reflecting the sky, golden hour lighting"

### 2. Incluye el Estilo

Ejemplos:
- "digital art"
- "oil painting"
- "photorealistic"
- "watercolor"
- "cyberpunk style"
- "anime style"

### 3. Menciona Detalles

- Iluminación: "golden hour", "dramatic lighting", "soft light"
- Colores: "vibrant colors", "pastel tones", "neon"
- Calidad: "detailed", "8k", "high quality", "masterpiece"

### 4. Usa Negative Prompts

```python
negative_prompt = "blurry, ugly, bad anatomy, bad proportions, watermark, text, signature, low quality, deformed"
```

---

## Integración con THAU (Modelo de Lenguaje)

### Futuro: THAU decide cuándo generar imágenes

```python
# El modelo THAU aprenderá a:
# 1. Detectar cuando necesita generar una imagen
# 2. Formular el prompt optimizado
# 3. Evaluar la calidad de la imagen generada
# 4. Mejorar el prompt si es necesario

# Ejemplo futuro:
user: "Explícame qué es recursión"
thau: "La recursión es cuando una función se llama a sí misma.
       Déjame mostrarte una visualización..."
       [genera imagen de espejos infinitos]
       "Como ves en esta imagen, es como espejos reflejándose entre sí."
```

---

## Comandos Rápidos

```bash
# Ejecutar demo completo
python demo_image_generation.py

# Generar una imagen rápida (Python)
python -c "
from capabilities.vision.image_generator import generate_image_quick
result = generate_image_quick('a cute cat')
print(result['path'])
"

# Iniciar API con generación de imágenes
python api/main.py

# Ver imágenes generadas
open data/generated_images/

# Ver estadísticas
curl http://localhost:8000/vision/stats

# Ver ejemplos
curl http://localhost:8000/vision/examples
```

---

## Próximos Pasos

1. **Entrenar THAU** para decidir cuándo generar imágenes
2. **Agregar más estilos** (anime, sketch, etc.)
3. **Image-to-Image**: Modificar imágenes existentes
4. **Inpainting**: Editar partes específicas
5. **ControlNet**: Mayor control sobre la composición

---

**¡THAU ahora puede generar y mostrar imágenes! 🎨🤖**

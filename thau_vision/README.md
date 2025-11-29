# THAU-Vision

> **Sistema de Vision-Language Model para THAU**

Un sistema completo para agregar capacidades de visión a THAU, permitiendo:
- Entender y describir imágenes
- Responder preguntas sobre contenido visual
- Aprender de imágenes etiquetadas
- Procesar cámara en tiempo real
- OCR y extracción de texto
- Detección y reconocimiento de objetos

## Arquitectura

```
┌─────────────────────────────────────────────────────────────────┐
│                        THAU-Vision                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   SigLIP     │    │  Projection  │    │   TinyLlama  │      │
│  │   Vision     │ -> │    MLP       │ -> │     LLM      │      │
│  │   Encoder    │    │              │    │              │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│       768d              768d -> 2048d         2048d             │
│                                                                  │
│  Image Input ──────────────────────────────> Text Output        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Instalación

```bash
# Dependencias base (ya instaladas si tienes THAU)
pip install torch transformers pillow

# Para visión
pip install timm

# Para cámara
pip install opencv-python

# Para API
pip install fastapi uvicorn
```

## Uso Rápido

### Captioning de Imágenes

```python
from thau_vision import THAUVisionModel

# Cargar modelo
model = THAUVisionModel(config_name="thau-vision-tiny")

# Generar caption
caption = model.caption("mi_imagen.jpg", "Describe esta imagen:")
print(caption)
```

### Preguntas sobre Imágenes (VQA)

```python
from thau_vision import THAUVisionModel

model = THAUVisionModel()

# Preguntar sobre imagen
respuesta = model.answer("foto.jpg", "¿Qué color es el objeto principal?")
print(respuesta)
```

### Procesamiento de Cámara en Tiempo Real

```python
from thau_vision.inference import CameraProcessor

# Crear procesador
processor = CameraProcessor(camera_id=0)

# Iniciar
processor.start()

# Hacer preguntas sobre lo que ve la cámara
respuesta = processor.ask("¿Qué ves en este momento?")
print(respuesta)

# Tomar snapshot
processor.snapshot("captura.jpg")

# Detener
processor.stop()
```

### API REST

```bash
# Iniciar servidor
python -m thau_vision.api.server

# Usar endpoints
curl -X POST http://localhost:8080/caption \
  -H "Content-Type: application/json" \
  -d '{"image_base64": "...", "style": "detailed"}'
```

## Entrenamiento

### Entrenar con Datos Demo

```bash
python thau_vision/train_thau_vision.py --demo
```

### Entrenar con Datos Propios

```bash
# Con archivo JSONL
python thau_vision/train_thau_vision.py \
  --data data/mis_imagenes.jsonl \
  --epochs 3 \
  --batch_size 2

# Con directorio de imágenes
python thau_vision/train_thau_vision.py \
  --images_dir ./imagenes \
  --data anotaciones.json \
  --epochs 5
```

### Formato de Datos

```json
// Captioning
{"image_path": "foto1.jpg", "caption": "Un gato naranja durmiendo"}

// VQA
{"image_path": "foto2.jpg", "question": "¿Qué animal es?", "answer": "Es un perro labrador"}

// Etiquetas
{"image_path": "foto3.jpg", "labels": ["manzana", "mesa", "frutero"]}
```

## Estructura del Módulo

```
thau_vision/
├── __init__.py              # Package init
├── README.md                # Este archivo
├── train_thau_vision.py     # Script de entrenamiento
│
├── models/                  # Modelos
│   ├── vision_encoder.py    # SigLIP/CLIP encoder
│   ├── projection.py        # MLP de proyección
│   └── thau_vlm.py         # Modelo VLM completo
│
├── training/                # Entrenamiento
│   ├── dataset.py          # Dataset y DataCollator
│   └── train_vision.py     # VisionTrainer
│
├── inference/              # Inferencia
│   ├── image_qa.py         # ImageQA helper
│   └── camera.py           # Procesador de cámara
│
├── api/                    # API REST
│   └── server.py          # FastAPI server
│
└── utils/                  # Utilidades
    └── image_utils.py     # Funciones de imagen
```

## Configuraciones Disponibles

| Config | Vision Encoder | Projection | Uso Recomendado |
|--------|---------------|------------|-----------------|
| thau-vision-tiny | SigLIP-Base (86M) | MLP | Desarrollo, pruebas |
| thau-vision-small | SigLIP-Large (307M) | MLP Deep | Balance calidad/velocidad |
| thau-vision-pro | SigLIP-SO400M (400M) | Resampler | Mejor calidad |

## API Endpoints

| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/caption` | POST | Generar caption de imagen |
| `/answer` | POST | Responder pregunta sobre imagen |
| `/identify` | POST | Identificar objetos |
| `/describe` | POST | Descripción detallada de escena |
| `/compare` | POST | Comparar dos imágenes |
| `/extract_text` | POST | OCR - extraer texto |
| `/classify` | POST | Clasificar en categorías |
| `/learn` | POST | Aprender nuevo objeto |

## Ejemplos de Uso Avanzado

### Enseñar a Reconocer Objetos

```python
from thau_vision.inference import ImageQA

qa = ImageQA(model_path="checkpoints/thau_vision")

# Enseñar nuevo objeto
features = qa.learn_object(
    image="mi_mascota.jpg",
    object_name="Firulais",
    description="Mi perro labrador dorado"
)
print(features)
# {'name': 'Firulais', 'visual_description': '...', 'colors': '...', ...}
```

### Comparar Imágenes

```python
qa = ImageQA(model_path="checkpoints/thau_vision")

comparacion = qa.compare_images(
    "foto_antes.jpg",
    "foto_despues.jpg",
    aspect="general"
)
print(comparacion)
```

### Sesión Interactiva de Cámara

```python
from thau_vision.inference import run_interactive_camera

# Iniciar sesión interactiva CLI
run_interactive_camera(camera_id=0)
```

## Limitaciones

- **Modelo base pequeño**: TinyLlama tiene capacidad limitada para razonamiento complejo
- **Alucinaciones**: Puede inventar detalles no presentes en las imágenes
- **Idioma**: Mejor rendimiento en español, soporte básico en inglés
- **Velocidad**: SigLIP-SO400M es más lento que las versiones más pequeñas

## Requisitos de Hardware

| Config | RAM Mínima | GPU | Notas |
|--------|------------|-----|-------|
| thau-vision-tiny | 8GB | Opcional | Funciona en CPU |
| thau-vision-small | 12GB | Recomendada | MPS/CUDA mejora velocidad |
| thau-vision-pro | 16GB | Recomendada | Mejor con GPU |

## Próximos Pasos

- [ ] Soporte para múltiples imágenes en una conversación
- [ ] Segmentación de objetos
- [ ] Tracking de objetos en video
- [ ] Integración con THAU Agent Framework
- [ ] Fine-tuning con datos específicos del usuario

---

*THAU-Vision - Construido con curiosidad y mucha ayuda de Claude*

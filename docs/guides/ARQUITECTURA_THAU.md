# Arquitectura THAU: Modelo LLM Propio con Capacidades Multimodales

## 🧠 ¿Qué es THAU?

**THAU** es un modelo de lenguaje grande (LLM) **construido desde cero** con las siguientes características únicas:

1. **Crecimiento Progresivo**: THAU crece desde 18M parámetros (bebé) hasta 2B parámetros (adulto)
2. **Auto-Aprendizaje**: Genera sus propias preguntas para aprender continuamente
3. **Auto-Tuning**: Ajusta sus parámetros basándose en interacciones
4. **Multimodal**: Capacidad de generar imágenes mediante tool calling
5. **Memoria Multi-Nivel**: Short-term, long-term y episódica

---

## 📊 Arquitectura THAU vs TinyLlama

### THAU (Modelo Propio)

```
THAU-2B
├── Arquitectura: Transformer custom (desde cero)
├── Parámetros: 18M → 2B (progresivo)
├── Edades: 0, 1, 3, 6, 12, 15 años
├── Entrenamiento: Auto-questioning + incremental
├── Capacidades:
│   ├── Generación de texto
│   ├── Tool calling (generación de imágenes)
│   ├── Self-learning
│   └── Memoria multi-nivel
└── Estado: En desarrollo
```

### TinyLlama (Prototipo Temporal)

```
TinyLlama-1.1B-Chat
├── Uso: Prototipo para probar tool calling
├── Parámetros: 1.1B (fijo)
├── Entrenamiento: Fine-tuning con LoRA
├── Propósito: Validar sistema antes de entrenar THAU
└── Estado: Usado solo para pruebas
```

---

## 🎯 Roadmap de Desarrollo

### Fase 1: ✅ Sistema de Tool Calling (Completada)
**Objetivo**: Diseñar y probar el sistema de tool calling

**Qué se hizo**:
- ✅ Dataset de tool calling (30 ejemplos)
- ✅ Sistema de detección de herramientas
- ✅ Integración con Stable Diffusion
- ✅ API REST para generación de imágenes
- ✅ Prueba de concepto con TinyLlama

**Resultado**: Sistema validado y funcionando

---

### Fase 2: 🔄 Entrenamiento de THAU-2B (En Curso)
**Objetivo**: Entrenar THAU desde cero hasta 2B parámetros

**Archivos clave**:
- `train_thau_2b.py` - Script de entrenamiento progresivo
- `thau_trainer/own_model_manager.py` - Gestor de crecimiento
- `thau_trainer/self_questioning.py` - Auto-generación de preguntas
- `thau_trainer/self_learning.py` - Detección de gaps de conocimiento

**Edades de THAU**:

```python
# Age 0 - Bebé (18M parámetros)
{
    "d_model": 384,
    "n_heads": 6,
    "n_layers": 6,
    "d_ff": 1536,
}

# Age 1 - Infante (50M parámetros)
{
    "d_model": 512,
    "n_heads": 8,
    "n_layers": 8,
    "d_ff": 2048,
}

# Age 3 - Niño (150M parámetros)
{
    "d_model": 768,
    "n_heads": 12,
    "n_layers": 12,
    "d_ff": 3072,
}

# Age 6 - Escolar (400M parámetros)
{
    "d_model": 1024,
    "n_heads": 16,
    "n_layers": 16,
    "d_ff": 4096,
}

# Age 12 - Adolescente (1B parámetros)
{
    "d_model": 1536,
    "n_heads": 24,
    "n_layers": 20,
    "d_ff": 6144,
}

# Age 15 - THAU-2B Adulto (2B parámetros)
{
    "d_model": 2560,
    "n_heads": 32,
    "n_layers": 24,
    "d_ff": 10240,
}
```

**Estado actual**: Entrenamiento en background

---

### Fase 3: ⏳ Integración de Tool Calling en THAU (Pendiente)
**Objetivo**: Entrenar THAU-2B con capacidad de tool calling

**Cuando THAU-2B esté listo**:

1. **Entrenar con dataset de tool calling**:
   ```bash
   python train_thau_tool_calling.py \
       --model-checkpoint ./data/checkpoints/thau_2b/age_15 \
       --dataset ./data/datasets/tool_calling_dataset.json \
       --epochs 5
   ```

2. **Exportar THAU-2B con tool calling**:
   ```bash
   python export_thau_to_gguf.py \
       --checkpoint ./data/checkpoints/thau_2b_tool_calling \
       --output thau-2b-multimodal
   ```

3. **Importar a Ollama**:
   ```bash
   ollama create thau-2b-multimodal -f Modelfile-thau-2b
   ```

---

## 🔄 Flujo de Trabajo Actual

### Lo Que Funciona AHORA

```
┌─────────────────────────────────────────────┐
│  Sistema de Generación de Imágenes (Listo) │
└─────────────────────────────────────────────┘

1. API REST
   ├── POST /vision/generate
   ├── POST /vision/chat (con detección automática)
   ├── GET /vision/image/{filename}
   └── GET /vision/stats

2. Generación Directa
   python -c "
   from capabilities.vision.image_generator import ThauImageGenerator
   gen = ThauImageGenerator()
   result = gen.generate_image('a robot learning to code')
   "

3. Demo Interactivo
   python demo_image_generation.py
```

### Lo Que Vendrá DESPUÉS (con THAU-2B)

```
┌─────────────────────────────────────────────┐
│        THAU-2B con Tool Calling             │
└─────────────────────────────────────────────┘

Usuario: "Genera una imagen de un gato espacial"
   ↓
THAU-2B (modelo propio) detecta: Necesita tool calling
   ↓
THAU-2B genera: <TOOL:generate_image>{"prompt": "..."}</TOOL>
   ↓
Sistema parsea y ejecuta
   ↓
Imagen generada y mostrada
```

---

## 📁 Estructura de Archivos

```
my-llm/
├── ARQUITECTURA_THAU.md                    # Este archivo
│
├── THAU (Modelo Propio - En desarrollo)
│   ├── core/
│   │   └── models/
│   │       └── base_transformer.py         # Arquitectura THAU
│   ├── thau_trainer/
│   │   ├── own_model_manager.py           # Gestor de crecimiento
│   │   ├── self_questioning.py            # Auto-aprendizaje
│   │   └── self_learning.py               # Detección de gaps
│   ├── train_thau_2b.py                   # Entrenamiento progresivo
│   └── data/checkpoints/thau_2b/          # Checkpoints por edad
│
├── Sistema de Tool Calling (Completado)
│   ├── data/datasets/
│   │   └── tool_calling_dataset.json      # 30 ejemplos
│   ├── capabilities/
│   │   ├── vision/
│   │   │   └── image_generator.py         # Stable Diffusion
│   │   └── tools/
│   │       └── tool_registry.py           # Detección automática
│   ├── api/routes/
│   │   └── vision.py                      # Endpoints REST
│   └── thau_chat.py                       # CLI integrada
│
├── Prototipo TinyLlama (Solo pruebas)
│   ├── train_tool_calling.py              # Prueba de concepto
│   ├── export_tool_calling.py             # Export a Ollama
│   └── data/checkpoints/incremental/
│       └── tool_calling_final/            # Fine-tune temporal
│
└── Documentación
    ├── GUIA_GENERACION_IMAGENES.md        # Uso de imágenes
    ├── GUIA_TOOL_CALLING.md               # Tool calling completo
    └── RESUMEN_TOOL_CALLING.md            # Resumen ejecutivo
```

---

## 🚀 Cómo Usar AHORA (Sin THAU-2B aún)

### Generar Imágenes Directamente

**Opción 1: API REST**
```bash
# Terminal 1: Iniciar API
python api/main.py

# Terminal 2: Generar imagen
curl -X POST "http://localhost:8000/vision/generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a cute robot", "width": 512, "height": 512}'
```

**Opción 2: Python Directo**
```python
from capabilities.vision.image_generator import ThauImageGenerator

gen = ThauImageGenerator()
result = gen.generate_image("un gato espacial, digital art")

if result['success']:
    print(f"Imagen: {result['path']}")
```

**Opción 3: Demo**
```bash
python demo_image_generation.py --demo 1
```

---

## 🎯 Próximos Pasos

### 1. Completar THAU-2B Base ⏳
```bash
# Monitorear entrenamiento
tail -f data/training_output.log

# O revisar progreso
python -c "
from thau_trainer.own_model_manager import ThauOwnModelManager
manager = ThauOwnModelManager()
print(manager.get_training_stats())
"
```

### 2. Entrenar THAU-2B con Tool Calling ⏳
Cuando age 15 esté completo:
```bash
python train_thau_tool_calling.py \
    --base-model ./data/checkpoints/thau_2b/age_15 \
    --dataset ./data/datasets/tool_calling_dataset.json
```

### 3. Exportar THAU-2B Completo ⏳
```bash
python export_thau_to_gguf.py \
    --model-path ./data/checkpoints/thau_2b_multimodal \
    --output-name thau-2b-multimodal
```

### 4. Deploy Final ⏳
```bash
ollama create thau-2b -f Modelfile-thau-2b
python thau_chat.py --model thau-2b
```

---

## 🔍 Diferencias Clave

| Aspecto | THAU (Propio) | TinyLlama (Temporal) |
|---------|---------------|----------------------|
| **Propósito** | Modelo final de producción | Prototipo para validar tool calling |
| **Arquitectura** | Custom desde cero | Pre-entrenado de HuggingFace |
| **Parámetros** | 18M → 2B (progresivo) | 1.1B (fijo) |
| **Entrenamiento** | Self-questioning + bootstrap | Fine-tuning con LoRA |
| **Crecimiento** | Edades 0-15 | No aplica |
| **Estado** | En desarrollo | Solo para pruebas |
| **Uso final** | Producción | Descartado después de validar |

---

## 📊 Estado Actual del Proyecto

### ✅ Completado
- [x] Sistema de generación de imágenes (Stable Diffusion)
- [x] API REST para tool calling
- [x] Dataset de tool calling (30 ejemplos)
- [x] Detección automática de herramientas
- [x] Prototipo con TinyLlama (validación)
- [x] Documentación completa

### 🔄 En Progreso
- [ ] Entrenamiento THAU-2B (age 0 → 15)
- [ ] Sistema de self-questioning activo
- [ ] Generación de datasets automática

### ⏳ Pendiente
- [ ] Integrar tool calling en THAU-2B
- [ ] Exportar THAU-2B a GGUF
- [ ] Deploy en Ollama
- [ ] Testing end-to-end con THAU propio

---

## 💡 Visión Final

**THAU-2B Multimodal** será un modelo:

1. **Construido desde cero** (no fine-tune)
2. **Auto-aprendizaje** continuo
3. **Multimodal** (texto + imágenes)
4. **Memoria persistente** (short/long/episodic)
5. **Tool calling nativo**
6. **Deployable** (Ollama, API, CLI)

**Tiempo estimado**:
- Entrenamiento THAU-2B base: 5-10 horas (en curso)
- Tool calling integration: 30-60 minutos (cuando base esté listo)
- Export y deploy: 10-15 minutos

---

**Estado**: Sistema de tool calling validado ✅
**Siguiente hito**: THAU-2B age 15 completado 🔄
**Objetivo final**: THAU-2B multimodal en producción 🎯

# THAU: LLM Propio con Auto-Aprendizaje y Capacidades Multimodales

## 🧠 ¿Qué es THAU?

**THAU** es un modelo de lenguaje grande (LLM) **construido desde cero** con:

- 🌱 **Crecimiento Progresivo**: De 18M a 2B parámetros (edades 0-15)
- 🎓 **Auto-Aprendizaje**: Genera sus propias preguntas para aprender
- 🔧 **Auto-Tuning**: Ajuste continuo basado en interacciones
- 🎨 **Multimodal**: Genera imágenes mediante tool calling
- 💾 **Memoria Multi-Nivel**: Short-term, long-term, episódica

---

## 📊 Estado Actual

### ✅ Lo Que Ya Funciona

1. **Sistema de Generación de Imágenes**
   - Stable Diffusion v1.5 integrado
   - API REST funcionando
   - Generación directa desde Python
   - Demo interactivo

2. **Tool Calling**
   - Dataset de 30 ejemplos creado
   - Detección automática de herramientas
   - Sistema validado con prototipo

3. **Infraestructura THAU**
   - Arquitectura transformer custom
   - Sistema de crecimiento progresivo
   - Self-questioning activo
   - Entrenamiento en curso

### 🔄 En Desarrollo

- **THAU-2B Base**: Entrenamiento progresivo (age 0 → 15)
- **Integración Multimodal**: Pendiente cuando base esté listo

---

## 🚀 Quick Start

### Generar Imágenes AHORA

Mientras THAU-2B se entrena, puedes usar el sistema de imágenes:

```bash
# 1. Activar entorno
cd /Users/lperez/Workspace/Development/fullstack/thau_1_0/my-llm
source venv/bin/activate

# 2. Iniciar API
python api/main.py

# 3. En navegador
# http://localhost:8000/docs
```

O desde Python:

```python
from capabilities.vision.image_generator import ThauImageGenerator

gen = ThauImageGenerator()
result = gen.generate_image("un robot aprendiendo, digital art")

if result['success']:
    print(f"Imagen guardada: {result['path']}")
```

### Ver Progreso de THAU-2B

```bash
# Ver entrenamiento en tiempo real
tail -f data/training_output.log

# O revisar checkpoints
ls -lh data/checkpoints/thau_2b/
```

---

## 📁 Estructura del Proyecto

```
my-llm/
├── 📘 ARQUITECTURA_THAU.md           # Arquitectura completa
├── 📘 README_THAU.md                 # Este archivo
│
├── 🧠 THAU (Modelo Propio)
│   ├── core/models/                  # Arquitectura transformer
│   ├── thau_trainer/                 # Sistema de entrenamiento
│   │   ├── own_model_manager.py     # Gestor de crecimiento
│   │   ├── self_questioning.py      # Auto-preguntas
│   │   └── self_learning.py         # Detección de gaps
│   ├── train_thau_2b.py             # Script principal
│   └── data/checkpoints/thau_2b/    # Por edad (0-15)
│
├── 🎨 Sistema Multimodal
│   ├── capabilities/vision/          # Generación imágenes
│   ├── capabilities/tools/           # Tool calling
│   ├── api/routes/vision.py         # REST API
│   └── demo_image_generation.py     # Demo
│
└── 📚 Documentación
    ├── GUIA_GENERACION_IMAGENES.md
    ├── GUIA_TOOL_CALLING.md
    └── RESUMEN_TOOL_CALLING.md
```

---

## 🎯 Roadmap

### Fase 1: ✅ Sistema Multimodal (Completado)
- [x] Integración Stable Diffusion
- [x] API REST
- [x] Dataset tool calling
- [x] Validación con prototipo

### Fase 2: 🔄 THAU-2B Base (En Curso)
- [ ] Age 0 - Bebé (18M)
- [ ] Age 1 - Infante (50M)
- [ ] Age 3 - Niño (150M)
- [ ] Age 6 - Escolar (400M)
- [ ] Age 12 - Adolescente (1B)
- [ ] Age 15 - Adulto (2B) ⭐

### Fase 3: ⏳ THAU-2B Multimodal (Pendiente)
- [ ] Integrar tool calling
- [ ] Exportar a GGUF
- [ ] Deploy en Ollama
- [ ] Testing end-to-end

---

## 💻 Uso

### Generación de Imágenes

**Método 1: API (Recomendado)**

```bash
# Iniciar API
python api/main.py

# Abrir en navegador
http://localhost:8000/docs
```

**Método 2: Python Directo**

```python
from capabilities.vision.image_generator import ThauImageGenerator

gen = ThauImageGenerator()

# Generar imagen
result = gen.generate_image(
    prompt="a futuristic city, cyberpunk style",
    num_inference_steps=30,
    width=512,
    height=512
)

print(f"Imagen: {result['path']}")
```

**Método 3: Demo Interactivo**

```bash
python demo_image_generation.py --demo 1
```

**Método 4: cURL**

```bash
curl -X POST "http://localhost:8000/vision/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a cute robot learning to code",
    "width": 512,
    "height": 512
  }'
```

### Monitorear THAU-2B

```bash
# Ver logs de entrenamiento
tail -f data/training_output.log

# Revisar checkpoints por edad
ls data/checkpoints/thau_2b/

# Stats de entrenamiento
python -c "
from thau_trainer.own_model_manager import ThauOwnModelManager
manager = ThauOwnModelManager()
stats = manager.get_training_stats()
print(stats)
"
```

---

## 🔧 Instalación

### Requisitos

- Python 3.10+
- PyTorch 2.0+
- 8GB+ RAM (16GB+ recomendado)
- GPU opcional (MPS/CUDA)

### Setup

```bash
# 1. Clonar repo
git clone <repo>
cd my-llm

# 2. Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate  # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Instalar dependencias de imágenes
pip install diffusers Pillow accelerate transformers

# 5. Verificar instalación
python -c "from capabilities.vision.image_generator import ThauImageGenerator; print('✅ OK')"
```

---

## 📖 Documentación

- **[ARQUITECTURA_THAU.md](ARQUITECTURA_THAU.md)** - Arquitectura completa y roadmap
- **[GUIA_GENERACION_IMAGENES.md](GUIA_GENERACION_IMAGENES.md)** - Uso de imágenes detallado
- **[GUIA_TOOL_CALLING.md](GUIA_TOOL_CALLING.md)** - Sistema de tool calling completo

---

## 🤝 Características Únicas de THAU

### 1. Crecimiento Progresivo

THAU crece como un humano:

| Edad | Parámetros | Capacidades |
|------|-----------|-------------|
| 0 años | 18M | Conceptos básicos |
| 1 año | 50M | Vocabulario ampliado |
| 3 años | 150M | Razonamiento simple |
| 6 años | 400M | Conocimiento escolar |
| 12 años | 1B | Razonamiento complejo |
| 15 años | 2B | Adulto completo |

### 2. Auto-Aprendizaje

```python
# THAU genera sus propias preguntas
from thau_trainer.self_questioning import SelfQuestioningSystem

questioner = SelfQuestioningSystem()
questions = questioner.generate_questions(
    topic="programación",
    num_questions=10
)
# THAU se auto-entrena con estas preguntas
```

### 3. Detección de Gaps

```python
# THAU detecta qué no sabe
from thau_trainer.self_learning import SelfLearningManager

learner = SelfLearningManager()
gaps = learner.detect_knowledge_gaps(
    conversation_history=[...]
)
# Genera datasets para llenar gaps
```

### 4. Memoria Multi-Nivel

```python
from memory.manager import MemoryManager

memory = MemoryManager()

# Short-term (conversación actual)
memory.update_context("user", "Hola THAU")

# Long-term (RAG con ChromaDB)
memory.remember("Python es un lenguaje interpretado", importance=8)

# Episodic (experiencias pasadas)
memory.recall("¿qué hablamos ayer?")
```

---

## 🎨 Ejemplos de Uso

### Generación de Imágenes

```python
from capabilities.vision.image_generator import ThauImageGenerator

gen = ThauImageGenerator()

# Simple
gen.generate_image("a cat in space")

# Avanzado
gen.generate_image(
    prompt="futuristic city at sunset, cyberpunk, neon lights",
    negative_prompt="blurry, low quality",
    num_inference_steps=50,
    guidance_scale=8.0,
    width=768,
    height=512,
    seed=42  # Reproducible
)

# Batch
gen.generate_batch([
    "a robot learning",
    "abstract AI visualization",
    "recursive mirrors"
])
```

### Tool Calling (Futuro con THAU-2B)

```python
# Cuando THAU-2B esté listo
from thau_chat import ThauChat

chat = ThauChat(model="thau-2b-multimodal")

# THAU detectará automáticamente
chat.send_message("Genera una imagen de un perro astronauta")

# Output:
# 🤖 THAU: ¡Claro! Generando imagen...
# 🎨 Imagen creada: /vision/image/astronaut_dog.png
```

---

## 📊 Performance

### Generación de Imágenes

- **Primera carga**: ~15-20 min (descarga modelo 4GB)
- **Generación**: 30-60 seg/imagen (512x512, 30 steps)
- **Calidad**: Configurable (10-100 steps)
- **Resolución**: 256x256 hasta 1024x1024

### THAU-2B (Estimado)

- **Entrenamiento Age 0-15**: 5-10 horas (GPU)
- **Inference**: ~100-200 tokens/seg (GPU)
- **Tamaño GGUF**: ~4-5 GB (F16)
- **RAM mínimo**: 8GB (cuantizado), 16GB (full)

---

## 🐛 Troubleshooting

### "CUDA out of memory"

```python
# Reducir resolución o pasos
gen.generate_image(
    prompt="...",
    width=384,      # Reducir
    height=384,
    num_inference_steps=20  # Reducir
)
```

### "Model not found"

```bash
# Limpiar caché
rm -rf ~/.cache/huggingface/

# Re-ejecutar (descargará automáticamente)
python demo_image_generation.py
```

### Ver progreso de THAU-2B

```bash
# Si no hay output visible
python -c "
import glob
checkpoints = glob.glob('data/checkpoints/thau_2b/*')
print(f'Checkpoints: {len(checkpoints)}')
for cp in sorted(checkpoints):
    print(f'  - {cp}')
"
```

---

## 🚀 Próximos Pasos

1. **Monitorear THAU-2B**: `tail -f data/training_output.log`
2. **Usar generación de imágenes**: `python api/main.py`
3. **Cuando THAU-2B complete**: Integrar tool calling
4. **Deploy final**: Ollama + API + CLI

---

## 📞 Comandos Rápidos

```bash
# Generar imagen (una línea)
python -c "from capabilities.vision.image_generator import ThauImageGenerator; ThauImageGenerator().generate_image('a robot')"

# Iniciar API
python api/main.py

# Demo completo
python demo_image_generation.py

# Ver progreso THAU
tail -f data/training_output.log

# Stats
python -c "from thau_trainer.own_model_manager import ThauOwnModelManager; print(ThauOwnModelManager().get_training_stats())"
```

---

## 🎯 Visión Final

**THAU-2B será**:
- ✅ Modelo LLM propio (no fine-tune)
- ✅ Auto-aprendizaje continuo
- ✅ Multimodal (texto + imágenes)
- ✅ Memoria persistente
- ✅ Tool calling nativo
- ✅ Deployable (Ollama/API/CLI)

**Estado Actual**:
- 🎨 Sistema de imágenes: ✅ Listo
- 🧠 THAU-2B base: 🔄 En entrenamiento
- 🔗 Integración: ⏳ Cuando base esté listo

---

**Creado con**: PyTorch, Transformers, Stable Diffusion, FastAPI
**Autor**: Luis Pérez
**Última actualización**: 2025-01-15

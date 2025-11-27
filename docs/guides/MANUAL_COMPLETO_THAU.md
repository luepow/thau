# 🤖 THAU - Manual Completo del Sistema

**Trainable Helpful AI Unit** - Sistema de Entrenamiento Autónomo para LLMs

---

## 📑 Tabla de Contenidos

1. [Introducción](#introducción)
2. [Características Principales](#características-principales)
3. [Arquitectura del Sistema](#arquitectura-del-sistema)
4. [Instalación](#instalación)
5. [Inicio Rápido](#inicio-rápido)
6. [Desarrollo Cognitivo](#desarrollo-cognitivo)
7. [Auto-Aprendizaje](#auto-aprendizaje)
8. [Memoria Vectorizada](#memoria-vectorizada)
9. [Aprendizaje Multilingüe](#aprendizaje-multilingüe)
10. [Protocolo MCP](#protocolo-mcp)
11. [API Endpoints](#api-endpoints)
12. [Casos de Uso](#casos-de-uso)
13. [Configuración Avanzada](#configuración-avanzada)
14. [Troubleshooting](#troubleshooting)
15. [Roadmap](#roadmap)

---

## Introducción

THAU es un sistema revolucionario que permite entrenar modelos LLM de forma autónoma, sin consumir tus tokens de API. El modelo aprende progresivamente como un humano, desde edad 0 (recién nacido) hasta 15+ años (adulto experto).

### ¿Qué hace diferente a THAU?

| Característica | THAU | Modelos Tradicionales |
|---|---|---|
| **Entrenamiento** | Autónomo, sin tokens | Requiere tokens/GPU costosos |
| **Desarrollo** | Progresivo por edades | Monolítico |
| **Memoria** | Vectorizada eficiente | Limitada al contexto |
| **Idiomas** | Multilingüe con fonética | Depende del dataset |
| **Auto-mejora** | Detecta y cubre brechas | Estático |
| **MCP** | Soporta herramientas | Limitado |

---

## Características Principales

### 1. 🧠 Desarrollo Cognitivo por Edades

THAU aprende progresivamente en 7 etapas:

- **Edad 0** (Recién Nacido): Palabras clave, respuestas simples
- **Edad 1-2** (Infante): Frases cortas, conceptos básicos
- **Edad 3-5** (Niño Pequeño): Explicaciones simples, causa-efecto
- **Edad 6-10** (Niño): Matemáticas básicas, lógica
- **Edad 11-12** (Pre-adolescente): Pensamiento abstracto
- **Edad 13-15** (Adolescente): Razonamiento complejo
- **Edad 15+** (Adulto): Expertise técnico, tool calling

### 2. 🔄 Auto-Generación de Datasets

THAU puede:
- Detectar brechas de conocimiento automáticamente
- Generar sus propios datasets de entrenamiento
- Cubrir áreas donde tiene respuestas inciertas
- Crear ejemplos apropiados para su edad cognitiva

### 3. 💾 Memoria Vectorizada Eficiente

- **FAISS** o numpy para búsqueda ultrarrápida
- **Sentence Transformers** para embeddings de calidad
- Compresión y gestión inteligente
- Recuperación semántica de interacciones previas

### 4. 🌍 Aprendizaje Multilingüe

THAU puede aprender múltiples idiomas:
- **Vocabulario**: Palabras con definiciones, ejemplos
- **Fonética**: Pronunciación IPA, división silábica
- **Gramática**: Reglas lingüísticas
- **Traducción**: Mapeo entre idiomas

Idiomas soportados: Español, Inglés, Francés, Alemán, Italiano, Portugués

### 5. 🔗 Protocolo MCP

Implementación completa del Model Context Protocol de Anthropic:
- **Herramientas**: web_search, execute_python, recall_memory, learn_word, generate_dataset
- **Recursos**: Acceso a conocimiento estructurado
- **Interoperabilidad**: Compatible con Claude y otros sistemas MCP

---

## Arquitectura del Sistema

```
my-llm/
├── thau_trainer/                 # Core del sistema
│   ├── cognitive_development.py  # Gestión de edades cognitivas
│   ├── self_learning.py          # Auto-generación de datasets
│   ├── vector_memory.py          # Memoria vectorizada
│   ├── language_learning.py      # Sistema multilingüe
│   ├── mcp_server.py             # Servidor MCP
│   └── integrated_trainer.py    # Coordinador principal
│
├── api/
│   └── thau_api_integrated.py   # API FastAPI completa
│
├── data/
│   ├── datasets/                # Datasets por edad
│   │   ├── age_0_newborn.jsonl
│   │   ├── age_1_infant.jsonl
│   │   ├── ...
│   │   └── auto_generated/      # Datasets auto-generados
│   ├── memory/                  # Índices vectoriales
│   ├── language/                # Vocabularios y fonética
│   └── logs/                    # Logs y progreso
│
└── start_thau.sh                # Script de inicio
```

---

## Instalación

### Requisitos Previos

1. **Python 3.10+**
   ```bash
   python3 --version
   ```

2. **Ollama** ([instalar](https://ollama.ai))
   ```bash
   curl https://ollama.ai/install.sh | sh
   ```

3. **Git**
   ```bash
   git --version
   ```

### Pasos de Instalación

```bash
# 1. Clonar repositorio
cd /path/to/your/projects
git clone <repo-url> thau_1_0
cd thau_1_0/my-llm

# 2. Crear entorno virtual
python3 -m venv venv
source venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Iniciar Ollama (en terminal separada)
ollama serve

# 5. Descargar modelo base
ollama pull qwen2.5-coder:1.5b-base

# 6. Verificar instalación
python thau_trainer/integrated_trainer.py
```

---

## Inicio Rápido

### Opción A: Usando el script

```bash
# Iniciar API server
./start_thau.sh

# O probar el sistema
./start_thau.sh test

# O probar MCP
./start_thau.sh mcp
```

### Opción B: Manual

```bash
# Activar entorno
source venv/bin/activate

# Iniciar API
python api/thau_api_integrated.py
```

### Primer Uso

```bash
# Verificar que está corriendo
curl http://localhost:8000/health

# Ver estado
curl http://localhost:8000/status

# Procesar primera interacción
curl -X POST http://localhost:8000/interact \
  -H "Content-Type: application/json" \
  -d '{
    "question": "¿Qué es Python?",
    "answer": "Python es un lenguaje de programación de alto nivel",
    "confidence": 0.9
  }'

# Ver documentación interactiva
# http://localhost:8000/docs
```

---

## Desarrollo Cognitivo

### Conceptos Clave

Cada edad tiene:
- **Capacidades**: Qué puede hacer
- **Dominios de aprendizaje**: Qué puede aprender
- **Complejidad de razonamiento**: Nivel 1-10
- **Longitud de contexto**: Tokens que puede manejar
- **Criterios de avance**: Requisitos para siguiente edad

### Ver Estado Actual

```bash
# API
curl http://localhost:8000/cognitive/status

# CLI
python -c "
from thau_trainer.cognitive_development import CognitiveDevelopmentManager
mgr = CognitiveDevelopmentManager()
import json
print(json.dumps(mgr.get_status(), indent=2))
"
```

### Forzar Avance de Edad

```bash
# Solo avanza si cumple criterios
curl -X POST http://localhost:8000/cognitive/advance
```

### Progreso de Ejemplo

```
Edad 0 → 100 ejemplos, 70% accuracy → Edad 1
Edad 1 → 200 ejemplos, 75% accuracy → Edad 3
Edad 3 → 500 ejemplos, 80% accuracy → Edad 6
...
```

---

## Auto-Aprendizaje

### Detección de Brechas

THAU detecta automáticamente cuando:
- Respuestas muy cortas (< 20 caracteres)
- Marcadores de incertidumbre ("no estoy seguro", "no sé")
- Confianza baja (< 0.6)

### Generación Automática

```bash
# Generar datasets para brechas
curl -X POST "http://localhost:8000/auto-improve?min_gaps=3"
```

### Proceso Completo

1. **Usuario interactúa** → THAU responde
2. **Sistema detecta brecha** → Registra tópico
3. **Auto-mejora ejecuta** → Genera 5-10 ejemplos
4. **Dataset creado** → Se importa a cola
5. **Entrenamiento** → THAU aprende

### Ver Brechas Detectadas

```bash
# Logs
cat data/logs/knowledge_gaps.jsonl | jq

# Stats
curl http://localhost:8000/stats/self-learning
```

---

## Memoria Vectorizada

### Características

- **Búsqueda semántica**: No necesita palabras exactas
- **Escalable**: Hasta 10,000+ vectores
- **Rápida**: Milisegundos por búsqueda
- **Eficiente**: Auto-limpieza

### Búsqueda

```bash
# API
curl -X POST http://localhost:8000/memory/recall \
  -H "Content-Type: application/json" \
  -d '{
    "query": "programación orientada a objetos",
    "k": 3
  }'

# Python
from thau_trainer.vector_memory import EfficientVectorMemory

memory = EfficientVectorMemory()
results = memory.search("machine learning", k=5)

for result in results:
    print(f"Score: {result['score']:.3f}")
    print(f"Text: {result['text']}")
```

### Estadísticas

```bash
curl http://localhost:8000/stats/memory
```

---

## Aprendizaje Multilingüe

### Añadir Idioma

```bash
curl -X POST "http://localhost:8000/language/add?language_code=en"
```

### Aprender Palabra

```bash
curl -X POST http://localhost:8000/language/learn-word \
  -H "Content-Type: application/json" \
  -d '{
    "word": "algorithm",
    "language": "en",
    "definition": "A step-by-step procedure to solve a problem",
    "pos": "noun",
    "examples": ["This sorting algorithm is efficient"]
  }'
```

### Fonética Automática

Para español, THAU genera automáticamente:
- **IPA**: Notación fonética internacional
- **Sílabas**: División silábica
- **Acentuación**: Sílaba tónica

Ejemplo:
```
computadora → /komputadora/ → com-pu-ta-do-ra (tónica: "do")
```

### Ver Progreso

```bash
curl http://localhost:8000/language/progress/es
```

---

## Protocolo MCP

### Herramientas Disponibles

```bash
# Listar
curl http://localhost:8000/mcp/tools
```

**Herramientas:**

1. **web_search**: Búsqueda web
2. **execute_python**: Ejecutar código Python
3. **recall_memory**: Buscar en memoria
4. **learn_word**: Aprender vocabulario
5. **generate_dataset**: Crear datasets

### Llamar Herramienta

```bash
curl -X POST http://localhost:8000/mcp/call \
  -H "Content-Type: application/json" \
  -d '{
    "tool_name": "execute_python",
    "arguments": {
      "code": "print(\"Hello from THAU!\")\nresult = 2 ** 10\nprint(f\"2^10 = {result}\")"
    }
  }'
```

### Integración con Claude

THAU puede comunicarse con Claude Desktop u otros clientes MCP:

1. Configurar cliente MCP para apuntar a `http://localhost:8000/mcp`
2. Claude puede llamar herramientas de THAU
3. Bidireccional: THAU puede usar herramientas externas

---

## API Endpoints

### Core

- `GET /` - Info del servidor
- `GET /status` - Estado completo
- `GET /health` - Health check

### Interacciones

- `POST /interact` - Procesar interacción
- `POST /memory/recall` - Buscar en memoria
- `POST /train` - Entrenar ahora
- `POST /auto-improve` - Auto-mejorar

### Desarrollo Cognitivo

- `GET /cognitive/status` - Estado cognitivo
- `POST /cognitive/advance` - Avanzar edad

### Idiomas

- `POST /language/add` - Añadir idioma
- `POST /language/learn-word` - Aprender palabra
- `GET /language/progress/{lang}` - Ver progreso

### MCP

- `GET /mcp/tools` - Listar herramientas
- `POST /mcp/call` - Ejecutar herramienta
- `GET /mcp/resources` - Listar recursos

### Estadísticas

- `GET /stats/memory` - Stats de memoria
- `GET /stats/self-learning` - Stats de auto-aprendizaje
- `GET /stats/datasets` - Datasets generados

---

## Casos de Uso

### Caso 1: Asistente Personal que Aprende

```python
import requests

BASE_URL = "http://localhost:8000"

# 1. Enseñarle sobre tu proyecto
requests.post(f"{BASE_URL}/interact", json={
    "question": "¿Cómo funciona mi módulo de autenticación?",
    "answer": "Tu módulo usa OAuth 2.0 con refresh tokens...",
    "confidence": 0.95
})

# 2. THAU recuerda
response = requests.post(f"{BASE_URL}/memory/recall", json={
    "query": "autenticación",
    "k": 3
})

print(response.json())
```

### Caso 2: Aprendizaje Multilingüe

```python
# Añadir francés
requests.post(f"{BASE_URL}/language/add?language_code=fr")

# Aprender vocabulario técnico
words = [
    ("ordinateur", "fr", "computadora/computer"),
    ("programmation", "fr", "programación/programming"),
    ("algorithme", "fr", "algoritmo/algorithm")
]

for word, lang, definition in words:
    requests.post(f"{BASE_URL}/language/learn-word", json={
        "word": word,
        "language": lang,
        "definition": definition
    })

# Ver progreso
progress = requests.get(f"{BASE_URL}/language/progress/fr")
print(f"Palabras aprendidas: {progress.json()['vocabulary_stats']['total_words']}")
```

### Caso 3: Entrenamiento Continuo

```python
# Loop de entrenamiento automático
import time

while True:
    # Usuario interactúa
    interaction = get_user_interaction()  # Tu lógica

    # THAU aprende
    requests.post(f"{BASE_URL}/interact", json=interaction)

    # Cada 100 interacciones, auto-mejorar
    if interaction_count % 100 == 0:
        requests.post(f"{BASE_URL}/auto-improve?min_gaps=5")

    # Cada 500, entrenar
    if interaction_count % 500 == 0:
        requests.post(f"{BASE_URL}/train")

    time.sleep(1)
```

---

## Configuración Avanzada

### Ajustar Criterios de Edad

Edita `thau_trainer/cognitive_development.py`:

```python
# Ejemplo: Facilitar avance de edad 3
advancement_criteria={
    "min_examples": 300,  # Reducir de 500
    "min_accuracy": 0.75,  # Reducir de 0.80
    "can_explain_simple_concepts": True
}
```

### Cambiar Intervalo de Auto-Mejora

En `start_thau.sh` o al inicializar:

```python
thau_trainer.start_auto_improvement_loop(interval_hours=1)  # Cada 1 hora
```

### Memoria Vectorizada: FAISS vs Numpy

Por defecto, THAU usa FAISS si está disponible, sino usa numpy.

Para forzar numpy:

```python
from thau_trainer.vector_memory import EfficientVectorMemory

memory = EfficientVectorMemory(index_type="flat")
```

Para IVF (rápido):

```python
memory = EfficientVectorMemory(index_type="IVF")
```

Para HNSW (muy rápido):

```python
memory = EfficientVectorMemory(index_type="HNSW")
```

---

## Troubleshooting

### "THAU no avanza de edad"

**Solución:**
```bash
# Ver progreso
curl http://localhost:8000/cognitive/status

# Verificar criterios
# - ¿Tiene suficientes ejemplos?
# - ¿El accuracy es suficiente?

# Añadir más datos apropiados para la edad
```

### "Auto-mejora no genera datasets"

**Causas:**
- Menos de `min_gaps` brechas detectadas
- Ollama no está corriendo

**Solución:**
```bash
# Verificar Ollama
curl http://localhost:11434/api/version

# Ver brechas detectadas
curl http://localhost:8000/stats/self-learning

# Forzar con min_gaps=1
curl -X POST "http://localhost:8000/auto-improve?min_gaps=1"
```

### "Memoria vectorizada muy lenta"

**Solución:**
```bash
# Instalar FAISS
pip install faiss-cpu

# O limpiar memoria
curl http://localhost:8000/stats/memory
# Si > 10,000 vectores, se auto-limpia
```

### "Error al importar sentence-transformers"

**Solución:**
```bash
pip install sentence-transformers
```

THAU funcionará sin ella (usa embeddings simples), pero es mejor tenerla.

---

## Roadmap

### Versión 1.1 (Q1 2025)

- [ ] Integración con APIs de búsqueda reales
- [ ] Soporte para más formatos de datasets (CSV, JSON)
- [ ] Dashboard web para monitoreo
- [ ] Exportar modelo entrenado a GGUF

### Versión 1.2 (Q2 2025)

- [ ] Multimodalidad (imágenes, audio)
- [ ] Integración con más idiomas (árabe, chino, japonés)
- [ ] Sistema de recompensas (RLHF)
- [ ] Distributed training

### Versión 2.0 (Q3 2025)

- [ ] Auto-arquitectura (THAU elige su estructura)
- [ ] Meta-aprendizaje (aprende a aprender mejor)
- [ ] Federated learning
- [ ] Integración con Claude 4 API

---

## Contribuir

THAU es de código abierto. Pull requests son bienvenidos!

1. Fork el proyecto
2. Crea tu feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la branch (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

---

## Licencia

MIT License - Ve `LICENSE` para detalles

---

## Agradecimientos

- **Anthropic** - Por el protocolo MCP
- **Ollama** - Por hacer LLMs locales accesibles
- **Meta** - Por FAISS
- **HuggingFace** - Por transformers y datasets

---

## Soporte

- **Documentación**: `THAU_FINAL_GUIDE.md`
- **GitHub Issues**: [Reportar bug](https://github.com/your-repo/issues)
- **Email**: support@thau-ai.com

---

**¡THAU crece mientras tú desarrollas!** 🌱🤖

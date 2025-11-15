# 🚀 Quickstart - Arquitecto de Software AI

Modelo LLM especializado en arquitectura de software, patrones de diseño y mejores prácticas de programación.

## 📋 Requisitos Completados

✅ Python 3.12.3
✅ Ollama instalado
✅ Dataset de entrenamiento (50+ conceptos de arquitectura)
✅ Scripts de entrenamiento y despliegue
✅ Modelfile para Ollama

## 🎯 Uso Rápido

### 1. Entrenar el Modelo (Primera vez)

```bash
# Activar el entorno virtual
source venv/bin/activate

# Entrenar con el dataset de arquitectura de software
python scripts/train_architecture.py --epochs 3 --batch-size 2
```

**Parámetros disponibles:**
- `--epochs`: Número de épocas (default: 3)
- `--batch-size`: Tamaño del batch (default: 2, ajustar según RAM)
- `--learning-rate`: Learning rate (default: 2e-4)
- `--dataset`: Ruta al dataset (default: ./data/datasets/architecture_training.jsonl)

**Tiempo estimado**: ~15-30 minutos en Apple Silicon M1/M2

### 2. Probar el Modelo Entrenado

```bash
# Ejecutar pruebas rápidas
python scripts/test_model.py
```

Esto probará el modelo con preguntas sobre:
- Patrones de diseño (Repository, Factory, Observer, etc.)
- Arquitecturas (Clean Architecture, Microservicios, Event-Driven)
- Principios SOLID
- Bases de datos y optimización

### 3. Desplegar a Ollama

```bash
# Crear el modelo en Ollama
./scripts/deploy_to_ollama.sh
```

### 4. Usar con Ollama

```bash
# Modo interactivo
ollama run architecture-expert

# Pregunta única
ollama run architecture-expert "¿Qué es el patrón Repository?"

# Pregunta compleja
ollama run architecture-expert "Explica cuándo usar microservicios vs monolito y dame un ejemplo de arquitectura"
```

## 📚 Ejemplos de Preguntas

El modelo está entrenado para responder:

**Patrones de Diseño:**
- "Explica el patrón Factory"
- "¿Cuándo usar Observer vs Pub/Sub?"
- "Diferencias entre Strategy y Template Method"

**Arquitecturas:**
- "¿Qué es Clean Architecture?"
- "Explica Event-Driven Architecture"
- "¿Cómo implementar CQRS con Event Sourcing?"

**Mejores Prácticas:**
- "Explica los principios SOLID"
- "¿Qué es DRY y cuándo aplicarlo?"
- "Diferencias entre ACID y BASE"

**Sistemas Distribuidos:**
- "¿Qué es el patrón Saga?"
- "Explica CAP theorem"
- "¿Cómo implementar circuit breaker?"

**Bases de Datos:**
- "¿Cuándo usar sharding?"
- "Explica indexing strategies"
- "Normalización vs denormalización"

**APIs y Seguridad:**
- "REST vs GraphQL"
- "¿Cómo funciona OAuth 2.0?"
- "Explica JWT y sus trade-offs"

**DevOps:**
- "¿Qué es CI/CD?"
- "Contenedores vs VMs"
- "Explica blue-green deployment"

## 🔧 Personalización

### Agregar Más Datos de Entrenamiento

Edita `data/datasets/architecture_training.jsonl` y agrega líneas en formato:

```json
{"instruction": "Tu pregunta", "input": "", "output": "Respuesta detallada con ejemplos"}
```

Luego re-entrena:

```bash
python scripts/train_architecture.py --epochs 3
```

### Ajustar Parámetros de Generación

Edita el `Modelfile`:

```dockerfile
# Más creativo
PARAMETER temperature 0.9

# Más determinista
PARAMETER temperature 0.3

# Mayor contexto
PARAMETER num_ctx 8192
```

Luego recrea el modelo:

```bash
ollama create architecture-expert -f Modelfile

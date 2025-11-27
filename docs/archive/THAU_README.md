# 🤖 THAU - Sistema de Entrenamiento Autónomo

**THAU** (Trainable Helpful AI Unit) es un sistema completo de entrenamiento autónomo para modelos LLM. Se entrena automáticamente con nuevos datos sin consumir tus tokens, versionándose automáticamente.

## 🌟 Características Principales

### ✅ Entrenamiento Autónomo
- **Sin intervención manual**: THAU se entrena solo cuando hay datos nuevos
- **Sin consumir tokens**: El entrenamiento es local, no usa APIs externas
- **Programado**: Se ejecuta cada X horas automáticamente
- **Incremental**: Sólo entrena con datos nuevos, no desde cero

### 📦 Versionado Automático
- Nomenclatura estilo industria: `thau-1.5b-v1`, `thau-1.5b-v2`, etc.
- Tamaños disponibles: 1.5b, 3b, 7b, 13b parámetros
- Cada entrenamiento incrementa la versión
- Historial completo de versiones

### 🔧 Capacidades Avanzadas
- **Chain-of-Thought**: Razonamiento paso a paso
- **Tool Calling**: Puede ejecutar herramientas (web search, code execution, etc.)
- **Arquitectura de Software**: Especializado en patrones y mejores prácticas
- **Análisis de Código**: Seguridad, performance, mejores prácticas

### 🌐 API REST Completa
- Agregar ejemplos de entrenamiento via HTTP
- Monitorear estado del modelo
- Forzar entrenamientos manuales
- Estadísticas en tiempo real

## 🚀 Quick Start

### 1. Instalación

```bash
cd my-llm

# Activar entorno virtual
source venv/bin/activate

# Instalar dependencias adicionales
pip install schedule click
```

### 2. Inicializar THAU

```bash
# Inicializar con tamaño de modelo (1.5b, 3b, 7b, 13b)
python scripts/thau_cli.py init --size 1.5b
```

### 3. Iniciar el Servicio de Entrenamiento

**Opción A: Con CLI (recomendado para desarrollo)**
```bash
python scripts/thau_cli.py start
```

**Opción B: Con API (recomendado para producción)**
```bash
python api/thau_api.py
```

El servicio se iniciará en `http://localhost:8000`

### 4. Agregar Datos de Entrenamiento

**Con CLI:**
```bash
python scripts/thau_cli.py add \
  "¿Qué es el patrón Strategy?" \
  "El patrón Strategy define una familia de algoritmos, los encapsula y los hace intercambiables..."
```

**Con API (curl):**
```bash
curl -X POST "http://localhost:8000/training/add" \
  -H "Content-Type: application/json" \
  -d '{
    "instruction": "¿Qué es el patrón Strategy?",
    "output": "El patrón Strategy define una familia de algoritmos..."
  }'
```

**Con API (Python):**
```python
import requests

requests.post("http://localhost:8000/training/add", json={
    "instruction": "¿Qué es el patrón Strategy?",
    "output": "El patrón Strategy define una familia de algoritmos..."
})
```

**Importar archivo JSONL:**
```bash
python scripts/thau_cli.py import-data data/datasets/mi_dataset.jsonl
```

### 5. Monitorear

**Ver estado:**
```bash
python scripts/thau_cli.py status
```

**Ver en API:**
```bash
curl http://localhost:8000/status
curl http://localhost:8000/stats
```

## 📖 Uso Completo

### Comandos CLI

```bash
# Ver todos los comandos
python scripts/thau_cli.py --help

# Inicializar modelo
python scripts/thau_cli.py init --size 1.5b

# Iniciar servicio
python scripts/thau_cli.py start

# Ver estado
python scripts/thau_cli.py status

# Agregar ejemplo
python scripts/thau_cli.py add "pregunta" "respuesta"

# Importar datos
python scripts/thau_cli.py import-data archivo.jsonl

# Forzar entrenamiento inmediato
python scripts/thau_cli.py train

# Ver versión actual
python scripts/thau_cli.py version
```

### API Endpoints

```bash
# Status del servicio
GET http://localhost:8000/status

# Agregar un ejemplo
POST http://localhost:8000/training/add
{
  "instruction": "pregunta",
  "output": "respuesta",
  "metadata": {"source": "manual"}
}

# Agregar lote de ejemplos
POST http://localhost:8000/training/batch
{
  "examples": [
    {"instruction": "...", "output": "..."},
    {"instruction": "...", "output": "..."}
  ]
}

# Forzar entrenamiento
POST http://localhost:8000/training/force

# Ver estadísticas
GET http://localhost:8000/stats

# Ver ejemplos pendientes
GET http://localhost:8000/examples/pending

# Iniciar servicio
POST http://localhost:8000/service/start

# Detener servicio
POST http://localhost:8000/service/stop

# Documentación interactiva
http://localhost:8000/docs
```

## 🔧 Configuración

El archivo `thau_trainer/config.py` contiene toda la configuración:

```python
# Intervalo de entrenamiento automático
auto_train_interval_hours = 24  # Cada 24 horas

# Mínimo de ejemplos nuevos para entrenar
min_new_examples = 10

# Parámetros de entrenamiento
batch_size = 2
epochs_per_training = 3
learning_rate = 2e-4

# LoRA
lora_r = 32
lora_alpha = 64

# Contexto
max_context_length = 2048
```

## 📊 Flujo de Entrenamiento

```
1. Usuario agrega datos vía CLI o API
   ↓
2. Datos van a cola de entrenamiento
   ↓
3. Servicio verifica cada hora si hay suficientes datos
   ↓
4. Si hay ≥10 ejemplos nuevos:
   - Carga modelo actual (o crea uno nuevo)
   - Configura LoRA adapters
   - Entrena con nuevos datos
   - Guarda como nueva versión
   - Actualiza en Ollama
   - Marca ejemplos como entrenados
   ↓
5. Modelo actualizado listo para usar
```

## 🎯 Versiones del Modelo

### Tamaños Disponibles

- **thau-1.5b**: 1.5 billones de parámetros (base: qwen2.5-coder:1.5b)
  - Rápido, eficiente, ideal para desarrollo
  - ~2GB RAM
  - ~30 tokens/segundo en M1

- **thau-3b**: 3 billones de parámetros
  - Balance performance/recursos
  - ~4GB RAM

- **thau-7b**: 7 billones de parámetros
  - Alto rendimiento
  - ~8GB RAM

- **thau-13b**: 13 billones de parámetros
  - Máxima calidad
  - ~16GB RAM

### Versionado

Cada entrenamiento incrementa la versión:
- `thau-1.5b-v1`: Versión inicial
- `thau-1.5b-v2`: Después del primer entrenamiento
- `thau-1.5b-v3`: Después del segundo entrenamiento
- ...

Puedes usar versiones específicas en Ollama:
```bash
ollama run thau-1.5b-v2
ollama run thau-1.5b-v3
```

## 💡 Ejemplos de Uso

### 1. Entrenar con conocimientos de tu proyecto

```bash
# Crear archivo con ejemplos
cat > mi_proyecto.jsonl <<EOF
{"instruction": "¿Cómo funciona el módulo de autenticación?", "output": "El módulo usa OAuth 2.0 con JWT..."}
{"instruction": "¿Dónde está la lógica de pagos?", "output": "En src/services/payment_service.py..."}
EOF

# Importar
python scripts/thau_cli.py import-data mi_proyecto.jsonl

# El modelo se entrenará automáticamente
```

### 2. Entrenar con conversaciones

```python
# Script para capturar conversaciones y agregar a THAU
import requests

def add_conversation(question, answer):
    requests.post("http://localhost:8000/training/add", json={
        "instruction": question,
        "output": answer,
        "metadata": {"source": "conversation"}
    })

# Durante tu sesión de código
add_conversation(
    "¿Cómo optimizo esta query SQL?",
    "Puedes agregar un índice en la columna user_id..."
)
```

### 3. Entrenar desde logs

```bash
# Parsear logs y convertir a JSONL
python scripts/parse_logs.py logs/conversations.log > training_data.jsonl

# Importar
python scripts/thau_cli.py import-data training_data.jsonl
```

## 📈 Monitoreo y Métricas

El dashboard de Ollama muestra:
- Versión actual del modelo
- Ejemplos entrenados
- Últim entrenamiento
- Ejemplos pendientes
- Progreso de entrenamiento

```bash
# Ver métricas en tiempo real
watch -n 5 "curl -s http://localhost:8000/stats | jq"
```

## 🔍 Debugging

### Ver logs

```bash
# Logs del servicio
tail -f data/logs/thau_service.log

# Logs de entrenamiento
tail -f data/logs/training_*.log
```

### Verificar datos

```bash
# Ver ejemplos pendientes
python scripts/thau_cli.py status

# Ver ejemplos en cola
ls -la data/training_queue/

# Ver ejemplos ya entrenados
head data/logs/trained_examples.jsonl
```

## 🎓 Mejores Prácticas

### 1. Calidad sobre Cantidad
- Agrega ejemplos bien formados
- Instrucciones claras
- Respuestas completas y precisas

### 2. Diversidad
- Varía los tipos de preguntas
- Cubre diferentes dominios
- Incluye ejemplos complejos y simples

### 3. Versionado Estratégico
- Espera acumular ~50-100 ejemplos antes de forzar entrenamiento
- Prueba cada nueva versión antes de usar en producción
- Mantén backups de versiones estables

### 4. Monitoreo
- Revisa periódicamente los stats
- Verifica que el auto-training funciona
- Revisa la calidad de las respuestas en nuevas versiones

## 🚀 Producción

### Docker

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY . /app

RUN pip install -r requirements.txt
RUN pip install schedule click

# Iniciar servicio al arrancar
CMD ["python", "api/thau_api.py"]
```

### Systemd Service

```ini
[Unit]
Description=THAU Training Service
After=network.target

[Service]
Type=simple
User=thau
WorkingDirectory=/home/thau/my-llm
ExecStart=/home/thau/my-llm/venv/bin/python api/thau_api.py
Restart=always

[Install]
WantedBy=multi-user.target
```

## 📚 Recursos

- **API Docs**: http://localhost:8000/docs
- **Ollama Docs**: https://ollama.ai/docs
- **LoRA**: https://arxiv.org/abs/2106.09685
- **PEFT**: https://huggingface.co/docs/peft

## 🤝 Contribuir

1. Agrega nuevos datasets a `data/datasets/`
2. Mejora los prompts de sistema
3. Optimiza parámetros de entrenamiento
4. Crea nuevas herramientas para tool calling

## 📄 Licencia

MIT License

---

**¡THAU se entrena mientras duermes!** 🌙🤖

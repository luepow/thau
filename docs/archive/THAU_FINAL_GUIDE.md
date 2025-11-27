# 🤖 THAU - Guía Completa del Sistema

## 🎯 ¿Qué es THAU?

**THAU** (Trainable Helpful AI Unit) es un sistema revolucionario de entrenamiento autónomo para modelos LLM que combina:

1. **Entrenamiento Autónomo**: Se entrena solo, sin consumir tus tokens
2. **Versionado Automático**: thau-1.5b-v1, thau-1.5b-v2, etc.
3. **Desarrollo Cognitivo**: Aprende como un humano, desde edad 0 hasta 15+ años
4. **Tool Calling**: Puede ejecutar herramientas (búsqueda web, código, etc.)
5. **Chain-of-Thought**: Razonamiento paso a paso

---

## 🧠 Sistema de Desarrollo Cognitivo

THAU aprende progresivamente como un niño humano:

### Edad 0: Recién Nacido 👶
- **Capacidades**: Palabras clave, respuestas simples sí/no
- **Aprende**: Vocabulario básico, reconocimiento de entidades
- **Contexto**: 128 tokens
- **Avanza con**: 100 ejemplos, 70% accuracy

### Edad 1-2: Infante 🍼
- **Capacidades**: Frases de 2-3 palabras, instrucciones simples
- **Aprende**: Lenguaje básico, conceptos simples (colores, números)
- **Contexto**: 256 tokens
- **Avanza con**: 200 ejemplos, 75% accuracy

### Edad 3-5: Niño Pequeño 🧒
- **Capacidades**: Explicaciones simples, causa-efecto
- **Aprende**: Conceptos cotidianos, categorización
- **Contexto**: 512 tokens
- **Avanza con**: 500 ejemplos, 80% accuracy

### Edad 6-10: Niño 📚
- **Capacidades**: Matemáticas básicas, lógica, lectura comprensiva
- **Aprende**: Ciencias básicas, reglas y patrones
- **Contexto**: 1024 tokens
- **Avanza con**: 1000 ejemplos, 85% accuracy

### Edad 11-12: Pre-adolescente 🎓
- **Capacidades**: Pensamiento abstracto, múltiples perspectivas
- **Aprende**: Álgebra, programación básica, tecnología
- **Contexto**: 2048 tokens
- **Avanza con**: 2000 ejemplos, 88% accuracy

### Edad 13-15: Adolescente 🚀
- **Capacidades**: Pensamiento crítico, razonamiento complejo
- **Aprende**: Matemáticas avanzadas, análisis crítico
- **Contexto**: 3072 tokens
- **Avanza con**: 3000 ejemplos, 90% accuracy

### Edad 15+: Adulto 🎯
- **Capacidades**: Razonamiento experto, tool calling, chain-of-thought
- **Aprende**: Especialización técnica, arquitectura de software
- **Contexto**: 4096 tokens
- **Mantiene**: 5000+ ejemplos, 92% accuracy

---

## 🚀 Inicio Rápido

### 1. Activar entorno

```bash
source venv/bin/activate
```

### 2. Inicializar THAU (primera vez)

```bash
# Iniciar con modelo de 1.5b parámetros
python scripts/thau_cli.py init --size 1.5b
```

### 3. Iniciar el servicio

```bash
# Opción A: Servicio en background (recomendado)
python api/thau_api.py

# Opción B: CLI directo
python scripts/thau_cli.py start
```

El servicio estará en `http://localhost:8000`
La documentación en `http://localhost:8000/docs`

### 4. Agregar datos de entrenamiento

**Opción A: Con API**
```bash
curl -X POST "http://localhost:8000/training/add" \
  -H "Content-Type: application/json" \
  -d '{
    "instruction": "¿Qué es un algoritmo?",
    "output": "Un algoritmo es un conjunto de pasos ordenados para resolver un problema..."
  }'
```

**Opción B: Con CLI**
```bash
python scripts/thau_cli.py add \
  "¿Qué es un algoritmo?" \
  "Un algoritmo es un conjunto de pasos ordenados..."
```

**Opción C: Importar archivo**
```bash
python scripts/thau_cli.py import-data data/datasets/mis_datos.jsonl
```

### 5. Ver estado

```bash
# Ver estado completo
python scripts/thau_cli.py status

# O por API
curl http://localhost:8000/status
```

---

## 📊 Monitoreo del Desarrollo

### Ver edad actual y progreso

```bash
python scripts/thau_cli.py status
```

Salida ejemplo:
```
========================================================================================
🤖 THAU Status
========================================================================================

Modelo: thau-1.5b-v3
Tamaño: 1.5b
Versión: v3

Desarrollo Cognitivo:
  Edad: 3 años (Niño Pequeño)
  Descripción: Razonamiento simple. Puede explicar conceptos básicos con ejemplos.

Progreso:
  Ejemplos en esta edad: 350 / 500 (70%)
  Accuracy actual: 82%
  Accuracy requerida: 80%
  Puede avanzar: No (faltan 150 ejemplos)

Total ejemplos: 750
Entrenamientos completados: 3
```

### Dashboards

**API:**
- Status: `http://localhost:8000/status`
- Stats completas: `http://localhost:8000/stats`
- Ejemplos pendientes: `http://localhost:8000/examples/pending`

---

## 💡 Ejemplos de Uso

### Caso 1: Entrenar desde cero

```bash
# Día 1: Iniciar en edad 0
python scripts/thau_cli.py init --size 1.5b
python api/thau_api.py &

# Agregar datos básicos (edad 0-1)
python scripts/thau_cli.py import-data data/datasets/age_0_newborn.jsonl
python scripts/thau_cli.py import-data data/datasets/age_1_infant.jsonl

# Día 2: THAU entrenó automáticamente y avanzó a edad 1
# Agregar datos de edad 3
python scripts/thau_cli.py import-data data/datasets/age_3_toddler.jsonl

# Semana 1: THAU va aprendiendo progresivamente
# Cada noche se entrena automáticamente
# Va avanzando de edad según su progreso
```

### Caso 2: Entrenar con tu proyecto

```python
# script para capturar interacciones
import requests

def teach_thau(question, answer):
    """Enseña a THAU con cada interacción"""
    requests.post("http://localhost:8000/training/add", json={
        "instruction": question,
        "output": answer,
        "metadata": {"source": "project", "date": "2025-01-13"}
    })

# Durante tu sesión
teach_thau(
    "¿Cómo funciona el módulo de autenticación?",
    "El módulo usa OAuth 2.0 con refresh tokens..."
)
```

### Caso 3: Forzar avance de edad

```bash
# Ver si puede avanzar
python scripts/thau_cli.py status

# Si cumple criterios, forzar entrenamiento
python scripts/thau_cli.py train

# THAU evaluará y avanzará si está listo
```

---

## ⚙️ Configuración Avanzada

### Cambiar intervalo de entrenamiento

Edita `thau_trainer/config.py`:

```python
auto_train_interval_hours = 6  # Entrenar cada 6 horas (en lugar de 24)
```

### Cambiar criterios de avance

Edita `thau_trainer/cognitive_development.py`:

```python
# Por ejemplo, para edad 3:
advancement_criteria={
    "min_examples": 300,  # Reducir de 500 a 300
    "min_accuracy": 0.75,  # Reducir de 0.80 a 0.75
    ...
}
```

### Cambiar tamaño del modelo

```bash
# Migrar a modelo más grande
python scripts/thau_cli.py init --size 7b

# Esto creará thau-7b-v1
```

---

## 🔍 Debugging

### Ver logs detallados

```bash
# Logs del servicio
tail -f data/logs/thau_service.log

# Logs de progreso cognitivo
cat data/logs/cognitive_progress.json | jq
```

### Ver datos en cola

```bash
# Ejemplos pendientes de entrenamiento
ls -la data/training_queue/

# Ejemplos ya entrenados
wc -l data/logs/trained_examples.jsonl
```

### Resetear desarrollo cognitivo

```bash
# CUIDADO: Esto resetea la edad a 0
rm data/logs/cognitive_progress.json

# Reiniciar servicio
```

---

## 📈 Métricas y KPIs

### Métricas clave a monitorear:

1. **Edad cognitiva**: ¿En qué etapa está?
2. **Progreso de edad**: ¿Qué % para siguiente edad?
3. **Accuracy promedio**: ¿Mejora con el tiempo?
4. **Ejemplos por día**: ¿Cuánto aprende?
5. **Tiempo entre avances**: ¿Qué tan rápido crece?

### Exportar métricas

```python
import requests

stats = requests.get("http://localhost:8000/stats").json()

print(f"Edad: {stats['cognitive']['current_age']}")
print(f"Progreso: {stats['cognitive']['progress']['progress_pct']}%")
print(f"Total ejemplos: {stats['service']['total_examples_trained']}")
```

---

## 🎯 Best Practices

### 1. Datos de calidad por edad

- **Edad 0-1**: Respuestas de 1-3 palabras
- **Edad 3-5**: Explicaciones de 1-2 frases
- **Edad 6-10**: Explicaciones con ejemplos
- **Edad 11+**: Razonamiento complejo multi-paso

### 2. Progresión natural

- No fuerces avances prematuros
- Deja que alcance los criterios naturalmente
- Más datos = mejor fundamento

### 3. Diversidad de datos

- Varía los tópicos en cada edad
- No te enfoques solo en un dominio
- Balance entre diferentes tipos de conocimiento

### 4. Monitoreo constante

- Revisa accuracy después de cada entrenamiento
- Verifica que las respuestas tengan sentido
- Testea el modelo en cada edad

---

## 🚀 Producción

### Docker Compose

```yaml
version: '3.8'

services:
  thau-api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
    environment:
      - AUTO_TRAIN_ENABLED=true
    restart: always

  ollama:
    image: ollama/ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama

volumes:
  ollama_data:
```

### Systemd Service

```ini
[Unit]
Description=THAU Training Service
After=network.target

[Service]
Type=simple
User=thau
WorkingDirectory=/opt/thau
ExecStart=/opt/thau/venv/bin/python api/thau_api.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

---

## 🤝 Contribuir

### Agregar nuevas edades

1. Define la etapa en `cognitive_development.py`
2. Crea dataset en `data/datasets/age_X_name.jsonl`
3. Ajusta criterios de avance

### Crear nuevas herramientas

1. Agrega la herramienta en `scripts/agent_runtime.py`
2. Crea ejemplos de uso en datasets
3. Entrena con ejemplos de tool calling

---

## 📚 Recursos

- **Ollama**: https://ollama.ai
- **Curriculum Learning**: https://arxiv.org/abs/2101.10382
- **LoRA**: https://arxiv.org/abs/2106.09685
- **Chain-of-Thought**: https://arxiv.org/abs/2201.11903

---

## 🆘 Troubleshooting

### "THAU no avanza de edad"
- Verifica criterios con `python scripts/thau_cli.py status`
- Asegúrate de tener suficientes ejemplos
- Chequea que el accuracy sea suficiente

### "Entrenamiento no se ejecuta automáticamente"
- Verifica que el servicio esté corriendo
- Revisa `data/logs/thau_service.log`
- Confirma que `auto_train_enabled = True`

### "Modelo no mejora"
- Revisa la calidad de los datos de entrenamiento
- Asegúrate de que los datos sean apropiados para la edad
- Considera incrementar epochs en `config.py`

---

**¡THAU crece mientras tú desarrollas!** 🌱🤖


# Guía Completa THAU-2B
## Modelo Propio con Auto-Aprendizaje y Crecimiento Progresivo

---

## Resumen Ejecutivo

THAU-2B es un modelo de lenguaje de **2B parámetros** entrenado **desde cero** con capacidades únicas:

- **🌱 Crecimiento Progresivo**: El modelo empieza pequeño y crece gradualmente
- **🧠 Auto-Aprendizaje**: Genera sus propias preguntas y aprende de ellas
- **💭 Auto-Preguntas**: Sistema autónomo de generación de conocimiento
- **📚 Auto-Generación de Datasets**: Crea sus propios datos de entrenamiento
- **🎯 Aprendizaje Incremental**: Mejora continuamente sin reentrenamiento completo

---

## Arquitectura THAU-2B

### Configuración del Modelo

```python
THAU_2B_CONFIG = TransformerConfig(
    vocab_size=32000,
    d_model=2560,          # Dimensión oculta
    n_heads=32,            # Cabezas de atención
    n_layers=24,           # Capas transformer
    d_ff=10240,            # Dimensión feed-forward (4x d_model)
    max_seq_length=4096,   # Ventana de contexto
    use_rotary_embeddings=True,  # RoPE para mejor posicionamiento
)
```

### Parámetros Totales

- **Embeddings**: 32,000 × 2,560 = ~82M
- **Por capa**: ~78.6M parámetros
- **24 capas**: 24 × 78.6M = ~1.9B
- **Total**: **~2B parámetros**

---

## Sistema de Crecimiento Progresivo

THAU crece por "edades cognitivas", como un ser humano:

| Edad | Parámetros | d_model | Capas | Capacidades |
|------|-----------|---------|-------|-------------|
| 0 | 18M | 256 | 2 | Palabras básicas |
| 1 | 35M | 384 | 3 | Frases simples |
| 3 | 110M | 512 | 6 | Explicaciones básicas |
| 6 | 450M | 768 | 12 | Razonamiento básico |
| 12 | 1B | 1024 | 24 | Razonamiento complejo |
| **15** | **2B** | **2560** | **24** | **THAU-2B completo** |

---

## Sistema de Auto-Aprendizaje

### 1. Auto-Preguntas (Self-Questioning)

El sistema genera preguntas automáticamente según la edad cognitiva:

```python
# Edad 0: Preguntas básicas
"¿Qué es {concepto}?"
"¿Cómo se usa {concepto}?"

# Edad 3+: Preguntas complejas
"¿Cómo se relaciona {concepto1} con {concepto2}?"
"¿Cuáles son las ventajas de {concepto}?"
```

**Límites de Seguridad:**
- Máximo 10 preguntas/hora
- Máximo 100 preguntas/día
- 30 segundos mínimo entre preguntas

### 2. Detección de Brechas de Conocimiento

El sistema detecta automáticamente cuando no sabe algo:

- Respuestas muy cortas (< 20 caracteres)
- Marcadores de incertidumbre ("no estoy seguro", "no sé")
- Confianza baja (< 0.6)

### 3. Auto-Generación de Datasets

Cuando detecta brechas, genera automáticamente datasets para cubr irlas:

```python
# Ejemplo: Brecha detectada en "algoritmos"
→ Genera 5-10 ejemplos sobre algoritmos
→ Guarda en data/datasets/auto_generated/
→ Entrena con los nuevos ejemplos
```

---

## Cómo Entrenar THAU-2B

### Opción 1: Entrenamiento Progresivo Completo

Entrena desde edad 0 hasta edad 15 (THAU-2B):

```bash
# Activar entorno virtual
source venv/bin/activate

# Entrenar hasta edad 15 (THAU-2B)
python train_thau_2b.py --target-age 15
```

**Tiempo estimado**: 5-10 horas (dependiendo del hardware)

### Opción 2: Entrenamiento por Fases

Entrena gradualmente edad por edad:

```bash
# Fase 1: Edad 0 (bebé)
python train_thau_2b.py --target-age 0

# Fase 2: Edad 1 (niño)
python train_thau_2b.py --target-age 1

# Fase 3: Edad 3 (escolar)
python train_thau_2b.py --target-age 3

# ...hasta edad 15
python train_thau_2b.py --target-age 15
```

### Opción 3: Continuar Entrenamiento Existente

Si ya tienes un checkpoint:

```python
from thau_trainer.own_model_manager import ThauOwnModelManager

manager = ThauOwnModelManager()
manager.load_checkpoint(Path("./data/model_checkpoints/age_12_final.pt"))
manager.advance_age(15)  # Avanzar a THAU-2B
```

---

## Ciclo de Auto-Aprendizaje

El entrenamiento sigue este ciclo automático:

```
1. [Bootstrap] → Datos iniciales básicos
         ↓
2. [Self-Question] → Genera pregunta automática
         ↓
3. [Answer] → Responde usando el modelo
         ↓
4. [Detect Gap] → ¿Respuesta de baja calidad?
         ↓
5. [Generate Dataset] → Crea ejemplos para mejorar
         ↓
6. [Train] → Entrena con nuevos ejemplos
         ↓
7. [Save Checkpoint] → Guarda progreso
         ↓
8. [Repeat] → Vuelve al paso 2
```

---

## Estructura de Archivos

```
my-llm/
├── config/
│   └── model_configs.py          # THAU_2B_CONFIG agregado
├── thau_trainer/
│   ├── own_model_manager.py      # Gestor del modelo propio (actualizado)
│   ├── self_questioning.py       # Sistema de auto-preguntas
│   └── self_learning.py          # Auto-generación de datasets
├── train_thau_2b.py              # Script principal de entrenamiento
├── data/
│   ├── model_checkpoints/        # Checkpoints guardados
│   ├── datasets/auto_generated/  # Datasets auto-generados
│   ├── training_stats/           # Estadísticas por fase
│   ├── logs/                     # Logs del sistema
│   └── self_questioning/         # Preguntas generadas
└── export/
    ├── export_to_gguf.py         # Exportador a formato GGUF
    └── Modelfile-thau            # Configuración para Ollama
```

---

## Exportar a Ollama

Una vez entrenado THAU-2B, exporta a Ollama:

### 1. Exportar a GGUF

```bash
python export/export_to_gguf.py \
  --model-path ./data/model_checkpoints/age_15_final.pt \
  --output-dir ./export/models \
  --quantization Q4_K_M
```

### 2. Crear Modelo en Ollama

```bash
# Crear Modelfile personalizado
cat > export/Modelfile-thau-2b <<EOF
FROM ./export/models/thau-2b-Q4_K_M.gguf

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx 4096

SYSTEM """Eres THAU, un modelo de 2B parámetros entrenado desde cero con
capacidades de auto-aprendizaje y razonamiento avanzado..."""
EOF

# Importar a Ollama
ollama create thau-2b -f export/Modelfile-thau-2b
```

### 3. Usar THAU-2B

```bash
# Modo interactivo
ollama run thau-2b

# Consulta directa
ollama run thau-2b "Explica qué es Clean Architecture"

# Desde Python
import ollama
response = ollama.chat(model='thau-2b', messages=[
    {'role': 'user', 'content': '¿Qué es SOLID?'}
])
print(response['message']['content'])
```

---

## Monitoreo y Métricas

### Ver Estadísticas de Entrenamiento

```python
from thau_trainer.own_model_manager import ThauOwnModelManager
import json

manager = ThauOwnModelManager()
manager.load_checkpoint(Path("./data/model_checkpoints/age_15_final.pt"))

stats = manager.get_stats()
print(json.dumps(stats, indent=2))
```

### Ver Estadísticas de Auto-Aprendizaje

```python
from thau_trainer.self_learning import SelfLearningManager

self_learning = SelfLearningManager()
stats = self_learning.get_stats()

print(f"Brechas detectadas: {stats['total_gaps_detected']}")
print(f"Datasets generados: {stats['total_datasets_generated']}")
print(f"Ejemplos totales: {stats['total_examples_generated']}")
```

### Ver Logs de Preguntas Auto-Generadas

```bash
# Ver últimas 10 preguntas
tail -10 data/self_questioning/activity_log.json

# Ver todas las Q&A del día
cat data/self_questioning/qa_$(date +%Y%m%d).jsonl
```

---

## Mejores Prácticas

### 1. Entrenamiento Incremental

- ✅ Entrena gradualmente por edades
- ✅ Guarda checkpoints cada 25 steps
- ✅ Monitorea la perplexity (debe bajar)
- ❌ No saltes edades abruptamente

### 2. Auto-Aprendizaje

- ✅ Respeta los límites de preguntas/hora
- ✅ Revisa las brechas detectadas periódicamente
- ✅ Valida los datasets auto-generados
- ❌ No deshabilites los límites de seguridad

### 3. Recursos

- **RAM**: Mínimo 16GB (32GB recomendado)
- **GPU**: MPS (Apple Silicon) o CUDA
- **Disco**: 20GB libres para checkpoints
- **Tiempo**: 5-10 horas para entrenamiento completo

---

## Solución de Problemas

### Error: "CUDA out of memory" / "MPS out of memory"

```python
# Reducir batch size en train_thau_2b.py
gradient_accumulation_steps=8  # Aumentar de 2 a 8
```

### Error: "Tokenizer retorna string"

Ya corregido en `own_model_manager.py` líneas 211-218 y 316-321

### El entrenamiento es muy lento

```bash
# Reducir steps por fase
python train_thau_2b.py --target-age 3 --steps-per-age 25
```

### No genera auto-preguntas

Verifica que Ollama esté corriendo:

```bash
ollama list  # Debe mostrar modelos disponibles
```

---

## Próximos Pasos

1. **Entrenar THAU-2B**: Ejecuta `python train_thau_2b.py --target-age 15`
2. **Exportar a Ollama**: Usa el modelo entrenado en producción
3. **Continuar Aprendizaje**: Deja que THAU se auto-entrene continuamente
4. **Fine-tuning Especializado**: Entrena en dominios específicos

---

## Comandos Rápidos

```bash
# Ver modelos disponibles en Ollama
ollama list

# Usar THAU actual (TinyLlama 1.1B)
ollama run thau:latest

# Entrenar THAU-2B desde cero
python train_thau_2b.py --target-age 15

# Verificar progreso
tail -f data/training_output.log

# Ver checkpoints guardados
ls -lh data/model_checkpoints/

# Exportar modelo entrenado
python export/export_to_gguf.py
```

---

## Contacto y Soporte

- **Logs**: `data/logs/`
- **Checkpoints**: `data/model_checkpoints/`
- **Datasets**: `data/datasets/auto_generated/`
- **Estadísticas**: `data/training_stats/`

---

**¡THAU-2B está listo para crecer y aprender! 🚀**

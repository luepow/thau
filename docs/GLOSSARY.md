# THAU - Glosario de Términos

Guía completa de términos, conceptos y acrónimos utilizados en THAU.

---

## A

### Adapter (Adaptador)
Componente que abstrae diferencias entre plataformas (CUDA, MPS, CPU). En THAU se usa para `DeviceManager` y `ModelAdapter`.

### Age (Edad Cognitiva)
Concepto único de THAU que representa el nivel de madurez del modelo. Cada edad tiene diferente cantidad de parámetros:
- **Age 0**: 256K parámetros (conceptos básicos)
- **Age 15**: 2B parámetros (capacidad completa)

### Agent (Agente)
Componente especializado del sistema THAU. Hay 11 tipos de agentes especializados en diferentes tareas (Code Writer, Debugger, Planner, etc.).

### Agent Orchestrator (Orquestador de Agentes)
Sistema que coordina y asigna tareas a los agentes especializados según el contexto.

### API Toolkit
Conjunto de herramientas para trabajar con APIs REST, webhooks, autenticación OAuth2, etc.

### Attention (Atención)
Mecanismo que permite al modelo "enfocarse" en partes relevantes de la entrada. THAU usa Multi-Head Attention con RoPE.

---

## B

### Batch Size (Tamaño de Lote)
Número de ejemplos procesados simultáneamente durante el entrenamiento. En THAU se ajusta automáticamente según memoria disponible.

### Beam Search
Estrategia de decodificación que explora múltiples secuencias simultáneamente, manteniendo las N más probables.

### BPE (Byte-Pair Encoding)
Algoritmo de tokenización que descompone texto en subpalabras. THAU usa el tokenizer de GPT-2.

---

## C

### Causal Mask (Máscara Causal)
Máscara que impide al modelo ver tokens futuros durante la generación, asegurando autorregresión.

### Checkpoint
Snapshot del modelo en un momento específico del entrenamiento. Incluye pesos, optimizador y metadatos.

### ChromaDB
Base de datos vectorial usada por THAU para memoria a largo plazo y búsqueda semántica.

### Cognitive Growth (Crecimiento Cognitivo)
Innovación de THAU: el modelo crece en parámetros según su "edad", desde 256K hasta 2B.

### Context Window (Ventana de Contexto)
Máxima cantidad de tokens que el modelo puede procesar simultáneamente. En THAU varía según edad.

### CORS (Cross-Origin Resource Sharing)
Mecanismo de seguridad HTTP. THAU Code Server lo configura para permitir acceso desde Electron.

---

## D

### d_ff (Feed-Forward Dimension)
Dimensión de la capa Feed-Forward en el transformer. Típicamente 4x d_model.

### d_model (Model Dimension)
Dimensión principal del modelo transformer. En THAU varía de 256 (Age 0) a 1536 (Age 15).

### DeviceManager
Componente que detecta y optimiza el uso de dispositivos (CUDA, MPS, CPU) automáticamente.

### Dropout
Técnica de regularización que desactiva aleatoriamente neuronas durante entrenamiento para evitar overfitting.

---

## E

### Embedding
Representación vectorial densa de tokens. THAU aprende embeddings de 256 a 1536 dimensiones.

### Encoder-Decoder
Arquitectura transformer con dos componentes. THAU usa solo decoder (GPT-style) para generación autorregresiva.

### Episodic Memory (Memoria Episódica)
Sistema de memoria que almacena experiencias temporales en SQLite con timestamp y contexto.

---

## F

### FastAPI
Framework web Python usado por THAU para crear REST APIs y WebSocket servers.

### Feed-Forward Network (FFN)
Red neuronal densa aplicada a cada posición. En THAU: Linear → GELU → Dropout → Linear.

### Fine-Tuning
Ajuste de un modelo preentrenado a una tarea específica. THAU usa LoRA para fine-tuning eficiente.

---

## G

### GELU (Gaussian Error Linear Unit)
Función de activación usada en THAU. Más suave que ReLU.

### Generation (Generación)
Proceso de producir nuevo texto token por token. THAU soporta greedy, sampling, beam search.

### Gradient Descent (Descenso de Gradiente)
Algoritmo de optimización usado para entrenar redes neuronales ajustando pesos según pérdida.

---

## H

### Hidden States
Representaciones internas del modelo en cada capa. THAU puede retornar hidden states para análisis.

### HuggingFace
Biblioteca de transformers. THAU usa sus tokenizers y puede cargar modelos compatibles.

---

## I

### Inference (Inferencia)
Proceso de usar un modelo entrenado para hacer predicciones sin actualizar pesos.

### Imagination (Imaginación Visual)
Capacidad única de THAU de generar imágenes usando VAE entrenado desde cero.

### Incremental Learning (Aprendizaje Incremental)
Capacidad de aprender de nuevas interacciones sin olvidar conocimiento previo. THAU usa LoRA.

---

## K

### KV Cache (Key-Value Cache)
Optimización que almacena keys/values de atención para generación más rápida. THAU lo implementa opcionalmente.

---

## L

### Latent Space (Espacio Latente)
Representación comprimida aprendida por VAE. THAU Visual usa 64 dimensiones latentes.

### Layer Normalization (LayerNorm)
Técnica de normalización aplicada a través de features. THAU usa Pre-LN (antes de atención/FFN).

### Learning Rate (Tasa de Aprendizaje)
Hiperparámetro que controla el tamaño del paso en gradient descent. THAU usa scheduling con warmup.

### Logits
Scores sin normalizar antes de softmax. THAU genera logits para cada token del vocabulario.

### Long-Term Memory (Memoria a Largo Plazo)
Sistema de memoria basado en ChromaDB para almacenar conocimiento persistente con búsqueda semántica.

### LoRA (Low-Rank Adaptation)
Técnica de fine-tuning eficiente que añade matrices de bajo rango. THAU lo usa para aprendizaje incremental.

---

## M

### Max Sequence Length (Longitud Máxima de Secuencia)
Máximo número de tokens procesables. Varía en THAU de 512 (Age 0) a 2048 (Age 15).

### MCP (Model Context Protocol)
Protocolo estándar para interoperabilidad entre sistemas de IA. THAU es MCP-compatible.

### Monaco Editor
Editor de código de VS Code. Usado en THAU Code Desktop para edición de código.

### MPS (Metal Performance Shaders)
Backend de aceleración en GPUs de Apple Silicon. DeviceManager lo detecta y usa automáticamente.

### Multi-Head Attention
Mecanismo de atención con múltiples "cabezas" que capturan diferentes aspectos. THAU usa 4-24 heads según edad.

---

## N

### n_heads (Number of Attention Heads)
Número de cabezas de atención paralelas. En THAU: 4 (Age 0) a 24 (Age 15).

### n_layers (Number of Layers)
Número de bloques transformer apilados. En THAU: 2 (Age 0) a 24 (Age 15).

### Nucleus Sampling (Top-p Sampling)
Estrategia de muestreo que selecciona de un subset dinámico de tokens con probabilidad acumulada p.

---

## O

### Orchestrator (Orquestador)
Componente que coordina agentes especializados. Ver `ThauAgentOrchestrator`.

### Overfitting (Sobreajuste)
Cuando el modelo memoriza datos de entrenamiento en vez de generalizar. THAU usa dropout y regularización.

---

## P

### Parameters (Parámetros)
Pesos del modelo neural. THAU crece de 256K a 2B parámetros según edad.

### Planner (Planificador)
Agente especializado que descompone tareas complejas en pasos ejecutables con estimaciones de tiempo/esfuerzo.

### Position Encoding (Codificación Posicional)
Información sobre posición de tokens en secuencia. THAU usa RoPE (Rotary Position Embedding).

### Preload Script
Script de Electron que actúa como puente seguro entre renderer y main process.

### PyTorch
Framework de deep learning usado por THAU como base para todos los modelos.

---

## Q

### Quantization (Cuantización)
Reducción de precisión numérica (float32 → int8) para ahorrar memoria. THAU soporta 8-bit y 4-bit.

---

## R

### RAG (Retrieval-Augmented Generation)
Técnica que combina búsqueda en bases de datos con generación. THAU usa ChromaDB para RAG.

### Reasoning (Razonamiento)
Capacidad de análisis lógico. THAU implementa Chain-of-Thought y planning.

### ReLU (Rectified Linear Unit)
Función de activación. THAU usa GELU en su lugar.

### REST API
Interfaz HTTP para interactuar con THAU. Ver `/api/agents/*`, `/api/planner/*`, etc.

### RoPE (Rotary Position Embedding)
Técnica avanzada de position encoding que codifica información relativa. THAU la implementa en atención.

---

## S

### Sampling
Proceso de seleccionar el siguiente token basándose en probabilidades. THAU soporta greedy, top-k, nucleus.

### Self-Attention (Auto-Atención)
Mecanismo donde cada token atiende a todos los demás en la secuencia.

### Self-Learning (Auto-Aprendizaje)
Capacidad de THAU de aprender de sus interacciones y auto-cuestionarse para mejorar.

### Self-Questioning (Auto-Cuestionamiento)
Técnica donde THAU genera preguntas sobre sus respuestas para identificar áreas de mejora.

### Session (Sesión)
Contexto de interacción con ID único. Mantiene historial de conversación y estado de agentes.

### Short-Term Memory (Memoria a Corto Plazo)
Buffer FIFO que mantiene contexto reciente de conversación (últimos N mensajes).

### Softmax
Función que convierte logits en distribución de probabilidad. Usada antes de sampling.

---

## T

### Temperature
Hiperparámetro que controla aleatoriedad en generación. Valores bajos → determinista, altos → creativo.

### THAU
Transformative Holistic Autonomous Unit - Sistema de IA con imaginación visual, auto-creación de herramientas y crecimiento cognitivo.

### THAU-2B
Modelo de lenguaje de THAU que crece de 256K a 2B parámetros según edad cognitiva.

### THAU Code Desktop
Aplicación Electron + React para interactuar con THAU (similar a Claude Code).

### THAU Vision
Sistema VAE de THAU para generación de imágenes desde texto (imaginación visual).

### Token
Unidad mínima de texto procesada por el modelo. THAU usa tokenizer BPE con ~50K tokens de vocabulario.

### Tokenizer
Componente que convierte texto en tokens. THAU usa GPT-2 tokenizer de HuggingFace.

### Tool Call (Invocación de Herramienta)
Capacidad del modelo de llamar funciones externas. THAU soporta tool calling nativo.

### Tool Factory (Fábrica de Herramientas)
Innovación única de THAU: crea herramientas ejecutables desde descripciones en lenguaje natural.

### Top-k Sampling
Estrategia de muestreo que selecciona aleatoriamente entre los k tokens más probables.

### Top-p Sampling
Ver Nucleus Sampling.

### Training (Entrenamiento)
Proceso de ajustar pesos del modelo usando gradient descent en datos de entrenamiento.

### Transformer
Arquitectura de red neuronal basada en atención. Base de THAU.

---

## V

### VAE (Variational Autoencoder)
Tipo de red neuronal que aprende representaciones latentes. THAU Visual lo usa para generar imágenes.

### Vocabulary (Vocabulario)
Conjunto de tokens conocidos por el modelo. THAU usa ~50,257 tokens (GPT-2).

---

## W

### WebSocket
Protocolo de comunicación bidireccional en tiempo real. THAU lo usa para chat streaming.

### Weight (Peso)
Parámetro aprendible de la red neuronal. THAU tiene 256K-2B pesos según edad.

---

## Conceptos Avanzados de THAU

### Auto-Tool Creation (Auto-Creación de Herramientas)
Capacidad única de THAU de generar código ejecutable de herramientas desde descripciones en lenguaje natural. Ejemplo:

```
Descripción: "Consultar API de clima y enviar email"
↓
Genera: weather_email_tool.py con:
- Consulta a API
- Parsing de respuesta
- Envío de email
- Manejo de errores
```

### Cognitive Age Scaling (Escalado por Edad Cognitiva)
Innovación de THAU donde la arquitectura del modelo escala dinámicamente:

| Edad | d_model | n_heads | n_layers | Parámetros |
|------|---------|---------|----------|------------|
| 0    | 256     | 4       | 2        | 256K       |
| 1    | 384     | 6       | 3        | 768K       |
| 3    | 512     | 8       | 6        | 1.7M       |
| 7    | 768     | 12      | 12       | 12M        |
| 15   | 1536    | 24      | 24       | 2B         |

### Visual Imagination Pipeline
Proceso completo de generación de imágenes en THAU Vision:

1. **Text Encoding**: Tokenizar prompt
2. **Latent Sampling**: Generar vector latente (64D)
3. **VAE Decoding**: Decodificar a imagen (32x32, 64x64, 128x128)
4. **Post-Processing**: Aplicar filtros y mejoras

### Agent Collaboration Pattern
Patrón donde múltiples agentes trabajan secuencialmente:

```
1. Planner → Descompone tarea
2. Code Writer → Implementa cada paso
3. Code Reviewer → Revisa implementación
4. Tester → Escribe tests
5. Documenter → Genera documentación
```

### Self-Learning Cycle
Ciclo continuo de mejora de THAU:

1. **Interacción**: Usuario hace pregunta
2. **Generación**: THAU responde
3. **Auto-Cuestionamiento**: "¿Qué preguntas relacionadas tendría?"
4. **Generación de Datos**: Crea pares pregunta-respuesta
5. **Fine-Tuning**: Entrena con LoRA
6. **Mejora**: Modelo actualizado con nuevo conocimiento

---

## Abreviaciones Comunes

- **API**: Application Programming Interface
- **BPE**: Byte-Pair Encoding
- **CORS**: Cross-Origin Resource Sharing
- **FFN**: Feed-Forward Network
- **GELU**: Gaussian Error Linear Unit
- **KV**: Key-Value
- **LoRA**: Low-Rank Adaptation
- **LLM**: Large Language Model
- **MCP**: Model Context Protocol
- **MPS**: Metal Performance Shaders
- **RAG**: Retrieval-Augmented Generation
- **REST**: Representational State Transfer
- **RoPE**: Rotary Position Embedding
- **VAE**: Variational Autoencoder

---

## Métricas y Medidas

### Loss (Pérdida)
Métrica de error del modelo. Valores más bajos = mejor. THAU monitorea:
- Training loss
- Validation loss
- Reconstruction loss (VAE)

### Perplexity (Perplejidad)
Medida de incertidumbre del modelo. Perplejidad = exp(loss). Valores bajos = mejor.

### BLEU Score
Métrica de calidad de traducción/generación. Compara texto generado con referencia.

### Latency (Latencia)
Tiempo de respuesta:
- **First Token Latency**: Tiempo hasta primer token (< 200ms ideal)
- **Generation Speed**: Tokens/segundo (30-50 tokens/s ideal)

---

## Referencias Técnicas

### Papers Implementados

1. **Attention Is All You Need** (Vaswani et al., 2017)
   - Arquitectura transformer base
   - Multi-head attention
   - Position encoding

2. **RoFormer** (Su et al., 2021)
   - Rotary Position Embedding
   - Atención relativa mejorada

3. **LoRA** (Hu et al., 2021)
   - Low-Rank Adaptation
   - Fine-tuning eficiente

4. **VAE** (Kingma & Welling, 2013)
   - Variational Autoencoders
   - Generación de imágenes

### Arquitecturas Inspiradoras

- **GPT**: Autoregressive language model
- **BERT**: Bidirectional encoding
- **T5**: Text-to-text framework
- **DALL-E**: Text-to-image generation
- **Claude**: Agent-based assistance

---

## Comandos Útiles

### Entrenamiento

```bash
# Entrenar THAU-2B
python train_thau_2b.py --target-age 15

# Entrenar THAU Vision
python train_thau_vision.py --age 0 --epochs 30

# Entrenar Tool Calling
python train_tool_calling.py --epochs 3
```

### API

```bash
# Iniciar servidor
python api/thau_code_server.py

# Health check
curl http://localhost:8001/health

# Crear tarea
curl -X POST http://localhost:8001/api/agents/task \
  -H "Content-Type: application/json" \
  -d '{"description": "Escribe una función", "role": "code_writer"}'
```

### Desktop App

```bash
# Iniciar desarrollo
cd thau-code-desktop && npm run dev

# Build producción
npm run build && npm run build:electron
```

---

## Troubleshooting

### Términos de Error Comunes

- **OOM (Out of Memory)**: Sin memoria GPU/RAM. Reducir batch_size o usar quantization.
- **CUDA Error**: Problema con GPU NVIDIA. Verificar drivers y PyTorch instalado con CUDA.
- **Tokenization Error**: Texto no puede tokenizarse. Verificar caracteres especiales.
- **Shape Mismatch**: Dimensiones de tensors incompatibles. Verificar config del modelo.

---

Este glosario es un documento vivo que se actualiza con nuevas características de THAU.

**Última actualización**: 2025-11-15
**Versión**: 2.0.0

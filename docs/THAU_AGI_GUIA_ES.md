# THAU AGI v2 - Guia Completa en Espanol

## Que es THAU AGI?

THAU AGI v2 es un sistema Proto-AGI (Inteligencia Artificial General Prototipo) que integra multiples capacidades avanzadas:

- **Ciclo ReAct**: Razona antes de actuar (THINK -> PLAN -> ACT -> OBSERVE -> REFLECT)
- **Aprendizaje Experiencial**: Aprende de interacciones pasadas
- **Metacognicion**: Se auto-evalua para mejorar
- **Busqueda Web**: Busca informacion en internet
- **Multi-Agente**: Colaboracion entre agentes especializados
- **Knowledge Base**: Base de conocimiento con RAG (Retrieval Augmented Generation)
- **Feedback Loop**: Mejora continua con retroalimentacion del usuario

---

## Instalacion Rapida

### Paso 1: Clonar y Preparar

```bash
# Clonar repositorio
git clone https://github.com/tu-usuario/my-llm.git
cd my-llm

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

### Paso 2: Instalar Ollama (Opcional pero Recomendado)

```bash
# En macOS
brew install ollama

# En Linux
curl -fsSL https://ollama.com/install.sh | sh

# Iniciar Ollama
ollama serve

# Descargar un modelo (en otra terminal)
ollama pull llama3.2
```

---

## Formas de Usar THAU

### Opcion 1: Interfaz Web con Gradio (Recomendado para Principiantes)

```bash
# Con Ollama (ejecuta localmente)
python scripts/gradio_thau_ollama.py

# Con modelo HuggingFace
python scripts/gradio_thau_agi.py
```

Luego abre tu navegador en: **http://localhost:7860**

### Opcion 2: Desde Python (Para Desarrolladores)

```python
from capabilities.proto_agi import ThauAGIv2, ThauConfig, ThauMode

# Crear agente
agent = ThauAGIv2()

# Chat simple
respuesta = agent.chat("Hola, que puedes hacer?")
print(respuesta)

# Ejecutar tarea
resultado = agent.run("Calcula 25 * 4 + 100", ThauMode.TASK)
print(resultado["response"])

# Investigar tema
resultado = agent.run("Investiga sobre machine learning", ThauMode.RESEARCH)
print(resultado["response"])
```

### Opcion 3: Demo Interactiva en Terminal

```bash
python scripts/test_thau_agi_v2.py --demo
```

---

## Herramientas Disponibles

THAU tiene 8 herramientas integradas:

| Herramienta | Descripcion | Ejemplo de Uso |
|-------------|-------------|----------------|
| `calculate` | Calculos matematicos | "Calcula 25 * 4 + 100" |
| `read_file` | Leer archivos | "Lee el archivo config.py" |
| `write_file` | Escribir archivos | "Escribe 'hola mundo' en test.txt" |
| `list_directory` | Listar directorios | "Lista los archivos del directorio actual" |
| `execute_python` | Ejecutar codigo Python | "Ejecuta print(2**10)" |
| `web_search` | Buscar en internet | "Busca en internet que es Python" |
| `fetch_url` | Obtener contenido de URL | "Obten el contenido de python.org" |
| `research` | Investigacion profunda | "Investiga sobre inteligencia artificial" |

---

## Modos de Operacion

THAU puede operar en 5 modos diferentes:

### 1. Modo Chat (Por defecto)
Para conversaciones casuales y preguntas generales.

```python
agent.chat("Hola, como estas?")
```

### 2. Modo Tarea
Para ejecutar tareas especificas con herramientas.

```python
agent.run("Calcula el factorial de 10", ThauMode.TASK)
```

### 3. Modo Investigacion
Para busqueda profunda de informacion.

```python
agent.run("Investiga sobre energia renovable", ThauMode.RESEARCH)
```

### 4. Modo Colaborativo
Usa multiples agentes especializados.

```python
agent.run("Desarrolla una funcion para ordenar listas", ThauMode.COLLABORATIVE)
```

### 5. Modo Aprendizaje
Aprendizaje intensivo de nuevos conceptos.

```python
agent.run("Aprende sobre patrones de diseno", ThauMode.LEARNING)
```

---

## Sistema de Feedback

THAU aprende de tu retroalimentacion:

```python
# Si la respuesta fue buena
agent.thumbs_up()

# Si la respuesta no fue correcta
agent.thumbs_down(reason="La respuesta estaba incompleta")

# Para corregir una respuesta
agent.correct("La respuesta correcta es...")
```

En la interfaz Gradio, usa los botones 👍 y 👎.

---

## Componentes del Sistema

### 1. Aprendizaje Experiencial

```python
from capabilities.proto_agi import ExperienceStore, Experience

# El sistema guarda automaticamente las experiencias
store = ExperienceStore()

# Buscar experiencias similares
experiencias = store.find_similar_experiences("calculo matematico")
```

### 2. Metacognicion

El motor metacognitivo evalua cada respuesta:
- Confianza de la respuesta (0-100%)
- Deteccion de incertidumbre
- Sugerencias de mejora

### 3. Knowledge Base con RAG

```python
from capabilities.proto_agi import KnowledgeStore, KnowledgeType

store = KnowledgeStore()

# Guardar conocimiento
store.store(
    content="Python es un lenguaje de programacion interpretado",
    knowledge_type=KnowledgeType.FACT,
    source="manual"
)

# Recuperar conocimiento relevante
resultados = store.retrieve("lenguaje programacion")
```

### 4. Sistema Multi-Agente

Agentes especializados disponibles:
- **CODER**: Escribe y revisa codigo
- **REVIEWER**: Revisa calidad del codigo
- **RESEARCHER**: Investiga informacion
- **PLANNER**: Planifica tareas complejas
- **TESTER**: Prueba funcionalidad

---

## Ejemplos Practicos

### Ejemplo 1: Calculadora Inteligente

```
Usuario: Calcula cuanto es 15% de 250 mas 50
THAU: Voy a calcular esto paso a paso:
      1. 15% de 250 = 250 * 0.15 = 37.5
      2. 37.5 + 50 = 87.5
      El resultado es 87.5
```

### Ejemplo 2: Explorar Archivos

```
Usuario: Lista los archivos Python en el directorio actual
THAU: Encontre los siguientes archivos .py:
      - main.py
      - config.py
      - utils.py
      - test_app.py
```

### Ejemplo 3: Investigacion Web

```
Usuario: Busca en internet las ultimas noticias sobre IA
THAU: He encontrado los siguientes resultados:
      1. "OpenAI lanza nuevo modelo GPT-5..."
      2. "Google anuncia avances en Gemini..."
      3. "Meta presenta Llama 3..."
```

### Ejemplo 4: Generar Codigo

```
Usuario: Escribe una funcion Python que calcule el factorial
THAU: def factorial(n):
          if n <= 1:
              return 1
          return n * factorial(n - 1)
```

---

## Configuracion Avanzada

### Configurar Componentes

```python
from capabilities.proto_agi import ThauConfig, ThauAGIv2

config = ThauConfig(
    # Modelo
    checkpoint_path="ruta/al/modelo",
    base_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",

    # Comportamiento
    max_iterations=10,
    confidence_threshold=0.6,

    # Activar/Desactivar features
    enable_learning=True,
    enable_metacognition=True,
    enable_web_search=True,
    enable_multi_agent=True,
    enable_knowledge_base=True,
    enable_feedback=True,

    # Limites
    max_tokens=500,
    timeout_seconds=120.0,

    # Debug
    verbose=True
)

agent = ThauAGIv2(config)
```

### Configurar Ollama

```python
from scripts.gradio_thau_ollama import ThauOllama, OllamaConfig

config = OllamaConfig(
    base_url="http://localhost:11434",
    model="llama3.2",
    temperature=0.7,
    max_tokens=1000,
    timeout=120
)

agent = ThauOllama(ollama_config=config)
```

---

## Solucionar Problemas

### Error: "No se pudo conectar con Ollama"

```bash
# Verificar que Ollama este corriendo
ollama serve

# En otra terminal, verificar modelos
ollama list
```

### Error: "Modelo no encontrado"

```bash
# Descargar el modelo
ollama pull llama3.2
```

### Error: "Web search no disponible"

```bash
# Instalar dependencias opcionales
pip install httpx beautifulsoup4 html2text
```

### Error de memoria

```python
# Usar modelo mas pequeno
config = ThauConfig(
    base_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    max_tokens=256
)
```

---

## Tests

```bash
# Tests rapidos (sin modelo)
python scripts/test_thau_agi_v2.py --quick

# Tests completos
python scripts/test_thau_agi_v2.py

# Benchmark de rendimiento
python scripts/test_thau_agi_v2.py --benchmark

# Test de web search
python scripts/test_web_search.py --quick
```

---

## Estructura del Proyecto

```
my-llm/
├── capabilities/
│   ├── proto_agi/
│   │   ├── __init__.py           # Exports principales
│   │   ├── thau_proto_agi.py     # Ciclo ReAct basico
│   │   ├── thau_agi.py           # AGI v1
│   │   ├── thau_agi_v2.py        # AGI v2 (unificado)
│   │   ├── experiential_learning.py  # Aprendizaje
│   │   ├── multi_agent.py        # Multi-agente
│   │   └── knowledge_base.py     # Knowledge + RAG
│   └── tools/
│       ├── web_search.py         # Busqueda web
│       └── system_tools.py       # Herramientas sistema
├── scripts/
│   ├── gradio_thau_agi.py        # UI con HuggingFace
│   ├── gradio_thau_ollama.py     # UI con Ollama
│   ├── test_thau_agi_v2.py       # Tests
│   └── test_web_search.py        # Tests web
└── docs/
    ├── THAU_AGI_GUIA_ES.md       # Esta guia
    └── THAU_AGI_GUIDE_EN.md      # English guide
```

---

## Contribuir

1. Fork el repositorio
2. Crea una rama: `git checkout -b mi-feature`
3. Haz commits: `git commit -m "Agrega feature"`
4. Push: `git push origin mi-feature`
5. Abre un Pull Request

---

## Licencia

MIT License - Ver archivo LICENSE

---

## Creditos

Desarrollado con amor para Thomas & Aurora.

**THAU** = **TH**omas + **AU**rora

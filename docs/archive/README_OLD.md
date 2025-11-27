# THAU - Unidad Autónoma Holística Transformativa

<div align="center">

![THAU](https://img.shields.io/badge/THAU-Sistema%20IA-blue?style=for-the-badge)
![Version](https://img.shields.io/badge/version-2.0.0-green?style=for-the-badge)
![License](https://img.shields.io/badge/license-MIT-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/python-3.10+-yellow?style=for-the-badge)
![Venezuela](https://img.shields.io/badge/Hecho%20en-Venezuela-yellow?style=for-the-badge)

**La Primera IA del Mundo con Imaginación Visual y Auto-Creación de Herramientas**

*"Donde la Inteligencia Crea su Propio Camino"*

[**Documentación Completa**](docs/README.md) | [**Inicio Rápido**](docs/getting-started/quickstart.md) | [**Instalación**](docs/getting-started/installation.md) | [**Glosario**](docs/GLOSSARY.md)

</div>

---

## 💝 La Historia detrás de THAU

**THAU** lleva el nombre de mis dos hijos, **Thomas** y **Aurora**, representando la curiosidad infinita, el aprendizaje constante y el crecimiento natural que caracteriza a la infancia. Como ellos aprenden del mundo creando sus propias herramientas y formas de entender la realidad, THAU fue diseñado para hacer lo mismo en el mundo de la inteligencia artificial.

> *"Así como Thomas y Aurora crecen, aprenden y crean cada día, THAU evoluciona de Age 0 (256K parámetros) a Age 15 (2B parámetros), desarrollando capacidades cognitivas más complejas con el tiempo."*

**De Venezuela 🇻🇪 para el Mundo**
Desarrollado con pasión por **Luis Eduardo Perez** ([@luepow](https://github.com/luepow))

---

---

## 🚀 ¿Qué es THAU?

**THAU** es un sistema de inteligencia artificial de última generación que combina capacidades únicas en el mundo:

### 🌟 Características Revolucionarias

#### 1. 🎨 Imaginación Visual Propia (THAU Vision)
Sistema VAE entrenado desde cero para generar imágenes desde descripciones de texto.

```python
vision = ThauVisionModel(age=0)
images = vision.imagine("un robot pintando")
```

#### 2. 🏭 Auto-Creación de Herramientas (Único en el Mundo)
THAU puede crear sus propias herramientas desde lenguaje natural.

```python
factory = ToolFactory()
tool = factory.create_tool("Consultar API de clima y enviar email")
# Genera: weather_email_tool.py listo para usar
```

#### 3. 🧠 Crecimiento Cognitivo (Age 0 → 15)
Modelo que evoluciona de 256K a 2B parámetros según su "edad cognitiva".

| Edad | Parámetros | Capacidad |
|------|-----------|-----------|
| 0 | 256K | Conceptos básicos |
| 3 | 1.7M | Párrafos coherentes |
| 7 | 12M | Razonamiento complejo |
| 15 | 2B | Capacidad completa |

#### 4. 🤖 Sistema de 11 Agentes Especializados
Orquestación inteligente de agentes expertos en diferentes tareas.

```
💬 General  ✍️ Code Writer  👀 Code Reviewer  🐛 Debugger
🔍 Researcher  📋 Planner  🏗️ Architect  🧪 Tester
📝 Documenter  🔌 API Specialist  🔒 Security  🎨 Visual Creator
```

#### 5. 📚 Self-Learning con Auto-Questioning
Aprende de sus interacciones y se auto-cuestiona para mejorar.

#### 6. 🔌 MCP Compatible
Interoperable con Claude, OpenAI y otros sistemas mediante Model Context Protocol.

---

## 📊 ¿Por Qué THAU es Único?

| Característica | THAU | Claude Code | GPT-4 | Copilot |
|---------------|------|-------------|-------|---------|
| Imaginación Visual Propia | ✅ | ❌ | DALL-E | ❌ |
| Auto-Creación de Herramientas | ✅ | ❌ | ❌ | ❌ |
| Crecimiento Cognitivo | ✅ | ❌ | ❌ | ❌ |
| 11 Agentes Especializados | ✅ | ❌ | ❌ | ❌ |
| Self-Learning | ✅ | ❌ | ❌ | ❌ |
| Desktop App | ✅ | ✅ | ✅ | ✅ |
| Open Source | ✅ | ❌ | ❌ | ❌ |
| MCP Compatible | ✅ | ✅ | ✅ | ❌ |

---

## 🎯 Casos de Uso

### Desarrollo Full-Stack
```python
# Planificar arquitectura
orchestrator.assign_task("Diseña un sistema de e-commerce", role="planner")

# Escribir código
orchestrator.assign_task("Implementa la API según el plan", role="code_writer")

# Revisar y testear
orchestrator.assign_task("Revisa el código y genera tests", role="code_reviewer")
```

### Generación Visual
```python
vision = ThauVisionModel(age=0)
images = vision.imagine([
    "logo de startup tech",
    "interfaz moderna de dashboard",
    "diagrama de arquitectura microservicios"
])
```

### Auto-Herramientas
```python
# THAU crea la herramienta automáticamente
tool = factory.create_tool(
    "Consultar GitHub API, encontrar issues abiertos y enviar reporte"
)
result = tool.execute(repo="microsoft/vscode")
```

---

## 🚀 Inicio Rápido (5 Minutos)

### 1. Instalar

```bash
# Clonar repositorio
git clone https://github.com/your-org/thau.git
cd thau

# Crear entorno virtual
python3 -m venv venv
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Configurar

```bash
cp .env.example .env
# Editar .env con tu configuración
```

### 3. Entrenar (Opcional)

```bash
# Entrenar THAU Vision
python train_thau_vision.py --age 0 --epochs 30

# Entrenar THAU-2B
python train_thau_2b.py --target-age 15

# Entrenar Software Engineering Expert
python train_software_engineering_expert.py
```

### 4. Usar

```bash
# Iniciar API Server
python api/thau_code_server.py

# O usar Desktop App
cd thau-code-desktop
npm install && npm run dev
```

---

## 📚 Documentación Completa

Toda la documentación está organizada en la carpeta [`docs/`](docs/README.md):

### Para Empezar
- [**Instalación**](docs/getting-started/installation.md) - Configuración paso a paso
- [**Inicio Rápido**](docs/getting-started/quickstart.md) - Primeros pasos
- [**Glosario**](docs/GLOSSARY.md) - Términos y conceptos

### Arquitectura
- [Visión General](docs/architecture/overview.md)
- [THAU-2B](docs/architecture/thau-2b.md) - Modelo de lenguaje
- [THAU Vision](docs/architecture/thau-vision.md) - Sistema visual
- [Sistema de Agentes](docs/architecture/thau-agents.md) - Agentes especializados

### Guías
- [Entrenamiento THAU-2B](docs/guides/training-thau-2b.md)
- [Generación de Imágenes](docs/guides/image-generation.md)
- [Tool Calling](docs/guides/tool-calling.md)
- [Sistema de Agentes](docs/guides/agent-system.md)

### API
- [REST API](docs/api/rest-api.md)
- [WebSocket](docs/api/websocket.md)
- [MCP Protocol](docs/api/mcp-protocol.md)

---

## 🎓 Entrenamientos Especializados

THAU puede entrenarse para ser experto en dominios específicos:

### Software Engineering Expert
```bash
python train_software_engineering_expert.py
```

Aprende:
- Desarrollo Backend (FastAPI, Django, bases de datos)
- Desarrollo Frontend (React, TypeScript, estado)
- Mejores Prácticas (SOLID, Clean Code, Design Patterns)
- Algoritmos y Estructuras de Datos
- Código simple y mantenible
- Decisiones con sentido común

---

## 🛠️ Componentes del Sistema

### THAU Core
```
thau/
├── core/                    # Modelos transformer
│   ├── models/             # TinyLLM, Attention, Layers
│   ├── tokenizer/          # Tokenización BPE
│   ├── training/           # Trainers, Optimizadores
│   └── inference/          # Generación de texto
├── thau_models/            # Modelos especializados
│   ├── vision_model.py     # THAU Vision (VAE)
│   └── tool_calling.py     # Invocación de herramientas
├── thau_agents/            # Sistema de agentes
│   ├── agent_system.py     # Orquestador
│   ├── planner.py          # Planificación
│   ├── tool_factory.py     # Auto-creación
│   └── mcp_integration.py  # Protocolo MCP
├── memory/                 # Sistema de memoria
│   ├── manager.py          # Coordinador
│   ├── short_term.py       # Buffer conversacional
│   ├── long_term.py        # ChromaDB (RAG)
│   └── episodic.py         # Memoria temporal
└── api/                    # REST API + WebSocket
    └── thau_code_server.py # Servidor principal
```

### THAU Code Desktop
```
thau-code-desktop/
├── src/
│   ├── components/         # React Components
│   │   ├── ChatInterface.tsx
│   │   ├── AgentPanel.tsx
│   │   ├── PlannerView.tsx
│   │   ├── ToolFactory.tsx
│   │   └── CodeEditor.tsx (Monaco)
│   ├── services/           # API & WebSocket
│   └── App.tsx
└── electron/               # Desktop wrapper
    ├── main.js
    └── preload.js
```

---

## 💻 Requisitos del Sistema

### Mínimos
- Python 3.10+
- 8GB RAM
- 10GB Disco
- CPU 4 cores

### Recomendados
- Python 3.11+
- 16GB+ RAM
- 50GB SSD
- GPU 8GB+ (NVIDIA o Apple Silicon)
- Node.js 18+ (para Desktop App)

---

## 🤝 Contribuir

¡Las contribuciones son bienvenidas! Por favor:

1. Fork el repositorio
2. Crea una rama de feature (`git checkout -b feature/amazing`)
3. Commit tus cambios (`git commit -m 'Add amazing feature'`)
4. Push a la rama (`git push origin feature/amazing`)
5. Abre un Pull Request

Ver [CONTRIBUTING.md](docs/development/contributing.md) para más detalles.

---

## 🔬 Investigación y Papers

THAU implementa y extiende múltiples trabajos de investigación:

- **Attention Is All You Need** (Vaswani et al., 2017) - Transformers
- **RoFormer** (Su et al., 2021) - RoPE
- **LoRA** (Hu et al., 2021) - Fine-tuning eficiente
- **Self-Questioning** (Kim et al., 2023) - Auto-mejora
- **VAE** (Kingma & Welling, 2013) - Generación visual

**Innovaciones propias:**
1. Tool Factory - Auto-creación de herramientas
2. Cognitive Growth - Escalado dinámico de parámetros
3. Multi-Agent Orchestration - Coordinación inteligente

---

## 📄 Licencia

MIT License - Ver [LICENSE](LICENSE) para detalles

---

## 🙏 Créditos y Agradecimientos

**Desarrollado con amor para Thomas y Aurora** ❤️

**Tecnologías utilizadas:**
- PyTorch 2.0+
- Transformers (HuggingFace)
- FastAPI
- React + TypeScript
- Electron
- Monaco Editor
- ChromaDB

**Agradecimientos especiales:**
- A la comunidad open source
- A todos los investigadores en IA
- A mi familia por el apoyo infinito

---

## 📞 Soporte y Comunidad

- **Documentación**: [docs/](docs/README.md)
- **GitHub**: [github.com/luepow/thau](https://github.com/luepow/thau)
- **Autor**: Luis Eduardo Perez ([@luepow](https://github.com/luepow))
- **País**: Venezuela 🇻🇪

---

<div align="center">

**Construido con pasión, dedicación y amor**

*Para Thomas, Aurora y el futuro de la inteligencia artificial*

---

### THAU - Unidad Autónoma Holística Transformativa

*"Donde la Inteligencia Crea su Propio Camino"*

---

🇻🇪 **Hecho en Venezuela para el Mundo** 🌎

---

[⭐ Star en GitHub](https://github.com/luepow/thau) | [📖 Leer Docs](docs/README.md) | [🚀 Empezar Ahora](docs/getting-started/installation.md)

---

**© 2025 Luis Eduardo Perez - Licencia MIT**

*Inspirado por la curiosidad infinita de Thomas y Aurora*

</div>

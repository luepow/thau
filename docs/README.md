# THAU - Documentación Oficial

![THAU Logo](https://img.shields.io/badge/THAU-AI%20System-blue)
![Version](https://img.shields.io/badge/version-2.0.0-green)
![License](https://img.shields.io/badge/license-MIT-blue)

> **The World's First AI with Visual Imagination, Self-Created Tools, and Cognitive Growth**

## ¿Qué es THAU?

**THAU** (Transformative Holistic Autonomous Unit) es un sistema de IA de última generación que combina capacidades únicas:

- **Imaginación Visual Propia**: Sistema VAE entrenado desde cero para generar imágenes
- **Auto-Creación de Herramientas**: Crea sus propias herramientas desde lenguaje natural (único en el mundo)
- **Crecimiento Cognitivo**: Modelo que evoluciona de 256K → 2B parámetros según "edad"
- **Sistema de Agentes**: 11 agentes especializados con orquestación inteligente
- **Self-Learning**: Aprende de sus interacciones y se auto-cuestiona
- **MCP Compatible**: Interoperable con Claude, OpenAI y otros sistemas

---

## 📚 Estructura de la Documentación

### 🚀 Para Empezar

- [**Instalación**](getting-started/installation.md) - Configuración del entorno
- [**Inicio Rápido**](getting-started/quickstart.md) - Primeros pasos con THAU
- [**Primeros Experimentos**](getting-started/first-steps.md) - Tus primeras interacciones

### 🏗️ Arquitectura

- [**Visión General**](architecture/overview.md) - Arquitectura completa del sistema
- [**THAU Core**](architecture/thau-core.md) - Modelo transformer base
- [**THAU-2B**](architecture/thau-2b.md) - Modelo de lenguaje con crecimiento cognitivo
- [**THAU Vision**](architecture/thau-vision.md) - Sistema de imaginación visual (VAE)
- [**Sistema de Agentes**](architecture/thau-agents.md) - Orquestación de agentes especializados

### 📖 Guías

- [**Entrenamiento THAU-2B**](guides/training-thau-2b.md) - Entrenar el modelo de lenguaje
- [**Generación de Imágenes**](guides/image-generation.md) - Usar THAU Vision
- [**Tool Calling**](guides/tool-calling.md) - Invocación de herramientas
- [**Sistema de Agentes**](guides/agent-system.md) - Usar los 11 agentes especializados
- [**Desktop App**](guides/desktop-app.md) - THAU Code Desktop (Electron)

### 🔌 API Reference

- [**REST API**](api/rest-api.md) - Endpoints HTTP
- [**WebSocket**](api/websocket.md) - Comunicación en tiempo real
- [**MCP Protocol**](api/mcp-protocol.md) - Model Context Protocol

### 👨‍💻 Desarrollo

- [**Contribuir**](development/contributing.md) - Guía para contribuidores
- [**Testing**](development/testing.md) - Estrategia de pruebas
- [**Deployment**](development/deployment.md) - Despliegue en producción

### 📋 Referencia

- [**Glosario**](GLOSSARY.md) - Términos y conceptos clave
- [**Configuración**](reference/configuration.md) - Variables de entorno y configuración
- [**Roadmap**](reference/roadmap.md) - Futuras características
- [**Changelog**](reference/changelog.md) - Historial de cambios

---

## 🌟 Características Principales

### 1. Imaginación Visual (THAU Vision)

Sistema VAE entrenado desde cero que permite a THAU generar imágenes:

```python
from thau_models.vision_model import ThauVisionModel

vision = ThauVisionModel(age=0)
images = vision.imagine("un robot pintando")
```

**Capacidades:**
- Generación de imágenes desde texto
- Latent space de 64 dimensiones
- Resoluciones: 32x32, 64x64, 128x128
- Training progresivo con mejora de calidad

### 2. Auto-Creación de Herramientas (Tool Factory)

THAU puede crear sus propias herramientas desde descripciones en lenguaje natural:

```python
from thau_agents.tool_factory import ToolFactory

factory = ToolFactory()
tool = factory.create_tool(
    "Enviar notificaciones por email con plantillas HTML"
)
# Genera: email_notification_tool.py listo para usar
```

**Ejemplos de herramientas creadas:**
- APIs REST con autenticación
- Webhooks con retry logic
- Integración con calendarios
- Procesamiento de PDFs
- Consultas a bases de datos

### 3. Crecimiento Cognitivo (THAU-2B)

Modelo que evoluciona en capacidad según su "edad cognitiva":

| Edad | Parámetros | d_model | Capas | Uso |
|------|-----------|---------|-------|-----|
| 0    | 256K      | 256     | 2     | Conceptos básicos |
| 1    | 768K      | 384     | 3     | Oraciones completas |
| 3    | 1.7M      | 512     | 6     | Párrafos coherentes |
| 7    | 12M       | 768     | 12    | Razonamiento complejo |
| 15   | 2B        | 1536    | 24    | Capacidad completa |

```python
from thau_trainer.own_model_manager import ThauOwnModelManager

manager = ThauOwnModelManager(age=15)  # 2B parámetros
response = manager.generate("Explica la teoría de la relatividad")
```

### 4. Sistema de Agentes

11 agentes especializados orquestados inteligentemente:

```python
from thau_agents.agent_system import ThauAgentOrchestrator

orchestrator = ThauAgentOrchestrator()
result = orchestrator.assign_task(
    "Revisar este código y sugerir mejoras",
    role="code_reviewer"
)
```

**Agentes disponibles:**
1. 💬 **General** - Asistente general
2. ✍️ **Code Writer** - Escribir código
3. 👀 **Code Reviewer** - Revisar código
4. 🐛 **Debugger** - Encontrar y corregir bugs
5. 🔍 **Researcher** - Investigar temas
6. 📋 **Planner** - Planificar tareas complejas
7. 🏗️ **Architect** - Diseñar arquitecturas
8. 🧪 **Tester** - Escribir tests
9. 📝 **Documenter** - Generar documentación
10. 🔌 **API Specialist** - Trabajar con APIs
11. 📊 **Data Analyst** - Analizar datos
12. 🔒 **Security** - Análisis de seguridad
13. 🎨 **Visual Creator** - Generar imágenes

### 5. Self-Learning con Auto-Questioning

THAU aprende de sus interacciones y se auto-cuestiona para mejorar:

```python
from thau_trainer.self_learning import SelfLearningTrainer

trainer = SelfLearningTrainer()
trainer.train_from_interaction(
    user_message="¿Cómo funciona la atención?",
    assistant_response="La atención es un mecanismo..."
)
# THAU se auto-pregunta:
# - "¿Qué otras preguntas podría hacer el usuario?"
# - "¿Cómo puedo explicar esto mejor?"
```

### 6. MCP (Model Context Protocol)

Compatible con el estándar MCP para interoperabilidad:

```python
from thau_agents.mcp_integration import ThauMCPServer

mcp_server = ThauMCPServer()
result = mcp_server.handle_tool_call(
    session_id="session_123",
    tool_name="generate_image",
    arguments={"prompt": "un robot", "num_images": 3}
)
```

---

## 🚀 Inicio Rápido (5 minutos)

### 1. Instalar Dependencias

```bash
# Clonar repositorio
cd thau

# Crear entorno virtual
python3 -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Configurar Variables de Entorno

```bash
cp .env.example .env
# Editar .env con tu configuración
```

### 3. Entrenar THAU Vision (Opcional)

```bash
python train_thau_vision.py --age 0 --epochs 30
```

### 4. Entrenar THAU-2B (Opcional)

```bash
python train_thau_2b.py --target-age 15
```

### 5. Iniciar API Server

```bash
python api/thau_code_server.py
```

### 6. Usar THAU Code Desktop

```bash
cd thau-code-desktop
npm install
npm run dev
```

---

## 📊 Comparación con Otros Sistemas

| Característica | THAU | Claude Code | GitHub Copilot | OpenAI GPT-4 |
|---------------|------|-------------|----------------|--------------|
| Chat Interface | ✅ | ✅ | ❌ | ✅ |
| Code Generation | ✅ | ✅ | ✅ | ✅ |
| **Visual Imagination** | ✅ **Único** | ❌ | ❌ | ✅ (DALL-E) |
| **Auto-Tool Creation** | ✅ **Único** | ❌ | ❌ | ❌ |
| **Cognitive Growth** | ✅ **Único** | ❌ | ❌ | ❌ |
| Specialized Agents | ✅ 11 tipos | ❌ | ❌ | ❌ |
| Task Planning | ✅ | ✅ | ❌ | ✅ |
| MCP Compatible | ✅ | ✅ | ❌ | ✅ |
| Self-Learning | ✅ | ❌ | ❌ | ❌ |
| Desktop App | ✅ Electron | ✅ Native | ✅ Extension | ✅ Web |
| Open Source | ✅ | ❌ | ❌ | ❌ |

---

## 🎯 Casos de Uso

### Desarrollo de Software

```python
# Planificar una característica compleja
orchestrator.assign_task(
    "Diseña un sistema de autenticación con JWT",
    role="planner"
)

# Escribir el código
orchestrator.assign_task(
    "Implementa el AuthService con las especificaciones del plan",
    role="code_writer"
)

# Revisar el código
orchestrator.assign_task(
    "Revisa el AuthService y sugiere mejoras",
    role="code_reviewer"
)

# Generar tests
orchestrator.assign_task(
    "Escribe tests unitarios para AuthService",
    role="tester"
)
```

### Generación de Contenido Visual

```python
# Generar imágenes para un proyecto
vision_model = ThauVisionModel(age=0)
images = vision_model.imagine([
    "logo de una startup tech",
    "interfaz de usuario moderna",
    "diagrama de arquitectura"
])
```

### Automatización con Herramientas

```python
# Crear herramienta personalizada
tool = factory.create_tool(
    "Consultar API de clima y enviar notificación si llueve"
)

# Usar la herramienta
result = tool.execute(city="Madrid")
```

---

## 🔬 Investigación y Publicaciones

THAU implementa y extiende múltiples papers de investigación:

- **Attention Is All You Need** (Vaswani et al., 2017) - Arquitectura Transformer
- **RoFormer** (Su et al., 2021) - Rotary Position Embedding
- **LoRA** (Hu et al., 2021) - Low-Rank Adaptation
- **Self-Questioning** (Kim et al., 2023) - Auto-mejora mediante preguntas
- **VAE** (Kingma & Welling, 2013) - Variational Autoencoders

**Innovaciones propias de THAU:**
1. **Tool Factory**: Auto-creación de herramientas desde lenguaje natural
2. **Cognitive Growth**: Escalado dinámico de parámetros según edad
3. **Visual Imagination**: VAE integrado en sistema de lenguaje

---

## 💻 Requisitos del Sistema

### Mínimos

- **CPU**: 4 cores
- **RAM**: 8GB
- **Disk**: 10GB
- **Python**: 3.10+
- **OS**: Linux, macOS, Windows

### Recomendados

- **GPU**: NVIDIA GPU con 8GB+ VRAM (para entrenamiento)
- **CPU**: 8+ cores
- **RAM**: 16GB+
- **Disk**: 50GB SSD
- **Python**: 3.11+

### Para Desktop App

- **Node.js**: 18+
- **npm**: 9+
- **Electron**: 28+

---

## 🤝 Comunidad y Soporte

- **Documentación**: [https://thau-docs.example.com](./README.md)
- **GitHub**: [https://github.com/your-org/thau](https://github.com)
- **Issues**: [GitHub Issues](https://github.com/your-org/thau/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/thau/discussions)
- **Email**: support@thau-ai.com

---

## 📄 Licencia

MIT License - Ver [LICENSE](../LICENSE) para detalles

---

## 🙏 Créditos

**THAU Team**

Tecnologías utilizadas:
- PyTorch
- Transformers (HuggingFace)
- FastAPI
- React + TypeScript
- Electron
- Monaco Editor

---

**Construido con pasión por el futuro de la IA**

*Haciendo el desarrollo de IA accesible para todos*

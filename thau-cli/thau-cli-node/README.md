# THAU CLI

![Version](https://img.shields.io/badge/version-2.0.0-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Node](https://img.shields.io/badge/node-18+-yellow)

**THAU CLI** es una herramienta de línea de comandos potente y profesional para trabajar con THAU, el sistema de IA con capacidades únicas de imaginación visual y auto-creación de herramientas.

Similar a **Claude Code**, pero con la flexibilidad de trabajar con modelos THAU o Ollama local.

---

## 🚀 Instalación

\`\`\`bash
cd thau-cli-node
npm install
npm link
\`\`\`

Verifica la instalación:
\`\`\`bash
thau --version
\`\`\`

---

## ✨ Nuevas Funcionalidades (v2.0)

✅ **Selección de Modelo** - THAU API o Ollama local  
✅ **Sistema de Permisos** - Control granular de archivos y comandos  
✅ **MCP Integration** - Conecta a servidores MCP  
✅ **Carpeta .thau/** - Configuración por proyecto (como Claude Code)  
✅ **thau.md** - Instrucciones personalizadas (como CLAUDE.md)  
✅ **11 Agentes Especializados** - Planner, Code Writer, Reviewer, etc.  
✅ **Auto-Inicialización** - Crea proyecto THAU automáticamente  

---

## 💻 Comandos Principales

### \`thau code\`
Modo interactivo de programación con agentes especializados.

\`\`\`bash
thau code
\`\`\`

**Agentes Disponibles:**
- \`general\` - Propósito general
- \`code_writer\` - Escribir código
- \`planner\` - Planificación de tareas
- \`code_reviewer\` - Revisión de código
- \`debugger\` - Depuración
- \`architect\` - Arquitectura
- \`test_writer\` - Generación de tests
- \`refactorer\` - Refactorización
- \`explainer\` - Explicación de código
- \`optimizer\` - Optimización
- \`security\` - Seguridad

---

## 🎮 Comandos Internos (Modo Interactivo)

### \`/model\`
Gestiona modelos disponibles.

\`\`\`
/model list               # Lista modelos disponibles
/model switch thau-api    # Cambia a THAU API
/model switch ollama:codellama  # Cambia a Ollama
\`\`\`

**Modelos Soportados:**
- \`thau-api\` - Servidor THAU (puerto 8000)
- \`ollama:codellama\` - Ollama CodeLlama
- \`ollama:mistral\` - Ollama Mistral
- \`ollama:llama2\` - Ollama Llama 2
- Cualquier modelo Ollama instalado localmente

### \`/mcp\`
Gestiona servidores MCP (Model Context Protocol).

\`\`\`
/mcp status                       # Ver conexiones MCP
/mcp connect thau-main            # Conectar a servidor
/mcp disconnect thau-main         # Desconectar
/mcp add my-server http://localhost:9000  # Agregar servidor
/mcp tools thau-main              # Listar herramientas
\`\`\`

### \`/permissions\`
Gestiona permisos de seguridad.

\`\`\`
/permissions show     # Ver permisos actuales
/permissions reset    # Resetear permisos
\`\`\`

### \`/exec <comando>\`
Ejecuta comandos de terminal con sistema de permisos.

\`\`\`
/exec ls -la
/exec npm install
/exec git status
\`\`\`

### Otros Comandos

- \`/help\` - Ayuda
- \`/agent <nombre>\` - Cambiar agente
- \`/clear\` - Limpiar historial
- \`/exit\` - Salir

---

## 📁 Carpeta \`.thau/\`

THAU crea automáticamente una carpeta \`.thau/\` en tu proyecto con:

\`\`\`
.thau/
├── thau.md              # Instrucciones personalizadas (como CLAUDE.md)
├── config.json          # Configuración del proyecto
├── prompts/             # Prompts personalizados
├── tools/               # Herramientas custom
└── memory/              # Memoria del proyecto
\`\`\`

### \`thau.md\` - Instrucciones Personalizadas

Similar a CLAUDE.md, puedes definir instrucciones específicas para tu proyecto.

THAU leerá estas instrucciones al trabajar en tu código.

---

## 🔧 Otros Comandos

### \`thau init\`
Inicializa proyecto THAU en el directorio actual.

### \`thau chat\`
Chat general con THAU.

### \`thau plan <tarea>\`
Crea un plan detallado para una tarea.

### \`thau create <tipo> [nombre]\`
Crea archivos de código (file, class, function, component).

### \`thau review <archivos>\`
Revisa código para bugs y mejoras.

### \`thau test <archivos>\`
Genera tests para código.

---

## 🌐 Configuración de Modelos

### THAU API Server

\`\`\`bash
# Iniciar servidor THAU
cd /path/to/thau
PYTHONPATH=. python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
\`\`\`

### Ollama (Modelos Locales)

\`\`\`bash
# Instalar
brew install ollama  # macOS

# Instalar modelos
ollama pull codellama
ollama pull mistral

# Usar en THAU CLI
thau code
/model switch ollama:codellama
\`\`\`

---

## 🔒 Sistema de Permisos

THAU incluye un sistema de permisos granular:

- ⚠️ Escritura de archivos (requiere confirmación)
- ⚠️ Eliminación de archivos (requiere confirmación doble)
- ⚠️ Ejecución de comandos (requiere aprobación)
- ✅ Lectura de archivos (auto-aprobado)

Archivo: \`~/.thau/permissions.json\`

---

## 📝 Ejemplo de Uso

\`\`\`bash
cd my-project
thau code

# Dentro del modo interactivo:
/agent planner
"Diseña una API REST para gestión de usuarios"

/agent code_writer
"Implementa el endpoint de registro"

/agent test_writer
"Genera tests para el registro"

/exec pytest tests/

/agent code_reviewer
"Revisa el código"
\`\`\`

---

## 👤 Autor

**Luis Eduardo Perez** ([@luepow](https://github.com/luepow))  
Desarrollado con ❤️ en Venezuela 🇻🇪

---

**THAU CLI** - Donde la Inteligencia Crea su Propio Camino

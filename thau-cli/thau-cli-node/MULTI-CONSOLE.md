# THAU Multi-Console System

## Descripción

El **Sistema Multi-Console** es una característica avanzada de THAU CLI que permite ejecutar múltiples sesiones de agentes simultáneamente, cada una con su propio contexto e historial independiente. Similar a tener múltiples pestañas o terminales, pero dentro de una sola interfaz de THAU.

## Arquitectura

### Componentes Principales

#### 1. `SessionManager` (src/lib/session-manager.js)
**Propósito**: Gestor central de sesiones de agentes

**Características**:
- Crea y gestiona múltiples sesiones independientes
- Cada sesión tiene:
  - ID único
  - Agente asignado
  - Modelo asignado
  - Historial de conversación separado
  - Timestamps de creación y última actividad
- Event-driven architecture (EventEmitter)
- Persistencia de sesiones (import/export JSON)

**Métodos principales**:
```javascript
createSession(agentName, model)    // Crea nueva sesión
getSession(sessionId)               // Obtiene sesión por ID
switchSession(sessionId)            // Cambia a otra sesión
listSessions()                      // Lista todas las sesiones
addMessage(sessionId, role, content) // Agrega mensaje al historial
getHistory(sessionId, limit)        // Obtiene historial de sesión
clearHistory(sessionId)             // Limpia historial
deleteSession(sessionId)            // Elimina sesión
changeAgent(sessionId, newAgent)    // Cambia agente de sesión
changeModel(sessionId, newModel)    // Cambia modelo de sesión
getStats()                          // Estadísticas generales
```

**Eventos emitidos**:
- `session:created` - Cuando se crea una sesión
- `session:deleted` - Cuando se elimina una sesión
- `session:switched` - Cuando se cambia de sesión activa
- `message:added` - Cuando se agrega un mensaje
- `history:cleared` - Cuando se limpia el historial
- `agent:changed` - Cuando se cambia el agente
- `model:changed` - Cuando se cambia el modelo

#### 2. `MultiConsole` (src/ui/MultiConsole.jsx)
**Propósito**: UI visual basada en Ink (React for Terminal)

**Características**:
- Interfaz de pestañas para múltiples sesiones
- Display en tiempo real de:
  - Sesiones activas (tabs)
  - Información de sesión actual (agente, modelo, mensajes)
  - Historial de conversación
  - Input interactivo
- Teclado shortcuts
- Indicador de procesamiento
- Sistema de ayuda integrado

**Layout**:
```
┌──────────────────────────────────────────────────────────┐
│ THAU Multi-Console - [1] general | [2] code_writer | ... │
├──────────────────────────────────────────────────────────┤
│ Agent: code_writer | Model: thau-api | Messages: 12      │
├──────────────────────────────────────────────────────────┤
│                                                            │
│ [Conversation History]                                     │
│                                                            │
│ You: Explicame este código                                │
│ THAU: Este código implementa...                           │
│                                                            │
├──────────────────────────────────────────────────────────┤
│ code_writer> Type your message...                         │
├──────────────────────────────────────────────────────────┤
│ Ctrl+N: New | Ctrl+W: Close | Ctrl+Tab: Next | Ctrl+C: Exit│
└──────────────────────────────────────────────────────────┘
```

**Keyboard Shortcuts**:
- `Ctrl+N` - Nueva sesión
- `Ctrl+W` - Cerrar sesión actual
- `Ctrl+Tab` - Cambiar a siguiente sesión
- `Alt+[1-9]` - Cambiar a sesión específica (1-9)
- `Ctrl+H` - Mostrar/ocultar ayuda
- `Ctrl+C` - Salir de THAU

**Comandos Internos**:
```bash
/help                # Muestra ayuda
/agent <nombre>      # Cambia agente de sesión actual
/model <nombre>      # Cambia modelo de sesión actual
/new [agente]        # Crea nueva sesión
/close               # Cierra sesión actual
/clear               # Limpia historial
/exit                # Sale de THAU
```

#### 3. `code-multi.js` (src/commands/code-multi.js)
**Propósito**: Comando de entrada al modo multi-console

**Flujo de ejecución**:
1. Verifica inicialización del proyecto (.thau/)
2. Check de conectividad (THAU API / Ollama)
3. Crea sesiones iniciales (general, code_writer, planner)
4. Renderiza la UI con Ink
5. Maneja el ciclo de vida de la aplicación

## Uso

### Instalación de Dependencias

Primero, asegúrate de tener instaladas las dependencias de Ink:

```bash
cd thau-cli-node
npm install ink ink-text-input react
```

### Ejecución

#### Modo Tradicional (Una sola sesión):
```bash
thau code
```

#### Modo Multi-Console (Múltiples sesiones):
```bash
thau code-multi
```

O puedes agregarlo como comando principal:

1. Edita `bin/thau`:
```javascript
#!/usr/bin/env node

const { Command } = require('commander');
const program = new Command();
const codeCommand = require('../src/commands/code');
const codeMultiCommand = require('../src/commands/code-multi');

program
  .version('2.0.0')
  .description('THAU CLI - AI Assistant with Multi-Console Support');

program
  .command('code')
  .description('Interactive coding mode (single session)')
  .action(codeCommand);

program
  .command('multi')
  .description('Multi-console mode (multiple sessions)')
  .action(codeMultiCommand);

program.parse(process.argv);
```

2. Usar:
```bash
thau multi
```

### Flujo de Trabajo Típico

**Escenario 1: Desarrollo con múltiples agentes**

1. Inicia THAU Multi-Console:
```bash
thau multi
```

2. Por defecto, tienes 3 sesiones:
   - `[1] general` - Propósito general
   - `[2] code_writer` - Escritura de código
   - `[3] planner` - Planificación

3. Trabaja en sesión de planner:
```
planner> "Diseña una API REST para gestión de usuarios"
```

4. Cambia a code_writer (Ctrl+2 o Alt+2):
```
code_writer> "Implementa el endpoint de registro basado en el plan"
```

5. Crea nueva sesión para tests (Ctrl+N):
```
/new test_writer
test_writer> "Genera tests para el endpoint de registro"
```

6. Alterna entre sesiones sin perder contexto

**Escenario 2: Debugging paralelo**

1. Sesión 1 (debugger): Analiza error
```
debugger> "Analiza este error de null pointer"
```

2. Sesión 2 (code_reviewer): Revisa código relacionado
```
code_reviewer> "Revisa esta función por posibles bugs"
```

3. Sesión 3 (explainer): Explica flujo
```
explainer> "Explica el flujo de autenticación"
```

4. Sesión 4 (code_writer): Aplica fix
```
code_writer> "Arregla el null pointer basado en el análisis"
```

## Ventajas del Sistema Multi-Console

### 1. **Contexto Independiente**
Cada sesión mantiene su propio historial y contexto, sin interferencias entre agentes.

### 2. **Productividad**
Trabaja en múltiples tareas simultáneamente sin perder el hilo de ninguna.

### 3. **Especialización**
Utiliza agentes especializados para diferentes aspectos del desarrollo:
- Planner para diseño
- Code Writer para implementación
- Test Writer para pruebas
- Code Reviewer para revisión
- Debugger para errores

### 4. **Organización**
Separa conceptualmente diferentes fases o componentes del proyecto.

### 5. **Persistencia**
Cada sesión mantiene su historial completo, permitiendo revisión posterior.

## Configuración Avanzada

### Sesiones Personalizadas

Puedes crear sesiones con configuraciones específicas:

```javascript
// En tu código
const sessionId = sessionManager.createSession('custom-agent', 'ollama:codellama');
```

### Persistencia de Sesiones

Exportar sesiones:
```javascript
const json = sessionManager.exportSessions();
fs.writeFileSync('sessions-backup.json', json);
```

Importar sesiones:
```javascript
const json = fs.readFileSync('sessions-backup.json', 'utf8');
sessionManager.importSessions(json);
```

### Estadísticas

Obtener estadísticas de uso:
```javascript
const stats = sessionManager.getStats();
console.log(stats);
// {
//   totalSessions: 5,
//   totalMessages: 127,
//   agentDistribution: { general: 2, code_writer: 2, planner: 1 },
//   activeSession: 'code_writer-1638...'
// }
```

## Integración con THAU API y Ollama

El multi-console soporta ambos backends transparentemente:

### THAU API
```javascript
// Automáticamente usa THAU API si está disponible
thauClient.sendTask(message, agent, context);
```

### Ollama (Fallback)
```javascript
// Si THAU API no está disponible, usa Ollama
await thauClient.switchModel('ollama', 'codellama');
```

### Cambio Dinámico
Cada sesión puede usar un modelo diferente:
```
[1] general (THAU API) | [2] code_writer (Ollama:codellama)
```

## Troubleshooting

### Error: "Cannot find module 'ink'"
```bash
npm install ink ink-text-input react
```

### Sesiones no se actualizan
Verifica que los eventos se estén emitiendo correctamente:
```javascript
sessionManager.on('message:added', (data) => {
  console.log('Message added:', data);
});
```

### Performance con muchas sesiones
Limita el historial por sesión:
```javascript
const history = sessionManager.getHistory(sessionId, 20); // Solo últimos 20 mensajes
```

### UI se ve mal en algunos terminales
Asegúrate de usar un terminal compatible con ANSI colors y UTF-8:
- ✅ iTerm2 (macOS)
- ✅ Windows Terminal
- ✅ Hyper
- ❌ CMD.exe antiguo

## Roadmap Futuro

### Funcionalidades Planificadas

1. **Sesiones Guardadas**: Guardar automáticamente sesiones al salir y restaurarlas al iniciar
2. **Temas Visuales**: Diferentes esquemas de colores y estilos
3. **Sesiones Compartidas**: Exportar/importar sesiones específicas
4. **Atajos Personalizables**: Configurar keyboard shortcuts personalizados
5. **Indicadores Visuales**: Badges para sesiones con mensajes no leídos
6. **Búsqueda en Historial**: Buscar en el historial de todas las sesiones
7. **Sesiones Persistentes**: SQLite para almacenar sesiones a largo plazo
8. **Modo Split**: Ver 2+ sesiones simultáneamente en split screen

## Código de Ejemplo

### Crear Aplicación Multi-Console Personalizada

```javascript
const React = require('react');
const { render } = require('ink');
const SessionManager = require('./src/lib/session-manager');
const MultiConsole = require('./src/ui/MultiConsole');
const ThauClient = require('./src/lib/client');

async function main() {
  const client = new ThauClient();
  const sessionManager = new SessionManager();

  // Crear sesiones personalizadas
  sessionManager.createSession('architect', 'thau-api');
  sessionManager.createSession('security', 'ollama:llama2');
  sessionManager.createSession('performance', 'thau-api');

  // Renderizar UI
  const { waitUntilExit } = render(
    React.createElement(MultiConsole, {
      sessionManager: sessionManager,
      thauClient: client,
      onExit: () => console.log('Bye!')
    })
  );

  await waitUntilExit();
}

main();
```

## Contribuciones

Este sistema es extensible. Puedes:

1. Crear componentes UI personalizados con Ink
2. Extender `SessionManager` con nuevos métodos
3. Implementar backends adicionales (OpenAI, Anthropic, etc.)
4. Mejorar la UI con más información contextual

## Autor

**Luis Eduardo Perez** ([@luepow](https://github.com/luepow))
Desarrollado con ❤️ en Venezuela 🇻🇪

---

**THAU CLI Multi-Console** - Donde Múltiples Inteligencias Trabajan en Armonía
